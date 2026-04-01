import asyncio
import gc
import json
import logging
import os
import platform
import re
import threading
import time
import uuid
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import dask.array as da
import numpy as np
import pystac
import rasterio
import rioxarray
import torch
import xarray as xr
from dask import config
from dask.diagnostics import ProgressBar
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

from .config import logging_config  # noqa: F401
from .geo_dask import (
    read_zarr_metadata,
    runModel,
    sum_overlapped_chunks,
)
from .utils.helpers import (
    asset_by_common_name,
    cmd_interface,
    get_directory,
    get_model,
    select_model_device,
    xarray_profile_info,
)
from .utils.polygon import gdf_to_yolo, geojson2coco, mask_to_poly_geojson
from .utils.post_inference import Config, buildings_splitter, clean_mask

logger = logging.getLogger(__name__)


class GeoInference:
    """
    Perform geospatial inference on imagery using a Torch-exported model.

    Args:
        model (str | None): Path or URL to a Torch-exported model artifact. If a URL is
            provided, it is downloaded into `work_dir`.
        work_dir (str | None): Directory for the resolved model and all outputs.
        mask_to_vec (bool): If True, convert the output mask to polygons (GeoJSON).
        mask_to_coco (bool): If True and `mask_to_vec` is True, also emit COCO JSON.
        mask_to_yolo (bool): If True and `mask_to_vec` is True, also emit YOLO CSV.
        device (str | None): Device selector forwarded to `select_model_device()`.
        multi_gpu (bool): Whether to enable multi-GPU selection in
            `select_model_device()`.
        gpu_id (int): GPU index used by `select_model_device()` when applicable.
        num_classes (int): Number of output classes.
        prediction_threshold (float): Probability threshold used during overlap
            reduction (`sum_overlapped_chunks`).
        post_inference (bool): Whether to run post-inference operations (e.g. cleaning,
            splitting) driven by model metadata and `sensor_name`.
        sam_checkpoint_path (str | None): Path to the SAM checkpoint (post-inference).
        sam_bpe_path (str | None): Path to the SAM BPE file (post-inference).
    """

    def __init__(
        self,
        model: str = None,
        work_dir: str = None,
        mask_to_vec: bool = False,
        mask_to_coco: bool = False,
        mask_to_yolo: bool = False,
        device: str = None,
        multi_gpu: bool = False,
        gpu_id: int = 0,
        num_classes: int = 5,
        prediction_threshold: float = 0.3,
        post_inference: bool = False,
        sam_checkpoint_path: str = None,
        sam_bpe_path: str = None,
    ):
        self.work_dir: Path = get_directory(work_dir)
        self.device = select_model_device(gpu_id, multi_gpu, device)
        self._model_path = str(
            get_model(model_path_or_url=model, work_dir=self.work_dir)
        )
        extra_files = {"metadata.json": ""}
        self.model = (
            torch.export.load(self._model_path, extra_files=extra_files)
            .module()
            .to(self.device)
        )
        raw = extra_files.get("metadata.json", "").strip()
        self.metadata = json.loads(raw) if raw else None
        self.mask_to_vec = mask_to_vec
        self.mask_to_coco = mask_to_coco
        self.mask_to_yolo = mask_to_yolo
        self.classes = num_classes
        self.prediction_threshold = prediction_threshold
        self.post_inference = post_inference
        self.raster_meta = None
        self.sam_checkpoint_path = sam_checkpoint_path
        self.sam_bpe_path = sam_bpe_path

    @torch.no_grad()
    def __call__(
        self,
        inference_input: Union[Path, str],
        sensor_name: str = None,
        bands_requested: List[str] = [],
        patch_size: int = 1024,
        workers: int = 0,
        bbox: str = None,
    ) -> str:

        async def run_async():

            # Start the periodic garbage collection task
            self.gc_task = asyncio.create_task(
                self.constant_gc(5)
            )  # Calls gc.collect() every 5 seconds
            # Run the main computation asynchronously
            self.mask_layer_name = await self.async_run_inference(
                inference_input=inference_input,
                sensor_name=sensor_name,
                bands_requested=bands_requested,
                patch_size=patch_size,
                workers=workers,
                bbox=bbox,
            )
            self.gc_task.cancel()

            try:
                await self.gc_task
            except asyncio.CancelledError:
                pass

        asyncio.run(run_async())
        return self.mask_layer_name

    async def async_run_inference(
        self,
        inference_input: Union[Path, str],
        sensor_name: str = None,
        bands_requested: List[str] = [],
        patch_size: int = 1024,
        workers: int = 0,
        bbox: str = None,
    ) -> None:
        """
        Perform geo inference on geospatial imagery using dask array.

        Args:
            inference_input Union[Path, str]: The path/url to the geospatial image to perform inference on.
            sensor_name (str): The name of the sensor to use for the inference.
            bands_requested List[str]: The requested bands to consider for the inference.
            patch_size (int): The size of the patches to use for inference.
            workers (int): Number of workers used by dask, Default = Nb of cores available on the host, minus 1.
            bbox (str): The bbox or extent of the image in this format "minx, miny, maxx, maxy".

        Returns:
            None

        """

        # configuring dask with proper number of workers, alternatively we could also use os.getenv('SLURM_CPUS_PER_TASK')
        if workers != 0:
            num_workers = workers
        elif "linux" in platform.uname().system.lower():
            num_workers = len(os.sched_getaffinity(0)) - 1
        else:
            num_workers = os.cpu_count() - 1
        print(f"running dask with {num_workers} workers")
        config.set(scheduler="threads", num_workers=num_workers)
        config.set(pool=ThreadPool(num_workers))

        if not isinstance(inference_input, (str, Path)):
            raise TypeError(
                f"Invalid raster type.\nGot {inference_input} of type {type(inference_input)}"
            )
        if not isinstance(bands_requested, (List)):
            raise ValueError(
                f"Requested bands should be a list."
                f"\nGot {bands_requested} of type {type(bands_requested)}"
            )
        if not isinstance(patch_size, int):
            raise TypeError(
                f"Invalid patch size. Patch size should be an integer..\nGot {patch_size}"
            )

        base_name = os.path.basename(
            Path(inference_input)
            if isinstance(inference_input, str)
            else inference_input
        )
        # it takes care of urls
        prefix_base_name = (
            base_name if not base_name.endswith(".tif") else base_name[:-4]
        )
        prefix_base_name = (
            prefix_base_name
            if not prefix_base_name.endswith(".zarr")
            else base_name[:-5]
        )
        u_id = uuid.uuid4().hex[:6]
        mask_path = self.work_dir.joinpath(prefix_base_name + f"_mask_{u_id}.tif")
        polygons_path = self.work_dir.joinpath(
            prefix_base_name + f"_polygons_{u_id}.geojson"
        )
        yolo_csv_path = self.work_dir.joinpath(prefix_base_name + f"_yolo_{u_id}.csv")
        coco_json_path = self.work_dir.joinpath(prefix_base_name + f"_coco_{u_id}.json")
        stride_patch_size = int(patch_size / 2)
        self.no_data = None
        self.input_dtype = None

        """ Processing starts"""
        start_time = time.time()
        try:
            raster_stac_item = False
            if isinstance(inference_input, pystac.Item):
                raster_stac_item = True
            else:
                try:
                    pystac.Item.from_file(str(inference_input))
                    raster_stac_item = True
                except Exception:
                    raster_stac_item = False
            self.json = None
            if not raster_stac_item:
                inference_input_path = Path(inference_input)
                if os.path.splitext(inference_input_path)[1].lower() == ".zarr":
                    aoi_dask_array = da.from_zarr(
                        inference_input,
                        chunks=(1, stride_patch_size, stride_patch_size),
                    )
                    meta_data_json = re.sub(r"\.zarr$", "", inference_input)
                    self.json = read_zarr_metadata(f"{meta_data_json}.json")
                else:
                    with rasterio.open(inference_input, "r") as src:
                        self.raster_meta = src.meta
                        self.raster = src
                        self.no_data = src.nodata
                        self.input_dtype = src.dtypes[0]

                    aoi_dask_array = rioxarray.open_rasterio(
                        inference_input,
                        chunks=(1, stride_patch_size, stride_patch_size),
                    )

                try:
                    if bands_requested:
                        if len(bands_requested) != 0:
                            if self.json is None:
                                logger.info("Bands are reordeing to bands_requested:")
                                aoi_dask_array = xr.concat(
                                    [
                                        aoi_dask_array[int(i) - 1, :, :]
                                        for i in bands_requested
                                    ],
                                    dim="band",
                                )
                            else:
                                logger.info("Bands are reordeing to bands_requested:")
                                aoi_dask_array = da.stack(
                                    [
                                        aoi_dask_array[int(i) - 1, :, :]
                                        for i in bands_requested
                                    ],
                                    axis=0,
                                )
                except Exception as e:
                    raise e
            else:
                assets = asset_by_common_name(inference_input)
                try:
                    bands_requested = {
                        band: assets[band.lower()] for band in bands_requested
                    }
                except KeyError:
                    raise KeyError(
                        f"Common names of the STAC assets ({assets.keys()}) do not match provided bands_requested keys ({bands_requested})."
                    )

                rio_gdal_options = {
                    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
                    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
                }
                all_bands_requested = []
                with rasterio.Env(**rio_gdal_options):
                    with rasterio.open(
                        bands_requested[next(iter(bands_requested))]["meta"].href, "r"
                    ) as src:
                        self.raster_meta = src.meta
                        self.raster = src
                        self.no_data = src.nodata
                        self.input_dtype = src.dtypes[0]
                    for key, value in bands_requested.items():
                        all_bands_requested.append(
                            rioxarray.open_rasterio(
                                value["meta"].href,
                                chunks=(1, stride_patch_size, stride_patch_size),
                            )
                        )
                aoi_dask_array = xr.concat(all_bands_requested, dim="band")
                del all_bands_requested

            is_float = False
            if self.no_data is None:
                if np.issubdtype(np.dtype(self.input_dtype), np.floating):
                    self.no_data = np.nan
                    is_float = True
                else:
                    self.no_data = 0
            if is_float:
                self.valid_mask = np.isfinite(aoi_dask_array).all(dim="band")
            else:
                self.valid_mask = (aoi_dask_array != self.no_data).all(dim="band")

            if bbox is not None:
                if not isinstance(bbox, (List)):
                    raise TypeError("bbox should be a list.")
                bbox = tuple(map(float, bbox))
                self.roi_window = from_bounds(
                    left=bbox[0],
                    bottom=bbox[1],
                    right=bbox[2],
                    top=bbox[3],
                    transform=self.raster_meta["transform"],
                )
                self.bbox_transform = from_origin(
                    bbox[0],
                    bbox[3],
                    self.raster_meta["transform"].a,
                    self.raster_meta["transform"].e
                    if self.raster_meta["transform"].e > 0
                    else -1 * self.raster_meta["transform"].e,
                )
                col_off, row_off = (
                    int(self.roi_window.col_off),
                    int(self.roi_window.row_off),
                )
                width, height = int(self.roi_window.width), int(self.roi_window.height)
                aoi_dask_array = aoi_dask_array[
                    :, row_off : row_off + height, col_off : col_off + width
                ]
                self.valid_mask = self.valid_mask[
                    row_off : row_off + height, col_off : col_off + width
                ]
                self.raster_meta.update(
                    {
                        "transform": self.bbox_transform,
                        "width": aoi_dask_array.shape[2],
                        "height": aoi_dask_array.shape[1],
                    }
                )
            self.original_shape = aoi_dask_array.shape
            # Pad the array to make dimensions multiples of the patch size
            pad_height = (
                stride_patch_size - aoi_dask_array.shape[1] % stride_patch_size
            ) % stride_patch_size
            pad_width = (
                stride_patch_size - aoi_dask_array.shape[2] % stride_patch_size
            ) % stride_patch_size
            padded = da.pad(
                aoi_dask_array.data if self.json is None else aoi_dask_array,
                ((0, 0), (0, pad_height), (0, pad_width)),
                mode="constant",
            )
            # Extra right/bottom margin so boundary chunks get full overlap (512×512)
            padded = da.pad(
                padded,
                ((0, 0), (0, stride_patch_size), (0, stride_patch_size)),
                mode="constant",
                constant_values=0,
            )
            aoi_dask_array = padded.rechunk(
                (aoi_dask_array.shape[0], stride_patch_size, stride_patch_size)
            )

            ordered_input: List[Any] = []
            sensor_meta: Dict[str, Any] = {}
            class_priors: Optional[list[float]] = None

            if self.metadata:
                if not sensor_name:
                    raise ValueError("Sensor name is required when model has metadata")
                if sensor_name not in self.metadata:
                    raise ValueError(f"Sensor name {sensor_name} not found in metadata")

                sensor_meta = self.metadata[sensor_name]
                ordered_input = sensor_meta.get("model_inputs", [])

                if "class_priors" in sensor_meta:
                    class_priors = sensor_meta["class_priors"]

            device_str = (
                str(self.device)
                if isinstance(self.device, torch.device)
                else self.device
            )
            aoi_dask_array = aoi_dask_array.map_overlap(
                runModel,
                model=self.model,
                ordered_input=ordered_input,
                patch_size=patch_size,
                device=device_str,
                num_classes=self.classes,
                no_data=self.no_data,
                chunks=(
                    self.classes + 1,
                    patch_size,
                    patch_size,
                ),
                depth={1: (0, stride_patch_size), 2: (0, stride_patch_size)},
                boundary="none",
                trim=False,
                dtype=np.float16,
            )
            aoi_dask_array = aoi_dask_array.map_overlap(
                sum_overlapped_chunks,
                chunk_size=patch_size,
                prediction_threshold=self.prediction_threshold,
                class_priors=class_priors,
                drop_axis=0,
                chunks=(
                    stride_patch_size,
                    stride_patch_size,
                ),
                depth={1: (stride_patch_size, 0), 2: (stride_patch_size, 0)},
                trim=False,
                boundary="none",
                dtype=np.uint8,
            )

            with ProgressBar() as pbar:
                pbar.register()
                # import rioxarray
                logger.info("Inference is running:")
                aoi_dask_array = xr.DataArray(
                    aoi_dask_array[: self.original_shape[1], : self.original_shape[2]],
                    dims=("y", "x"),
                    attrs=self.json
                    if self.json is not None
                    else xarray_profile_info(self.raster_meta),
                )
                aoi_dask_array = aoi_dask_array.where(self.valid_mask, other=255)
                aoi_dask_array.rio.write_nodata(255, inplace=True)
                aoi_dask_array.rio.to_raster(
                    mask_path, tiled=True, lock=threading.Lock()
                )

            if self.post_inference:
                if "building" in sensor_meta["class_labels"]:
                    building_class_index = int(sensor_meta["class_labels"]["building"])
                    gsd = sensor_meta["gsd"]
                    splitter_config = Config(
                        building_class_index=building_class_index,
                        road_class_index=3,
                        gsd=gsd,
                        device=device_str,
                        checkpoint_path=self.sam_checkpoint_path,
                        bpe_path=self.sam_bpe_path,
                    )
                    mask_path = buildings_splitter(
                        inference_input, mask_path, splitter_config
                    )

                min_area_m2 = sensor_meta.get("min_area_m2")
                min_area_m2 = (
                    {int(k): float(v) for k, v in min_area_m2.items()}
                    if min_area_m2
                    else None
                )
                if min_area_m2 is not None:
                    road_class_index = None
                    if "road" in sensor_meta["class_labels"]:
                        road_class_index = int(sensor_meta["class_labels"]["road"])
                    clean_config = Config(
                        road_class_index=road_class_index,
                        gsd=gsd,
                        min_area_m2=min_area_m2,
                    )
                    mask_path = clean_mask(mask_path, clean_config)

            if self.mask_to_vec:
                mask_to_poly_geojson(mask_path, polygons_path)
                if self.mask_to_yolo:
                    gdf_to_yolo(polygons_path, mask_path, yolo_csv_path)
                if self.mask_to_coco:
                    geojson2coco(mask_path, polygons_path, coco_json_path)
            total_time = time.time() - start_time
            logger.info(
                "Extraction Completed in {:.0f}m {:.0f}s".format(
                    total_time // 60, total_time % 60
                )
            )
            torch.cuda.empty_cache()
            return mask_path.name

        except Exception as e:
            print(f"Processing on the Dask cluster failed due to: {e}")
            raise e

    async def constant_gc(self, interval_seconds):
        while True:
            gc.collect()  # Call garbage collection
            await asyncio.sleep(interval_seconds)  # Wait for the specified interval


def main() -> None:
    arguments = cmd_interface()
    geo_inference = GeoInference(
        model=arguments["model"],
        work_dir=arguments["work_dir"],
        mask_to_vec=arguments["vec"],
        mask_to_coco=arguments["coco"],
        mask_to_yolo=arguments["yolo"],
        multi_gpu=arguments["multi_gpu"],
        device=arguments["device"],
        gpu_id=arguments["gpu_id"],
        num_classes=arguments["classes"],
        prediction_threshold=arguments["prediction_threshold"],
        post_inference=arguments.get("post_inference", False),
        sam_checkpoint_path=arguments.get("sam_checkpoint_path"),
        sam_bpe_path=arguments.get("sam_bpe_path"),
    )
    inference_mask_layer_name = geo_inference(
        inference_input=arguments["image"],
        sensor_name=arguments["sensor_name"],
        bands_requested=arguments["bands_requested"],
        patch_size=arguments["patch_size"],
        workers=arguments["workers"],
        bbox=arguments["bbox"],
    )
    print(inference_mask_layer_name)


if __name__ == "__main__":
    main()
