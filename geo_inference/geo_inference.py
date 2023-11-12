import time
import torch
import logging

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torchgeo.datasets import stack_samples
from utils.polygon import mask_to_poly_geojson, gdf_to_yolo
from utils.helpers import get_device, get_model, get_directory
from geo_blocks import RasterDataset, InferenceSampler, InferenceMerge

from config.logging_config import logger
logger = logging.getLogger(__name__)


class GeoInference:
    """
    A class for performing geo inference on geospatial imagery using a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model to use.
        work_dir (str): The directory where the model and output files will be saved.
        batch_size (int): The batch size to use for inference.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        device (str): The device to use for inference (either "cpu" or "gpu").
        gpu_id (int): The ID of the GPU to use for inference (if device is "gpu").

    Attributes:
        batch_size (int): The batch size to use for inference.
        work_dir (Path): The directory where the model and output files will be saved.
        device (torch.device): The device to use for inference.
        mask_to_vec (bool): Whether to convert the output mask to vector format.
        model (torch.jit.ScriptModule): The pre-trained model to use for inference.
        classes (int): The number of classes in the output of the model.

    """

    def __init__(self,
                 model_name: str = None,
                 work_dir: str = None,
                 batch_size: int = 1,
                 mask_to_vec: bool = False,
                 device: str = "gpu",
                 gpu_id: int = 0):
        self.batch_size = batch_size
        self.work_dir: Path = get_directory(work_dir)
        self.device = get_device(device=device, 
                                 gpu_id=gpu_id)
        model_path: Path = get_model(model_name=model_name, 
                               work_dir=self.work_dir)
        self.mask_to_vec = mask_to_vec
        self.model = torch.jit.load(model_path, map_location=self.device)
        dummy_input = torch.ones((1, 3, 32, 32), device=self.device, dtype=torch.float)
        with torch.no_grad():
            self.classes = self.model(dummy_input).shape[1]
        
    def __call__(self, tiff_image, patch_size: int = 512, stride_size: int = 256) -> None:
        """
        Perform geo inference on geospatial imagery.

        Args:
            tiff_image (str): The path to the geospatial image to perform inference on.
            patch_size (int): The size of the patches to use for inference.
            stride_size (int): The stride to use between patches.

        Returns:
            None

        """
        mask_path = self.work_dir.joinpath(Path(tiff_image).stem + "_mask.tif")
        polygons_path = self.work_dir.joinpath(Path(tiff_image).stem + "_polygons.geojson")
        yolo_csv_path = self.work_dir.joinpath(Path(tiff_image).stem + "_yolo.csv")
        
        dataset = RasterDataset(tiff_image, bounding_box=None)
        sampler = InferenceSampler(dataset, size=patch_size, stride=stride_size)
        roi_height = sampler.im_height 
        roi_width = sampler.im_width
        h_padded, w_padded = roi_height + patch_size, roi_width + patch_size
        output_meta = dataset.src.meta
        merge_patches = InferenceMerge(height=h_padded, width=w_padded, classes=self.classes, device=self.device)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler, collate_fn=stack_samples)
        
        start_time = time.time()
        
        for batch in tqdm(dataloader, desc='extracting features', unit='batch', total=len(dataloader)):
            image_tensor = batch["image"].to(self.device)
            window_tensor = batch["window"].unsqueeze(1).to(self.device)
            pixel_xy = batch["pixel_coords"]
            output = self.model(image_tensor) 
            merge_patches.merge_on_cpu(batch=output, windows=window_tensor, pixel_coords=pixel_xy)
        merge_patches.save_as_tiff(output_meta, mask_path)
        
        if self.mask_to_vec:
            mask_to_poly_geojson(mask_path, polygons_path)
            gdf_to_yolo(polygons_path, mask_path, yolo_csv_path)
        
        end_time = time.time() - start_time
        
        logger.info('Extraction Completed in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))
                    
if __name__ == "__main__":
    pass