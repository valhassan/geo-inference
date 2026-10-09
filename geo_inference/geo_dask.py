import json
import logging
import threading
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.signal.windows as w
import torch
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

ENTROPY_WEIGHT_FLOOR: float = 1e-2
_CUDA_LOCK = threading.Lock()
_BATCH_CAP: dict = {}
# Input dtypes copied to the device as is and cast to float32 there.
_DEVICE_CAST_DTYPES = (np.uint8, np.int16, np.float16, np.float32)


@lru_cache(maxsize=None)
def _hann_1d(patch_size: int, flat_start: bool, flat_end: bool) -> np.ndarray:
    """1D Hann taper; a half is held at its peak (1.0) where the patch touches the raster edge."""
    step = patch_size >> 1
    win = w.hann(M=patch_size, sym=False).astype(np.float32)
    if flat_start:
        win[:step] = win[step]
    if flat_end:
        win[step:] = win[step]
    win.flags.writeable = False
    return win


@lru_cache(maxsize=64)
def patch_window(
    patch_size: int, first_row: bool, last_row: bool, first_col: bool, last_col: bool
) -> np.ndarray:
    """
    2D blending window for a patch. Separable, so edge and corner variants are the outer
    product of the row/column tapers with their border halves flattened.
    """
    win = np.outer(
        _hann_1d(patch_size, first_row, last_row),
        _hann_1d(patch_size, first_col, last_col),
    )
    win.flags.writeable = False
    return win


@lru_cache(maxsize=64)
def _device_window(
    patch_size: int, first_row: bool, last_row: bool, first_col: bool, last_col: bool, device: str
) -> torch.Tensor:
    """`patch_window` kept resident on the device, so it is not copied over for every batch."""
    return torch.from_numpy(
        patch_window(patch_size, first_row, last_row, first_col, last_col).copy()
    ).to(device)


def _is_nodata(tile: np.ndarray, no_data: Optional[float]) -> bool:
    if no_data is None or np.isnan(no_data):
        return not np.isfinite(tile).any()
    return bool(np.all(tile == no_data))


def _forward(
    model,
    batch: np.ndarray,
    windows: torch.Tensor,
    extra_inputs: list,
    device: str,
    acc: torch.Tensor,
    batch_offsets: list,
    step: int,
) -> None:
    """
    Run one batch through the model, weight its logits by window x entropy confidence and
    overlap-add them into `acc` (C + 1, H, W) on the device, so only the accumulated chunk
    is copied back to the host instead of every patch.
    """
    is_cuda = device.startswith("cuda")
    tensor = torch.from_numpy(batch)
    if is_cuda:
        tensor = tensor.pin_memory()
    with (_CUDA_LOCK if is_cuda else nullcontext()), torch.inference_mode():
        # Copy in the native dtype (e.g. uint8, 4x smaller than float32) and cast on the device.
        tensor = tensor.to(device, non_blocking=True).float()
        y = model(tensor, *extra_inputs)
        if isinstance(y, (tuple, list)):
            y = y[0]
        logits = y.float()  # (B, C, H, W)
        if logits.shape[2:] != windows.shape[1:]:
            raise ValueError(
                f"Model output shape {tuple(logits.shape[1:])} does not match patch {windows.shape[1:]}"
            )
        num_classes = logits.shape[1]
        if num_classes != acc.shape[0] - 1:
            raise ValueError(
                f"Model returned {num_classes} classes, expected {acc.shape[0] - 1}"
            )

        # ENTROPY WEIGHTING
        log_probs = torch.log_softmax(logits, dim=1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)  # (B, H, W)
        if num_classes > 1:
            normalized_entropy = torch.clamp(entropy / float(np.log(num_classes)), 0.0, 1.0)
        else:
            normalized_entropy = torch.zeros_like(entropy)
        blend_factor = (1.0 - normalized_entropy) + ENTROPY_WEIGHT_FLOOR

        weights = windows * blend_factor
        weighted = logits * weights.unsqueeze(1)
        patch_size = windows.shape[1]
        for k, (i, j) in enumerate(batch_offsets):
            rows = slice(i * step, i * step + patch_size)
            cols = slice(j * step, j * step + patch_size)
            acc[:-1, rows, cols] += weighted[k]
            acc[-1, rows, cols] += weights[k]


def runModel(
    chunk_data: np.ndarray,
    model,
    ordered_input: list,
    patch_size: int,
    device: str,
    no_data: Optional[float],
    num_classes: int = 5,
    block_info=None,
    patches_per_chunk: int = 1,
    batch_size: int = 1,
):
    """
    Run the model on every patch of a chunk and overlap-add the windowed logits.

    The chunk spans `patches_per_chunk` stride cells per axis (stride = patch_size / 2) plus the
    right/bottom neighbour's first cell added by map_overlap, so it holds
    `patches_per_chunk` patches per axis (one fewer on the last chunk, which has no neighbour).
    Patches are inferred in batches of up to `batch_size`.

    Returns:
        float16 array (num_classes + 1, (patches_per_chunk + 1) * stride, same): the summed
        weighted logits and, in the last channel, the summed weights. The trailing stride rows
        and columns spill into the next chunks and are combined by `sum_overlapped_chunks`.
    """
    step = patch_size >> 1
    out_size = (patches_per_chunk + 1) * step
    empty = np.zeros((num_classes + 1, out_size, out_size), dtype=np.float16)
    if chunk_data is None or chunk_data.size == 0:
        return empty

    num_chunks = block_info[0]["num-chunks"]
    chunk_location = block_info[0]["chunk-location"]
    # Patch p covers stride cells p and p + 1, so the raster has (cells - 1) patches per axis.
    total_rows = num_chunks[1] * patches_per_chunk - 1
    total_cols = num_chunks[2] * patches_per_chunk - 1
    row0 = chunk_location[1] * patches_per_chunk
    col0 = chunk_location[2] * patches_per_chunk
    n_rows = min(chunk_data.shape[1] // step - 1, patches_per_chunk)
    n_cols = min(chunk_data.shape[2] // step - 1, patches_per_chunk)

    offsets = []
    for i in range(n_rows):
        for j in range(n_cols):
            tile = chunk_data[:, i * step : i * step + patch_size, j * step : j * step + patch_size]
            if not _is_nodata(tile, no_data):
                offsets.append((i, j))
    if not offsets:
        return empty

    try:
        extra_inputs = [
            torch.as_tensor(extra_input, dtype=torch.float32, device=torch.device(device))
            for extra_input in (ordered_input or [])
        ]
        acc = torch.zeros(
            (num_classes + 1, out_size, out_size), dtype=torch.float32, device=device
        )
        start = 0
        while start < len(offsets):
            size = max(1, min(batch_size, _BATCH_CAP.get(device, batch_size)))
            batch_offsets = offsets[start : start + size]
            batch = np.stack(
                [
                    chunk_data[:, i * step : i * step + patch_size, j * step : j * step + patch_size]
                    for i, j in batch_offsets
                ]
            )
            if batch.dtype not in _DEVICE_CAST_DTYPES:
                batch = batch.astype(np.float32)
            windows = torch.stack(
                [
                    _device_window(
                        patch_size,
                        row0 + i == 0,
                        row0 + i == total_rows - 1,
                        col0 + j == 0,
                        col0 + j == total_cols - 1,
                        device,
                    )
                    for i, j in batch_offsets
                ]
            )
            try:
                _forward(
                    model, batch, windows, extra_inputs, device, acc, batch_offsets, step
                )
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                with _CUDA_LOCK:
                    _BATCH_CAP[device] = size // 2
                torch.cuda.empty_cache()
                logger.warning(f"CUDA out of memory with batch {size}; retrying with {size // 2}")
                continue
            start += len(batch_offsets)
        return acc.half().cpu().numpy()

    except Exception as e:
        logging.error(f"Error occured in RunModel: {e}")
        return empty


def sum_overlapped_chunks(
    aoi_chunk: np.ndarray,
    chunk_size: int,
    prediction_threshold: float = 0.3,
    block_info=None,
    patches_per_chunk: int = 1,
):
    """
    Combine a chunk's overlap-added logits with the spill from its top/left neighbours, normalize
    by the summed weights and turn the scores into class labels.

    map_overlap prepends the last stride rows (columns) of the top (left) neighbour, except on
    the first row (column) of chunks, so the offset is read from the block shape.
    @param aoi_chunk: np.ndarray, output of runModel with the top/left overlap.
            chunk_size: int, the patch size.
            prediction_threshold: float, threshold for binary segmentation.
            block_info: none, this is having all the info about the chunk relative to the whole data (dask array)
            patches_per_chunk: int, stride cells per chunk and axis used by runModel.
    @return: reday-to-save chunks
    """
    step = chunk_size // 2
    own = patches_per_chunk * step
    if aoi_chunk is None or aoi_chunk.size == 0:
        return np.zeros((own, own), dtype=np.uint8)

    top = aoi_chunk.shape[1] - (patches_per_chunk + 1) * step
    left = aoi_chunk.shape[2] - (patches_per_chunk + 1) * step
    if top not in (0, step) or left not in (0, step):
        logging.error(
            f" In sum_overlapped_chunks the chunk shape {aoi_chunk.shape} does not match "
            f"patch size {chunk_size} and {patches_per_chunk} patches per chunk"
        )
        return np.zeros((own, own), dtype=np.uint8)

    full_array = aoi_chunk[:, top : top + own, left : left + own].astype(np.float32)
    if top:
        full_array[:, :step, :] += aoi_chunk[:, :step, left : left + own]
    if left:
        full_array[:, :, :step] += aoi_chunk[:, top : top + own, :step]
    if top and left:
        full_array[:, :step, :step] += aoi_chunk[:, :step, :step]

    final_result = np.divide(
        full_array[:-1, :, :],
        full_array[-1, :, :][np.newaxis, :, :],
        out=np.zeros_like(full_array[:-1, :, :]),
        where=full_array[-1, :, :] != 0,
    )
    if final_result.shape[0] == 1:
        return np.where(final_result > prediction_threshold, 1, 0).squeeze(0).astype(np.uint8)
    return np.argmax(final_result, axis=0).astype(np.uint8)


def read_zarr_metadata(metadata_json: Union[Path, str]):
    try:
        with open(metadata_json, "r") as metadat_json:
            metadata = json.load(metadat_json)
            lines = metadata["transform"].strip().split("\n")
            matrix_values = []
            for line in lines:
                values = line.strip("|").split(",")
                matrix_values.extend(float(val.strip()) for val in values)
            # Create and return the Affine object
            trs = Affine(
                matrix_values[0],
                matrix_values[1],
                matrix_values[2],
                matrix_values[3],
                matrix_values[4],
                matrix_values[5],
            )
            metadata.update(
                {
                    "crs": metadata["crs"],
                    "transform": trs,
                    "count": metadata["count"],
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "driver": metadata["driver"],
                    "dtype": metadata["dtype"],
                    "BIGTIFF": metadata.get("BIGTIFF", "Unknown"),
                    "compress": metadata.get("compress", "Unknown"),
                }
            )
            return metadata
    except FileNotFoundError:
        logging.error(f"Error: The file '{metadata_json}' was not found.")
    except json.JSONDecodeError:
        logging.error("Error: Failed to decode JSON from the file.")
