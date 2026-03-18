import json
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.signal.windows as w
import torch
from rasterio.transform import Affine

logger = logging.getLogger(__name__)


def runModel(
    chunk_data: np.ndarray,
    model,
    ordered_input: list,
    patch_size: int,
    device: str,
    no_data: Optional[float],
    num_classes: int = 5,
    block_info=None,
):
    num_chunks = block_info[0]["num-chunks"]
    chunk_location = block_info[0]["chunk-location"]

    if chunk_data is None or chunk_data.size == 0:
        return np.zeros((num_classes + 1, patch_size, patch_size), dtype=np.float16)

    if (
        (no_data is None and not np.isfinite(chunk_data).any())
        or (
            no_data is not None
            and np.isnan(no_data)
            and not np.isfinite(chunk_data).any()
        )
        or (
            no_data is not None
            and not np.isnan(no_data)
            and np.all(chunk_data == no_data)
        )
    ):
        return np.zeros((num_classes + 1, patch_size, patch_size), dtype=np.float16)

    try:
        step = patch_size >> 1
        window = w.hann(M=patch_size, sym=False).astype(np.float32, copy=False)
        window = window[:, np.newaxis] * window[np.newaxis, :]
        final_window = window

        if chunk_location[2] >= num_chunks[2] - 2 and chunk_location[1] == 0:
            window_u = np.vstack(
                [
                    np.tile(window[step : step + 1, :], (step, 1)),
                    window[step:, :],
                ]
            )
            window_r = np.hstack(
                [
                    window[:, :step],
                    np.tile(window[:, step : step + 1], (1, step)),
                ]
            )
            final_window = np.block(
                [
                    [window_u[:step, :step], np.ones((step, step), dtype=np.float32)],
                    [window_r[step:, :step], window_r[step:, step:]],
                ]
            )
        elif chunk_location[2] >= num_chunks[2] - 2 and (
            chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2
        ):
            # left egde window
            final_window = np.hstack(
                [
                    window[:, :step],
                    np.tile(window[:, step : step + 1], (1, step)),
                ]
            )
        elif chunk_location[2] >= num_chunks[2] - 2 and (
            chunk_location[1] >= num_chunks[1] - 2
        ):
            # bottom right window
            window_r = np.hstack(
                [
                    window[:, :step],
                    np.tile(window[:, step : step + 1], (1, step)),
                ]
            )
            window_b = np.vstack(
                [
                    window[:step, :],
                    np.tile(window[step : step + 1, :], (step, 1)),
                ]
            )
            final_window = np.block(
                [
                    [window_r[:step, :step], window_r[:step, step:]],
                    [window_b[step:, :step], np.ones((step, step), dtype=np.float32)],
                ]
            )
        elif chunk_location[1] >= num_chunks[1] - 2 and (
            chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
        ):
            # bottom egde window
            final_window = np.vstack(
                [
                    window[:step, :],
                    np.tile(window[step : step + 1, :], (step, 1)),
                ]
            )
        elif chunk_location[1] >= num_chunks[1] - 2 and chunk_location[2] == 0:
            # bottom left window
            window_l = np.hstack(
                [
                    np.tile(window[:, step : step + 1], (1, step)),
                    window[:, step:],
                ]
            )
            window_b = np.vstack(
                [
                    window[:step, :],
                    np.tile(window[step : step + 1, :], (step, 1)),
                ]
            )
            final_window = np.block(
                [
                    [window_l[:step, :step], window_l[:step, step:]],
                    [np.ones((step, step), dtype=np.float32), window_b[step:, step:]],
                ]
            )
        elif chunk_location[1] == 0 and chunk_location[2] == 0:
            # Top left window
            window_u = np.vstack(
                [
                    np.tile(window[step : step + 1, :], (step, 1)),
                    window[step:, :],
                ]
            )
            window_l = np.hstack(
                [
                    np.tile(window[:, step : step + 1], (1, step)),
                    window[:, step:],
                ]
            )
            final_window = np.block(
                [
                    [np.ones((step, step), dtype=np.float32), window_u[:step, step:]],
                    [window_l[step:, :step], window_l[step:, step:]],
                ]
            )
        elif chunk_location[2] == 0 and (
            chunk_location[1] > 0 and chunk_location[1] < num_chunks[1]
        ):
            # top edge window
            final_window = np.hstack(
                [
                    np.tile(window[:, step : step + 1], (1, step)),
                    window[:, step:],
                ]
            )
        elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2) and (
            chunk_location[1] == 0
        ):
            # top edge window
            final_window = np.vstack(
                [
                    np.tile(window[step : step + 1, :], (step, 1)),
                    window[step:, :],
                ]
            )
        elif (chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 2) and (
            chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 2
        ):
            final_window = window

        tensor = torch.as_tensor(
            chunk_data, device=torch.device(device), dtype=torch.float32
        )
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        if ordered_input:
            extra_inputs = [
                torch.as_tensor(
                    extra_input, dtype=torch.float32, device=torch.device(device)
                )
                for extra_input in ordered_input
            ]
        else:
            extra_inputs = []

        with torch.no_grad():
            if extra_inputs:
                y = model(tensor, *extra_inputs)
            else:
                y = model(tensor)
        out = y[0].float().detach().cpu().numpy()

        if out.shape[1:] == final_window.shape and out.shape[1:] == (
            patch_size,
            patch_size,
        ):
            if out.shape[0] != num_classes:
                return np.zeros(
                    (num_classes + 1, patch_size, patch_size), dtype=np.float16
                )

            result = np.empty((num_classes + 1, patch_size, patch_size), dtype=np.float16)
            result[:-1, :, :] = (out * final_window).astype(np.float16, copy=False)
            result[-1, :, :] = final_window.astype(np.float16, copy=False)
            return result
        return np.zeros((num_classes + 1, patch_size, patch_size), dtype=np.float16)

    except Exception as e:
        logging.error(f"Error occured in RunModel: {e}")
        return np.zeros((num_classes + 1, patch_size, patch_size), dtype=np.float16)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Release unused memory


def sum_overlapped_chunks(
    aoi_chunk: np.ndarray,
    chunk_size: int,
    prediction_threshold: float = 0.3,
    block_info=None,
):
    """
    This function is for summing up the overlapped parts of the patches in order to reduce the edge artifact.
    After running the model, we run this function on neighbor chunks.
    @param aoi_chunk: np.ndarray, this is a chunk of data in dask array.
            aoi_chunk: int, the size of chunk data that we want to feed the model with
            chunk_size: int , the size of each patch on which the model should be run.
            prediction_threshold: float, threshold for binary segmentation.
            block_info: none, this is having all the info about the chunk relative to the whole data (dask array)
    @return: reday-to-save chunks
    """
    if aoi_chunk.size > 0 and aoi_chunk is not None:
        num_chunks = block_info[0]["num-chunks"]
        chunk_location = block_info[0]["chunk-location"]
        full_array = None
        if (chunk_location[1] == 0 or chunk_location[1] == num_chunks[1] - 1) and (
            chunk_location[2] == 0 or chunk_location[2] == num_chunks[2] - 1
        ):
            """ All 4 corners"""
            full_array = aoi_chunk[
                :,
                : int(chunk_size / 2),
                : int(chunk_size / 2),
            ]
        elif (chunk_location[1] == 0 or chunk_location[1] == num_chunks[1] - 1) and (
            chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1
        ):
            """ Top and bottom edges but not corners"""
            full_array = (
                aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
            )
        elif (chunk_location[2] == 0 or chunk_location[2] == num_chunks[2] - 1) and (
            chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
        ):
            """ Left and right edges but not corners"""
            full_array = (
                aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
            )
        elif (chunk_location[2] > 0 and chunk_location[2] < num_chunks[2] - 1) and (
            chunk_location[1] > 0 and chunk_location[1] < num_chunks[1] - 1
        ):
            """ Middle chunks """
            full_array = (
                aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    : int(chunk_size / 2),
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
                + aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    : int(chunk_size / 2),
                ]
                + aoi_chunk[
                    :,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                    int(chunk_size / 2) : int(chunk_size / 2) * 2,
                ]
            )

        if full_array is None:
            return np.zeros((int(chunk_size / 2), int(chunk_size / 2)), dtype=np.uint8)

        if full_array.shape != (
            aoi_chunk.shape[0],
            int(chunk_size / 2),
            int(chunk_size / 2),
        ):
            logging.error(
                f" In sum_overlapped_chunks the shape of full_array is not {(6, int(chunk_size / 2), int(chunk_size / 2))}"
                f" The size of it {full_array.shape}"
            )
            return np.zeros((int(chunk_size / 2), int(chunk_size / 2)), dtype=np.uint8)
        else:
            final_result = np.divide(
                full_array[:-1, :, :],
                full_array[-1, :, :][np.newaxis, :, :],
                out=np.zeros_like(full_array[:-1, :, :], dtype=float),
                where=full_array[-1, :, :] != 0,
            )
            if final_result.shape[0] == 1:
                final_result = (
                    np.where(final_result > prediction_threshold, 1, 0)
                    .squeeze(0)
                    .astype(np.uint8)
                )
            else:
                final_result = np.argmax(final_result, axis=0).astype(np.uint8)
            return final_result


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
