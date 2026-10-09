import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from scipy.ndimage import binary_closing
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage import morphology
from skimage.morphology import disk

logger = logging.getLogger(__name__)


@dataclass
class Config:
    building_class_index: Optional[int] = None
    road_class_index: Optional[int] = None
    background_index: int = 0
    gsd: Optional[float] = None
    min_area_m2: Optional[dict] = field(default_factory=dict)

    @classmethod
    def from_sensor(cls, sensor_meta: dict) -> Optional["Config"]:
        """Return a config when the sensor has area thresholds, else None."""
        areas = sensor_meta.get("min_area_m2")
        if not areas:
            return None
        labels = sensor_meta["class_labels"]
        return cls(
            building_class_index=(
                int(labels["building"]) if "building" in labels else None
            ),
            road_class_index=int(labels["road"]) if "road" in labels else None,
            gsd=sensor_meta["gsd"],
            min_area_m2={int(k): float(v) for k, v in areas.items()},
        )


def _close(
    block: np.ndarray, close_radius: dict, background_index: int
) -> np.ndarray:
    """Closing on background pixels only: reconnects small gaps (roads by default)."""
    out = block.copy()
    for class_idx, radius in close_radius.items():
        binary = out == class_idx
        if not binary.any():
            continue
        closed = binary_closing(binary, structure=disk(radius))
        # only add pixels — never remove existing class pixels
        out[closed & ~binary & (out == background_index)] = class_idx
    return out


def _clean_block(
    block: np.ndarray,
    min_px: dict,
    close_radius: dict,
    background_index: int,
) -> np.ndarray:
    """Whole-array reference cleanup: per class, in order, closing then small region removal."""
    out = block.copy()
    for class_idx, min_size in min_px.items():
        if class_idx in close_radius:
            out = _close(out, {class_idx: close_radius[class_idx]}, background_index)
        binary = out == class_idx
        if binary.any():
            out[binary & ~morphology.remove_small_objects(binary, max_size=min_size)] = (
                background_index
            )
    return out


def _union_edges(a: np.ndarray, b: np.ndarray, off_a: int, off_b: int) -> np.ndarray:
    """Global label pairs of components touching across a shared tile edge."""
    touch = (a > 0) & (b > 0)
    return np.stack([a[touch] + off_a, b[touch] + off_b], axis=1)


def clean_mask(
    mask_path: str | Path,
    cfg: Config,
    chunk_size: int = 4096,
    num_workers: Optional[int] = None,
) -> Path:
    """
    Sensor-aware morphological cleanup of a multiclass segmentation mask.

    For each non-background class:
      1. Morphological closing (optional, configurable per class) — reconnects
         broken linear features before the size filter runs.
      2. Small region removal — any 4-connected component of at most the sensor-derived
         min_area_m2 threshold is relabelled as background.

    Classes are cleaned in ``min_area_m2`` order, so road closing can fill pixels freed by
    classes listed before it.

    The mask is processed in `chunk_size` tiles so memory is bounded by the tile size
    (times `num_workers`) rather than the raster size. Closing is local and only needs a
    small halo. Components are labelled per tile and joined across tile edges, so their
    sizes — and therefore the result — match a whole-raster pass exactly. When road is
    not the first class, the classes before it are cleaned in a first tiled stage.

    Args:
        mask_path: path to the raw mask to clean
        cfg:       Config
        chunk_size: tile size in pixels
        num_workers: threads cleaning tiles concurrently (default: min(8, cpu count))

    Returns:
        Path to the cleaned mask (``<stem>_cleaned.tif``).
        The input mask_path is never modified.
    """
    mask_path = Path(mask_path)
    out_path = mask_path.with_name(mask_path.stem + "_cleaned.tif")
    num_workers = num_workers or min(8, os.cpu_count() or 1)

    min_px = {
        class_idx: max(1, int(min_m2 / cfg.gsd**2))
        for class_idx, min_m2 in cfg.min_area_m2.items()
    }
    road_radius_px = max(1, round(1.5 / cfg.gsd))
    close_radius = (
        {cfg.road_class_index: road_radius_px}
        if cfg.road_class_index in cfg.min_area_m2
        else {}
    )

    # Only closing depends on order (it fills background, including pixels freed by
    # earlier classes), so classes before the road class are cleaned in a first stage.
    classes = list(min_px)
    split = classes.index(cfg.road_class_index) if close_radius else 0
    if split == 0:
        _clean_tiled(mask_path, out_path, min_px, close_radius,
                     cfg.background_index, chunk_size, num_workers)
    else:
        tmp_path = mask_path.with_name(mask_path.stem + "_cleaning.tif")
        try:
            _clean_tiled(mask_path, tmp_path, {k: min_px[k] for k in classes[:split]}, {},
                         cfg.background_index, chunk_size, num_workers)
            _clean_tiled(tmp_path, out_path, {k: min_px[k] for k in classes[split:]},
                         close_radius, cfg.background_index, chunk_size, num_workers)
        finally:
            tmp_path.unlink(missing_ok=True)
    logger.info(f"saved {out_path}")
    return out_path


def _clean_tiled(
    mask_path: Path,
    out_path: Path,
    min_px: dict,
    close_radius: dict,
    background_index: int,
    chunk_size: int,
    num_workers: int,
) -> None:
    """One tiled stage: closing, then exact small region removal for `min_px` classes."""
    halo = 2 * max(close_radius.values()) + 2 if close_radius else 0

    with rasterio.open(mask_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
    tiles = [
        Window(c, r, min(chunk_size, width - c), min(chunk_size, height - r))
        for r in range(0, height, chunk_size)
        for c in range(0, width, chunk_size)
    ]
    n_cols = -(-width // chunk_size)

    def labelled_tile(win: Window) -> tuple[np.ndarray, dict]:
        """Closed core of a tile and its per-class 4-connected labels."""
        with rasterio.open(mask_path) as src:
            outer = Window(win.col_off - halo, win.row_off - halo,
                           win.width + 2 * halo, win.height + 2 * halo)
            outer = outer.intersection(Window(0, 0, width, height))
            block = src.read(1, window=outer)
        r0, c0 = win.row_off - outer.row_off, win.col_off - outer.col_off
        core = _close(block, close_radius, background_index)[
            r0 : r0 + win.height, c0 : c0 + win.width
        ]
        labels = {k: ndimage.label(core == k, output=np.int32) for k in min_px}
        return core, labels

    def summarize(win: Window) -> dict:
        _, labels = labelled_tile(win)
        return {
            k: (n, np.bincount(lab.ravel(), minlength=n + 1)[1:],
                lab[0], lab[-1], lab[:, 0], lab[:, -1])
            for k, (lab, n) in labels.items()
        }

    # Pass 1: per-tile component sizes and border labels.
    with ThreadPoolExecutor(num_workers) as pool:
        summaries = list(pool.map(summarize, tiles))

    # Join components across tile edges and decide which global components to keep.
    keep, offsets = {}, {}
    for k, min_size in min_px.items():
        counts = np.array([s[k][0] for s in summaries])
        offsets[k] = np.concatenate([[0], np.cumsum(counts)[:-1]])
        total = int(counts.sum())
        sizes = np.concatenate([s[k][1] for s in summaries]) if total else np.zeros(0)
        pairs = [np.zeros((0, 2), np.int64)]
        for i, s in enumerate(summaries):
            if (i + 1) % n_cols and i + 1 < len(summaries):  # right neighbour
                pairs.append(_union_edges(s[k][5], summaries[i + 1][k][4],
                                          offsets[k][i], offsets[k][i + 1]))
            if i + n_cols < len(summaries):  # bottom neighbour
                pairs.append(_union_edges(s[k][3], summaries[i + n_cols][k][2],
                                          offsets[k][i], offsets[k][i + n_cols]))
        pairs = np.concatenate(pairs) - 1  # global labels are 1-based
        graph = coo_matrix(
            (np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(total, total)
        )
        _, comp = connected_components(graph, directed=False)
        comp_size = np.bincount(comp, weights=sizes)
        # label 0 (not this class) maps to True so it is never touched
        keep[k] = np.concatenate([[True], comp_size[comp] > min_size])
    del summaries

    # Pass 2: relabel each tile (deterministic) and drop components that are too small.
    def clean_tile(i: int) -> np.ndarray:
        core, labels = labelled_tile(tiles[i])
        for k, (lab, n) in labels.items():
            off = offsets[k][i]
            lut = np.r_[True, keep[k][off + 1 : off + n + 1]]
            core[~lut[lab]] = background_index
        return core

    block = math.gcd(chunk_size, 512)
    profile.update(tiled=block >= 16, blockxsize=block, blockysize=block)
    if not profile["tiled"]:
        profile.pop("blockxsize"), profile.pop("blockysize")
    with rasterio.open(out_path, "w", **profile) as dst, ThreadPoolExecutor(num_workers) as pool:
        # submit in waves so finished tiles never pile up waiting to be written
        for start in range(0, len(tiles), num_workers):
            idx = range(start, min(start + num_workers, len(tiles)))
            for i, data in zip(idx, pool.map(clean_tile, idx)):
                dst.write(data, 1, window=tiles[i])
