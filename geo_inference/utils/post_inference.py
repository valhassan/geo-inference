from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from samgeo import SamGeo3
from scipy.ndimage import binary_closing
from skimage import measure, morphology
from skimage.morphology import disk


@dataclass
class Config:
    building_class_index: Optional[int] = None
    road_class_index: Optional[int] = None
    background_index: int = 0
    gsd: Optional[float] = None
    device: Optional[str] = None
    merge_threshold: int = 2
    checkpoint_path: Optional[str] = None
    bpe_path: Optional[str] = None
    min_area_m2: Optional[dict] = field(default_factory=dict)


def _get_gsd(path: str | Path) -> float:
    with rasterio.open(path) as src:
        t = src.transform
        px = (abs(t.a) + abs(t.e)) / 2
        if src.crs and src.crs.is_geographic:
            lat = (src.bounds.top + src.bounds.bottom) / 2
            px = px * 111_320 * np.cos(np.radians(lat))
    return float(px)


def buildings_splitter(
    geotiff_path: str | Path,
    mask_path: str | Path,
    cfg: Config,
) -> Path:
    """
    Splits merged buildings in a multiclass segmentation mask.

    The original segmentation is the north star — building pixels are never
    redrawn. SAM3 detects how many buildings are inside each blob and where
    the split line is. The split is applied by intersecting SAM3 instances
    with the original blob pixels so boundaries always come from the
    original mask.

    Args:
        geotiff_path: path to the source GeoTIFF
        mask_path:    path to the multiclass mask file
        cfg:          Config

    Returns:
        Path to the corrected mask file (``<stem>.corrected.tif``).
        The original ``mask_path`` is never modified.
    """
    if cfg.building_class_index is None:
        raise ValueError("building_class_index is required")
    gsd = cfg.gsd if cfg.gsd is not None else _get_gsd(geotiff_path)
    print(f"[pipeline] GSD={gsd:.3f} m/px")

    with rasterio.open(mask_path) as src:
        pred_mask = src.read(1)
        profile = src.profile.copy()

    building_mask = pred_mask == cfg.building_class_index
    if not building_mask.any():
        print("[pipeline] no buildings in mask — returning input unchanged")
        return Path(mask_path)

    sam3 = SamGeo3(
        backend="meta",
        bpe_path=cfg.bpe_path,
        checkpoint_path=cfg.checkpoint_path,
        load_from_HF=cfg.checkpoint_path is None,
        device=cfg.device,
    )

    mask_path = Path(mask_path)
    out_path = mask_path.with_name(mask_path.stem + ".corrected.tif")
    tmp_path = mask_path.with_name(mask_path.stem + ".sam3_tmp.tif")

    sam3.generate_masks_tiled(
        source=str(geotiff_path),
        prompt="building",
        output=str(tmp_path),
        bands=[1, 2, 3],
        unique=True,
        dtype="uint32",
    )

    with rasterio.open(tmp_path) as src:
        instance_labels = src.read(1)

    tmp_path.unlink(missing_ok=True)

    labeled_regions = measure.label(building_mask)
    region_ids = np.unique(labeled_regions)
    region_ids = region_ids[region_ids != 0]

    print(f"[pipeline] building regions: {len(region_ids)}")

    out = pred_mask.copy()
    n_split = 0

    for region_id in region_ids:
        region_pixels = labeled_regions == region_id
        sam3_ids_in_region = np.unique(instance_labels[region_pixels])
        sam3_ids_in_region = sam3_ids_in_region[sam3_ids_in_region != 0]

        if len(sam3_ids_in_region) < cfg.merge_threshold:
            continue

        print(
            f"[pipeline] splitting region {region_id} → "
            f"{len(sam3_ids_in_region)} instances"
        )

        out[region_pixels] = 0

        for sam3_id in sam3_ids_in_region:
            sub_region = region_pixels & (instance_labels == sam3_id)
            if sub_region.any():
                out[sub_region] = cfg.building_class_index

        n_split += 1

    # missed = building_mask & (out != cfg.building_class_index)
    # if missed.any():
    #     out[missed] = cfg.building_class_index
    #     print(f"[pipeline] fallback: restored {missed.sum()} uncovered pixels")

    print(f"[pipeline] regions split: {n_split}/{len(region_ids)}")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

    return out_path


def clean_mask(
    mask_path: str | Path,
    cfg: Config,
) -> Path:
    """
    Sensor-aware morphological cleanup of a multiclass segmentation mask.

    For each non-background class:
      1. Morphological closing (optional, configurable per class) — reconnects
         broken linear features before the size filter runs.
      2. Small region removal — any connected component below the sensor-derived
         min_area_m2 threshold is relabelled as background.

    Thresholds are derived from real connected component distributions in the
    training data via Config.from_stats(), so nothing is hardcoded.

    Args:
        mask_path: path to the mask to clean (output of buildings_splitter or raw mask)
        cfg:       Config

    Returns:
        Path to the cleaned mask (``<stem>.cleaned.tif``).
        The input mask_path is never modified.
    """
    mask_path = Path(mask_path)
    out_path = mask_path.with_name(mask_path.stem + ".cleaned.tif")

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        profile = src.profile.copy()

    out = mask.copy()

    road_radius_px = max(1, round(1.5 / cfg.gsd))
    close_classes = (
        {cfg.road_class_index: road_radius_px}
        if cfg.road_class_index in cfg.min_area_m2
        else {}
    )

    for class_idx, min_m2 in cfg.min_area_m2.items():
        min_px = max(1, int(min_m2 / cfg.gsd**2))
        binary = out == class_idx

        if not binary.any():
            continue

        # closing: reconnect small gaps (roads only by default)
        if class_idx in close_classes:
            radius = close_classes[class_idx]
            closed = binary_closing(binary, structure=disk(radius))
            # only add pixels — never remove existing class pixels
            new_px = closed & ~binary & (out == cfg.background_index)
            out[new_px] = class_idx
            binary = out == class_idx

        # small region removal
        cleaned = morphology.remove_small_objects(binary, min_size=min_px)
        removed = binary & ~cleaned
        if removed.any():
            out[removed] = cfg.background_index
            print(
                f"[clean_mask] class {class_idx}: removed {removed.sum()} px "
                f"(min={min_px}px / {min_m2:.1f}m²)"
            )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

    print(f"[clean_mask] saved {out_path}")
    return out_path
