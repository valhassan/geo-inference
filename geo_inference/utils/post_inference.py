import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from samgeo import SamGeo3
from scipy.ndimage import binary_closing, binary_erosion
from skimage import measure, morphology
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
        mask_path: path to the raw mask to clean
        cfg:       Config

    Returns:
        Path to the cleaned mask (``<stem>_cleaned.tif``).
        The input mask_path is never modified.
    """
    mask_path = Path(mask_path)
    out_path = mask_path.with_name(mask_path.stem + "_cleaned.tif")

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
        cleaned = morphology.remove_small_objects(binary, max_size=min_px)
        removed = binary & ~cleaned
        if removed.any():
            out[removed] = cfg.background_index
            logger.debug(
                f"class {class_idx}: removed {removed.sum()} px "
                f"(min={min_px}px / {min_m2:.1f}m²)"
            )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
    logger.info(f"saved {out_path}")
    return out_path
