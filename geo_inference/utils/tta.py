from typing import Callable

import kornia as krn
import torch

_RADTTA_CLAHE_CLIP = 10.0
_RADTTA_CLAHE_GRID = (32, 32)
_TTA_INPUT_SCALE = 255.0


def geometric_tta(
    x: torch.Tensor,
) -> list[tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]]:
    """Geometric Augmentation."""
    t = krn.geometry.transform
    x_01 = x / _TTA_INPUT_SCALE

    return [
        (x, lambda y: y),
        (t.hflip(x_01) * _TTA_INPUT_SCALE, lambda y: t.hflip(y)),
        (t.vflip(x_01) * _TTA_INPUT_SCALE, lambda y: t.vflip(y)),
    ]


def radiometric_tta(
    x: torch.Tensor,
) -> torch.Tensor:
    """Radiometric Augmentation."""
    x_01 = x / _TTA_INPUT_SCALE
    clahe = krn.enhance.equalize_clahe(
        x_01, clip_limit=_RADTTA_CLAHE_CLIP, grid_size=_RADTTA_CLAHE_GRID
    )
    return clahe * _TTA_INPUT_SCALE
