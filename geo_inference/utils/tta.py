from typing import Callable

import kornia as krn
import torch

_RADTTA_CLAHE_CLIP = 10.0
_RADTTA_CLAHE_GRID = (32, 32)
_TTA_INPUT_SCALE = 255.0


def _angle_tensor(
    batch_size: int, angle_deg: float, device: torch.device
) -> torch.Tensor:
    return torch.full((batch_size,), angle_deg, device=device, dtype=torch.float32)


def geometric_tta(
    x: torch.Tensor,
) -> list[tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]]:
    """Geometric Augmentation."""
    t = krn.geometry.transform
    x_01 = x / _TTA_INPUT_SCALE
    B = x.shape[0]
    dev = x.device

    return [
        (x, lambda y: y),
        (t.hflip(x_01) * _TTA_INPUT_SCALE, lambda y: t.hflip(y)),
        (t.vflip(x_01) * _TTA_INPUT_SCALE, lambda y: t.vflip(y)),
        (
            t.rotate(
                x_01,
                _angle_tensor(B, 90.0, dev),
                mode="bilinear",
                align_corners=False,
            )
            * _TTA_INPUT_SCALE,
            lambda y: t.rotate(
                y,
                _angle_tensor(y.shape[0], -90.0, y.device),
                mode="bilinear",
                align_corners=False,
            ),
        ),
        (
            t.rot180(x_01) * _TTA_INPUT_SCALE,
            lambda y: t.rot180(y),
        ),
        (
            t.rotate(
                x_01,
                _angle_tensor(B, 270.0, dev),
                mode="bilinear",
                align_corners=False,
            )
            * _TTA_INPUT_SCALE,
            lambda y: t.rotate(
                y,
                _angle_tensor(y.shape[0], -270.0, y.device),
                mode="bilinear",
                align_corners=False,
            ),
        ),
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
