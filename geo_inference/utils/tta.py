from typing import Callable

import kornia as krn
import torch

_RADTTA_CLAHE_CLIP = 10.0
_RADTTA_CLAHE_GRID = (32, 32)


def geometric_tta(
    x: torch.Tensor,
) -> list[tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]]:
    t = krn.geometry.transform

    return [
        (x, lambda y: y),
        (t.hflip(x), lambda y: t.hflip(y)),
        (t.vflip(x), lambda y: t.vflip(y)),
        (
            t.rotate(x, 90.0, mode="bilinear", align_corners=False),
            lambda y: t.rotate(y, -90.0, mode="bilinear", align_corners=False),
        ),
        (
            t.rot180(x, mode="bilinear", align_corners=False),
            lambda y: t.rot180(y, mode="bilinear", align_corners=False),
        ),
        (
            t.rotate(x, 270.0, mode="bilinear", align_corners=False),
            lambda y: t.rotate(y, -270.0, mode="bilinear", align_corners=False),
        ),
    ]


def radiometric_tta(
    x: torch.Tensor,
) -> torch.Tensor:
    clahe = krn.enhance.equalize_clahe(
        x, clip_limit=_RADTTA_CLAHE_CLIP, tile_grid_size=_RADTTA_CLAHE_GRID
    )
    return clahe
