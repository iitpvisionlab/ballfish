"""
MIT License
Copyright (c) 2024 Arseniy Terekhin
Copyright (c) 2024 Dmitry Sidorchuk
"""

from __future__ import annotations
from typing import Literal, TypeAlias
import torch
from torch import Tensor
import torch.nn.functional as F

Mode: TypeAlias = Literal["bilinear", "nearest", "bicubic"]
PaddingMode: TypeAlias = Literal["zeros", "border", "reflection"]


def projective_transform(
    src: Tensor,
    inverse_matrix: Tensor,
    dst_size: tuple[int, int],
    mode: Mode = "bilinear",
    padding_mode: PaddingMode = "zeros",
) -> Tensor:
    """
    Parameters
    ----------
    src
        N, C, H, W format
    inverse_matrix
        3x3 "destination -> source" matrix
    dst_size
        Output size (height, width)
    mode
        Interpolation mode
    padding_mode
        What value to use for outside pixels
    """
    N, _, H, W = src.size()
    h_out, w_out = dst_size

    space_w = torch.linspace(0.5, w_out - 0.5, w_out, dtype=torch.float32)
    space_h = torch.linspace(0.5, h_out - 0.5, h_out, dtype=torch.float32)

    grid_y, grid_x = torch.meshgrid(space_h, space_w, indexing="ij")
    grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=2)

    grid_p = torch.bmm(
        grid.view(N, h_out * w_out, 3),
        inverse_matrix.unsqueeze(0).transpose(1, 2),
    )

    grid = (grid_p[..., :2] / grid_p[..., 2:]).reshape(N, h_out, w_out, 2)

    # `grid_sample` work in a weird coordinate system
    grid *= 2.0
    w_h = torch.tensor((W, H))
    grid = (grid - w_h) / w_h

    return F.grid_sample(
        src, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )
