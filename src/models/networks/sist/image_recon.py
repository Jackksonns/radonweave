# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""Image reconstruction module for SIST."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ResidualConvStack2d


class ImageNoiseEstimator(nn.Module):
    """Estimate image-domain noise from sinogram noise."""

    def __init__(
        self,
        conv1d_channels: int = 64,
        conv1d_kernel: int = 3,
        residual_blocks: int = 5,
        residual_channels: int = 64,
    ) -> None:
        super().__init__()
        padding = conv1d_kernel // 2
        self.conv1d = nn.Conv1d(1, conv1d_channels, kernel_size=conv1d_kernel, padding=padding)
        self.residual_stack = ResidualConvStack2d(
            in_channels=conv1d_channels,
            hidden_channels=residual_channels,
            out_channels=residual_channels,
            num_blocks=residual_blocks,
        )
        self.out_conv = nn.Conv2d(residual_channels, 1, kernel_size=3, padding=1)

    def forward(self, sinogram_noise: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
        if sinogram_noise.dim() == 4 and sinogram_noise.size(1) == 1:
            sinogram_noise = sinogram_noise.squeeze(1)
        if sinogram_noise.dim() != 3:
            raise ValueError("sinogram_noise must have shape (B, P, D) or (B, 1, P, D)")

        batch_size, num_views, det_bins = sinogram_noise.shape
        x = sinogram_noise.reshape(batch_size * num_views, 1, det_bins)
        x = self.conv1d(x)
        x = x.view(batch_size, num_views, -1, det_bins).permute(0, 2, 1, 3)
        x = self.residual_stack(x)
        x = F.interpolate(x, size=image_shape, mode="bilinear", align_corners=False)
        x = self.out_conv(x)
        return x
