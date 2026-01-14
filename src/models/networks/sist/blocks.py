# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""Reusable building blocks for SIST."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TwoLayerMLP(nn.Module):
    """Two linear layers with activation after each layer."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden = out_dim if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class RepeatedMLP(nn.Module):
    """Repeat a two-layer MLP block N times."""

    def __init__(self, in_dim: int, out_dim: int, repeats: int) -> None:
        super().__init__()
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        blocks = []
        for i in range(repeats):
            blocks.append(TwoLayerMLP(in_dim if i == 0 else out_dim, out_dim))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ResidualConvBlock2d(nn.Module):
    """Two 2D conv layers with a residual skip."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class ResidualConvStack2d(nn.Module):
    """Optional channel projection + stacked residual blocks."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.proj_in = (
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            if in_channels != hidden_channels
            else nn.Identity()
        )
        self.blocks = nn.Sequential(
            *[ResidualConvBlock2d(hidden_channels, kernel_size=kernel_size) for _ in range(num_blocks)]
        )
        self.proj_out = (
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
            if out_channels != hidden_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.blocks(x)
        x = self.proj_out(x)
        return x
