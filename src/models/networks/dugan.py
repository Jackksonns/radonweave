# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
DUGAN generator components (simplified for radonweave pipeline).

We re-use the generator from the original DUGAN repo, without the GAN
discriminators. The generator is a RED-CNN–style encoder/decoder with skip
connections. This file is intentionally minimal to avoid extra dependencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DUGANGenerator(nn.Module):
    """
    RED-CNN style generator used in DUGAN.

    Args:
        in_channels: input channels (LoDoPaB uses 1)
        out_channels: feature width (original DUGAN default 32)
        num_layers: number of conv pairs in encoder/decoder (original 10)
        kernel_size: convolution kernel size (original 3)
        padding: padding size (original 1)
    """

    def __init__(self, in_channels=1, out_channels=32, num_layers=10,
                 kernel_size=3, padding=1):
        super().__init__()
        encoder = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=1, padding=padding)
        ]
        decoder = [
            nn.ConvTranspose2d(out_channels, in_channels,
                               kernel_size=kernel_size, stride=1,
                               padding=padding)
        ]
        for _ in range(num_layers):
            encoder.append(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=kernel_size, stride=1, padding=padding)
            )
            decoder.append(
                nn.ConvTranspose2d(out_channels, out_channels,
                                   kernel_size=kernel_size, stride=1,
                                   padding=padding)
            )
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = []
        for block in self.encoder:
            residuals.append(x)
            x = F.relu(block(x), inplace=True)
        for residual, block in zip(residuals[::-1], self.decoder[::-1]):
            x = F.relu(block(x) + residual, inplace=True)
        return x


__all__ = ["DUGANGenerator"]
