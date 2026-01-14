# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
from dataclasses import dataclass

import torch
from torch import nn

from .attention import ChannelAttention, SpatialAttention
from .blocks import DenseConnectionBlock


@dataclass
class AttRDNConfig:
    in_channels: int = 1
    num_features: int = 64
    num_blocks: int = 4
    num_layers_per_block: int = 4
    growth_rate: int = 32
    leaky_relu_slope: float = 0.01
    attention_reduction: int = 16
    attention_spatial_kernel: int = 7
    use_local_residual: bool = False
    use_global_residual: bool = True
    global_residual_source: str = "sfe1"


class AttRDN(nn.Module):
    def __init__(self, config: AttRDNConfig):
        super().__init__()
        self.config = config

        self.sfe1 = nn.Conv2d(
            config.in_channels, config.num_features, kernel_size=3, padding=1, bias=True
        )
        self.sfe2 = nn.Conv2d(
            config.num_features, config.num_features, kernel_size=3, padding=1, bias=True
        )

        self.blocks = nn.ModuleList(
            [
                DenseConnectionBlock(
                    in_channels=config.num_features,
                    growth_rate=config.growth_rate,
                    num_layers=config.num_layers_per_block,
                    leaky_relu_slope=config.leaky_relu_slope,
                    use_residual=config.use_local_residual,
                )
                for _ in range(config.num_blocks)
            ]
        )

        self.global_feature_fusion = nn.Conv2d(
            config.num_features * config.num_blocks,
            config.num_features,
            kernel_size=1,
            padding=0,
            bias=True,
        )

        self.channel_attention = ChannelAttention(
            config.num_features, reduction=config.attention_reduction
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=config.attention_spatial_kernel
        )

        self.out_conv = nn.Conv2d(
            config.num_features, config.in_channels, kernel_size=3, padding=1, bias=True
        )

    def forward(self, x, return_noise=False):
        shallow_1 = self.sfe1(x)
        shallow_2 = self.sfe2(shallow_1)

        features = shallow_2
        block_outputs = []
        for block in self.blocks:
            features = block(features)
            block_outputs.append(features)

        fused = self.global_feature_fusion(torch.cat(block_outputs, dim=1))
        attended = self.channel_attention(fused)
        attended = self.spatial_attention(attended)

        if self.config.use_global_residual:
            if self.config.global_residual_source == "sfe1":
                residual = shallow_1
            elif self.config.global_residual_source == "sfe2":
                residual = shallow_2
            else:
                raise ValueError(
                    "global_residual_source must be 'sfe1' or 'sfe2'."
                )
            attended = attended + residual

        noise = self.out_conv(attended)
        denoised = x - noise

        if return_noise:
            return denoised, noise
        return denoised
