# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
import torch
from torch import nn


class DenseConnectionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        growth_rate,
        num_layers,
        leaky_relu_slope=0.2,
        use_residual=False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            layer_in_channels = in_channels + idx * growth_rate
            self.layers.append(
                nn.Conv2d(
                    layer_in_channels,
                    growth_rate,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
            )

        self.local_feature_fusion = nn.Conv2d(
            in_channels + num_layers * growth_rate,
            in_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        features = [x]
        for conv in self.layers:
            out = conv(torch.cat(features, dim=1))
            out = self.activation(out)
            features.append(out)

        fused = self.local_feature_fusion(torch.cat(features, dim=1))
        if self.use_residual:
            return fused + x
        return fused
