# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
FedM2CT global SpatialNet (adapted from FedM2CT-main).

We keep the feature FiLM modulation (gamma/beta from MLP),
using a default feature vector of length 8 (zeros if not provided).
"""
import torch
import torch.nn as nn


def conv3x3(in_chn, out_chn, bias=True):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)


class SA_layer(nn.Module):
    """Channel attention from the original utils.py."""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.filter1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln1 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.filter2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.ln2 = nn.GroupNorm(num_channels=planes, num_groups=1, affine=False)
        self.relu2 = nn.LeakyReLU(0.2, True)
        self.sa = SA_layer(7)

    def forward(self, x):
        out = self.relu1(self.ln1(self.filter1(x)))
        out = self.ln2(self.filter2(out))
        out = self.relu2(out)
        out = self.sa(out) * out
        return out + x


class MLP(nn.Module):
    """Maps feature_vec (len=8) to FiLM gamma/beta."""

    def __init__(self, input_ch, planes, n):
        super().__init__()
        self.planes = planes
        self.num_FiLM = n
        self.model = nn.Sequential(
            nn.Linear(input_ch, int(0.5 * n * planes), bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(int(0.5 * n * planes), 1 * n * planes, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1 * n * planes, 2 * n * planes, bias=True),
        )

    def forward(self, x):
        out = self.model(x)
        out1 = out[:, : self.num_FiLM * self.planes]
        out2 = out[:, self.num_FiLM * self.planes :]
        return out1, out2


class SpatialNet(nn.Module):
    """
    Global SpatialNet (image-domain) from FedM2CT.
    Input: [B,1,H,W], feature_vec: [B,8].
    Output: [B,1,H,W].
    """

    def __init__(self, num_filters: int = 32):
        super().__init__()
        planes = num_filters * 4
        conv_start = [
            nn.Conv2d(1, planes, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=planes, num_groups=1, affine=False),
            nn.LeakyReLU(0.2, True),
        ]
        layer = lambda: nn.Sequential(ResidualBlock(planes), ResidualBlock(planes), ResidualBlock(planes))
        conv_end = [
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, padding=0, bias=True),
        ]
        self.MLP = MLP(8, planes, 4)
        self.model_start = nn.Sequential(*conv_start)
        self.layer0 = layer()
        self.layer1 = layer()
        self.layer2 = layer()
        self.layer3 = layer()
        self.model_end = nn.Sequential(*conv_end)

    def forward(self, x, feature_vec=None):
        b, _, _, _ = x.size()
        if feature_vec is None:
            feature_vec = torch.zeros(b, 8, device=x.device, dtype=x.dtype)
        gamma, beta = self.MLP(feature_vec)
        planes = gamma.shape[1] // 4
        gamma_0 = gamma[:, :planes].view(b, planes, 1, 1)
        beta_0 = beta[:, :planes].view(b, planes, 1, 1)
        gamma_1 = gamma[:, planes : 2 * planes].view(b, planes, 1, 1)
        beta_1 = beta[:, planes : 2 * planes].view(b, planes, 1, 1)
        gamma_2 = gamma[:, 2 * planes : 3 * planes].view(b, planes, 1, 1)
        beta_2 = beta[:, 2 * planes : 3 * planes].view(b, planes, 1, 1)
        gamma_3 = gamma[:, 3 * planes : 4 * planes].view(b, planes, 1, 1)
        beta_3 = beta[:, 3 * planes : 4 * planes].view(b, planes, 1, 1)

        out = self.model_start(x)
        out = self.layer0(out); out = gamma_0 * out + beta_0
        out = self.layer1(out); out = gamma_1 * out + beta_1
        out = self.layer2(out); out = gamma_2 * out + beta_2
        out = self.layer3(out); out = gamma_3 * out + beta_3
        out = self.model_end(out)
        return out


__all__ = ["SpatialNet"]
