# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
Official RED-CNN architecture (from RED-CNN-master/networks.py) adapted for PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class REDCNN(nn.Module):
    def __init__(self, out_channels: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_channels, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_channels, 1, kernel_size=5, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out = out + residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out = out + residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out = out + residual_1
        out = self.relu(out)
        return out


__all__ = ["REDCNN"]
