# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
from typing import Iterable, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window_1d = gauss.view(1, 1, -1)
    window_2d = window_1d.transpose(2, 1) @ window_1d
    return window_2d


def _create_window(window_size: int, channels: int, sigma: float, device, dtype):
    window = _gaussian_window(window_size, sigma, device, dtype)
    window = window.expand(channels, 1, window_size, window_size)
    return window


def _ssim(
    x,
    y,
    data_range: float,
    window_size: int,
    sigma: float,
    k1: float = 0.01,
    k2: float = 0.03,
):
    if x.shape != y.shape:
        raise ValueError("Input shapes must match for SSIM.")

    channels = x.size(1)
    window = _create_window(window_size, channels, sigma, x.device, x.dtype)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=channels)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(x * x, window, padding=window_size // 2, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(y * y, window, padding=window_size // 2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=channels) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    )
    cs_map = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)

    ssim_val = ssim_map.mean(dim=[1, 2, 3])
    cs_val = cs_map.mean(dim=[1, 2, 3])
    return ssim_val, cs_val


def ms_ssim(
    x,
    y,
    data_range: Optional[float] = None,
    window_size: int = 11,
    sigma: float = 1.5,
    weights: Optional[Iterable[float]] = None,
):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    weights = torch.tensor(weights, device=x.device, dtype=x.dtype)
    levels = weights.numel()

    if data_range is None:
        data_range = (y.max() - y.min()).clamp(min=1e-6)
        data_range = float(data_range.item())

    mcs = []
    for level in range(levels):
        ssim_val, cs_val = _ssim(x, y, data_range, window_size, sigma)
        if level < levels - 1:
            mcs.append(cs_val)
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
            y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=0)
        else:
            ssim_last = ssim_val

    mcs_stack = torch.stack(mcs, dim=0)
    eps = 1e-6
    mcs_stack = torch.clamp(mcs_stack, min=eps, max=1.0)
    ssim_last = torch.clamp(ssim_last, min=eps, max=1.0)
    mcs_power = torch.prod(mcs_stack ** weights[:-1].view(-1, 1), dim=0)
    ms_ssim_val = mcs_power * (ssim_last ** weights[-1])
    ms_ssim_val = torch.nan_to_num(ms_ssim_val, nan=0.0, posinf=1.0, neginf=0.0)
    return ms_ssim_val.mean()


class L1MSSSIMLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.15,
        data_range: Optional[float] = None,
        window_size: int = 11,
        sigma: float = 1.5,
        weights: Optional[Iterable[float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.data_range = data_range
        self.window_size = window_size
        self.sigma = sigma
        self.weights = weights

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ms_ssim_val = ms_ssim(
            pred,
            target,
            data_range=self.data_range,
            window_size=self.window_size,
            sigma=self.sigma,
            weights=self.weights,
        )
        return self.alpha * l1 + (1.0 - self.alpha) * (1.0 - ms_ssim_val)
