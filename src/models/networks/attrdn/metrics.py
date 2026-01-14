# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
from typing import Optional

import numpy as np
import torch

from .losses import _ssim


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def psnr(pred: np.ndarray, target: np.ndarray, data_range: Optional[float] = None) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float("inf")
    if data_range is None:
        data_range = target.max() - target.min()
        data_range = max(float(data_range), 1e-6)
    return float(20 * np.log10(data_range) - 10 * np.log10(mse))


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: Optional[float] = None,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    if data_range is None:
        data_range = float((target.max() - target.min()).clamp(min=1e-6).item())
    ssim_val, _ = _ssim(pred, target, data_range, window_size, sigma)
    return float(ssim_val.mean().item())


def fbp_reconstruction(
    sinogram: np.ndarray, theta: np.ndarray, filter_name: str = "ramp"
) -> np.ndarray:
    try:
        from skimage.transform import iradon
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for radon/iradon. Install scikit-image."
        ) from exc
    return iradon(sinogram, theta=theta, filter_name=filter_name, circle=False)
