# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""Top-level SIST model definition."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .image_recon import ImageNoiseEstimator
from .sinogram_transformer import SinogramTransformer
from .unet import UNet


class SISTModel(nn.Module):
    """SIST model combining sinogram and image modules."""

    def __init__(
        self,
        sinogram_transformer: SinogramTransformer,
        image_noise_estimator: ImageNoiseEstimator,
        image_denoiser: UNet,
    ) -> None:
        super().__init__()
        self.sinogram_transformer = sinogram_transformer
        self.image_noise_estimator = image_noise_estimator
        self.image_denoiser = image_denoiser

    def forward(self, sinogram_ld: torch.Tensor, image_ld: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_sinogram = self.sinogram_transformer(sinogram_ld)
        sinogram_noise = sinogram_ld - pred_sinogram
        image_noise = self.image_noise_estimator(sinogram_noise, image_ld.shape[-2:])
        pred_image = self.image_denoiser(image_ld - image_noise)
        return {
            "pred_sinogram": pred_sinogram,
            "pred_image": pred_image,
            "pred_noise": image_noise,
        }


def build_sist_model(config: Dict) -> SISTModel:
    model_cfg = config["model"]
    image_noise_cfg = config["image_noise"]
    unet_cfg = config["unet"]

    sinogram_transformer = SinogramTransformer(
        detector_bins=model_cfg["detector_bins"],
        embed_dim=model_cfg["embed_dim"],
        mlp_repeats=model_cfg["mlp_repeats"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        qkv_shared=model_cfg["qkv_shared"],
        max_views=model_cfg["max_views"],
        tail_mlp_repeats=model_cfg["tail_mlp_repeats"],
        tail_residual_blocks=model_cfg["tail_residual_blocks"],
        tail_refine_channels=model_cfg["tail_refine_channels"],
        use_layer_norm=model_cfg["use_layer_norm"],
        mlp_ratio=model_cfg["mlp_ratio"],
    )

    image_noise_estimator = ImageNoiseEstimator(
        conv1d_channels=image_noise_cfg["conv1d_channels"],
        conv1d_kernel=image_noise_cfg["conv1d_kernel"],
        residual_blocks=image_noise_cfg["residual_blocks"],
        residual_channels=image_noise_cfg["residual_channels"],
    )

    image_denoiser = UNet(
        in_channels=unet_cfg["in_channels"],
        out_channels=unet_cfg["out_channels"],
        base_channels=unet_cfg["base_channels"],
        depth=unet_cfg["depth"],
    )
    return SISTModel(sinogram_transformer, image_noise_estimator, image_denoiser)
