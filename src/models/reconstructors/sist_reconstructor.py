# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
SIST Reconstructor (based on sist-info reproduction).

Uses LoDoPaB sinograms as input and an FBP image as auxiliary input.
Returns reconstructed image for training/validation while keeping full
SIST outputs for loss computation.
"""
import os
from typing import Dict, Callable

import numpy as np
import torch
import torch.nn as nn

from src.models.networks.sist import build_sist_model, SISTLoss
from copy import deepcopy

from src.models.reconstructors.standard_learned_reconstructor import StandardLearnedReconstructor
from src.utils.registry import register


class _SISTPairDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, fbp_dataset, part: str):
        self.base_dataset = base_dataset
        self.fbp_dataset = fbp_dataset
        self.part = part
        self._len = min(base_dataset.get_len(part), fbp_dataset.get_len(part))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        sinogram, gt = self.base_dataset.get_sample(idx, part=self.part)
        fbp, _ = self.fbp_dataset.get_sample(idx, part=self.part)
        sinogram = torch.from_numpy(np.asarray(sinogram)).float()
        fbp = torch.from_numpy(np.asarray(fbp)).float()
        gt = torch.from_numpy(np.asarray(gt)).float()
        if sinogram.ndim == 2:
            sinogram = sinogram.unsqueeze(0)
        if fbp.ndim == 2:
            fbp = fbp.unsqueeze(0)
        if gt.ndim == 2:
            gt = gt.unsqueeze(0)
        return (sinogram, fbp), gt


class SISTWrapper(nn.Module):
    accepts_tuple = True

    def __init__(self, model: nn.Module, fbp_op=None):
        super().__init__()
        self.model = model
        self.fbp_op = fbp_op
        self.last = {}

    def _fbp_from_sinogram(self, sinogram: torch.Tensor) -> torch.Tensor:
        if self.fbp_op is None:
            raise RuntimeError("fbp_op is required to compute image_ld for SIST.")
        if sinogram.dim() == 4 and sinogram.size(1) == 1:
            sinogram = sinogram.squeeze(1)
        if sinogram.dim() == 2:
            sinogram = sinogram.unsqueeze(0)
        sinogram_np = sinogram.detach().cpu().numpy()
        images = []
        for i in range(sinogram_np.shape[0]):
            fbp = self.fbp_op(sinogram_np[i])
            fbp_arr = np.asarray(fbp)
            images.append(torch.from_numpy(fbp_arr))
        image = torch.stack(images, dim=0).unsqueeze(1)
        return image.to(sinogram.device, dtype=sinogram.dtype)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            sinogram_ld, image_ld = inputs
        else:
            sinogram_ld = inputs
            image_ld = self._fbp_from_sinogram(sinogram_ld)
        if sinogram_ld.dim() == 4 and sinogram_ld.size(1) == 1:
            sinogram_ld = sinogram_ld.squeeze(1)
        elif sinogram_ld.dim() == 2:
            sinogram_ld = sinogram_ld.unsqueeze(0)
        outputs = self.model(sinogram_ld, image_ld)
        self.last = {
            "sinogram_ld": sinogram_ld,
            "image_ld": image_ld,
            "pred_sinogram": outputs["pred_sinogram"],
            "pred_image": outputs["pred_image"],
            "pred_noise": outputs["pred_noise"],
        }
        return outputs["pred_image"]


@register("reconstructor", "sist")
class SISTReconstructor(StandardLearnedReconstructor):
    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        # sinogram transformer
        "detector_bins": {"default": 513, "retrain": True},
        "embed_dim": {"default": 512, "retrain": True},
        "mlp_repeats": {"default": 1, "retrain": True},
        "num_layers": {"default": 6, "retrain": True},
        "num_heads": {"default": 8, "retrain": True},
        "qkv_shared": {"default": True, "retrain": True},
        "max_views": {"default": 1000, "retrain": True},
        "tail_mlp_repeats": {"default": 1, "retrain": True},
        "tail_residual_blocks": {"default": 5, "retrain": True},
        "tail_refine_channels": {"default": 64, "retrain": True},
        "use_layer_norm": {"default": True, "retrain": True},
        "mlp_ratio": {"default": 4.0, "retrain": True},
        # image noise estimator
        "conv1d_channels": {"default": 64, "retrain": True},
        "conv1d_kernel": {"default": 3, "retrain": True},
        "residual_blocks": {"default": 5, "retrain": True},
        "residual_channels": {"default": 64, "retrain": True},
        # unet
        "unet_in_channels": {"default": 1, "retrain": True},
        "unet_out_channels": {"default": 1, "retrain": True},
        "unet_base_channels": {"default": 64, "retrain": True},
        "unet_depth": {"default": 4, "retrain": True},
        # loss
        "w_sinogram_l1": {"default": 1.0, "retrain": True},
        "w_sisl": {"default": 1.0, "retrain": True},
        "w_image_l1": {"default": 1.0, "retrain": True},
        "w_noise_l1": {"default": 1.0, "retrain": True},
        "local_reduction": {"default": "mean", "retrain": False},
        "global_reduction": {"default": "mean", "retrain": False},
        "beam": {"default": "parallel", "retrain": False},
        "angle_coverage": {"default": "pi", "retrain": False},
        "conjugate_mode": {"default": "auto", "retrain": False},
        # fbp params
        "filter_type": {"default": "Hann", "retrain": False},
        "frequency_scaling": {"default": 1.0, "retrain": False},
    })

    def __init__(self, ray_trafo=None, **kwargs):
        if ray_trafo is None:
            ray_trafo = kwargs.pop("ray_trafo", None)
        if ray_trafo is None:
            ray_trafo = kwargs.pop("op", None)
        kwargs.pop("op", None)
        self.sist_params = {}
        for key in self.HYPER_PARAMS.keys():
            if key in kwargs:
                self.sist_params[key] = kwargs.pop(key)
        super().__init__(op=ray_trafo, **kwargs)
        self.hyper_params.update(self.sist_params)
        self.model = None
        self._loss = None
        self._wrapper = None
        self._fbp_op = None

    # ===== Data =====
    def build_dataloaders(self, data_cfg, dataset=None, ray_trafo=None):
        from src.data import get_standard_dataset
        from torch.utils.data import DataLoader
        from src.data.fbp_dataset import generate_fbp_cache_files, get_cached_fbp_dataset

        impl = data_cfg.get("impl", "astra_cuda")
        dataset = dataset or get_standard_dataset("lodopab", impl=impl)
        ray_trafo = ray_trafo or dataset.get_ray_trafo(impl=impl)

        cache_files_cfg = data_cfg.get("cache_files", {})
        cache_files_raw = {
            "train": cache_files_cfg.get("train", ("./cache_lodopab_train_fbp.npy", None)),
            "validation": cache_files_cfg.get("validation", ("./cache_lodopab_validation_fbp.npy", None)),
        }
        cache_files = {}
        for k, v in cache_files_raw.items():
            if isinstance(v, str):
                cache_files[k] = (v, None)
            elif isinstance(v, (list, tuple)) and len(v) >= 2:
                cache_files[k] = (v[0], v[1])
            else:
                cache_files[k] = (None, None)
        cache_size = data_cfg.get("cache_size", {})
        if data_cfg.get("generate_cache"):
            generate_fbp_cache_files(
                dataset,
                ray_trafo,
                cache_files,
                size=cache_size,
                filter_type=self.hyper_params.get("filter_type", "Hann"),
                frequency_scaling=self.hyper_params.get("frequency_scaling", 1.0),
            )
        for part, pair in list(cache_files.items()):
            path, gt_path = pair
            if isinstance(path, str) and not os.path.exists(path) and not data_cfg.get("generate_cache"):
                cache_files[part] = (None, gt_path)

        fbp_dataset = get_cached_fbp_dataset(
            dataset,
            ray_trafo,
            cache_files,
            size=cache_size,
            filter_type=self.hyper_params.get("filter_type", "Hann"),
            frequency_scaling=self.hyper_params.get("frequency_scaling", 1.0),
        )

        train_dataset = _SISTPairDataset(dataset, fbp_dataset, part="train")
        val_dataset = _SISTPairDataset(dataset, fbp_dataset, part="validation")

        train_cfg = data_cfg.get("train", {})
        val_cfg = data_cfg.get("val", {})
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.get("batch_size", 4),
            shuffle=train_cfg.get("shuffle", True),
            num_workers=train_cfg.get("num_workers", 0),
            pin_memory=train_cfg.get("pin_memory", True),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_cfg.get("batch_size", 1),
            shuffle=val_cfg.get("shuffle", False),
            num_workers=val_cfg.get("num_workers", 0),
            pin_memory=val_cfg.get("pin_memory", True),
        )
        return train_loader, val_loader

    # ===== Model =====
    def build_model(self, cfg):
        if self.model is not None:
            return self.model
        model_cfg = {
            "model": {
                "detector_bins": int(self.hyper_params.get("detector_bins", 513)),
                "embed_dim": int(self.hyper_params.get("embed_dim", 512)),
                "mlp_repeats": int(self.hyper_params.get("mlp_repeats", 1)),
                "num_layers": int(self.hyper_params.get("num_layers", 6)),
                "num_heads": int(self.hyper_params.get("num_heads", 8)),
                "qkv_shared": bool(self.hyper_params.get("qkv_shared", True)),
                "max_views": int(self.hyper_params.get("max_views", 1000)),
                "tail_mlp_repeats": int(self.hyper_params.get("tail_mlp_repeats", 1)),
                "tail_residual_blocks": int(self.hyper_params.get("tail_residual_blocks", 5)),
                "tail_refine_channels": int(self.hyper_params.get("tail_refine_channels", 64)),
                "use_layer_norm": bool(self.hyper_params.get("use_layer_norm", True)),
                "mlp_ratio": float(self.hyper_params.get("mlp_ratio", 4.0)),
            },
            "image_noise": {
                "conv1d_channels": int(self.hyper_params.get("conv1d_channels", 64)),
                "conv1d_kernel": int(self.hyper_params.get("conv1d_kernel", 3)),
                "residual_blocks": int(self.hyper_params.get("residual_blocks", 5)),
                "residual_channels": int(self.hyper_params.get("residual_channels", 64)),
            },
            "unet": {
                "in_channels": int(self.hyper_params.get("unet_in_channels", 1)),
                "out_channels": int(self.hyper_params.get("unet_out_channels", 1)),
                "base_channels": int(self.hyper_params.get("unet_base_channels", 64)),
                "depth": int(self.hyper_params.get("unet_depth", 4)),
            },
        }
        core_model = build_sist_model(model_cfg)
        from odl.tomo import fbp_op

        self._fbp_op = fbp_op(
            self.op,
            filter_type=self.hyper_params.get("filter_type", "Hann"),
            frequency_scaling=self.hyper_params.get("frequency_scaling", 1.0),
        )
        self._wrapper = SISTWrapper(core_model, fbp_op=self._fbp_op)
        self.model = self._wrapper
        return self.model

    # ===== Loss =====
    def get_loss_fn(self) -> Callable:
        if self._loss is None:
            self._loss = SISTLoss(
                w_sinogram_l1=float(self.hyper_params.get("w_sinogram_l1", 1.0)),
                w_sisl=float(self.hyper_params.get("w_sisl", 1.0)),
                w_image_l1=float(self.hyper_params.get("w_image_l1", 1.0)),
                w_noise_l1=float(self.hyper_params.get("w_noise_l1", 1.0)),
                local_reduction=str(self.hyper_params.get("local_reduction", "mean")),
                global_reduction=str(self.hyper_params.get("global_reduction", "mean")),
                beam=str(self.hyper_params.get("beam", "parallel")),
                angle_coverage=self.hyper_params.get("angle_coverage", "pi"),
                conjugate_mode=str(self.hyper_params.get("conjugate_mode", "auto")),
            )

        def loss_fn(pred_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
            if self._wrapper is None or not self._wrapper.last:
                raise RuntimeError("SISTWrapper does not have cached outputs for loss.")
            last = self._wrapper.last
            pred_sinogram = last["pred_sinogram"]
            pred_noise = last["pred_noise"]
            image_ld = last["image_ld"]
            target_image = target_image.float()
            target_sinogram = self._forward_project(target_image, pred_sinogram.shape[-2:])
            target_noise = image_ld - target_image
            loss_dict = self._loss(
                pred_sinogram,
                target_sinogram,
                last["pred_image"],
                target_image,
                pred_noise,
                target_noise,
            )
            return loss_dict["total"]

        return loss_fn

    def _forward_project(self, image: torch.Tensor, sinogram_shape) -> torch.Tensor:
        if image.dim() == 4 and image.size(1) == 1:
            image = image[:, 0]
        image_np = image.detach().cpu().numpy()
        sinograms = []
        for i in range(image_np.shape[0]):
            sino = self.op(image_np[i])
            sinograms.append(np.asarray(sino))
        sinograms = np.stack(sinograms, axis=0)
        sino_t = torch.from_numpy(sinograms).to(image.device, dtype=image.dtype)
        return sino_t


__all__ = ["SISTReconstructor"]
