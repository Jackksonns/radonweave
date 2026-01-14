# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
AttRDN reconstructor (attrdn-info reproduction) integrated into radonweave.

Pipeline:
- AttRDN denoises sinograms.
- FBP reconstructs an image for evaluation/visualization.
- Loss uses L1 + MS-SSIM on sinograms (target sinogram is forward-projected
  from ground truth image).
"""
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from odl.tomo import fbp_op

from src.models.networks.attrdn import AttRDN, AttRDNConfig
from src.models.networks.attrdn.losses import L1MSSSIMLoss
from src.models.reconstructors.standard_learned_reconstructor import StandardLearnedReconstructor
from src.utils.registry import register


class _AttrDNWrapper(nn.Module):
    def __init__(self, model: nn.Module, fbp_op):
        super().__init__()
        self.model = model
        self.fbp_op = fbp_op
        self.last = {}

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        if sinogram.ndim == 2:
            sinogram = sinogram.unsqueeze(0).unsqueeze(0)
        elif sinogram.ndim == 3:
            sinogram = sinogram.unsqueeze(1)

        denoised = self.model(sinogram)
        self.last["denoised_sinogram"] = denoised

        den_np = denoised.detach().float().cpu().numpy()
        recons = []
        for i in range(den_np.shape[0]):
            fbp = self.fbp_op(den_np[i, 0])
            recons.append(np.asarray(fbp))
        recon = torch.from_numpy(np.stack(recons, axis=0)).unsqueeze(1)
        return recon.to(sinogram.device, dtype=sinogram.dtype)


class _AttrDNLoss:
    def __init__(self, reconstructor, loss_fn):
        self.reconstructor = reconstructor
        self.loss_fn = loss_fn

    def _forward_project(self, target_img: torch.Tensor) -> torch.Tensor:
        op = getattr(self.reconstructor, "non_normed_op", None) or self.reconstructor.op
        img_np = target_img.detach().float().cpu().numpy()
        sinos = []
        for i in range(img_np.shape[0]):
            img = img_np[i, 0] if img_np.ndim == 4 else img_np[i]
            sino = op(img)
            sinos.append(np.asarray(sino))
        sinos = torch.from_numpy(np.stack(sinos, axis=0)).unsqueeze(1)
        return sinos.to(target_img.device, dtype=target_img.dtype)

    def __call__(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        denoised = self.reconstructor._wrapper.last.get("denoised_sinogram")
        if denoised is None:
            raise RuntimeError("AttrDN loss called before forward; missing denoised sinogram.")
        target_sino = self._forward_project(target_img)
        # Compute loss in full precision to avoid AMP-induced NaNs.
        use_cuda = denoised.is_cuda and torch.cuda.is_available()
        if use_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                den = torch.nan_to_num(denoised.float())
                tgt = torch.nan_to_num(target_sino.float())
                return self.loss_fn(den, tgt)
        den = torch.nan_to_num(denoised.float())
        tgt = torch.nan_to_num(target_sino.float())
        return self.loss_fn(den, tgt)


@register("reconstructor", "attrdn")
class AttRDNReconstructor(StandardLearnedReconstructor):
    HYPER_PARAMS = deepcopy(StandardLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        # AttRDN config (attrdn-info)
        "num_features": {"default": 64, "retrain": True},
        "num_blocks": {"default": 4, "retrain": True},
        "num_layers_per_block": {"default": 4, "retrain": True},
        "growth_rate": {"default": 32, "retrain": True},
        "leaky_relu_slope": {"default": 0.01, "retrain": True},
        "attention_reduction": {"default": 16, "retrain": True},
        "attention_spatial_kernel": {"default": 7, "retrain": True},
        "use_local_residual": {"default": False, "retrain": True},
        "use_global_residual": {"default": True, "retrain": True},
        "global_residual_source": {"default": "sfe1", "retrain": True},
        # Loss config (attrdn-info)
        "loss_alpha": {"default": 0.15, "retrain": False},
        "loss_window_size": {"default": 11, "retrain": False},
        "loss_sigma": {"default": 1.5, "retrain": False},
        "loss_weights": {
            "default": [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            "retrain": False,
        },
        "loss_data_range": {"default": None, "retrain": False},
        # FBP params
        "filter_type": {"default": "Hann", "retrain": False},
        "frequency_scaling": {"default": 1.0, "retrain": False},
    })

    def __init__(self, op=None, ray_trafo=None, **kwargs):
        if ray_trafo is None:
            ray_trafo = kwargs.pop("ray_trafo", None)
        if op is None:
            op = kwargs.pop("op", None)
        if op is None:
            op = ray_trafo
        kwargs.pop("op", None)

        if "use_cuda" not in kwargs:
            kwargs["use_cuda"] = True

        attrdn_params = {}
        for key in self.HYPER_PARAMS.keys():
            if key in kwargs:
                attrdn_params[key] = kwargs.pop(key)

        super().__init__(op=op, **kwargs)
        self.hyper_params.update(attrdn_params)
        self.model = None
        self._wrapper = None
        self._loss = None
        self.fbp_op = None

    def build_dataloaders(self, data_cfg, dataset=None, ray_trafo=None):
        from src.data import get_standard_dataset
        from torch.utils.data import DataLoader

        impl = data_cfg.get("impl", "astra_cuda")
        dataset = dataset or get_standard_dataset("lodopab", impl=impl)

        train_dataset = dataset.create_torch_dataset(
            part="train",
            reshape=((1,) + dataset.space[0].shape, (1,) + dataset.space[1].shape),
        )
        val_dataset = dataset.create_torch_dataset(
            part="validation",
            reshape=((1,) + dataset.space[0].shape, (1,) + dataset.space[1].shape),
        )

        train_cfg = data_cfg.get("train", {})
        val_cfg = data_cfg.get("val", {})

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.get("batch_size", 2),
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

    def build_model(self, cfg):
        config = AttRDNConfig(
            in_channels=1,
            num_features=self.hyper_params["num_features"],
            num_blocks=self.hyper_params["num_blocks"],
            num_layers_per_block=self.hyper_params["num_layers_per_block"],
            growth_rate=self.hyper_params["growth_rate"],
            leaky_relu_slope=self.hyper_params["leaky_relu_slope"],
            attention_reduction=self.hyper_params["attention_reduction"],
            attention_spatial_kernel=self.hyper_params["attention_spatial_kernel"],
            use_local_residual=self.hyper_params["use_local_residual"],
            use_global_residual=self.hyper_params["use_global_residual"],
            global_residual_source=self.hyper_params["global_residual_source"],
        )
        core_model = AttRDN(config)

        self.fbp_op = fbp_op(
            self.op,
            filter_type=self.hyper_params["filter_type"],
            frequency_scaling=self.hyper_params["frequency_scaling"],
        )
        self._wrapper = _AttrDNWrapper(core_model, self.fbp_op)

        loss_fn = L1MSSSIMLoss(
            alpha=self.hyper_params["loss_alpha"],
            data_range=self.hyper_params["loss_data_range"],
            window_size=self.hyper_params["loss_window_size"],
            sigma=self.hyper_params["loss_sigma"],
            weights=self.hyper_params["loss_weights"],
        )
        self._loss = _AttrDNLoss(self, loss_fn)
        self.loss_function = self._loss

        self.model = self._wrapper
        return self.model

    def description(self) -> Dict:
        return {
            "name": "attrdn",
            "model": "AttRDN (attrdn-info)",
            "loss": "L1+MS-SSIM (sinogram)",
            "filter_type": self.hyper_params["filter_type"],
            "frequency_scaling": self.hyper_params["frequency_scaling"],
        }
