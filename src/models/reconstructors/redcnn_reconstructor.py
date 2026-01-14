# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
RED-CNN Reconstructor integrated with radonweave Trainer.

Uses FBP inputs (cache-aware) and trains the official RED-CNN generator
with MSE loss in the image domain.
"""
import os
import torch
import torch.nn as nn

from src.models.networks.redcnn import REDCNN
from src.models.reconstructors.standard_learned_reconstructor import StandardLearnedReconstructor
from src.utils.registry import register


@register("reconstructor", "redcnn")
class REDCNNReconstructor(StandardLearnedReconstructor):
    HYPER_PARAMS = {
        "out_channels": {"default": 96, "retrain": True},
        "filter_type": {"default": "Hann", "retrain": False},
        "frequency_scaling": {"default": 1.0, "retrain": False},
    }

    def __init__(self, ray_trafo=None, **kwargs):
        if ray_trafo is None:
            ray_trafo = kwargs.pop("ray_trafo", None)
        if ray_trafo is None:
            ray_trafo = kwargs.pop("op", None)
        kwargs.pop("op", None)
        self.redcnn_params = {}
        for k in ["out_channels", "filter_type", "frequency_scaling"]:
            if k in kwargs:
                self.redcnn_params[k] = kwargs.pop(k)
        super().__init__(op=ray_trafo, **kwargs)
        self.hyper_params.update(self.redcnn_params)
        self.model = None
        self.loss_fn = nn.MSELoss()

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
        # normalize cache files to (fbp_path, gt_path)
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

        # If cache file missing and not generating, fall back to on-the-fly FBP
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

        train_dataset = fbp_dataset.create_torch_dataset(
            part="train",
            reshape=((1,) + fbp_dataset.space[0].shape, (1,) + fbp_dataset.space[1].shape),
        )
        val_dataset = fbp_dataset.create_torch_dataset(
            part="validation",
            reshape=((1,) + fbp_dataset.space[0].shape, (1,) + fbp_dataset.space[1].shape),
        )
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
        if self.model is None:
            self.model = REDCNN(out_channels=self.hyper_params.get("out_channels", 96))
        return self.model

    # ===== Loss =====
    def get_loss_fn(self):
        return self.loss_fn


__all__ = ["REDCNNReconstructor"]
