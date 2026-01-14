# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
DUGAN Reconstructor (GAN version) integrated into the radonweave Trainer.

Key points:
- Uses LoDoPaB FBP inputs (cache-aware, like other fbp_* models).
- Trains generator + two discriminators (image / gradient) with LS-GAN loss,
  CutMix consistency (CR) and EMA generator (for eval/checkpoint).
- Trainer can stay single-optimizer for other models; a GAN branch is
  activated via `trainer.gan_mode: true` (and recon.gan_mode flag).
"""
import os
import random
from copy import deepcopy
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.networks.dugan import DUGANGenerator
from src.models.networks.dugan_wrapper import UNet, turn_on_spectral_norm
from src.models.reconstructors.standard_learned_reconstructor import StandardLearnedReconstructor
from src.utils.registry import register


def ls_gan(inputs, targets):
    """Least squares GAN loss."""
    if isinstance(targets, (float, int)):
        targets = torch.ones_like(inputs) * targets
    return torch.mean((inputs - targets) ** 2)


class SobelOperator(nn.Module):
    """Sobel magnitude (channel-wise) used in original DUGAN."""

    def __init__(self, epsilon: float = 1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer(
            "conv_x",
            torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)[None, None, :, :] / 4,
        )
        self.register_buffer(
            "conv_y",
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)[None, None, :, :] / 4,
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)
        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)
        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)
        x = x.view(b, c, h, w)
        return x


def warmup(warmup_iter: int, cutmix_prob: float, n_iter: int) -> float:
    return min(n_iter * cutmix_prob / max(1, warmup_iter), cutmix_prob)


def cutmix(mask_size):
    mask = torch.ones(mask_size, device="cuda" if torch.cuda.is_available() else None)
    lam = random.betavariate(1.0, 1.0)
    _, _, height, width = mask_size
    cx = random.uniform(0, width)
    cy = random.uniform(0, height)
    w = width * (1 - lam) ** 0.5
    h = height * (1 - lam) ** 0.5
    x0 = int(max(cx - w / 2, 0))
    x1 = int(min(cx + w / 2, width))
    y0 = int(max(cy - h / 2, 0))
    y1 = int(min(cy + h / 2, height))
    mask[:, :, y0:y1, x0:x1] = 0
    return mask


def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target


@register("reconstructor", "dugan")
class DUGANReconstructor(StandardLearnedReconstructor):
    """
    DUGAN with GAN training (generator + discriminators).
    """

    gan_mode = True  # flag for Trainer to enter GAN branch

    HYPER_PARAMS = {
        # generator
        "dugan_num_layers": {"default": 10, "retrain": True},
        "dugan_num_channels": {"default": 32, "retrain": True},
        "dugan_kernel_size": {"default": 3, "retrain": True},
        "dugan_padding": {"default": 1, "retrain": True},
        "repeat_num": {"default": 6, "retrain": True},  # discriminator depth
        # GAN settings
        "use_grad_discriminator": {"default": True, "retrain": False},
        "d_iter": {"default": 1, "retrain": False},
        "cutmix_prob": {"default": 0.5, "retrain": False},
        "cutmix_warmup_iter": {"default": 1000, "retrain": False},
        "img_gen_loss_weight": {"default": 0.1, "retrain": False},
        "grad_gen_loss_weight": {"default": 0.1, "retrain": False},
        "pix_loss_weight": {"default": 1.0, "retrain": False},
        "grad_loss_weight": {"default": 20.0, "retrain": False},
        "cr_loss_weight": {"default": 1.0, "retrain": False},
        "moving_average": {"default": 0.999, "retrain": False},
        # filters / cache
        "filter_type": {"default": "Hann", "retrain": False},
        "frequency_scaling": {"default": 1.0, "retrain": False},
        # lr for G/D (fallback to optim params if None)
        "g_lr": {"default": None, "retrain": False},
        "d_lr": {"default": None, "retrain": False},
    }

    def __init__(self, ray_trafo=None, **kwargs):
        if ray_trafo is None:
            ray_trafo = kwargs.pop("ray_trafo", None)
        if ray_trafo is None:
            ray_trafo = kwargs.pop("op", None)
        # 避免重复传递 op 给父类
        kwargs.pop("op", None)
        self.dugan_params = {}
        self.loss_params = {}
        for k in [
            "dugan_num_layers",
            "dugan_num_channels",
            "dugan_kernel_size",
            "dugan_padding",
            "repeat_num",
            "loss_type",
            "use_grad_discriminator",
            "d_iter",
            "cutmix_prob",
            "cutmix_warmup_iter",
            "img_gen_loss_weight",
            "grad_gen_loss_weight",
            "pix_loss_weight",
            "grad_loss_weight",
            "cr_loss_weight",
            "moving_average",
            "filter_type",
            "frequency_scaling",
            "g_lr",
            "d_lr",
        ]:
            if k in kwargs:
                self.dugan_params[k] = kwargs.pop(k)
        super().__init__(op=ray_trafo, **kwargs)
        self.hyper_params.update(self.dugan_params)
        self.generator = None
        self.ema_generator = None
        self.img_discriminator = None
        self.grad_discriminator = None
        self.sobel = None
        self.g_optimizer = None
        self.img_d_optimizer = None
        self.grad_d_optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # keep model attr for saving (will set to ema_generator in gan_setup)
        self.model = None

    # ===== Data =====
    def build_dataloaders(self, data_cfg, dataset=None, ray_trafo=None):
        """FBP inputs with optional cache (same pattern as fbp_*)."""
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
            if path in (".", "./", ""):
                cache_files[part] = (None, gt_path)
                continue
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
            part="train", reshape=((1,) + fbp_dataset.space[0].shape, (1,) + fbp_dataset.space[1].shape)
        )
        val_dataset = fbp_dataset.create_torch_dataset(
            part="validation", reshape=((1,) + fbp_dataset.space[0].shape, (1,) + fbp_dataset.space[1].shape)
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

    # ===== Build models / opts =====
    def build_model(self, cfg):
        if self.generator is None:
            self.generator = DUGANGenerator(
                in_channels=1,
                out_channels=self.hyper_params.get("dugan_num_channels", 32),
                num_layers=self.hyper_params.get("dugan_num_layers", 10),
                kernel_size=self.hyper_params.get("dugan_kernel_size", 3),
                padding=self.hyper_params.get("dugan_padding", 1),
            ).to(self.device)
            self.model = self.generator  # temporary; will be overwritten by EMA in gan_setup
        return self.generator

    def gan_setup(self, device, optim_cfg: Dict):
        """Instantiate discriminators, EMA, optimizers (called by Trainer in GAN mode)."""
        self.device = device
        self.build_model(optim_cfg)  # ensure generator exists
        repeat_num = int(self.hyper_params.get("repeat_num", 6))
        self.img_discriminator = turn_on_spectral_norm(
            UNet(repeat_num=repeat_num, use_discriminator=True, conv_dim=64, use_sigmoid=False, in_channels=1)
        ).to(self.device)
        self.grad_discriminator = (
            deepcopy(self.img_discriminator) if self.hyper_params.get("use_grad_discriminator", True) else None
        )
        self.sobel = SobelOperator().to(self.device)
        self.ema_generator = deepcopy(self.generator).to(self.device)
        # lrs
        base_lr = float(optim_cfg.get("params", {}).get("lr", 2e-4))
        g_lr = float(self.hyper_params.get("g_lr") or base_lr)
        d_lr = float(self.hyper_params.get("d_lr") or base_lr)
        weight_decay = float(optim_cfg.get("params", {}).get("weight_decay", 0.0))
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=g_lr, weight_decay=weight_decay)
        self.img_d_optimizer = torch.optim.Adam(self.img_discriminator.parameters(), lr=d_lr)
        if self.grad_discriminator is not None:
            self.grad_d_optimizer = torch.optim.Adam(self.grad_discriminator.parameters(), lr=d_lr)
        else:
            self.grad_d_optimizer = None
        # save EMA as primary model for checkpoints
        self.model = self.ema_generator

    # ===== Helpers =====
    def _update_ema(self):
        m = float(self.hyper_params.get("moving_average", 0.999))
        for old_param, new_param in zip(self.ema_generator.parameters(), self.generator.parameters()):
            old_param.data.mul_(m).add_(new_param.data, alpha=1.0 - m)

    def _pad_for_disc(self, x: torch.Tensor) -> torch.Tensor:
        repeat_num = int(self.hyper_params.get("repeat_num", 6))
        if repeat_num <= 0:
            return x
        multiple = 2 ** repeat_num
        _, _, h, w = x.shape
        pad_h = (multiple - (h % multiple)) % multiple
        pad_w = (multiple - (w % multiple)) % multiple
        if pad_h == 0 and pad_w == 0:
            return x
        # Pad right/bottom so every downsample keeps even shapes in the discriminator.
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    # ===== GAN training step =====
    def gan_training_step(self, batch, n_iter, scaler: torch.amp.GradScaler, use_amp: bool):
        """
        batch: (x=FBP, y=GT)
        Returns dict with key 'loss_g_total' for Trainer logging.
        """
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        self.generator.train()
        self.img_discriminator.train()
        if self.grad_discriminator is not None:
            self.grad_discriminator.train()

        # forward generator
        with torch.amp.autocast("cuda", enabled=use_amp):
            gen_y = self.generator(x)
            grad_gen = self.sobel(gen_y)
            grad_x = self.sobel(x)
            grad_y = self.sobel(y)

        # Discriminator steps
        apply_cutmix = random.random() < warmup(
            int(self.hyper_params.get("cutmix_warmup_iter", 1000)),
            float(self.hyper_params.get("cutmix_prob", 0.5)),
            n_iter,
        )
        cr_w = float(self.hyper_params.get("cr_loss_weight", 1.0))

        def _disc_step(discriminator, optimizer, real_in, fake_in, source_in):
            real_in = self._pad_for_disc(real_in)
            fake_in = self._pad_for_disc(fake_in)
            source_in = self._pad_for_disc(source_in)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                real_enc, real_dec = discriminator(real_in)
                fake_enc, fake_dec = discriminator(fake_in.detach())
                source_enc, source_dec = discriminator(source_in)
                disc_loss = (
                    ls_gan(real_enc, 1.0)
                    + ls_gan(real_dec, 1.0)
                    + ls_gan(fake_enc, 0.0)
                    + ls_gan(fake_dec, 0.0)
                    + ls_gan(source_enc, 0.0)
                    + ls_gan(source_dec, 0.0)
                )
                total_loss = disc_loss
                if apply_cutmix:
                    mask = cutmix(real_dec.size()).to(real_dec)
                    cutmix_input = mask_src_tgt(real_in, fake_in.detach(), mask)
                    cutmix_enc, cutmix_dec = discriminator(cutmix_input)
                    cutmix_loss = ls_gan(cutmix_enc, 0.0) + ls_gan(cutmix_dec, mask)
                    cr_loss = F.mse_loss(cutmix_dec, mask_src_tgt(real_dec, fake_dec, mask))
                    total_loss = total_loss + cutmix_loss + cr_loss * cr_w
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            return total_loss.detach()

        loss_d_img = _disc_step(self.img_discriminator, self.img_d_optimizer, y, gen_y, x)
        loss_d_grad = None
        if self.grad_discriminator is not None:
            loss_d_grad = _disc_step(self.grad_discriminator, self.grad_d_optimizer, grad_y, grad_gen, grad_x)

        # Generator step
        loss_g_total = torch.tensor(0.0, device=self.device)
        if n_iter % int(self.hyper_params.get("d_iter", 1)) == 0:
            self.g_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                gen_y_disc = self._pad_for_disc(gen_y)
                img_gen_enc, img_gen_dec = self.img_discriminator(gen_y_disc)
                img_gen_loss = ls_gan(img_gen_enc, 1.0) + ls_gan(img_gen_dec, 1.0)
                total = img_gen_loss * float(self.hyper_params.get("img_gen_loss_weight", 0.1))

                if self.grad_discriminator is not None:
                    grad_gen_disc = self._pad_for_disc(grad_gen)
                    grad_gen_enc, grad_gen_dec = self.grad_discriminator(grad_gen_disc)
                    grad_gen_loss = ls_gan(grad_gen_enc, 1.0) + ls_gan(grad_gen_dec, 1.0)
                    total = total + grad_gen_loss * float(self.hyper_params.get("grad_gen_loss_weight", 0.1))
                pix_loss = F.mse_loss(gen_y, y)
                grad_loss = F.l1_loss(grad_gen, grad_y)
                total = (
                    total
                    + pix_loss * float(self.hyper_params.get("pix_loss_weight", 1.0))
                    + grad_loss * float(self.hyper_params.get("grad_loss_weight", 20.0))
                )
            scaler.scale(total).backward()
            scaler.step(self.g_optimizer)
            loss_g_total = total.detach()

        scaler.update()
        self._update_ema()

        return {
            "loss_g_total": loss_g_total,
            "loss_d_img": loss_d_img,
            "loss_d_grad": loss_d_grad,
        }

    @torch.no_grad()
    def gan_infer(self, x: torch.Tensor, use_ema: bool = True):
        """Inference for validation/test (uses EMA by default)."""
        model = self.ema_generator if use_ema and self.ema_generator is not None else self.generator
        model.eval()
        with torch.no_grad():
            y_hat = model(x.to(self.device, non_blocking=True))
            return torch.clamp(y_hat, 0.0, 1.0)

    def get_eval_model(self, use_ema: bool = True):
        """Return nn.Module used for saving / eval."""
        return self.ema_generator if use_ema and self.ema_generator is not None else self.generator

    # ===== Loss for val (pixel + grad) =====
    def get_loss_fn(self) -> Callable:
        pix_w = float(self.hyper_params.get("pix_loss_weight", 1.0))
        grad_w = float(self.hyper_params.get("grad_loss_weight", 20.0))
        sobel = SobelOperator()

        def dugan_val_loss(pred: torch.Tensor, target: torch.Tensor):
            pred = pred.float()
            target = target.float()
            _sobel = sobel.to(pred.device)
            loss_pix = F.mse_loss(pred, target)
            loss_grad = F.l1_loss(_sobel(pred), _sobel(target))
            return pix_w * loss_pix + grad_w * loss_grad

        return dugan_val_loss


__all__ = ["DUGANReconstructor"]
