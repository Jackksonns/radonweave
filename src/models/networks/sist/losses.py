# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""Losses for SIST, including SISL components."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .geometry import conjugate_sinogram


def _center_crop(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, h, w = x.shape
    start_h = max((h - target_h) // 2, 0)
    start_w = max((w - target_w) // 2, 0)
    return x[:, start_h : start_h + target_h, start_w : start_w + target_w]


def local_inner_structure_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Local inner-structure loss based on second-order derivatives."""
    if pred.dim() == 4 and pred.size(1) == 1:
        pred = pred.squeeze(1)
    if target.dim() == 4 and target.size(1) == 1:
        target = target.squeeze(1)

    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError("pred/target must have shape (B, P, D) or (B, 1, P, D)")

    diff = pred - target
    d_ii = diff[:, 2:, :] - 2 * diff[:, 1:-1, :] + diff[:, :-2, :]
    d_jj = diff[:, :, 2:] - 2 * diff[:, :, 1:-1] + diff[:, :, :-2]
    d_ij = diff[:, 1:, 1:] - diff[:, 1:, :-1] - diff[:, :-1, 1:] + diff[:, :-1, :-1]

    target_h = min(d_ii.shape[1], d_jj.shape[1], d_ij.shape[1])
    target_w = min(d_ii.shape[2], d_jj.shape[2], d_ij.shape[2])

    d_ii = _center_crop(d_ii, target_h, target_w)
    d_jj = _center_crop(d_jj, target_h, target_w)
    d_ij = _center_crop(d_ij, target_h, target_w)

    value = torch.sqrt(d_ii**2 + d_jj**2 + 2 * d_ij**2 + eps)
    if reduction == "sum":
        return value.sum()
    if reduction == "mean":
        return value.mean()
    raise ValueError("reduction must be 'mean' or 'sum'")


def global_inner_structure_loss(
    pred: torch.Tensor,
    reduction: str = "mean",
    beam: str = "fan",
    angle_coverage: str | float | int = "2pi",
    conjugate_mode: str = "auto",
) -> torch.Tensor:
    """Global inner-structure loss using conjugate sinogram consistency."""
    conj = conjugate_sinogram(
        pred,
        beam=beam,
        angle_coverage=angle_coverage,
        conjugate_mode=conjugate_mode,
    )
    diff = pred - conj
    if reduction == "sum":
        return torch.linalg.norm(diff.reshape(diff.size(0), -1), ord=2, dim=1).sum()
    if reduction == "mean":
        return torch.linalg.norm(diff.reshape(diff.size(0), -1), ord=2, dim=1).mean()
    raise ValueError("reduction must be 'mean' or 'sum'")


def sisl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    local_reduction: str = "mean",
    global_reduction: str = "mean",
    beam: str = "fan",
    angle_coverage: str | float | int = "2pi",
    conjugate_mode: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (SISL, L_C, L_S)."""
    lc = global_inner_structure_loss(
        pred,
        reduction=global_reduction,
        beam=beam,
        angle_coverage=angle_coverage,
        conjugate_mode=conjugate_mode,
    )
    ls = local_inner_structure_loss(pred, target, reduction=local_reduction)
    return lc + ls, lc, ls


class SISTLoss(nn.Module):
    """Total loss for SIST."""

    def __init__(
        self,
        w_sinogram_l1: float = 1.0,
        w_sisl: float = 1.0,
        w_image_l1: float = 1.0,
        w_noise_l1: float = 1.0,
        local_reduction: str = "mean",
        global_reduction: str = "mean",
        beam: str = "fan",
        angle_coverage: str | float | int = "2pi",
        conjugate_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.w_sinogram_l1 = w_sinogram_l1
        self.w_sisl = w_sisl
        self.w_image_l1 = w_image_l1
        self.w_noise_l1 = w_noise_l1
        self.local_reduction = local_reduction
        self.global_reduction = global_reduction
        self.beam = beam
        self.angle_coverage = angle_coverage
        self.conjugate_mode = conjugate_mode
        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred_sinogram: torch.Tensor,
        target_sinogram: torch.Tensor,
        pred_image: torch.Tensor,
        target_image: torch.Tensor,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        sisl, lc, ls = sisl_loss(
            pred_sinogram,
            target_sinogram,
            local_reduction=self.local_reduction,
            global_reduction=self.global_reduction,
            beam=self.beam,
            angle_coverage=self.angle_coverage,
            conjugate_mode=self.conjugate_mode,
        )
        loss_sino = self.l1(pred_sinogram, target_sinogram)
        loss_img = self.l1(pred_image, target_image)
        loss_noise = self.l1(pred_noise, target_noise)

        total = (
            self.w_sinogram_l1 * loss_sino
            + self.w_sisl * sisl
            + self.w_image_l1 * loss_img
            + self.w_noise_l1 * loss_noise
        )
        return {
            "total": total,
            "sinogram_l1": loss_sino,
            "sisl": sisl,
            "lc": lc,
            "ls": ls,
            "image_l1": loss_img,
            "noise_l1": loss_noise,
        }
