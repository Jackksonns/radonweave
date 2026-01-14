# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""Geometry helpers for SISL conjugate sinogram computation."""

from __future__ import annotations

from typing import Tuple

import math

import torch


def _parse_angle_coverage(angle_coverage: str | float | int) -> float:
    if isinstance(angle_coverage, str):
        value = angle_coverage.strip().lower()
        if value in {"pi", "180", "180deg", "180-degree"}:
            return math.pi
        if value in {"2pi", "360", "360deg", "360-degree"}:
            return 2.0 * math.pi
        return float(value)
    return float(angle_coverage)


def _resolve_conjugate_mode(
    num_views: int,
    beam: str,
    angle_coverage: str | float | int,
    conjugate_mode: str,
) -> Tuple[int, bool]:
    mode = conjugate_mode.lower()
    if mode == "auto":
        if beam.lower() == "parallel":
            coverage = _parse_angle_coverage(angle_coverage)
            mode = "flip_only" if coverage <= math.pi + 1e-6 else "shift_and_flip"
        else:
            mode = "shift_and_flip"

    if mode == "shift_and_flip":
        if num_views % 2 != 0:
            raise ValueError("num_views must be even for shift_and_flip conjugate mapping")
        return num_views // 2, True
    if mode == "flip_only":
        return 0, True
    if mode == "identity":
        return 0, False
    raise ValueError(f"Unsupported conjugate_mode: {conjugate_mode}")


def build_conjugate_indices(
    num_views: int,
    det_bins: int,
    view_shift: int = 0,
    flip_detectors: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    view_idx = torch.arange(num_views)
    conj_view_idx = (view_idx + view_shift) % num_views
    if flip_detectors:
        conj_det_idx = torch.arange(det_bins - 1, -1, -1)
    else:
        conj_det_idx = torch.arange(det_bins)
    return conj_view_idx, conj_det_idx


def conjugate_sinogram(
    sinogram: torch.Tensor,
    conj_view_idx: torch.Tensor | None = None,
    conj_det_idx: torch.Tensor | None = None,
    beam: str = "fan",
    angle_coverage: str | float | int = "2pi",
    conjugate_mode: str = "auto",
) -> torch.Tensor:
    """Return conjugate sinogram using precomputed indices or geometry settings."""
    if sinogram.dim() == 4 and sinogram.size(1) == 1:
        sinogram = sinogram.squeeze(1)
    if sinogram.dim() != 3:
        raise ValueError("sinogram must have shape (B, P, D) or (B, 1, P, D)")

    _, num_views, det_bins = sinogram.shape
    if conj_view_idx is None or conj_det_idx is None:
        view_shift, flip_detectors = _resolve_conjugate_mode(
            num_views=num_views,
            beam=beam,
            angle_coverage=angle_coverage,
            conjugate_mode=conjugate_mode,
        )
        conj_view_idx, conj_det_idx = build_conjugate_indices(
            num_views=num_views,
            det_bins=det_bins,
            view_shift=view_shift,
            flip_detectors=flip_detectors,
        )

    conj_view_idx = conj_view_idx.to(sinogram.device)
    conj_det_idx = conj_det_idx.to(sinogram.device)
    return sinogram.index_select(1, conj_view_idx).index_select(2, conj_det_idx)
