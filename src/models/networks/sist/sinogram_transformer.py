# -*- coding: utf-8 -*-
# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""Sinogram Transformer module for SIST."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .blocks import RepeatedMLP, ResidualConvStack2d


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional shared Q/K/V projection."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_shared: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_shared = qkv_shared
        if qkv_shared:
            self.qkv = nn.Linear(embed_dim, embed_dim)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.k = nn.Linear(embed_dim, embed_dim)
            self.v = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qkv_shared:
            q = k = v = self.qkv(x)
        else:
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)

        q = self._shape(q)
        k = self._shape(k)
        v = self._shape(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(x.shape[0], -1, self.embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """ViT-style transformer encoder layer (pre-norm)."""

    def __init__(
        self,
        self_embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_shared: bool = True,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(self_embed_dim) if use_layer_norm else nn.Identity()
        self.attn = MultiHeadSelfAttention(self_embed_dim, num_heads, qkv_shared=qkv_shared)
        self.norm2 = nn.LayerNorm(self_embed_dim) if use_layer_norm else nn.Identity()
        hidden_dim = int(self_embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self_embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SinogramTransformer(nn.Module):
    """Sinogram Transformer module with head, transformer encoder, and tail."""

    def __init__(
        self,
        detector_bins: int,
        embed_dim: int = 1024,
        mlp_repeats: int = 1,
        num_layers: int = 6,
        num_heads: int = 6,
        qkv_shared: bool = True,
        max_views: int = 1152,
        tail_mlp_repeats: Optional[int] = None,
        tail_residual_blocks: int = 5,
        tail_refine_channels: int = 64,
        use_layer_norm: bool = True,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.detector_bins = detector_bins
        self.embed_dim = embed_dim
        self.max_views = max_views
        self.head_mlp = RepeatedMLP(detector_bins, embed_dim, mlp_repeats)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_views, embed_dim))
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self_embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_shared=qkv_shared,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )
        tail_repeats = mlp_repeats if tail_mlp_repeats is None else tail_mlp_repeats
        self.tail_mlp = RepeatedMLP(embed_dim, detector_bins, tail_repeats)
        self.tail_refine = ResidualConvStack2d(
            in_channels=1,
            hidden_channels=tail_refine_channels,
            out_channels=1,
            num_blocks=tail_residual_blocks,
        )

    def forward(self, sinogram: torch.Tensor) -> torch.Tensor:
        if sinogram.dim() == 4 and sinogram.size(1) == 1:
            sinogram = sinogram.squeeze(1)
        if sinogram.dim() != 3:
            raise ValueError("sinogram must have shape (B, P, D) or (B, 1, P, D)")
        batch_size, num_views, det_bins = sinogram.shape
        if det_bins != self.detector_bins:
            raise ValueError("sinogram detector bins mismatch")
        if num_views > self.max_views:
            raise ValueError("num_views exceeds max_views in positional embedding")

        head = self.head_mlp(sinogram)
        x = head + self.pos_embed[:, :num_views, :]
        for layer in self.layers:
            x = layer(x)
        tail_in = x + head
        s_prime = self.tail_mlp(tail_in)
        refined = self.tail_refine(s_prime.unsqueeze(1)).squeeze(1)
        return refined
