# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
"""
DnST (Denoising Swin Transformer) for Low-Dose CT Image Denoising

This module implements the DnST architecture as described in 
"Denoising Swin Transformer and Perceptual Peak Signal-to-Noise Ratio 
for Low-Dose CT Image Denoising" by Boyan Zhang et al.

Key Features:
- Swin Transformer-based architecture for image denoising
- Patch partition with patch_size=1 (pixel-level processing)
- 24 Swin Transformer blocks for feature extraction
- Residual learning framework
- Combined MSE and perceptual loss training

Architecture:
1. Patch Partition (1x1 convolution, no actual patching)
2. Linear Embedding
3. Multiple Swin Transformer Blocks
4. Linear De-embedding
5. Residual Mapping

The network processes low-dose CT images and outputs denoised images
with improved visual quality and structural detail preservation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with Window-based Multi-head Self Attention (W-MSA).
    
    Implements the basic building block of Swin Transformer with:
    - Layer normalization
    - Window-based multi-head self attention
    - MLP (Feed-forward network)
    - Residual connections
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads, default=8
        window_size (int): Window size for W-MSA, default=8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim, default=4.0
        qkv_bias (bool): Enable bias for qkv if True, default=True
        attn_drop (float): Attention dropout rate, default=0.0
        proj_drop (float): Projection dropout rate, default=0.0
        mlp_drop (float): Dropout rate in MLP, default=0.0
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0
    ):
        super(SwinTransformerBlock, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Window-based Multi-head Self Attention
        self.attn = WindowMultiheadAttention(
            dim, num_heads=num_heads, window_size=window_size,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        
        # MLP (Feed-forward network)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, mlp_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Swin Transformer Block.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Output tensor with same shape as input
        """
        # Save shape for reshaping
        B, C, H, W = x.shape
        
        # Reshape for attention (B, C, H, W) -> (B*H*W, C)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Window-based Attention with residual
        attn_out = self.norm1(x_reshaped)
        attn_out = self.attn(attn_out.view(B, H, W, C))
        attn_out = attn_out.view(-1, C)
        x_reshaped = x_reshaped + attn_out
        
        # MLP with residual
        mlp_out = self.norm2(x_reshaped)
        mlp_out = self.mlp(mlp_out)
        x_reshaped = x_reshaped + mlp_out
        
        # Reshape back to (B, C, H, W)
        x = x_reshaped.view(B, H, W, C).permute(0, 3, 1, 2)
        
        return x


class WindowMultiheadAttention(nn.Module):
    """
    Window-based Multi-head Self Attention Module.
    
    Applies multi-head self attention within local windows to reduce
    computational complexity.
    
    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads
        window_size (int): Window size for attention
        qkv_bias (bool): Enable bias for qkv
        attn_drop (float): Attention dropout rate
        proj_drop (float): Projection dropout rate
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super(WindowMultiheadAttention, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, H, W, C)
        
        Returns:
            Output tensor with same shape
        """
        B, H, W, C = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.reshape(B * H * W, C)
        
        # QKV projection
        qkv = self.qkv(x_flat).reshape(B * H * W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(0, 2, 1, 3)  # (B*H*W, num_heads, 3, C//num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Attention output
        x_attn = (attn @ v).transpose(1, 2).reshape(B * H * W, C)
        
        # Projection
        x_out = self.proj(x_attn)
        x_out = self.proj_drop(x_out)
        
        # Reshape back
        x_out = x_out.reshape(B, H, W, C)
        
        return x_out


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-forward Network).
    
    Consists of two linear layers with activation function and dropout.
    
    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features
        out_features (int, optional): Number of output features (defaults to in_features)
        act_layer (nn.Module, optional): Activation function, default=GELU
        drop (float): Dropout rate, default=0.0
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DnST(nn.Module):
    """
    Denoising Swin Transformer (DnST) for Low-Dose CT Image Denoising.
    
    This network combines Swin Transformer blocks with residual learning
    for effective denoising of low-dose CT images while preserving
    structural details.
    
    Architecture:
    1. Patch Partition (Linear Embedding with 1x1 convolution)
    2. Multiple Swin Transformer Blocks
    3. Linear De-embedding (1x1 convolution)
    4. Residual Mapping
    
    Args:
        in_channels (int): Number of input channels (usually 1 for CT), default=1
        out_channels (int): Number of output channels (usually 1), default=1
        patch_size (int): Patch size for embedding (1 for pixel-level), default=1
        num_features (int): Number of feature channels, default=96
        num_swin_blocks (int): Number of Swin Transformer blocks (24 in paper), default=24
        num_heads (int): Number of attention heads in Swin blocks, default=8
        window_size (int): Window size for attention, default=8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim, default=4.0
        qkv_bias (bool): Enable bias for qkv, default=True
        attn_drop (float): Attention dropout rate, default=0.0
        proj_drop (float): Projection dropout rate, default=0.0
        mlp_drop (float): Dropout rate in MLP, default=0.0
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        patch_size: int = 1,
        num_features: int = 96,
        num_swin_blocks: int = 24,
        num_heads: int = 8,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0
    ):
        super(DnST, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.num_features = num_features
        self.num_swin_blocks = num_swin_blocks
        
        # Patch Partition: Linear Embedding with 1x1 convolution
        # This projects from in_channels to num_features without changing spatial dimensions
        self.patch_partition = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, padding=0
        )
        
        # Swin Transformer Blocks
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=num_features,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                mlp_drop=mlp_drop
            )
            for _ in range(num_swin_blocks)
        ])
        
        # Linear De-embedding: Projects back to out_channels
        self.linear_deembedding = nn.Conv2d(
            num_features, out_channels, kernel_size=1, padding=0
        )
        
        # Residual Mapping: For residual learning
        # The network learns the residual (noise) and adds it to input
        self.residual_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DnST.
        
        Args:
            x: Input low-dose CT image (B, C, H, W)
               Typically: (batch_size, 1, height, width)
        
        Returns:
            Denoised image with same shape as input
        """
        # Patch Partition: Linear Embedding
        feat = self.patch_partition(x)
        
        # Swin Transformer Blocks: Feature extraction with residual connections
        res = feat
        for swin_block in self.swin_blocks:
            feat = swin_block(feat)
        
        # Add residual from early features
        feat = feat + res
        
        # Linear De-embedding: Project back to image space
        out = self.linear_deembedding(feat)
        
        # Residual Learning: Output = Input + Network(Input)
        # The network learns to output the noise/residual
        out = x + out * self.residual_weight
        
        return out


class SimpleDnST(nn.Module):
    """
    Simplified DnST for quick testing and validation.
    
    Uses fewer Swin blocks and smaller feature dimensions
    for faster computation while maintaining the core architecture concepts.
    
    Args:
        in_channels (int): Number of input channels, default=1
        out_channels (int): Number of output channels, default=1
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(SimpleDnST, self).__init__()
        
        # Use smaller configuration
        num_features = 48
        num_swin_blocks = 6  # Reduced from 24
        num_heads = 4  # Reduced from 8
        
        # Patch Partition
        self.patch_partition = nn.Conv2d(in_channels, num_features, kernel_size=1)
        
        # Swin Transformer Blocks (simplified)
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=num_features,
                num_heads=num_heads,
                window_size=4,  # Smaller window
                mlp_ratio=2.0  # Smaller ratio
            )
            for _ in range(num_swin_blocks)
        ])
        
        # Linear De-embedding
        self.linear_deembedding = nn.Conv2d(num_features, out_channels, kernel_size=1)
        
        # Residual weight
        self.residual_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of simplified DnST"""
        feat = self.patch_partition(x)
        res = feat
        
        for swin_block in self.swin_blocks:
            feat = swin_block(feat)
        
        feat = feat + res
        out = self.linear_deembedding(feat)
        out = x + out * self.residual_weight
        
        return out


class LightweightDnST(nn.Module):
    """
    Lightweight DnST for resource-constrained environments.
    
    Minimal configuration suitable for inference on CPU or embedded devices.
    
    Args:
        in_channels (int): Number of input channels, default=1
        out_channels (int): Number of output channels, default=1
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(LightweightDnST, self).__init__()
        
        num_features = 32
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=1),
            nn.LayerNorm(num_features),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
        
        # Simple attention block
        self.attention = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size=1)
        )
        
        self.residual_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of lightweight DnST"""
        feat = self.features(x)
        attn = self.attention(feat)
        feat = feat * attn
        out = self.recon(feat)
        return x + out * self.residual_weight
