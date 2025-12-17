#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models/meg_encoder2.py

[Re-factored Version]
- Backbone: SimpleConv (Brain Magick style) from File 2.
- Branch: Local-only (Global/Sentence branch removed).
- Application: MEG decoding.
"""
from __future__ import annotations
import math
import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
__all__ = ["UltimateMEGEncoder"]


# ---------------- 基础模块 (Spatial & Subject) ----------------

class FourierEmb(nn.Module):
    def __init__(self, k: int = 32, margin: float = 0.1):
        super().__init__()
        self.k = int(k); self.margin = float(margin)
        fx = torch.arange(1, k + 1).view(1, 1, -1)
        fy = torch.arange(1, k + 1).view(1, 1, -1)
        self.register_buffer("_fx", fx, persistent=False)
        self.register_buffer("_fy", fy, persistent=False)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[..., 0:1] * (1.0 - 2 * self.margin) + self.margin
        y = xy[..., 1:2] * (1.0 - 2 * self.margin) + self.margin
        phase_x = (2 * math.pi * (x * self._fx)).float()
        phase_y = (2 * math.pi * (y * self._fy)).float()
        cosx, sinx = torch.cos(phase_x), torch.sin(phase_x)
        cosy, siny = torch.cos(phase_y), torch.sin(phase_y)
        tgt_dtype = x.dtype
        cosx, sinx = cosx.to(tgt_dtype), sinx.to(tgt_dtype)
        cosy, siny = cosy.to(tgt_dtype), siny.to(tgt_dtype)
        feat1 = torch.einsum("bck,bcm->bckm", cosx, cosy)
        feat2 = torch.einsum("bck,bcm->bckm", sinx, siny)
        return torch.cat(
            [feat1.reshape(x.size(0), x.size(1), -1), feat2.reshape(x.size(0), x.size(1), -1)],
            dim=-1,
        )


class SpatialAttention(nn.Module):
    """
    将可变的传感器位置映射到固定的 spatial_channels (默认270)。
    """
    def __init__(self, spatial_channels: int, fourier_k: int = 32, dropout_p: float = 0.0, dropout_radius: float = 0.2):
        super().__init__()
        self.spatial_channels = int(spatial_channels)
        self.fourier = FourierEmb(fourier_k, margin=0.1)
        pos_dim = 2 * fourier_k * fourier_k
        self.query = nn.Sequential(
            nn.Linear(pos_dim, pos_dim),
            nn.SiLU(),
            nn.Linear(pos_dim, spatial_channels),
        )
        self.dropout_p = float(dropout_p)
        self.dropout_r = float(dropout_radius)

    def _make_mask(self, xy: torch.Tensor) -> torch.Tensor:
        B, C, _ = xy.shape
        if not (self.dropout_p > 0.0 and self.training):
            return torch.zeros(B, C, dtype=torch.bool, device=xy.device)
        if torch.rand(1, device=xy.device).item() > self.dropout_p:
            return torch.zeros(B, C, dtype=torch.bool, device=xy.device)
        cx = torch.rand(B, device=xy.device); cy = torch.rand(B, device=xy.device)
        r = torch.full((B,), self.dropout_r, device=xy.device)
        dx2 = (xy[..., 0] - cx[:, None]) ** 2 + (xy[..., 1] - cy[:, None]) ** 2
        return dx2 <= (r[:, None] ** 2)

    def forward(self, meg_ct: torch.Tensor, sensor_locs: torch.Tensor) -> torch.Tensor:
        xy = sensor_locs[..., :2]
        pos_feat = self.fourier(xy)
        q = self.query(pos_feat)
        mask = self._make_mask(xy)
        if mask.any():
            all_true = mask.all(dim=1)
            if all_true.any():
                mask[all_true, 0] = False
            q = q.masked_fill(mask.unsqueeze(-1), float("-inf"))
        bad = torch.isinf(q).all(dim=1, keepdim=True)
        if bad.any():
            q = torch.where(bad.expand_as(q), torch.zeros_like(q), q)
        attn = torch.softmax(q, dim=1)
        attn = torch.nan_to_num(attn)
        return torch.einsum("bct,bcs->bst", meg_ct, attn)  # [B,spatial_channels,T]


class SubjectLayers(nn.Module):
    def __init__(self, channels: int, n_subjects: int):
        super().__init__()
        self.subject_convs = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in range(n_subjects)])
        for conv in self.subject_convs:
            with torch.no_grad():
                conv.weight.zero_()
                ch = conv.weight.shape[0]
                conv.weight[torch.arange(ch), torch.arange(ch), 0] = 1.0
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor, subj_idx: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0: return x
        if (subj_idx == subj_idx[0]).all():
            return self.subject_convs[int(subj_idx[0].item())](x)
        out = torch.empty_like(x)
        for u in torch.unique(subj_idx):
            m = subj_idx == u
            if m.any():
                out[m] = self.subject_convs[int(u.item())](x[m])
        return out


# ---------------- SimpleConv Components (Brain Magick / File 2) ----------------

class DilatedGLUBlock(nn.Module):
    """
    Conv1d -> BatchNorm -> GELU -> GLU -> Dropout -> Residual
    """
    def __init__(self, d_model: int, kernel_size: int, dilation: int, glu_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.glu_mult = glu_mult
        pad = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(d_model, d_model * glu_mult, kernel_size, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(d_model * glu_mult)
        self.act = nn.GELU()
        self.glu = nn.GLU(dim=1) 
        self.dropout = nn.Dropout(dropout)
        
        out_dim = (d_model * glu_mult) // 2
        if out_dim != d_model:
            self.out_proj = nn.Conv1d(out_dim, d_model, 1)
        else:
            self.out_proj = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.glu(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x + residual 


class SimpleConvBackbone(nn.Module):
    """
    Stack of DilatedGLUBlocks with periodic dilation reset.
    """
    def __init__(self, d_model: int, depth: int, kernel_size: int, dilation_period: int, glu_mult: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** (i % dilation_period)
            self.blocks.append(
                DilatedGLUBlock(d_model, kernel_size, dilation, glu_mult, dropout)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------- UltimateMEGEncoder (Local Only) ----------------
class UltimateMEGEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        n_subjects: int,
        spatial_channels: int = 270,        # Standard MEG virtual channels
        fourier_k: int = 32,
        d_model: int = 320,                 # Hidden dimension
        out_channels: int = 1024,           # Output dimension (e.g. for Wav2Vec2 targets)
        # --- SimpleConv Params (From File 2) ---
        backbone_depth: int = 10,           # Deeper than original File 1
        backbone_kernel: int = 3,
        dilation_period: int = 5,           # Reset dilation every 5 layers [1, 2, 4, 8, 16]
        glu_mult: int = 2,
        dropout: float = 0.0,
        # ----------------------------
        subject_layer_pos: Literal["early", "late", "none"] = "early",
        use_subjects: bool = True,
        spatial_dropout_p: float = 0.0,
        spatial_dropout_radius: float = 0.2,
        out_timesteps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning(f"UltimateMEGEncoder: Ignoring unused kwargs: {list(kwargs.keys())}")

        self.in_channels = in_channels
        self.subject_layer_pos = subject_layer_pos
        self.use_subjects = use_subjects
        self.d_model = d_model
        self.out_timesteps = out_timesteps

        # 1. Spatial Merger (Variable Sensors -> Fixed Channels)
        self.spatial = SpatialAttention(spatial_channels, fourier_k, spatial_dropout_p, spatial_dropout_radius)
        
        # 2. Subject Layers (Early: Apply on Spatial Channels)
        if self.use_subjects and subject_layer_pos == "early":
            self.subj_layer = SubjectLayers(spatial_channels, n_subjects)
        else:
            self.subj_layer = None

        # 3. Projection to Hidden Dimension
        # Map from Spatial Channels (270) to Backbone Width (e.g., 320)
        self.pre_linear = nn.Conv1d(spatial_channels, d_model, 1)

        # 4. Backbone (SimpleConv / Brain Magick Architecture)
        self.backbone = SimpleConvBackbone(
            d_model=d_model,
            depth=backbone_depth,
            kernel_size=backbone_kernel,
            dilation_period=dilation_period,
            glu_mult=glu_mult,
            dropout=dropout
        )

        # 5. Output Head (Local Window)
        self.proj = nn.Conv1d(d_model, out_channels, 1)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        if (out_timesteps is None) or (int(out_timesteps) <= 0):
            self.out_pool = nn.Identity()
        else:
            self.out_pool = nn.AdaptiveAvgPool1d(self.out_timesteps)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        meg_win: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
        **kwargs  # Absorb extra args like meg_sent_full if passed by accident
    ) -> torch.Tensor:
        """
        Forward pass for Local Window Encoding.
        Input:
            meg_win: [B, C, T]
            sensor_locs: [B, C, 3]
            subj_idx: [B]
        Output:
            y_local: [B, out_channels, out_timesteps]
        """
        device = next(self.parameters()).device
        meg_win = meg_win.to(device, non_blocking=True)

        # 1. Spatial Attention
        z = self.spatial(meg_win, sensor_locs) # [B, 270, T]

        # 2. Subject Alignment
        if self.subj_layer is not None:
            z = self.subj_layer(z, subj_idx)

        # 3. Projection to Backbone Dim
        z = self.pre_linear(z) # [B, 320, T]

        # 4. Backbone Processing
        z = self.backbone(z)   # [B, 320, T]

        # 5. Output Projection & Pooling
        y_local = self.proj(z)
        y_local = self.out_pool(y_local).contiguous() # [B, 1024, T_out]
        
        return torch.nan_to_num(y_local)