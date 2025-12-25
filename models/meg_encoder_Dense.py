#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models/meg_encoder.py

MEG / EEG encoder with sentence-level global readout using TIME-wise
Attentive Statistics Pooling (TIME-ASP).

Key design choices (this version):
- Decouple internal MEG width (d_model) from text-side output dimension (text_dim).
- Sentence-level readout is length-robust: TIME-ASP → Linear(d_model → text_dim).
- Multi-stage temporal downsampling is disabled by default to avoid
  sentence-length leakage (pre_down_tcap=0, token_pool_max_T=0).
- Optional memory slots are retained for analysis / visualization only
  and are not part of the main global path.

This file is a cleaned, documented, and reproducible version of the original
implementation. All numerical behavior and forward logic are preserved.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
__all__ = ["UltimateMEGEncoder", "AttentiveStatsPool1D"]


# =============================================================================
# Basic building blocks
# =============================================================================
class FourierEmb(nn.Module):
    """
    2D Fourier feature embedding for sensor coordinates.

    Input:
      xy : [B, C, 2] normalized sensor coordinates

    Output:
      features : [B, C, 2 * k * k]
    """
    def __init__(self, k: int = 32, margin: float = 0.1):
        super().__init__()
        self.k = int(k)
        self.margin = float(margin)

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
            [
                feat1.reshape(x.size(0), x.size(1), -1),
                feat2.reshape(x.size(0), x.size(1), -1),
            ],
            dim=-1,
        )


class SpatialAttention(nn.Module):
    """
    Spatial attention module that merges variable sensor layouts into
    a fixed number of spatial channels.
    """
    def __init__(
        self,
        spatial_channels: int,
        fourier_k: int = 32,
        dropout_p: float = 0.0,
        dropout_radius: float = 0.2,
    ):
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
        """
        Circular spatial dropout during training.
        """
        B, C, _ = xy.shape
        if not (self.dropout_p > 0.0 and self.training):
            return torch.zeros(B, C, dtype=torch.bool, device=xy.device)

        if torch.rand(1, device=xy.device).item() > self.dropout_p:
            return torch.zeros(B, C, dtype=torch.bool, device=xy.device)

        cx = torch.rand(B, device=xy.device)
        cy = torch.rand(B, device=xy.device)
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

        return torch.einsum("bct,bcs->bst", meg_ct, attn)


class SubjectLayers(nn.Module):
    """
    Subject-specific 1×1 convolutions, identity-initialized.
    """
    def __init__(self, channels: int, n_subjects: int):
        super().__init__()
        self.subject_convs = nn.ModuleList(
            [nn.Conv1d(channels, channels, 1) for _ in range(n_subjects)]
        )

        for conv in self.subject_convs:
            with torch.no_grad():
                conv.weight.zero_()
                ch = conv.weight.shape[0]
                conv.weight[torch.arange(ch), torch.arange(ch), 0] = 1.0
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor, subj_idx: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return x

        if (subj_idx == subj_idx[0]).all():
            return self.subject_convs[int(subj_idx[0].item())](x)

        out = torch.empty_like(x)
        for u in torch.unique(subj_idx):
            m = subj_idx == u
            if m.any():
                out[m] = self.subject_convs[int(u.item())](x[m])
        return out


# =============================================================================
# Backbone blocks
# =============================================================================
class PaperDilatedBlock(nn.Module):
    """
    Two-stage dilated CNN block with GLU projection.
    """
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.res1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.glu = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1),
            nn.GLU(dim=1),
        )
        self.out = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.res1(x)
        y = y + self.res2(y)
        y = self.glu(y)
        return self.out(y)


# =============================================================================
# Attentive Statistics Pooling (TIME-ASP)
# =============================================================================
class AttentiveStatsPool1D(nn.Module):
    """
    Time-wise attentive statistics pooling.

    Input:
      z : [B, D, T]

    Output:
      g : [B, D]

    Computes attention-weighted mean and standard deviation over time,
    followed by a 1×1 projection. This operation is length-agnostic.
    """
    def __init__(self, d_model: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        H = hidden or d_model

        self.attn = nn.Sequential(
            nn.Conv1d(d_model, H, 1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(H, 1, 1),
        )
        self.proj = nn.Conv1d(2 * d_model, d_model, 1)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.eps = 1e-5

    def forward(self, z_bdt: torch.Tensor) -> torch.Tensor:
        e = self.attn(z_bdt)                         # [B,1,T]
        a = torch.softmax(e, dim=-1)                 # [B,1,T]
        mu = torch.sum(a * z_bdt, dim=-1, keepdim=True)
        var = torch.sum(a * (z_bdt - mu) ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        feat = torch.cat([mu, std], dim=1)           # [B,2D,1]
        out = self.proj(feat).squeeze(-1)            # [B,D]
        return torch.nan_to_num(out)


# =============================================================================
# UltimateMEGEncoder (local + sentence global)
# =============================================================================
class UltimateMEGEncoder(nn.Module):
    """
    Full MEG encoder with a length-robust sentence-level global readout.

    Local path:
      MEG window → spatial attention → backbone → local features

    Global path (sentence mode):
      Full sentence MEG → TIME-ASP → Linear(d_model → text_dim)
    """
    def __init__(
        self,
        *,
        in_channels: int,
        n_subjects: int,
        spatial_channels: int = 270,
        fourier_k: int = 32,
        d_model: int = 320,
        text_dim: Optional[int] = None,
        out_channels: int = 1024,
        backbone_depth: int = 5,
        backbone_type: Literal["cnn", "conformer"] = "cnn",
        subject_layer_pos: Literal["early", "late", "none"] = "early",
        use_subjects: bool = True,
        spatial_dropout_p: float = 0.0,
        spatial_dropout_radius: float = 0.2,
        nhead: int = 8,
        dropout: float = 0.1,
        out_timesteps: Optional[int] = None,
        # --- sentence mode ---
        context_mode: Literal["none", "sentence"] = "sentence",
        context_memory_len: int = 12,
        freeze_ctx_local: bool = True,
        detach_context: bool = False,
        global_frontend: Literal["shared", "separate_full"] = "separate_full",
        warm_start_global: bool = False,
        pre_down_tcap: int = 0,
        token_pool_max_T: int = 0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning(
                f"UltimateMEGEncoder: ignoring unused kwargs: {list(kwargs.keys())}"
            )

        self.context_mode = context_mode
        self.subject_layer_pos = subject_layer_pos
        self.use_subjects = use_subjects
        self.d_model = d_model
        self.text_dim = int(text_dim) if text_dim is not None else int(d_model)
        self.detach_context = bool(detach_context)
        self.global_frontend = global_frontend
        self.warm_start_global = bool(warm_start_global)
        self.pre_down_tcap = int(pre_down_tcap)

        # ------------------------------------------------------------------
        # Local frontend
        # ------------------------------------------------------------------
        self.spatial = SpatialAttention(
            spatial_channels,
            fourier_k,
            spatial_dropout_p,
            spatial_dropout_radius,
        )
        self.pre_S = nn.Conv1d(spatial_channels, spatial_channels, 1)
        self.to_d = nn.Conv1d(spatial_channels, d_model, 1)

        if self.use_subjects and subject_layer_pos != "none":
            self.subjS = SubjectLayers(spatial_channels, n_subjects)
            self.subjD = SubjectLayers(d_model, n_subjects)
        else:
            self.subjS = None
            self.subjD = None

        self.backbone = nn.Sequential(
            *[PaperDilatedBlock(d_model, dropout=0.2) for _ in range(backbone_depth)]
        )

        # ------------------------------------------------------------------
        # Global (sentence) frontend
        # ------------------------------------------------------------------
        if self.global_frontend == "shared":
            self.spatial_g = self.spatial
            self.pre_S_g = self.pre_S
            self.subjS_g = self.subjS
            self.subjD_g = self.subjD
            self.to_d_g = self.to_d
            self.backbone_g = self.backbone
        else:
            self.spatial_g = SpatialAttention(
                spatial_channels,
                fourier_k,
                spatial_dropout_p,
                spatial_dropout_radius,
            )
            self.pre_S_g = nn.Conv1d(spatial_channels, spatial_channels, 1)
            self.subjS_g = SubjectLayers(spatial_channels, n_subjects) if self.subjS is not None else None
            self.subjD_g = SubjectLayers(d_model, n_subjects) if self.subjD is not None else None
            self.to_d_g = nn.Conv1d(spatial_channels, d_model, 1)
            self.backbone_g = nn.Sequential(
                *[PaperDilatedBlock(d_model, dropout=0.2) for _ in range(backbone_depth)]
            )

            if self.warm_start_global:
                self._warm_start_from_local_()

        # ------------------------------------------------------------------
        # Sentence-level TIME-ASP head (main global path)
        # ------------------------------------------------------------------
        self.sent_pool_time = AttentiveStatsPool1D(d_model, hidden=d_model, dropout=dropout)
        self.sent_proj = nn.Linear(d_model, self.text_dim)

        nn.init.xavier_uniform_(self.sent_proj.weight)
        nn.init.zeros_(self.sent_proj.bias)

        # ------------------------------------------------------------------
        # Local output head (kept for compatibility)
        # ------------------------------------------------------------------
        self.tail = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.proj = nn.Conv1d(d_model, out_channels, 1)

        if out_timesteps is None or int(out_timesteps) <= 0:
            self.out_pool = nn.Identity()
        else:
            self.out_pool = nn.AdaptiveAvgPool1d(int(out_timesteps))

        self._init()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _warm_start_from_local_(self):
        def _copy(dst: nn.Module, src: nn.Module):
            if dst is None or src is None:
                return
            dst.load_state_dict(src.state_dict())

        _copy(self.spatial_g, self.spatial)
        _copy(self.pre_S_g, self.pre_S)
        _copy(self.subjS_g, self.subjS)
        _copy(self.subjD_g, self.subjD)
        _copy(self.to_d_g, self.to_d)
        _copy(self.backbone_g, self.backbone)

    # ------------------------------------------------------------------
    # Encoding paths
    # ------------------------------------------------------------------
    def _encode_local(
        self,
        x_bct: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
    ) -> torch.Tensor:
        z = self.spatial(x_bct, sensor_locs)
        z = self.pre_S(z)

        if self.subjS is not None and self.subject_layer_pos == "early":
            z = self.subjS(z, subj_idx)

        z = self.to_d(z)

        if self.subjD is not None and self.subject_layer_pos == "late":
            z = self.subjD(z, subj_idx)

        z = self.backbone(z)
        return z

    def _encode_sentence(
        self,
        meg_sent_full: torch.Tensor,
        meg_sent_full_mask: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
    ) -> torch.Tensor:
        z = self.spatial_g(meg_sent_full, sensor_locs)
        z = self.pre_S_g(z)

        if self.subjS_g is not None and self.subject_layer_pos == "early":
            z = self.subjS_g(z, subj_idx)

        z = self.to_d_g(z)

        if self.subjD_g is not None and self.subject_layer_pos == "late":
            z = self.subjD_g(z, subj_idx)

        if self.pre_down_tcap and z.size(-1) > self.pre_down_tcap:
            while z.size(-1) > self.pre_down_tcap:
                z = F.avg_pool1d(z, kernel_size=2, stride=2)

        z = self.backbone_g(z)
        z = torch.nan_to_num(z)

        g = self.sent_pool_time(z)
        g = self.sent_proj(g)
        g = torch.nan_to_num(g)

        return g.detach() if self.detach_context else g

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        meg_win: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
        *,
        meg_sent_full: Optional[torch.Tensor] = None,
        meg_sent_full_mask: Optional[torch.Tensor] = None,
        return_global: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        device = next(self.parameters()).device

        z_local = self._encode_local(
            meg_win.to(device, non_blocking=True),
            sensor_locs,
            subj_idx,
        )
        y_local = self.out_pool(self.proj(self.tail(z_local)))
        y_local = torch.nan_to_num(y_local)

        if not return_global or self.context_mode == "none":
            return y_local

        assert meg_sent_full is not None and meg_sent_full_mask is not None, (
            "Sentence mode requires meg_sent_full and meg_sent_full_mask."
        )

        g = self._encode_sentence(
            meg_sent_full.to(device, non_blocking=True),
            meg_sent_full_mask,
            sensor_locs,
            subj_idx,
        )

        return y_local, g
