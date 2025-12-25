#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models/meg_encoder_audio_T.py

CNN-only MEG encoder with deterministic Temporal Pyramid Pooling (TPP).

Design goals:
- Remove all Conformer / Transformer components.
- Retain SpatialAttention and SubjectLayers for light cross-subject adaptation.
- Sentence-level head uses deterministic multi-scale TPP
  (levels = [1, 2, 4, 8] by default).
- Each TPP bin computes (mean | std), followed by an MLP to audio_dim.
- Output sentence tokens: [B, L_total, audio_dim], with strict slot ordering
  identical to the teacher model.
- Time masks: True = padding; mask-aware statistics.
- Optional L2 normalization on sentence tokens.
- A local window branch is retained for interface compatibility.

This file is intentionally deterministic and reproducible:
no attention over time, no stochastic pooling, no ordering ambiguity.
"""

from __future__ import annotations
import math
from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UltimateMEGEncoderTPP"]


# =============================================================================
# Fourier positional embedding for sensor coordinates
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

        tgt_dtype = xy.dtype
        cosx, sinx = cosx.to(tgt_dtype), sinx.to(tgt_dtype)
        cosy, siny = cosy.to(tgt_dtype), siny.to(tgt_dtype)

        f1 = torch.einsum("bck,bcm->bckm", cosx, cosy)
        f2 = torch.einsum("bck,bcm->bckm", sinx, siny)

        return torch.cat(
            [
                f1.reshape(x.size(0), x.size(1), -1),
                f2.reshape(x.size(0), x.size(1), -1),
            ],
            dim=-1,
        )


# =============================================================================
# Spatial attention (sensor merger)
# =============================================================================
class SpatialAttention(nn.Module):
    """
    Position-driven spatial attention with optional circular dropout.

    Maps variable sensor layouts to a fixed number of spatial channels.
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
        Circular spatial dropout for robustness.
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
        """
        Input:
          meg_ct      : [B, C_in, T]
          sensor_locs : [B, C_in, 2(+)]

        Output:
          merged      : [B, spatial_channels, T]
        """
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
            q = torch.where(bad, torch.zeros_like(q), q)

        attn = torch.softmax(q, dim=1)
        attn = torch.nan_to_num(attn)

        return torch.einsum("bct,bcs->bst", meg_ct, attn)


# =============================================================================
# Subject-specific adaptation layers
# =============================================================================
class SubjectLayers(nn.Module):
    """
    Per-subject 1×1 convolutions, identity-initialized.

    Provides light subject adaptation without altering representation scale.
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
# CNN backbone blocks
# =============================================================================
class PaperDilatedBlock(nn.Module):
    """
    Two-stage dilated residual block followed by GLU projection.
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
# Deterministic Temporal Pyramid Pooling
# =============================================================================
class TemporalPyramidPoolingDet(nn.Module):
    """
    Deterministic Temporal Pyramid Pooling.

    For each level in `levels`, the valid time axis is evenly partitioned
    into bins. Each bin computes (mean | std) in a mask-aware manner.

    Input:
      z_bdt    : [B, D, T]
      time_mask: [B, T], True = padding

    Output:
      tokens   : [B, sum(levels), d_out]
    """
    def __init__(self, d_model: int, levels: List[int], d_out: int, dropout: float = 0.1):
        super().__init__()
        self.levels = list(levels)
        self.L_total = int(sum(levels))
        self.eps = 1e-6

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_out),
        )

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _bin_bounds(Tv: int, n_bins: int, device: torch.device):
        idx = torch.linspace(0, Tv, steps=n_bins + 1, device=device)
        idx = torch.floor(idx).to(torch.long)
        start, end = idx[:-1], idx[1:]
        end = torch.maximum(end, start)
        return start, end

    def forward(
        self,
        z_bdt: torch.Tensor,
        time_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, D, T = z_bdt.shape
        device = z_bdt.device

        valid = (
            torch.ones(B, T, dtype=torch.bool, device=device)
            if time_mask is None
            else ~time_mask.to(device)
        )

        w = valid.to(z_bdt.dtype)
        zw = z_bdt * w.unsqueeze(1)
        z2w = (z_bdt ** 2) * w.unsqueeze(1)

        psum = torch.cumsum(zw, dim=-1)
        psum2 = torch.cumsum(z2w, dim=-1)
        wsum = torch.cumsum(w, dim=-1)

        out_levels = []

        for nb in self.levels:
            toks_b = []
            for b in range(B):
                Tv = int(wsum[b, -1].item())
                if Tv <= 0:
                    toks_b.append(torch.zeros(nb, 2 * D, device=device))
                    continue

                s, e = self._bin_bounds(Tv, nb, device)
                s_idx = torch.floor(s.float() * T / max(1, Tv)).long().clamp(0, T)
                e_idx = torch.floor(e.float() * T / max(1, Tv)).long().clamp(0, T)
                e_idx = torch.maximum(e_idx, s_idx)

                feats = []
                for i in range(nb):
                    si, ei = int(s_idx[i]), int(e_idx[i])
                    if ei <= si:
                        feats.append(torch.zeros(2 * D, device=device))
                        continue

                    if si > 0:
                        sumz = psum[b, :, ei - 1] - psum[b, :, si - 1]
                        sumz2 = psum2[b, :, ei - 1] - psum2[b, :, si - 1]
                        cnt = wsum[b, ei - 1] - wsum[b, si - 1]
                    else:
                        sumz = psum[b, :, ei - 1]
                        sumz2 = psum2[b, :, ei - 1]
                        cnt = wsum[b, ei - 1]

                    cnt = cnt.clamp_min(1.0)
                    mu = sumz / cnt
                    var = (sumz2 / cnt) - mu.square()
                    std = torch.sqrt(var.clamp_min(0.0) + self.eps)
                    feats.append(torch.cat([mu, std], dim=-1))

                toks_b.append(torch.stack(feats, dim=0))

            feat = torch.stack(toks_b, dim=0)
            tok = self.mlp(feat)
            out_levels.append(tok)

        out = torch.cat(out_levels, dim=1)
        return torch.nan_to_num(out)


# =============================================================================
# UltimateMEGEncoderTPP
# =============================================================================
class UltimateMEGEncoderTPP(nn.Module):
    """
    CNN-only MEG encoder with deterministic TPP sentence tokens.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        n_subjects: int,
        spatial_channels: int = 270,
        fourier_k: int = 32,
        d_model: int = 320,
        backbone_depth: int = 5,
        subject_layer_pos: Literal["early", "late", "none"] = "early",
        use_subjects: bool = True,
        dropout: float = 0.1,
        audio_dim: int = 2048,
        tpp_levels: Optional[List[int]] = None,
        tpp_slots: Optional[int] = None,
        out_channels: int = 1024,
        out_timesteps: Optional[int] = None,
        **_ignore,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.audio_dim = int(audio_dim)
        self.levels = tpp_levels or [1, 2, 4, 8]
        self.L = sum(self.levels)

        if (tpp_slots is not None) and (int(tpp_slots) != self.L):
            print(
                f"[warn] tpp_slots({tpp_slots}) != sum(levels)({self.L}); "
                f"using levels={self.levels}"
            )

        # --- Frontend ---
        self.spatial = SpatialAttention(spatial_channels, fourier_k)
        self.pre_S = nn.Conv1d(spatial_channels, spatial_channels, 1)
        self.to_d = nn.Conv1d(spatial_channels, d_model, 1)

        if use_subjects and subject_layer_pos != "none":
            self.subjS = SubjectLayers(spatial_channels, n_subjects)
            self.subjD = SubjectLayers(d_model, n_subjects)
        else:
            self.subjS = None
            self.subjD = None

        # --- Backbone ---
        self.backbone = nn.Sequential(
            *[PaperDilatedBlock(d_model, dropout=0.2) for _ in range(backbone_depth)]
        )

        # --- Sentence-level TPP head ---
        self.tpp = TemporalPyramidPoolingDet(
            d_model, self.levels, d_out=self.audio_dim, dropout=dropout
        )
        self.tok_norm = nn.LayerNorm(self.audio_dim)

        # --- Local branch (compatibility) ---
        self.tail = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, 1),
        )
        self.proj = nn.Conv1d(d_model, out_channels, 1)
        self.out_pool = (
            nn.Identity()
            if out_timesteps is None or int(out_timesteps) <= 0
            else nn.AdaptiveAvgPool1d(int(out_timesteps))
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Sentence-level encoding
    # ------------------------------------------------------------------
    def encode_sentence_tokens(
        self,
        meg_sent_full: torch.Tensor,
        meg_sent_full_mask: Optional[torch.Tensor],
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        x = meg_sent_full.to(device, non_blocking=True)
        loc = sensor_locs.to(device, non_blocking=True)
        s = subj_idx.to(device, non_blocking=True)

        z = self.spatial(x, loc)
        z = self.pre_S(z)
        if self.subjS is not None:
            z = self.subjS(z, s)

        z = self.to_d(z)
        if self.subjD is not None:
            z = self.subjD(z, s)

        z = self.backbone(z)

        tmask = None
        if meg_sent_full_mask is not None:
            m = meg_sent_full_mask.to(device, dtype=torch.bool, non_blocking=True)
            T = z.size(-1)
            if m.size(1) != T:
                if m.size(1) > T:
                    m = m[:, :T]
                else:
                    pad = torch.zeros(
                        m.size(0), T - m.size(1),
                        dtype=torch.bool, device=device
                    )
                    m = torch.cat([m, pad], dim=1)
            tmask = m

        toks = self.tpp(z, time_mask=tmask)
        toks = self.tok_norm(torch.nan_to_num(toks))

        if normalize:
            toks = F.normalize(toks, dim=-1)

        return toks

    # ------------------------------------------------------------------
    # Local window encoding (compatibility path)
    # ------------------------------------------------------------------
    def encode_local_window(
        self,
        meg_win: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        x = meg_win.to(device, non_blocking=True)
        loc = sensor_locs.to(device, non_blocking=True)
        s = subj_idx.to(device, non_blocking=True)

        z = self.spatial(x, loc)
        z = self.pre_S(z)
        if self.subjS is not None:
            z = self.subjS(z, s)

        z = self.to_d(z)
        if self.subjD is not None:
            z = self.subjD(z, s)

        z = self.backbone(z)
        y = self.out_pool(self.proj(self.tail(z))).contiguous()
        return torch.nan_to_num(y)

    def forward(
        self,
        meg_win: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
        *,
        meg_sent_full: Optional[torch.Tensor] = None,
        meg_sent_full_mask: Optional[torch.Tensor] = None,
        return_global: bool = False,
    ):
        y_local = self.encode_local_window(meg_win, sensor_locs, subj_idx)
        if not return_global:
            return y_local

        assert meg_sent_full is not None, "meg_sent_full required for sentence TPP"
        toks = self.encode_sentence_tokens(
            meg_sent_full,
            meg_sent_full_mask,
            sensor_locs,
            subj_idx,
            normalize=True,
        )
        return y_local, toks
