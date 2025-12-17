#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models/eeg_encoder.py

Configuration Alignment with Brain Magick (SimpleConv):
- Backbone: SimpleConv (DilatedGLUBlock)
- Dilation: Periodic (1, 2, 4, 8, 16...) per 5 layers
- GLU: Multiplier = 2
- Structure: Spatial Merger (270) -> Subject Layer (270) -> Linear (320) -> Backbone (320)
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


# ---------------- Basic Modules ----------------
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
    Corresponds to 'merger' in config.yaml.
    Maps variable sensors to fixed 'spatial_channels' (270).
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


# ---------------- SimpleConv Components (Brain Magick) ----------------

class DilatedGLUBlock(nn.Module):
    """
    Corresponds to config:
    kernel_size=3, glu=2, batch_norm=true, gelu=true
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
    Corresponds to config:
    depth=10, dilation_period=5
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


# ---------------- ASP (Pooling) & Global Utils ----------------
class AttentiveStatsPool1D(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        H = hidden or d_model
        self.attn = nn.Sequential(
            nn.Conv1d(d_model, H, 1, bias=True),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(H, 1, 1, bias=True),
        )
        self.proj = nn.Conv1d(2 * d_model, d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.eps = 1e-5

    def forward(self, z_bdt: torch.Tensor) -> torch.Tensor:
        e = self.attn(z_bdt)
        a = torch.softmax(e, dim=-1)
        mu = torch.sum(a * z_bdt, dim=-1, keepdim=True)
        var = torch.sum(a * (z_bdt - mu) ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        feat = torch.cat([mu, std], dim=1)
        out = self.proj(feat).squeeze(-1)
        return torch.nan_to_num(out)


class LearnableTokenPooler(nn.Module):
    def __init__(self, d_model: int, L: int, nhead: int = 4, dropout: float = 0.1, max_T: int = 0):
        super().__init__()
        self.L = int(L)
        self.max_T = int(max_T)
        self.q = nn.Parameter(torch.randn(self.L, d_model) / math.sqrt(d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        if self.max_T > 0:
            self.down = nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model, bias=False, stride=2),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, 1, bias=True),
            )
        else:
            self.down = nn.Identity()

    def forward(self, z_bdt: torch.Tensor, mask_bt: torch.Tensor | None = None) -> torch.Tensor:
        x = z_bdt.transpose(1, 2)
        if mask_bt is None:
            m = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        else:
            m = mask_bt if mask_bt.dtype == torch.bool else (mask_bt > 0.5)
            if m.size(1) != x.size(1):
                m = F.interpolate(m.float().unsqueeze(1), size=x.size(1), mode="nearest").squeeze(1).bool()
        while isinstance(self.down, nn.Sequential) and x.size(1) > self.max_T:
            x = self.down(x.transpose(1, 2)).transpose(1, 2)
            m = F.interpolate(m.float().unsqueeze(1), size=x.size(1), mode="nearest").squeeze(1).bool()
        bad = m.all(dim=1)
        if bad.any():
            m[bad, 0] = False
        B, T, D = x.shape
        q = self.q.unsqueeze(0).expand(B, self.L, D)
        y, _ = self.attn(q, x, x, key_padding_mask=m)
        y = torch.nan_to_num(y)
        return self.norm(y)


# ---------------- UltimateMEGEncoder ----------------
class UltimateMEGEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        n_subjects: int,
        spatial_channels: int = 270,        # Config: merger_channels
        fourier_k: int = 32,
        d_model: int = 320,                 # Config: hidden.meg
        text_dim: int | None = None,
        out_channels: int = 1024,
        # --- SimpleConv Specifics (Brain Magick) ---
        backbone_depth: int = 10,           # Config: depth
        backbone_kernel: int = 3,           # Config: kernel_size
        dilation_period: int = 5,           # Config: dilation_period
        glu_mult: int = 2,                  # Config: glu
        dropout: float = 0.0,               # Config: dropout (for backbone)
        # ----------------------------
        subject_layer_pos: Literal["early", "late", "none"] = "early",
        use_subjects: bool = True,
        spatial_dropout_p: float = 0.2,     # Config: merger_dropout
        out_timesteps: Optional[int] = None,
        # --- Global / Sentence ---
        context_mode: Literal["none", "sentence"] = "sentence",
        context_memory_len: int = 12,
        detach_context: bool = False,
        global_frontend: Literal["shared", "separate_full"] = "separate_full",
        warm_start_global: bool = False,
        pre_down_tcap: int = 0,
        token_pool_max_T: int = 0,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning(f"UltimateMEGEncoder: Ignoring unused kwargs: {list(kwargs.keys())}")

        self.in_channels = in_channels
        self.subject_layer_pos = subject_layer_pos
        self.use_subjects = use_subjects
        self.context_mode = context_mode
        self.d_model = d_model
        self.text_dim = int(text_dim) if (text_dim is not None) else int(d_model)
        self.out_timesteps = out_timesteps
        self.detach_context = detach_context
        self.global_frontend = global_frontend
        self.warm_start_global = bool(warm_start_global)
        self.pre_down_tcap = int(pre_down_tcap) if pre_down_tcap is not None else 0

        # --- Architecture Construction (Aligned with Config) ---
        
        # 1. Spatial Merger (Config: merger=true)
        # Maps sensors -> 270 channels
        self.spatial = SpatialAttention(spatial_channels, fourier_k, spatial_dropout_p, 0.2)
        
        # 2. Subject Layers (Config: subject_layers=true, subject_layers_dim='input')
        # 'input' here implies before the main hidden projection.
        # So we apply it on spatial_channels (270).
        if self.use_subjects and subject_layer_pos == "early":
            self.subj_layer = SubjectLayers(spatial_channels, n_subjects)
        else:
            self.subj_layer = None

        # 3. Initial Linear (Config: initial_linear=270, but hidden.meg=320)
        # We need to project from Spatial (270) to Backbone (320).
        self.pre_linear = nn.Conv1d(spatial_channels, d_model, 1)

        # 4. Backbone (Config: simpleconv)
        self.backbone = SimpleConvBackbone(
            d_model=d_model,
            depth=backbone_depth,
            kernel_size=backbone_kernel,
            dilation_period=dilation_period,
            glu_mult=glu_mult,
            dropout=dropout
        )

        # --- Global / Sentence Branch Handling ---
        
        if self.global_frontend == "separate_full":
            self.spatial_g = SpatialAttention(spatial_channels, fourier_k, spatial_dropout_p, 0.2)
            if self.subj_layer is not None:
                self.subj_layer_g = SubjectLayers(spatial_channels, n_subjects)
            else:
                self.subj_layer_g = None
            self.pre_linear_g = nn.Conv1d(spatial_channels, d_model, 1)
            self.backbone_g = SimpleConvBackbone(
                d_model=d_model,
                depth=backbone_depth,
                kernel_size=backbone_kernel,
                dilation_period=dilation_period,
                glu_mult=glu_mult,
                dropout=dropout
            )
            if self.warm_start_global:
                self._warm_start_from_local_()
        else:
            # Shared frontend
            self.spatial_g = self.spatial
            self.subj_layer_g = self.subj_layer
            self.pre_linear_g = self.pre_linear
            self.backbone_g = self.backbone

        # --- Heads ---
        # Sentence-level head
        self.sent_pool_time = AttentiveStatsPool1D(d_model, hidden=d_model, dropout=dropout)
        self.sent_proj = nn.Linear(d_model, self.text_dim)
        nn.init.xavier_uniform_(self.sent_proj.weight)
        nn.init.zeros_(self.sent_proj.bias)

        # Token Memory (Optional)
        self.token_pool = LearnableTokenPooler(d_model, context_memory_len, nhead=8, dropout=dropout, max_T=token_pool_max_T)
        self.mem_enc = nn.Identity()

        # Local Output Head
        self.proj = nn.Conv1d(d_model, out_channels, 1)
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

    def _warm_start_from_local_(self):
        def _copy_(dst, src):
            if dst is None or src is None: return
            dst.load_state_dict(src.state_dict())
        _copy_(self.spatial_g, self.spatial)
        _copy_(self.subj_layer_g, self.subj_layer)
        _copy_(self.pre_linear_g, self.pre_linear)
        _copy_(self.backbone_g, self.backbone)

    def _encode_stream(self, x_bct: torch.Tensor, sensor_locs: torch.Tensor, subj_idx: torch.Tensor,
                       spatial_mod, subj_mod, linear_mod, backbone_mod) -> torch.Tensor:
        # 1. Spatial Merge (B, C_in, T) -> (B, 270, T)
        z = spatial_mod(x_bct, sensor_locs)
        
        # 2. Subject Layer (Early) -> operates on 270 channels
        if subj_mod is not None:
            z = subj_mod(z, subj_idx)

        # 3. Linear Projection (B, 270, T) -> (B, 320, T)
        z = linear_mod(z)
            
        # 4. Backbone (SimpleConv)
        z = backbone_mod(z) # (B, 320, T)
        return z

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
        meg_win = meg_win.to(device, non_blocking=True)

        # --- Local Branch ---
        local_feat = self._encode_stream(
            meg_win, sensor_locs, subj_idx,
            self.spatial, self.subj_layer, self.pre_linear, self.backbone
        )
        y_local = self.out_pool(self.proj(local_feat)).contiguous()
        y_local = torch.nan_to_num(y_local)

        if (not return_global) or (self.context_mode == "none"):
            return y_local

        # --- Global / Sentence Branch ---
        assert (meg_sent_full is not None) and (meg_sent_full_mask is not None)
        if meg_sent_full.device != device:
            meg_sent_full = meg_sent_full.to(device, non_blocking=True)
            
        global_feat = self._encode_stream(
            meg_sent_full, sensor_locs, subj_idx,
            self.spatial_g, self.subj_layer_g, self.pre_linear_g, self.backbone_g
        )
        
        m = meg_sent_full_mask
        if m.dtype != torch.bool: m = m > 0.5
        if m.size(1) != global_feat.size(-1):
            m = F.interpolate(m.float().unsqueeze(1), size=global_feat.size(-1), mode="nearest").squeeze(1).bool()
        
        g = self.sent_pool_time(global_feat) # [B, D]
        g = self.sent_proj(g)                # [B, Text_Dim]
        g = torch.nan_to_num(g)
        if self.detach_context:
            g = g.detach()

        return y_local, g