#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
models/meg_encoder.py  (sentence-mode global: minimal, length-robust, TIME-ASP)

本版关键变更：
- 将 MEG 内部宽度 d_model 与文本侧输出维度 text_dim 解耦（例如 d_model=256, text_dim=1024）。
- 句级读头：sent_pool_time 后接 Linear(d_model → text_dim)。
- 默认关闭多重下采样（pre_down_tcap=0、token_pool_max_T=0），避免时长泄露。
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


# ---------------- 基础模块 ----------------
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


class PaperDilatedBlock(nn.Module):
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
        self.glu = nn.Sequential(nn.Conv1d(d_model, d_model * 2, 1), nn.GLU(dim=1))
        self.out = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        y = x + self.res1(x)
        y = y + self.res2(y)
        y = self.glu(y)
        return self.out(y)


class DepthwiseConv1d(nn.Module):
    def __init__(self, ch: int, k: int, pad: int):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, k, padding=pad, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return self.pw(F.gelu(self.dw(x)))


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, ff_mult: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.Conv1d(d_model, d_model * ff_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * ff_mult, d_model, 1),
        )
        self.mhsa = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.conv = DepthwiseConv1d(d_model, conv_kernel, pad=conv_kernel // 2)
        self.ff2 = nn.Sequential(
            nn.Conv1d(d_model, d_model * ff_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * ff_mult, d_model, 1),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = x + 0.5 * self.ff1(x)
        y_t = self.norm1(y.transpose(1, 2))
        sa, _ = self.mhsa(y_t, y_t, y_t, need_weights=False)
        y = y + self.drop(sa.transpose(1, 2))
        y = y + self.drop(self.conv(self.norm2(y.transpose(1, 2)).transpose(1, 2)))
        y = y + 0.5 * self.ff2(self.norm3(y.transpose(1, 2)).transpose(1, 2))
        return y


class ConformerBackbone(nn.Module):
    def __init__(self, d_model: int, depth: int, nhead: int = 8, ff_mult: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([ConformerBlock(d_model, nhead, ff_mult, conv_kernel, dropout) for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


# ---------------- ASP（时间维） ----------------
class AttentiveStatsPool1D(nn.Module):
    """输入: z [B, D, T] → 输出: token [B, D]（加权均值+方差→1×1 投影）。"""
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
        e = self.attn(z_bdt)                         # [B,1,T]
        a = torch.softmax(e, dim=-1)                 # [B,1,T]  权重和=1（长度盲）
        mu = torch.sum(a * z_bdt, dim=-1, keepdim=True)
        var = torch.sum(a * (z_bdt - mu) ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        feat = torch.cat([mu, std], dim=1)          # [B,2D,1]
        out = self.proj(feat).squeeze(-1)           # [B,D]
        return torch.nan_to_num(out)


# ---------------- 槽聚合（保留以兼容/分析，不参与 g 主路） ----------------
class MemReadout(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, normalize_out: bool = False, use_layernorm: bool = True):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)
        self.norm = nn.LayerNorm(d_model) if bool(use_layernorm) else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.normalize_out = bool(normalize_out)

    def forward(self, mem_bld: torch.Tensor) -> torch.Tensor:
        w = self.scorer(mem_bld)              # [B,L,1]
        a = torch.softmax(w, dim=1)           # [B,L,1]
        g = torch.sum(a * mem_bld, dim=1)     # [B,D]
        g = self.dropout(self.norm(g))
        g = torch.nan_to_num(g)
        if self.normalize_out:
            g = F.normalize(g, dim=-1)
            g = torch.nan_to_num(g)
        return g


# ---------------- Learnable Token Pooler（PMA；仅保留供 mem 可视化） ----------------
class LearnableTokenPooler(nn.Module):
    def __init__(self, d_model: int, L: int, nhead: int = 4, dropout: float = 0.1, max_T: int = 0):
        super().__init__()
        self.L = int(L)
        self.max_T = int(max_T)   # 默认 0：不下采样，避免时长泄露
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
        x = z_bdt.transpose(1, 2)  # [B,T,D]
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
        y, _ = self.attn(q, x, x, key_padding_mask=m)  # [B,L,D]
        y = torch.nan_to_num(y)
        return self.norm(y)


# ---------------- UltimateMEGEncoder ----------------
class UltimateMEGEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        n_subjects: int,
        spatial_channels: int = 270,
        fourier_k: int = 32,
        d_model: int = 320,                     # MEG 内部宽度
        text_dim: int | None = None,            # 文本侧对齐维度（如 1024）
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
        readout_dropout: float = 0.0,
        # --- global(sentence)
        context_mode: Literal["none", "sentence"] = "sentence",
        context_memory_len: int = 12,
        mem_enc_layers: int = 0,
        mem_enc_heads: int = 8,
        mem_dropout_p: float = 0.15,
        freeze_ctx_local: bool = True,
        detach_context: bool = False,
        global_frontend: Literal["shared", "separate_full"] = "separate_full",
        warm_start_global: bool = False,
        slot_agg: Literal["mean", "asp"] = "mean",
        ctx_token_mbatch: int = 64,
        pre_down_tcap: int = 0,        # 0：关闭句前限长（避免泄露长度）
        token_pool_max_T: int = 0,     # 0：PMA 不下采样（避免泄露长度）
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning(f"UltimateMEGEncoder: 忽略未使用参数: {list(kwargs.keys())}")

        assert backbone_type in ("cnn", "conformer")
        assert subject_layer_pos in ("early", "late", "none")
        assert context_mode in ("none", "sentence")
        assert global_frontend in ("shared", "separate_full")
        assert slot_agg in ("mean", "asp")

        self.in_channels = in_channels
        self.subject_layer_pos = subject_layer_pos
        self.use_subjects = use_subjects
        self.context_mode = context_mode
        self.d_model = d_model
        self.text_dim = int(text_dim) if (text_dim is not None) else int(d_model)
        self.out_timesteps = out_timesteps
        self.ctx_mbatch = int(max(1, ctx_token_mbatch))
        self.ctx_L = int(context_memory_len)
        self.mem_dropout_p = float(mem_dropout_p)
        self.freeze_ctx_local = bool(freeze_ctx_local)
        self.detach_context = bool(detach_context)
        self.global_frontend = global_frontend
        self.warm_start_global = bool(warm_start_global)
        self.slot_agg = slot_agg
        self.pre_down_tcap = int(pre_down_tcap) if pre_down_tcap is not None else 0

        # ---------- Local 前端 ----------
        self.spatial = SpatialAttention(spatial_channels, fourier_k, spatial_dropout_p, spatial_dropout_radius)
        self.pre_S = nn.Conv1d(spatial_channels, spatial_channels, 1)
        self.to_d = nn.Conv1d(spatial_channels, d_model, 1)
        if self.use_subjects and subject_layer_pos != "none":
            self.subjS = SubjectLayers(spatial_channels, n_subjects)
            self.subjD = SubjectLayers(d_model, n_subjects)
        else:
            self.subjS = None
            self.subjD = None

        self.backbone = (
            nn.Sequential(*[PaperDilatedBlock(d_model, dropout=0.2) for _ in range(backbone_depth)])
            if backbone_type == "cnn"
            else ConformerBackbone(d_model, depth=backbone_depth, nhead=nhead, ff_mult=4, conv_kernel=15, dropout=dropout)
        )

        # ---------- Sentence 前端（共享/分离） ----------
        if self.global_frontend == "shared":
            self.spatial_g = self.spatial
            self.pre_S_g = self.pre_S
            self.subjS_g = self.subjS
            self.subjD_g = self.subjD
            self.to_d_g = self.to_d
            self.backbone_g = self.backbone
        else:
            self.spatial_g = SpatialAttention(spatial_channels, fourier_k, spatial_dropout_p, spatial_dropout_radius)
            self.pre_S_g = nn.Conv1d(spatial_channels, spatial_channels, 1)
            if self.use_subjects and subject_layer_pos != "none":
                self.subjS_g = SubjectLayers(spatial_channels, n_subjects)
                self.subjD_g = SubjectLayers(d_model, n_subjects)
            else:
                self.subjS_g = None
                self.subjD_g = None
            self.to_d_g = nn.Conv1d(spatial_channels, d_model, 1)
            self.backbone_g = (
                nn.Sequential(*[PaperDilatedBlock(d_model, dropout=0.2) for _ in range(backbone_depth)])
                if backbone_type == "cnn"
                else ConformerBackbone(d_model, depth=backbone_depth, nhead=nhead, ff_mult=4, conv_kernel=15, dropout=dropout)
            )
            if self.warm_start_global:
                self._warm_start_from_local_()

        # ---------- 句级记忆（保留可选，不参与 g 主路径） ----------
        self.token_pool = LearnableTokenPooler(d_model, context_memory_len, nhead=mem_enc_heads,
                                               dropout=dropout, max_T=int(token_pool_max_T))
        self.mem_norm = nn.LayerNorm(d_model)
        if int(mem_enc_layers) > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model, mem_enc_heads, d_model * 2, dropout, batch_first=True, norm_first=True, activation="gelu",
            )
            self.mem_enc = nn.TransformerEncoder(enc_layer, num_layers=mem_enc_layers)
        else:
            self.mem_enc = nn.Identity()

        self.readout = MemReadout(
            d_model,
            dropout=readout_dropout,
            normalize_out=False,
            use_layernorm=(self.context_mode != "sentence")
        )
        self.slot_pool_sent = AttentiveStatsPool1D(d_model, hidden=d_model, dropout=dropout) if self.slot_agg == "asp" else None

        # ---------- 句级“时间 ASP”读头（主路径） ----------
        self.sent_pool_time = AttentiveStatsPool1D(d_model, hidden=d_model, dropout=dropout)
        # 由 d_model → text_dim（如 256 → 1024）
        self.sent_proj = nn.Linear(d_model, self.text_dim)
        nn.init.xavier_uniform_(self.sent_proj.weight)
        nn.init.zeros_(self.sent_proj.bias)

        # ---------- 局部输出头（保留但可在外部训练脚本中移除） ----------
        self.tail = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.GELU(), nn.Conv1d(d_model, d_model, 1))
        self.proj = nn.Conv1d(d_model, out_channels, 1)
        if (out_timesteps is None) or (int(out_timesteps) <= 0):
            self.out_timesteps = None
            self.out_pool = nn.Identity()
        else:
            self.out_timesteps = int(out_timesteps)
            self.out_pool = nn.AdaptiveAvgPool1d(self.out_timesteps)

        self._init()

    # ======= 初始化 / 工具 =======
    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _warm_start_from_local_(self):
        def _copy_params_(dst: nn.Module, src: nn.Module):
            if (dst is None) or (src is None):
                return
            with torch.no_grad():
                for (n_d, p_d), (n_s, p_s) in zip(dst.named_parameters(), src.named_parameters()):
                    if p_d.shape == p_s.shape:
                        p_d.copy_(p_s)
        _copy_params_(self.spatial_g, self.spatial)
        _copy_params_(self.pre_S_g, self.pre_S)
        _copy_params_(self.subjS_g, self.subjS)
        _copy_params_(self.subjD_g, self.subjD)
        _copy_params_(self.to_d_g, self.to_d)
        _copy_params_(self.backbone_g, self.backbone)

    def set_local_trainable(self, trainable: bool, last_n_backbone_blocks: Optional[int] = None):
        mods = [self.spatial, self.pre_S, self.to_d, self.subjS, self.subjD, self.backbone]
        def _set(m, flag):
            if m is None: return
            for p in m.parameters(): p.requires_grad = flag
        if (last_n_backbone_blocks is None) or (not isinstance(self.backbone, (nn.Sequential, ConformerBackbone))):
            for m in mods: _set(m, trainable)
        else:
            for m in mods: _set(m, False)
            _set(self.to_d, trainable)
            if isinstance(self.backbone, nn.Sequential):
                for b in self.backbone[-last_n_backbone_blocks:]:
                    _set(b, trainable)
            else:
                for b in self.backbone.blocks[-last_n_backbone_blocks:]:
                    _set(b, trainable)

    def set_global_trainable(self, trainable: bool, last_n_backbone_blocks: Optional[int] = None):
        if self.global_frontend == "shared": return
        def _set(m, flag):
            if m is None: return
            for p in m.parameters(): p.requires_grad = flag
        for m in (self.spatial_g, self.pre_S_g, self.subjS_g, self.subjD_g, self.to_d_g):
            _set(m, trainable)
        if last_n_backbone_blocks is None or not isinstance(self.backbone_g, (nn.Sequential, ConformerBackbone)):
            _set(self.backbone_g, trainable)
        else:
            if isinstance(self.backbone_g, nn.Sequential):
                for b in self.backbone_g[-last_n_backbone_blocks:]:
                    _set(b, trainable)
            else:
                for b in self.backbone_g.blocks[-last_n_backbone_blocks:]:
                    _set(b, trainable)

    # ----- Local encode -----
    def _encode_local(self, x_bct: torch.Tensor, loc_bC3: torch.Tensor, subj_b: torch.Tensor) -> torch.Tensor:
        if x_bct.size(1) != loc_bC3.size(1):
            raise RuntimeError(f"Channel mismatch: MEG C={x_bct.size(1)} vs sensor_locs C={loc_bC3.size(1)}.")
        z = self.spatial(x_bct, loc_bC3)
        z = self.pre_S(z)
        if (self.subjS is not None) and (self.subject_layer_pos == "early"):
            z = self.subjS(z, subj_b)
        z = self.to_d(z)
        if (self.subjD is not None) and (self.subject_layer_pos == "late"):
            z = self.subjD(z, subj_b)
        z = self.backbone(z)  # [B,D,T]
        return z

    # ----- Global (sentence) -----
    def _global_from_sentence(
        self,
        meg_sent_full: torch.Tensor,        # [B,C,Tf]
        meg_sent_full_mask: torch.Tensor,   # [B,Tf] bool
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if meg_sent_full.device != device:
            meg_sent_full = meg_sent_full.to(device, non_blocking=True)

        # 句级专用前端（shared / separate_full）
        z_ct = self.spatial_g(meg_sent_full, sensor_locs)   # [B, C', T]
        z_ct = self.pre_S_g(z_ct)
        if (self.subjS_g is not None) and (self.subject_layer_pos == "early"):
            z_ct = self.subjS_g(z_ct, subj_idx)

        z_dt = self.to_d_g(z_ct)                            # [B, D, T]
        if (self.subjD_g is not None) and (self.subject_layer_pos == "late"):
            z_dt = self.subjD_g(z_dt, subj_idx)

        # 默认关闭句前限长，避免直接暴露时长
        if getattr(self, "pre_down_tcap", 0) and z_dt.size(-1) > self.pre_down_tcap:
            Tcap = int(self.pre_down_tcap)
            while z_dt.size(-1) > Tcap:
                z_dt = F.avg_pool1d(z_dt, kernel_size=2, stride=2, ceil_mode=False)

        # 主干编码
        z = self.backbone_g(z_dt)                           # [B,D,T]
        z = torch.nan_to_num(z)

        # 对齐 mask 尺寸（若提供）
        m = meg_sent_full_mask
        if m.dtype != torch.bool:
            m = m > 0.5
        if m.size(1) != z.size(-1):
            m = F.interpolate(m.float().unsqueeze(1), size=z.size(-1), mode="nearest").squeeze(1).bool()

        # ===== 主路径：时间 ASP → 句向量 g（长度盲）=====
        g = self.sent_pool_time(z)           # [B, d_model]
        g = self.sent_proj(g)                # [B, text_dim]
        g = torch.nan_to_num(g)
        g = g.detach() if self.detach_context else g

        # ===== 可选 mem：保留以便分析；不用于 g =====
        mem = None
        if self.ctx_L > 0:
            mem = self.token_pool(z, mask_bt=m)      # [B,L,D]
            mem = self.mem_norm(mem)
            if self.training and self.mem_dropout_p > 0:
                mem = F.dropout(mem, p=self.mem_dropout_p, training=True)
            mem = self.mem_enc(mem)                  # 0 层时为 Identity
            mem = mem.detach() if self.detach_context else mem

        return mem, g

    # ----- Forward -----
    def forward(
        self,
        meg_win: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
        *,
        meg_sent_full: Optional[torch.Tensor] = None,      # [B,C,Tf]
        meg_sent_full_mask: Optional[torch.Tensor] = None, # [B,Tf] bool
        return_global: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        local_bdt = self._encode_local(meg_win.to(device, non_blocking=True), sensor_locs, subj_idx)  # [B,D,T]
        y_local = self.out_pool(self.proj(self.tail(local_bdt))).contiguous()
        y_local = torch.nan_to_num(y_local)

        if (not return_global) or (self.context_mode == "none"):
            return y_local

        assert (meg_sent_full is not None) and (meg_sent_full_mask is not None), \
            "context_mode=sentence 需要提供 meg_sent_full [B,C,Tf] 和 meg_sent_full_mask [B,Tf]"
        mem, g = self._global_from_sentence(meg_sent_full, meg_sent_full_mask, sensor_locs, subj_idx)
        return y_local, g
