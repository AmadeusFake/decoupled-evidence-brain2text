#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#音频锚点
models/meg_encoder_bkp2.py

新增：完整句级分支拆分（separate_full）与若干稳定性修复
----------------------------------------------------------------
UltimateMEGEncoder 现支持 3 种句级前端模式：
- "shared":          句级复用 local 的 spatial/pre_S/subject/to_d/backbone（完全共享）
- "separate":        句级仅分离 to_d/backbone；spatial/subject 仍与 local 共享（原先做法）
- "separate_full":   句级完全分离 spatial/pre_S/subject/to_d/backbone（本次新增，默认 warm-start 自 local）

本版关键修复/增强：
1) SpatialAttention 的随机圆盘 dropout：确保不会“全通道被遮掉”导致 softmax NaN；
   若某个位置所有通道均被 -inf 屏蔽，回退为均匀分布再 softmax。
2) window→global 的 mask：仍保留自动兜底，但显式告警（建议调用方显式传入 meg_sent_mask，True=pad）。
3) RPE 命名与用法澄清：实现为“绝对位置嵌入（APE）”，保持入参名不变以兼容老脚本。
4) separate_full 分支参数 warm-start：句级前端权重从 local 拷贝，提升稳定性。
5) 窗口 token 缓存：增加 TTL 语义（按“调用步”淘汰），与 LRU 上限并存。
6) 可选邻近抑制（use_near_suppress）：对相邻时间 token 做局部均值扣除（半径=near_radius，强度=near_alpha）。
7) 句级 mask 尺寸对齐保护：若与时序长度不符，做 nearest 插值到匹配长度。
"""

from __future__ import annotations
import math
import logging
import contextlib
from typing import Optional, Literal, Dict, Tuple, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["UltimateMEGEncoder", "ContextualReRanker", "AttentiveStatsPool1D"]


# ========== 基础模块 ==========
class FourierEmb(nn.Module):
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
            [feat1.reshape(x.size(0), x.size(1), -1), feat2.reshape(x.size(0), x.size(1), -1)],
            dim=-1,
        )


class SpatialAttention(nn.Module):
    """
    位置驱动的通道注意力：
    - 训练时可随机丢弃一个“圆盘区域”的通道（spatial dropout），增强鲁棒性。
    - 本版修复：避免“全通道被遮掉”造成 softmax NaN；若某位置全被 -inf，回退为均匀分布。
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
        cx = torch.rand(B, device=xy.device)
        cy = torch.rand(B, device=xy.device)
        r = torch.full((B,), self.dropout_r, device=xy.device)
        dx2 = (xy[..., 0] - cx[:, None]) ** 2 + (xy[..., 1] - cy[:, None]) ** 2
        return dx2 <= (r[:, None] ** 2)

    def forward(self, meg_ct: torch.Tensor, sensor_locs: torch.Tensor) -> torch.Tensor:
        xy = sensor_locs[..., :2]                              # [B,C,2]
        pos_feat = self.fourier(xy)                            # [B,C,S]
        q = self.query(pos_feat)                               # [B,C,S]

        # 圆盘随机 dropout（True=drop）
        mask = self._make_mask(xy)                             # [B,C]
        if mask.any():
            # 若某样本全 True，强制保留一个通道
            all_true = mask.all(dim=1)
            if all_true.any():
                mask[all_true, 0] = False
            q = q.masked_fill(mask.unsqueeze(-1), float("-inf"))

        # 若某个位置 S 下所有通道均为 -inf，回退为 0（对应均匀 softmax）
        bad = torch.isinf(q).all(dim=1, keepdim=True)          # [B,1,S]
        if bad.any():
            q = torch.where(bad.expand_as(q), torch.zeros_like(q), q)

        attn = torch.softmax(q, dim=1)                         # [B,C,S]
        attn = torch.nan_to_num(attn)
        out = torch.einsum("bct,bcs->bst", meg_ct, attn)       # [B,spatial_channels,T]
        return out


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


# --- 连续位置标量 MLP（默认关闭）---
class PosScalarMLP(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s.view(-1, 1))


class LearnableTokenPooler(nn.Module):
    """
    将变长时间/窗口序列池化为固定 L 个“记忆 token”。
    - 输入:  z_bdt [B, D, T]，mask_bt [B, T] (True=pad)
    - 输出:  mem [B, L, D]
    """
    def __init__(self, d_model: int, L: int, nhead: int = 4, dropout: float = 0.1, max_T: int = 512):
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
        y, _ = self.attn(q, x, x, key_padding_mask=m)
        y = torch.nan_to_num(y)
        return self.norm(y)


class AttentiveStatsPool1D(nn.Module):
    """输入: z [B, D, T] → 输出: token [B, D]"""
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


# ========== 句向量读头 ==========
class MemReadout(nn.Module):
    """将句级记忆 mem [B,L,D] 聚合为 g [B,D]，顺序不敏感。"""
    def __init__(self, d_model: int, dropout: float = 0.0, normalize_out: bool = False):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1, bias=True)
        nn.init.xavier_uniform_(self.scorer.weight)
        nn.init.zeros_(self.scorer.bias)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_out = bool(normalize_out)

    def forward(self, mem_bld: torch.Tensor) -> torch.Tensor:
        w = self.scorer(mem_bld)          # [B,L,1]
        a = torch.softmax(w, dim=1)
        g = torch.sum(a * mem_bld, dim=1) # [B,D]
        g = self.dropout(self.norm(g))
        g = torch.nan_to_num(g)
        if self.normalize_out:
            g = F.normalize(g, dim=-1)
            g = torch.nan_to_num(g)
        return g


# ========== UltimateMEGEncoder ==========
class UltimateMEGEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        n_subjects: int,
        spatial_channels: int = 270,
        fourier_k: int = 32,
        d_model: int = 320,
        out_channels: int = 1024,
        backbone_depth: int = 5,
        backbone_type: Literal["cnn", "conformer"] = "cnn",
        subject_layer_pos: Literal["early", "late", "none"] = "early",
        use_subjects: bool = True,
        spatial_dropout_p: float = 0.0,
        spatial_dropout_radius: float = 0.2,
        context_mode: Literal["none", "window", "sentence"] = "window",
        context_memory_len: int = 16,
        mem_enc_layers: int = 1,
        mem_enc_heads: int = 8,
        mem_dropout_p: float = 0.0,
        freeze_ctx_local: bool = True,
        detach_context: bool = False,
        cache_ttl: int = 256,
        cache_refresh_p: float = 0.05,   # 目前未使用，保留
        nhead: int = 8,
        dropout: float = 0.1,
        out_timesteps: Optional[int] = None,
        # 位置/近邻相关
        use_rpe: bool = False,           # 兼容参数名；实现为 APE（绝对位置嵌入）
        rpe_max_rel: int = 32,
        rpe_scale: float = 0.0,
        use_mem_pos: bool = False,
        pos_mlp_hidden: int = 128,
        use_near_suppress: bool = False,
        near_radius: int = 2,
        near_alpha: float = 0.2,
        # token 编码缓存与小 batch 编码
        ctx_token_mbatch: int = 64,
        # 窗口 token 聚合
        window_token_agg: Literal["mean", "asp"] = "asp",
        asp_hidden: Optional[int] = None,
        # 句向量读头
        readout_dropout: float = 0.0,
        # 句级前端共享/分离
        global_frontend: Literal["shared", "separate", "separate_full"] = "shared",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.warning(f"UltimateMEGEncoder: 忽略未使用参数: {list(kwargs.keys())}")
        assert backbone_type in ("cnn", "conformer")
        assert subject_layer_pos in ("early", "late", "none")
        assert context_mode in ("none", "window", "sentence")
        assert global_frontend in ("shared", "separate", "separate_full")

        self.in_channels = in_channels
        self.subject_layer_pos = subject_layer_pos
        self.use_subjects = use_subjects
        self.context_mode = context_mode
        self.d_model = d_model
        self.out_timesteps = out_timesteps
        self.ctx_mbatch = int(max(1, ctx_token_mbatch))

        self.ctx_L = int(context_memory_len)
        self.mem_dropout_p = float(mem_dropout_p)
        self.freeze_ctx_local = bool(freeze_ctx_local)
        self.detach_context = bool(detach_context)
        self.cache_ttl = int(cache_ttl)
        self.cache_refresh_p = float(cache_refresh_p)

        self.use_rpe = bool(use_rpe)  # 实际为 APE（绝对）
        self.rpe_scale = float(rpe_scale)
        self.use_mem_pos = bool(use_mem_pos)
        self.use_near_suppress = bool(use_near_suppress)
        self.near_radius = int(max(0, near_radius))
        self.near_alpha = float(near_alpha)
        self.rpe_max_rel = int(rpe_max_rel)

        self.global_frontend = global_frontend

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

        # ---------- Sentence 前端（根据模式选择共享/分离） ----------
        if self.global_frontend == "shared":
            # 全部共享
            self.spatial_g = self.spatial
            self.pre_S_g = self.pre_S
            self.subjS_g = self.subjS
            self.subjD_g = self.subjD
            self.to_d_g = self.to_d
            self.backbone_g = self.backbone
        elif self.global_frontend == "separate":
            # 仅分离 to_d / backbone
            self.spatial_g = self.spatial
            self.pre_S_g = self.pre_S
            self.subjS_g = self.subjS
            self.subjD_g = self.subjD
            self.to_d_g = nn.Conv1d(spatial_channels, d_model, 1)
            self.backbone_g = (
                nn.Sequential(*[PaperDilatedBlock(d_model, dropout=0.2) for _ in range(backbone_depth)])
                if backbone_type == "cnn"
                else ConformerBackbone(d_model, depth=backbone_depth, nhead=nhead, ff_mult=4, conv_kernel=15, dropout=dropout)
            )
        else:  # "separate_full"
            # 完全分离 spatial/pre_S/subject/to_d/backbone
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

        # ---------- 窗口 token 聚合（仅 window 全局流使用） ----------
        self.window_token_agg = window_token_agg
        self.win_pool = (
            AttentiveStatsPool1D(d_model, hidden=asp_hidden or d_model, dropout=dropout)
            if self.window_token_agg == "asp" else None
        )

        # ---------- 句级记忆 & 读头 ----------
        self.token_pool = LearnableTokenPooler(d_model, context_memory_len, nhead=mem_enc_heads, dropout=dropout, max_T=512)
        self.mem_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, mem_enc_heads, d_model * 2, dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.mem_enc = nn.TransformerEncoder(enc_layer, num_layers=mem_enc_layers)
        self.mem_pos_mlp = PosScalarMLP(d_model, hidden=pos_mlp_hidden, dropout=0.0)

        # APE（绝对位置嵌入）：名称沿用 rpe 以兼容外部配置
        self.rpe = nn.Embedding(2 * self.rpe_max_rel + 2, d_model) if self.use_rpe else None
        if self.rpe is not None:
            nn.init.normal_(self.rpe.weight, std=0.02)
            with torch.no_grad():
                self.rpe.weight[0].zero_()  # 0 保留给 pad

        self.readout = MemReadout(d_model, dropout=readout_dropout, normalize_out=False)

        # ---------- 局部输出头（基线保持） ----------
        self.tail = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.GELU(), nn.Conv1d(d_model, d_model, 1))
        self.proj = nn.Conv1d(d_model, out_channels, 1)
        if (out_timesteps is None) or (int(out_timesteps) <= 0):
            self.out_timesteps = None
            self.out_pool = nn.Identity()
            try:
                logger.info("UltimateMEGEncoder: out_timesteps<=0 → no time pooling (Identity).")
            except Exception:
                pass
        else:
            self.out_timesteps = int(out_timesteps)
            self.out_pool = nn.AdaptiveAvgPool1d(self.out_timesteps)

        self._init()

        # ---------- separate_full warm-start ----------
        def _copy_params_(dst: nn.Module, src: nn.Module):
            if (dst is None) or (src is None): return
            with torch.no_grad():
                for (n_d, p_d), (n_s, p_s) in zip(dst.named_parameters(), src.named_parameters()):
                    if p_d.shape == p_s.shape:
                        p_d.copy_(p_s)

        if self.global_frontend == "separate_full":
            _copy_params_(self.spatial_g, self.spatial)
            _copy_params_(self.pre_S_g, self.pre_S)
            _copy_params_(self.subjS_g, self.subjS)
            _copy_params_(self.subjD_g, self.subjD)
            _copy_params_(self.to_d_g, self.to_d)
            _copy_params_(self.backbone_g, self.backbone)

        # ---------- 缓存（仅全局流 window 使用） ----------
        # 存储为 {key: (token_cpu_fp32[TENSOR], last_seen_step[int])}
        self._win_token_cache: "OrderedDict[int, tuple[torch.Tensor, int]]" = OrderedDict()
        self._win_cache_hits = 0
        self._win_cache_miss = 0
        self._win_cache_max = 500_000
        self._step = 0  # 用于 TTL

        # 自动 mask 告警只打一遍
        self._warned_auto_mask = False

    # ======= 工具 =======
    def clear_win_cache(self):
        self._win_token_cache.clear()
        self._win_cache_hits = 0
        self._win_cache_miss = 0

    def set_local_trainable(self, trainable: bool, last_n_backbone_blocks: Optional[int] = None):
        """切换 local 路径是否参与训练；可选仅解冻 backbone 的末尾 N 个 block。"""
        mods = [self.spatial, self.pre_S, self.to_d, self.subjS, self.subjD, self.backbone]

        def _set(m, flag):
            if m is None:
                return
            for p in m.parameters():
                p.requires_grad = flag

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
        self.freeze_ctx_local = not trainable
        if trainable:
            self.clear_win_cache()

    def set_global_trainable(self, trainable: bool, last_n_backbone_blocks: Optional[int] = None):
        """控制句级（global）前端的可训练性。对于不同模式，作用模块不同："""
        def _set(m, flag):
            if m is None:
                return
            for p in m.parameters():
                p.requires_grad = flag

        if self.global_frontend == "shared":
            # 共享模式下不单独控制
            return

        if self.global_frontend == "separate":
            _set(self.to_d_g, trainable)
            if last_n_backbone_blocks is None or not isinstance(self.backbone_g, (nn.Sequential, ConformerBackbone)):
                _set(self.backbone_g, trainable)
            else:
                if isinstance(self.backbone_g, nn.Sequential):
                    for b in self.backbone_g[-last_n_backbone_blocks:]:
                        _set(b, trainable)
                else:
                    for b in self.backbone_g.blocks[-last_n_backbone_blocks:]:
                        _set(b, trainable)
        else:  # "separate_full"
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

    @property
    def context_memory_len(self) -> int:
        return self.ctx_L

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

    # ----- Radiate indices -----
    @staticmethod
    def _radiate_indices(N: int, anchors: torch.Tensor, K: int, stride: int = 1, radius: Optional[int] = None) -> List[torch.Tensor]:
        B = anchors.numel()
        out = []
        r = radius if (radius is not None and radius > 0) else (N - 1)
        for b in range(B):
            i0 = int(anchors[b].item())
            seq: List[int] = [i0]
            d = 1
            taken = 1
            while taken < K and d <= r:
                if (d - 1) % max(1, stride) == 0:
                    l = i0 - d
                    if l >= 0:
                        seq.append(l); taken += 1
                        if taken >= K: break
                    rr = i0 + d
                    if rr < N:
                        seq.append(rr); taken += 1
                        if taken >= K: break
                d += 1
            seq = sorted(set([x for x in seq if 0 <= x < N]))[:K]
            if len(seq) == 0:
                seq = [min(max(i0, 0), N - 1)]
            out.append(torch.tensor(seq, dtype=torch.long, device=anchors.device))
        return out

    def _add_mem_continuous_pos_(self, mem_bLd: torch.Tensor) -> torch.Tensor:
        if not self.use_mem_pos:
            return mem_bLd
        B, L, D = mem_bLd.shape
        pos = torch.linspace(0, 1, steps=L, device=mem_bLd.device, dtype=mem_bLd.dtype).view(1, L).expand(B, -1)
        p = self.mem_pos_mlp(pos.reshape(B * L)).view(B, L, D)
        if p.dtype != mem_bLd.dtype:
            p = p.to(mem_bLd.dtype)
        return mem_bLd + p

    # ----- Global from window seq：变长 N → 固定 L → g -----
    def _global_from_window_seq(
        self,
        meg_sent: torch.Tensor,  # [B, N, C, T]
        sensor_locs: torch.Tensor,  # [B, C, 3]
        subj_idx: torch.Tensor,  # [B]
        anchor_idx: Optional[torch.Tensor] = None,  # [B]
        select_topk: Optional[int] = None,
        radiate_stride: int = 1,
        radiate_radius: Optional[int] = None,
        meg_sent_keys: Optional[torch.Tensor] = None,
        meg_sent_mask: Optional[torch.Tensor] = None,  # [B,N] True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._step += 1
        B, N, C, T = meg_sent.shape
        device = next(self.parameters()).device

        if (anchor_idx is not None) and (select_topk is not None) and (select_topk > 0) and (N > select_topk):
            idx_list = self._radiate_indices(N, anchor_idx.to(device), K=select_topk, stride=radiate_stride, radius=radiate_radius)
            gather = []
            gather_mask = [] if (meg_sent_mask is not None) else None
            for b in range(B):
                idx_b = idx_list[b].to(meg_sent.device)
                gather.append(meg_sent[b].index_select(0, idx_b))
                if gather_mask is not None:
                    gather_mask.append(meg_sent_mask[b].index_select(0, idx_b))
            meg_sent = torch.stack(gather, dim=0)  # [B,K,C,T]
            if gather_mask is not None:
                meg_sent_mask = torch.stack(gather_mask, dim=0)  # [B,K]
            N = meg_sent.size(1)

        # Mask 兜底：建议上游显式传入；兜底仅在首次缺失时打印一次警告
        if meg_sent_mask is None:
            if not self._warned_auto_mask:
                logger.warning("UltimateMEGEncoder: meg_sent_mask 未提供，回退到自动检测（全零窗口为 pad）。建议显式传入以避免误判。")
                self._warned_auto_mask = True
            with torch.no_grad():
                meg_sent_mask = meg_sent.abs().sum(dim=(2, 3)) == 0  # [B,N]
        mask_bt = meg_sent_mask

        flat_cpu = meg_sent.reshape(B * N, C, T)
        tok_flat = torch.empty(B * N, self.d_model, device=device, dtype=self.to_d.weight.dtype)

        use_cache = self.freeze_ctx_local and (meg_sent_keys is not None)
        keys_flat_cpu = meg_sent_keys.reshape(-1).to("cpu") if (use_cache and meg_sent_keys is not None) else None

        hit_mask = torch.zeros(B * N, dtype=torch.bool, device=device)
        if use_cache:
            hit_idx_list: List[int] = []
            cache_toks: List[torch.Tensor] = []
            for i in range(B * N):
                k = int(keys_flat_cpu[i].item()) if keys_flat_cpu is not None else -1
                if k >= 0 and (k in self._win_token_cache):
                    tok_cpu, last_seen = self._win_token_cache[k]
                    # TTL 检查
                    if (self._step - last_seen) <= self.cache_ttl:
                        hit_idx_list.append(i)
                        cache_toks.append(tok_cpu.to(dtype=tok_flat.dtype, device=device))
                        self._win_token_cache.move_to_end(k)
                        # 更新 last_seen
                        self._win_token_cache[k] = (tok_cpu, self._step)
            if hit_idx_list:
                hit_idx = torch.tensor(hit_idx_list, device=device, dtype=torch.long)
                hit_mask[hit_idx] = True
                tok_flat.index_copy_(0, hit_idx, torch.stack(cache_toks, dim=0))
                self._win_cache_hits += len(hit_idx_list)

        miss_idx = (~hit_mask).nonzero(as_tuple=False).squeeze(1)
        self._win_cache_miss += int(miss_idx.numel())

        if miss_idx.numel() > 0:
            toks_miss = self._encode_local_tokens_chunked_(flat_cpu, miss_idx, sensor_locs, subj_idx, B, N)
            tok_flat.index_copy_(0, miss_idx, toks_miss)
            if use_cache:
                km = keys_flat_cpu.index_select(0, miss_idx.to("cpu"))
                for j in range(km.numel()):
                    k = int(km[j].item())
                    if k >= 0:
                        # 存 CPU float32 + 当前 step
                        self._win_token_cache[k] = (tok_flat[miss_idx[j].item()].detach().to(dtype=torch.float32).cpu(), self._step)
                        self._win_token_cache.move_to_end(k)
                        # 尺寸上限驱逐（LRU）
                        if len(self._win_token_cache) > self._win_cache_max:
                            self._win_token_cache.popitem(last=False)

        tok = tok_flat.view(B, N, self.d_model)  # [B, N, D]
        tok = torch.nan_to_num(tok)

        # APE（绝对）注入到 token（非 logits）
        if self.use_rpe and (self.rpe is not None):
            pos_idx = torch.arange(N, device=device).clamp(max=self.rpe_max_rel) + 1  # [N]
            ape = self.rpe(pos_idx).to(tok.dtype).unsqueeze(0)  # [1,N,D]
            tok = tok + self.rpe_scale * ape

        # 可选邻近抑制：扣除局部均值，鼓励 token 多样性
        if self.use_near_suppress and (self.near_radius > 0) and (self.near_alpha > 0.0):
            # tok: [B,N,D] -> [B,D,N]
            x = tok.transpose(1, 2)
            k = 2 * self.near_radius + 1
            w = torch.ones(self.d_model, 1, k, device=tok.device, dtype=tok.dtype) / float(k)
            pad = self.near_radius
            local = F.conv1d(x, w, padding=pad, groups=self.d_model)
            x = x - self.near_alpha * local
            tok = x.transpose(1, 2)

        mask_bt = mask_bt.to(device).to(torch.bool)
        mem = self.token_pool(tok.transpose(1, 2), mask_bt=mask_bt)  # [B, L, D]
        mem = self.mem_norm(mem)
        if self.training and self.mem_dropout_p > 0:
            mem = F.dropout(mem, p=self.mem_dropout_p, training=True)
        mem = self.mem_enc(mem)
        mem = self._add_mem_continuous_pos_(mem)
        g = self.readout(mem)  # [B, D]
        g = g.detach() if self.detach_context else g
        return mem.detach() if self.detach_context else mem, g

    def _encode_local_tokens_chunked_(
        self,
        flat_cpu: torch.Tensor,  # [B*N, C, T] on CPU(pinned)
        miss_idx: torch.Tensor,  # [M]
        sensor_locs: torch.Tensor,  # [B, C, 3] on DEVICE
        subj_idx: torch.Tensor,  # [B] on DEVICE
        B: int,
        N: int,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        D = self.d_model
        out = torch.empty(miss_idx.numel(), D, device=device, dtype=self.to_d.weight.dtype)
        mbs = max(1, self.ctx_mbatch)
        for s in range(0, miss_idx.numel(), mbs):
            e = min(miss_idx.numel(), s + mbs)
            idx_chunk = miss_idx[s:e]
            b_idx = torch.div(idx_chunk, N, rounding_mode="floor")
            x = flat_cpu.index_select(0, idx_chunk).to(device, non_blocking=True)
            loc = sensor_locs.index_select(0, b_idx)
            sub = subj_idx.index_select(0, b_idx)
            z = self._encode_local(x, loc, sub)  # [m, D, T]
            tok = self.win_pool(z) if (self.win_pool is not None) else z.mean(dim=2)
            out[s:e] = tok.to(out.dtype)
        return out

    # ----- Global (full sentence) -----
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

        # 使用“句级专用”完整前端（根据模式可能是共享/半共享/完全分离）
        z_ct = self.spatial_g(meg_sent_full, sensor_locs)   # [B, C', T]
        z_ct = self.pre_S_g(z_ct)
        if (self.subjS_g is not None) and (self.subject_layer_pos == "early"):
            z_ct = self.subjS_g(z_ct, subj_idx)

        z_dt = self.to_d_g(z_ct)
        if (self.subjD_g is not None) and (self.subject_layer_pos == "late"):
            z_dt = self.subjD_g(z_dt, subj_idx)
        z = self.backbone_g(z_dt)                           # [B,D,T]
        z = torch.nan_to_num(z)

        # mask 尺寸对齐（若不符则 nearest 插值到 z 的 T）
        m = meg_sent_full_mask
        if m.dtype != torch.bool:
            m = m > 0.5
        if m.size(1) != z.size(-1):
            m = F.interpolate(m.float().unsqueeze(1), size=z.size(-1), mode="nearest").squeeze(1).bool()

        mem = self.token_pool(z, mask_bt=m)  # [B,L,D]
        mem = self.mem_norm(mem)
        if self.training and self.mem_dropout_p > 0:
            mem = F.dropout(mem, p=self.mem_dropout_p, training=True)
        mem = self.mem_enc(mem)
        mem = self._add_mem_continuous_pos_(mem)
        g = self.readout(mem)  # [B,D]
        g = g.detach() if self.detach_context else g
        return mem.detach() if self.detach_context else mem, g

    # ----- Forward：返回局部输出；可选返回句向量 -----
    def forward(
        self,
        meg_win: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
        # window-mode 全局输入
        meg_sent: Optional[torch.Tensor] = None,  # [B,N,C,T]
        anchor_idx: Optional[torch.Tensor] = None,  # [B]
        select_topk: Optional[int] = None,
        radiate_stride: int = 1,
        radiate_radius: Optional[int] = None,
        meg_sent_keys: Optional[torch.Tensor] = None,
        meg_sent_mask: Optional[torch.Tensor] = None,  # [B,N] True=pad
        # sentence-mode 全局输入
        meg_sent_full: Optional[torch.Tensor] = None,  # [B,C,Tf]
        meg_sent_full_mask: Optional[torch.Tensor] = None,  # [B,Tf] bool
        # flags
        return_global: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        local_bdt = self._encode_local(meg_win.to(device, non_blocking=True), sensor_locs, subj_idx)  # [B,D,T]
        y_local = self.out_pool(self.proj(self.tail(local_bdt))).contiguous()  # [B, out_dim, T_or_1]
        y_local = torch.nan_to_num(y_local)

        if not return_global or self.context_mode == "none":
            return y_local

        if self.context_mode == "window":
            assert meg_sent is not None, "context_mode=window 需要提供 meg_sent [B,N,C,T]"
            _, g = self._global_from_window_seq(
                meg_sent=meg_sent,
                sensor_locs=sensor_locs,
                subj_idx=subj_idx,
                anchor_idx=anchor_idx,
                select_topk=select_topk,
                radiate_stride=radiate_stride,
                radiate_radius=radiate_radius,
                meg_sent_keys=meg_sent_keys,
                meg_sent_mask=meg_sent_mask,
            )
        elif self.context_mode == "sentence":
            assert (meg_sent_full is not None) and (meg_sent_full_mask is not None), \
                "context_mode=sentence 需要提供 meg_sent_full [B,C,Tf] 和 meg_sent_full_mask [B,Tf]"
            _, g = self._global_from_sentence(meg_sent_full, meg_sent_full_mask, sensor_locs, subj_idx)
        else:
            g = torch.zeros(local_bdt.size(0), self.d_model, device=device, dtype=local_bdt.dtype)

        return y_local, g


# ========== 旁路：Contextual Re-ranking（logit 偏置） ==========
class AnchorQuery(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, s_b: torch.Tensor) -> torch.Tensor:
        return self.net(s_b.clamp(0, 1).unsqueeze(-1))


class MemAttnPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wm = nn.Linear(d_model, d_model, bias=False)

    def forward(self, mem_bld: torch.Tensor, q_bd: torch.Tensor) -> torch.Tensor:
        B, L, D = mem_bld.shape
        q = self.Wq(q_bd).unsqueeze(1)  # [B,1,D]
        m = self.Wm(mem_bld)            # [B,L,D]
        score = torch.matmul(q, m.transpose(1, 2)).squeeze(1) / math.sqrt(float(D))
        a = score.softmax(dim=-1)       # [B,L]
        c = torch.einsum("bl,bld->bd", a, mem_bld)  # [B,D]
        return c


class ContextReRankHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim, bias=True)
        self.alpha = nn.Parameter(torch.tensor(-3.0))
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, ctx_bd: torch.Tensor, audio_bkd: torch.Tensor) -> torch.Tensor:
        ctx_proj = self.proj(ctx_bd)  # [B,D]
        bias = torch.einsum("bd,bkd->bk", ctx_proj, audio_bkd)  # [B,K]
        return torch.sigmoid(self.alpha) * bias


class ContextualReRanker(nn.Module):
    def __init__(self, d_model: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.anchor_query = AnchorQuery(d_model, hidden=128, dropout=0.0)
        self.mem_pool = MemAttnPool(d_model)
        self.rerank = ContextReRankHead(d_model, out_dim)
        self.dropout_p = float(dropout)

    @contextlib.contextmanager
    def _tmp_flags(self, encoder, detach_context: bool):
        old = bool(getattr(encoder, "detach_context", True))
        try:
            encoder.detach_context = bool(detach_context)
            yield
        finally:
            encoder.detach_context = old

    def forward(
        self,
        encoder: UltimateMEGEncoder,
        *,
        meg_sent: torch.Tensor,
        sensor_locs: torch.Tensor,
        subj_idx: torch.Tensor,
        anchor_idx: torch.Tensor,
        audio_bank: torch.Tensor,
        meg_sent_keys: torch.Tensor | None = None,
        meg_sent_mask: torch.Tensor | None = None,
        train_mem: bool = False,
        select_topk: Optional[int] = None,
        radiate_stride: int = 1,
        radiate_radius: Optional[int] = None,
    ) -> torch.Tensor:
        device = next(encoder.parameters()).device
        ctx_mgr = torch.enable_grad if train_mem else torch.no_grad
        with ctx_mgr():
            with self._tmp_flags(encoder, detach_context=not train_mem):
                mem_bld, _ = encoder._global_from_window_seq(
                    meg_sent=meg_sent.to(device, non_blocking=True),
                    sensor_locs=sensor_locs.to(device, non_blocking=True),
                    subj_idx=subj_idx.to(device, non_blocking=True),
                    anchor_idx=anchor_idx,
                    select_topk=select_topk,
                    radiate_stride=radiate_stride,
                    radiate_radius=radiate_radius,
                    meg_sent_keys=meg_sent_keys,
                    meg_sent_mask=meg_sent_mask,
                )  # [B,L,D]
        B, N, _, _ = meg_sent.shape
        s = anchor_idx.float() / (float(N) - 1.0 + 1e-8)  # [B]
        q = self.anchor_query(s.to(device))               # [B,D]
        ctx_vec = self.mem_pool(mem_bld, q)               # [B,D]
        if self.training and self.dropout_p > 0.0:
            ctx_vec = F.dropout(ctx_vec, p=self.dropout_p, training=True)
        bias = self.rerank(ctx_vec, audio_bank.to(device))  # [B,K]
        return bias
