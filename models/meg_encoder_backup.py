# filename: models/meg_encoder.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backup/shotscreen
Ultimate MEG Encoder_backup (paper-aligned, context-ready, signature-compatible)

默认 = 论文基线：
- SpatialAttention(基于 2D 传感器坐标的加权聚合) -> 1x1 -> CNN 膨胀残差主干
- subject early 层（可选 late/none）
- 尾部 1x1 + 投影到 1024，**不做时间池化**（保持输入 T）

可选扩展（默认关闭）：
- backbone_type: 'cnn' | 'conformer'
- context_mode: 'none' | 'window' | 'sentence'（带 GlobalEncoder / PerceiverResampler + CrossAttention）
- 兼容旧参：若构造函数里出现过时键，会 warning 但忽略
"""

from __future__ import annotations
import math, logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ------------------ Fourier positional embeddings ------------------ #
class FourierEmb(nn.Module):
    def __init__(self, k: int = 32, margin: float = 0.1):
        super().__init__()
        self.k = int(k); self.margin = float(margin)
        fx = torch.arange(1, k + 1).view(1, 1, -1)
        fy = torch.arange(1, k + 1).view(1, 1, -1)
        self.register_buffer("_fx", fx, persistent=False)
        self.register_buffer("_fy", fy, persistent=False)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: [B,C,2] in [0,1]
        x = xy[..., 0:1] * (1.0 - 2 * self.margin) + self.margin
        y = xy[..., 1:2] * (1.0 - 2 * self.margin) + self.margin
        phase_x = 2 * math.pi * (x * self._fx)  # [B,C,K]
        phase_y = 2 * math.pi * (y * self._fy)
        cosx, sinx = torch.cos(phase_x), torch.sin(phase_x)
        cosy, siny = torch.cos(phase_y), torch.sin(phase_y)
        feat1 = torch.einsum("bck,bcm->bckm", cosx, cosy)
        feat2 = torch.einsum("bck,bcm->bckm", sinx, siny)
        return torch.cat([feat1.reshape(x.size(0), x.size(1), -1),
                          feat2.reshape(x.size(0), x.size(1), -1)], dim=-1)  # [B,C,2K^2]


class SpatialAttention(nn.Module):
    """
    基于二维传感器坐标（Fourier 投影）的空间注意力： [B,C,T] -> [B,S,T]
    """
    def __init__(self, spatial_channels: int, fourier_k: int = 32,
                 dropout_p: float = 0.0, dropout_radius: float = 0.2):
        super().__init__()
        self.spatial_channels = int(spatial_channels)
        self.fourier = FourierEmb(fourier_k, margin=0.1)
        pos_dim = 2 * fourier_k * fourier_k
        self.query = nn.Sequential(
            nn.Linear(pos_dim, pos_dim), nn.SiLU(),
            nn.Linear(pos_dim, spatial_channels)
        )
        self.dropout_p = float(dropout_p)
        self.dropout_r = float(dropout_radius)

    def _make_mask(self, xy: torch.Tensor) -> torch.Tensor:
        B, C, _ = xy.shape
        if (self.dropout_p <= 0.0) or (not self.training):
            return torch.zeros(B, C, dtype=torch.bool, device=xy.device)
        if torch.rand(1, device=xy.device).item() > self.dropout_p:
            return torch.zeros(B, C, dtype=torch.bool, device=xy.device)
        cx = torch.rand(B, device=xy.device)
        cy = torch.rand(B, device=xy.device)
        r = torch.full((B,), self.dropout_r, device=xy.device)
        dx2 = (xy[..., 0] - cx[:, None]) ** 2 + (xy[..., 1] - cy[:, None]) ** 2
        return dx2 <= (r[:, None] ** 2)

    def forward(self, meg_ct: torch.Tensor, sensor_locs: torch.Tensor) -> torch.Tensor:
        xy = sensor_locs[..., :2]                    # [B,C,2]
        pos_feat = self.fourier(xy)                  # [B,C,2K^2]
        q = self.query(pos_feat)                     # [B,C,S]
        mask = self._make_mask(xy)                   # [B,C]
        if mask.any():
            q = q.masked_fill(mask.unsqueeze(-1), float('-inf'))
        attn = torch.softmax(q, dim=1)               # [B,C,S]
        return torch.einsum("bct,bcs->bst", meg_ct, attn)  # [B,S,T]


class SubjectLayers(nn.Module):
    """Per-subject 1×1 conv（单位映射初始化）"""
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
        B = x.size(0); out = torch.empty_like(x)
        for i in range(B):
            out[i:i+1] = self.subject_convs[int(subj_idx[i].item())](x[i:i+1])
        return out


# ------------------ Baseline CNN 主干（论文对齐版） ------------------ #
class PaperDilatedBlock(nn.Module):
    """
    论文对齐次序：
    1) 残差块 #1：Conv(dil=2) + BN + GELU + Dropout
    2) 残差块 #2：Conv(dil=4) + BN + GELU + Dropout
    3) 非残差：   1×1 Conv -> GLU    （**不做 BN**）
    4) 1×1 输出
    """
    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.res1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # 非残差 + GLU（无 BN）
        self.glu = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1, bias=True),
            nn.GLU(dim=1)
        )
        self.out = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        y = x + self.res1(x)
        y = y + self.res2(y)
        y = self.glu(y)         # 通道回到 d_model
        return self.out(y)


# ------------------ Conformer 主干（可选） ------------------ #
class DepthwiseConv1d(nn.Module):
    def __init__(self, ch: int, k: int, pad: int):
        super().__init__()
        self.dw = nn.Conv1d(ch, ch, k, padding=pad, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1)
    def forward(self, x):  # [B,D,T]
        return self.pw(F.gelu(self.dw(x)))

class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, ff_mult: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.Conv1d(d_model, d_model*ff_mult,1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(d_model*ff_mult, d_model,1))
        self.mhsa = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.conv = DepthwiseConv1d(d_model, conv_kernel, pad=conv_kernel//2)
        self.ff2 = nn.Sequential(nn.Conv1d(d_model, d_model*ff_mult,1), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(d_model*ff_mult, d_model,1))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model); self.drop = nn.Dropout(dropout)
    def forward(self, x):  # x: [B,D,T]
        y = x + 0.5*self.ff1(x)
        y_t = self.norm1(y.transpose(1,2))
        sa,_ = self.mhsa(y_t, y_t, y_t, need_weights=False)
        y = y + self.drop(sa.transpose(1,2))
        y = y + self.drop(self.conv(self.norm2(y.transpose(1,2)).transpose(1,2)))
        y = y + 0.5*self.ff2(self.norm3(y.transpose(1,2)).transpose(1,2))
        return y

class ConformerBackbone(nn.Module):
    def __init__(self, d_model: int, depth: int, nhead: int = 8, ff_mult: int = 4, conv_kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([ConformerBlock(d_model, nhead, ff_mult, conv_kernel, dropout) for _ in range(depth)])
    def forward(self, x):
        for b in self.blocks: x = b(x)
        return x


# ------------------ 全局上下文（可选） ------------------ #
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 16384):
        super().__init__(); self.drop=nn.Dropout(dropout); self.emb=nn.Embedding(max_len, d_model)
    def forward(self, x):  # [B,N,D]
        n = x.size(1); pos = torch.arange(n, device=x.device).unsqueeze(0)
        return self.drop(x + self.emb(pos))

class GlobalEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, nhead: int, dropout: float):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1,1,d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pos = LearnedPositionalEncoding(d_model, dropout=dropout)
    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor]):
        x = self.pos(tokens)                                   # [B,N,D]
        cls = self.cls.expand(x.size(0), 1, -1)               # [B,1,D]
        x = torch.cat([cls, x], dim=1)                        # [B,1+N,D]
        if padding_mask is not None:
            pad0 = torch.zeros(x.size(0),1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([pad0, padding_mask], dim=1)
        y = self.enc(x, src_key_padding_mask=padding_mask)    # [B,1+N,D]
        return y[:,0]                                         # [B,D]

class PerceiverResampler(nn.Module):
    def __init__(self, d_model: int, num_latents: int = 64, nhead: int = 8, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model))
        self.blocks = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, sent_tokens_btd: torch.Tensor) -> torch.Tensor:
        B = sent_tokens_btd.size(0)
        z = self.latents.expand(B, -1, -1)  # [B, M, D]
        for attn in self.blocks:
            upd, _ = attn(query=z, key=sent_tokens_btd, value=sent_tokens_btd, need_weights=False)
            z = self.norm(z + upd)
        return z  # [B, M, D]

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, local_btd: torch.Tensor, global_mem: torch.Tensor) -> torch.Tensor:
        if global_mem.dim() == 2:  # [B,D] -> [B,1,D]
            global_mem = global_mem.unsqueeze(1)
        out, _ = self.attn(local_btd, global_mem, global_mem, need_weights=False)
        return self.norm(local_btd + out)


# ------------------ UltimateMEGEncoder ------------------ #
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
        backbone_type: Literal["cnn","conformer"] = "cnn",
        subject_layer_pos: Literal["early","late","none"] = "early",
        use_subjects: bool = True,
        spatial_dropout_p: float = 0.0,
        spatial_dropout_radius: float = 0.2,
        # 兼容旧脚本：
        use_context: bool = False,
        # 新上下文开关：
        context_mode: Literal["none","window","sentence"] | None = None,
        global_context_type: Literal["latent","cls"] = "latent",
        global_depth: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        num_latents: int = 64,
        latent_layers: int = 2,
        out_timesteps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        ignored = []
        for k in ("enable_global_stream", "global_spatial_channels"):
            if k in kwargs:
                ignored.append(k); kwargs.pop(k)
        if ignored:
            logger.warning(f"UltimateMEGEncoder: 忽略过时参数: {ignored}")
        if kwargs:
            logger.warning(f"UltimateMEGEncoder: 未使用参数被忽略: {list(kwargs.keys())}")

        if context_mode is None:
            context_mode = "window" if use_context else "none"
        assert backbone_type in ("cnn","conformer")
        assert subject_layer_pos in ("early","late","none")
        assert context_mode in ("none","window","sentence")

        self.in_channels = in_channels
        self.subject_layer_pos = subject_layer_pos
        self.use_subjects = use_subjects
        self.context_mode = context_mode
        self.global_context_type = global_context_type
        self.d_model = d_model
        self.out_timesteps = out_timesteps  # None -> 不池化

        # 局部编码：空间注意力 -> (early subject) -> 1x1 到 d_model -> (late subject) -> 主干
        self.spatial = SpatialAttention(spatial_channels, fourier_k, spatial_dropout_p, spatial_dropout_radius)
        self.pre_S  = nn.Conv1d(spatial_channels, spatial_channels, 1)
        self.to_d   = nn.Conv1d(spatial_channels, d_model, 1)

        if self.use_subjects and subject_layer_pos != "none":
            self.subjS = SubjectLayers(spatial_channels, n_subjects)
            self.subjD = SubjectLayers(d_model, n_subjects)
        else:
            self.subjS = None; self.subjD = None

        if backbone_type == "cnn":
            self.backbone = nn.Sequential(*[PaperDilatedBlock(d_model, dropout=0.2) for _ in range(backbone_depth)])
        else:
            self.backbone = ConformerBackbone(d_model, depth=backbone_depth, nhead=nhead, ff_mult=4, conv_kernel=15, dropout=dropout)

        # 全局上下文（默认关闭）
        if self.context_mode == "window":
            self.global_enc = GlobalEncoder(d_model, num_layers=global_depth, nhead=nhead, dropout=dropout)
            self.fuse_ctx   = CrossAttentionFusion(d_model, nhead=nhead, dropout=dropout)
            self.resampler  = None
        elif self.context_mode == "sentence":
            if self.global_context_type == "latent":
                self.resampler = PerceiverResampler(d_model, num_latents=num_latents, nhead=nhead, layers=latent_layers, dropout=dropout)
                self.fuse_ctx  = CrossAttentionFusion(d_model, nhead=nhead, dropout=dropout)
                self.global_enc = None
            else:
                self.global_enc = GlobalEncoder(d_model, num_layers=global_depth, nhead=nhead, dropout=dropout)
                self.fuse_ctx   = CrossAttentionFusion(d_model, nhead=nhead, dropout=dropout)
                self.resampler  = None
        else:
            self.global_enc = None
            self.resampler  = None
            self.fuse_ctx   = None

        # 尾部映射 + 投影 + 可选池化（默认不池化）
        self.tail = nn.Sequential(nn.Conv1d(d_model, d_model, 1), nn.GELU(), nn.Conv1d(d_model, d_model, 1))
        self.proj = nn.Conv1d(d_model, out_channels, 1)
        self.out_pool = (nn.AdaptiveAvgPool1d(out_timesteps) if out_timesteps is not None else nn.Identity())

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _encode_local(self, x_bct: torch.Tensor, loc_bC3: torch.Tensor, subj_b: torch.Tensor) -> torch.Tensor:
        z = self.spatial(x_bct, loc_bC3)    # [B,S,T]
        z = self.pre_S(z)
        if (self.subjS is not None) and (self.subject_layer_pos == "early"):
            z = self.subjS(z, subj_b)
        z = self.to_d(z)                    # [B,D,T]
        if (self.subjD is not None) and (self.subject_layer_pos == "late"):
            z = self.subjD(z, subj_b)
        z = self.backbone(z)                # [B,D,T]
        return z

    def _global_from_window_seq(self, meg_sent: torch.Tensor, sensor_locs: torch.Tensor, subj_idx: torch.Tensor,
                                meg_sent_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B,N,C,T = meg_sent.shape
        flat = meg_sent.reshape(B*N, C, T)
        locs_flat = sensor_locs.unsqueeze(1).expand(-1, N, -1, -1).reshape(B*N, C, 3)
        subj_flat = subj_idx.unsqueeze(1).expand(-1, N).reshape(B*N)
        win_feat = self._encode_local(flat, locs_flat, subj_flat).mean(dim=2)  # [B*N,D]
        tokens = win_feat.view(B, N, self.d_model)                              # [B,N,D]
        g_vec = self.global_enc(tokens, meg_sent_mask) if self.global_enc is not None else tokens.mean(dim=1)
        return g_vec  # [B,D]

    def _global_from_sentence(self, meg_sent_full: torch.Tensor, sensor_locs: torch.Tensor, subj_idx: torch.Tensor) -> torch.Tensor:
        z = self._encode_local(meg_sent_full, sensor_locs, subj_idx)            # [B,D,Tf]
        tokens = z.transpose(1, 2)                                              # [B,Tf,D]
        if self.global_context_type == "latent":
            mem = self.resampler(tokens)                                        # [B,M,D]
            return mem
        else:
            g_vec = self.global_enc(tokens, padding_mask=None)                  # [B,D]
            return g_vec

    def forward(
        self,
        meg_win: torch.Tensor,               # [B,C,T]
        sensor_locs: torch.Tensor,           # [B,C,3]
        subj_idx: torch.Tensor,              # [B]
        meg_sent: Optional[torch.Tensor] = None,            # [B,N,C,T]（window 模式）
        meg_sent_mask: Optional[torch.Tensor] = None,       # [B,N]
        meg_sent_full: Optional[torch.Tensor] = None,       # [B,C,T_full]（sentence 模式）
        center_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 统一变量名：local_bDT
        local_bdt = self._encode_local(meg_win, sensor_locs, subj_idx)          # [B,D,T]

        if self.context_mode == "window":
            if meg_sent is None:
                raise ValueError("context_mode='window' 需要 meg_sent=[B,N,C,T]")
            g_vec = self._global_from_window_seq(meg_sent, sensor_locs, subj_idx, meg_sent_mask)  # [B,D]
            local_bdt = self.fuse_ctx(local_bdt.transpose(1,2), g_vec).transpose(1,2).contiguous()
        elif self.context_mode == "sentence":
            if (meg_sent_full is None) and (meg_sent is None):
                raise ValueError("context_mode='sentence' 需要 meg_sent_full 或 meg_sent")
            if meg_sent_full is not None:
                mem = self._global_from_sentence(meg_sent_full, sensor_locs, subj_idx)            # [B,M,D] 或 [B,D]
            else:
                mem = self._global_from_window_seq(meg_sent, sensor_locs, subj_idx, meg_sent_mask) # [B,D]
            local_bdt = self.fuse_ctx(local_bdt.transpose(1,2), mem).transpose(1,2).contiguous()

        x = self.tail(local_bdt)
        y = self.proj(x)                                 # [B,1024,T]
        y = self.out_pool(y)                             # 默认不池化，保持 T
        return y
