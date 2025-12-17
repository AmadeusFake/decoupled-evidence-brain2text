#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 基础组件 ======
class TimeFourierPos(nn.Module):
    def __init__(self, d_model: int, T: int, k: int = 32, dropout: float = 0.0):
        super().__init__()
        self.T = int(T); self.k = int(k); self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(2 * self.k, d_model)
        t = torch.arange(self.T, dtype=torch.float32).view(-1, 1)           # [T,1]
        freqs = torch.arange(1, self.k + 1, dtype=torch.float32).view(1, -1)  # [1,k]
        phases = 2.0 * torch.pi * (t @ freqs) / float(self.T)               # [T,k]
        pos = torch.cat([torch.sin(phases), torch.cos(phases)], dim=1)      # [T,2k]
        self.register_buffer("pos", pos, persistent=False)
    def forward(self, B: int) -> torch.Tensor:
        x = self.proj(self.pos)
        x = self.drop(x)
        return x.unsqueeze(0).expand(B, -1, -1)  # [B,T,d]

def _build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    sin, cos = freqs.sin()[None, None, :, :], freqs.cos()[None, None, :, :]
    return sin, cos

def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    half = sin.size(-1)
    x1, x2 = x[..., :half], x[..., half:half*2]
    x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if x.size(-1) > 2 * half:
        x_rot = torch.cat([x_rot, x[..., 2*half:]], dim=-1)
    return x_rot

def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, dropout_p: float = 0.0, is_causal: bool = False) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal)
    except RuntimeError:
        Dh = q.size(-1)
        attn = (q @ k.transpose(-1, -2)) / math.sqrt(max(1.0, float(Dh)))
        attn = attn.softmax(dim=-1)
        if dropout_p and q.requires_grad:
            attn = F.dropout(attn, p=dropout_p, training=True)
        return attn @ v

class SelfAttentionRope(nn.Module):
    def __init__(self, d_model: int, heads: int = 8, dropout: float = 0.1, rope_theta: float = 10000.0):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads; self.dh = d_model // heads; self.rope_theta = float(rope_theta)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        B, T, D = x_btd.shape
        qkv = self.qkv(x_btd)
        q, k, v = qkv.chunk(3, dim=-1)
        def to_heads(z): return z.view(B, T, self.h, self.dh).transpose(1, 2).contiguous()
        qh, kh, vh = map(to_heads, (q, k, v))
        sin, cos = _build_rope_cache(T, self.dh, self.rope_theta, x_btd.device, x_btd.dtype)
        qh = _apply_rope(qh, sin, cos); kh = _apply_rope(kh, sin, cos)
        q_ = qh.reshape(B * self.h, T, self.dh)
        k_ = kh.reshape(B * self.h, T, self.dh)
        v_ = vh.reshape(B * self.h, T, self.dh)
        out = _sdpa(q_, k_, v_, dropout_p=self.drop.p if self.training else 0.0, is_causal=False)
        out = out.reshape(B, self.h, T, self.dh).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)

class MQACrossAttention(nn.Module):
    def __init__(self, d_model: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads; self.dh = d_model // heads
        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, self.dh, bias=True)
        self.Wv = nn.Linear(d_model, self.dh, bias=True)
        self.Wo = nn.Linear(d_model, d_model, bias=True)
        self.drop = nn.Dropout(dropout)
        self.pre_norm_q = nn.LayerNorm(d_model)
    def forward(self, q_btd: torch.Tensor, mem_bLd: torch.Tensor, pos_bias_bd: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = q_btd.shape
        L = mem_bLd.size(1)
        q_in = self.pre_norm_q(q_btd)
        if pos_bias_bd is not None:
            q_in = q_in + pos_bias_bd.unsqueeze(1)
        q = self.Wq(q_in).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.Wk(mem_bLd).unsqueeze(1).expand(B, self.h, L, self.dh)
        v = self.Wv(mem_bLd).unsqueeze(1).expand(B, self.h, L, self.dh)
        q_ = q.reshape(B * self.h, T, self.dh)
        k_ = k.reshape(B * self.h, L, self.dh)
        v_ = v.reshape(B * self.h, L, self.dh)
        ctx = _sdpa(q_, k_, v_, dropout_p=self.drop.p if self.training else 0.0, is_causal=False)
        ctx = ctx.reshape(B, self.h, T, self.dh).transpose(1, 2).reshape(B, T, D)
        return self.Wo(ctx)

class DepthwiseGLU(nn.Module):
    def __init__(self, d_model: int, k: int = 15, dropout: float = 0.1):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv1d(d_model, d_model, k, padding=pad, groups=d_model, bias=False)
        self.pw = nn.Conv1d(d_model, d_model * 2, 1, bias=True)
        self.drop = nn.Dropout(dropout)
    def forward(self, x_btd: torch.Tensor) -> torch.Tensor:
        z = x_btd.transpose(1, 2)
        z = self.dw(z)
        z = self.pw(z)
        z = F.glu(z, dim=1)
        z = self.drop(z)
        return z.transpose(1, 2)

# ====== 多尺度记忆 ======
def build_multiscale_mem(mem_bLd: torch.Tensor, L_short: int = 32, L_mid: int = 8, L_long: int = 1) -> torch.Tensor:
    B, L, D = mem_bLd.shape
    def pool(x, Lout):
        if Lout == L:
            return x
        return F.adaptive_avg_pool1d(x.transpose(1, 2), Lout).transpose(1, 2).contiguous()
    m_short = pool(mem_bLd, min(L_short, L))
    m_mid   = pool(mem_bLd, min(L_mid,  max(1, L)))
    m_long  = pool(mem_bLd, 1 if L_long <= 1 else min(L_long, max(1, L)))
    return torch.cat([m_short, m_mid, m_long], dim=1)

# ====== Perceiver 解码器 ======
class TimeDecoderPerceiver(nn.Module):
    def __init__(self, d_model: int, audio_d: int, T: int,
                 layers: int = 3, heads: int = 8, time_fourier_k: int = 32, dropout: float = 0.1,
                 L_short: int = 32, L_mid: int = 8, L_long: int = 1):
        super().__init__()
        self.time_pos = TimeFourierPos(d_model, T, k=time_fourier_k, dropout=dropout)
        self.layers = nn.ModuleList()
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, audio_d)
        self.L_short, self.L_mid, self.L_long = int(L_short), int(L_mid), int(L_long)
        for _ in range(layers):
            self.layers.append(nn.ModuleDict(dict(
                norm1=nn.LayerNorm(d_model),
                sa=SelfAttentionRope(d_model, heads=heads, dropout=dropout),
                norm2=nn.LayerNorm(d_model),
                ca=MQACrossAttention(d_model, heads=heads, dropout=dropout),
                norm3=nn.LayerNorm(d_model),
                dglu=DepthwiseGLU(d_model, k=15, dropout=dropout),
                gate=nn.Sequential(nn.Linear(d_model, d_model//8), nn.SiLU(), nn.Linear(d_model//8, 1))
            )))
        self.pos_mlp = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        nn.init.zeros_(self.pos_mlp[-1].bias)
    def forward(self, mem_bLd: torch.Tensor, pos_scalar_b: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = mem_bLd.shape
        T = self.time_pos.T
        mem_ms = build_multiscale_mem(mem_bLd, self.L_short, self.L_mid, self.L_long)
        q = self.time_pos(B)
        p_vec = None
        if pos_scalar_b is not None:
            s = pos_scalar_b
            if s.dim() == 2 and s.size(1) == 1:
                s = s.squeeze(1)
            s = s.to(q.dtype).clamp(-1.0, 1.0).mul_(0.5).add_(0.5)  # [-1,1] -> [0,1]
            if self.training:
                s = (s + torch.randn_like(s) * 0.02).clamp(0.0, 1.0)
            p_vec = self.pos_mlp(s.view(B,1))
        for blk in self.layers:
            y = blk.norm1(q)
            q = q + blk.sa(y)
            y = blk.norm2(q)
            q = q + blk.ca(y, mem_ms, pos_bias_bd=p_vec)
            y = blk.norm3(q)
            delta = blk.dglu(y)
            gate = torch.sigmoid(blk.gate(y).clamp(-20, 20))
            q = q + gate * delta
        q = self.norm_out(q)
        y = self.proj_out(q).transpose(1, 2)
        return torch.nan_to_num(y, 0.0, 0.0, 0.0)
