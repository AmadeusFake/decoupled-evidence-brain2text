# tools/qccp_rerank.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
from typing import List, Dict, Tuple, Any, Optional
import torch
import torch.nn.functional as F

EPS = 1e-8

# ---------- 句子分组：尽量鲁棒的键推断 ----------
_CAND_KEYS = [
    "sentence_id", "sent_id", "sentence_uid",
    "utt_id", "utterance_id", "segment_id",
    "original_sentence_id", "sentence_path", "sentence_audio_path", "transcript_path",
]

def _sent_key(row: dict) -> Tuple[str, str]:
    # 1) 优先显式句级 id
    for k in _CAND_KEYS:
        if k in row and row[k]:
            return ("k:"+k, str(row[k]))
    # 2) 退化：按原音频 + 近似句边界（若有）
    a = str(row.get("original_audio_path", ""))
    so = row.get("original_sentence_onset_in_audio_s", None)
    eo = row.get("original_sentence_offset_in_audio_s", None)
    if so is not None and eo is not None:
        return ("audio+sent", f"{a}::{float(so):.3f}-{float(eo):.3f}")
    # 3) 最后兜底：按原音频归并（后续再用时间衰减区分）
    return ("audio", a)

def group_rows_by_sentence(rows: List[dict]) -> Dict[Tuple[str,str], List[int]]:
    g: Dict[Tuple[str,str], List[int]] = {}
    for i, r in enumerate(rows):
        k = _sent_key(r)
        g.setdefault(k, []).append(i)
    # 按时间排序组内索引（便于构造时间核）
    def _tcenter(rr: dict) -> float:
        s0 = float(rr.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(rr.get("local_window_offset_in_audio_s", s0))
        return 0.5 * (s0 + s1)
    for k, idxs in g.items():
        idxs.sort(key=lambda i: _tcenter(rows[i]))
    return g

def window_centers(rows: List[dict], idxs: List[int]) -> torch.Tensor:
    t = []
    for i in idxs:
        r = rows[i]
        s0 = float(r.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(r.get("local_window_offset_in_audio_s", s0))
        t.append(0.5 * (s0 + s1))
    return torch.tensor(t, dtype=torch.float32)

# ---------- QCCP 先验与重排 ----------
@torch.no_grad()
def qccp_rerank_group(
    base_logits_bo: torch.Tensor,    # [B, O] 本组 B 个查询对全池 O 的“局部分数”（clip 相似度）
    times_b: torch.Tensor,           # [B] 该组窗口的中心时间（秒）
    topk: int = 128,                 # 只重排 top-K 候选（外部保留原分数）
    q_quantile: float = 0.9,         # 分位阈值 q（用于 ReLU(s - m) 的 m）
    half_life_s: float = 2.0,        # 时间核半衰期（秒）
    gamma: float = 0.7,              # 先验权重（加到 logit 上）
    gate: bool = True,               # 启用自适应门控（避免“噪音先验”）
) -> torch.Tensor:
    """
    返回：加了先验后的 logits（同形状 [B,O]；仅 top-K 处被增益，其余等于原值）
    公式（简化版）：
      support_i(j) = Σ_{u≠i} κ(|t_u - t_i|) * ReLU( s(u,j) - m_u )
      S'_i(j) = S_i(j) + β_i * γ * norm(support_i(j))
      其中 m_u = quantile_q( S_u(:) )；κ(d) = 0.5 ** (d / half_life_s)
    """
    B, O = base_logits_bo.shape
    device = base_logits_bo.device

    # 1) 组内每行的阈值 m_u（分位数/或可换 mean+α·std）
    #    注意 B 很小（句内窗口数），对 O 做 quantile 可承受
    m_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)  # [B,1]

    # 2) 时间核 κ
    dt = torch.abs(times_b.view(-1,1) - times_b.view(1,-1))   # [B,B]
    kappa = (0.5 ** (dt / max(1e-6, half_life_s))).to(device) # [B,B]
    kappa.fill_diagonal_(0.0)  # 不给自己投票

    # 3) 每个查询 i 的 top-K 索引
    topk_idx = torch.topk(base_logits_bo, k=min(topk, O), dim=1, largest=True, sorted=False).indices  # [B,K]
    out = base_logits_bo.clone()

    # 4) 计算支持：对每个 i，只在其 top-K 列上累积邻居票
    #    support[i, k] = sum_u kappa[i,u] * relu( S[u, j_k] - m[u] )
    #    向量化实现
    gather_u_j = []
    for i in range(B):
        idx_j = topk_idx[i]                     # [K]
        Suj = base_logits_bo[:, idx_j]          # [B,K] 取所有 u 对这些 j 的分数
        thresh = m_b1                           # [B,1]
        votes = F.relu(Suj - thresh)            # [B,K]
        weights = kappa[i].unsqueeze(1)         # [B,1]
        support = (weights * votes).sum(dim=0)  # [K]
        # ---- 归一化（防止句长依赖）----
        norm = (kappa[i].sum() + EPS)
        support = support / norm
        # ---- 门控：证据尖锐度（熵）越低，门开得越大 ----
        beta_i = gamma
        if gate:
            p = F.softmax(support, dim=0)                   # K 分布
            ent = -(p * (p + EPS).log()).sum() / math.log(p.numel())
            beta_i = gamma * (1.0 - float(ent))             # ∈(0, γ]
        # 写回
        out[i, idx_j] = out[i, idx_j] + beta_i * support
        gather_u_j.append((idx_j, support))
    return out
