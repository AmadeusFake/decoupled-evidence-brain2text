#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrieval_window_vote.py  (window-level)

核心改动（相对你给的版本）：
- 新增 vote_mode 三选一：off | old | twostage（默认 twostage）
- 新增“两阶段句票” window_vote_rerank():
    阶段1（句级）：Top-K ∩ 分位门槛 → 聚合到句子（mean/max/logsumexp），句长归一，挑 Top-S 句，并用句分布熵调制 beta
    阶段2（句内）：把句支持度按 softmax/Top-r/均匀 分配回“该句所有候选窗口”（可选 softmax 温度）
  这样既能稳健提升“选对句子”的概率，又能**改变句内排名**，避免“平均加票导致句内顺序不变”的问题。
- 旧版 window_vote_boost() 保留为 vote_mode=old 以便回溯复现实验。
- 其它：参数化更细、加入句长归一（none|count|sqrt|log）、轻微负票（demote_alpha，可关）、混合权重 mix_alpha。

你的 ckpt 选择、模型加载、QCCP、主流程等保留。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------- 常量 -------------------------
TARGET_T = 360     # 对齐 T（示例：3s*120Hz）
AUDIO_D = 1024
EPS = 1e-8

# ------------------------- I/O & 工具 -------------------------
def log(msg: str):
    print(msg, flush=True)

def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def content_id_of(r: dict) -> str:
    """定义窗口级唯一键：原音频 + [窗口起止秒]"""
    if r.get("content_id"):
        return r["content_id"]
    a = r["original_audio_path"]
    s0 = float(r["local_window_onset_in_audio_s"])
    s1 = float(r["local_window_offset_in_audio_s"])
    return f"{a}::{s0:.3f}-{s1:.3f}"

def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D audio array, got {x.shape}")
    if x.shape[0] == AUDIO_D:
        return x
    if x.shape[1] == AUDIO_D:
        return x.T
    # 取更接近 AUDIO_D 的那一维为特征维
    return x if abs(x.shape[0]-AUDIO_D) < abs(x.shape[1]-AUDIO_D) else x.T

def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(1) == T:
        return x
    return F.interpolate(x.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0)

def window_centers(rows: List[dict], idxs: List[int]) -> torch.Tensor:
    t = []
    for i in idxs:
        r = rows[i]
        s0 = float(r.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(r.get("local_window_offset_in_audio_s", s0))
        t.append(0.5*(s0+s1))
    return torch.tensor(t, dtype=torch.float32)

# ------------------------- 句子别名（避免 KeyError） -------------------------
_CAND_SENT_KEYS = [
    "sentence_id", "sentence_uid",
    "utt_id", "utterance_id", "segment_id",
    "original_sentence_id",
    "sentence_path", "sentence_audio_path", "transcript_path",
]

def _round3(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return None

def sentence_aliases(row: dict):
    """
    生成“句子别名”列表。形如：
      ('k:sentence_id','04_1_0.0_0'), ('audio+sent','/a.wav::0.000-5.100'), ('audio','/a.wav')
    """
    aliases = []
    # 显式 id
    for k in _CAND_SENT_KEYS:
        v = row.get(k)
        if v not in (None, ""):
            aliases.append((f"k:{k}", str(v)))
    # 音频 + 句子起止秒
    a = str(row.get("original_audio_path", "") or row.get("sentence_audio_path", "") or row.get("audio_path", ""))
    so = (row.get("global_segment_onset_in_audio_s", None)
          if row.get("global_segment_onset_in_audio_s", None) is not None
          else row.get("original_sentence_onset_in_audio_s", None))
    eo = (row.get("global_segment_offset_in_audio_s", None)
          if row.get("global_segment_offset_in_audio_s", None) is not None
          else row.get("original_sentence_offset_in_audio_s", None))
    if a and so is not None and eo is not None:
        so3, eo3 = _round3(so), _round3(eo)
        if so3 and eo3:
            aliases.append(("audio+sent", f"{a}::{so3}-{eo3}"))
    # 仅音频兜底
    if a:
        aliases.append(("audio", a))
    return aliases

def build_sentence_index_with_alias(candidate_rows: list):
    canon2idx = {}
    alias2idx = {}
    cand_sent_idx = []
    for r in candidate_rows:
        aliases = sentence_aliases(r)
        if not aliases:
            cand_sent_idx.append(-1)
            continue
        canon = aliases[0]
        if canon not in canon2idx:
            canon2idx[canon] = len(canon2idx)
        sidx = canon2idx[canon]
        for a in aliases:
            if a not in alias2idx:
                alias2idx[a] = sidx
        cand_sent_idx.append(sidx)
    return canon2idx, alias2idx, cand_sent_idx

def lookup_sent_idx(row: dict, alias2idx: dict):
    for a in sentence_aliases(row):
        if a in alias2idx:
            return alias2idx[a]
    return None

# ------------------------- subject 映射（从 records 读取） -------------------------
def _normalize_subject_key(x: Any) -> Optional[str]:
    """
    将 'S01' / '01' / 1 / '1' 等全部归一到两位数字字符串 '01'。
    """
    if x is None:
        return None
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    n = int(m.group(1))
    return f"{n:02d}"

def read_subject_mapping_from_records(run_dir: Path) -> Dict[str, int]:
    """
    读取 run_dir/records/subject_mapping.json，支持两种格式：
    - { "mapping": {"01":0,...}, "order":[...] }
    - { "01":0, "02":1, ... }
    返回：{'01':0,'02':1,...}
    """
    p = run_dir / "records" / "subject_mapping.json"
    assert p.exists(), f"[SUBJECT] subject_mapping.json not found: {p}"
    obj = json.loads(p.read_text(encoding="utf-8"))
    if "mapping" in obj and isinstance(obj["mapping"], dict):
        raw = obj["mapping"]
    else:
        raw = obj
    out = {}
    for k, v in raw.items():
        nk = _normalize_subject_key(k)
        if nk is not None:
            out[nk] = int(v)
    assert out, "[SUBJECT] empty mapping after normalization"
    return out

def subject_indices_for_rows_by_records(rows: List[dict], subj_map: Dict[str, int]) -> List[int]:
    miss = 0
    idxs = []
    for r in rows:
        sid = _normalize_subject_key(r.get("subject_id"))
        if sid is not None and sid in subj_map:
            idxs.append(subj_map[sid])
        else:
            idxs.append(0)  # fallback
            miss += 1
    if miss > 0:
        log(f"[WARN] {miss}/{len(rows)} rows have no subject match to records mapping; fallback subj_idx=0.")
    return idxs

# ------------------------- 模型加载 -------------------------
def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        with open(rec, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("model_cfg", {}) or cfg.get("enc_cfg", {}) or {}
    return {}

def choose_ckpt_path(args) -> Path:
    if args.use_best_ckpt:
        best_txt = Path(args.run_dir) / "records" / "best_checkpoint.txt"
        assert best_txt.exists(), f"best_checkpoint.txt not found at {best_txt}"
        ckpt = best_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path(args.run_dir) / ckpt_path).resolve()
        assert ckpt_path.exists(), f"best ckpt not found: {ckpt_path}"
        log(f"[INFO] Using BEST checkpoint from records: {ckpt_path}")
        return ckpt_path
    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"--ckpt_path not found: {ckpt_path}"
    return ckpt_path

# 你的 window 级 encoder
from models.meg_encoder import UltimateMEGEncoder

def _read_logit_scale_exp(ckpt_path: Path) -> Optional[float]:
    """
    若 ckpt 里有 learnable logit_scale，返回 exp(logit_scale)。没有则 None。
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)
    for k in ("model.scorer.logit_scale", "scorer.logit_scale", "logit_scale"):
        v = state.get(k, None)
        if v is not None:
            try:
                return float(torch.exp(v).item())
            except Exception:
                try:
                    return float(np.exp(float(v)))
                except Exception:
                    pass
    return None

def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str) -> Tuple[UltimateMEGEncoder, Dict[str,Any]]:
    model_cfg = load_cfg_from_records(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", {}) or hp.get("enc_cfg", {})
    assert model_cfg, "找不到 model_cfg/enc_cfg（records/config.json 或 ckpt.hyper_parameters）"

    if "out_timesteps" in UltimateMEGEncoder.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None  # 保持与训练的一致，无时间池化

    model = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state.items():
        nk = k[6:] if k.startswith("model.") else k
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        log(f"[WARN] Missing keys: {len(missing)}（示例）{missing[:10]}")
    if unexpected:
        log(f"[WARN] Unexpected keys: {len(unexpected)}（示例）{unexpected[:10]}")
    model.eval().to(device)

    meta = {"logit_scale_exp": _read_logit_scale_exp(ckpt_path)}
    if meta["logit_scale_exp"] is not None:
        log(f"[INFO] Found exp(logit_scale) in ckpt: {meta['logit_scale_exp']:.6f}")
    return model, meta

@torch.no_grad()
def encode_meg_batch(model, batch_rows: List[dict], device: str, subj_map: Dict[str,int]) -> torch.Tensor:
    megs, locs, sidx = [], [], []
    miss = 0
    for r in batch_rows:
        mp = r["meg_win_path"]; lp = r["sensor_coordinates_path"]
        assert mp and Path(mp).exists(), f"missing meg_win_path: {mp}"
        assert lp and Path(lp).exists(), f"missing sensor_coordinates_path: {lp}"
        x = np.load(mp, allow_pickle=False).astype(np.float32)
        x = ensure_meg_CxT(x)
        megs.append(torch.from_numpy(x))
        loc = np.load(lp, allow_pickle=False).astype(np.float32)
        locs.append(torch.from_numpy(loc))
        sid = _normalize_subject_key(r.get("subject_id"))
        if sid is not None and sid in subj_map:
            sidx.append(subj_map[sid])
        else:
            sidx.append(0); miss += 1
    if miss:
        log(f"[WARN] {miss}/{len(batch_rows)} rows miss subject in records mapping; use subj_idx=0")

    meg = torch.stack(megs, 0).to(device)   # [B,C,T]
    loc = torch.stack(locs, 0).to(device)   # [B,C,3]
    sid = torch.tensor(sidx, dtype=torch.long, device=device)

    y = model(meg_win=meg, sensor_locs=loc, subj_idx=sid)  # -> [B,1024,T?]
    if y.dim() != 3 or y.size(1) != AUDIO_D:
        raise RuntimeError(f"encoder must output [B,1024,T], got {tuple(y.shape)}")
    if y.size(2) != TARGET_T:
        y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)
    return y  # [B,1024,360]

# ------------------------- 候选池构建（唯一窗口 + 代表行） -------------------------
@torch.no_grad()
def load_audio_pool_unique(test_rows: List[dict], device: str, dtype: torch.dtype):
    uniq: Dict[str, int] = {}
    ids: List[str] = []
    feats = []
    rep_rows = []

    for r in test_rows:
        cid = content_id_of(r)
        if cid in uniq:
            continue
        uniq[cid] = len(ids)
        ids.append(cid)
        rep_rows.append(r)

    for r in tqdm(rep_rows, desc="Load candidates"):
        p = r["audio_feature_path"]
        a = np.load(p, allow_pickle=False).astype(np.float32)
        a = ensure_audio_DxT(a)
        ta = torch.from_numpy(a)
        ta = maybe_interp_1DT(ta, TARGET_T)
        feats.append(ta)

    A = torch.stack(feats, 0).to(device=device, dtype=dtype)  # [O,1024,360]
    return A, ids, rep_rows

# ------------------------- 相似度（clip-style 与训练一致） -------------------------
def compute_logits_clip(queries: torch.Tensor, pool: torch.Tensor,
                        scale: Optional[float] = None) -> torch.Tensor:
    """
    queries: [b,1024,360], pool: [O,1024,360]
    仅候选端 L2（按 (C,T)），与训练一致；不做时间池化
    可选 scale=exp(logit_scale)（若 ckpt 有则使用）
    """
    q = queries.to(torch.float32)
    A = pool.to(torch.float32)
    inv_norms = 1.0 / (A.norm(dim=(1, 2), p=2) + EPS)  # [O]
    logits = torch.einsum("bct,oct,o->bo", q, A, inv_norms)
    if scale is not None:
        logits = logits * float(scale)
    return logits.to(torch.float32)

# ------------------------- QCCP（可选） -------------------------
def qccp_rerank_group(
    base_logits_bo: torch.Tensor,
    times_b: torch.Tensor,
    topk: int = 128,
    q_quantile: float = 0.9,
    half_life_s: float = 2.0,
    gamma: float = 0.7,
    gate: bool = False,
) -> torch.Tensor:
    B, O = base_logits_bo.shape
    device = base_logits_bo.device

    m_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)  # [B,1]
    dt = torch.abs(times_b.view(-1,1) - times_b.view(1,-1))   # [B,B]
    kappa = (0.5 ** (dt / max(1e-6, half_life_s))).to(device)
    kappa.fill_diagonal_(0.0)

    K = min(topk, O)
    topk_idx = torch.topk(base_logits_bo, k=K, dim=1, largest=True, sorted=False).indices  # [B,K]
    out = base_logits_bo.clone()

    for i in range(B):
        idx_j = topk_idx[i]
        Suj = base_logits_bo[:, idx_j]         # [B,K]
        votes = (Suj - m_b1).clamp_min_(0.0)   # [B,K]
        weights = kappa[i].unsqueeze(1)        # [B,1]
        support = (weights * votes).sum(dim=0) # [K]
        support = support / (kappa[i].sum() + EPS)

        beta_i = gamma
        if gate and K > 1:
            p = F.softmax(support, dim=0)
            ent = -(p * (p + EPS).log()).sum() / np.log(K)
            beta_i = gamma * float(1.0 - ent)

        out[i, idx_j] = out[i, idx_j] + beta_i * support

    return out

# ------------------------- 旧版 Window-Vote（保留回溯） -------------------------
def _precompute_sentence_buckets(cand_sent_idx_o: torch.Tensor) -> Dict[int, torch.Tensor]:
    buckets = {}
    uniq = torch.unique(cand_sent_idx_o)
    for s in uniq.tolist():
        if s < 0:
            continue
        buckets[int(s)] = torch.nonzero(cand_sent_idx_o == s, as_tuple=False).view(-1)
    return buckets

def window_vote_boost(
    logits_bo: torch.Tensor,     # [B,O]
    cand_sent_idx_o: torch.Tensor,  # [O]
    topk: int = 1024,
    gamma: float = 0.7,
    gate: bool = True,
    q_quantile: float = 0.9,
    agg: str = "logsumexp",      # "logsumexp" | "mean" | "max"
) -> torch.Tensor:
    """
    旧版：对被选中的句子“全句等量加票”。不改变句内相对次序（这就是你遇到的现象）。
    """
    B, O = logits_bo.shape
    device = logits_bo.device
    K = min(topk, O)

    buckets = _precompute_sentence_buckets(cand_sent_idx_o)

    topk_scores, topk_idx = torch.topk(logits_bo, k=K, dim=1, largest=True, sorted=True)  # [B,K]
    sent_idx_topk = cand_sent_idx_o[topk_idx]  # [B,K]

    out = logits_bo.clone()

    for b in range(B):
        idx = topk_idx[b]
        sco = topk_scores[b]
        sids = sent_idx_topk[b]

        if gate:
            thr = torch.quantile(sco, q=q_quantile)
            m = (sco >= thr)
            idx, sco, sids = idx[m], sco[m], sids[m]

        valid = (sids >= 0)
        if not torch.any(valid):
            continue
        idx, sco, sids = idx[valid], sco[valid], sids[valid]

        us, inv = torch.unique(sids, return_inverse=True)

        if agg == "mean":
            num = torch.bincount(inv, minlength=us.numel()).clamp_min_(1)
            sums = torch.zeros_like(num, dtype=sco.dtype)
            sums.index_add_(0, inv, sco)
            sent_support = sums / num
        elif agg == "max":
            sent_support = torch.full((us.numel(),), fill_value=-1e9, dtype=sco.dtype, device=device)
            sent_support.scatter_reduce_(0, inv, sco, reduce="amax", include_self=True)
        else:
            sent_support = torch.full((us.numel(),), fill_value=-1e9, dtype=sco.dtype, device=device)
            for k in range(us.numel()):
                vals = sco[inv == k]
                m = torch.max(vals)
                sent_support[k] = m + torch.log(torch.clamp(torch.exp(vals - m).sum(), min=EPS))

        beta = gamma
        if gate and us.numel() > 1:
            p = F.softmax(sent_support, dim=0)
            ent = -(p * (p + EPS).log()).sum() / np.log(float(us.numel()))
            beta = gamma * float(1.0 - ent)

        for si, s_id in enumerate(us.tolist()):
            sup = beta * sent_support[si]
            if sup <= 0:
                continue
            bucket_idx = buckets.get(int(s_id), None)
            if bucket_idx is not None and bucket_idx.numel() > 0:
                out[b, bucket_idx] += sup

    return out

# ------------------------- 新版：两阶段窗口重排（推荐） -------------------------
def _sent_len_norm_factor(n: int, mode: str = "sqrt") -> float:
    if n <= 0 or mode == "none":
        return 1.0
    if mode == "count":
        return 1.0 / float(n)
    if mode == "sqrt":
        return 1.0 / float(np.sqrt(n))
    if mode == "log":
        return 1.0 / float(np.log2(n + 1.0))
    return 1.0

def window_vote_rerank(
    logits_bo: torch.Tensor,            # [B,O] 原始分数
    cand_sent_idx_o: torch.Tensor,      # [O]  每个候选窗口的句索引（-1 表示未知）
    *,
    topk_window: int = 1024,            # 句票只在 Top-K 内做统计
    q_quantile: float = 0.8,            # 筛强候选
    sent_agg: str = "mean",             # mean|max|logsumexp
    sent_top_m: int = 3,                # 每句取最强 m 个窗口统计
    sent_topS: int = 3,                 # 只对 Top-S 句子加票
    sent_norm: str = "sqrt",            # none|count|sqrt|log
    gamma: float = 0.7,                 # 票重
    gate_entropy: bool = True,          # 用句级分布熵调节 beta
    intra_mode: str = "softmax",        # softmax|topr|none
    intra_topr: int = 4,                # topr 模式下只分配到前 r 个
    intra_temp: float = 2.0,            # softmax 温度
    demote_alpha: float = 0.0,          # 对非 Top-S 句子轻微降权（0 关闭）
    mix_alpha: float = 1.0,             # 与原分数混合：out = (1-mix)*base + mix*boost
) -> torch.Tensor:
    """
    两阶段重排：
      1) 句级：在 Top-K ∩ q 分位门槛内聚合到句子，做句长归一，只保留 Top-S 句；beta 可被句分布熵调制；
      2) 句内：把“句支持度”再分配给该句的窗口（softmax/Top-r/均匀），从而改写句内排名。
    """
    B, O = logits_bo.shape
    device = logits_bo.device
    K = int(min(topk_window, O))
    if K <= 0:
        return logits_bo

    # 预建句桶
    buckets = _precompute_sentence_buckets(cand_sent_idx_o)
    bucket_sizes = {s: (idx.numel() if idx is not None else 0) for s, idx in buckets.items()}

    base = logits_bo
    boost = logits_bo.clone()

    topk_scores, topk_idx = torch.topk(base, k=K, dim=1, largest=True, sorted=True)  # [B,K]
    sent_idx_topk = cand_sent_idx_o[topk_idx]  # [B,K]

    for b in range(B):
        idx = topk_idx[b]
        sco = topk_scores[b]
        sids = sent_idx_topk[b]

        # 过滤未知句 / 分位门槛
        valid = (sids >= 0)
        if valid.any():
            idx, sco, sids = idx[valid], sco[valid], sids[valid]
        else:
            continue

        thr = torch.quantile(sco, q=q_quantile)
        m = (sco >= thr)
        idx, sco, sids = idx[m], sco[m], sids[m]
        if idx.numel() == 0:
            continue

        # 按句聚合（只取每句最强 m 个）
        us, inv = torch.unique(sids, return_inverse=True)  # us: [S_sel]
        S_sel = us.numel()
        if S_sel == 0:
            continue

        # 先为每个句子收集它的 top-m 窗口得分
        sent_scores = []
        for k_s in range(S_sel):
            mask = (inv == k_s)
            vals = sco[mask]
            if vals.numel() == 0:
                sent_scores.append(torch.tensor(-1e9, device=device))
                continue
            # 取 top-m
            m_take = min(sent_top_m, int(vals.numel()))
            top_vals = torch.topk(vals, k=m_take, largest=True, sorted=False).values

            if sent_agg == "max":
                agg = torch.max(top_vals)
            elif sent_agg == "logsumexp":
                m0 = torch.max(top_vals)
                agg = m0 + torch.log(torch.clamp(torch.exp(top_vals - m0).sum(), min=EPS))
            else:  # mean（推荐更稳）
                agg = torch.mean(top_vals)

            # 句长归一
            s_id = int(us[k_s].item())
            n_windows = bucket_sizes.get(s_id, 1)
            agg = agg * _sent_len_norm_factor(n_windows, mode=sent_norm)
            sent_scores.append(agg)

        sent_scores = torch.stack(sent_scores, dim=0)  # [S_sel]

        # 只保留 Top-S 句子
        keepS = min(int(sent_topS), int(sent_scores.numel()))
        if keepS <= 0:
            continue
        topS_val, topS_idx_local = torch.topk(sent_scores, k=keepS, largest=True, sorted=True)
        us_sel = us[topS_idx_local]  # 被选中的句 id

        # 句级熵门控
        beta = gamma
        if gate_entropy and keepS > 1:
            p = F.softmax(topS_val, dim=0)
            ent = -(p * (p + EPS).log()).sum() / np.log(float(keepS))
            beta = gamma * float(1.0 - ent)

        # 给 Top-S 句加票，并决定“句内分配”
        for k_s in range(keepS):
            s_id = int(us_sel[k_s].item())
            sup = beta * float(topS_val[k_s].item())
            if sup <= 0:
                continue
            bucket_idx = buckets.get(s_id, None)
            if bucket_idx is None or bucket_idx.numel() == 0:
                continue

            # 句内分配权重
            base_slice = base[b, bucket_idx]
            if intra_mode == "softmax":
                p = F.softmax(base_slice / max(1e-6, intra_temp), dim=0)
                boost[b, bucket_idx] += sup * p
            elif intra_mode == "topr":
                r = min(intra_topr, int(bucket_idx.numel()))
                if r <= 0:
                    continue
                top_r_idx = torch.topk(base_slice, k=r, largest=True, sorted=False).indices
                sel = bucket_idx[top_r_idx]
                boost[b, sel] += sup / float(r)
            else:  # none: 均匀分配（不建议，等价旧版句内不改序）
                boost[b, bucket_idx] += sup / float(bucket_idx.numel())

        # 对非 Top-S 句轻微降权（可选）
        if demote_alpha > 0.0:
            non_sel_mask = torch.ones(S_sel, dtype=torch.bool, device=device)
            non_sel_mask[topS_idx_local] = False
            if non_sel_mask.any():
                non_us = us[non_sel_mask]
                # 用被选句的均值做一个温和的“阈值”
                tau = float(topS_val.mean().item())
                for s_id_t in non_us.tolist():
                    s_id_t = int(s_id_t)
                    idx_t = buckets.get(s_id_t, None)
                    if idx_t is None or idx_t.numel() == 0:
                        continue
                    boost[b, idx_t] -= demote_alpha * tau

    # 与原分数线性混合（更稳）
    out = (1.0 - mix_alpha) * base + mix_alpha * boost
    return out

# ------------------------- 评测主流程 -------------------------
def evaluate(args):
    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if "16" in amp or "fp16" in amp else None)

    test_rows = read_jsonl(Path(args.test_manifest))
    log(f"[INFO] test rows = {len(test_rows):,}")

    # 候选池（唯一）+ 代表行（句别名）
    A, pool_ids, candidate_rows = load_audio_pool_unique(test_rows, device=device, dtype=torch.float32)
    O = A.size(0)
    log(f"[INFO] candidate windows O={O}")

    canon2idx, alias2idx, cand_sent_idx = build_sentence_index_with_alias(candidate_rows)
    S = len(canon2idx)
    log(f"[INFO] sentences S={S}")
    cand_sent_idx = torch.tensor(cand_sent_idx, dtype=torch.long, device=device)

    # GT 窗口索引
    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in test_rows]

    # GT 句索引（仅日志）
    gt_sent_idx = []
    missing = 0
    for r in test_rows:
        s = lookup_sent_idx(r, alias2idx)
        if s is None:
            aliases = sentence_aliases(r)
            canon = aliases[0] if aliases else ("unknown", str(r.get("sentence_id","")))
            if canon not in canon2idx:
                canon2idx[canon] = len(canon2idx)
            s = canon2idx[canon]
            for a in aliases:
                if a not in alias2idx:
                    alias2idx[a] = s
            missing += 1
        gt_sent_idx.append(s)
    if missing > 0:
        log(f"[WARN] {missing} test rows' sentences were not in candidate pool; created empty buckets.")

    # subject 映射（从 records 读取）
    run_dir = Path(args.run_dir)
    subj_map = read_subject_mapping_from_records(run_dir)
    log(f"[SUBJECT] loaded from records: {len(subj_map)} subjects (keys like {sorted(list(subj_map.keys()))[:3]}...)")

    # 模型 & ckpt & 温度
    ckpt_path = choose_ckpt_path(args)
    model, meta = load_model_from_ckpt(ckpt_path, run_dir, device=device)
    scale = meta.get("logit_scale_exp", None) if args.use_ckpt_logit_scale else None

    # 统计
    topk_list = [int(x) for x in args.topk.split(",")]
    recalls = {k: 0 for k in topk_list}
    mrr_sum = 0.0
    ranks: List[int] = []

    save_topk_k = max(0, int(args.save_topk)) if args.save_topk is not None else 0
    preds_topk_file = preds_tsv_file = None
    if save_topk_k > 0:
        out_dir = run_dir / "results" / "retrieval_window_vote"
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_topk_path = out_dir / f"preds_topk{save_topk_k}.jsonl"
        preds_tsv_path = out_dir / f"preds_topk{save_topk_k}.tsv"
        preds_topk_file = open(preds_topk_path, "w", encoding="utf-8")
        preds_tsv_file = open(preds_tsv_path, "w", encoding="utf-8")
        preds_tsv_file.write("query_index\trank\tgt_cid\tpred_cids\n")

    # 分组（供可选 QCCP 用）
    def process_query_indices(q_indices: List[int]):
        nonlocal mrr_sum
        rows = [test_rows[i] for i in q_indices]
        if autocast_dtype is None:
            Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)     # [B,1024,360]
        else:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)

        logits = compute_logits_clip(Y, A, scale=scale)  # [B,O]

        if args.use_qccp:
            times = window_centers(test_rows, q_indices).to(device=logits.device)
            logits = qccp_rerank_group(
                logits, times,
                topk=args.qccp_topk, q_quantile=args.qccp_q,
                half_life_s=args.qccp_half_life_s, gamma=args.qccp_gamma, gate=args.qccp_gate
            )

        # 句票模式选择
        if args.vote_mode == "off":
            pass
        elif args.vote_mode == "old":
            logits = window_vote_boost(
                logits, cand_sent_idx_o=cand_sent_idx,
                topk=args.topk_window, gamma=args.gamma,
                gate=args.gate, q_quantile=args.q_quantile, agg=args.sent_agg
            )
        else:  # twostage
            logits = window_vote_rerank(
                logits, cand_sent_idx_o=cand_sent_idx,
                topk_window=args.topk_window, q_quantile=args.q_quantile,
                sent_agg=args.sent_agg, sent_top_m=args.sent_top_m, sent_topS=args.sent_topS,
                sent_norm=args.sent_norm, gamma=args.gamma, gate_entropy=args.gate_entropy,
                intra_mode=args.intra_mode, intra_topr=args.intra_topr, intra_temp=args.intra_temp,
                demote_alpha=args.demote_alpha, mix_alpha=args.mix_alpha
            )

        for j_in_group, global_j in enumerate(q_indices):
            g = gt_index[global_j]
            s = logits[j_in_group]
            rank = int((s > s[g]).sum().item()) + 1
            ranks.append(rank)
            mrr_sum += 1.0 / rank
            for k in topk_list:
                recalls[k] += int(rank <= k)

            if save_topk_k > 0:
                topk_scores, topk_idx_local = torch.topk(s, k=save_topk_k, largest=True, sorted=True)
                pred_cids = [pool_ids[int(t)] for t in topk_idx_local.tolist()]
                rec = {
                    "query_index": int(global_j),
                    "gt_rank": int(rank),
                    "gt_cid": pool_ids[g],
                    "pred_cids": pred_cids,
                    "pred_scores": [float(x) for x in topk_scores.tolist()],
                    "vote_mode": args.vote_mode,
                    "window_vote": {
                        "topk": int(args.topk_window), "gamma": float(args.gamma),
                        "q": float(args.q_quantile), "gate": bool(args.gate),
                        "agg": args.sent_agg, "sent_top_m": int(args.sent_top_m),
                        "sent_topS": int(args.sent_topS), "sent_norm": args.sent_norm,
                        "intra_mode": args.intra_mode, "intra_topr": int(args.intra_topr),
                        "intra_temp": float(args.intra_temp), "demote_alpha": float(args.demote_alpha),
                        "mix_alpha": float(args.mix_alpha), "gate_entropy": bool(args.gate_entropy)
                    },
                    "qccp_used": bool(args.use_qccp),
                }
                preds_topk_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                preds_tsv_file.write(f"{global_j}\t{rank}\t{pool_ids[g]}\t{','.join(pred_cids)}\n")

    if args.use_qccp:
        # 句分组（可选）
        sent2idx: Dict[Tuple[str,str], List[int]] = {}
        for i, r in enumerate(test_rows):
            als = sentence_aliases(r)
            k = als[0] if als else ("unknown", content_id_of(r))
            sent2idx.setdefault(k, []).append(i)
        groups = list(sent2idx.values())
        log(f"[INFO] QCCP mode: sentences={len(groups)}, avg windows/sent={len(test_rows)/max(1,len(groups)):.2f}")
        for g_idx in tqdm(range(len(groups)), desc="Evaluate (QCCP + Vote)"):
            process_query_indices(groups[g_idx])
    else:
        B = len(test_rows)
        cs = int(args.chunk_size)
        for st in tqdm(range(0, B, cs), desc="Evaluate (Vote)"):
            ed = min(B, st + cs)
            process_query_indices(list(range(st, ed)))

    # 汇总
    num_queries = len(test_rows)
    metrics = {
        "num_queries": num_queries,
        "pool_size": O,
        "recall_at": {str(k): recalls[k] / num_queries for k in topk_list},
        "mrr": mrr_sum / num_queries,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "topk_list": topk_list,
        "use_qccp": bool(args.use_qccp),
        "vote_mode": args.vote_mode,
        "window_vote": {
            "topk": int(args.topk_window), "gamma": float(args.gamma),
            "q": float(args.q_quantile), "gate": bool(args.gate),
            "agg": args.sent_agg, "sent_top_m": int(args.sent_top_m),
            "sent_topS": int(args.sent_topS), "sent_norm": args.sent_norm,
            "intra_mode": args.intra_mode, "intra_topr": int(args.intra_topr),
            "intra_temp": float(args.intra_temp), "demote_alpha": float(args.demote_alpha),
            "mix_alpha": float(args.mix_alpha), "gate_entropy": bool(args.gate_entropy)
        },
        "use_ckpt_logit_scale": bool(args.use_ckpt_logit_scale),
    }
    out_dir = run_dir / "results" / "retrieval_window_vote"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.save_json) if args.save_json else (out_dir / "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    out_ranks = Path(args.save_ranks) if args.save_ranks else (out_dir / "ranks.txt")
    with open(out_ranks, "w", encoding="utf-8") as f:
        for r in ranks:
            f.write(str(int(r)) + "\n")

    log("==== Retrieval (Window-Vote) Results ====")
    log(json.dumps(metrics, indent=2, ensure_ascii=False))
    log(f"[INFO] Metrics saved to: {out_json.as_posix()}")
    log(f"[INFO] Ranks saved to  : {out_ranks.as_posix()}")
    if save_topk_k > 0:
        preds_topk_file.close()
        preds_tsv_file.close()

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16", "16-mixed"])

    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--chunk_size", type=int, default=256)

    # 评分缩放
    p.add_argument("--use_ckpt_logit_scale", action="store_true", help="若 ckpt 内有 learnable logit_scale，则使用之")

    # —— 句票模式
    p.add_argument("--vote_mode", type=str, default="twostage", choices=["off", "old", "twostage"])

    # —— old 模式 & 基础门控
    p.add_argument("--topk_window", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--q_quantile", type=float, default=0.8)
    p.add_argument("--sent_agg", type=str, default="mean", choices=["logsumexp", "mean", "max"])

    # —— twostage 细粒度参数
    p.add_argument("--sent_top_m", type=int, default=3)
    p.add_argument("--sent_topS", type=int, default=3)
    p.add_argument("--sent_norm", type=str, default="sqrt", choices=["none", "count", "sqrt", "log"])
    p.add_argument("--gate_entropy", action="store_true")  # 默认 False -> 更稳，显式开关以复现实验
    p.add_argument("--intra_mode", type=str, default="softmax", choices=["softmax", "topr", "none"])
    p.add_argument("--intra_topr", type=int, default=4)
    p.add_argument("--intra_temp", type=float, default=2.0)
    p.add_argument("--demote_alpha", type=float, default=0.0)
    p.add_argument("--mix_alpha", type=float, default=1.0)

    # QCCP（可选）
    p.add_argument("--use_qccp", action="store_true")
    p.add_argument("--qccp_topk", type=int, default=128)
    p.add_argument("--qccp_q", type=float, default=0.9)
    p.add_argument("--qccp_half_life_s", type=float, default=2.0)
    p.add_argument("--qccp_gamma", type=float, default=0.7)
    p.add_argument("--qccp_gate", action="store_true")

    # 输出
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--save_ranks", type=str, default="")
    p.add_argument("--save_topk", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    evaluate(args)
