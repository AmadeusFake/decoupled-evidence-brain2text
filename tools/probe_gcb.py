#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/probe_gcb.py  —  Standalone GCB Probe (no changes to your eval code)

功能：
- 读取 test.jsonl 与 run_dir（从 ckpt / records 恢复模型配置）
- 构建“唯一候选窗口池”（audio features）
- 以“句子组”为单位：编码 MEG → 计算 base logits → 做一次 GCB 支持度快照
- 额外诊断：hubness（非GT被选为top1的组计数）、支持分布偏斜度（Gini/HHI）、句长偏置相关性、
  per-query 上限 r=1 对照（one-vote-per-query）下的 gt_in_topS 变化
- 可选：模拟施加 GCB（不写回模型，只在脚本内计算），对比 rank 改善/伤害/不变
- 输出：
  - <out_dir>/gcb_probe.jsonl          逐组记录
  - <out_dir>/gcb_probe_summary.json   总体统计
  - <out_dir>/hubness_top.json         非GT top1 频次最高的句子（前K）
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------- 常量 -------------------------
TARGET_T = 360     # 时间对齐
AUDIO_D = 1024
EPS = 1e-8

# ------------------------- I/O & 工具 -------------------------
def log(msg: str):
    print(msg, flush=True)

def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def content_id_of(r: dict) -> str:
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
    return x if abs(x.shape[0]-AUDIO_D) < abs(x.shape[1]-AUDIO_D) else x.T

def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(1) == T:
        return x
    return F.interpolate(x.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0)

# ------------------------- 句子别名 -------------------------
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
    aliases = []
    for k in _CAND_SENT_KEYS:
        v = row.get(k)
        if v not in (None, ""):
            aliases.append((f"k:{k}", str(v)))
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
    if a:
        aliases.append(("audio", a))
    return aliases

def build_sentence_index_with_alias(candidate_rows: list):
    canon2idx = {}
    alias2idx = {}
    cand_sent_idx = []
    sid2name = {}  # 追加：可读名

    for r in candidate_rows:
        aliases = sentence_aliases(r)
        if not aliases:
            cand_sent_idx.append(-1)
            continue
        canon = aliases[0]
        if canon not in canon2idx:
            sidx = len(canon2idx)
            canon2idx[canon] = sidx
            sid2name[sidx] = f"{canon[0]}::{canon[1]}"
        sidx = canon2idx[canon]
        for a in aliases:
            if a not in alias2idx:
                alias2idx[a] = sidx
        cand_sent_idx.append(sidx)

    return canon2idx, alias2idx, cand_sent_idx, sid2name

def sent_key_for_group(r: dict) -> Tuple[str, str]:
    als = sentence_aliases(r)
    return als[0] if als else ("unknown", content_id_of(r))

# ------------------------- subject 映射 -------------------------
def _normalize_subject_key(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    n = int(m.group(1))
    return f"{n:02d}"

def read_subject_mapping_from_records(run_dir: Path) -> Dict[str, int]:
    p = run_dir / "records" / "subject_mapping.json"
    assert p.exists(), f"[SUBJECT] subject_mapping.json not found: {p}"
    obj = json.loads(p.read_text(encoding="utf-8"))
    raw = obj["mapping"] if "mapping" in obj and isinstance(obj["mapping"], dict) else obj
    out = {}
    for k, v in raw.items():
        nk = _normalize_subject_key(k)
        if nk is not None:
            out[nk] = int(v)
    assert out, "[SUBJECT] empty mapping after normalization"
    return out

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

from models.meg_encoder_Dense import UltimateMEGEncoder  # 依赖你的工程位置

def _read_logit_scale_exp(ckpt_path: Path) -> Optional[float]:
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
        model_cfg["out_timesteps"] = None  # 评测端不做时间池化

    model = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    new_state = { (k[6:] if k.startswith("model.") else k): v for k, v in state.items() }
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

# ------------------------- 候选池 -------------------------
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

# ------------------------- 相似度（clip-style） -------------------------
def compute_logits_clip(queries: torch.Tensor, pool: torch.Tensor,
                        scale: Optional[float] = None) -> torch.Tensor:
    q = queries.to(torch.float32)
    A = pool.to(torch.float32)
    inv_norms = 1.0 / (A.norm(dim=(1, 2), p=2) + EPS)  # [O]
    logits = torch.einsum("bct,oct,o->bo", q, A, inv_norms)
    if scale is not None:
        logits = logits * float(scale)
    return logits.to(torch.float32)

# ------------------------- 句桶 + 统计工具 -------------------------
def _precompute_sentence_buckets(cand_sent_idx_o: torch.Tensor) -> Dict[int, torch.Tensor]:
    buckets = {}
    uniq = torch.unique(cand_sent_idx_o)
    for s in uniq.tolist():
        if s < 0:
            continue
        buckets[int(s)] = torch.nonzero(cand_sent_idx_o == s, as_tuple=False).view(-1)
    return buckets

def gini_coefficient(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

def hhi_index(x: np.ndarray) -> float:
    s = np.asarray(x, dtype=np.float64)
    tot = s.sum()
    if tot <= 0:
        return 0.0
    p = s / tot
    return float(np.sum(p * p))

def pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(x) != len(y):
        return None
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# ------------------------- GCB 快照（仅诊断，不改 logits） -------------------------
@torch.no_grad()
def gcb_support_snapshot(
    base_logits_bo: torch.Tensor,
    cand_sent_idx_o: torch.Tensor,
    buckets: Dict[int, torch.Tensor],
    *,
    topk: int,
    q_quantile: float,
    agg: str,
    sent_norm: str,
    topS: int,
):
    B, O = base_logits_bo.shape
    device = base_logits_bo.device
    K = min(int(topk), O)
    snap = {
        "keep_total": 0,
        "num_sents_kept": 0,
        "beta": None,
        "topS": [],
        "topS_val": [],
        "entropy": None,
        "per_sent": {},  # sid -> {"support": float, "bucket": int, "kept": int}
        "unlabeled_kept": 0,
        "gini": None,
        "hhi": None,
        "gt_in_topS_cap1": None,  # r=1 per-query 对照
        "margin_cap1": None,
    }
    if K <= 0 or B == 0 or O == 0:
        return snap

    topk_scores, topk_idx = torch.topk(base_logits_bo, k=K, dim=1, largest=True, sorted=False)
    thr_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)
    keep_mask = topk_scores >= thr_b1
    sids = cand_sent_idx_o[topk_idx]
    valid_mask = keep_mask & (sids >= 0)
    snap["unlabeled_kept"] = int((keep_mask & (sids < 0)).sum().item())
    kept = valid_mask.sum().item()
    snap["keep_total"] = int(kept)
    if kept == 0:
        return snap

    sids_all = sids[valid_mask]  # [M]
    vals_all = (topk_scores - thr_b1).clamp_min_(0.0)[valid_mask].float()
    us, inv = torch.unique(sids_all, return_inverse=True)
    S_sel = us.numel()
    snap["num_sents_kept"] = int(S_sel)

    # 句级支持度
    if agg == "mean":
        counts = torch.bincount(inv, minlength=S_sel).clamp_min_(1)
        sums = torch.zeros(S_sel, dtype=vals_all.dtype, device=device)
        sums.index_add_(0, inv, vals_all)
        sent_support = sums / counts
        kept_per_sent = counts
    elif agg == "max":
        kept_per_sent = torch.bincount(inv, minlength=S_sel).clamp_min_(1)
        sent_support = torch.full((S_sel,), fill_value=-1e-9, dtype=vals_all.dtype, device=device)
        sent_support.scatter_reduce_(0, inv, vals_all, reduce="amax", include_self=True)
    else:  # logsumexp
        kept_per_sent = torch.bincount(inv, minlength=S_sel).clamp_min_(1)
        sent_support = torch.empty(S_sel, dtype=vals_all.dtype, device=device)
        for k in range(S_sel):
            vk = vals_all[inv == k]
            m0 = torch.max(vk)
            sent_support[k] = m0 + torch.log(torch.clamp(torch.exp(vk - m0).sum(), min=EPS))

    # 句长归一
    bucket_sizes = [int(buckets.get(int(s.item()), torch.empty(0, device=device)).numel()) for s in us]
    def _norm(n: int) -> float:
        if sent_norm == "none": return 1.0
        if sent_norm == "count": return 1.0 / max(1.0, float(n))
        if sent_norm == "sqrt":  return 1.0 / float(np.sqrt(max(1, n)))
        if sent_norm == "log":   return 1.0 / float(np.log2(max(2, n)))
        return 1.0
    norms = torch.tensor([_norm(n) for n in bucket_sizes], dtype=sent_support.dtype, device=device)
    sent_support = sent_support * norms  # [S_sel]

    # Gini/HHI（看是否被少数句子劫持）
    sup_np = sent_support.detach().cpu().numpy()
    snap["gini"] = gini_coefficient(sup_np)
    snap["hhi"] = hhi_index(sup_np)

    # Top-S
    keepS = min(int(topS), int(sent_support.numel())) if topS > 0 else int(sent_support.numel())
    if keepS > 0 and sent_support.numel() > 0:
        topS_val, topS_idx = torch.topk(sent_support, k=keepS, largest=True, sorted=True)
        us_sel = us[topS_idx]
        p = torch.softmax(topS_val, dim=0)
        ent = -(p * (torch.log(p + 1e-8))).sum() / np.log(float(max(1, keepS)))
        beta = float(1.0 - float(ent.item()))
        snap["entropy"] = float(ent.item())
        snap["beta"] = beta
        snap["topS"] = [int(s.item()) for s in us_sel]
        snap["topS_val"] = [float(v.item()) for v in topS_val]

    # 每句统计
    for i, sid in enumerate(us.tolist()):
        snap["per_sent"][int(sid)] = {
            "support": float(sent_support[i].item()),
            "bucket": int(bucket_sizes[i]),
            "kept": int(torch.bincount(inv, minlength=S_sel)[i].item()),
        }

    # 对照：per-query cap r=1 的句级支持（每个查询对同一句只投一票=最大票）
    # 复用上面的 kept索引 (inv==k -> 属于句k的条目)，再按查询 dim 归并
    # 构造 (query, sid) 的最大 (s - thr) 作为该查询对该句的票
    # 先把 kept 的 (query_idx, sid, val) 取出来：
    kept_q = torch.nonzero(valid_mask, as_tuple=False)[:, 0]  # [M]
    kept_sid = sids_all
    kept_val = vals_all
    # 对每个 (q, sid) 取 max：
    # 建字典累加（避免 GPU 稀疏）：转到 CPU
    q_list = kept_q.detach().cpu().tolist()
    sid_list = kept_sid.detach().cpu().tolist()
    val_list = kept_val.detach().cpu().tolist()
    qsid2max: Dict[Tuple[int,int], float] = {}
    for q, s, v in zip(q_list, sid_list, val_list):
        key = (int(q), int(s))
        if key not in qsid2max or v > qsid2max[key]:
            qsid2max[key] = float(v)
    # 汇总到句级
    cap1_support: Dict[int, float] = {}
    for (q, s), v in qsid2max.items():
        cap1_support[s] = cap1_support.get(s, 0.0) + v
    if cap1_support:
        # 与 sent_support 同一 sid 空间对齐
        cap1_vec = np.array([cap1_support.get(int(s.item()), 0.0) for s in us], dtype=np.float64)
        # 同样做 Top-S
        if keepS > 0 and cap1_vec.size > 0:
            topS_idx2 = np.argsort(-cap1_vec)[:keepS]
            topS_val2 = cap1_vec[topS_idx2]
            snap["topS_val_cap1"] = [float(x) for x in topS_val2.tolist()]
            snap["topS_cap1"] = [int(us[i].item()) for i in topS_idx2]
            snap["gini_cap1"] = gini_coefficient(cap1_vec)
            snap["hhi_cap1"] = hhi_index(cap1_vec)
    return snap

# ------------------------- GCB 应用（仅“模拟重排”） -------------------------
@torch.no_grad()
def gcb_apply_to_group(
    base_logits_bo: torch.Tensor,
    cand_sent_idx_o: torch.Tensor,
    buckets: Dict[int, torch.Tensor],
    *,
    topk: int = 512,
    q_quantile: float = 0.9,
    agg: str = "logsumexp",
    sent_norm: str = "sqrt",
    topS: int = 2,
    gamma: float = 0.9,
    gate_entropy: bool = True,
    intra_mode: str = "topr",
    intra_temp: float = 1.5,
    intra_topr: int = 2,
    mix_alpha: float = 1.0,
) -> torch.Tensor:
    B, O = base_logits_bo.shape
    device = base_logits_bo.device
    K = min(int(topk), O)
    if K <= 0 or B == 0 or O == 0:
        return base_logits_bo

    topk_scores, topk_idx = torch.topk(base_logits_bo, k=K, dim=1, largest=True, sorted=False)
    thr_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)
    keep_mask = topk_scores >= thr_b1
    sids = cand_sent_idx_o[topk_idx]
    valid_mask = keep_mask & (sids >= 0)
    if not valid_mask.any():
        return base_logits_bo

    sids_all = sids[valid_mask]
    vals_all = (topk_scores - thr_b1).clamp_min_(0.0)[valid_mask].float()
    us, inv = torch.unique(sids_all, return_inverse=True)
    S_sel = us.numel()

    if agg == "mean":
        counts = torch.bincount(inv, minlength=S_sel).clamp_min_(1)
        sums = torch.zeros(S_sel, dtype=vals_all.dtype, device=device)
        sums.index_add_(0, inv, vals_all)
        sent_support = sums / counts
    elif agg == "max":
        sent_support = torch.full((S_sel,), fill_value=-1e-9, dtype=vals_all.dtype, device=device)
        sent_support.scatter_reduce_(0, inv, vals_all, reduce="amax", include_self=True)
    else:  # logsumexp
        sent_support = torch.empty(S_sel, dtype=vals_all.dtype, device=device)
        for k in range(S_sel):
            vk = vals_all[inv == k]
            m0 = torch.max(vk)
            sent_support[k] = m0 + torch.log(torch.clamp(torch.exp(vk - m0).sum(), min=EPS))

    # 句长归一
    bucket_sizes = [int(buckets.get(int(s.item()), torch.empty(0, device=device)).numel()) for s in us]
    def _norm(n: int) -> float:
        if sent_norm == "none": return 1.0
        if sent_norm == "count": return 1.0 / max(1.0, float(n))
        if sent_norm == "sqrt":  return 1.0 / float(np.sqrt(max(1, n)))
        if sent_norm == "log":   return 1.0 / float(np.log2(max(2, n)))
        return 1.0
    norms = torch.tensor([_norm(n) for n in bucket_sizes], dtype=sent_support.dtype, device=device)
    sent_support = sent_support * norms

    keepS = min(int(topS), int(sent_support.numel())) if topS > 0 else int(sent_support.numel())
    if keepS <= 0:
        return base_logits_bo
    topS_val, topS_idx = torch.topk(sent_support, k=keepS, largest=True, sorted=True)
    us_sel = us[topS_idx]

    beta = gamma
    if gate_entropy and keepS > 1:
        p = F.softmax(topS_val, dim=0)
        ent = -(p * (p + EPS).log()).sum() / np.log(float(keepS))
        beta = gamma * float(1.0 - ent)

    boost_o = torch.zeros(O, dtype=base_logits_bo.dtype, device=device)
    for k in range(keepS):
        sid = int(us_sel[k].item())
        sup = beta * float(topS_val[k].item())
        if sup <= 0:
            continue
        bucket_idx = buckets.get(sid, None)
        if bucket_idx is None or bucket_idx.numel() == 0:
            continue

        if intra_mode == "softmax":
            base_mean = base_logits_bo[:, bucket_idx].mean(dim=0)
            p = F.softmax(base_mean / max(1e-6, intra_temp), dim=0)
            boost_o[bucket_idx] += sup * p
        elif intra_mode == "topr":
            r = min(int(intra_topr), int(bucket_idx.numel()))
            if r <= 0:
                continue
            base_mean = base_logits_bo[:, bucket_idx].mean(dim=0)
            top_idx = torch.topk(base_mean, k=r, largest=True, sorted=False).indices
            boost_o[bucket_idx[top_idx]] += sup / float(r)
        else:  # flat
            boost_o[bucket_idx] += sup / float(bucket_idx.numel())

    return base_logits_bo + float(mix_alpha) * boost_o.unsqueeze(0)

# ------------------------- 主流程（探针） -------------------------
def main(args):
    torch.set_float32_matmul_precision("high")
    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if "16" in amp or "fp16" in amp else None)

    test_rows = read_jsonl(Path(args.test_manifest))
    log(f"[INFO] test rows = {len(test_rows):,}")

    # 候选池
    A, pool_ids, candidate_rows = load_audio_pool_unique(test_rows, device=device, dtype=torch.float32)
    O = A.size(0)
    log(f"[INFO] candidate windows O={O}")

    # 句 index
    canon2idx, alias2idx, cand_sent_idx, sid2name = build_sentence_index_with_alias(candidate_rows)
    S = len(canon2idx)
    log(f"[INFO] sentences S={S}")
    cand_sent_idx = torch.tensor(cand_sent_idx, dtype=torch.long, device=device)
    buckets = _precompute_sentence_buckets(cand_sent_idx)

    # gt → pool index
    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in test_rows]

    # 组装“句子组”
    sent2idx: Dict[Tuple[str,str], List[int]] = {}
    for i, r in enumerate(test_rows):
        k = sent_key_for_group(r)
        sent2idx.setdefault(k, []).append(i)
    groups = list(sent2idx.values())
    log(f"[INFO] Grouping by sentence: groups={len(groups)}, avg windows/sent={len(test_rows)/max(1,len(groups)):.2f}")

    # 模型
    run_dir = Path(args.run_dir)
    subj_map = read_subject_mapping_from_records(run_dir)
    log(f"[SUBJECT] loaded from records: {len(subj_map)} subjects")
    ckpt_path = choose_ckpt_path(args)
    model, meta = load_model_from_ckpt(ckpt_path, run_dir, device=device)
    scale = meta.get("logit_scale_exp", None) if args.use_ckpt_logit_scale else None

    # 输出
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "results" / "gcb_probe")
    out_dir.mkdir(parents=True, exist_ok=True)
    f_probe = open(out_dir / "gcb_probe.jsonl", "w", encoding="utf-8")

    # 汇总
    n_queries = len(test_rows)
    gt_in_topS_flags = []
    margins = []
    betas = []
    keeps = []
    unlabeled_kept = []
    gini_list = []
    hhi_list = []
    gini_cap1_list = []
    hhi_cap1_list = []
    imp = harm = same = 0
    simulated = 0

    # 句长偏置相关性（全局）收集
    all_len_for_corr: List[float] = []
    all_sup_for_corr: List[float] = []

    # hubness：统计“非GT top1 句”的出现频次
    hub_counter: Dict[int, int] = {}

    def encode_rows(indices: List[int]) -> torch.Tensor:
        rows = [test_rows[i] for i in indices]
        if autocast_dtype is None:
            Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        else:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        return Y

    for g_idx in tqdm(range(len(groups)), desc="Probe (Grouped)"):
        q_indices = groups[g_idx]
        Y = encode_rows(q_indices)
        with torch.no_grad():
            logits = compute_logits_clip(Y, A, scale=scale)
            logits_pre = logits.clone()

        # 快照
        snap = gcb_support_snapshot(
            logits, cand_sent_idx, buckets,
            topk=args.gcb_topk, q_quantile=args.gcb_q, agg=args.gcb_agg,
            sent_norm=args.gcb_norm, topS=args.gcb_topS
        )

        # 组内 GT 句 id（用组内首个 query 的 gt）
        g0 = gt_index[q_indices[0]]
        gt_sid = int(cand_sent_idx[g0].item()) if int(cand_sent_idx[g0].item()) >= 0 else None

        # margin(best - gt)
        if gt_sid is not None and snap["per_sent"]:
            gt_sup = snap["per_sent"].get(gt_sid, {"support": 0.0})["support"]
            if len(snap["topS"]) > 0:
                best_sid = snap["topS"][0]
                best_sup = snap["per_sent"].get(best_sid, {"support": 0.0})["support"]
                margin = float(best_sup - gt_sup)
            else:
                best_sid = None; best_sup = None; margin = None
        else:
            gt_sup = None; best_sid = None; best_sup = None; margin = None

        gt_in_topS = (gt_sid in snap["topS"]) if (gt_sid is not None) else None
        if gt_in_topS is not None:
            gt_in_topS_flags.append(int(bool(gt_in_topS)))
        if margin is not None and math.isfinite(margin):
            margins.append(margin)
        if snap["beta"] is not None:
            betas.append(float(snap["beta"]))
        keeps.append(int(snap["keep_total"]))
        unlabeled_kept.append(int(snap["unlabeled_kept"]))
        if snap["gini"] is not None:
            gini_list.append(float(snap["gini"]))
        if snap["hhi"] is not None:
            hhi_list.append(float(snap["hhi"]))
        if "gini_cap1" in snap and snap["gini_cap1"] is not None:
            gini_cap1_list.append(float(snap["gini_cap1"]))
        if "hhi_cap1" in snap and snap["hhi_cap1"] is not None:
            hhi_cap1_list.append(float(snap["hhi_cap1"]))

        # 供长度偏置相关性：把每句的 (bucket_size, support) 收集
        for sid_str, recs in snap["per_sent"].items():
            all_len_for_corr.append(float(recs["bucket"]))
            all_sup_for_corr.append(float(recs["support"]))

        # hubness：统计“非GT top1”
        if len(snap["topS"]) > 0:
            top1_sid = int(snap["topS"][0])
            if gt_sid is None or top1_sid != gt_sid:
                hub_counter[top1_sid] = hub_counter.get(top1_sid, 0) + 1

        # 可选：模拟 GCB 重排，看 rank 改变
        rank_delta = None
        if args.simulate_rerank:
            simulated += 1
            logits_post = gcb_apply_to_group(
                logits, cand_sent_idx, buckets,
                topk=args.gcb_topk, q_quantile=args.gcb_q, agg=args.gcb_agg,
                sent_norm=args.gcb_norm, topS=args.gcb_topS, gamma=args.gcb_gamma,
                gate_entropy=args.gcb_gate_entropy, intra_mode=args.gcb_intra_mode,
                intra_temp=args.gcb_intra_temp, intra_topr=args.gcb_intra_topr,
                mix_alpha=1.0,
            )
            improved = harmed = same_ct = 0
            for j_in_group, global_j in enumerate(q_indices):
                gidx = gt_index[global_j]
                s_pre = logits_pre[j_in_group]; r_pre = int((s_pre > s_pre[gidx]).sum().item()) + 1
                s_post = logits_post[j_in_group]; r_post = int((s_post > s_post[gidx]).sum().item()) + 1
                if r_post < r_pre: improved += 1
                elif r_post > r_pre: harmed += 1
                else: same_ct += 1
            imp += improved; harm += harmed; same += same_ct
            rank_delta = {"improved": improved, "harmed": harmed, "same": same_ct}

        # 记录
        rec = {
            "group_first_query": int(q_indices[0]),
            "group_size": len(q_indices),
            "gt_sid": gt_sid,
            "gt_in_topS": gt_in_topS,
            "gt_support": gt_sup,
            "best_sid": best_sid,
            "best_support": best_sup,
            "margin_best_minus_gt": margin,
            "beta": snap["beta"],
            "entropy": snap["entropy"],
            "keep_total": snap["keep_total"],
            "num_sents_kept": snap["num_sents_kept"],
            "unlabeled_kept": snap["unlabeled_kept"],
            "gini": snap["gini"],
            "hhi": snap["hhi"],
            "topS": snap["topS"],
            "topS_val": snap["topS_val"],
            "rank_delta": rank_delta,
            "flags": {
                "gcb_topk": args.gcb_topk,
                "gcb_q": args.gcb_q,
                "gcb_agg": args.gcb_agg,
                "gcb_norm": args.gcb_norm,
                "gcb_topS": args.gcb_topS,
                "simulate_rerank": bool(args.simulate_rerank),
            }
        }
        f_probe.write(json.dumps(rec, ensure_ascii=False) + "\n")

    f_probe.close()

    # 汇总
    def safe_stats(arr):
        if not arr:
            return None
        a = np.array(arr, dtype=float)
        return {
            "count": int(a.size),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p05": float(np.percentile(a, 5)) if a.size >= 20 else None,
            "p95": float(np.percentile(a, 95)) if a.size >= 20 else None,
        }

    # 长度偏置相关性
    len_sup_corr = pearson_corr(all_len_for_corr, all_sup_for_corr)

    summary = {
        "queries": n_queries,
        "groups": len(groups),
        "gt_in_topS": (float(np.mean(gt_in_topS_flags)) if gt_in_topS_flags else None),
        "margin_stats": safe_stats(margins),
        "beta_stats": safe_stats(betas),
        "keep_windows_stats": safe_stats(keeps),
        "unlabeled_kept_ratio_stats": (
            (lambda r: {
                "mean": float(np.mean(r)),
                "median": float(np.median(r)),
                "p95": float(np.percentile(r, 95)) if len(r) >= 20 else None
            })(list(np.array(unlabeled_kept) / np.maximum(1, np.array(keeps))))
            if keeps else None
        ),
        "gini_stats": safe_stats(gini_list),
        "hhi_stats": safe_stats(hhi_list),
        "gini_cap1_stats": safe_stats(gini_cap1_list),
        "hhi_cap1_stats": safe_stats(hhi_cap1_list),
        "len_support_pearson": len_sup_corr,
        "simulate_rerank": bool(args.simulate_rerank),
        "rank_delta_overall": (
            {
                "improved": int(imp),
                "harmed": int(harm),
                "same": int(same),
                "improved_ratio": float(imp / max(1, (imp + harm + same))),
                "harmed_ratio": float(harm / max(1, (imp + harm + same))),
            } if args.simulate_rerank else None
        ),
        "notes": "hubness_top 保存为单独文件；sid2name 提供可读别名映射。",
    }

    with open(out_dir / "gcb_probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 导出 hubness top-K
    topK = max(1, int(args.hubness_topk))
    hub_items = sorted(hub_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:topK]
    hub_json = []
    for sid, cnt in hub_items:
        name = (sid2name.get(sid) or f"sid:{sid}")
        bsize = int(buckets.get(int(sid), torch.empty(0, device=device)).numel())
        hub_json.append({"sid": int(sid), "name": name, "count": int(cnt), "bucket": bsize})
    with open(out_dir / "hubness_top.json", "w", encoding="utf-8") as f:
        json.dump(hub_json, f, indent=2, ensure_ascii=False)

    # 也保存 sid2name 便于查看
    with open(out_dir / "sid2name.json", "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in sid2name.items()}, f, indent=2, ensure_ascii=False)

    log("[DONE] Probe files:")
    log(f"  - {out_dir / 'gcb_probe.jsonl'}")
    log(f"  - {out_dir / 'gcb_probe_summary.json'}")
    log(f"  - {out_dir / 'hubness_top.json'}")
    log(f"  - {out_dir / 'sid2name.json'}")

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16", "16-mixed"])
    p.add_argument("--out_dir", type=str, default="")

    # ---------- 评分缩放 ----------
    p.add_argument("--use_ckpt_logit_scale", action="store_true")

    # ---------- GCB 快照/模拟 参数 ----------
    p.add_argument("--gcb_topk", type=int, default=512)
    p.add_argument("--gcb_q", type=float, default=0.90)
    p.add_argument("--gcb_agg", type=str, default="logsumexp", choices=["mean", "max", "logsumexp"])
    p.add_argument("--gcb_norm", type=str, default="sqrt", choices=["none", "count", "sqrt", "log"])
    p.add_argument("--gcb_topS", type=int, default=2)
    p.add_argument("--gcb_gamma", type=float, default=0.9)
    p.add_argument("--gcb_gate_entropy", action="store_true")
    p.add_argument("--gcb_intra_mode", type=str, default="topr", choices=["flat", "softmax", "topr"])
    p.add_argument("--gcb_intra_temp", type=float, default=1.5)
    p.add_argument("--gcb_intra_topr", type=int, default=2)

    # ---------- 是否模拟一次 GCB 重排并统计 rank 变化 ----------
    p.add_argument("--simulate_rerank", action="store_true")

    # ---------- 额外导出 ----------
    p.add_argument("--hubness_topk", type=int, default=50)

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)



