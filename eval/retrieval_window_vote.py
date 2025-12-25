#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrieval_window_vote.py

Minimal, scan-aligned retrieval evaluator with optional context mechanisms.

Pipeline (fixed order, identical to training protocol):
  Encode (MEG) →
  Similarity (candidate-only L2, CLIP-style) →
  QCCP (hop-only, optional) →
  Window-Vote (scan-aligned, optional) →
  GCB (sentence-level soft consensus, post-hoc, optional)

Design principles:
- Candidate pool is de-duplicated by content_id (window-level uniqueness).
- Evaluation matches training: only candidate embeddings are normalized.
- All context mechanisms operate *within sentence groups* only.
- No cross-sentence leakage is introduced.

Optional diagnostics:
  --dump_per_query        : per-query ranks and hits (TSV)
  --dump_sentence_metrics: per-sentence aggregated metrics (TSV)
  --seed                 : reproducibility for stochastic ops (if any)
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

# =============================================================================
# Global constants (must match training / frontend assumptions)
# =============================================================================
TARGET_T = 360          # target temporal resolution after interpolation
AUDIO_D = 1024          # embedding dimensionality
EPS = 1e-8


# =============================================================================
# Basic utilities
# =============================================================================
def log(msg: str):
    """Flush-print helper for deterministic logging."""
    print(msg, flush=True)


def read_jsonl(p: Path) -> List[dict]:
    """Read a JSONL file into a list of dictionaries."""
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def content_id_of(r: dict) -> str:
    """
    Canonical content identifier for candidate pooling.

    If `content_id` is provided, use it directly.
    Otherwise, fall back to (audio_path, window_onset, window_offset).
    """
    if r.get("content_id"):
        return r["content_id"]

    a = r["original_audio_path"]
    s0 = float(r["local_window_onset_in_audio_s"])
    s1 = float(r["local_window_offset_in_audio_s"])
    return f"{a}::{s0:.3f}-{s1:.3f}"


def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    """
    Ensure audio features are shaped as [D, T] with D == AUDIO_D.
    Transpose if necessary.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D audio array, got {x.shape}")

    if x.shape[0] == AUDIO_D:
        return x
    if x.shape[1] == AUDIO_D:
        return x.T

    # Heuristic fallback: choose orientation closer to AUDIO_D
    return x if abs(x.shape[0] - AUDIO_D) < abs(x.shape[1] - AUDIO_D) else x.T


def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    """
    Ensure MEG windows are shaped as [C, T].
    Convention: channels <= time.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T


def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Interpolate a [D, T'] tensor to target length T if needed.
    """
    if x.size(1) == T:
        return x
    return F.interpolate(
        x.unsqueeze(0),
        size=T,
        mode="linear",
        align_corners=False
    ).squeeze(0)

def get_meg_encoder_class(name: str):
    """
    Factory for selecting MEG encoder backbone.

    Parameters
    ----------
    name : {"dense", "exp"}
        Encoder variant.

    Returns
    -------
    Encoder class (UltimateMEGEncoder).
    """
    name = name.lower()
    if name == "dense":
        from models.meg_encoder_Dense import UltimateMEGEncoder
        return UltimateMEGEncoder
    elif name == "exp":
        from models.meg_encoder_ExpDilated import UltimateMEGEncoder
        return UltimateMEGEncoder
    else:
        raise ValueError(f"Unknown meg_encoder: {name}")

def window_centers(rows: List[dict], idxs: List[int]) -> torch.Tensor:
    """
    Compute temporal centers (in seconds) for a group of windows.
    Used for QCCP hop-based neighborhood construction.
    """
    centers = []
    for i in idxs:
        r = rows[i]
        s0 = float(r.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(r.get("local_window_offset_in_audio_s", s0))
        centers.append(0.5 * (s0 + s1))
    return torch.tensor(centers, dtype=torch.float32)


# =============================================================================
# Sentence aliasing / indexing
# =============================================================================
_CAND_SENT_KEYS = [
    "sentence_id", "sentence_uid", "utt_id", "utterance_id", "segment_id",
    "original_sentence_id", "sentence_path",
    "sentence_audio_path", "transcript_path"
]


def _round3(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return None


def sentence_aliases(row: dict):
    """
    Generate a list of possible (key, value) aliases identifying the same sentence.
    The first alias is treated as canonical.
    """
    aliases = []

    for k in _CAND_SENT_KEYS:
        v = row.get(k)
        if v not in (None, ""):
            aliases.append((f"k:{k}", str(v)))

    audio = (
        row.get("original_audio_path")
        or row.get("sentence_audio_path")
        or row.get("audio_path")
        or ""
    )

    so = (
        row.get("global_segment_onset_in_audio_s")
        if row.get("global_segment_onset_in_audio_s") is not None
        else row.get("original_sentence_onset_in_audio_s")
    )
    eo = (
        row.get("global_segment_offset_in_audio_s")
        if row.get("global_segment_offset_in_audio_s") is not None
        else row.get("original_sentence_offset_in_audio_s")
    )

    if audio and so is not None and eo is not None:
        so3, eo3 = _round3(so), _round3(eo)
        if so3 and eo3:
            aliases.append(("audio+sent", f"{audio}::{so3}-{eo3}"))

    if audio:
        aliases.append(("audio", audio))

    return aliases


def build_sentence_index_with_alias(candidate_rows: list):
    """
    Build sentence indices for the candidate pool with alias resolution.

    Returns:
      canon2idx    : canonical sentence -> sentence index
      alias2idx    : alias -> sentence index
      cand_sent_idx: per-candidate sentence index (or -1 if unknown)
    """
    canon2idx, alias2idx, cand_sent_idx = {}, {}, []

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


# =============================================================================
# Subject mapping
# =============================================================================
def _normalize_subject_key(x: Any) -> Optional[str]:
    """
    Normalize subject identifiers to a two-digit string ("01", "02", ...).
    """
    if x is None:
        return None
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return f"{int(m.group(1)):02d}"


def read_subject_mapping_from_records(run_dir: Path) -> Dict[str, int]:
    """
    Load subject mapping from records directory.

    Supported sources (priority order):
      1) records/subject_mapping.json
      2) records/subject_mapping_snapshot.json
      3) records/subject_mapping_path.txt -> external JSON
    """
    rec_dir = run_dir / "records"

    p = rec_dir / "subject_mapping.json"
    if not p.exists():
        snap = rec_dir / "subject_mapping_snapshot.json"
        if snap.exists():
            p = snap
        else:
            txt = rec_dir / "subject_mapping_path.txt"
            if txt.exists():
                target = Path(txt.read_text(encoding="utf-8").strip())
                assert target.exists(), f"[SUBJECT] Invalid path in {txt}: {target}"
                p = target
            else:
                raise FileNotFoundError(
                    f"[SUBJECT] No subject mapping found under {rec_dir}"
                )

    obj = json.loads(p.read_text(encoding="utf-8"))
    raw = obj["mapping"] if "mapping" in obj and isinstance(obj["mapping"], dict) else obj

    out: Dict[str, int] = {}
    for k, v in raw.items():
        nk = _normalize_subject_key(k)
        if nk is not None:
            out[nk] = int(v)

    assert out, "[SUBJECT] Empty mapping after normalization"
    return out


# =============================================================================
# Model loading
# =============================================================================
def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    """Load model configuration from records/config.json if present."""
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        cfg = json.loads(rec.read_text(encoding="utf-8"))
        return cfg.get("model_cfg", {}) or cfg.get("enc_cfg", {}) or {}
    return {}


def choose_ckpt_path(args) -> Path:
    """Resolve checkpoint path (explicit or best_checkpoint.txt)."""
    if args.use_best_ckpt:
        best_txt = Path(args.run_dir) / "records" / "best_checkpoint.txt"
        assert best_txt.exists(), f"Missing {best_txt}"
        ckpt = best_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path(args.run_dir) / ckpt_path).resolve()
        assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
        log(f"[INFO] Using BEST checkpoint: {ckpt_path}")
        return ckpt_path

    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"--ckpt_path not found: {ckpt_path}"
    return ckpt_path


def _read_logit_scale_exp(ckpt_path: Path) -> Optional[float]:
    """
    Attempt to extract exp(logit_scale) from checkpoint state dict.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)

    for k in ("model.scorer.logit_scale", "scorer.logit_scale", "logit_scale"):
        v = state.get(k)
        if v is not None:
            try:
                return float(torch.exp(v).item())
            except Exception:
                try:
                    return float(np.exp(float(v)))
                except Exception:
                    pass
    return None
def load_model_from_ckpt(
    ckpt_path: Path,
    run_dir: Path,
    device: str,
    meg_encoder: str,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load MEG encoder from checkpoint and records config, with selectable backbone.

    Parameters
    ----------
    ckpt_path : Path
        Lightning checkpoint path.
    run_dir : Path
        Run directory containing records/config.json.
    device : str
        Target device ("cpu" or "cuda").
    meg_encoder : str
        Encoder backbone selector: "dense" or "exp".

    Returns
    -------
    model : torch.nn.Module
        Instantiated encoder model loaded with state_dict (strict=False).
    meta : Dict[str, Any]
        Extra metadata (e.g., exp(logit_scale) if present in checkpoint).
    """
    EncoderCls = get_meg_encoder_class(meg_encoder)

    model_cfg = load_cfg_from_records(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", {}) or hp.get("enc_cfg", {})

    assert model_cfg, "No model_cfg / enc_cfg found"

    # Evaluation: disable temporal pooling inside encoder (if supported)
    if "out_timesteps" in EncoderCls.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None

    model = EncoderCls(**model_cfg)

    state = ckpt.get("state_dict", ckpt)
    new_state = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        log(f"[WARN] Missing keys: {len(missing)} (e.g. {missing[:8]})")
    if unexpected:
        log(f"[WARN] Unexpected keys: {len(unexpected)} (e.g. {unexpected[:8]})")

    model.eval().to(device)

    meta = {"logit_scale_exp": _read_logit_scale_exp(ckpt_path)}
    if meta["logit_scale_exp"] is not None:
        log(f"[INFO] exp(logit_scale) = {meta['logit_scale_exp']:.6f}")

    return model, meta


@torch.no_grad()
def encode_meg_batch(
    model,
    batch_rows: List[dict],
    device: str,
    subj_map: Dict[str, int]
) -> torch.Tensor:
    """
    Encode a batch of MEG windows into [B, 1024, T] embeddings.
    """
    megs, locs, subj_idx = [], [], []
    miss = 0

    for r in batch_rows:
        mp = r["meg_win_path"]
        lp = r["sensor_coordinates_path"]

        assert mp and Path(mp).exists(), f"Missing meg_win_path: {mp}"
        assert lp and Path(lp).exists(), f"Missing sensor_coordinates_path: {lp}"

        x = np.load(mp, allow_pickle=False).astype(np.float32)
        x = ensure_meg_CxT(x)
        megs.append(torch.from_numpy(x))

        loc = np.load(lp, allow_pickle=False).astype(np.float32)
        locs.append(torch.from_numpy(loc))

        sid = _normalize_subject_key(r.get("subject_id"))
        if sid is not None and sid in subj_map:
            subj_idx.append(subj_map[sid])
        else:
            subj_idx.append(0)
            miss += 1

    if miss:
        log(f"[WARN] {miss}/{len(batch_rows)} rows missing subject mapping; using subj_idx=0")

    meg = torch.stack(megs).to(device)
    loc = torch.stack(locs).to(device)
    sid = torch.tensor(subj_idx, dtype=torch.long, device=device)

    y = model(meg_win=meg, sensor_locs=loc, subj_idx=sid)
    if y.dim() != 3 or y.size(1) != AUDIO_D:
        raise RuntimeError(f"Expected [B,1024,T], got {tuple(y.shape)}")

    if y.size(2) != TARGET_T:
        y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)

    return y


# =============================================================================
# Candidate pool construction (unique windows)
# =============================================================================
@torch.no_grad()
def load_audio_pool_unique(
    test_rows: List[dict],
    device: str,
    dtype: torch.dtype
):
    """
    Load and de-duplicate candidate audio windows.

    Returns:
      A        : [O, 1024, 360] tensor
      pool_ids : list of content_ids
      rows     : representative rows for each candidate
    """
    uniq: Dict[str, int] = {}
    pool_ids: List[str] = []
    feats = []
    rep_rows = []

    for r in test_rows:
        cid = content_id_of(r)
        if cid in uniq:
            continue
        uniq[cid] = len(pool_ids)
        pool_ids.append(cid)
        rep_rows.append(r)

    for r in tqdm(rep_rows, desc="Load candidates"):
        a = np.load(r["audio_feature_path"], allow_pickle=False).astype(np.float32)
        a = ensure_audio_DxT(a)
        ta = maybe_interp_1DT(torch.from_numpy(a), TARGET_T)
        feats.append(ta)

    A = torch.stack(feats).to(device=device, dtype=dtype)
    return A, pool_ids, rep_rows


# =============================================================================
# Similarity (candidate-only normalization, CLIP-style)
# =============================================================================
def compute_logits_clip(
    queries: torch.Tensor,
    pool: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Compute query–candidate logits using candidate-only L2 normalization.
    """
    q = queries.to(torch.float32)
    A = pool.to(torch.float32)

    inv_norms = 1.0 / (A.norm(dim=(1, 2), p=2) + EPS)
    logits = torch.einsum("bct,oct,o->bo", q, A, inv_norms)

    if scale is not None:
        logits = logits * float(scale)

    return logits.to(torch.float32)


# =============================================================================
# Sentence buckets
# =============================================================================
def _precompute_sentence_buckets(
    cand_sent_idx_o: torch.Tensor
) -> Dict[int, torch.Tensor]:
    """
    Precompute candidate indices grouped by sentence index.
    """
    buckets = {}
    uniq = torch.unique(cand_sent_idx_o)
    for s in uniq.tolist():
        if s < 0:
            continue
        buckets[int(s)] = torch.nonzero(
            cand_sent_idx_o == s,
            as_tuple=False
        ).view(-1)
    return buckets


def _sent_len_norm_factor(n: int, mode: str = "none") -> float:
    """
    Sentence-length normalization factor.
    """
    if n <= 0 or mode == "none":
        return 1.0
    if mode in ("bucket_count", "count", "kept_count"):
        return 1.0 / float(n)
    if mode in ("bucket_sqrt", "sqrt"):
        return 1.0 / float(np.sqrt(max(1.0, n)))
    if mode == "log":
        return 1.0 / float(np.log2(n + 1.0))
    return 1.0


# =============================================================================
# GCB: sentence-level soft consensus (post-hoc)
# =============================================================================
@torch.no_grad()
def gcb_apply_to_group(
    base_logits_bo: torch.Tensor,
    cand_sent_idx_o: torch.Tensor,
    buckets: Dict[int, torch.Tensor],
    *,
    topk: int = 128,
    q_quantile: float = 0.95,
    top_m: int = 3,
    sent_norm: str = "bucket_sqrt",
    topS: int = 3,
    gamma: float = 0.7,
) -> torch.Tensor:
    """
    Apply Group Context Boost (GCB) to a sentence group.
    """
    B, O = base_logits_bo.shape
    if B == 0 or O == 0:
        return base_logits_bo

    K = min(int(topk), O)
    topk_scores, topk_idx = torch.topk(
        base_logits_bo, k=K, dim=1, largest=True, sorted=False
    )

    thr_b1 = torch.quantile(
        base_logits_bo, q=q_quantile, dim=1, keepdim=True
    )

    keep_mask = topk_scores >= thr_b1
    sids = cand_sent_idx_o[topk_idx]
    valid_mask = keep_mask & (sids >= 0)

    if not valid_mask.any():
        return base_logits_bo

    sids_all = sids[valid_mask]
    vals_all = (topk_scores - thr_b1).clamp_min_(0.0)[valid_mask].float()

    us, inv = torch.unique(sids_all, return_inverse=True)
    device = base_logits_bo.device

    sent_support = torch.empty(us.numel(), device=device)
    for k in range(us.numel()):
        vk = vals_all[inv == k]
        if vk.numel() == 0:
            sent_support[k] = -1e9
            continue
        m_take = min(max(1, int(top_m)), int(vk.numel()))
        sent_support[k] = torch.topk(vk, m_take).values.mean()

    bucket_sizes = [
        int(buckets.get(int(s.item()), torch.empty(0)).numel())
        for s in us
    ]
    norms = torch.tensor(
        [_sent_len_norm_factor(n, sent_norm) for n in bucket_sizes],
        device=device
    )
    sent_support = sent_support * norms

    keepS = min(int(topS), sent_support.numel()) if topS > 0 else sent_support.numel()
    if keepS <= 0:
        return base_logits_bo

    topS_val, topS_idx = torch.topk(sent_support, keepS)
    us_sel = us[topS_idx]

    boost_o = torch.zeros(O, device=device)
    for k in range(keepS):
        sid = int(us_sel[k].item())
        sup = gamma * float(topS_val[k].item())
        if sup <= 0:
            continue
        idx = buckets.get(sid)
        if idx is not None and idx.numel() > 0:
            boost_o[idx] += sup

    return base_logits_bo + boost_o.unsqueeze(0)


# =============================================================================
# QCCP: hop-only contextual reweighting
# =============================================================================
@torch.no_grad()
def qccp_rerank_group(
    base_logits_bo: torch.Tensor,
    *,
    times_b: Optional[torch.Tensor] = None,
    hops: int = 1,
    alpha: float = 0.6,
    topk: int = 128,
    q_quantile: float = 0.9,
) -> torch.Tensor:
    """
    Query-Conditioned Context Propagation (hop-only variant).
    """
    B, O = base_logits_bo.shape
    if B == 0 or O == 0:
        return base_logits_bo

    device = base_logits_bo.device
    K = min(int(topk), O)
    topk_idx = torch.topk(base_logits_bo, K, dim=1).indices
    thr_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)
    out = base_logits_bo.clone()

    if times_b is None:
        order = torch.arange(B, device=device)
    else:
        order = torch.argsort(times_b)

    pos = torch.empty(B, device=device, dtype=torch.long)
    pos[order] = torch.arange(B, device=device)

    W = torch.zeros(B, B, device=device)
    for i in range(B):
        pi = int(pos[i])
        lo = max(0, pi - hops)
        hi = min(B - 1, pi + hops)
        for j in order[lo:hi + 1]:
            d = abs(int(pos[j]) - pi)
            if d > 0:
                W[i, j] = alpha ** d

    for i in range(B):
        idx_i = topk_idx[i]
        Suj = base_logits_bo[:, idx_i]
        votes = (Suj - thr_b1).clamp_min_(0.0)
        support = (W[i].unsqueeze(1) * votes).sum(dim=0)
        denom = W[i].sum().clamp_min_(1e-6)
        out[i, idx_i] += support / denom

    return out


# =============================================================================
# Window-Vote: scan-aligned, sentence-consistent voting
# =============================================================================
@torch.no_grad()
def window_vote_rerank(
    logits_bo: torch.Tensor,
    cand_sent_idx_o: torch.Tensor,
    *,
    topk_window: int = 128,
    q_quantile: float = 0.95,
    sent_top_m: int = 3,
    sent_topS: int = 3,
    sent_norm: str = "bucket_sqrt",
    gamma: float = 0.7,
) -> torch.Tensor:
    """
    Scan-aligned window voting within sentence groups.
    """
    B, O = logits_bo.shape
    if B == 0 or O == 0:
        return logits_bo

    device = logits_bo.device
    K = min(int(topk_window), O)
    buckets = _precompute_sentence_buckets(cand_sent_idx_o)
    bucket_sizes = {s: idx.numel() for s, idx in buckets.items()}

    boost = logits_bo.clone()
    topk_scores, topk_idx = torch.topk(logits_bo, K, dim=1, sorted=True)
    sent_idx_topk = cand_sent_idx_o[topk_idx]

    for b in range(B):
        sco = topk_scores[b]
        idx = topk_idx[b]
        sids = sent_idx_topk[b]

        valid = sids >= 0
        if not valid.any():
            continue

        sco, idx, sids = sco[valid], idx[valid], sids[valid]
        thr = torch.quantile(sco, q=q_quantile)
        keep = sco >= thr
        if not keep.any():
            continue

        sco, idx, sids = sco[keep], idx[keep], sids[keep]
        us, inv = torch.unique(sids, return_inverse=True)

        sent_scores = []
        for k in range(us.numel()):
            vals = sco[inv == k]
            m_take = min(max(1, sent_top_m), vals.numel())
            agg = torch.topk(vals, m_take).values.mean()
            n = bucket_sizes.get(int(us[k]), 1)
            sent_scores.append(agg * _sent_len_norm_factor(n, sent_norm))

        sent_scores = torch.stack(sent_scores)
        keepS = min(sent_topS, sent_scores.numel())
        if keepS <= 0:
            continue

        topS_val, topS_idx = torch.topk(sent_scores, keepS)
        us_sel = us[topS_idx]

        for k in range(keepS):
            sup = gamma * float(topS_val[k])
            if sup <= 0:
                continue
            bid = buckets.get(int(us_sel[k]))
            if bid is not None:
                boost[b, bid] += sup

    return boost


# =============================================================================
# Evaluation
# =============================================================================
def evaluate(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = args.device
    amp = args.amp.lower()
    autocast_dtype = (
        torch.bfloat16 if amp == "bf16"
        else torch.float16 if amp in ("fp16", "16-mixed")
        else None
    )

    test_rows = read_jsonl(Path(args.test_manifest))
    log(f"[INFO] test rows = {len(test_rows):,}")

    A, pool_ids, candidate_rows = load_audio_pool_unique(
        test_rows, device=device, dtype=torch.float32
    )
    O = A.size(0)
    log(f"[INFO] candidate windows O={O}")

    _, _, cand_sent_idx = build_sentence_index_with_alias(candidate_rows)
    cand_sent_idx = torch.tensor(cand_sent_idx, device=device)
    buckets = _precompute_sentence_buckets(cand_sent_idx)

    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in test_rows]

    run_dir = Path(args.run_dir)
    subj_map = read_subject_mapping_from_records(run_dir)
    log(f"[SUBJECT] loaded {len(subj_map)} subjects")

    ckpt_path = choose_ckpt_path(args)

    # -------- ONLY CHANGE: pass args.meg_encoder --------
    model, meta = load_model_from_ckpt(
        ckpt_path=ckpt_path,
        run_dir=run_dir,
        device=device,
        meg_encoder=args.meg_encoder,
    )
    # -----------------------------------------------
    scale = meta["logit_scale_exp"] if args.use_ckpt_logit_scale else None

    topk_list = [int(x) for x in args.topk.split(",")]
    recalls = {k: 0 for k in topk_list}
    ranks, mrr_sum = [], 0.0

    sent2idx: Dict[Tuple[str, str], List[int]] = {}
    for i, r in enumerate(test_rows):
        key = sentence_aliases(r)[0] if sentence_aliases(r) else ("unknown", content_id_of(r))
        sent2idx.setdefault(key, []).append(i)

    for sent_key, q_indices in tqdm(sent2idx.items(), desc="Evaluate (Grouped)"):
        rows = [test_rows[i] for i in q_indices]

        if autocast_dtype is None:
            Y = encode_meg_batch(model, rows, device, subj_map)
        else:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                Y = encode_meg_batch(model, rows, device, subj_map)

        logits = compute_logits_clip(Y, A, scale)

        if not args.no_qccp:
            times = window_centers(test_rows, q_indices).to(device)
            logits = qccp_rerank_group(
                logits,
                times_b=times,
                hops=args.qccp_hops,
                alpha=args.qccp_alpha,
                topk=args.qccp_topk,
                q_quantile=args.qccp_q,
            )

        if not args.no_windowvote:
            logits = window_vote_rerank(
                logits,
                cand_sent_idx,
                topk_window=args.topk_window,
                q_quantile=args.q_quantile,
                sent_top_m=args.sent_top_m,
                sent_topS=args.sent_topS,
                sent_norm=args.sent_norm,
                gamma=args.gamma,
            )

        if not args.no_gcb:
            logits = gcb_apply_to_group(
                logits,
                cand_sent_idx,
                buckets,
                topk=args.gcb_topk,
                q_quantile=args.gcb_q,
                top_m=args.gcb_top_m,
                sent_norm=args.gcb_norm,
                topS=args.gcb_topS,
                gamma=args.gcb_gamma,
            )

        for j, gi in enumerate(q_indices):
            g = gt_index[gi]
            s = logits[j]
            rank = int((s > s[g]).sum()) + 1
            ranks.append(rank)
            mrr_sum += 1.0 / rank
            for k in topk_list:
                recalls[k] += int(rank <= k)

    num_q = len(test_rows)
    metrics = {
        "num_queries": num_q,
        "pool_size": O,
        "recall_at": {str(k): recalls[k] / num_q for k in topk_list},
        "mrr": mrr_sum / num_q,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }

    out_dir = run_dir / "results" / "retrieval_final_min"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    log("==== Retrieval Results ====")
    log(json.dumps(metrics, indent=2))

# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--ckpt_path", default="")
    p.add_argument("--use_best_ckpt", action="store_true")

    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", default="bf16", choices=["off", "bf16", "fp16", "16-mixed"])
    p.add_argument("--topk", default="1,5,10")

    p.add_argument("--no_qccp", action="store_true")
    p.add_argument("--qccp_hops", type=int, default=1)
    p.add_argument("--qccp_alpha", type=float, default=0.6)
    p.add_argument("--qccp_topk", type=int, default=128)
    p.add_argument("--qccp_q", type=float, default=0.9)

    p.add_argument("--no_windowvote", action="store_true")
    p.add_argument("--topk_window", type=int, default=128)
    p.add_argument("--q_quantile", type=float, default=0.95)
    p.add_argument("--sent_top_m", type=int, default=3)
    p.add_argument("--sent_topS", type=int, default=3)
    p.add_argument("--sent_norm", default="bucket_sqrt")
    p.add_argument("--gamma", type=float, default=0.7)

    p.add_argument("--no_gcb", action="store_true")
    p.add_argument("--gcb_topk", type=int, default=128)
    p.add_argument("--gcb_q", type=float, default=0.95)
    p.add_argument("--gcb_top_m", type=int, default=3)
    p.add_argument("--gcb_norm", default="bucket_sqrt")
    p.add_argument("--gcb_topS", type=int, default=3)
    p.add_argument("--gcb_gamma", type=float, default=0.7)
    p.add_argument(
        "--meg_encoder",
        default="dense",
        choices=["dense", "exp"],
        help="MEG encoder backbone: dense or exp",
    )

    p.add_argument("--use_ckpt_logit_scale", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    evaluate(args)
