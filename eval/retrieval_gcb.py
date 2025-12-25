#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrieval_gcb_decode.py

Base vs. GCB (post-hoc only; no QCCP, no Window-Vote) retrieval evaluation,
with qualitative long-sentence decoding examples.

Pipeline:
  Encode (MEG) →
  Similarity (base, candidate-only L2 normalization) →
  GCB (sentence-level soft consensus, across=sum; post-hoc)

Key additions compared to the base evaluator:
- Sentence-level "decoding" visualization:
  For each anchor_word_idx, take the Top-1 candidate window and map it back
  to a word token, then reconstruct the full sentence.
- Automatic selection of long-sentence examples:
  improved / unchanged / worsened, with configurable thresholds.
- Only GCB is kept. QCCP and Window-Vote are fully removed.

Default outputs (under runs/<run_dir>/results/retrieval_gcb_decode/):
  - metrics.json
  - ranks.txt
  - examples_long_sentences.json
  - (optional) preds_topkK.jsonl / preds_topkK.tsv

Required fields per test_manifest row:
  - global_segment_text
  - anchor_word_idx
  - original_audio_path
  - local_window_onset_in_audio_s / local_window_offset_in_audio_s
  - audio_feature_path
  - meg_win_path
  - sensor_coordinates_path
  - subject_id
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
# Constants (must match training / frontend assumptions)
# =============================================================================
TARGET_T = 360
AUDIO_D = 1024
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

    If `content_id` is present, use it directly.
    Otherwise, fall back to (audio_path, window_onset, window_offset).
    """
    if r.get("content_id"):
        return r["content_id"]
    a = r["original_audio_path"]
    s0 = float(r["local_window_onset_in_audio_s"])
    s1 = float(r["local_window_offset_in_audio_s"])
    return f"{a}::{s0:.3f}-{s1:.3f}"


def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    """Ensure audio features are shaped as [D, T] with D == AUDIO_D."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D audio array, got {x.shape}")
    if x.shape[0] == AUDIO_D:
        return x
    if x.shape[1] == AUDIO_D:
        return x.T
    return x if abs(x.shape[0] - AUDIO_D) < abs(x.shape[1] - AUDIO_D) else x.T


def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    """Ensure MEG windows are shaped as [C, T]."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T


def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    """Interpolate a [D, T'] tensor to target length T if needed."""
    if x.size(1) == T:
        return x
    return F.interpolate(
        x.unsqueeze(0),
        size=T,
        mode="linear",
        align_corners=False
    ).squeeze(0)


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
      canon2idx     : canonical sentence -> sentence index
      alias2idx     : alias -> sentence index
      cand_sent_idx : per-candidate sentence index (or -1 if unknown)
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
    """Normalize subject identifiers to a two-digit string."""
    if x is None:
        return None
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return f"{int(m.group(1)):02d}"


def read_subject_mapping_from_records(run_dir: Path) -> Dict[str, int]:
    """
    Load subject mapping from records/subject_mapping.json.
    """
    p = run_dir / "records" / "subject_mapping.json"
    assert p.exists(), f"[SUBJECT] not found: {p}"

    obj = json.loads(p.read_text(encoding="utf-8"))
    raw = obj["mapping"] if "mapping" in obj and isinstance(obj["mapping"], dict) else obj

    out = {}
    for k, v in raw.items():
        nk = _normalize_subject_key(k)
        if nk is not None:
            out[nk] = int(v)

    assert out, "[SUBJECT] empty mapping after normalization"
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
        assert best_txt.exists(), f"best_checkpoint.txt not found: {best_txt}"
        ckpt = best_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path(args.run_dir) / ckpt_path).resolve()
        assert ckpt_path.exists(), f"best ckpt not found: {ckpt_path}"
        log(f"[INFO] Using BEST checkpoint: {ckpt_path}")
        return ckpt_path

    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"--ckpt_path not found: {ckpt_path}"
    return ckpt_path


from models.meg_encoder import UltimateMEGEncoder


def _read_logit_scale_exp(ckpt_path: Path) -> Optional[float]:
    """Attempt to extract exp(logit_scale) from checkpoint."""
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


def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str):
    """Load UltimateMEGEncoder from checkpoint and records config."""
    model_cfg = load_cfg_from_records(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", {}) or hp.get("enc_cfg", {})

    assert model_cfg, "no model_cfg/enc_cfg found"

    # Evaluation mode: disable temporal pooling
    if "out_timesteps" in UltimateMEGEncoder.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None

    model = UltimateMEGEncoder(**model_cfg)

    state = ckpt.get("state_dict", ckpt)
    new_state = {
        (k[6:] if k.startswith("model.") else k): v
        for k, v in state.items()
    }

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        log(f"[WARN] Missing keys: {len(missing)} (e.g., {missing[:8]})")
    if unexpected:
        log(f"[WARN] Unexpected keys: {len(unexpected)} (e.g., {unexpected[:8]})")

    model.eval().to(device)

    meta = {"logit_scale_exp": _read_logit_scale_exp(ckpt_path)}
    if meta["logit_scale_exp"] is not None:
        log(f"[INFO] exp(logit_scale) in ckpt: {meta['logit_scale_exp']:.6f}")

    return model, meta


@torch.no_grad()
def encode_meg_batch(
    model,
    batch_rows: List[dict],
    device: str,
    subj_map: Dict[str, int]
) -> torch.Tensor:
    """Encode a batch of MEG windows into [B, 1024, T] embeddings."""
    megs, locs, sidx = [], [], []
    miss = 0

    for r in batch_rows:
        mp = r["meg_win_path"]
        lp = r["sensor_coordinates_path"]

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
            sidx.append(0)
            miss += 1

    if miss:
        log(f"[WARN] {miss}/{len(batch_rows)} rows miss subject mapping; using subj_idx=0")

    meg = torch.stack(megs).to(device)
    loc = torch.stack(locs).to(device)
    sid = torch.tensor(sidx, dtype=torch.long, device=device)

    y = model(meg_win=meg, sensor_locs=loc, subj_idx=sid)
    if y.dim() != 3 or y.size(1) != AUDIO_D:
        raise RuntimeError(f"encoder must output [B,1024,T], got {tuple(y.shape)}")

    if y.size(2) != TARGET_T:
        y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)

    return y


# =============================================================================
# Candidate pool (unique windows)
# =============================================================================
@torch.no_grad()
def load_audio_pool_unique(
    test_rows: List[dict],
    device: str,
    dtype: torch.dtype
):
    """Load and de-duplicate candidate audio windows."""
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
        a = np.load(r["audio_feature_path"], allow_pickle=False).astype(np.float32)
        a = ensure_audio_DxT(a)
        ta = maybe_interp_1DT(torch.from_numpy(a), TARGET_T)
        feats.append(ta)

    A = torch.stack(feats).to(device=device, dtype=dtype)
    return A, ids, rep_rows


# =============================================================================
# Similarity (candidate-only normalization; CLIP-style)
# =============================================================================
def compute_logits_clip(
    queries: torch.Tensor,
    pool: torch.Tensor,
    scale: Optional[float] = None
) -> torch.Tensor:
    q = queries.to(torch.float32)
    A = pool.to(torch.float32)
    inv_norms = 1.0 / (A.norm(dim=(1, 2), p=2) + EPS)
    logits = torch.einsum("bct,oct,o->bo", q, A, inv_norms)
    if scale is not None:
        logits = logits * float(scale)
    return logits.to(torch.float32)


# =============================================================================
# Sentence buckets (for GCB)
# =============================================================================
def _precompute_sentence_buckets(
    cand_sent_idx_o: torch.Tensor
) -> Dict[int, torch.Tensor]:
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
# GCB (sentence-level soft consensus; across=sum)
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
        sid = int(us_sel[k])
        sup = gamma * float(topS_val[k])
        if sup <= 0:
            continue
        idx = buckets.get(sid)
        if idx is not None and idx.numel() > 0:
            boost_o[idx] += sup

    return base_logits_bo + boost_o.unsqueeze(0)


# =============================================================================
# Simple tokenization and decoding helpers
# =============================================================================
def tokenize_simple(s: str) -> List[str]:
    """Whitespace-based tokenization (must stay stable)."""
    return s.strip().split()


def build_cid2word(candidate_rows: List[dict]) -> Dict[str, str]:
    """
    Map candidate content_id to its corresponding anchor word token.
    """
    cid2w = {}
    for r in candidate_rows:
        cid = content_id_of(r)
        sent = (r.get("global_segment_text") or "").strip()
        toks = tokenize_simple(sent)
        ai = int(r.get("anchor_word_idx", -1))
        w = toks[ai] if 0 <= ai < len(toks) else "[UNK]"
        cid2w[cid] = w
    return cid2w


# =============================================================================
# Main evaluation
# =============================================================================
def evaluate(args):
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
    cand_sent_idx = torch.tensor(cand_sent_idx, dtype=torch.long, device=device)
    buckets = _precompute_sentence_buckets(cand_sent_idx)

    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in test_rows]

    run_dir = Path(args.run_dir)
    subj_map = read_subject_mapping_from_records(run_dir)
    log(f"[SUBJECT] loaded {len(subj_map)} subjects")

    ckpt_path = choose_ckpt_path(args)
    model, meta = load_model_from_ckpt(ckpt_path, run_dir, device)
    scale = meta["logit_scale_exp"] if args.use_ckpt_logit_scale else None

    topk_list = [int(x) for x in args.topk.split(",")]
    recalls = {"base": {k: 0 for k in topk_list}, "post": {k: 0 for k in topk_list}}
    mrr_sum_base = 0.0
    mrr_sum_post = 0.0
    ranks_post: List[int] = []

    out_dir = run_dir / "results" / "retrieval_gcb_decode"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group queries by sentence
    def sent_key_for_group(r: dict) -> Tuple[str, str]:
        als = sentence_aliases(r)
        return als[0] if als else ("unknown", content_id_of(r))

    sent2idx: Dict[Tuple[str, str], List[int]] = {}
    for i, r in enumerate(test_rows):
        sent2idx.setdefault(sent_key_for_group(r), []).append(i)

    groups = list(sent2idx.values())
    log(f"[INFO] Grouping by sentence: groups={len(groups)}, "
        f"avg windows/sent={len(test_rows)/max(1,len(groups)):.2f}")

    cid2word = build_cid2word(candidate_rows)

    long_sent_records = []

    @torch.no_grad()
    def encode_group(q_indices: List[int]):
        rows = [test_rows[i] for i in q_indices]
        if autocast_dtype is None:
            Y = encode_meg_batch(model, rows, device, subj_map)
        else:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                Y = encode_meg_batch(model, rows, device, subj_map)

        base = compute_logits_clip(Y, A, scale=scale)
        post = gcb_apply_to_group(
            base, cand_sent_idx, buckets,
            topk=args.gcb_topk,
            q_quantile=args.gcb_q,
            top_m=args.gcb_top_m,
            sent_norm=args.gcb_norm,
            topS=args.gcb_topS,
            gamma=args.gcb_gamma,
        )
        return base, post

    for q_idx in tqdm(groups, desc="Evaluate (Grouped)"):
        base, post = encode_group(q_idx)

        for j_local, global_j in enumerate(q_idx):
            gt = gt_index[global_j]
            s_b = base[j_local]
            s_p = post[j_local]

            rank_b = int((s_b > s_b[gt]).sum()) + 1
            rank_p = int((s_p > s_p[gt]).sum()) + 1

            mrr_sum_base += 1.0 / rank_b
            mrr_sum_post += 1.0 / rank_p
            ranks_post.append(rank_p)

            for k in topk_list:
                recalls["base"][k] += int(rank_b <= k)
                recalls["post"][k] += int(rank_p <= k)

        rows = [test_rows[i] for i in q_idx]
        sent_text = (rows[0].get("global_segment_text") or "").strip()
        gold_tokens = tokenize_simple(sent_text)
        n_tokens = len(gold_tokens)
        n_windows = len(q_idx)

        if n_windows < args.min_windows or n_tokens < args.min_tokens:
            continue

        base_tokens = ["[PAD]"] * n_tokens
        post_tokens = ["[PAD]"] * n_tokens

        for j_local, global_j in enumerate(q_idx):
            ai = int(test_rows[global_j].get("anchor_word_idx", j_local))
            if not (0 <= ai < n_tokens):
                continue

            cid_b = pool_ids[int(torch.argmax(base[j_local]))]
            cid_p = pool_ids[int(torch.argmax(post[j_local]))]

            base_tokens[ai] = cid2word.get(cid_b, "[UNK]")
            post_tokens[ai] = cid2word.get(cid_p, "[UNK]")

        mask = [t != "[PAD]" for t in base_tokens]
        denom = max(1, sum(mask))

        base_acc = sum(
            bt == gt for bt, gt, m in zip(base_tokens, gold_tokens, mask) if m
        ) / denom
        post_acc = sum(
            pt == gt for pt, gt, m in zip(post_tokens, gold_tokens, mask) if m
        ) / denom

        long_sent_records.append({
            "sentence_text": sent_text,
            "subject_id": str(rows[0].get("subject_id", "")),
            "n_windows": n_windows,
            "n_tokens": n_tokens,
            "base_text": " ".join(base_tokens),
            "post_text": " ".join(post_tokens),
            "base_acc": base_acc,
            "post_acc": post_acc,
            "delta": post_acc - base_acc,
        })

    num_queries = len(test_rows)
    metrics = {
        "num_queries": num_queries,
        "pool_size": O,
        "recall_at": {
            "base": {str(k): recalls["base"][k] / num_queries for k in topk_list},
            "post": {str(k): recalls["post"][k] / num_queries for k in topk_list},
        },
        "mrr": {
            "base": mrr_sum_base / num_queries,
            "post": mrr_sum_post / num_queries,
        },
        "mean_rank_post": float(np.mean(ranks_post)),
        "median_rank_post": float(np.median(ranks_post)),
        "topk_list": topk_list,
        "flags": {
            "use_ckpt_logit_scale": bool(args.use_ckpt_logit_scale),
            "only_gcb": True,
        },
    }

    out_json = Path(args.save_json) if args.save_json else (out_dir / "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    out_ranks = Path(args.save_ranks) if args.save_ranks else (out_dir / "ranks.txt")
    with open(out_ranks, "w", encoding="utf-8") as f:
        for r in ranks_post:
            f.write(str(int(r)) + "\n")

    delta_thr = float(args.delta_threshold)
    eps = float(args.unchanged_eps)

    improved = [r for r in long_sent_records if r["delta"] >= delta_thr]
    unchanged = [r for r in long_sent_records if abs(r["delta"]) < eps]
    worsened = [r for r in long_sent_records if r["delta"] <= -delta_thr]

    improved.sort(key=lambda x: (-x["delta"], -x["post_acc"], -x["n_windows"]))
    unchanged.sort(key=lambda x: (-x["post_acc"], -x["n_windows"]))
    worsened.sort(key=lambda x: (x["delta"], -x["n_windows"]))

    examples = {
        "improved": improved[:args.n_improved],
        "unchanged": unchanged[:args.n_unchanged],
        "worsened": worsened[:args.n_worsened],
    }

    with open(out_dir / "examples_long_sentences.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    log("==== Retrieval Results (Base vs GCB) ====")
    ra_b = metrics["recall_at"]["base"]
    ra_p = metrics["recall_at"]["post"]
    log(f"[BASE] R@1={ra_b.get('1',0.0):.4f}  "
        f"R@5={ra_b.get('5',0.0):.4f}  "
        f"R@10={ra_b.get('10',0.0):.4f}  "
        f"MRR={metrics['mrr']['base']:.4f}")
    log(f"[POST] R@1={ra_p.get('1',0.0):.4f}  "
        f"R@5={ra_p.get('5',0.0):.4f}  "
        f"R@10={ra_p.get('10',0.0):.4f}  "
        f"MRR={metrics['mrr']['post']:.4f}")
    log(f"[INFO] Metrics saved to: {out_json.as_posix()}")
    log(f"[INFO] Ranks saved to  : {out_ranks.as_posix()}")
    log(f"[INFO] Long-sentence examples saved to: "
        f"{(out_dir / 'examples_long_sentences.json').as_posix()}")


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

    # GCB parameters
    p.add_argument("--gcb_topk", type=int, default=128)
    p.add_argument("--gcb_q", type=float, default=0.95)
    p.add_argument("--gcb_top_m", type=int, default=3)
    p.add_argument("--gcb_norm", default="bucket_sqrt", choices=["bucket_sqrt"])
    p.add_argument("--gcb_topS", type=int, default=3)
    p.add_argument("--gcb_gamma", type=float, default=0.7)

    # Output
    p.add_argument("--use_ckpt_logit_scale", action="store_true")
    p.add_argument("--save_json", default="")
    p.add_argument("--save_ranks", default="")

    # Long-sentence selection
    p.add_argument("--min_windows", type=int, default=20)
    p.add_argument("--min_tokens", type=int, default=20)
    p.add_argument("--delta_threshold", type=float, default=0.05)
    p.add_argument("--unchanged_eps", type=float, default=1e-6)
    p.add_argument("--n_improved", type=int, default=6)
    p.add_argument("--n_unchanged", type=int, default=3)
    p.add_argument("--n_worsened", type=int, default=3)

    return p.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    evaluate(args)
