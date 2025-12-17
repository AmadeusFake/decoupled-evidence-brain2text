#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-stage full evaluation:
1) Sentence-level retrieval (Stage-1) — with unique sentence pool metrics.
2) Window-level baseline (Stage-0) — content_id group-wise de-dup (group-max).
3) Window-level two-stage re-ranking (Stage-2) — sentence prior -> window scores, then group-max evaluate.

- Global(sentence) model: run_dir A  (e.g., StageG_CLIP_gOnly...)
- Local(window)   model: run_dir B  (e.g., none_baseline_safe...)

Outputs JSON for all three.
"""

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.meg_encoder import UltimateMEGEncoder

# ---------------- Const ----------------
TARGET_T = 360
AUDIO_D = 1024
EPS = 1e-8

def log(x: str): print(x, flush=True)

# ---------------- IO ----------------
def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---------------- Keys & helpers ----------------
def _maybe_float(r: dict, k: str, default: Optional[float] = None) -> Optional[float]:
    v = r.get(k, default)
    try:
        return None if v is None else float(v)
    except Exception:
        return default

def content_id_of(r: dict) -> str:
    if r.get("content_id"): return r["content_id"]
    audio_path = r.get("original_audio_path", r.get("audio_feature_path", ""))
    onset_s = float(r.get("local_window_onset_in_audio_s", r.get("onset_in_audio_s", 0.0)))
    offset_s = float(r.get("local_window_offset_in_audio_s", r.get("offset_in_audio_s", 0.0)))
    return f"{Path(audio_path).stem}::{onset_s:.3f}-{offset_s:.3f}"

def sentence_key_of(r: dict) -> str:
    audio_path = r.get("original_audio_path", r.get("audio_feature_path", ""))
    stem = Path(audio_path).stem
    # prefer explicit sentence boundary if provided
    s_on = r.get("sentence_onset_in_audio_s", r.get("sent_onset_in_audio_s"))
    s_off = r.get("sentence_offset_in_audio_s", r.get("sent_offset_in_audio_s"))
    if s_on is not None and s_off is not None:
        return f"{stem}::SENT[{float(s_on):.3f}-{float(s_off):.3f}]"
    if r.get("sentence_id"):
        return f"{stem}::SID[{r['sentence_id']}]"
    if r.get("sentence_idx") is not None:
        return f"{stem}::IDX[{int(r['sentence_idx'])}]"
    return f"{stem}::WHOLE"

def window_relpos_of(r: dict, bounds: Dict[str, Tuple[float, float]]) -> float:
    s_key = sentence_key_of(r)
    s_on, s_off = bounds.get(s_key, (None, None))
    if s_on is None or s_off is None:
        s_on = _maybe_float(r, "sentence_onset_in_audio_s", _maybe_float(r, "sent_onset_in_audio_s", None))
        s_off = _maybe_float(r, "sentence_offset_in_audio_s", _maybe_float(r, "sent_offset_in_audio_s", None))
    if s_on is None or s_off is None or s_off <= s_on:
        return 0.5
    w_on = _maybe_float(r, "local_window_onset_in_audio_s", _maybe_float(r, "onset_in_audio_s", 0.0)) or 0.0
    w_off = _maybe_float(r, "local_window_offset_in_audio_s", _maybe_float(r, "offset_in_audio_s", 0.0)) or 0.0
    w_c = 0.5 * (w_on + w_off)
    rpos = (w_c - s_on) / max(s_off - s_on, 1e-3)
    return float(min(1.0, max(0.0, rpos)))

# ---------------- Shapes ----------------
def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, f"audio array ndim=2 required, got {x.shape}"
    return x if x.shape[0] == AUDIO_D else x.T

def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(-1) == T: return x
    s2 = x.dim() == 2
    if s2: x = x.unsqueeze(0)
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    return x.squeeze(0) if s2 else x

# -------- Robust MEG + loc alignment to target channels --------
def _get_target_C_from_encoder(enc) -> int:
    for attr in ("in_channels", "input_channels", "spatial_channels"):
        if hasattr(enc, attr):
            try:
                v = int(getattr(enc, attr))
                if v > 0:
                    return v
            except Exception:
                pass
    return 208  # fallback

def ensure_meg_CxT_target(x: np.ndarray, target_C: int) -> np.ndarray:
    """
    Robust CxT:
    - If a dimension equals target_C, treat it as channel dim; transpose if needed.
    - Else fallback to heuristic (C <= T).
    - Finally, crop/pad channels to target_C.
    """
    assert x.ndim == 2, f"MEG array must be 2D, got {x.shape}"
    r, c = x.shape
    if r == target_C and c != target_C:
        CxT = x
    elif c == target_C and r != target_C:
        CxT = x.T
    else:
        CxT = x if r <= c else x.T
    C, T = CxT.shape
    if C == target_C:
        return CxT.astype(np.float32)
    if C > target_C:
        return CxT[:target_C, :].astype(np.float32)
    pad = np.zeros((target_C - C, T), dtype=CxT.dtype)
    return np.concatenate([CxT, pad], axis=0).astype(np.float32)

def ensure_locs_Cx3_target(l: np.ndarray, target_C: int) -> np.ndarray:
    """
    Align sensor_locs to [target_C, 3]:
    - Auto transpose to Cx3.
    - Crop/pad channels to target_C.
    - If malformed, return zeros.
    """
    if l.ndim != 2:
        return np.zeros((target_C, 3), dtype=np.float32)
    r, c = l.shape
    if c == 3:
        Cx3 = l
    elif r == 3:
        Cx3 = l.T
    else:
        return np.zeros((target_C, 3), dtype=np.float32)
    C, D = Cx3.shape
    if D != 3:
        return np.zeros((target_C, 3), dtype=np.float32)
    if C == target_C:
        return Cx3.astype(np.float32)
    if C > target_C:
        return Cx3[:target_C, :].astype(np.float32)
    pad = np.zeros((target_C - C, 3), dtype=Cx3.dtype)
    return np.concatenate([Cx3, pad], axis=0).astype(np.float32)

# ---------------- ckpt & encoder ----------------
def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "records" / "config.json"
    if p.exists():
        cfg = json.loads(p.read_text("utf-8"))
        return cfg.get("model_cfg", cfg.get("enc_cfg", {}))
    return {}

def choose_ckpt(run_dir: Path) -> Path:
    best_txt = run_dir / "records" / "best_checkpoint.txt"
    s = best_txt.read_text("utf-8").strip().splitlines()[0]
    ckpt_path = Path(s) if s.startswith("/") else (run_dir / s)
    assert ckpt_path.exists(), f"best ckpt not found: {ckpt_path}"
    return ckpt_path

def infer_tau_from_ckpt(ckpt: dict, default=0.0) -> float:
    hp = ckpt.get("hyper_parameters", {})
    ls = hp.get("logit_scale", None) or (hp.get("model_cfg", {}) or hp.get("enc_cfg", {})).get("logit_scale", None)
    if ls is None:
        for k in ("model.logit_scale", "logit_scale"):
            if k in ckpt.get("state_dict", {}):
                ls = float(ckpt["state_dict"][k].float().mean().item()); break
    if not ls or ls <= 0: return default
    return float(1.0/ls)

def load_encoder(run_dir: Path, device: str) -> Tuple[UltimateMEGEncoder, float]:
    model_cfg = load_cfg_from_records(run_dir)
    ckpt_path = choose_ckpt(run_dir)
    try: ckpt = torch.load(ckpt_path, map_location="cpu")
    except TypeError: ckpt = torch.load(ckpt_path, map_location="cpu")
    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", hp.get("enc_cfg", {}))
    if "out_timesteps" in inspect.signature(UltimateMEGEncoder).parameters:
        model_cfg["out_timesteps"] = None
    enc = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    state = { (k.replace("model.", "", 1) if k.startswith("model.") else k): v for k,v in state.items() }
    miss, unexp = enc.load_state_dict(state, strict=False)
    if miss:  log(f"[{run_dir.name}] missing keys: {len(miss)}")
    if unexp: log(f"[{run_dir.name}] unexpected keys: {len(unexp)}")
    enc.eval().to(device)
    tau = infer_tau_from_ckpt(ckpt, default=0.0)
    return enc, tau

# ---------------- subject map & bounds ----------------
def load_subject_map(run_dir: Path, train_rows: List[dict]) -> Dict[str,int]:
    snap = run_dir / "records" / "subject_mapping_snapshot.json"
    if snap.exists():
        data = json.loads(snap.read_text("utf-8"))
        m = data.get("mapping", {})
        if m:
            return {str(k): int(v) for k,v in m.items()}
    ids = sorted({str(r["subject_id"]) for r in train_rows if "subject_id" in r})
    return {sid:i for i, sid in enumerate(ids)}

def build_sentence_bounds(rows: List[dict]) -> Dict[str, Tuple[float,float]]:
    tmp: DefaultDict[str, List[Tuple[float,float]]] = defaultdict(list)
    for r in rows:
        s_key = sentence_key_of(r)
        s_on = r.get("sentence_onset_in_audio_s", r.get("sent_onset_in_audio_s"))
        s_off = r.get("sentence_offset_in_audio_s", r.get("sent_offset_in_audio_s"))
        if s_on is not None and s_off is not None:
            tmp[s_key].append((float(s_on), float(s_off)))
        else:
            w_on = _maybe_float(r, "local_window_onset_in_audio_s", _maybe_float(r, "onset_in_audio_s", None))
            w_off = _maybe_float(r, "local_window_offset_in_audio_s", _maybe_float(r, "offset_in_audio_s", None))
            if w_on is not None and w_off is not None:
                tmp[s_key].append((float(w_on), float(w_off)))
    bounds = {}
    for k, spans in tmp.items():
        ons = [a for a,_ in spans]; offs=[b for _,b in spans]
        bounds[k] = (min(ons), max(offs))
    return bounds

# ---------------- pools ----------------
@torch.no_grad()
def load_audio_window_pool(test_rows: List[dict], device: str) -> Tuple[torch.Tensor, List[str], List[str], Dict[str,List[int]]]:
    """
    Return:
      pool: [O,C,T] audio features (unique by content_id)
      pool_ids: list[str] content_ids in the same order
      pool_sents: list[str] sentence_key for each content_id
      sent_to_cand_indices: mapping from sentence_key -> list of pool indices (windows belonging to that sentence)
    NOTE: We load UNIQUE BY content_id to align with window-eval de-dup (group-max).
    """
    uniq: Dict[str, dict] = {}
    for r in test_rows:
        cid = content_id_of(r)
        if cid not in uniq:
            uniq[cid] = {
                "audio_path": r["audio_feature_path"],
                "sent_key": sentence_key_of(r)
            }
    pool_ids = list(uniq.keys())
    feats = []
    pool_sents = []
    for cid in tqdm(pool_ids, desc="Loading audio windows"):
        arr = ensure_audio_DxT(np.load(uniq[cid]["audio_path"]).astype(np.float32))
        t = torch.from_numpy(arr)
        t = maybe_interp_1DT(t, TARGET_T).to(device)
        feats.append(t)
        pool_sents.append(uniq[cid]["sent_key"])
    pool = torch.stack(feats, 0)  # [O,C,T]
    sent_to_cand_indices: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(pool_sents):
        sent_to_cand_indices[s].append(i)
    return pool, pool_ids, pool_sents, sent_to_cand_indices

@torch.no_grad()
def build_sentence_audio_pool(test_rows: List[dict], device: str) -> Tuple[torch.Tensor, List[str]]:
    """
    For each unique sentence_key, make a single [C] vector as sentence-level audio embedding
    by averaging the time-mean of all its windows.
    Return:
      sent_audio: [S, C]
      sent_keys: list[str]
    """
    groups: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
    for r in test_rows:
        s_key = sentence_key_of(r)
        a = ensure_audio_DxT(np.load(r["audio_feature_path"]).astype(np.float32))
        t = torch.from_numpy(a).to(device)
        t = maybe_interp_1DT(t, TARGET_T)   # [C,T]
        groups[s_key].append(t.mean(dim=-1))  # [C]
    sent_keys = sorted(groups.keys())
    vecs = [ torch.stack(groups[k],0).mean(dim=0) for k in sent_keys ]
    sent_audio = torch.stack(vecs, 0)  # [S,C]
    return sent_audio, sent_keys

# ---------------- encode ----------------
def pick_sentence_path(r: dict) -> str:
    for k in ["meg_sentence_full_path","meg_sentence_path","meg_sent_full_path","meg_sent_path","meg_sentence_full","meg_sentence"]:
        if k in r and r[k] and Path(r[k]).exists():
            return r[k]
    return r["meg_sentence_full_path"] if "meg_sentence_full_path" in r else r["meg_win_path"]

@torch.no_grad()
def encode_meg_any(
    enc: UltimateMEGEncoder,
    arrs: List[np.ndarray],
    locs: List[np.ndarray],
    sidx: List[int],
    device: str,
    amp: Optional[str],
) -> torch.Tensor:
    """
    Robust batch encode:
    - Align MEG to target C (from encoder), fix orientation, crop/pad channels.
    - Interp EACH sample to TARGET_T before stacking (so stack won't fail on variable sentence length).
    - Align sensor_locs to [C,3].
    """
    if len(arrs) == 0:
        return torch.zeros(0, getattr(enc, "out_channels", 0), TARGET_T, device=device)

    target_C = _get_target_C_from_encoder(enc)

    megs_list = []
    locs_list = []
    for x_np, l_np in zip(arrs, locs):
        x_aligned = ensure_meg_CxT_target(x_np, target_C)          # [C,T]
        x_t = torch.from_numpy(x_aligned)                           # cpu
        x_t = maybe_interp_1DT(x_t, TARGET_T)                       # [C,TARGET_T]
        megs_list.append(x_t)

        l_aligned = ensure_locs_Cx3_target(l_np, target_C)          # [C,3]
        locs_list.append(torch.from_numpy(l_aligned))

    megs = torch.stack(megs_list, 0).to(device)                     # [B,C,T]
    locs_t = torch.stack(locs_list, 0).to(device)                   # [B,C,3]
    sidx_t = torch.tensor(sidx, dtype=torch.long, device=device)

    sig = inspect.signature(enc.forward)
    kwargs = {}
    if "sensor_locs" in sig.parameters: kwargs["sensor_locs"] = locs_t
    if "subj_idx"    in sig.parameters: kwargs["subj_idx"]    = sidx_t

    dtype = torch.bfloat16 if amp=="bf16" else (torch.float16 if amp=="fp16" else None)
    with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype is not None)):
        if "meg_win" in sig.parameters:
            y = enc(meg_win=megs, **kwargs)
        else:
            y = enc(megs, **kwargs)

    if y.dim() == 2:
        y = y.unsqueeze(-1).repeat(1,1,TARGET_T)
    return maybe_interp_1DT(y, TARGET_T)

@torch.no_grad()
def encode_windows(enc, rows: List[dict], device: str, amp: Optional[str]) -> torch.Tensor:
    arrs = [np.load(r["meg_win_path"]) for r in rows]
    locs = [np.load(r["sensor_coordinates_path"]) for r in rows]
    sidx = [int(r["_subject_idx"]) for r in rows]
    return encode_meg_any(enc, arrs, locs, sidx, device, amp)  # [B,C,T]

@torch.no_grad()
def encode_sentences(enc, rows: List[dict], device: str, amp: Optional[str]) -> torch.Tensor:
    paths = [pick_sentence_path(r) for r in rows]
    arrs  = [np.load(p) for p in paths]
    locs  = [np.load(r["sensor_coordinates_path"]) for r in rows]
    sidx  = [int(r["_subject_idx"]) for r in rows]
    y = encode_meg_any(enc, arrs, locs, sidx, device, amp)  # [B,C,T]
    return y

# ---------------- sim ----------------
def clip_sim(q: torch.Tensor, p: torch.Tensor, tau: float = 0.0) -> torch.Tensor:
    if q.dim() == 3: q = q.flatten(1)
    if p.dim() == 3: p = p.flatten(1)
    q = F.normalize(q, p=2, dim=1)
    p = F.normalize(p, p=2, dim=1)
    out = q @ p.t()
    return out if tau<=0 else (out / tau)

# ---------------- ranks & metrics ----------------
def compute_metrics_from_ranks(ranks: List[int], topk: List[int]) -> Dict[str,Any]:
    n = len(ranks)
    recalls = {str(k): float(sum(1 for r in ranks if r<=k))/n for k in topk}
    mrr = float(sum(1.0/r for r in ranks))/n
    return {
        "num_queries": n,
        "recall_at": recalls,
        "mrr": mrr,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }

# ---------------- main eval ----------------
def evaluate(args):
    device = args.device
    amp = args.amp.lower()
    if torch.cuda.is_available():
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    # data
    test_rows_raw = read_jsonl(Path(args.test_manifest))
    train_rows_raw = read_jsonl(Path(args.train_manifest))

    # load maps/bounds
    subj_map = load_subject_map(Path(args.sent_run_dir), train_rows_raw)
    sent_bounds = build_sentence_bounds(test_rows_raw)

    # filter & annotate
    rows: List[dict] = []
    for r in test_rows_raw:
        if not (Path(r.get("sensor_coordinates_path","")).exists()
                and Path(r.get("meg_win_path","")).exists()
                and Path(r.get("audio_feature_path","")).exists()):
            continue
        rr = dict(r)
        rr["_subject_idx"] = subj_map.get(str(r.get("subject_id")), 0)
        rr["_sent_key"] = sentence_key_of(r)
        rows.append(rr)
    assert rows, "No valid rows."

    # pools
    audio_pool, pool_cids, pool_sents, sent_to_cand = load_audio_window_pool(rows, device)
    groups_for_eval = [[i] for i in range(len(pool_cids))]  # already unique by content_id

    # sentence-level audio pool
    sentence_audio, sentence_keys = build_sentence_audio_pool(rows, device)  # [S,C], list[str]
    sentkey_to_sid = {k:i for i,k in enumerate(sentence_keys)}

    # ground-truth ids
    gt_cand_idx = [ pool_cids.index(content_id_of(r)) for r in rows ]
    gt_sent_idx = [ sentkey_to_sid[sentence_key_of(r)] for r in rows ]

    # encoders
    sent_enc, tau_s_ckpt = load_encoder(Path(args.sent_run_dir), device)
    win_enc,  tau_w_ckpt = load_encoder(Path(args.win_run_dir),  device)
    tau_s = args.tau_sent if args.tau_sent>0 else (tau_s_ckpt if tau_s_ckpt>0 else 0.0)
    tau_w = args.tau_win  if args.tau_win>0 else (tau_w_ckpt if tau_w_ckpt>0 else 0.0)
    log(f"[tau] sentence={tau_s:.5f}, window={tau_w:.5f}")

    B = args.batch_size
    topk = [int(x) for x in args.topk.split(",")]

    # ==== Stage-1: sentence-level retrieval ====
    sent_ranks: List[int] = []
    for i in tqdm(range(0, len(rows), B), desc="Stage-1 SentEval"):
        batch = rows[i:i+B]
        Sq = encode_sentences(sent_enc, batch, device, amp).mean(dim=-1)  # [B,C]
        sims = clip_sim(Sq, sentence_audio, tau=tau_s)  # [B,S]
        for j in range(len(batch)):
            gt = gt_sent_idx[i+j]
            s = sims[j]
            rnk = int((s > s[gt]).sum().item()) + 1
            sent_ranks.append(rnk)
    sent_metrics = compute_metrics_from_ranks(sent_ranks, topk)

    # ==== Stage-0: window baseline (local model only) ====
    base_ranks: List[int] = []
    for i in tqdm(range(0, len(rows), B), desc="Stage-0 WinBaseline"):
        batch = rows[i:i+B]
        Qw = encode_windows(win_enc, batch, device, amp)  # [B,C,T]
        base = clip_sim(Qw, audio_pool, tau=tau_w)       # [B,O]
        for j in range(len(batch)):
            scores = base[j]  # [O]
            gt = gt_cand_idx[i+j]
            gt_score = scores[gt]
            rnk = int((scores > gt_score).sum().item()) + 1
            base_ranks.append(rnk)
    base_metrics = compute_metrics_from_ranks(base_ranks, topk)

    # ==== Stage-2: two-stage re-rank (sentence prior -> window) ====
    ts_ranks: List[int] = []
    alpha = float(args.alpha_sent)
    cap   = float(args.bias_cap)

    # map each candidate window to its sentence id
    sentkey_to_sid = {k:i for i,k in enumerate(sentence_keys)}
    pool_sent_sid = torch.tensor([sentkey_to_sid[s] for s in pool_sents], device=device, dtype=torch.long)  # [O]

    for i in tqdm(range(0, len(rows), B), desc="Stage-2 Two-Stage"):
        batch = rows[i:i+B]

        # window scores (base)
        Qw = encode_windows(win_enc, batch, device, amp)     # [B,C,T]
        base = clip_sim(Qw, audio_pool, tau=tau_w)           # [B,O]

        # sentence prior per query
        Sq   = encode_sentences(sent_enc, batch, device, amp).mean(dim=-1)  # [B,C]
        Spri = clip_sim(Sq, sentence_audio, tau=tau_s)                      # [B,Sent]
        s_prior = Spri.index_select(dim=1, index=pool_sent_sid)             # [B,O]
        s_prior = s_prior - s_prior.mean(dim=1, keepdim=True)
        s_prior = torch.clamp(s_prior, -cap, cap)

        final = base + alpha * s_prior

        for j in range(len(batch)):
            scores = final[j]    # [O]
            gt = gt_cand_idx[i+j]
            gt_score = scores[gt]
            rnk = int((scores > gt_score).sum().item()) + 1
            ts_ranks.append(rnk)

    ts_metrics = compute_metrics_from_ranks(ts_ranks, topk)

    # ==== Summary ====
    summary = {
        "settings": {
            "sent_run_dir": args.sent_run_dir,
            "win_run_dir":  args.win_run_dir,
            "tau_sent": tau_s,
            "tau_win":  tau_w,
            "alpha_sent": alpha,
            "bias_cap": cap,
            "batch_size": B,
        },
        "sentence_retrieval": sent_metrics,
        "window_baseline":    base_metrics,
        "window_two_stage":   ts_metrics,
    }
    print("\n==== FINAL RESULTS ====")
    print(json.dumps(summary, indent=2))

    out_dir = Path(args.save_json_dir) if args.save_json_dir else (Path(args.sent_run_dir)/"results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "two_stage_full_eval.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[Saved] {out_path}")

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser("Two-stage full eval (sentence + window baseline + 2-stage)")
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--test_manifest",  required=True, type=str)
    p.add_argument("--sent_run_dir",   required=True, type=str, help="global-only run dir (StageG...)")
    p.add_argument("--win_run_dir",    required=True, type=str, help="local baseline run dir")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp",    type=str, default="bf16", choices=["off","bf16","fp16"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--topk",       type=str, default="1,5,10")
    p.add_argument("--tau_sent", type=float, default=0.0, help="0 -> infer from ckpt if possible")
    p.add_argument("--tau_win",  type=float, default=0.0, help="0 -> infer from ckpt if possible")
    p.add_argument("--alpha_sent", type=float, default=1.0, help="weight of sentence prior in Stage-2")
    p.add_argument("--bias_cap",   type=float, default=0.10, help="clamp |prior| after centering")
    p.add_argument("--save_json_dir", type=str, default="")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
