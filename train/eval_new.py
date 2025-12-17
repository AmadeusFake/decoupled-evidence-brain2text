#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eval (time-preserving rerank + sentence/position gates)

- none:     local-only baseline (unchanged)
- window:   time-preserving rerank using neighboring-window bank (same-sentence)
- sentence: time-preserving rerank using full-sentence MEG (per sample)

All gates are parameter-free (softmax/sigmoid/Gaussian).
Works with existing checkpoints; no training required.
"""

import argparse
import json
import inspect
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, DefaultDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Assuming models and utils are in PYTHONPATH
from models.meg_encoder import UltimateMEGEncoder

# ---------- Constants ----------
TARGET_T = 360
AUDIO_D = 1024
EPS = 1e-8

# ---------- Logging ----------
def log(msg: str):
    print(msg, flush=True)

# ---------- IO ----------
def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---------- Keys / Meta ----------
def content_id_of(r: dict) -> str:
    if r.get("content_id"):
        return r["content_id"]
    audio_path = r.get("original_audio_path", r.get("audio_feature_path", ""))
    onset_s = float(r.get("local_window_onset_in_audio_s", r.get("onset_in_audio_s", 0.0)))
    offset_s = float(r.get("local_window_offset_in_audio_s", r.get("offset_in_audio_s", 0.0)))
    return f"{Path(audio_path).stem}::{onset_s:.3f}-{offset_s:.3f}"

def _maybe_float(r: dict, k: str, default: Optional[float] = None) -> Optional[float]:
    v = r.get(k, default)
    try:
        return None if v is None else float(v)
    except Exception:
        return default

def sentence_key_of(r: dict) -> str:
    audio_path = r.get("original_audio_path", r.get("audio_feature_path", ""))
    stem = Path(audio_path).stem
    s_on = r.get("sentence_onset_in_audio_s", r.get("sent_onset_in_audio_s"))
    s_off = r.get("sentence_offset_in_audio_s", r.get("sent_offset_in_audio_s"))
    if s_on is not None and s_off is not None:
        return f"{stem}::SENT[{float(s_on):.3f}-{float(s_off):.3f}]"
    if r.get("sentence_idx") is not None:
        return f"{stem}::IDX[{int(r['sentence_idx'])}]"
    if r.get("utt_id") is not None:
        return f"{stem}::UTT[{r['utt_id']}]"
    return f"{stem}::WHOLE"

def window_relpos_of(r: dict, sent_bounds: Dict[str, Tuple[float, float]]) -> float:
    sent_key = sentence_key_of(r)
    s_on, s_off = sent_bounds.get(sent_key, (None, None))
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

# ---------- Shapes ----------
def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D audio array, got {x.shape}")
    return x if x.shape[0] == AUDIO_D else x.T

def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(-1) == T:
        return x
    is_2d = x.dim() == 2
    if is_2d:
        x = x.unsqueeze(0)
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    if is_2d:
        x = x.squeeze(0)
    return x

# ---------- Run config / ckpt ----------
def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        with open(rec, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("model_cfg", cfg.get("enc_cfg", {}))
    return {}

def choose_ckpt_path(args) -> Path:
    if args.use_best_ckpt:
        best_txt_path = Path(args.run_dir) / "records" / "best_checkpoint.txt"
        assert best_txt_path.exists(), f"best_checkpoint.txt not found at {best_txt_path}"
        ckpt_str = best_txt_path.read_text(encoding="utf-8").strip().splitlines()[0]
        ckpt_path = Path(ckpt_str) if ckpt_str.startswith("/") else (Path(args.run_dir) / ckpt_str)
        assert ckpt_path.exists(), f"Best checkpoint not found: {ckpt_path}"
        log(f"Using BEST checkpoint from records: {ckpt_path}")
        return ckpt_path
    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"--ckpt_path not found: {ckpt_path}"
    return ckpt_path

def infer_tau_from_ckpt(ckpt: dict, default: float = 0.0) -> float:
    hps = ckpt.get("hyper_parameters", {})
    logit_scale = None
    if "logit_scale" in hps:
        logit_scale = float(hps["logit_scale"])
    elif "model_cfg" in hps and isinstance(hps["model_cfg"], dict):
        logit_scale = float(hps["model_cfg"].get("logit_scale", 0.0))
    if logit_scale is None:
        for k in ("model.logit_scale", "logit_scale"):
            if k in ckpt.get("state_dict", {}):
                with torch.no_grad():
                    ls = ckpt["state_dict"][k].float().mean().item()
                logit_scale = float(ls)
                break
    if logit_scale is None or logit_scale <= 0:
        return default
    return float(1.0 / logit_scale)

def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str) -> Tuple[UltimateMEGEncoder, float]:
    model_cfg = load_cfg_from_records(run_dir)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    if not model_cfg:
        hps = ckpt.get("hyper_parameters", {})
        model_cfg = hps.get("model_cfg", hps.get("enc_cfg", {}))
    assert model_cfg, "Model config not found in records or checkpoint"

    # Eval 模式；强制不做时间池化
    if "out_timesteps" in inspect.signature(UltimateMEGEncoder).parameters:
        model_cfg["out_timesteps"] = None

    model = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing: log(f"[WARN] Missing keys during model load: {len(missing)}")
    if unexpected: log(f"[WARN] Unexpected keys during model load: {len(unexpected)}")
    model.eval().to(device)

    tau_from_ckpt = infer_tau_from_ckpt(ckpt, default=0.0)
    return model, tau_from_ckpt

# ---------- Subject map ----------
def load_subject_map(run_dir: Path, train_rows: List[dict]) -> Dict[str, int]:
    snap = run_dir / "records" / "subject_mapping_snapshot.json"
    if snap.exists():
        with open(snap, "r", encoding="utf-8") as f:
            data = json.load(f)
        sub_map = data.get("mapping", {})
        if sub_map:
            log(f"Loaded subject map from run records: {len(sub_map)} subjects.")
            return {str(k): int(v) for k, v in sub_map.items()}
    log("Building subject map from TRAIN manifest as fallback.")
    ids = sorted({str(r["subject_id"]) for r in train_rows if "subject_id" in r})
    return {sid: i for i, sid in enumerate(ids)}

def filter_and_annotate_rows(rows: List[dict], sub_map: Dict[str, int], strict_subjects: bool) -> List[dict]:
    out = []
    miss_sub, miss_art = 0, 0
    for r in rows:
        sid = str(r.get("subject_id"))
        if strict_subjects and sid not in sub_map:
            miss_sub += 1
            continue
        if not all(Path(r.get(p, "")).exists() for p in ["sensor_coordinates_path", "meg_win_path", "audio_feature_path"]):
            miss_art += 1
            continue
        rr = dict(r)
        rr["_subject_idx"] = sub_map.get(sid, 0)
        rr["_sent_key"] = sentence_key_of(r)
        out.append(rr)
    if miss_sub: log(f"[Note] Skipped {miss_sub} test rows (subject unseen).")
    if miss_art: log(f"[Note] Skipped {miss_art} rows with missing files.")
    return out

# ---------- Sentence bounds (for relative position) ----------
def build_sentence_bounds(rows: List[dict]) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    tmp: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)
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
    for k, spans in tmp.items():
        ons = [a for a, _ in spans]
        offs = [b for _, b in spans]
        bounds[k] = (min(ons), max(offs))
    return bounds

# ---------- Audio pool (plus sentence meta & sentence-level cache) ----------
def load_audio_pool_unique(
    test_rows: List[dict],
    sent_bounds: Dict[str, Tuple[float, float]],
    device: str,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, List[str], List[str], torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns:
      pool_tensor: [O, C, T]
      pool_ids:    list[str]
      pool_sents:  list[str]
      pool_rpos:   [O]
      sent_audio_cache: dict[sent_key] -> [C]
    """
    unique_audio_paths: Dict[str, str] = {}
    pool_sents: Dict[str, str] = {}
    pool_rpos_vals: Dict[str, float] = {}
    for r in test_rows:
        cid = content_id_of(r)
        if cid in unique_audio_paths:
            continue
        unique_audio_paths[cid] = r["audio_feature_path"]
        s_key = sentence_key_of(r)
        pool_sents[cid] = s_key
        pool_rpos_vals[cid] = window_relpos_of(r, sent_bounds)

    pool_ids = list(unique_audio_paths.keys())
    feats = []
    for cid in tqdm(pool_ids, desc="Loading & Aligning Audio Pool"):
        path = unique_audio_paths[cid]
        audio_arr = ensure_audio_DxT(np.load(path).astype(np.float32))
        audio_tensor = torch.from_numpy(audio_arr)
        audio_tensor = maybe_interp_1DT(audio_tensor, TARGET_T)
        feats.append(audio_tensor)
    pool_tensor = torch.stack(feats, 0).to(device=device, dtype=dtype)  # [O,C,T]

    # sentence-level audio summary cache (mean over time of all windows in sentence)
    sent_groups: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
    for idx, cid in enumerate(pool_ids):
        s_key = pool_sents[cid]
        sent_groups[s_key].append(pool_tensor[idx].mean(dim=-1, keepdim=False))  # [C]
    sent_audio_cache: Dict[str, torch.Tensor] = {}
    for s_key, vecs in sent_groups.items():
        v = torch.stack(vecs, 0).mean(dim=0)  # [C]
        sent_audio_cache[s_key] = v

    pool_sents_list = [pool_sents[cid] for cid in pool_ids]
    pool_rpos = torch.tensor([pool_rpos_vals[cid] for cid in pool_ids], device=device, dtype=torch.float32)  # [O]
    return pool_tensor, pool_ids, pool_sents_list, pool_rpos, sent_audio_cache

# ---------- Base logits ----------
def compute_base_logits(queries: torch.Tensor, pool: torch.Tensor, sim: str, tau: float) -> torch.Tensor:
    q = queries.float()
    p = pool.float()
    if sim == "clip":
        qf = F.normalize(q.flatten(1), p=2, dim=1)
        pf = F.normalize(p.flatten(1), p=2, dim=1)
        logits = torch.matmul(qf, pf.t())
    elif sim == "cosine":
        qn = F.normalize(q.flatten(1), p=2, dim=1)
        pn = F.normalize(p.flatten(1), p=2, dim=1)
        logits = torch.matmul(qn, pn.t())
    elif sim == "dot":
        logits = torch.matmul(q.flatten(1), p.flatten(1).t())
    else:
        raise ValueError(f"Unsupported similarity: {sim}")
    if tau > 0:
        logits = logits / tau
    return logits

# ---------- Gates ----------
def time_gate(q: torch.Tensor, c: torch.Tensor, mode: str = "softmax", tau_t: float = 0.10) -> torch.Tensor:
    """
    q,c: [B,C,T] -> w: [B,T]
    """
    qn = F.normalize(q, p=2, dim=1)
    cn = F.normalize(c, p=2, dim=1)
    sim_t = (qn * cn).sum(dim=1)  # [B,T]
    if mode == "softmax":
        w = torch.softmax(sim_t / max(tau_t, 1e-4), dim=-1)
    elif mode == "sigmoid":
        w = torch.sigmoid(sim_t)
        w = w / (w.sum(dim=-1, keepdim=True) + EPS)
    else:
        w = torch.full_like(sim_t, 1.0 / sim_t.size(-1))
    return w  # [B,T]

def gaussian_pos_gate(r_q: torch.Tensor, r_o: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    r_q: [B], r_o: [B,O] -> [B,O]
    """
    delta = r_o - r_q.unsqueeze(1)
    return torch.exp(- (delta / max(sigma, 1e-4)) ** 2)

def sigmoid_sent_gate(s_q: torch.Tensor, s_o: torch.Tensor, tau_sent: float) -> torch.Tensor:
    """
    s_q: [B,C], s_o: [B,O,C] -> [B,O]
    """
    s_qn = F.normalize(s_q, p=2, dim=1).unsqueeze(1)  # [B,1,C]
    s_on = F.normalize(s_o, p=2, dim=2)               # [B,O,C]
    cos = (s_qn * s_on).sum(dim=-1)                   # [B,O]
    return torch.sigmoid(cos / max(tau_sent, 1e-4))

# ---------- Build context bank (window/sentence) ----------
def build_by_sentence(rows: List[dict]) -> Dict[str, List[dict]]:
    by_sent: DefaultDict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_sent[sentence_key_of(r)].append(r)
    for s_key in by_sent:
        by_sent[s_key].sort(key=lambda x: _maybe_float(x, "local_window_onset_in_audio_s", _maybe_float(x, "onset_in_audio_s", 0.0)) or 0.0)
    return by_sent

def select_neighbor_windows(r: dict, by_sent: Dict[str, List[dict]], max_windows: int, stride: int, exclude_self: bool, radius: int) -> List[dict]:
    s_key = sentence_key_of(r)
    seq = by_sent.get(s_key, [])
    if not seq:
        return []
    r_on = _maybe_float(r, "local_window_onset_in_audio_s", _maybe_float(r, "onset_in_audio_s", 0.0)) or 0.0
    idx = min(range(len(seq)), key=lambda i: abs((_maybe_float(seq[i], "local_window_onset_in_audio_s", _maybe_float(seq[i], "onset_in_audio_s", 0.0)) or 0.0) - r_on))
    cand_idx = []
    for k in range(1, radius + 1):
        if idx - k >= 0:
            cand_idx.append(idx - k)
        if idx + k < len(seq):
            cand_idx.append(idx + k)
    cand_idx = sorted(set(cand_idx))
    if exclude_self and idx in cand_idx:
        cand_idx.remove(idx)
    cand_idx = cand_idx[::max(1, stride)][:max_windows]
    return [seq[i] for i in cand_idx]

# ---------- Encode ----------
@torch.no_grad()
def encode_meg(
    model: UltimateMEGEncoder,
    arrs: List[np.ndarray],
    locs: List[np.ndarray],
    sidx: List[int],
    device: str,
    autocast_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    if len(arrs) == 0:
        # 这里返回空张量（不会走到插值），调用侧会特殊处理
        return torch.empty(0, model.out_channels if hasattr(model, "out_channels") else 0, TARGET_T, device=device)
    megs = torch.stack([torch.from_numpy(ensure_meg_CxT(x)) for x in arrs]).to(device)
    locs_t = torch.stack([torch.from_numpy(l) for l in locs]).to(device)
    sidx_t = torch.tensor(sidx, dtype=torch.long, device=device)
    sig = inspect.signature(model.forward)
    kwargs = {}
    if "sensor_locs" in sig.parameters: kwargs["sensor_locs"] = locs_t
    if "subj_idx" in sig.parameters:    kwargs["subj_idx"]    = sidx_t
    with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
        y = model(meg_win=megs, **kwargs) if "meg_win" in sig.parameters else model(megs, **kwargs)
    if y.dim() == 2:
        y = y.unsqueeze(-1).repeat(1, 1, TARGET_T)
    y = maybe_interp_1DT(y, TARGET_T)
    return y

@torch.no_grad()
def encode_meg_batch_with_context(
    model: UltimateMEGEncoder,
    batch_rows: List[dict],
    by_sent: Optional[Dict[str, List[dict]]],
    device: str,
    context_mode: str,
    autocast_dtype: Optional[torch.dtype],
    ctx_max_windows: int = 6,
    ctx_stride: int = 1,
    exclude_self_from_ctx: bool = False,
    ctx_exclude_radius: int = 2,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # 1) query local window
    q_arrs = [np.load(r["meg_win_path"]) for r in batch_rows]
    locs  = [np.load(r["sensor_coordinates_path"]) for r in batch_rows]
    sidx  = [int(r["_subject_idx"]) for r in batch_rows]
    q = encode_meg(model, q_arrs, locs, sidx, device, autocast_dtype)  # [B,C,T]

    if context_mode == "none":
        return q, None

    if context_mode == "sentence":
        sent_paths = []
        for r in batch_rows:
            p = r.get("meg_sent_path", r.get("meg_sentence_path", None))
            if p is None or not Path(p).exists():
                p = r["meg_win_path"]
            sent_paths.append(p)
        c_arrs = [np.load(p) for p in sent_paths]
        c = encode_meg(model, c_arrs, locs, sidx, device, autocast_dtype)
        return q, c

    if context_mode == "window":
        assert by_sent is not None
        banks: List[torch.Tensor] = []
        for r in batch_rows:
            neigh = select_neighbor_windows(r, by_sent, max_windows=ctx_max_windows, stride=ctx_stride,
                                            exclude_self=exclude_self_from_ctx, radius=ctx_exclude_radius)
            if len(neigh) == 0:
                banks.append(torch.zeros(0, q.size(1), TARGET_T, device=device))
                continue
            arrs = [np.load(x["meg_win_path"]) for x in neigh]
            locs_n = [np.load(x["sensor_coordinates_path"]) for x in neigh]
            sidx_n = [int(x["_subject_idx"]) for x in neigh]
            y = encode_meg(model, arrs, locs_n, sidx_n, device, autocast_dtype)  # [K,C,T]
            banks.append(y)
        # reduce 统一到 [B,C,T]
        c_list = []
        for b in banks:
            if b.numel() == 0:
                c_list.append(torch.zeros_like(q[0:1]))
            else:
                c_list.append(b.mean(dim=0, keepdim=True))
        c = torch.cat(c_list, dim=0)
        return q, c

    raise ValueError(f"Unknown context_mode: {context_mode}")

# ---------- Evaluation ----------
def evaluate(args):
    device = args.device
    autocast_dtype = torch.bfloat16 if args.amp == "bf16" else (torch.float16 if "16" in args.amp else None)

    # --- Load data & meta ---
    test_rows_raw = read_jsonl(Path(args.test_manifest))
    train_rows_raw = read_jsonl(Path(args.train_manifest))
    run_dir = Path(args.run_dir)

    sub_map = load_subject_map(run_dir, train_rows_raw)
    sent_bounds = build_sentence_bounds(test_rows_raw)
    test_rows = filter_and_annotate_rows(test_rows_raw, sub_map, args.strict_subjects)
    assert test_rows, "No valid test rows after filtering."

    # --- Candidate Pool (+ sentence meta & rpos) ---
    audio_pool, pool_ids, pool_sents, pool_rpos, sent_audio_cache = load_audio_pool_unique(
        test_rows, sent_bounds, device, dtype=torch.float32
    )
    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_indices = [cid_to_index[content_id_of(r)] for r in test_rows]

    # --- Model ---
    ckpt_path = choose_ckpt_path(args)
    model, tau_from_ckpt = load_model_from_ckpt(ckpt_path, run_dir, device)
    tau = args.tau if args.tau > 0 else (tau_from_ckpt if tau_from_ckpt > 0 else 0.0)
    if tau > 0 and args.sim == "clip":
        log(f"[Info] Using temperature tau = {tau:.5f} (ckpt or arg)")

    # --- Group by sentence (for window mode) ---
    by_sent = build_by_sentence(test_rows) if args.context_mode == "window" else None

    # --- Loop ---
    B = args.batch_size
    topk_vals = [int(k) for k in args.topk.split(",")]
    recalls = {k: 0 for k in topk_vals}
    mrr_sum = 0.0
    ranks: List[int] = []

    do_rerank = (args.context_mode != "none") and (args.rerank_topk > 0)

    pbar = tqdm(range(0, len(test_rows), B), desc=f"Evaluating ({args.context_mode})")
    for i in pbar:
        batch_rows = test_rows[i: i + B]

        # encode query & context
        q, c = encode_meg_batch_with_context(
            model, batch_rows, by_sent, device, args.context_mode, autocast_dtype,
            ctx_max_windows=args.ctx_max_windows, ctx_stride=args.ctx_stride,
            exclude_self_from_ctx=args.exclude_self_from_ctx, ctx_exclude_radius=args.ctx_exclude_radius
        )  # q:[B,C,T], c:[B,C,T] or None

        base = compute_base_logits(q, audio_pool, args.sim, tau)  # [B,O]
        final = base.clone()

        if do_rerank:
            c_used = c if c is not None else q

            # sentence-level query summary s_q : [B,C]
            s_q = c_used.mean(dim=-1)  # [B,C]

            # r_q: [B]
            r_q = torch.tensor(
                [window_relpos_of(r, sent_bounds) for r in batch_rows],
                device=device, dtype=torch.float32
            )

            # TopK indices per query
            K = min(args.rerank_topk, audio_pool.size(0))
            topk_vals_idx = torch.topk(base, k=K, dim=1, largest=True, sorted=False).indices  # [B,K]

            # time gate w_t: [B,T]
            if args.no_time_gate:
                w = torch.full((q.size(0), q.size(-1)), 1.0 / q.size(-1), device=device, dtype=q.dtype)
            else:
                w = time_gate(q, c_used, mode=args.rerank_time_pool, tau_t=0.10)

            # chunked rerank to save memory
            chunk = max(1, args.rerank_chunk)
            Bsize, _ = topk_vals_idx.shape

            for start in range(0, K, chunk):
                end = min(K, start + chunk)
                sel_idx = topk_vals_idx[:, start:end]  # [B,Ck]

                # gather candidate frames: [B,Ck,C,T]
                cand = audio_pool.index_select(0, sel_idx.reshape(-1)).reshape(
                    Bsize, -1, audio_pool.size(1), audio_pool.size(2)
                )

                # sentence gate候选摘要：使用各候选所属句的缓存向量
                s_o_list = []
                r_o = torch.zeros(Bsize, sel_idx.size(1), device=device, dtype=torch.float32)
                for b in range(Bsize):
                    idx_b = sel_idx[b]  # [Ck]
                    s_vecs = []
                    # 句摘要
                    for oid in idx_b.tolist():
                        s_key = pool_sents[oid]
                        s_vecs.append(sent_audio_cache[s_key])  # [C] (tensor on device)
                    s_o_b = torch.stack(s_vecs, dim=0)  # [Ck,C]
                    s_o_list.append(s_o_b)
                    # 位置
                    r_o[b] = pool_rpos.index_select(0, idx_b)

                s_o = torch.stack(s_o_list, dim=0)  # [B,Ck,C]

                # frame-wise dot -> bias: [B,Ck,T] -> [B,Ck]
                frame_dot = (c_used.unsqueeze(1) * cand).sum(dim=2)  # [B,Ck,T]
                bias = (w.unsqueeze(1) * frame_dot).sum(dim=-1)      # [B,Ck]

                # sentence gate: [B,Ck]
                if args.no_sent_gate:
                    g_sent = torch.ones_like(bias)
                else:
                    g_sent = sigmoid_sent_gate(s_q, s_o, tau_sent=args.rerank_tau_sent)

                # position gate: [B,Ck]
                if args.no_pos_gate:
                    g_pos = torch.ones_like(bias)
                else:
                    g_pos = gaussian_pos_gate(r_q, r_o, sigma=args.rerank_sigma_pos)

                # robust normalize (row-wise center + clamp)
                bias = bias - bias.mean(dim=1, keepdim=True)
                bias = torch.clamp(bias, -args.bias_cap, args.bias_cap)

                # combine
                delta = args.rerank_alpha * bias * g_sent * g_pos  # [B,Ck]

                # scatter add into final logits
                final.scatter_add_(dim=1, index=sel_idx, src=delta)

        # ranking
        for j in range(len(batch_rows)):
            gt = gt_indices[i + j]
            scores = final[j]
            gt_score = scores[gt]
            rank = (scores > gt_score).sum().item() + 1
            ranks.append(rank)
            mrr_sum += 1.0 / rank
            for k in topk_vals:
                if rank <= k:
                    recalls[k] += 1

    # --- Report ---
    num_queries = len(test_rows)
    metrics = {
        "num_queries": num_queries,
        "pool_size": audio_pool.size(0),
        "similarity": args.sim,
        "recall_at": {str(k): v / num_queries for k, v in recalls.items()},
        "mrr": mrr_sum / num_queries,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "context_mode": args.context_mode,
        "rerank": do_rerank,
    }
    log("==== Retrieval Results ====")
    print(json.dumps(metrics, indent=2))

    out_dir = Path(args.run_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{args.context_mode}.json"
    if args.save_json:
        out_path = Path(args.save_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log(f"Metrics saved to: {out_path}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate (time-preserving rerank + sentence/position gates).")
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--strict_subjects", action="store_true")
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--sim", type=str, default="clip", choices=["clip", "cosine", "dot"])
    p.add_argument("--tau", type=float, default=0.0, help="temperature for base sim; 0 means infer from ckpt if possible")
    p.add_argument("--subject_map", type=str, default="auto", choices=["auto", "from_run", "train"])
    p.add_argument("--context_mode", type=str, default="none", choices=["none", "window", "sentence"])

    # window-mode sampling
    p.add_argument("--ctx_max_windows", type=int, default=16)
    p.add_argument("--ctx_stride", type=int, default=1)
    p.add_argument("--exclude_self_from_ctx", action="store_true")
    p.add_argument("--ctx_exclude_radius", type=int, default=2)

    # rerank hyperparams
    p.add_argument("--rerank_topk", type=int, default=200)
    p.add_argument("--rerank_alpha", type=float, default=1.0)
    p.add_argument("--bias_cap", type=float, default=0.10, help="clamp |bias| <= bias_cap after rowwise centering")
    p.add_argument("--rerank_tau_bias", type=float, default=0.06)   # reserved
    p.add_argument("--rerank_tau_sent", type=float, default=0.25)
    p.add_argument("--rerank_sigma_pos", type=float, default=0.25)
    p.add_argument("--rerank_time_pool", type=str, default="softmax", choices=["softmax", "sigmoid", "uniform"])
    p.add_argument("--rerank_bank_reduce", type=str, default="rms", choices=["rms", "mean"])
    p.add_argument("--rerank_chunk", type=int, default=512)

    # gates toggles
    p.add_argument("--no_sent_gate", action="store_true", help="disable sentence gate (g_sent=1)")
    p.add_argument("--no_pos_gate", action="store_true", help="disable position gate (g_pos=1)")
    p.add_argument("--no_time_gate", action="store_true", help="disable time gate (uniform w_t)")

    return p.parse_args()

# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    evaluate(args)
