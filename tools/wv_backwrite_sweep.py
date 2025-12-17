#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/wv_backwrite_sweep.py

目的：
- 对 Window-Vote 的“回写参数”做补扫：topk_window, q_quantile, sent_topS, gamma
- 保持其它设置与“扫描最优族”一致：sent_top_m=3，sent_norm=bucket_sqrt，across=sum
- 默认关闭 QCCP（更干净），可用 --with_qccp 开启；默认在 WV 后追加固定 GCB（--with_gcb/--no_gcb 可控）

输出：
- <run_dir>/results/wv_backwrite_sweep/wv_sweep.csv
- <run_dir>/results/wv_backwrite_sweep/wv_sweep.json
"""

import argparse, json, csv, re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------- 常量 -------------------------
TARGET_T = 360
AUDIO_D = 1024
EPS = 1e-8

# ------------------------- 日志 -------------------------
def log(msg: str):
    print(msg, flush=True)

# ------------------------- I/O -------------------------
def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def content_id_of(r: dict) -> str:
    if r.get("content_id"):  # 允许外部预置键
        return r["content_id"]
    a = r["original_audio_path"]
    s0 = float(r["local_window_onset_in_audio_s"])
    s1 = float(r["local_window_offset_in_audio_s"])
    return f"{a}::{s0:.3f}-{s1:.3f}"

def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(1) == T: return x
    return F.interpolate(x.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0)

def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2: raise ValueError(f"expect 2D audio array, got {x.shape}")
    if x.shape[0] == AUDIO_D: return x
    if x.shape[1] == AUDIO_D: return x.T
    return x if abs(x.shape[0]-AUDIO_D) < abs(x.shape[1]-AUDIO_D) else x.T

def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2: raise ValueError(f"expect 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

def window_centers(rows: List[dict], idxs: List[int]) -> torch.Tensor:
    t = []
    for i in idxs:
        r = rows[i]
        s0 = float(r.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(r.get("local_window_offset_in_audio_s", s0))
        t.append(0.5*(s0+s1))
    return torch.tensor(t, dtype=torch.float32)

# ------------------------- 句别名/分桶 -------------------------
_CAND_SENT_KEYS = [
    "sentence_id","sentence_uid","utt_id","utterance_id","segment_id",
    "original_sentence_id","sentence_path","sentence_audio_path","transcript_path",
]

def _round3(x):
    try: return f"{float(x):.3f}"
    except Exception: return None

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
        if so3 and eo3: aliases.append(("audio+sent", f"{a}::{so3}-{eo3}"))
    if a: aliases.append(("audio", a))
    return aliases

def build_sentence_index_with_alias(candidate_rows: list):
    canon2idx, alias2idx, cand_sent_idx = {}, {}, []
    for r in candidate_rows:
        als = sentence_aliases(r)
        if not als:
            cand_sent_idx.append(-1); continue
        canon = als[0]
        if canon not in canon2idx:
            canon2idx[canon] = len(canon2idx)
        sidx = canon2idx[canon]
        for a in als:
            alias2idx.setdefault(a, sidx)
        cand_sent_idx.append(sidx)
    return canon2idx, alias2idx, cand_sent_idx

def _precompute_sentence_buckets(cand_sent_idx_o: torch.Tensor) -> Dict[int, torch.Tensor]:
    buckets = {}
    uniq = torch.unique(cand_sent_idx_o)
    for s in uniq.tolist():
        if s < 0: continue
        buckets[int(s)] = torch.nonzero(cand_sent_idx_o == s, as_tuple=False).view(-1)
    return buckets

# ------------------------- 模型/相似度 -------------------------
def compute_logits_clip(Q: torch.Tensor, A: torch.Tensor, scale: Optional[float]) -> torch.Tensor:
    q = Q.to(torch.float32)
    a = A.to(torch.float32)
    inv = 1.0 / (a.norm(dim=(1,2), p=2) + EPS)  # [O]
    logits = torch.einsum("bct,oct,o->bo", q, a, inv)
    if scale is not None: logits = logits * float(scale)
    return logits.to(torch.float32)

def _normalize_subject_key(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x); m = re.search(r"(\d+)", s)
    if not m: return None
    return f"{int(m.group(1)):02d}"

def read_subject_mapping_from_records(run_dir: Path) -> Dict[str, int]:
    p = run_dir / "records" / "subject_mapping.json"
    assert p.exists(), f"[SUBJECT] not found: {p}"
    obj = json.loads(p.read_text(encoding="utf-8"))
    raw = obj["mapping"] if "mapping" in obj and isinstance(obj["mapping"], dict) else obj
    out = {}
    for k, v in raw.items():
        nk = _normalize_subject_key(k)
        if nk is not None: out[nk] = int(v)
    assert out, "[SUBJECT] empty mapping"
    return out

def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        cfg = json.loads(rec.read_text(encoding="utf-8"))
        return cfg.get("model_cfg", {}) or cfg.get("enc_cfg", {}) or {}
    return {}

def choose_ckpt_path(run_dir: Path, use_best_ckpt: bool, ckpt_path: str) -> Path:
    if use_best_ckpt:
        best_txt = run_dir / "records" / "best_checkpoint.txt"
        assert best_txt.exists(), f"best_checkpoint.txt not found: {best_txt}"
        ckpt = best_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        p = Path(ckpt)
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        assert p.exists(), f"best ckpt not found: {p}"
        log(f"[INFO] Using BEST checkpoint: {p}")
        return p
    p = Path(ckpt_path)
    assert p.exists(), f"--ckpt_path not found: {p}"
    return p

from models.meg_encoder import UltimateMEGEncoder  # 按你的工程结构

def _read_logit_scale_exp(ckpt_path: Path) -> Optional[float]:
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)
    for k in ("model.scorer.logit_scale","scorer.logit_scale","logit_scale"):
        v = state.get(k, None)
        if v is not None:
            try: return float(torch.exp(v).item())
            except Exception:
                try: return float(np.exp(float(v)))
                except Exception: pass
    return None

def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str) -> Tuple[UltimateMEGEncoder, Dict[str,Any]]:
    model_cfg = load_cfg_from_records(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", {}) or hp.get("enc_cfg", {})
    assert model_cfg, "no model_cfg/enc_cfg found"
    if "out_timesteps" in UltimateMEGEncoder.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None

    model = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    new_state = { (k[6:] if k.startswith("model.") else k): v for k, v in state.items() }
    model.load_state_dict(new_state, strict=False)
    model.eval().to(device)

    meta = {"logit_scale_exp": _read_logit_scale_exp(ckpt_path)}
    if meta["logit_scale_exp"] is not None:
        log(f"[INFO] exp(logit_scale) in ckpt: {meta['logit_scale_exp']:.6f}")
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
        if sid is not None and sid in subj_map: sidx.append(subj_map[sid])
        else: sidx.append(0); miss += 1
    if miss:
        log(f"[WARN] {miss}/{len(batch_rows)} rows miss subject mapping; use subj_idx=0")
    meg = torch.stack(megs, 0).to(device)
    loc = torch.stack(locs, 0).to(device)
    sid = torch.tensor(sidx, dtype=torch.long, device=device)
    y = model(meg_win=meg, sensor_locs=loc, subj_idx=sid)  # [B,1024,T]
    if y.size(2) != TARGET_T:
        y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)
    return y

@torch.no_grad()
def load_audio_pool_unique(test_rows: List[dict], device: str, dtype: torch.dtype):
    uniq: Dict[str, int] = {}; ids: List[str] = []; feats = []; rep_rows = []
    for r in test_rows:
        cid = content_id_of(r)
        if cid in uniq: continue
        uniq[cid] = len(ids); ids.append(cid); rep_rows.append(r)
    for r in tqdm(rep_rows, desc="Load candidates"):
        a = np.load(r["audio_feature_path"], allow_pickle=False).astype(np.float32)
        a = ensure_audio_DxT(a)
        ta = maybe_interp_1DT(torch.from_numpy(a), TARGET_T)
        feats.append(ta)
    A = torch.stack(feats, 0).to(device=device, dtype=dtype)  # [O,1024,360]
    return A, ids, rep_rows

# ------------------------- QCCP（可选，默认关） -------------------------
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
    B, O = base_logits_bo.shape
    if B == 0 or O == 0: return base_logits_bo
    device = base_logits_bo.device
    K = min(int(topk), O)
    topk_idx = torch.topk(base_logits_bo, k=K, dim=1, largest=True, sorted=False).indices
    thr_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)
    out = base_logits_bo.clone()

    if times_b is None:
        order = torch.arange(B, device=device)
    else:
        order = torch.argsort(times_b)
    pos = torch.empty(B, dtype=torch.long, device=device)
    pos[order] = torch.arange(B, device=device)

    W = torch.zeros(B, B, dtype=base_logits_bo.dtype, device=device)
    for i in range(B):
        pi = int(pos[i].item())
        lo = max(0, pi - int(hops)); hi = min(B-1, pi + int(hops))
        idx = order[lo:hi+1]
        for j in idx.tolist():
            if j == i: continue
            pj = int(pos[j].item()); d = abs(pj - pi)
            if d == 0 or d > int(hops): continue
            W[i, j] = float(alpha) ** float(d)

    for i in range(B):
        idx_i = topk_idx[i]
        Suj = base_logits_bo[:, idx_i]
        votes = (Suj - thr_b1).clamp_min_(0.0)
        weights = W[i].unsqueeze(1)
        support = (weights * votes).sum(dim=0)
        denom = W[i].sum().clamp_min_(1e-6)
        support = support / denom
        out[i, idx_i] = out[i, idx_i] + support
    return out

# ------------------------- WV（mean@topm=3, across=sum, norm=bucket_sqrt） -------------------------
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
    B, O = logits_bo.shape
    if B == 0 or O == 0: return logits_bo
    device = logits_bo.device
    K = int(min(topk_window, O))
    buckets = _precompute_sentence_buckets(cand_sent_idx_o)
    bucket_sizes = {s: (idx.numel() if idx is not None else 0) for s, idx in buckets.items()}

    base = logits_bo
    boost = logits_bo.clone()

    topk_scores, topk_idx = torch.topk(base, k=K, dim=1, largest=True, sorted=True)
    sent_idx_topk = cand_sent_idx_o[topk_idx]

    for b in range(B):
        idx = topk_idx[b]; sco = topk_scores[b]; sids = sent_idx_topk[b]
        valid = (sids >= 0)
        if not valid.any(): continue
        idx, sco, sids = idx[valid], sco[valid], sids[valid]

        thr = torch.quantile(sco, q=q_quantile)
        m = (sco >= thr)
        idx, sco, sids = idx[m], sco[m], sids[m]
        if idx.numel() == 0: continue

        us, inv = torch.unique(sids, return_inverse=True)
        if us.numel() == 0: continue

        # mean@topm + bucket_sqrt
        sent_scores = []
        for k_s in range(us.numel()):
            vals = sco[inv == k_s]
            m_take = min(max(1, int(sent_top_m)), int(vals.numel()))
            top_vals = torch.topk(vals, k=m_take, largest=True, sorted=False).values
            agg_val = torch.mean(top_vals)
            s_id = int(us[k_s].item())
            n_windows = bucket_sizes.get(s_id, 1)
            # 句长归一
            if sent_norm == "bucket_sqrt":
                agg_val = agg_val / float(np.sqrt(max(1.0, n_windows)))
            sent_scores.append(agg_val)

        sent_scores = torch.stack(sent_scores, dim=0)
        keepS = min(int(sent_topS), int(sent_scores.numel())) if sent_topS > 0 else int(sent_scores.numel())
        if keepS <= 0: continue
        topS_val, topS_idx_local = torch.topk(sent_scores, k=keepS, largest=True, sorted=True)
        us_sel = us[topS_idx_local]

        # across=sum（不改句内排序）
        for k_s in range(keepS):
            s_id = int(us_sel[k_s].item()); sup = gamma * float(topS_val[k_s].item())
            if sup <= 0: continue
            bucket_idx = buckets.get(s_id, None)
            if bucket_idx is None or bucket_idx.numel() == 0: continue
            boost[b, bucket_idx] += sup

    return boost  # base + sup

# ------------------------- GCB（固定参数，默认开） -------------------------
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
    if B == 0 or O == 0: return base_logits_bo
    K = min(int(topk), O)

    topk_scores, topk_idx = torch.topk(base_logits_bo, k=K, dim=1, largest=True, sorted=False)  # [B,K]
    thr_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)                   # [B,1]
    keep_mask = topk_scores >= thr_b1
    sids = cand_sent_idx_o[topk_idx]
    valid_mask = keep_mask & (sids >= 0)
    if not valid_mask.any(): return base_logits_bo

    sids_all = sids[valid_mask]
    vals_all = (topk_scores - thr_b1).clamp_min_(0.0)[valid_mask].float()
    us, inv = torch.unique(sids_all, return_inverse=True)
    if us.numel() == 0: return base_logits_bo

    # mean@topm per sentence
    sent_support = torch.empty(us.numel(), dtype=vals_all.dtype, device=base_logits_bo.device)
    for k in range(us.numel()):
        vk = vals_all[inv == k]
        m_take = min(max(1, int(top_m)), int(vk.numel()))
        top_vals = torch.topk(vk, k=m_take, largest=True, sorted=False).values
        sent_support[k] = torch.mean(top_vals)

    # 句长归一
    bucket_sizes = [int(buckets.get(int(s.item()), torch.empty(0, device=base_logits_bo.device)).numel()) for s in us]
    norms = torch.tensor([1.0/float(np.sqrt(max(1.0, n))) for n in bucket_sizes],
                         dtype=sent_support.dtype, device=base_logits_bo.device)
    sent_support = sent_support * norms

    keepS = min(int(topS), int(sent_support.numel())) if topS > 0 else int(sent_support.numel())
    if keepS <= 0: return base_logits_bo
    topS_val, topS_idx = torch.topk(sent_support, k=keepS, largest=True, sorted=True)
    us_sel = us[topS_idx]

    boost_o = torch.zeros(O, dtype=base_logits_bo.dtype, device=base_logits_bo.device)
    for k in range(keepS):
        sid = int(us_sel[k].item()); sup = gamma * float(topS_val[k].item())
        if sup <= 0: continue
        bucket_idx = buckets.get(sid, None)
        if bucket_idx is None or bucket_idx.numel() == 0: continue
        boost_o[bucket_idx] += sup

    return base_logits_bo + boost_o.unsqueeze(0)

# ------------------------- 编码 + 缓存 base logits -------------------------
@torch.no_grad()
def evaluate_and_cache_logits(args):
    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if "16" in amp or "fp16" in amp else None)

    test_rows = read_jsonl(Path(args.test_manifest))
    log(f"[INFO] test rows = {len(test_rows):,}")

    A, pool_ids, candidate_rows = load_audio_pool_unique(test_rows, device=device, dtype=torch.float32)
    O = A.size(0)
    log(f"[INFO] candidate windows O={O}")

    _, _, cand_sent_idx = build_sentence_index_with_alias(candidate_rows)
    cand_sent_idx = torch.tensor(cand_sent_idx, dtype=torch.long, device=device)
    buckets = _precompute_sentence_buckets(cand_sent_idx)

    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in test_rows]

    run_dir = Path(args.run_dir)
    subj_map = read_subject_mapping_from_records(run_dir)
    ckpt_path = choose_ckpt_path(run_dir, args.use_best_ckpt, args.ckpt_path)
    model, meta = load_model_from_ckpt(ckpt_path, run_dir, device=device)
    scale = meta.get("logit_scale_exp", None) if args.use_ckpt_logit_scale else None

    # 分句组
    def sent_key_for_group(r: dict) -> Tuple[str, str]:
        als = sentence_aliases(r)
        return als[0] if als else ("unknown", content_id_of(r))

    sent2idx: Dict[Tuple[str,str], List[int]] = {}
    for i, r in enumerate(test_rows):
        k = sent_key_for_group(r)
        sent2idx.setdefault(k, []).append(i)
    groups = list(sent2idx.values())
    if args.sample_groups and args.sample_groups > 0:
        groups = groups[:args.sample_groups]
    log(f"[INFO] Groups={len(groups)}, avg windows/sent={len(test_rows)/max(1,len(groups)):.2f}")

    # 编码 + base logits（可选 QCCP）
    base_logits_per_group: List[torch.Tensor] = []
    for g in tqdm(groups, desc="Encode+Logits (Grouped)"):
        rows = [test_rows[i] for i in g]
        if autocast_dtype is None:
            Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        else:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        logits = compute_logits_clip(Y, A, scale=scale)

        if args.with_qccp:
            times = window_centers(test_rows, g).to(device=logits.device)
            logits = qccp_rerank_group(
                logits, times_b=times,
                hops=args.qccp_hops, alpha=args.qccp_alpha,
                topk=args.qccp_topk, q_quantile=args.qccp_q
            )
        base_logits_per_group.append(logits.detach().cpu())  # 存 CPU，复用

    meta_pack = {
        "test_rows": test_rows,
        "pool_ids": pool_ids,
        "gt_index_all": gt_index,
        "groups": groups,
        "cand_sent_idx": cand_sent_idx.cpu(),
        "buckets": {k: v.cpu() for k, v in buckets.items()},
        "run_dir": run_dir,
        "pool_size": O,
    }
    return base_logits_per_group, meta_pack

# ------------------------- WV 补扫主流程 -------------------------
def run_wv_sweep(args):
    base_logits_per_group, meta = evaluate_and_cache_logits(args)
    test_rows = meta["test_rows"]
    pool_ids = meta["pool_ids"]
    gt_index_all = meta["gt_index_all"]
    groups = meta["groups"]
    cand_sent_idx = meta["cand_sent_idx"]
    buckets = meta["buckets"]
    run_dir = meta["run_dir"]
    O = meta["pool_size"]

    # 参数网格
    def parse_list(s, typ=float):
        return [typ(x.strip()) for x in s.split(",") if x.strip()]

    topk_windows = parse_list(args.wv_topk_window_list, int)
    q_list = parse_list(args.wv_q_list, float)
    topS_list = parse_list(args.wv_topS_list, int)
    gamma_list = parse_list(args.wv_gamma_list, float)
    sent_top_m = int(args.wv_sent_top_m)  # 固定 3（可改）

    out_dir = run_dir / "results" / "wv_backwrite_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "wv_sweep.csv"
    json_path = out_dir / "wv_sweep.json"

    rows_csv = []
    best = []

    topk_list = [int(x) for x in args.topk.split(",")]
    for topk_w in topk_windows:
        for q in q_list:
            for topS in topS_list:
                for gamma in gamma_list:
                    recalls = {k: 0 for k in topk_list}
                    mrr_sum = 0.0
                    ranks = []

                    # 逐组应用 WV（可选再接 GCB）
                    for g, logits_cpu in zip(groups, base_logits_per_group):
                        s = logits_cpu.to(torch.float32)
                        s = window_vote_rerank(
                            s.to(cand_sent_idx.device),
                            cand_sent_idx_o=cand_sent_idx.to(s.device),
                            topk_window=topk_w,
                            q_quantile=q,
                            sent_top_m=sent_top_m,
                            sent_topS=topS,
                            sent_norm="bucket_sqrt",
                            gamma=gamma
                        )
                        if args.with_gcb:
                            s = gcb_apply_to_group(
                                s, cand_sent_idx.to(s.device), {k: v.to(s.device) for k, v in buckets.items()},
                                topk=args.gcb_topk, q_quantile=args.gcb_q,
                                top_m=args.gcb_top_m, sent_norm=args.gcb_norm,
                                topS=args.gcb_topS, gamma=args.gcb_gamma
                            )

                        # 统计
                        for j_in_group, global_j in enumerate(g):
                            gidx = gt_index_all[global_j]
                            row = s[j_in_group]
                            rank = int((row > row[gidx]).sum().item()) + 1
                            ranks.append(rank)
                            mrr_sum += 1.0 / rank
                            for K in topk_list:
                                recalls[K] += int(rank <= K)

                    num_q = sum(len(g) for g in groups)
                    res = {
                        "topk_window": topk_w, "q": q,
                        "sent_top_m": sent_top_m, "sent_topS": topS, "gamma": gamma,
                        "with_gcb": bool(args.with_gcb),
                        "pool_size": O,
                        "num_queries": num_q,
                        "R@1": recalls.get(1, 0)/num_q if num_q else 0.0,
                        "R@5": recalls.get(5, 0)/num_q if num_q else 0.0,
                        "R@10": recalls.get(10, 0)/num_q if num_q else 0.0,
                        "MRR": mrr_sum/num_q if num_q else 0.0,
                        "MeanRank": float(np.mean(ranks)) if ranks else 0.0,
                        "MedRank": float(np.median(ranks)) if ranks else 0.0,
                    }
                    rows_csv.append(res)
                    best.append((res["R@1"], res["R@5"], res["MRR"], res))

    # 写 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        for r in rows_csv: w.writerow(r)

    # 写 JSON（按 R@1、R@5、MRR 排序）
    rows_sorted = sorted(rows_csv, key=lambda r: (r["R@1"], r["R@5"], r["MRR"]), reverse=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"best_top10": rows_sorted[:10], "all": rows_sorted}, f, indent=2, ensure_ascii=False)

    log("\n=== WV back-write sweep: Top 10 by R@1 ===")
    for i, r in enumerate(rows_sorted[:10]):
        log(f"[#{i+1}] R1={r['R@1']:.4f} R5={r['R@5']:.4f} MRR={r['MRR']:.4f}  "
            f"| topk_w={r['topk_window']} q={r['q']} topS={r['sent_topS']} gamma={r['gamma']} "
            f"{' +GCB' if r['with_gcb'] else ''}")

    log(f"[DONE] WV sweep saved to:\n  - {csv_path}\n  - {json_path}")

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off","bf16","fp16","16-mixed"])
    p.add_argument("--use_ckpt_logit_scale", action="store_true")

    # WV 网格（补扫重点）
    p.add_argument("--wv_topk_window_list", type=str, default="64,128")
    p.add_argument("--wv_q_list", type=str, default="0.95,0.98")
    p.add_argument("--wv_topS_list", type=str, default="1,2")
    p.add_argument("--wv_gamma_list", type=str, default="0.3,0.5,0.7")
    p.add_argument("--wv_sent_top_m", type=int, default=3)   # 固定 3（可改）

    # 评测指标
    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--sample_groups", type=int, default=0, help="仅前 N 个句组做快速扫；0=全部")

    # 可选：QCCP（默认关）
    p.add_argument("--with_qccp", action="store_true")
    p.add_argument("--qccp_hops", type=int, default=1)
    p.add_argument("--qccp_alpha", type=float, default=0.6)
    p.add_argument("--qccp_topk", type=int, default=128)
    p.add_argument("--qccp_q", type=float, default=0.9)

    # 可选：WV 之后追加固定 GCB（默认开）
    p.add_argument("--with_gcb", action="store_true", default=True)
    p.add_argument("--gcb_topk", type=int, default=128)
    p.add_argument("--gcb_q", type=float, default=0.95)
    p.add_argument("--gcb_top_m", type=int, default=3)
    p.add_argument("--gcb_norm", type=str, default="bucket_sqrt")
    p.add_argument("--gcb_topS", type=int, default=3)
    p.add_argument("--gcb_gamma", type=float, default=0.7)

    return p.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    run_wv_sweep(args)
