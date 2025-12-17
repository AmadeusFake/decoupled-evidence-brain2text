#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrieval_window_vote.py — Minimal, scan-aligned evaluator (+ per-query/per-sentence dump)
Pipeline: Encode → Similarity → QCCP(hop) → WindowVote(soft-consensus, across=sum) → GCB(soft-consensus, post)
Default params unchanged. Added:
  --dump_per_query: write per_query.tsv (query-level ranks & hits)
  --dump_sentence_metrics: write sentence_metrics.tsv (sentence-level aggregated metrics)
  --seed: control shuffling and reproducibility of any stochastic ops (if any)
"""

import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------- 常量 -------------------------
TARGET_T = 360
AUDIO_D = 1024
EPS = 1e-8

# ------------------------- 工具 -------------------------
def log(msg: str):
    print(msg, flush=True)

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

def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D audio array, got {x.shape}")
    if x.shape[0] == AUDIO_D: return x
    if x.shape[1] == AUDIO_D: return x.T
    return x if abs(x.shape[0]-AUDIO_D) < abs(x.shape[1]-AUDIO_D) else x.T

def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(1) == T: return x
    return F.interpolate(x.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0)

def window_centers(rows: List[dict], idxs: List[int]) -> torch.Tensor:
    t = []
    for i in idxs:
        r = rows[i]
        s0 = float(r.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(r.get("local_window_offset_in_audio_s", s0))
        t.append(0.5*(s0+s1))
    return torch.tensor(t, dtype=torch.float32)

# ------------------------- 句子别名/索引 -------------------------
_CAND_SENT_KEYS = ["sentence_id","sentence_uid","utt_id","utterance_id","segment_id",
                   "original_sentence_id","sentence_path","sentence_audio_path","transcript_path"]

def _round3(x):
    try: return f"{float(x):.3f}"
    except Exception: return None

def sentence_aliases(row: dict):
    aliases = []
    for k in _CAND_SENT_KEYS:
        v = row.get(k)
        if v not in (None, ""): aliases.append((f"k:{k}", str(v)))
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
        aliases = sentence_aliases(r)
        if not aliases: cand_sent_idx.append(-1); continue
        canon = aliases[0]
        if canon not in canon2idx: canon2idx[canon] = len(canon2idx)
        sidx = canon2idx[canon]
        for a in aliases:
            if a not in alias2idx: alias2idx[a] = sidx
        cand_sent_idx.append(sidx)
    return canon2idx, alias2idx, cand_sent_idx

# ------------------------- subject 映射 -------------------------
def _normalize_subject_key(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x); m = re.search(r"(\d+)", s)
    if not m: return None
    return f"{int(m.group(1)):02d}"

def read_subject_mapping_from_records(run_dir: Path) -> Dict[str, int]:
    """
    尝试多种来源来读取 subject mapping：
      1) records/subject_mapping.json （原 MEG 用法）
      2) records/subject_mapping_snapshot.json （有些 run 会保存快照）
      3) records/subject_mapping_path.txt 指向的全局 JSON（例如 EEG union）
    """
    rec_dir = run_dir / "records"

    # 1) 首选 subject_mapping.json
    p = rec_dir / "subject_mapping.json"
    if not p.exists():
        # 2) 其次看看 snapshot
        snap = rec_dir / "subject_mapping_snapshot.json"
        if snap.exists():
            p = snap
        else:
            # 3) 最后使用 path.txt 里指定的全局路径
            txt = rec_dir / "subject_mapping_path.txt"
            if txt.exists():
                target = Path(txt.read_text(encoding="utf-8").strip())
                assert target.exists(), f"[SUBJECT] path in {txt} does not exist: {target}"
                p = target
            else:
                raise FileNotFoundError(
                    f"[SUBJECT] no subject mapping found under {rec_dir} "
                    f"(expected subject_mapping.json / subject_mapping_snapshot.json / subject_mapping_path.txt)"
                )

    obj = json.loads(p.read_text(encoding="utf-8"))
    # 兼容两种结构：{"mapping": {...}} 或直接 {...}
    raw = obj["mapping"] if "mapping" in obj and isinstance(obj["mapping"], dict) else obj

    out: Dict[str, int] = {}
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
        cfg = json.loads(rec.read_text(encoding="utf-8"))
        return cfg.get("model_cfg", {}) or cfg.get("enc_cfg", {}) or {}
    return {}

def choose_ckpt_path(args) -> Path:
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

from models.meg_encoder2 import UltimateMEGEncoder

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
        model_cfg["out_timesteps"] = None  # 评测端不做时间池化

    model = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    new_state = { (k[6:] if k.startswith("model.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:   log(f"[WARN] Missing keys: {len(missing)} (e.g., {missing[:8]})")
    if unexpected: log(f"[WARN] Unexpected keys: {len(unexpected)} (e.g., {unexpected[:8]})")
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

    y = model(meg_win=meg, sensor_locs=loc, subj_idx=sid)  # [B,1024,T?]
    if y.dim() != 3 or y.size(1) != AUDIO_D:
        raise RuntimeError(f"encoder must output [B,1024,T], got {tuple(y.shape)}")
    if y.size(2) != TARGET_T:
        y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)
    return y

# ------------------------- 候选池 -------------------------
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

# ------------------------- 相似度 -------------------------
def compute_logits_clip(queries: torch.Tensor, pool: torch.Tensor,
                        scale: Optional[float] = None) -> torch.Tensor:
    q = queries.to(torch.float32)
    A = pool.to(torch.float32)
    inv_norms = 1.0 / (A.norm(dim=(1, 2), p=2) + EPS)  # [O]
    logits = torch.einsum("bct,oct,o->bo", q, A, inv_norms)
    if scale is not None: logits = logits * float(scale)
    return logits.to(torch.float32)

# ------------------------- 句桶 -------------------------
def _precompute_sentence_buckets(cand_sent_idx_o: torch.Tensor) -> Dict[int, torch.Tensor]:
    buckets = {}
    uniq = torch.unique(cand_sent_idx_o)
    for s in uniq.tolist():
        if s < 0: continue
        buckets[int(s)] = torch.nonzero(cand_sent_idx_o == s, as_tuple=False).view(-1)
    return buckets

def _sent_len_norm_factor(n: int, mode: str = "none") -> float:
    if n <= 0 or mode == "none": return 1.0
    if mode in ("bucket_count","count","kept_count"): return 1.0 / float(n)
    if mode in ("bucket_sqrt","sqrt"):               return 1.0 / float(np.sqrt(max(1.0, n)))
    if mode == "log":                                return 1.0 / float(np.log2(n + 1.0))
    return 1.0

# ------------------------- GCB（soft-consensus, intra=sum） -------------------------
@torch.no_grad()
def gcb_apply_to_group(
    base_logits_bo: torch.Tensor,
    cand_sent_idx_o: torch.Tensor,
    buckets: Dict[int, torch.Tensor],
    *,
    topk: int = 128,
    q_quantile: float = 0.95,
    top_m: int = 3,               # mean@topm
    sent_norm: str = "bucket_sqrt",
    topS: int = 3,
    gamma: float = 0.7,
) -> torch.Tensor:
    B, O = base_logits_bo.shape
    if B == 0 or O == 0: return base_logits_bo
    K = min(int(topk), O)

    topk_scores, topk_idx = torch.topk(base_logits_bo, k=K, dim=1, largest=True, sorted=False)  # [B,K]
    thr_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)                   # [B,1]
    keep_mask = topk_scores >= thr_b1                                                            # [B,K]
    sids = cand_sent_idx_o[topk_idx]                                                             # [B,K]
    valid_mask = keep_mask & (sids >= 0)
    if not valid_mask.any(): return base_logits_bo

    sids_all = sids[valid_mask]                                                                  # [M]
    vals_all = (topk_scores - thr_b1).clamp_min_(0.0)[valid_mask].float()                        # [M]
    us, inv = torch.unique(sids_all, return_inverse=True)                                        # [S_sel]
    S_sel = us.numel()
    device = base_logits_bo.device

    # mean@topm per sentence
    sent_support = torch.empty(S_sel, dtype=vals_all.dtype, device=device)
    for k in range(S_sel):
        vk = vals_all[inv == k]
        if vk.numel() == 0:
            sent_support[k] = -1e9
            continue
        m_take = min(max(1, int(top_m)), int(vk.numel()))
        top_vals = torch.topk(vk, k=m_take, largest=True, sorted=False).values
        sent_support[k] = torch.mean(top_vals)

    # 句长归一
    bucket_sizes = [int(buckets.get(int(sid.item()), torch.empty(0, device=device)).numel()) for sid in us]
    norms = torch.tensor([_sent_len_norm_factor(n, sent_norm) for n in bucket_sizes],
                         dtype=sent_support.dtype, device=device)
    sent_support = sent_support * norms

    keepS = min(int(topS), int(sent_support.numel())) if topS > 0 else int(sent_support.numel())
    if keepS <= 0: return base_logits_bo
    topS_val, topS_idx = torch.topk(sent_support, k=keepS, largest=True, sorted=True)
    us_sel = us[topS_idx]

    # across=sum: 句内所有窗口统一 +sup（不改句内排序）
    boost_o = torch.zeros(O, dtype=base_logits_bo.dtype, device=device)
    for k in range(keepS):
        sid = int(us_sel[k].item()); sup = gamma * float(topS_val[k].item())
        if sup <= 0: continue
        bucket_idx = buckets.get(sid, None)
        if bucket_idx is None or bucket_idx.numel() == 0: continue
        boost_o[bucket_idx] += sup

    return base_logits_bo + boost_o.unsqueeze(0)

# ------------------------- QCCP（hop-only） -------------------------
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

    # 以时间排序的 hop 邻域
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

# ------------------------- Window-Vote（scan-aligned, across=sum） -------------------------
@torch.no_grad()
def window_vote_rerank(
    logits_bo: torch.Tensor,
    cand_sent_idx_o: torch.Tensor,
    *,
    topk_window: int = 128,
    q_quantile: float = 0.95,
    sent_top_m: int = 3,                 # mean@topm
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
        S_sel = us.numel()
        if S_sel == 0: continue

        # mean@topm + bucket_sqrt
        sent_scores = []
        for k_s in range(S_sel):
            vals = sco[inv == k_s]
            m_take = min(max(1, int(sent_top_m)), int(vals.numel()))
            top_vals = torch.topk(vals, k=m_take, largest=True, sorted=False).values
            agg_val = torch.mean(top_vals)
            s_id = int(us[k_s].item())
            n_windows = bucket_sizes.get(s_id, 1)
            agg_val = agg_val * _sent_len_norm_factor(n_windows, mode=sent_norm)
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

    return boost  # 直接返回 boost（已含 base + sup）

# ------------------------- 统计辅助（新增） -------------------------
def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)

def _write_tsv_header(fh, cols):
    fh.write("\t".join(cols) + "\n")

# ------------------------- 主流程 -------------------------
def evaluate(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if "16" in amp or "fp16" in amp else None)

    test_rows = read_jsonl(Path(args.test_manifest))
    log(f"[INFO] test rows = {len(test_rows):,}")

    A, pool_ids, candidate_rows = load_audio_pool_unique(test_rows, device=device, dtype=torch.float32)
    O = A.size(0)
    log(f"[INFO] candidate windows O={O}")

    canon2idx, alias2idx, cand_sent_idx = build_sentence_index_with_alias(candidate_rows)
    cand_sent_idx = torch.tensor(cand_sent_idx, dtype=torch.long, device=device)
    buckets = _precompute_sentence_buckets(cand_sent_idx)

    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in test_rows]

    run_dir = Path(args.run_dir)
    subj_map = read_subject_mapping_from_records(run_dir)
    log(f"[SUBJECT] loaded {len(subj_map)} subjects")

    ckpt_path = choose_ckpt_path(args)
    model, meta = load_model_from_ckpt(ckpt_path, run_dir, device=device)
    scale = meta.get("logit_scale_exp", None) if args.use_ckpt_logit_scale else None

    topk_list = [int(x) for x in args.topk.split(",")]
    recalls = {k: 0 for k in topk_list}
    mrr_sum = 0.0
    ranks: List[int] = []

    out_dir = run_dir / "results" / "retrieval_final_min"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_topk_k = max(0, int(args.save_topk)) if args.save_topk is not None else 0
    preds_topk_file = preds_tsv_file = None
    if save_topk_k > 0:
        preds_topk_path = out_dir / f"preds_topk{save_topk_k}.jsonl"
        preds_tsv_path = out_dir / f"preds_topk{save_topk_k}.tsv"
        preds_topk_file = open(preds_topk_path, "w", encoding="utf-8")
        preds_tsv_file = open(preds_tsv_path, "w", encoding="utf-8")
        preds_tsv_file.write("query_index\trank\tgt_cid\tpred_cids\n")

    # 新增：per-query 统计输出
    per_query_fh = None
    if args.dump_per_query:
        per_query_path = out_dir / "per_query.tsv"
        per_query_fh = open(per_query_path, "w", encoding="utf-8")
        _write_tsv_header(per_query_fh, [
            "query_index","sentence_key","sentence_idx","group_size",
            "rank","hit@1","hit@5","hit@10","mrr_contrib"
        ])

    # 分句组，应用 QCCP/GCB/WindowVote（都是组内机制）
    def sent_key_for_group(r: dict) -> Tuple[str, str]:
        als = sentence_aliases(r)
        return als[0] if als else ("unknown", content_id_of(r))

    sent2idx: Dict[Tuple[str,str], List[int]] = {}
    for i, r in enumerate(test_rows):
        k = sent_key_for_group(r)
        sent2idx.setdefault(k, []).append(i)
    groups = list(sent2idx.values())
    log(f"[INFO] Grouping by sentence: groups={len(groups)}, avg windows/sent={len(test_rows)/max(1,len(groups)):.2f}")

    # 为句级统计先缓存 per-sentence 的分子/分母
    sentence_stats = {}  # key -> dict(accumulators)

    def process_query_indices(q_indices: List[int], sent_key: Tuple[str,str]):
        nonlocal mrr_sum
        rows = [test_rows[i] for i in q_indices]
        if autocast_dtype is None:
            Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        else:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        logits = compute_logits_clip(Y, A, scale=scale)

        # QCCP（hop）— 由命令行决定是否启用（你要求“去掉 qccp”，运行时传 --no_qccp 即可）
        if not args.no_qccp:
            times = window_centers(test_rows, q_indices).to(device=logits.device)
            logits = qccp_rerank_group(
                logits, times_b=times,
                hops=args.qccp_hops, alpha=args.qccp_alpha,
                topk=args.qccp_topk, q_quantile=args.qccp_q
            )

        # Window-Vote（scan aligned）
        if not args.no_windowvote:
            logits = window_vote_rerank(
                logits, cand_sent_idx_o=cand_sent_idx,
                topk_window=args.topk_window, q_quantile=args.q_quantile,
                sent_top_m=args.sent_top_m, sent_topS=args.sent_topS,
                sent_norm=args.sent_norm, gamma=args.gamma
            )

        # GCB（post）
        if not args.no_gcb:
            logits = gcb_apply_to_group(
                logits, cand_sent_idx, buckets,
                topk=args.gcb_topk, q_quantile=args.gcb_q,
                top_m=args.gcb_top_m, sent_norm=args.gcb_norm,
                topS=args.gcb_topS, gamma=args.gcb_gamma
            )

        # 统计
        skey = f"{sent_key[0]}::{sent_key[1]}"
        acc = sentence_stats.setdefault(skey, {"n":0,"hit1":0,"hit5":0,"hit10":0,"mrr_sum":0.0,"ranks":[]})
        group_size = len(q_indices)
        for j_in_group, global_j in enumerate(q_indices):
            g = gt_index[global_j]
            s = logits[j_in_group]
            rank = int((s > s[g]).sum().item()) + 1
            ranks.append(rank)
            mrr_sum += 1.0 / rank
            for k in topk_list:
                recalls[k] += int(rank <= k)

            # 句级累加
            acc["n"] += 1
            acc["hit1"] += int(rank <= 1)
            acc["hit5"] += int(rank <= 5)
            acc["hit10"] += int(rank <= 10)
            acc["mrr_sum"] += (1.0 / rank)
            acc["ranks"].append(rank)

            if save_topk_k > 0:
                topk_scores, topk_idx_local = torch.topk(s, k=save_topk_k, largest=True, sorted=True)
                pred_cids = [pool_ids[int(t)] for t in topk_idx_local.tolist()]
                rec = {
                    "query_index": int(global_j),
                    "gt_rank": int(rank),
                    "gt_cid": pool_ids[g],
                    "pred_cids": pred_cids,
                    "pred_scores": [float(x) for x in topk_scores.tolist()],
                    "flags": {
                        "no_qccp": bool(args.no_qccp),
                        "no_gcb": bool(args.no_gcb),
                        "no_windowvote": bool(args.no_windowvote),
                    }
                }
                preds_topk_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                preds_tsv_file.write(f"{global_j}\t{rank}\t{pool_ids[g]}\t{','.join(pred_cids)}\n")

            if per_query_fh is not None:
                hit1 = int(rank <= 1); hit5 = int(rank <= 5); hit10 = int(rank <= 10)
                per_query_fh.write("\t".join(map(str, [
                    int(global_j), skey, -1, group_size,
                    int(rank), hit1, hit5, hit10, f"{1.0/float(rank):.8f}"
                ])) + "\n")

    # 遍历组
    for sent_key, q_indices in tqdm(list(sent2idx.items()), desc="Evaluate (Grouped)"):
        process_query_indices(q_indices, sent_key)

    num_queries = len(test_rows)
    metrics = {
        "num_queries": num_queries,
        "pool_size": O,
        "recall_at": {str(k): recalls[k] / num_queries for k in topk_list},
        "mrr": mrr_sum / num_queries,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "topk_list": topk_list,
        "flags": {
            "qccp": not args.no_qccp,
            "gcb": not args.no_gcb,
            "windowvote": not args.no_windowvote,
            "use_ckpt_logit_scale": bool(args.use_ckpt_logit_scale),
        }
    }
    out_dir = run_dir / "results" / "retrieval_final_min"
    out_json = Path(args.save_json) if args.save_json else (out_dir / "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    out_ranks = Path(args.save_ranks) if args.save_ranks else (out_dir / "ranks.txt")
    with open(out_ranks, "w", encoding="utf-8") as f:
        for r in ranks: f.write(str(int(r)) + "\n")

    log("==== Retrieval Results ====")
    ra = metrics["recall_at"]
    log(f"[RESULT] R@1={ra.get('1',0.0):.4f}  R@5={ra.get('5',0.0):.4f}  R@10={ra.get('10',0.0):.4f}  "
        f"MRR={metrics['mrr']:.4f}  MeanRank={metrics['mean_rank']:.1f}  MedRank={metrics['median_rank']:.1f}")
    log(json.dumps(metrics["flags"], indent=2, ensure_ascii=False))
    log(f"[INFO] Metrics saved to: {out_json.as_posix()}")
    log(f"[INFO] Ranks saved to  : {out_ranks.as_posix()}")

    if save_topk_k > 0:
        preds_topk_file.close()
        preds_tsv_file.close()
    if per_query_fh is not None:
        per_query_fh.close()

    # 新增：句级统计导出（Nature 统计将基于它做 paired bootstrap）
    if args.dump_sentence_metrics:
        sent_path = out_dir / "sentence_metrics.tsv"
        with open(sent_path, "w", encoding="utf-8") as fh:
            _write_tsv_header(fh, ["sentence_key","n_windows","R1","R5","R10","MRR","MedR"])
            for skey, acc in sentence_stats.items():
                n = acc["n"]
                r1 = _safe_div(acc["hit1"], n)
                r5 = _safe_div(acc["hit5"], n)
                r10 = _safe_div(acc["hit10"], n)
                mrr = _safe_div(acc["mrr_sum"], n)
                medr = float(np.median(acc["ranks"])) if acc["ranks"] else float("nan")
                fh.write("\t".join([skey, str(n),
                                    f"{r1:.8f}", f"{r5:.8f}", f"{r10:.8f}", f"{mrr:.8f}", f"{medr:.6f}"]) + "\n")
        log(f"[INFO] Sentence metrics saved to: {sent_path.as_posix()}")

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

    # QCCP（hop-only）
    p.add_argument("--no_qccp", action="store_true")
    p.add_argument("--qccp_hops", type=int, default=1)
    p.add_argument("--qccp_alpha", type=float, default=0.6)
    p.add_argument("--qccp_topk", type=int, default=128)
    p.add_argument("--qccp_q", type=float, default=0.9)

    # Window-Vote（scan 最优）
    p.add_argument("--no_windowvote", action="store_true")
    p.add_argument("--topk_window", type=int, default=128)
    p.add_argument("--q_quantile", type=float, default=0.95)
    p.add_argument("--sent_top_m", type=int, default=3)
    p.add_argument("--sent_topS", type=int, default=3)
    p.add_argument("--sent_norm", type=str, default="bucket_sqrt", choices=["bucket_sqrt"])
    p.add_argument("--gamma", type=float, default=0.7)

    # GCB（post, soft-consensus）
    p.add_argument("--no_gcb", action="store_true")
    p.add_argument("--gcb_topk", type=int, default=128)
    p.add_argument("--gcb_q", type=float, default=0.95)
    p.add_argument("--gcb_top_m", type=int, default=3)
    p.add_argument("--gcb_norm", type=str, default="bucket_sqrt", choices=["bucket_sqrt"])
    p.add_argument("--gcb_topS", type=int, default=3)
    p.add_argument("--gcb_gamma", type=float, default=0.7)

    # 评分缩放 + 输出
    p.add_argument("--use_ckpt_logit_scale", action="store_true")
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--save_ranks", type=str, default="")
    p.add_argument("--save_topk", type=int, default=0)

    # 新增输出与可重复性
    p.add_argument("--dump_per_query", action="store_true")
    p.add_argument("--dump_sentence_metrics", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    evaluate(args)
