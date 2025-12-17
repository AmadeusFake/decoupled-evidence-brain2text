#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fair(er) one-shot evaluation report for MEG↔Audio retrieval.

改动要点（相对你的原版）：
- 兼容池大小的指标：
  * Success@p%（p∈{1,5,10}%）：前百分位命中率
  * MPR（Mean Percentile Rank）与 1-MPR
  * Calibrated R@k：去机会水平后的 R@k（chance=0，完美=1）
  * Cross-only 的 R@k/MRR（先剔除同句子候选）
  * Cross pairwise AUC：P(score_gt > 随机 cross 负样本)
- 同时保留原始 overall/same-only 指标与图表
- 自动从 run_dir/records/config.json 解析 manifest（如未传参）
- 追加两张图：
  * success_at_pct_curve.png（overall/same/cross）
  * calibrated_r_at_k_bar.png（overall/same/cross）
"""

import argparse, json, inspect, re, math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, DefaultDict
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------ Model import ------------
from models.meg_encoder import UltimateMEGEncoder

# ------------ Constants ------------
TARGET_T = 360
AUDIO_D  = 1024

# ------------ Utils ------------
def log(x): print(x, flush=True)

def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def content_id_of(r: dict) -> str:
    if r.get("content_id"): return r["content_id"]
    audio_path = r.get("original_audio_path", r.get("audio_feature_path", ""))
    onset = float(r.get("local_window_onset_in_audio_s", r.get("onset_in_audio_s", 0.0)))
    offset= float(r.get("local_window_offset_in_audio_s", r.get("offset_in_audio_s", 0.0)))
    return f"{Path(audio_path).stem}::{onset:.3f}-{offset:.3f}"

def _maybe_float(r: dict, k: str, default: Optional[float] = None) -> Optional[float]:
    v = r.get(k, default)
    try: return None if v is None else float(v)
    except Exception: return default

def sentence_key_of(r: dict) -> str:
    audio_path = r.get("original_audio_path", r.get("audio_feature_path", ""))
    stem = Path(audio_path).stem
    s_on = r.get("sentence_onset_in_audio_s", r.get("sent_onset_in_audio_s"))
    s_off= r.get("sentence_offset_in_audio_s", r.get("sent_offset_in_audio_s"))
    if s_on is not None and s_off is not None:
        return f"{stem}::SENT[{float(s_on):.3f}-{float(s_off):.3f}]"
    if r.get("sentence_idx") is not None:
        return f"{stem}::IDX[{int(r['sentence_idx'])}]"
    if r.get("utt_id") is not None:
        return f"{stem}::UTT[{r['utt_id']}]"
    return f"{stem}::WHOLE"

def window_relpos_of(r: dict, sent_bounds: Dict[str, Tuple[float, float]]) -> float:
    s_key = sentence_key_of(r)
    s_on, s_off = sent_bounds.get(s_key, (None, None))
    if s_on is None or s_off is None:
        s_on = _maybe_float(r, "sentence_onset_in_audio_s", _maybe_float(r, "sent_onset_in_audio_s", None))
        s_off= _maybe_float(r, "sentence_offset_in_audio_s", _maybe_float(r, "sent_offset_in_audio_s", None))
    if s_on is None or s_off is None or s_off <= s_on: return 0.5
    w_on = _maybe_float(r, "local_window_onset_in_audio_s", _maybe_float(r, "onset_in_audio_s", 0.0)) or 0.0
    w_off= _maybe_float(r, "local_window_offset_in_audio_s", _maybe_float(r, "offset_in_audio_s", 0.0)) or 0.0
    w_c = 0.5 * (w_on + w_off)
    return float(min(1.0, max(0.0, (w_c - s_on) / max(s_off - s_on, 1e-3))))

def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2: raise ValueError(f"Expected 2D audio, got {x.shape}")
    return x if x.shape[0] == AUDIO_D else x.T

def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2: raise ValueError(f"Expected 2D MEG, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(-1) == T: return x
    twoD = (x.dim()==2)
    if twoD: x = x.unsqueeze(0)
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    if twoD: x = x.squeeze(0)
    return x

def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        with open(rec, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        # 兼容不同保存格式
        if "cfg" in cfg and isinstance(cfg["cfg"], dict):
            return cfg["cfg"]
        return cfg
    return {}

def auto_find_manifests(run_dir: Path, given_train: str, given_test: str) -> Tuple[str, str]:
    if given_train and given_test:
        return given_train, given_test
    cfg = load_cfg_from_records(run_dir)
    # 多层兜底
    for root in (cfg, cfg.get("args", {}), cfg.get("Args", {})):
        if isinstance(root, dict):
            tr = root.get("train_manifest") or root.get("train_path")
            te = root.get("test_manifest")  or root.get("test_path")
            if tr and te:
                return str(tr), str(te)
    raise RuntimeError("无法自动解析 train/test manifest，请手工传入 --train_manifest 与 --test_manifest")

def choose_ckpt_path(args) -> Path:
    if args.use_best_ckpt:
        best_txt = Path(args.run_dir)/"records"/"best_checkpoint.txt"
        assert best_txt.exists(), f"best_checkpoint.txt not found: {best_txt}"
        ckpt_str = best_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        ckpt = Path(ckpt_str) if ckpt_str.startswith("/") else (Path(args.run_dir)/ckpt_str)
        assert ckpt.exists(), f"Best ckpt not found: {ckpt}"
        log(f"Using BEST checkpoint: {ckpt}")
        return ckpt
    ckpt = Path(args.ckpt_path); assert ckpt.exists(), f"ckpt not found: {ckpt}"
    return ckpt

def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str) -> UltimateMEGEncoder:
    # 从 records 拿 enc_cfg / model_cfg
    model_cfg = load_cfg_from_records(run_dir).get("enc_cfg") \
                or load_cfg_from_records(run_dir).get("model_cfg", {})
    try: ckpt = torch.load(ckpt_path, map_location="cpu")
    except TypeError: ckpt = torch.load(ckpt_path, map_location="cpu")
    if not model_cfg:
        hps = ckpt.get("hyper_parameters", {})
        model_cfg = hps.get("model_cfg", hps.get("enc_cfg", {}))
    assert model_cfg, "Model config not found in ckpt or records/config.json"
    # 强制评估时不下采样时间步（兼容 sentence 模式）
    if "out_timesteps" in inspect.signature(UltimateMEGEncoder).parameters:
        model_cfg["out_timesteps"] = None
    model = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    new_state = {k.replace("model.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing: log(f"[WARN] missing keys: {len(missing)}")
    if unexpected: log(f"[WARN] unexpected keys: {len(unexpected)}")
    return model.eval().to(device)

def load_subject_map(run_dir: Path, train_rows: List[dict]) -> Dict[str, int]:
    snap = run_dir / "records" / "subject_mapping_snapshot.json"
    if snap.exists():
        with open(snap, "r", encoding="utf-8") as f:
            data = json.load(f)
        mp = data.get("mapping", {})
        if mp:
            return {str(k): int(v) for k, v in mp.items()}
    ids = sorted({str(r["subject_id"]) for r in train_rows if "subject_id" in r})
    return {sid: i for i, sid in enumerate(ids)}

def filter_and_annotate_rows(rows: List[dict], sub_map: Dict[str,int], strict_subjects: bool) -> List[dict]:
    out = []
    for r in rows:
        sid = str(r.get("subject_id"))
        if strict_subjects and sid not in sub_map: continue
        if not all(Path(r.get(p,"")).exists() for p in ["sensor_coordinates_path","meg_win_path","audio_feature_path"]):
            continue
        rr = dict(r)
        rr["_subject_idx"] = sub_map.get(sid, 0)
        rr["_sent_key"] = sentence_key_of(r)
        out.append(rr)
    return out

def build_sentence_bounds(rows: List[dict]) -> Dict[str, Tuple[float,float]]:
    bounds: Dict[str, Tuple[float,float]] = {}
    tmp: DefaultDict[str, List[Tuple[float,float]]] = defaultdict(list)
    for r in rows:
        s_key = sentence_key_of(r)
        s_on = r.get("sentence_onset_in_audio_s", r.get("sent_onset_in_audio_s"))
        s_off= r.get("sentence_offset_in_audio_s", r.get("sent_offset_in_audio_s"))
        if s_on is not None and s_off is not None:
            tmp[s_key].append((float(s_on), float(s_off)))
        else:
            w_on = _maybe_float(r,"local_window_onset_in_audio_s", _maybe_float(r,"onset_in_audio_s",None))
            w_off= _maybe_float(r,"local_window_offset_in_audio_s", _maybe_float(r,"offset_in_audio_s",None))
            if w_on is not None and w_off is not None:
                tmp[s_key].append((float(w_on), float(w_off)))
    for k, spans in tmp.items():
        ons = [a for a,_ in spans]; offs=[b for _,b in spans]
        bounds[k] = (min(ons), max(offs))
    return bounds

@torch.no_grad()
def encode_meg(model: UltimateMEGEncoder, arrs, locs, sidx, device, autocast_dtype):
    if len(arrs)==0:
        return torch.empty(0, getattr(model,"out_channels",0), TARGET_T, device=device)
    megs = torch.stack([torch.from_numpy(ensure_meg_CxT(x)) for x in arrs]).to(device)
    locs_t = torch.stack([torch.from_numpy(l) for l in locs]).to(device)
    sidx_t = torch.tensor(sidx, dtype=torch.long, device=device)
    sig = inspect.signature(model.forward); kwargs={}
    if "sensor_locs" in sig.parameters: kwargs["sensor_locs"]=locs_t
    if "subj_idx"    in sig.parameters: kwargs["subj_idx"]=sidx_t
    with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
        y = model(meg_win=megs, **kwargs) if "meg_win" in sig.parameters else model(megs, **kwargs)
    if y.dim()==2: y = y.unsqueeze(-1).repeat(1,1,TARGET_T)
    return maybe_interp_1DT(y, TARGET_T)

def compute_base_logits(queries: torch.Tensor, pool: torch.Tensor, sim: str, tau: float) -> torch.Tensor:
    q = queries.float(); p = pool.float()
    if sim in ("clip","cosine"):
        qn = F.normalize(q.flatten(1), p=2, dim=1)
        pn = F.normalize(p.flatten(1), p=2, dim=1)
        logits = qn @ pn.t()
    elif sim=="dot":
        logits = q.flatten(1) @ p.flatten(1).t()
    else:
        raise ValueError(sim)
    if tau>0: logits = logits / tau
    return logits

def load_audio_pool_unique(test_rows, sent_bounds, device, dtype=torch.float32):
    uniq_paths: Dict[str,str] = {}
    pool_sents: Dict[str,str] = {}
    pool_rpos_vals: Dict[str,float] = {}
    for r in test_rows:
        cid = content_id_of(r)
        if cid in uniq_paths: continue
        uniq_paths[cid] = r["audio_feature_path"]
        s_key = sentence_key_of(r)
        pool_sents[cid] = s_key
        pool_rpos_vals[cid] = window_relpos_of(r, sent_bounds)
    pool_ids = list(uniq_paths.keys())
    feats=[]
    for cid in tqdm(pool_ids, desc="Loading & Aligning Audio Pool"):
        arr = ensure_audio_DxT(np.load(uniq_paths[cid]).astype(np.float32))
        t = torch.from_numpy(arr); t = maybe_interp_1DT(t, TARGET_T)
        feats.append(t)
    pool = torch.stack(feats,0).to(device=device, dtype=dtype)  # [O,C,T]
    sent_groups: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
    for i,cid in enumerate(pool_ids):
        sent_groups[pool_sents[cid]].append(pool[i].mean(-1))
    sent_audio_cache = {k: torch.stack(v,0).mean(0) for k,v in sent_groups.items()}
    pool_sents_list = [pool_sents[cid] for cid in pool_ids]
    pool_rpos = torch.tensor([pool_rpos_vals[cid] for cid in pool_ids], device=device, dtype=torch.float32)
    return pool, pool_ids, pool_sents_list, pool_rpos, sent_audio_cache

def build_by_sentence(pool_sents: List[str], device: str):
    O=len(pool_sents); all_idx = torch.arange(O, device=device, dtype=torch.long)
    tmp: DefaultDict[str, List[int]] = defaultdict(list)
    for i,sk in enumerate(pool_sents): tmp[sk].append(i)
    by_sent = {sk: torch.tensor(v, device=device, dtype=torch.long) for sk,v in tmp.items()}
    not_by = {}
    for sk, idx in by_sent.items():
        mask = torch.ones(O, dtype=torch.bool, device=device); mask[idx]=False
        not_by[sk] = all_idx[mask]
    return by_sent, not_by

# ------------ Text features ------------
STOPWORDS = {
    "the","a","an","of","and","to","in","on","at","for","from","as","with","by","that","this","is","are",
    "was","were","be","been","it","its","into","over","under","but","or","if","so","not","no","than","then",
    "she","he","they","i","you","we","his","her","their","our","your","me","him","them","us"
}
PRONOUNS = {"i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","her","our",
            "their","mine","yours","ours","theirs"}
TOKEN_RE = re.compile(r"[A-Za-z']+")
def tokenize_lower(s: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(s or "")]

def text_features(s: str) -> Dict[str, float]:
    if not s:
        return {"n_char":0,"n_tok":0,"avg_tok_len":0.0,"ttr":0.0,"stop_ratio":0.0,"long_ratio":0.0,"has_pron":0.0}
    toks = tokenize_lower(s); n_tok=len(toks)
    types=set(toks)
    stop_ratio = (sum(1 for t in toks if t in STOPWORDS)/n_tok) if n_tok>0 else 0.0
    long_ratio = (sum(1 for t in toks if len(t)>=6)/n_tok) if n_tok>0 else 0.0
    return {
        "n_char": len(s),
        "n_tok": n_tok,
        "avg_tok_len": (sum(len(t) for t in toks)/n_tok) if n_tok>0 else 0.0,
        "ttr": (len(types)/n_tok) if n_tok>0 else 0.0,
        "stop_ratio": stop_ratio,
        "long_ratio": long_ratio,
        "has_pron": 1.0 if any(t in PRONOUNS for t in toks) else 0.0
    }

LEN_BINS = [(0,5),(6,10),(11,15),(16,25),(26,10**9)]
def bucket_len(n_tok: int) -> str:
    for a,b in LEN_BINS:
        if a<=n_tok<=b: return f"{a}-{b if b<10**9 else 'INF'}"
    return "UNK"
def bucket_ratio(x: float, edges=(0.0,0.2,0.4,0.6,0.8,1.0)) -> str:
    for i in range(len(edges)-1):
        if edges[i] <= x < edges[i+1]: return f"[{edges[i]:.1f},{edges[i+1]:.1f})"
    return f"[{edges[-2]:.1f},{edges[-1]:.1f}]"

# ------------ Simple PCA (numpy) ------------
def pca_2d(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(0, keepdims=True)
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T

# ------------ Plot helpers ------------
def figsave(path: Path, dpi=180):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def bar(ax, xs, ys, title, xlabel, ylabel):
    ax.bar(xs, ys)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

def stacked(ax, xs, parts: Dict[str, List[float]], title, xlabel, ylabel):
    bottom = np.zeros(len(xs))
    for name, vals in parts.items():
        ax.bar(xs, vals, bottom=bottom, label=name)
        bottom += np.array(vals)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

def heatmap(ax, M: np.ndarray, xt, yt, title):
    im = ax.imshow(M, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt, fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def safe_div(a,b): return float(a)/float(b) if b and b>0 else 0.0

# ------------ Fairness helpers ------------
def summarize_rank_pool(ranks, pools, topk_vals, pcts=(0.01, 0.05, 0.10)):
    import numpy as np
    ranks = np.asarray(ranks, dtype=np.float64)
    pools = np.asarray(pools, dtype=np.float64)
    eps = 1e-9
    if len(ranks)==0:  # 空集时的防守
        return {
            "MPR": None, "one_minus_MPR": None,
            "success_at_pct": {f"{int(p*100)}%": None for p in pcts},
            "calibrated_R@k": {str(k): None for k in topk_vals},
        }
    # Mean Percentile Rank（越低越好）以及 1-MPR（越高越好）
    mpr = np.mean((ranks - 1.0) / np.maximum(1.0, pools - 1.0))
    one_minus_mpr = 1.0 - mpr
    # Success@p%（归一化 top-k）
    succ_pct = {}
    for p in pcts:
        k_dyn = np.ceil(p * pools)
        succ_pct[f"{int(p*100)}%"] = float(np.mean(ranks <= k_dyn))
    # Calibrated R@k：去 chance（k/O）并归一化
    cal_r_at_k = {}
    for k in topk_vals:
        vals = []
        for r, o in zip(ranks, pools):
            o = max(1.0, o)
            chance = min(1.0, k / o)
            if chance >= 1.0 - eps:
                continue
            hit = 1.0 if r <= k else 0.0
            vals.append((hit - chance) / (1.0 - chance))
        cal_r_at_k[str(k)] = float(np.mean(vals)) if vals else None
    return {
        "MPR": float(mpr),
        "one_minus_MPR": float(one_minus_mpr),
        "success_at_pct": succ_pct,
        "calibrated_R@k": cal_r_at_k,
    }

# ------------ CLI ------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", type=str, default="")
    p.add_argument("--train_manifest", type=str, default="")
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off","bf16","fp16"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--k_max", type=int, default=50, help="max k for CMC curve")
    p.add_argument("--strict_subjects", action="store_true")
    p.add_argument("--sim", type=str, default="clip", choices=["clip","cosine","dot"])
    p.add_argument("--tau", type=float, default=0.0, help="fixed temperature; 0 disables scaling")
    p.add_argument("--context_mode", type=str, default="none", choices=["none","window","sentence"])
    p.add_argument("--viz_topN", type=int, default=30, help="Top-N sentences for mini confusion")
    p.add_argument("--hard_topM", type=int, default=20, help="Top-M hardest sentences figure")
    return p.parse_args()

# ------------ Main ------------
def main():
    args = parse_args()
    device = args.device
    autocast_dtype = torch.bfloat16 if args.amp=="bf16" else (torch.float16 if "16" in args.amp or args.amp=="fp16" else None)

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "results_fancy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 自动解析 manifest（如未显式给出）
    train_manifest, test_manifest = auto_find_manifests(run_dir, args.train_manifest, args.test_manifest)
    log(f"[data] train={train_manifest}")
    log(f"[data] test ={test_manifest}")

    test_rows_raw  = read_jsonl(Path(test_manifest))
    train_rows_raw = read_jsonl(Path(train_manifest))
    sub_map = load_subject_map(run_dir, train_rows_raw)
    sent_bounds = build_sentence_bounds(test_rows_raw)
    test_rows = filter_and_annotate_rows(test_rows_raw, sub_map, args.strict_subjects)
    assert test_rows, "No valid test rows"

    # text features
    texts = [r.get("global_segment_text","") for r in test_rows]
    tfeats = [text_features(t) for t in texts]
    len_bins  = [bucket_len(int(tf["n_tok"])) for tf in tfeats]
    long_bins = [bucket_ratio(float(tf["long_ratio"])) for tf in tfeats]
    stop_bins = [bucket_ratio(float(tf["stop_ratio"])) for tf in tfeats]
    pron_bins = ["1" if tf["has_pron"]>0.5 else "0" for tf in tfeats]

    # pool
    pool, pool_ids, pool_sents, pool_rpos, _ = load_audio_pool_unique(test_rows, sent_bounds, device, dtype=torch.float32)
    O=pool.size(0)
    cid2idx = {cid:i for i,cid in enumerate(pool_ids)}
    gt_indices = [cid2idx[content_id_of(r)] for r in test_rows]
    by_sent, not_by = build_by_sentence(pool_sents, device=device)

    # model
    ckpt = choose_ckpt_path(args)
    model = load_model_from_ckpt(ckpt, run_dir, device)
    tau = float(args.tau) if args.tau>0 else 0.0
    if tau>0 and args.sim=="clip": log(f"[Info] Fixed tau={tau:.4f}")

    # accumulators
    B = args.batch_size
    topk_vals = [int(x) for x in args.topk.split(",")]
    ranks_all: List[int] = []
    recalls_overall = {k:0 for k in topk_vals}
    recalls_same    = {k:0 for k in topk_vals}
    recalls_cross   = {k:0 for k in topk_vals}
    mrr_sum = 0.0
    guarded1 = 0

    # pool-size aware
    ranks_overall,  pools_overall  = [], []
    ranks_same,     pools_same     = [], []
    ranks_cross,    pools_cross    = [], []
    mrr_same_sum,   mrr_cross_sum  = 0.0, 0.0
    cnt_same,       cnt_cross      = 0, 0
    auc_cross_vals: List[float]    = []

    err_within=0; err_cross=0
    abs_dr_bins = [i/10.0 for i in range(11)]
    def b_absdr(v):
        for i in range(10):
            if abs_dr_bins[i] <= v < abs_dr_bins[i+1]:
                return f"[{abs_dr_bins[i]:.1f},{abs_dr_bins[i+1]:.1f})"
        return f"[{abs_dr_bins[-2]:.1f},{abs_dr_bins[-1]:.1f}]"
    hist_absdr = {b_absdr(i/10.0):0 for i in range(10)}

    # text-bucket accumulators
    def Z(): return {"cnt":0,"r1":0,"r5":0,"r10":0,"mrr":0.0,"e_within":0,"e_cross":0}
    acc_len  = defaultdict(Z); acc_long = defaultdict(Z)
    acc_stop = defaultdict(Z); acc_pron = defaultdict(Z)

    # sentence-level stats for figures
    sent_acc = defaultdict(lambda: {"cnt":0,"r10":0,"mrr":0.0})
    # confusion (cross top-1 flow)
    conf_counts = Counter()
    def short_name(sk, L=24):
        n = sk.split("::")[-1]
        return n if len(n)<=L else n[:L-3]+"..."

    # CMC up to k_max
    k_max = min(max(topk_vals+[args.k_max]), O)
    cmc_counts = np.zeros(k_max, dtype=np.int64)  # count of ranks ≤ k (1-indexed)

    # loop
    for i0 in tqdm(range(0, len(test_rows), B), desc=f"Eval fancy ({args.context_mode})"):
        rows = test_rows[i0:i0+B]
        arrs = [np.load(r["meg_win_path"]) for r in rows]
        locs = [np.load(r["sensor_coordinates_path"]) for r in rows]
        sidx = [int(r["_subject_idx"]) for r in rows]
        q = encode_meg(model, arrs, locs, sidx, device, autocast_dtype)

        base = compute_base_logits(q, pool, args.sim, tau)  # [b,O]

        for j, r in enumerate(rows):
            scores = base[j]; gt = gt_indices[i0+j]
            s_key = r["_sent_key"]
            s_same = by_sent.get(s_key, torch.empty(0, dtype=torch.long, device=device))
            s_cross= not_by.get(s_key, torch.arange(O, device=device, dtype=torch.long))

            gt_score = scores[gt]
            rank = int((scores > gt_score).sum().item()) + 1
            ranks_all.append(rank); mrr_sum += 1.0/rank
            for k in topk_vals:
                if rank<=k: recalls_overall[k]+=1
            # CMC
            if rank<=k_max: cmc_counts[rank-1:] += 1

            # 记录 overall 的 (rank, pool)
            ranks_overall.append(rank); pools_overall.append(int(O))

            # same-sentence-only rank
            if s_same.numel()>0:
                s_scores = scores.index_select(0, s_same)
                rank_same = int((s_scores > gt_score).sum().item()) + 1
                for k in topk_vals:
                    if rank_same<=k: recalls_same[k]+=1
                ranks_same.append(rank_same); pools_same.append(int(s_same.numel()))
                mrr_same_sum += 1.0/rank_same; cnt_same += 1

            # cross-only rank 以及 cross AUC
            if s_cross.numel()>0:
                c_scores   = scores.index_select(0, s_cross)
                rank_cross = int((c_scores > gt_score).sum().item()) + 1
                for k in topk_vals:
                    if rank_cross<=k: recalls_cross[k]+=1
                ranks_cross.append(rank_cross); pools_cross.append(int(s_cross.numel()))
                mrr_cross_sum += 1.0/rank_cross; cnt_cross += 1
                auc_cross_vals.append(float((gt_score > c_scores).float().mean().item()))

            # guarded@1
            best_cross = scores.index_select(0, s_cross).max().item() if s_cross.numel()>0 else -1e9
            guarded1 += int(gt_score >= best_cross)

            # error kind & stats
            top_pred = int(torch.argmax(scores).item())
            err_type = "none"
            if top_pred != gt:
                # within vs cross
                same_best, same_best_idx = -1e9, -1
                if s_same.numel()>0:
                    same_no_gt = s_same[s_same != gt]
                    if same_no_gt.numel()>0:
                        s_vals = scores.index_select(0, same_no_gt)
                        same_best = s_vals.max().item()
                        same_best_idx = int(same_no_gt[torch.argmax(s_vals).item()].item())
                cross_best, cross_best_idx = -1e9, -1
                if s_cross.numel()>0:
                    cross_vals = scores.index_select(0, s_cross)
                    cross_best = cross_vals.max().item()
                    cross_best_idx = int(s_cross[torch.argmax(cross_vals).item()].item())

                if cross_best >= same_best:
                    err_type = "cross"
                    err_cross += 1
                    conf_counts[(s_key, pool_sents[top_pred])] += 1
                else:
                    err_type = "within"
                    err_within += 1
                    rq = window_relpos_of(r, sent_bounds)
                    rw = float(pool_rpos[same_best_idx].item()) if 0<=same_best_idx<O else 0.5
                    hist_absdr[b_absdr(abs(rq-rw))] += 1

            # sentence-level aggregation
            sent_acc[s_key]["cnt"] += 1
            if rank<=10: sent_acc[s_key]["r10"] += 1
            sent_acc[s_key]["mrr"] += 1.0/rank

            # text buckets
            tf = tfeats[i0+j]; lb=len_bins[i0+j]; lgb=long_bins[i0+j]; stb=stop_bins[i0+j]; prb=pron_bins[i0+j]
            for acc in (acc_len[lb], acc_long[lgb], acc_stop[stb], acc_pron[prb]):
                acc["cnt"] += 1
                acc["r1"]  += int(rank<=1); acc["r5"] += int(rank<=5); acc["r10"] += int(rank<=10)
                acc["mrr"] += 1.0/rank
                if err_type=="cross":  acc["e_cross"]  += 1
                if err_type=="within": acc["e_within"] += 1

    N = len(test_rows)
    summary_same = {
        "recall_at": {str(k): (recalls_same[k]/max(1,cnt_same)) for k in topk_vals},
        "mrr": (mrr_same_sum/max(1,cnt_same)) if cnt_same>0 else None,
        "queries_count": cnt_same,
    }
    summary_cross = {
        "recall_at": {str(k): (recalls_cross[k]/max(1,cnt_cross)) for k in topk_vals},
        "mrr": (mrr_cross_sum/max(1,cnt_cross)) if cnt_cross>0 else None,
        "queries_count": cnt_cross,
    }

    # size-normalized summaries
    norm_overall = summarize_rank_pool(ranks_overall, pools_overall, topk_vals)
    norm_same    = summarize_rank_pool(ranks_same,    pools_same,    topk_vals) if cnt_same  else {}
    norm_cross   = summarize_rank_pool(ranks_cross,   pools_cross,   topk_vals) if cnt_cross else {}
    mean_auc_cross = float(np.mean(auc_cross_vals)) if auc_cross_vals else None

    # overall naive recalls
    recalls_overall = {str(k): recalls_overall[k]/N for k in topk_vals}

    summary = {
        "num_queries": N,
        "pool_size": int(O),
        "similarity": args.sim,
        "recall_at_overall": recalls_overall,
        "same_only": summary_same,
        "cross_only": summary_cross,
        "guarded_top1": guarded1/N,
        "mrr": mrr_sum/N,
        "mean_rank": float(np.mean(ranks_all)) if ranks_all else None,
        "median_rank": float(np.median(ranks_all)) if ranks_all else None,
        "error_breakdown": {
            "errors_total": int(N - sum(1 for r in ranks_all if r==1)),
            "within": int(err_within),
            "cross": int(err_cross),
            "within_ratio": err_within/max(1, (N - sum(1 for r in ranks_all if r==1))),
            "cross_ratio":  err_cross /max(1, (N - sum(1 for r in ranks_all if r==1))),
        },
        "within_sentence_abs_delta_r_hist": hist_absdr,
        "text_buckets": {
            "len":  {k: _pack_bucket(v) for k,v in acc_len.items()},
            "long": {k: _pack_bucket(v) for k,v in acc_long.items()},
            "stop": {k: _pack_bucket(v) for k,v in acc_stop.items()},
            "pron": {k: _pack_bucket(v) for k,v in acc_pron.items()},
        },
        "size_normalized": {
            "overall": norm_overall,
            "same_only": norm_same,
            "cross_only": norm_cross,
        },
        "cross_auc_pairwise": mean_auc_cross,
        "context_mode": args.context_mode,
    }

    # save summary json
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --------- Figures ---------
    # CMC
    ks = np.arange(1, k_max+1)
    cmc = (np.asarray((np.arange(k_max)>=0), dtype=float)*0)  # dummy init
    cmc = cmc  # no-op
    cmc = None
    # 使用已累计的 cmc_counts
    ks = np.arange(1, k_max+1)
    cmc = (np.asarray(cmc_counts, dtype=np.float64) / max(1,N))
    plt.figure(figsize=(6,4))
    plt.plot(ks, cmc, marker="o", linewidth=2)
    plt.title("CMC (Recall@k)"); plt.xlabel("k"); plt.ylabel("Recall")
    plt.grid(True, alpha=0.3); figsave(out_dir/"cmc_curve.png")

    # rank histogram (log bins)
    ranks = np.array(ranks_all) if ranks_all else np.array([1])
    bins = np.unique(np.logspace(0, math.log10(max(ranks.max(),10)), num=20, dtype=int))
    plt.figure(figsize=(6,4))
    plt.hist(ranks, bins=bins, edgecolor="black")
    plt.xscale("log"); plt.title("Rank distribution (log bins)")
    plt.xlabel("rank"); plt.ylabel("count"); plt.grid(True, alpha=0.3)
    figsave(out_dir/"rank_hist_log.png")

    # same vs overall (R@10)
    plt.figure(figsize=(5,4))
    xs = ["overall","same-sent-only"]
    ys = [summary["recall_at_overall"].get("10",0.0), summary["same_only"]["recall_at"].get("10",0.0) if cnt_same else 0.0]
    ax = plt.gca(); bar(ax, xs, ys, "R@10: overall vs same-sentence only", "", "Recall")
    figsave(out_dir/"same_vs_overall.png")

    # error pie
    err_tot = summary["error_breakdown"]["errors_total"]
    plt.figure(figsize=(4.8,4.8))
    plt.pie([err_within, err_cross, max(0, err_tot-err_within-err_cross)],
            labels=["within","cross","others"], autopct="%1.1f%%")
    plt.title("Error type split"); figsave(out_dir/"error_pie.png")

    # abs Δr histogram
    plt.figure(figsize=(8,4))
    bins_k = list(hist_absdr.keys()); vals = [hist_absdr[b] for b in bins_k]
    ax = plt.gca(); bar(ax, bins_k, vals, "|Δr| histogram (within errors)", "abs Δr", "count")
    plt.xticks(rotation=45); figsave(out_dir/"abs_dr_hist.png")

    # R@10 by len
    plt.figure(figsize=(7,4))
    xs = sorted(acc_len.keys(), key=lambda s: (int(s.split('-')[0]) if s!='UNK' else 99999))
    ys = [safe_div(acc_len[x]["r10"], acc_len[x]["cnt"]) for x in xs]
    ax = plt.gca(); bar(ax, xs, ys, "R@10 by sentence length (tokens)", "len bucket", "R@10")
    figsave(out_dir/"r10_by_len.png")

    # errors by len (stacked)
    plt.figure(figsize=(7,4))
    parts = {
        "within":[safe_div(acc_len[x]["e_within"], acc_len[x]["cnt"]) for x in xs],
        "cross": [safe_div(acc_len[x]["e_cross"],  acc_len[x]["cnt"]) for x in xs],
    }
    ax = plt.gca(); stacked(ax, xs, parts, "Error type by length", "len bucket", "ratio")
    figsave(out_dir/"errors_by_len.png")

    # R@10 by stop/long/pron
    plt.figure(figsize=(7,4))
    xs = sorted(acc_stop.keys())
    ys = [safe_div(acc_stop[x]["r10"], acc_stop[x]["cnt"]) for x in xs]
    bar(plt.gca(), xs, ys, "R@10 vs stopword ratio", "stop ratio bin", "R@10")
    figsave(out_dir/"r10_by_stop.png")

    plt.figure(figsize=(7,4))
    xs = sorted(acc_long.keys())
    ys = [safe_div(acc_long[x]["r10"], acc_long[x]["cnt"]) for x in xs]
    bar(plt.gca(), xs, ys, "R@10 vs long-word ratio", "long ratio bin", "R@10")
    figsave(out_dir/"r10_by_long.png")

    plt.figure(figsize=(4.8,4))
    xs = ["no-pron","has-pron"]
    ys = [safe_div(acc_pron["0"]["r10"], acc_pron["0"]["cnt"]),
          safe_div(acc_pron["1"]["r10"], acc_pron["1"]["cnt"])]
    bar(plt.gca(), xs, ys, "R@10 vs pronoun presence", "", "R@10")
    figsave(out_dir/"r10_by_pron.png")

    # == 新增：Success@p% 曲线（overall / same-only / cross-only）==
    def _succ_at_pct_series(norm):
        s = norm.get("success_at_pct", {}) if norm else {}
        xs = ["1%","5%","10%"]
        ys = [s.get(x, None) if s else None for x in xs]
        return xs, ys
    xs = ["1%","5%","10%"]
    xs_o, ys_o = _succ_at_pct_series(norm_overall)
    xs_s, ys_s = _succ_at_pct_series(norm_same) if cnt_same else (xs, [None,None,None])
    xs_c, ys_c = _succ_at_pct_series(norm_cross) if cnt_cross else (xs, [None,None,None])
    plt.figure(figsize=(6.8,4.6))
    plt.plot(xs_o, ys_o, marker="o", label="overall")
    if cnt_same:  plt.plot(xs_s, ys_s, marker="o", label="same-only")
    if cnt_cross: plt.plot(xs_c, ys_c, marker="o", label="cross-only")
    plt.ylim(0,1); plt.grid(True, alpha=0.3)
    plt.title("Success@p% (pool-size normalized)")
    plt.xlabel("p"); plt.ylabel("success rate")
    plt.legend()
    figsave(out_dir/"success_at_pct_curve.png")

    # == 新增：Calibrated R@k 比较条形图 ==
    def _calR(norm, ks):
        d = norm.get("calibrated_R@k", {}) if norm else {}
        return [d.get(str(k), None) for k in ks]
    ks_plot = topk_vals
    plt.figure(figsize=(7.2,4.6))
    width = 0.25
    x = np.arange(len(ks_plot))
    yo = _calR(norm_overall, ks_plot)
    plt.bar(x - width, [0 if v is None else v for v in yo], width=width, label="overall")
    if cnt_same:
        ys = _calR(norm_same, ks_plot)
        plt.bar(x, [0 if v is None else v for v in ys], width=width, label="same-only")
    if cnt_cross:
        yc = _calR(norm_cross, ks_plot)
        plt.bar(x + width, [0 if v is None else v for v in yc], width=width, label="cross-only")
    plt.xticks(x, [f"R@{k}" for k in ks_plot])
    plt.ylim(0,1); plt.grid(True, axis="y", alpha=0.3)
    plt.title("Calibrated R@k (chance=0, perfect=1)")
    plt.legend()
    figsave(out_dir/"calibrated_r_at_k_bar.png")

    # Mini confusion (Top-N)
    if len(conf_counts)>0 and args.viz_topN>1:
        score_sent = Counter()
        for (src,dst),c in conf_counts.items():
            score_sent[src]+=c; score_sent[dst]+=c
        top_sents = [s for s,_ in score_sent.most_common(args.viz_topN)]
        idx = {s:i for i,s in enumerate(top_sents)}
        Nn = len(top_sents)
        M = np.zeros((Nn,Nn), dtype=np.int32)
        for (src,dst),c in conf_counts.items():
            if src in idx and dst in idx:
                M[idx[src], idx[dst]] += c
        plt.figure(figsize=(max(6, Nn*0.35), max(6, Nn*0.35)))
        ax = plt.gca()
        xt = [short_name(s) for s in top_sents]
        yt = [short_name(s) for s in top_sents]
        heatmap(ax, M, xt, yt, f"Mini Confusion (Top-{Nn} sentences, cross top-1)")
        figsave(out_dir/"mini_confusion_topN.png")

    # Sentence feature PCA
    sent_keys = list(sent_acc.keys())
    feat_mat=[]; color_r10=[]; sizes=[]
    for sk in sent_keys:
        idxs = [i for i,r in enumerate(test_rows) if r["_sent_key"]==sk]
        vecs = []
        for i in idxs:
            tf = tfeats[i]
            vecs.append([tf["n_tok"], tf["avg_tok_len"], tf["ttr"], tf["stop_ratio"],
                         tf["long_ratio"], tf["has_pron"]])
        v = np.array(vecs).mean(0) if vecs else np.zeros(6)
        feat_mat.append(v)
        s = sent_acc[sk]
        rr10 = safe_div(s["r10"], s["cnt"])
        color_r10.append(rr10)
        sizes.append(10 + 90*min(1.0, s["cnt"]/10.0))
    if len(feat_mat)>=3:
        X = np.array(feat_mat, dtype=np.float32)
        XY = pca_2d(X)
        plt.figure(figsize=(6.5,5.2))
        sc = plt.scatter(XY[:,0], XY[:,1], c=np.array(color_r10), s=np.array(sizes), cmap="viridis")
        plt.colorbar(sc, label="Sentence-level R@10")
        plt.title("Sentence text-feature PCA (colored by R@10)")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, alpha=0.3)
        figsave(out_dir/"sentence_feature_pca.png")

    # Hardest sentences Top-M
    M_top = args.hard_topM
    items=[]
    for sk, s in sent_acc.items():
        items.append((sk, safe_div(s["r10"], s["cnt"]), s["cnt"], s["mrr"]/max(1,s["cnt"])))
    items.sort(key=lambda x: (x[1], x[3]))
    topM = items[:min(M_top, len(items))]
    if topM:
        plt.figure(figsize=(8, max(4, 0.35*len(topM))))
        xs = [short_name(sk) for sk,_,_,_ in topM]
        r10s  = [v for _,v,_,_ in topM]
        mrrs  = [v for *_,v in [(a,b,c,d) for (a,b,c,d) in topM]]
        y = np.arange(len(topM))
        plt.barh(y, [1.0 - r for r in r10s], label="1 - R@10")
        plt.plot([1.0 - m for m in mrrs], y, "o", label="1 - MRR")
        plt.yticks(y, xs); plt.gca().invert_yaxis()
        plt.xlabel("hardness (higher = harder)")
        plt.title(f"Hardest sentences (Top-{len(topM)})")
        plt.legend(); plt.grid(True, axis="x", alpha=0.3)
        figsave(out_dir/"hardest_sentences_topM.png")

    # --------- HTML report ---------
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Fancy Eval Report (Fair)</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif; margin:24px;}}
h2{{margin-top:28px;}}
img{{max-width:100%; height:auto; border:1px solid #ddd; padding:4px; margin:10px 0;}}
.small{{color:#555;}}
.grid{{display:grid; grid-template-columns:1fr 1fr; gap:18px;}}
</style></head><body>
<h1>Fancy Eval Report (Fair)</h1>
<p class="small">run_dir: {run_dir.as_posix()}</p>
<pre class="small">{json.dumps(summary, indent=2)}</pre>

<h2>Core</h2>
<div class="grid">
  <div><img src="cmc_curve.png"><p class="small">CMC (Recall@k)</p></div>
  <div><img src="rank_hist_log.png"><p class="small">Rank distribution (log bins)</p></div>
  <div><img src="same_vs_overall.png"><p class="small">R@10: overall vs same-sentence only</p></div>
  <div><img src="error_pie.png"><p class="small">Error type split</p></div>
</div>

<h2>Within-sentence dynamics</h2>
<img src="abs_dr_hist.png"><p class="small">|Δr| histogram (within errors)</p>

<h2>Text buckets</h2>
<div class="grid">
  <div><img src="r10_by_len.png"><p class="small">R@10 by length</p></div>
  <div><img src="errors_by_len.png"><p class="small">Error type by length</p></div>
  <div><img src="r10_by_stop.png"><p class="small">R@10 vs stopword ratio</p></div>
  <div><img src="r10_by_long.png"><p class="small">R@10 vs long-word ratio</p></div>
  <div><img src="r10_by_pron.png"><p class="small">R@10 vs pronoun</p></div>
</div>

<h2>Pool-size normalized views</h2>
<div class="grid">
  <div><img src="success_at_pct_curve.png"><p class="small">Success@p% (overall/same/cross)</p></div>
  <div><img src="calibrated_r_at_k_bar.png"><p class="small">Calibrated R@k (chance=0, perfect=1)</p></div>
</div>

<h2>Confusions & sentence space</h2>
<img src="mini_confusion_topN.png"><p class="small">Mini confusion (Top-N sentences, cross top-1)</p>
<img src="sentence_feature_pca.png"><p class="small">Sentence text-feature PCA colored by R@10</p>

<h2>Hardest sentences</h2>
<img src="hardest_sentences_topM.png"><p class="small">Top-M hardest sentences</p>

</body></html>
"""
    (out_dir/"report.html").write_text(html, encoding="utf-8")
    log(f"Saved report to: {out_dir/'report.html'}")
    log("Done.")

def _pack_bucket(acc: Dict[str, float]) -> Dict[str, float]:
    return {
        "cnt": int(acc["cnt"]),
        "R@1": safe_div(acc["r1"], acc["cnt"]),
        "R@5": safe_div(acc["r5"], acc["cnt"]),
        "R@10":safe_div(acc["r10"],acc["cnt"]),
        "MRR": safe_div(acc["mrr"],acc["cnt"]),
        "err_within_ratio": safe_div(acc["e_within"], acc["cnt"]),
        "err_cross_ratio":  safe_div(acc["e_cross"],  acc["cnt"]),
    }

if __name__ == "__main__":
    if torch.cuda.is_available():
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    main()
