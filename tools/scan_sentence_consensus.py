#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/scan_sentence_consensus.py  —  Quick sweep for sentence-level consensus

目标：
- 在“不改动模型、不做重排”的前提下，离线评估“句级共识”的公式哪种最好。
- 给定一组 query（同一句子的窗口），我们从每个 query 的全库 logits 中，先取 topK & 量化门槛 q，
  再对每个候选句桶聚合得到 per-query 的句分数；最后跨 query 再做一次聚合（和/均值/投票等），
  得到该组对每个句子的支持度，Top-1 是否命中 GT 句。

输出：
- <run_dir>/results/sent_consensus_scan/scan_summary.json
- <run_dir>/results/sent_consensus_scan/scan_summary.csv
- 终端打印 Top-N 最佳配置

共识维度（可扫）：
- 窗口→句（单个 query 内）聚合：max / mean(top-m) / logsumexp(top-m) / soft-avg(T)
- 归一：按“本 query 保留的该句窗口数”或“句桶全长”（kept_count / bucket_count / bucket_sqrt / none）
- 跨 query 聚合：sum / mean / vote（每个 query 把该句作为 per-query top1 计票）
- 证据口径：topK ∈ {64,128,256,...}，q ∈ {0.95,0.98,...}，top-m ∈ {1,3,5}，soft T ∈ {1.5,2.5}

备注：
- 依赖：models.meg_encoder.UltimateMEGEncoder（与你现有工程一致）
- 与 eval / probe 脚本同风格：支持 --use_best_ckpt / --use_ckpt_logit_scale / --amp bf16 等
"""

import argparse
import csv
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
TARGET_T = 360       # 对齐 T
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
    if x.size(1) == T: return x
    return F.interpolate(x.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0)

# ------------------------- 句子别名/分桶 -------------------------
_CAND_SENT_KEYS = [
    "sentence_id","sentence_uid","utt_id","utterance_id","segment_id",
    "original_sentence_id","sentence_path","sentence_audio_path","transcript_path",
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

def sent_key_for_group(r: dict) -> Tuple[str, str]:
    als = sentence_aliases(r)
    return als[0] if als else ("unknown", content_id_of(r))

def _precompute_sentence_buckets(cand_sent_idx_o: torch.Tensor) -> Dict[int, torch.Tensor]:
    buckets = {}
    uniq = torch.unique(cand_sent_idx_o)
    for s in uniq.tolist():
        if s < 0: continue
        buckets[int(s)] = torch.nonzero(cand_sent_idx_o == s, as_tuple=False).view(-1)
    return buckets

# ------------------------- subject 映射/模型加载 -------------------------
def _normalize_subject_key(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x)
    m = re.search(r"(\d+)", s)
    if not m: return None
    return f"{int(m.group(1)):02d}"

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

from models.meg_encoder import UltimateMEGEncoder

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
    model.load_state_dict(new_state, strict=False)
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
        if cid in uniq: continue
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

# ------------------------- 共识公式（窗口→句；句→组） -------------------------
def _norm_factor(count_kept:int, bucket_size:int, mode:str)->float:
    if mode == "none": return 1.0
    if mode == "kept_count": return 1.0 / max(1.0, float(count_kept))
    if mode == "bucket_count": return 1.0 / max(1.0, float(bucket_size))
    if mode == "bucket_sqrt":  return 1.0 / math.sqrt(max(1.0, float(bucket_size)))
    return 1.0

def per_query_sentence_scores(
    s_logits_o: torch.Tensor,                # [O]
    buckets: Dict[int, torch.Tensor],        # sid -> idx(o)
    *, topk:int, q:float,
    agg:str, topm:int, soft_T:float,
    norm:str, cand_sent_idx_o: torch.Tensor,
) -> Dict[int, float]:
    """
    返回：sid -> score（该 query 对每个句子的支持度）
    """
    O = s_logits_o.size(0)
    K = min(int(topk), O)
    vals, idx = torch.topk(s_logits_o, k=K, largest=True, sorted=True)  # [K]
    thr = torch.quantile(s_logits_o, q=q) if 0.0 < q < 1.0 else vals[-1]
    keep = vals >= thr
    idx_kept = idx[keep]
    sid_kept = cand_sent_idx_o[idx_kept]

    # 收集到每句
    sid2vals: Dict[int, List[float]] = {}
    for j in range(idx_kept.numel()):
        sid = int(sid_kept[j].item())
        if sid < 0: continue
        sid2vals.setdefault(sid, []).append(float(vals[keep][j].item()))

    out: Dict[int, float] = {}
    for sid, arr in sid2vals.items():
        arr.sort(reverse=True)
        take = arr[:max(1, int(topm))] if agg in ("mean","lse","soft") else arr[:1]
        if len(take) == 0:
            continue
        if agg == "max":
            score = take[0]
            count_kept = 1
        elif agg == "mean":
            score = float(np.mean(take))
            count_kept = len(arr)
        elif agg == "lse":
            m0 = max(take)
            score = m0 + float(np.log(max(EPS, np.sum(np.exp(np.array(take) - m0)))))
            count_kept = len(arr)
        else:  # soft
            v = np.array(take, dtype=np.float64)
            p = np.exp(v / max(1e-6, float(soft_T)))
            p = p / max(EPS, p.sum())
            score = float((p * v).sum())
            count_kept = len(arr)

        bucket_size = int(buckets.get(sid, torch.empty(0)).numel())
        score *= _norm_factor(count_kept=count_kept, bucket_size=bucket_size, mode=norm)
        out[sid] = score
    return out

def across_queries_aggregate(
    per_query_scores: List[Dict[int,float]],  # len = group_size
    mode: str, num_sents: int
) -> Dict[int, float]:
    if mode in ("sum","mean"):
        acc: Dict[int, float] = {}
        for d in per_query_scores:
            for sid, v in d.items():
                acc[sid] = acc.get(sid, 0.0) + float(v)
        if mode == "mean" and len(per_query_scores) > 0:
            for k in list(acc.keys()):
                acc[k] /= float(len(per_query_scores))
        return acc
    elif mode == "vote":
        votes: Dict[int, int] = {}
        for d in per_query_scores:
            if not d: continue
            # 该 query 下分数最高的句子投 1 票
            sid = max(d.items(), key=lambda x: x[1])[0]
            votes[sid] = votes.get(sid, 0) + 1
        return {k: float(v) for k, v in votes.items()}
    else:
        raise ValueError(f"unknown across mode: {mode}")

# ------------------------- 主流程（扫描或单配） -------------------------
def evaluate_groups_logits(
    model, A, pool_ids, candidate_rows, test_rows, device, subj_map, scale,
    cand_sent_idx_o: torch.Tensor, buckets: Dict[int, torch.Tensor],
    sample_groups: int = 0
):
    # 组装“句子组”
    sent2idx: Dict[Tuple[str,str], List[int]] = {}
    for i, r in enumerate(test_rows):
        k = sent_key_for_group(r)
        sent2idx.setdefault(k, []).append(i)
    groups = list(sent2idx.values())
    if sample_groups and sample_groups > 0:
        groups = groups[:sample_groups]
    log(f"[INFO] Groups={len(groups)} (avg windows/sent={len(test_rows)/max(1,len(groups)):.2f})")

    # 预先把整个 test 的 gt 索引映射好
    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in test_rows]
    gt_sid_per_group = []
    logits_per_group: List[torch.Tensor] = []

    @torch.no_grad()
    def encode_meg_batch(rows: List[dict]) -> torch.Tensor:
        megs, locs, sidx = [], [], []
        miss = 0
        for r in rows:
            mp = r["meg_win_path"]; lp = r["sensor_coordinates_path"]
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
            log(f"[WARN] {miss}/{len(rows)} rows miss subject mapping; fallback subj_idx=0")

        meg = torch.stack(megs, 0).to(device)
        loc = torch.stack(locs, 0).to(device)
        sid = torch.tensor(sidx, dtype=torch.long, device=device)

        y = model(meg_win=meg, sensor_locs=loc, subj_idx=sid)  # [B,1024,T]
        if y.size(2) != TARGET_T:
            y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)
        return y

    # 编码 + logits（每个组一次）
    for g in tqdm(groups, desc="Encode+Logits (Grouped)"):
        rows = [test_rows[i] for i in g]
        Y = encode_meg_batch(rows)                  # [B,1024,T]
        logits = compute_logits_clip(Y, A, scale=scale)  # [B,O]
        logits_per_group.append(logits.detach())
        # 该组 GT 句 sid（以组首为准）
        g0 = gt_index[g[0]]
        gt_sid = int(cand_sent_idx_o[g0].item()) if cand_sent_idx_o[g0].item() >= 0 else -1
        gt_sid_per_group.append(gt_sid)

    return groups, logits_per_group, gt_sid_per_group

def run_single_or_scan(args):
    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if "16" in amp or "fp16" in amp else None)

    # 数据与池
    test_rows = read_jsonl(Path(args.test_manifest))
    log(f"[INFO] test rows = {len(test_rows):,}")
    A, pool_ids, candidate_rows = load_audio_pool_unique(test_rows, device=device, dtype=torch.float32)
    O = A.size(0); log(f"[INFO] candidate windows O={O}")

    # 分桶
    canon2idx, alias2idx, cand_sent_idx = build_sentence_index_with_alias(candidate_rows)
    S = len(canon2idx); log(f"[INFO] sentences S={S}")
    cand_sent_idx = torch.tensor(cand_sent_idx, dtype=torch.long, device=device)
    buckets = _precompute_sentence_buckets(cand_sent_idx)

    # 模型
    run_dir = Path(args.run_dir)
    subj_map = read_subject_mapping_from_records(run_dir)
    ckpt_path = choose_ckpt_path(args)
    model, meta = load_model_from_ckpt(ckpt_path, run_dir, device=device)
    scale = meta.get("logit_scale_exp", None) if args.use_ckpt_logit_scale else None

    # 编码 + logits（缓存）
    groups, logits_per_group, gt_sid_per_group = evaluate_groups_logits(
        model, A, pool_ids, candidate_rows, test_rows, device, subj_map, scale,
        cand_sent_idx_o=cand_sent_idx, buckets=buckets, sample_groups=args.sample_groups
    )

    # 估计窗口级 R@1（作为独立性上界的 p）
    # 注意：这里是“窗口检索对窗口GT”的 R@1，不聚合
    # 只做一个粗略估计：取每个组的第一条 query 计算是否把其 GT 窗口排第1
    # （不完全等同你的主评测，但足够估算）
    top1_hit = 0; total_q = 0
    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index_all = [cid_to_index[content_id_of(r)] for r in test_rows]
    for g, logits in zip(groups, logits_per_group):
        s = logits[0]  # 组内第一条
        gidx = gt_index_all[g[0]]
        rank = int((s > s[gidx]).sum().item()) + 1
        top1_hit += int(rank == 1); total_q += 1
    p_window = (top1_hit / max(1,total_q))

    # 扫描参数
    if args.mode == "scan":
        topk_list   = [int(x) for x in args.topk_list.split(",")] if args.topk_list else [64,128,256]
        q_list      = [float(x) for x in args.q_list.split(",")]  if args.q_list else [0.95,0.98]
        agg_list    = args.agg_list.split(",") if args.agg_list else ["max","mean","lse"]
        topm_list   = [int(x) for x in args.topm_list.split(",")] if args.topm_list else [1,3]
        norm_list   = args.norm_list.split(",") if args.norm_list else ["kept_count","bucket_count","bucket_sqrt","none"]
        across_list = args.across_list.split(",") if args.across_list else ["sum","vote"]
        soft_T_list = [float(x) for x in (args.soft_T_list.split(",") if args.soft_T_list else ["1.5","2.5"])]

        combos = []
        for topk in topk_list:
            for q in q_list:
                for agg in agg_list:
                    for norm in norm_list:
                        if agg == "soft":
                            for T in soft_T_list:
                                combos.append((topk,q,agg,3,norm,"sum",T))
                                combos.append((topk,q,agg,3,norm,"vote",T))
                        elif agg in ("mean","lse"):
                            for m in topm_list:
                                combos.append((topk,q,agg,m,norm,"sum",0.0))
                                combos.append((topk,q,agg,m,norm,"vote",0.0))
                        else:  # max
                            combos.append((topk,q,agg,1,norm,"sum",0.0))
                            combos.append((topk,q,agg,1,norm,"vote",0.0))
    else:
        combos = [(args.topk, args.q, args.agg, args.topm, args.norm, args.across, args.soft_T)]

    # 扫描
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "results" / "sent_consensus_scan")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scan_summary.csv"
    json_path = out_dir / "scan_summary.json"

    rows_csv = []
    best_print = []

    for (topk,q,agg,topm,norm,across,soft_T) in tqdm(combos, desc="Scan combos"):
        # 逐组计算句级支持并评估
        top1_ok = 0; top2_ok = 0; top3_ok = 0; top5_ok = 0
        margins = []
        for g_idx, (g, logits, gt_sid) in enumerate(zip(groups, logits_per_group, gt_sid_per_group)):
            per_query = []
            for j in range(logits.size(0)):
                d = per_query_sentence_scores(
                    logits[j], buckets,
                    topk=topk, q=q, agg=agg, topm=topm, soft_T=soft_T,
                    norm=norm, cand_sent_idx_o=cand_sent_idx
                )
                per_query.append(d)
            s_scores = across_queries_aggregate(per_query, mode=across, num_sents=len(buckets))

            if not s_scores:
                # 没有任何句得分，跳过（算错误）
                pass
            else:
                # 排序选前若干
                sorted_pairs = sorted(s_scores.items(), key=lambda x: x[1], reverse=True)
                sids_sorted = [sid for sid,_ in sorted_pairs]
                if len(sids_sorted) >= 1 and gt_sid == sids_sorted[0]:
                    top1_ok += 1
                if gt_sid in sids_sorted[:2]: top2_ok += 1
                if gt_sid in sids_sorted[:3]: top3_ok += 1
                if gt_sid in sids_sorted[:5]: top5_ok += 1
                # margin(top1 - gt)
                if gt_sid in s_scores:
                    best_sid = sids_sorted[0]
                    margins.append(float(s_scores[best_sid] - s_scores[gt_sid]))
                else:
                    # GT 句无得分，记为一个负 margin（用最小值-1）
                    if sids_sorted:
                        margins.append(float(s_scores[sids_sorted[0]] - 0.0))
                    else:
                        margins.append(0.0)

        N = len(groups)
        acc1 = top1_ok / max(1,N)
        acc2 = top2_ok / max(1,N)
        acc3 = top3_ok / max(1,N)
        acc5 = top5_ok / max(1,N)
        margin_med = float(np.median(margins)) if margins else 0.0

        rows_csv.append({
            "topk": topk, "q": q, "agg": agg, "topm": topm, "norm": norm,
            "across": across, "soft_T": soft_T,
            "groups": N, "sent_top1": acc1, "sent_top2": acc2, "sent_top3": acc3, "sent_top5": acc5,
            "margin_median": margin_med,
            "window_p_est": p_window
        })
        best_print.append((acc1, acc2, acc3, acc5, margin_med, (topk,q,agg,topm,norm,across,soft_T)))

    # 保存 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)

    # 保存 JSON（按 acc1 排序）
    rows_sorted = sorted(rows_csv, key=lambda r: (r["sent_top1"], r["sent_top2"], r["sent_top3"]), reverse=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_groups": len(logits_per_group),
            "window_r1_estimate": p_window,
            "independence_upper_bound_mean": float(np.mean([1.0 - (1.0 - p_window) ** len(g) for g in groups])),
            "best_top5": rows_sorted[:5],
            "all": rows_sorted
        }, f, indent=2, ensure_ascii=False)

    # 打印前 10
    print("\n=== Top 10 configs by sent_top1 ===")
    for k,(acc1,acc2,acc3,acc5,mm,params) in enumerate(sorted(best_print, key=lambda x:x[0], reverse=True)[:10]):
        topk,q,agg,topm,norm,across,soft_T = params
        print(f"[#{k+1}] acc1={acc1:.4f} acc2={acc2:.4f} acc3={acc3:.4f} acc5={acc5:.4f} "
              f"med_margin={mm:.3f} | topk={topk} q={q} agg={agg} m={topm} norm={norm} across={across} T={soft_T}")

    log(f"[DONE] results saved to:\n  - {csv_path}\n  - {json_path}")

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")
    p.add_argument("--use_ckpt_logit_scale", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off","bf16","fp16","16-mixed"])
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--sample_groups", type=int, default=0, help="仅取前 N 组做快速扫描；0=全部")

    p.add_argument("--mode", type=str, default="scan", choices=["scan","single"])

    # 扫描列表（字符串，逗号分隔）
    p.add_argument("--topk_list", type=str, default="64,128,256")
    p.add_argument("--q_list", type=str, default="0.95,0.98")
    p.add_argument("--agg_list", type=str, default="max,mean,lse,soft")
    p.add_argument("--topm_list", type=str, default="1,3,5")
    p.add_argument("--norm_list", type=str, default="kept_count,bucket_count,bucket_sqrt,none")
    p.add_argument("--across_list", type=str, default="sum,vote")
    p.add_argument("--soft_T_list", type=str, default="1.5,2.5")

    # 单一配置模式
    p.add_argument("--topk", type=int, default=128)
    p.add_argument("--q", type=float, default=0.98)
    p.add_argument("--agg", type=str, default="mean", choices=["max","mean","lse","soft"])
    p.add_argument("--topm", type=int, default=3)
    p.add_argument("--norm", type=str, default="kept_count",
                   choices=["kept_count","bucket_count","bucket_sqrt","none"])
    p.add_argument("--across", type=str, default="sum", choices=["sum","mean","vote"])
    p.add_argument("--soft_T", type=float, default=2.5)

    return p.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    run_single_or_scan(args)

