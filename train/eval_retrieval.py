#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Eval (paper-aligned & train-aligned, subject-consistent) + QCCP 重排

关键点：
1) 与训练一致的被试映射（优先从 run_dir/records 读取；否则回退 TRAIN 构建）
2) 支持 --use_best_ckpt 从 run_dir/records/best_checkpoint.txt 取 ckpt
3) 相似度默认 sim=clip（仅候选端 L2、无温度），对齐到 T=360，不做时间池化
4) 新增 QCCP（Query-Conditioned Contextual Prior）句内先验重排：
   - 先按“句子”分组，同句内窗口一起前向、一起重排
   - V(i,j)=Σ_u κ(|t_u - t_i|)*ReLU(S(u,j)-m_u)，m_u 为分位阈值
   - 仅对每个查询的 top-K 候选加先验，带熵门控（可关）
   - 完全零训练，仅在评测阶段对 logits 做增量
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# model
from models.meg_encoder2 import UltimateMEGEncoder

TARGET_T = 360
AUDIO_D = 1024
EPS = 1e-8


# ------------------------- 通用工具 -------------------------
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
    return x if abs(x.shape[0] - AUDIO_D) < abs(x.shape[1] - AUDIO_D) else x.T


def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T


def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(1) == T:
        return x
    return F.interpolate(x.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0)


def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        with open(rec, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("model_cfg", {})
    return {}


def choose_ckpt_path(args) -> Path:
    if args.use_best_ckpt:
        best_txt = Path(args.run_dir) / "records" / "best_checkpoint.txt"
        assert best_txt.exists(), f"best_checkpoint.txt not found at {best_txt}"
        ckpt = best_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        ckpt_path = Path(ckpt)
        assert ckpt_path.exists(), f"best ckpt not found: {ckpt_path}"
        log(f"Using BEST checkpoint from records: {ckpt_path}")
        return ckpt_path
    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"--ckpt_path not found: {ckpt_path}"
    return ckpt_path


def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str) -> UltimateMEGEncoder:
    model_cfg = load_cfg_from_records(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")  # 保持兼容
    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", {})
    assert model_cfg, "找不到 model_cfg（既不在 run_dir/records/config.json，也不在 ckpt.hyper_parameters）"

    if "out_timesteps" in UltimateMEGEncoder.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None

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
    return model


def build_subject_map_from_train(train_rows: List[dict]) -> Dict[str, int]:
    ids = sorted({str(r["subject_id"]) for r in train_rows})
    return {sid: i for i, sid in enumerate(ids)}


def load_subject_map_from_run(run_dir: Path) -> Dict[str, int]:
    """优先读取 snapshot；否则读 config.json 里的 subject_mapping_path。"""
    snap = run_dir / "records" / "subject_mapping_snapshot.json"
    if snap.exists():
        with open(snap, "r", encoding="utf-8") as f:
            d = json.load(f)
        mp = d.get("mapping") or d.get("map") or {}
        if mp:
            return {str(k): int(v) for k, v in mp.items()}
    cfg_p = run_dir / "records" / "config.json"
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        p = cfg.get("subject_mapping_path")
        if p:
            jp = Path(p)
            if jp.exists():
                with open(jp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                mp = d.get("mapping") or d.get("map") or {}
                if mp:
                    return {str(k): int(v) for k, v in mp.items()}
    return {}


def annotate_subject_indices(rows: List[dict], sub_map: Dict[str, int],
                             strict_subjects: bool, note_prefix: str) -> Tuple[List[dict], str, Dict[str, int]]:
    filtered = []
    skipped_subjects = 0
    missing_artifacts = 0
    for r in rows:
        sid = str(r["subject_id"])
        if strict_subjects and sid not in sub_map:
            skipped_subjects += 1
            continue
        if not Path(r.get("sensor_coordinates_path", "")).exists() or not Path(r.get("meg_win_path", "")).exists():
            missing_artifacts += 1
            continue
        r2 = dict(r)
        r2["_subject_idx"] = sub_map.get(sid, 0 if not strict_subjects else -1)
        if strict_subjects and r2["_subject_idx"] < 0:
            skipped_subjects += 1
            continue
        filtered.append(r2)

    if skipped_subjects > 0:
        log(f"[Note] skipped test rows with unseen subjects: {skipped_subjects}")
    if missing_artifacts > 0:
        log(f"[Note] skipped rows missing MEG/coords: {missing_artifacts}")

    note = f"{note_prefix}; strict_subjects={bool(strict_subjects)}; skipped_subjects={skipped_subjects}; missing_artifacts={missing_artifacts}"
    return filtered, note, sub_map


@torch.no_grad()
def encode_meg_batch(model, batch_rows: List[dict], device: str) -> torch.Tensor:
    megs, locs, sidx = [], [], []
    for r in batch_rows:
        x = np.load(r["meg_win_path"]).astype(np.float32)
        x = ensure_meg_CxT(x)
        megs.append(torch.from_numpy(x))
        loc = np.load(r["sensor_coordinates_path"]).astype(np.float32)
        locs.append(torch.from_numpy(loc))
        sidx.append(r["_subject_idx"])

    meg = torch.stack(megs, 0).to(device)  # [B,C,T]
    loc = torch.stack(locs, 0).to(device)  # [B,C,3]
    sid = torch.tensor(sidx, dtype=torch.long, device=device)

    y = model(meg_win=meg, sensor_locs=loc, subj_idx=sid,
              meg_sent=None, meg_sent_mask=None)  # -> [B,1024,T?]
    if y.dim() != 3 or y.size(1) != AUDIO_D:
        raise RuntimeError(f"encoder must output [B,1024,T], got {tuple(y.shape)}")

    if y.size(2) != TARGET_T:
        y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)
    return y  # [B,1024,360]


@torch.no_grad()
def load_audio_pool_unique(test_rows: List[dict], device: str, dtype: torch.dtype) -> Tuple[torch.Tensor, List[str]]:
    uniq: Dict[str, str] = {}
    for r in test_rows:
        cid = content_id_of(r)
        if cid not in uniq:
            uniq[cid] = r["audio_feature_path"]

    ids = list(uniq.keys())
    feats = []
    for cid in tqdm(ids, desc="Loading & aligning audio pool"):
        p = uniq[cid]
        a = np.load(p).astype(np.float32)
        a = ensure_audio_DxT(a)
        ta = torch.from_numpy(a)
        ta = maybe_interp_1DT(ta, TARGET_T)
        feats.append(ta)
    A = torch.stack(feats, 0).to(device=device, dtype=dtype)  # [O,1024,360]
    return A, ids


def compute_logits(
    queries: torch.Tensor,   # [b,1024,360]
    pool: torch.Tensor,      # [O,1024,360]
    sim: str = "clip",       # clip | cosine | dot
    tau: float = 0.0,
) -> torch.Tensor:
    q = queries.to(torch.float32)
    A = pool.to(torch.float32)

    if sim == "dot":
        logits = torch.einsum("bct,oct->bo", q, A)

    elif sim == "clip":
        inv_norms = 1.0 / (A.norm(dim=(1, 2), p=2) + EPS)   # [O]
        logits = torch.einsum("bct,oct,o->bo", q, A, inv_norms)
        if tau and tau > 0:
            logits = logits / tau

    elif sim == "cosine":
        qn = q / (q.norm(dim=(1, 2), keepdim=True, p=2) + EPS)
        an = A / (A.norm(dim=(1, 2), keepdim=True, p=2) + EPS)
        logits = torch.einsum("bct,oct->bo", qn, an)
        if tau and tau > 0:
            logits = logits / tau
    else:
        raise ValueError(f"Unsupported sim: {sim}")

    return logits.to(torch.float32)


def ranks_from_scores(scores: torch.Tensor, gt_index: int) -> int:
    gt = scores[gt_index]
    better = (scores > gt).sum().item()
    return int(better) + 1


# ------------------------- QCCP：句内先验重排 -------------------------
_CAND_SENT_KEYS = [
    "sentence_id", "sent_id", "sentence_uid",
    "utt_id", "utterance_id", "segment_id",
    "original_sentence_id", "sentence_path", "sentence_audio_path", "transcript_path",
]

def _sent_key(row: dict) -> Tuple[str, str]:
    # 1) 优先显式句级 id
    for k in _CAND_SENT_KEYS:
        if k in row and row[k]:
            return ("k:"+k, str(row[k]))
    # 2) 退化：按原音频 + 近似句边界（若有）
    a = str(row.get("original_audio_path", ""))
    so = row.get("original_sentence_onset_in_audio_s", None)
    eo = row.get("original_sentence_offset_in_audio_s", None)
    if so is not None and eo is not None:
        return ("audio+sent", f"{a}::{float(so):.3f}-{float(eo):.3f}")
    # 3) 最后兜底：按原音频归并
    return ("audio", a)

def group_rows_by_sentence(rows: List[dict]) -> Dict[Tuple[str,str], List[int]]:
    g: Dict[Tuple[str,str], List[int]] = {}
    for i, r in enumerate(rows):
        k = _sent_key(r)
        g.setdefault(k, []).append(i)
    # 组内按时间排序（用于时间核）
    def _tcenter(rr: dict) -> float:
        s0 = float(rr.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(rr.get("local_window_offset_in_audio_s", s0))
        return 0.5 * (s0 + s1)
    for k, idxs in g.items():
        idxs.sort(key=lambda i: _tcenter(rows[i]))
    return g

def window_centers(rows: List[dict], idxs: List[int]) -> torch.Tensor:
    t = []
    for i in idxs:
        r = rows[i]
        s0 = float(r.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(r.get("local_window_offset_in_audio_s", s0))
        t.append(0.5 * (s0 + s1))
    return torch.tensor(t, dtype=torch.float32)

@torch.no_grad()
def qccp_rerank_group(
    base_logits_bo: torch.Tensor,    # [B,O]
    times_b: torch.Tensor,           # [B] （秒）
    topk: int = 128,
    q_quantile: float = 0.9,
    half_life_s: float = 2.0,
    gamma: float = 0.7,
    gate: bool = True,
) -> torch.Tensor:
    """
    support_i(j) = Σ_{u≠i} κ(|t_u - t_i|) * ReLU( S(u,j) - m_u ),  m_u = quantile_q(S_u(:))
    logits'_i(j) = logits_i(j) + β_i * γ * norm(support_i(j))，仅在 top-K 上加成
    """
    B, O = base_logits_bo.shape
    device = base_logits_bo.device

    # 分位阈值
    m_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)  # [B,1]

    # 时间核
    dt = torch.abs(times_b.view(-1,1) - times_b.view(1,-1))   # [B,B]
    kappa = (0.5 ** (dt / max(1e-6, half_life_s))).to(device) # [B,B]
    kappa.fill_diagonal_(0.0)

    K = min(topk, O)
    topk_idx = torch.topk(base_logits_bo, k=K, dim=1, largest=True, sorted=False).indices  # [B,K]
    out = base_logits_bo.clone()

    for i in range(B):
        idx_j = topk_idx[i]                  # [K]
        Suj = base_logits_bo[:, idx_j]       # [B,K]
        votes = (Suj - m_b1).clamp_min_(0.0) # [B,K]
        weights = kappa[i].unsqueeze(1)      # [B,1]
        support = (weights * votes).sum(dim=0)  # [K]
        # 归一化（句长无关）
        support = support / (kappa[i].sum() + EPS)

        beta_i = gamma
        if gate and K > 1:
            p = F.softmax(support, dim=0)
            ent = -(p * (p + EPS).log()).sum() / np.log(K)
            beta_i = gamma * float(1.0 - ent)  # ∈(0, γ]

        out[i, idx_j] = out[i, idx_j] + beta_i * support

    return out


# ------------------------- 主评测流程 -------------------------
def evaluate(args):
    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if "16" in amp or "fp16" in amp else None)

    test_rows = read_jsonl(Path(args.test_manifest))
    train_rows = read_jsonl(Path(args.train_manifest))

    # ---------- Subject map（关键修复） ----------
    run_dir = Path(args.run_dir)
    subject_note = ""
    sub_map: Dict[str, int] = {}

    if args.subject_map == "from_run" or args.subject_map == "auto":
        sub_map = load_subject_map_from_run(run_dir)
        if sub_map:
            subject_note = "subject map from RUN (records)"
            log(f"Loaded subject map from run: {len(sub_map)} subjects.")
        elif args.subject_map == "from_run":
            raise RuntimeError("subject_map=from_run but no mapping was found under run_dir/records")
    if not sub_map:
        sub_map = build_subject_map_from_train(train_rows)
        subject_note = "subject map built from TRAIN (fallback)"
        log(f"Built subject map from TRAIN: {len(sub_map)} subjects.")

    Path(run_dir / "records").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "records" / "subjects_eval_used.json", "w", encoding="utf-8") as f:
        json.dump({"n_subjects": len(sub_map), "map": sub_map, "note": subject_note}, f, ensure_ascii=False, indent=2)
    log(f"Saved subject mapping used for eval -> {(run_dir / 'records' / 'subjects_eval_used.json').as_posix()}")

    # 用映射标注/过滤 TEST rows
    filtered, subject_note, _ = annotate_subject_indices(test_rows, sub_map, args.strict_subjects, subject_note)
    assert filtered, "No valid test rows after filtering."

    # ===== 句子分组（用于 QCCP）=====
    sent2idx = group_rows_by_sentence(filtered)
    groups = list(sent2idx.values())
    avg_w = sum(len(g) for g in groups) / max(1, len(groups))
    log(f"Grouped test rows into {len(groups)} sentences (avg windows/sent = {avg_w:.2f})")

    # 候选池（内容级唯一）
    A, pool_ids = load_audio_pool_unique(filtered, device=device, dtype=torch.float32)
    pool_size = A.size(0)
    log(f"Candidate pool size (unique test audio windows): {pool_size}")
    assert A.dim() == 3 and A.shape[1:] == (AUDIO_D, TARGET_T), f"A shape wrong: {tuple(A.shape)}"

    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in filtered]

    # 模型 & ckpt
    ckpt_path = choose_ckpt_path(args)
    model = load_model_from_ckpt(ckpt_path, run_dir, device=device)

    # 评估
    topk_list = [int(x) for x in args.topk.split(",")]
    recalls = {k: 0 for k in topk_list}
    mrr_sum = 0.0
    ranks: List[int] = []

    save_topk_k = max(0, int(args.save_topk)) if args.save_topk is not None else 0
    preds_topk_file = None
    preds_tsv_file = None
    if save_topk_k > 0:
        out_dir = run_dir / "results" / "retrieval_fullpool"
        out_dir.mkdir(parents=True, exist_ok=True)
        preds_topk_path = out_dir / f"preds_topk{save_topk_k}.jsonl"
        preds_tsv_path = out_dir / f"preds_topk{save_topk_k}.tsv"
        preds_topk_file = open(preds_topk_path, "w", encoding="utf-8")
        preds_tsv_file = open(preds_tsv_path, "w", encoding="utf-8")
        preds_tsv_file.write("query_index\trank\tgt_cid\tpred_cids\n")

    # QCCP 超参
    qccp_topk = int(getattr(args, "qccp_topk", 128))
    qccp_q    = float(getattr(args, "qccp_q", 0.9))
    qccp_hs   = float(getattr(args, "qccp_half_life_s", 2.0))
    qccp_gam  = float(getattr(args, "qccp_gamma", 0.7))
    qccp_gate = bool(getattr(args, "qccp_gate", False))  # 注意：flag 类参数，默认 False；传 --qccp_gate 才启用

    num_queries = 0
    with torch.no_grad():
        pbar = tqdm(range(len(groups)), desc="Evaluating (QCCP)")
        for gi in pbar:
            idxs = groups[gi]
            batch_rows = [filtered[i] for i in idxs]

            if autocast_dtype is None:
                Y = encode_meg_batch(model, batch_rows, device=device)     # [B,1024,360]
            else:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    Y = encode_meg_batch(model, batch_rows, device=device)

            logits = compute_logits(queries=Y, pool=A, sim=args.sim, tau=args.tau)  # [B,O]

            # 句内时间（秒）
            times = window_centers(filtered, idxs).to(device=logits.device)

            # QCCP 重排
            logits_qccp = qccp_rerank_group(
                logits, times,
                topk=qccp_topk, q_quantile=qccp_q,
                half_life_s=qccp_hs, gamma=qccp_gam, gate=qccp_gate
            )

            # 统计
            for j_in_group, global_j in enumerate(idxs):
                g = gt_index[global_j]
                s = logits_qccp[j_in_group]
                rank = ranks_from_scores(s, g)
                ranks.append(rank)
                mrr_sum += 1.0 / rank
                for k in topk_list:
                    recalls[k] += int(rank <= k)

                if save_topk_k > 0:
                    topk_scores, topk_idx = torch.topk(s, k=save_topk_k, largest=True, sorted=True)
                    pred_cids = [pool_ids[int(t)] for t in topk_idx.tolist()]
                    rec = {
                        "query_index": int(global_j),
                        "gt_rank": int(rank),
                        "gt_cid": pool_ids[g],
                        "pred_cids": pred_cids,
                        "pred_scores": [float(x) for x in topk_scores.tolist()],
                        "hit@K": any(idx == g for idx in topk_idx.tolist()),
                        "qccp": {"topk": qccp_topk, "q": qccp_q, "half_life_s": qccp_hs, "gamma": qccp_gam, "gate": qccp_gate},
                    }
                    preds_topk_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    preds_tsv_file.write(f"{global_j}\t{rank}\t{pool_ids[g]}\t{','.join(pred_cids)}\n")

            num_queries += len(idxs)

    if preds_topk_file is not None:
        preds_topk_file.close()
    if preds_tsv_file is not None:
        preds_tsv_file.close()

    metrics = {
        "num_queries": num_queries,
        "pool_size": pool_size,
        "sim": args.sim,
        "tau": None if (args.tau is None or args.tau <= 0) else float(args.tau),
        "recall_at": {str(k): recalls[k] / num_queries for k in topk_list},
        "mrr": mrr_sum / num_queries,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "topk_list": topk_list,
        "note": subject_note,
        "qccp": {
            "topk": qccp_topk, "q": qccp_q,
            "half_life_s": qccp_hs, "gamma": qccp_gam, "gate": qccp_gate
        }
    }

    out_dir = run_dir / "results" / "retrieval_fullpool"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.save_json) if args.save_json else (out_dir / "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    out_ranks = Path(args.save_ranks) if args.save_ranks else (out_dir / "ranks.txt")
    with open(out_ranks, "w", encoding="utf-8") as f:
        for r in ranks:
            f.write(str(int(r)) + "\n")

    log("==== Retrieval Results (test full pool + QCCP) ====")
    log(json.dumps(metrics, indent=2, ensure_ascii=False))
    log(f"Metrics saved to: {out_json.as_posix()}")
    log(f"Ranks saved to  : {out_ranks.as_posix()}")


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="", help="显式 ckpt 路径；若配合 --use_best_ckpt 则忽略此项")
    p.add_argument("--use_best_ckpt", action="store_true", help="从 run_dir/records/best_checkpoint.txt 读取最佳 ckpt")
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16", "16-mixed"])
    p.add_argument("--batch_size", type=int, default=256)  # 保留参数以兼容，但当前按句分组不使用
    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--strict_subjects", action="store_true",
                   help="仅保留 TEST 中 subject 存在于映射的样本（建议开启）")
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--save_ranks", type=str, default="")
    p.add_argument("--save_topk", type=int, default=0)

    p.add_argument("--sim", type=str, default="clip", choices=["clip", "cosine", "dot"],
                   help="clip=仅候选端 L2（与训练一致）；cosine=两端 L2；dot=纯点积（不推荐）")
    p.add_argument("--tau", type=float, default=0.0, help="温度：>0 时 logits/=tau")

    p.add_argument("--subject_map", type=str, default="auto", choices=["auto", "from_run", "train"],
                   help="auto 从 run_dir/records 读取（找不到回退 train）；from_run 只用 run_dir；train 只用 TRAIN 构建")

    # QCCP 先验参数（默认保守）
    p.add_argument("--qccp_topk", type=int, default=128, help="仅对前 K 进行先验重排")
    p.add_argument("--qccp_q", type=float, default=0.9, help="邻居票的分位阈值 q")
    p.add_argument("--qccp_half_life_s", type=float, default=2.0, help="时间核半衰期（秒）")
    p.add_argument("--qccp_gamma", type=float, default=0.7, help="先验加权系数 γ")
    p.add_argument("--qccp_gate", action="store_true", help="启用基于熵的门控")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    evaluate(args)
