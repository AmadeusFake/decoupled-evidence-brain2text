#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEG-compatible Retrieval Eval (full-pool, no re-ranking) with robust subject mapping & forward-call

- 适配 EEG/MEG 窗口：row["meg_win_path"] -> [C,T]，row["sensor_coordinates_path"] -> [C,3]
- 动态解析模型类（ckpt.hyper_parameters.model_target），前向自适配（只传模型声明的参数；含同义名映射；并有位置参数回退）
- 编码器输出统一到 [B,1024,360]，相似度 "clip" 仅候选端 L2
- 无任何重排；严格模式下自动 subject 归一化，失败则回退为非严格以保证可评
"""

from __future__ import annotations
import argparse
import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

TARGET_T = 360
AUDIO_D = 1024
EPS = 1e-8


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
        raise ValueError(f"expect 2D MEG/EEG array, got {x.shape}")
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
        return cfg.get("model_cfg", {}) or {}
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


def _resolve_model_ctor(ckpt, default="models.eeg_encoder:UltimateMEGEncoder"):
    target = None
    if isinstance(ckpt, dict):
        hp = ckpt.get("hyper_parameters", {}) or {}
        for key in ["model_target", "model_cls", "model_class", "model_name"]:
            val = hp.get(key, None)
            if isinstance(val, str) and ":" in val:
                target = val
                break
    if target is None:
        target = default
    mod, cls = target.split(":")
    return getattr(importlib.import_module(mod), cls)


def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")  # 显式自用权衡，保留默认安全提示
    model_cfg = load_cfg_from_records(run_dir)
    if not model_cfg:
        model_cfg = ckpt.get("hyper_parameters", {}).get("model_cfg", {})
    assert model_cfg, "找不到 model_cfg（既不在 run_dir/records/config.json，也不在 ckpt.hyper_parameters）"

    ctor = _resolve_model_ctor(ckpt)
    if "out_timesteps" in ctor.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None

    model = ctor(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    new_state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        log(f"[WARN] Missing keys: {len(missing)}（示例）{missing[:10]}")
    if unexpected:
        log(f"[WARN] Unexpected keys: {len(unexpected)}（示例）{unexpected[:10]}")
    model.eval().to(device)
    return model


# ---------- Subject mapping ----------
def build_subject_map_from_train(train_rows: List[dict]) -> Dict[str, int]:
    ids = sorted({str(r["subject_id"]) for r in train_rows})
    return {sid: i for i, sid in enumerate(ids)}


def load_subject_map_from_run(run_dir: Path) -> Dict[str, int]:
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


def _norm_identity(s: str) -> str: return str(s)
def _norm_lower(s: str) -> str: return str(s).lower()
def _norm_strip_prefixes(s: str) -> str:
    s = str(s)
    s = re.sub(r"^(sub-|subject[-_]?)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^[sS]", "", s)
    return s
def _norm_digits_only(s: str) -> str:
    digits = re.sub(r"\D+", "", str(s))
    return digits.lstrip("0") or "0"
def _norm_s_then_digits(s: str) -> str:
    return _norm_digits_only(_norm_strip_prefixes(s))

NORM_CANDIDATES = [
    ("identity", _norm_identity),
    ("lower", _norm_lower),
    ("strip_prefixes", _norm_strip_prefixes),
    ("digits_only", _norm_digits_only),
    ("strip_then_digits", _norm_s_then_digits),
]


def choose_best_normalizer(sub_map: Dict[str, int], test_rows: List[dict]) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
    test_sids = {str(r["subject_id"]) for r in test_rows}
    report = {"n_test_unique": len(test_sids), "candidates": []}
    best = ("identity", sub_map, 0, 0)
    for name, fn in NORM_CANDIDATES:
        norm_map: Dict[str, int] = {}
        collisions = 0
        for k, v in sub_map.items():
            nk = fn(k)
            if nk in norm_map and norm_map[nk] != v:
                collisions += 1
            norm_map[nk] = v
        inter = len({fn(s) for s in test_sids} & set(norm_map.keys()))
        report["candidates"].append({"name": name, "intersection": inter, "collisions": collisions})
        if inter > best[2] and collisions == 0:
            best = (name, norm_map, inter, collisions)
    report["picked"] = {"name": best[0], "intersection": best[2], "collisions": best[3]}
    return best[0], best[1], report


def annotate_subject_indices(rows: List[dict], sub_map: Dict[str, int],
                             strict_subjects: bool, note_prefix: str,
                             normalizer_name: str):
    fn = dict(NORM_CANDIDATES)[normalizer_name]
    filtered = []
    skipped_subjects = 0
    missing_artifacts = 0
    missing_examples = []
    missing_file_examples = []

    for r in rows:
        sid_raw = str(r["subject_id"])
        sid = fn(sid_raw)
        if strict_subjects and sid not in sub_map:
            if skipped_subjects < 5:
                missing_examples.append((sid_raw, sid))
            skipped_subjects += 1
            continue
        if not Path(r.get("sensor_coordinates_path", "")).exists() or not Path(r.get("meg_win_path", "")).exists():
            if missing_artifacts < 5:
                missing_file_examples.append({
                    "sensor_coordinates_path": r.get("sensor_coordinates_path", ""),
                    "meg_win_path": r.get("meg_win_path", "")
                })
            missing_artifacts += 1
            continue
        r2 = dict(r)
        r2["_subject_idx"] = sub_map.get(sid, 0 if not strict_subjects else -1)
        if strict_subjects and r2["_subject_idx"] < 0:
            if skipped_subjects < 5:
                missing_examples.append((sid_raw, sid))
            skipped_subjects += 1
            continue
        filtered.append(r2)

    note = (f"{note_prefix}; strict_subjects={bool(strict_subjects)}; "
            f"skipped_subjects={skipped_subjects}; missing_artifacts={missing_artifacts}; "
            f"normalizer={normalizer_name}")
    diag = {
        "skipped_subjects": skipped_subjects,
        "missing_artifacts": missing_artifacts,
        "examples_unmatched_subject": missing_examples,
        "examples_missing_files": missing_file_examples,
    }
    return filtered, note, sub_map, diag


# ---------- Encoder forward (robust) ----------
@torch.no_grad()
def encode_meg_batch(model, batch_rows: List[dict], device: str) -> torch.Tensor:
    megs, locs, sidx = [], [], []
    for r in batch_rows:
        x = np.load(r["meg_win_path"]).astype(np.float32)   # EEG/MEG window [C,T]
        x = ensure_meg_CxT(x)
        megs.append(torch.from_numpy(x))
        loc = np.load(r["sensor_coordinates_path"]).astype(np.float32)   # [C,3]
        locs.append(torch.from_numpy(loc))
        sidx.append(int(r["_subject_idx"]))

    meg = torch.stack(megs, 0).to(device)   # [B,C,T]
    loc = torch.stack(locs, 0).to(device)   # [B,C,3]
    sid = torch.tensor(sidx, dtype=torch.long, device=device)

    sig = inspect.signature(model.forward)
    params = set(sig.parameters.keys())

    # map common synonyms to what the model actually accepts
    def first_present(cands):
        for c in cands:
            if c in params:
                return c
        return None

    kw = {}
    name_meg = first_present(["meg_win", "eeg_win", "eeg", "x", "inputs", "signal"])
    if name_meg: kw[name_meg] = meg

    name_locs = first_present(["sensor_locs", "sensor_coords", "sensor_positions", "sensor_pos", "locs", "coords"])
    if name_locs: kw[name_locs] = loc

    name_sid = first_present(["subj_idx", "subject_idx", "subject_index", "subject_id", "sid"])
    if name_sid: kw[name_sid] = sid

    # Only pass optional keys if they exist in signature
    for opt_name, opt_val in [("meg_sent", None), ("meg_sent_mask", None)]:
        if opt_name in params:
            kw[opt_name] = opt_val

    try:
        if kw:
            y = model(**kw)
        else:
            # fallback positional attempts
            try:
                y = model(meg, loc, sid)
            except TypeError:
                try:
                    y = model(meg, sid)
                except TypeError:
                    y = model(meg)
    except TypeError as e:
        # final hard fallback: try common positional patterns
        try:
            y = model(meg, loc, sid)
        except TypeError:
            try:
                y = model(meg, sid)
            except TypeError:
                y = model(meg)

    # unify to [B,1024,T]
    if y.dim() == 3 and y.size(1) != AUDIO_D and y.size(2) == AUDIO_D:
        y = y.transpose(1, 2)
    assert y.dim() == 3 and y.size(1) == AUDIO_D, f"encoder must output [B,1024,T], got {tuple(y.shape)}"
    if y.size(2) != TARGET_T:
        y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)
    return y


# ---------- Audio pool & scoring ----------
@torch.no_grad()
def load_audio_pool_unique(test_rows: List[dict], device: str, dtype: torch.dtype):
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


def compute_logits(queries: torch.Tensor, pool: torch.Tensor, sim: str = "clip", tau: float = 0.0) -> torch.Tensor:
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


def evaluate(args):
    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if "16" in amp or "fp16" in amp else None)

    test_rows = read_jsonl(Path(args.test_manifest))
    train_rows = read_jsonl(Path(args.train_manifest))

    run_dir = Path(args.run_dir)
    subject_note = ""
    sub_map: Dict[str, int] = {}

    if args.subject_map in ("from_run", "auto"):
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

    norm_name, normed_sub_map, norm_report = choose_best_normalizer(sub_map, test_rows)
    log(f"[Subject-Norm] test_unique={norm_report['n_test_unique']}, "
        f"candidates={norm_report['candidates']}, picked={norm_report['picked']}")
    sub_map = normed_sub_map

    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "records" / "subjects_eval_used.json", "w", encoding="utf-8") as f:
        json.dump({"n_subjects": len(sub_map), "map": sub_map, "note": subject_note, "normalizer": norm_name,
                   "report": norm_report}, f, ensure_ascii=False, indent=2)
    log(f"Saved subject mapping -> {(run_dir / 'records' / 'subjects_eval_used.json').as_posix()}")

    filtered, subject_note, _, diag = annotate_subject_indices(
        test_rows, sub_map, args.strict_subjects, subject_note, normalizer_name=norm_name
    )
    if not filtered and args.strict_subjects:
        log("[WARN] Strict subject filtering removed all rows. Falling back to non-strict (unseen -> index 0).")
        filtered, subject_note, _, diag = annotate_subject_indices(
            test_rows, sub_map, False, subject_note + " (fallback_non_strict)", normalizer_name=norm_name
        )
    assert filtered, f"No valid test rows after filtering. diag={diag}"

    A, pool_ids = load_audio_pool_unique(filtered, device=device, dtype=torch.float32)
    pool_size = A.size(0)
    log(f"Candidate pool size (unique test audio windows): {pool_size}")
    assert A.dim() == 3 and A.shape[1:] == (AUDIO_D, TARGET_T), f"A shape wrong: {tuple(A.shape)}"

    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in filtered]

    ckpt_path = choose_ckpt_path(args)
    model = load_model_from_ckpt(ckpt_path, run_dir, device=device)

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

    B = max(1, int(args.batch_size))
    num_queries = 0
    with torch.no_grad():
        for start in tqdm(range(0, len(filtered), B), desc="Evaluating"):
            end = min(len(filtered), start + B)
            batch_rows = filtered[start:end]

            if autocast_dtype is None:
                Y = encode_meg_batch(model, batch_rows, device=device)     # [b,1024,360]
            else:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    Y = encode_meg_batch(model, batch_rows, device=device)

            logits = compute_logits(queries=Y, pool=A, sim=args.sim, tau=args.tau)  # [b,O]

            for j in range(logits.size(0)):
                global_j = start + j
                g = gt_index[global_j]
                s = logits[j]
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
                    }
                    preds_topk_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    preds_tsv_file.write(f"{global_j}\t{rank}\t{pool_ids[g]}\t{','.join(pred_cids)}\n")

            num_queries += (end - start)

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

    log("==== Retrieval Results (test full pool, NO re-ranking) ====")
    log(json.dumps(metrics, indent=2, ensure_ascii=False))
    log(f"Metrics saved to: {out_json.as_posix()}")
    log(f"Ranks saved to  : {out_ranks.as_posix()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="off", choices=["off", "bf16", "fp16", "16-mixed"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--strict_subjects", action="store_true",
                   help="严格要求 test 被试必须在训练映射内；匹配失败将自动尝试归一化并在必要时回退为非严格。")
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--save_ranks", type=str, default="")
    p.add_argument("--save_topk", type=int, default=0)
    p.add_argument("--sim", type=str, default="clip", choices=["clip", "cosine", "dot"])
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--subject_map", type=str, default="auto", choices=["auto", "from_run", "train"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    evaluate(args)
