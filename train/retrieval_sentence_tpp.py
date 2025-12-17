#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.meg_encoder_audio_T import UltimateMEGEncoderTPP

def log(msg: str): print(msg, flush=True)

def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# ---------- 严格维度校正 ----------
def ensure_meg_CxT_strict(x: np.ndarray, expected_C: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D MEG, got {x.shape}")
    if x.shape[0] == expected_C:   # [C,T]
        return x
    if x.shape[1] == expected_C:   # [T,C] -> [C,T]
        return x.T
    raise ValueError(f"MEG shape {x.shape} incompatible with expected_C={expected_C}")

def ensure_coords_Cx3_strict(loc: np.ndarray, expected_C: int) -> np.ndarray:
    if loc.ndim != 2:
        raise ValueError(f"expect 2D coords, got {loc.shape}")
    if loc.shape == (expected_C, 3): return loc
    if loc.shape == (3, expected_C): return loc.T
    raise ValueError(f"coords shape {loc.shape} incompatible with expected_C={expected_C}")

def ensure_tpp_LxD_strict(a: np.ndarray, expected_D: int) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError(f"expect 2D TPP, got {a.shape}")
    if a.shape[1] == expected_D:   # [L,D]
        return a
    if a.shape[0] == expected_D:   # [D,L] -> [L,D]
        return a.T
    raise ValueError(f"TPP shape {a.shape} incompatible with expected_D={expected_D}")

# ---------- 其它工具 ----------
def sentence_cid(r: dict) -> str:
    a = str(r["original_audio_path"])
    s0 = float(r["global_segment_onset_in_audio_s"])
    s1 = float(r["global_segment_offset_in_audio_s"])
    return f"{a}::{s0:.3f}-{s1:.3f}"

def build_subject_map_from_train(rows: List[dict]) -> Dict[str, int]:
    ids = sorted({str(r["subject_id"]) for r in rows})
    return {sid: i for i, sid in enumerate(ids)}

def pad_time_BCT(megs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(megs); C = int(megs[0].size(0))
    Tmax = max(int(x.size(1)) for x in megs)
    dev, dt = megs[0].device, megs[0].dtype
    meg = torch.zeros(B, C, Tmax, device=dev, dtype=dt)
    msk = torch.ones(B, Tmax, device=dev, dtype=torch.bool)
    for i, x in enumerate(megs):
        T = int(x.size(1)); meg[i, :, :T] = x; msk[i, :T] = False
    return meg, msk

# ---------- enc 配置 / ckpt ----------
def load_enc_cfg(run_dir: Path) -> Dict[str, Any]:
    cfg_p = run_dir / "records" / "config.json"
    assert cfg_p.exists(), f"records/config.json not found under {run_dir}"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if isinstance(cfg.get("enc_cfg"), dict): return cfg["enc_cfg"]
    if isinstance(cfg.get("cfg", {}).get("enc_cfg"), dict): return cfg["cfg"]["enc_cfg"]
    raise RuntimeError("enc_cfg missing in records/config.json")

def load_train_args(run_dir: Path) -> Dict[str, Any]:
    cfg_p = run_dir / "records" / "config.json"
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "args" in cfg and isinstance(cfg["args"], dict):
        return cfg["args"]
    if "cfg" in cfg and "args" in cfg["cfg"] and isinstance(cfg["cfg"]["args"], dict):
        return cfg["cfg"]["args"]
    return {}

def choose_ckpt_path(run_dir: Path, use_best: bool, ckpt_path: str) -> Path:
    if use_best:
        p = run_dir / "records" / "best_checkpoint.txt"
        assert p.exists(), f"{p} not found"
        ck = p.read_text(encoding="utf-8").strip().splitlines()[0]
        log(f"[ckpt] using BEST from records: {ck}")
        ck = Path(ck)
        if not ck.is_absolute():
            ck = (run_dir / ck).resolve()
        assert ck.exists(), f"best ckpt not found: {ck}"
        return ck
    ck = Path(ckpt_path); assert ck.exists(), f"--ckpt_path not found: {ck}"
    return ck

def read_logit_scale_from_ckpt(ckpt: Path) -> Optional[float]:
    """返回 exp(logit_scale)；若不存在返回 None"""
    sd = torch.load(ckpt, map_location="cpu")
    state = sd.get("state_dict", sd)
    for k in ("scorer.logit_scale", "model.scorer.logit_scale"):
        v = state.get(k, None)
        if v is not None:
            try:
                return float(torch.exp(v).item())
            except Exception:
                return float(np.exp(float(v)))
    return None

@torch.no_grad()
def load_encoder_from_ckpt(ckpt: Path, enc_cfg: Dict, device: str) -> UltimateMEGEncoderTPP:
    model = UltimateMEGEncoderTPP(**enc_cfg).to(device).eval()
    sd = torch.load(ckpt, map_location="cpu")
    state = sd.get("state_dict", sd)
    enc_state = {}
    for k, v in state.items():
        if k.startswith("enc."): enc_state[k[4:]] = v
        elif k.startswith("model.enc."): enc_state[k[10:]] = v
    missing, unexpected = model.load_state_dict(enc_state, strict=False)
    if missing:   log(f"[warn] missing keys: {len(missing)} (e.g. {missing[:6]})")
    if unexpected: log(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:6]})")
    return model

# ---------- 打分（加入长度归一化 + 外部缩放） ----------
def late_interaction_logits(q_tok: torch.Tensor, k_tok: torch.Tensor,
                            k_mask: Optional[torch.Tensor],
                            length_norm: str="Lq",
                            scale_override: Optional[float]=None,
                            temp_mode_fallback: str="fixed",
                            tau_fallback: float=0.07) -> torch.Tensor:
    """
    - 与训练一致：先 max over key-slots，再 sum over query-slots，最后长度归一化；
    - 若提供 scale_override（来自 learnable logit_scale 的 exp值 或 1/tau），则用它；
    - 否则按 fallback 的 temp_mode/tau（通常不会影响排序，只为数值一致）。
    """
    q = F.normalize(q_tok, dim=-1)
    k = F.normalize(k_tok, dim=-1)
    S = torch.einsum("bid,ojd->boij", q, k)   # [B,O,Lq,Lk]
    if k_mask is not None:
        m = k_mask.view(1, k_tok.size(0), 1, k_tok.size(1))
        S = S.masked_fill(m, float("-inf"))
    s = S.max(dim=-1).values.sum(dim=-1)      # [B,O]

    # 长度归一化
    if length_norm == "Lq":
        s = s / (q.size(1) + 1e-6)
    elif length_norm == "sqrt":
        s = s / ((q.size(1) * k.size(1)) ** 0.5 + 1e-6)

    s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))

    if scale_override is not None:
        s = s * float(scale_override)
    else:
        if temp_mode_fallback == "fixed" and tau_fallback and tau_fallback > 0:
            s = s * (1.0 / float(tau_fallback))
        # temp_mode_fallback == "none" -> 不缩放

    return s

def pad_k_chunk(k_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    O = len(k_list); D = int(k_list[0].size(1))
    Lmax = max(int(t.size(0)) for t in k_list)
    dev, dt = k_list[0].device, k_list[0].dtype
    k_pad = torch.zeros(O, Lmax, D, device=dev, dtype=dt)
    m_pad = torch.ones(O, Lmax, device=dev, dtype=dev.type if hasattr(dev, "type") else torch.bool).bool()
    # 上一行为了兼容 CPU 设备构造 dtype；直接 bool 也可以
    m_pad = torch.ones(O, Lmax, device=dev, dtype=torch.bool)
    for i, t in enumerate(k_list):
        L = int(t.size(0)); k_pad[i, :L] = t; m_pad[i, :L] = False
    return k_pad, m_pad

# ---------- 编码 ----------
@torch.no_grad()
def encode_sentence_batch(
    model: UltimateMEGEncoderTPP,
    batch_rows: List[dict],
    sub_map: Dict[str, int],
    device: str,
    autocast_dtype: Optional[torch.dtype],
    expected_C: int,
) -> torch.Tensor:
    megs, locs, sids = [], [], []
    for r in batch_rows:
        mp = r["meg_sentence_full_path"]; lp = r["sensor_coordinates_path"]
        assert mp and Path(mp).exists(), f"missing meg_sentence_full_path: {mp}"
        assert lp and Path(lp).exists(), f"missing sensor_coordinates_path: {lp}"
        meg = ensure_meg_CxT_strict(np.load(mp, allow_pickle=False).astype(np.float32), expected_C)   # [C,T]
        loc = ensure_coords_Cx3_strict(np.load(lp, allow_pickle=False).astype(np.float32), expected_C)# [C,3]
        megs.append(torch.from_numpy(meg).to(device))
        locs.append(torch.from_numpy(loc).to(device))
        sids.append(sub_map[str(r["subject_id"])])

    locs_b = torch.stack(locs, 0)                                # [B,C,3]
    sids_b = torch.tensor(sids, dtype=torch.long, device=device) # [B]
    meg_bct, mask_bT = pad_time_BCT([t for t in megs])           # [B,C,Tmax], [B,Tmax]

    if autocast_dtype is None or device.startswith("cpu"):
        q = model.encode_sentence_tokens(
            meg_sent_full=meg_bct, meg_sent_full_mask=mask_bT,
            sensor_locs=locs_b, subj_idx=sids_b, normalize=False
        )
    else:
        with torch.amp.autocast("cuda", dtype=autocast_dtype):
            q = model.encode_sentence_tokens(
                meg_sent_full=meg_bct, meg_sent_full_mask=mask_bT,
                sensor_locs=locs_b, subj_idx=sids_b, normalize=False
            )
    return q  # [B,Lq,D]

# ---------- 主流程 ----------
def evaluate(args):
    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if ("16" in amp or "fp16" in amp) else None)

    # enc 配置（拿到 expected_C / expected_D）
    run_dir = Path(args.run_dir)
    enc_cfg = load_enc_cfg(run_dir)
    expected_C = int(enc_cfg.get("in_channels", 208))
    expected_D = int(enc_cfg.get("audio_dim", 2048))

    # 训练时的核心超参（与检索对齐）
    train_args = load_train_args(run_dir)
    length_norm = str(train_args.get("length_norm", args.length_norm))
    temp_mode_tr = str(train_args.get("temp_mode", "learnable"))  # learnable | fixed | none
    tau_tr = float(train_args.get("tau", args.tau))

    log(f"[config] length_norm={length_norm}, temp_mode(train)={temp_mode_tr}, tau(train)={tau_tr}")

    # 数据 + subject map
    test_rows_all = read_jsonl(Path(args.test_manifest))
    train_rows = read_jsonl(Path(args.train_manifest))
    sub_map = build_subject_map_from_train(train_rows)
    log(f"[subject] built from TRAIN: {len(sub_map)}")

    # 选句级查询（subject+sentence 唯一）
    picked: Dict[Tuple[str,str], dict] = {}
    miss = 0
    for r in test_rows_all:
        sid = str(r["subject_id"])
        if args.strict_subjects and sid not in sub_map: continue
        mp, lp = r.get("meg_sentence_full_path",""), r.get("sensor_coordinates_path","")
        if (not mp) or (not Path(mp).exists()) or (not lp) or (not Path(lp).exists()):
            miss += 1; continue
        key = (sid, sentence_cid(r))
        picked.setdefault(key, r)
    if miss: log(f"[note] skipped rows missing sentence MEG/coords: {miss}")
    queries = list(picked.values())
    assert queries, "no valid sentence queries after filtering"
    log(f"[queries] unique subject-sentence pairs: {len(queries)}")

    # 候选池（句级唯一）
    uniq_audio: Dict[str, str] = {}
    for r in queries:
        cid = sentence_cid(r)
        ap = r.get("audio_sentence_feature_path","")
        assert ap and Path(ap).exists(), f"audio_sentence_feature_path missing for cid={cid}"
        uniq_audio[cid] = ap
    pool_ids = list(uniq_audio.keys())
    log(f"[pool] sentence-level candidates: {len(pool_ids)}")

    # 载入候选（严格对齐到 [L,D]）
    k_list: List[torch.Tensor] = []
    for cid in tqdm(pool_ids, desc="Loading sentence-level audio TPP"):
        a = ensure_tpp_LxD_strict(np.load(uniq_audio[cid], allow_pickle=False).astype(np.float32), expected_D)
        k_list.append(torch.from_numpy(a).to(device))

    # GT 索引
    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[sentence_cid(r)] for r in queries]

    # 模型与温度（learnable 优先）
    ckpt = choose_ckpt_path(run_dir, args.use_best_ckpt, args.ckpt_path)
    model = load_encoder_from_ckpt(ckpt, enc_cfg, device=device)

    scale_override = None
    if temp_mode_tr == "learnable":
        logit_scale_exp = read_logit_scale_from_ckpt(ckpt)
        if logit_scale_exp is not None:
            scale_override = float(logit_scale_exp)          # s *= exp(logit_scale)
            log(f"[scaler] use learnable logit_scale: exp(logit_scale)={scale_override:.6f}")
        else:
            log("[scaler] learnable logit_scale not found in ckpt; fall back to 'none'")
    elif temp_mode_tr == "fixed":
        if tau_tr and tau_tr > 0:
            scale_override = 1.0 / float(tau_tr)
            log(f"[scaler] use fixed tau from train: tau={tau_tr:.6f} -> scale={scale_override:.6f}")
    else:
        log("[scaler] temp_mode(train)=none -> no scaling")

    # 评测
    K_list = [int(x) for x in args.topk.split(",")]
    hits = {K: 0 for K in K_list}; mrr_sum = 0.0; ranks: List[int] = []
    pool_size = len(pool_ids)

    with torch.no_grad():
        for s in tqdm(range(0, len(queries), args.query_batch), desc="Evaluating"):
            batch_rows = queries[s: s+args.query_batch]
            q_tok = encode_sentence_batch(model, batch_rows, sub_map, device,
                                          autocast_dtype=autocast_dtype, expected_C=expected_C)
            logits_all = []
            for p in range(0, pool_size, args.pool_chunk):
                k_chunk = k_list[p: p+args.pool_chunk]
                k_pad, k_mask = pad_k_chunk(k_chunk)
                s_bo = late_interaction_logits(
                    q_tok, k_pad, k_mask=k_mask,
                    length_norm=length_norm,
                    scale_override=scale_override,
                    temp_mode_fallback=args.temp_mode,  # 兜底
                    tau_fallback=args.tau
                )
                logits_all.append(s_bo)
            logits = torch.cat(logits_all, dim=1)  # [B, pool_size]

            for i in range(len(batch_rows)):
                gi = gt_index[s+i]
                sc = logits[i]; gt = sc[gi]
                rank = int((sc > gt).sum().item()) + 1
                ranks.append(rank); mrr_sum += 1.0 / rank
                for K in K_list: hits[K] += int(rank <= K)

    num_q = len(queries)
    metrics = {
        "num_queries": num_q,
        "pool_size": pool_size,
        "topk": {str(K): hits[K] / num_q for K in K_list},
        "mrr": mrr_sum / num_q,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "length_norm_used": length_norm,
        "train_temp_mode": temp_mode_tr,
        "train_tau": (tau_tr if temp_mode_tr == "fixed" else None),
        "scale_used": (scale_override if scale_override is not None else None),
        "fallback_temp_mode": args.temp_mode,
        "fallback_tau": (args.tau if args.temp_mode == "fixed" else None),
    }

    out_dir = run_dir / "results" / "retrieval_sentence"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / "ranks.txt", "w", encoding="utf-8") as f:
        for r in ranks: f.write(str(int(r)) + "\n")

    log("==== Sentence-level Retrieval (TPP) ====")
    log(json.dumps(metrics, indent=2, ensure_ascii=False))
    log(f"metrics -> {(out_dir / 'metrics.json').as_posix()}")
    log(f"ranks   -> {(out_dir / 'ranks.txt').as_posix()}")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off","bf16","fp16","16-mixed"])

    # 兜底（若 ckpt/records 无法提供，则退回到这里）
    p.add_argument("--temp_mode", type=str, default="fixed", choices=["fixed","none"])
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--length_norm", type=str, default="Lq", choices=["none","Lq","sqrt"])

    p.add_argument("--query_batch", type=int, default=64)
    p.add_argument("--pool_chunk", type=int, default=2048)
    p.add_argument("--topk", type=str, default="1,5,10")
    p.add_argument("--strict_subjects", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    evaluate(args)
