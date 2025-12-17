#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sentence-level retrieval (GLOBAL branch, Stage-G style)

- Query:  per-sample full-sentence MEG -> enc._global_from_sentence -> g
- Pool:   dedup TEXT sentence vectors (one per unique sentence)
- Scoring: CLIP-style (L2 on selectable sides) with temperature (ckpt logit_scale if available)
- Dim align: use ckpt's meg_to_text.weight if d_meg != text_dim; else require d_meg==text_dim
- No input time resampling: right-pad per batch and pass meg_sent_full_mask

Outputs:
  <sent_run_dir>/results/sent_eval_textpool.json
"""

import argparse, json, re, inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from models.meg_encoder import UltimateMEGEncoder

# ----------------- Utils -----------------
def log(x: str): print(x, flush=True)

def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

_SENT_RE = re.compile(r'([^/\\]+)::(\d+)-(\d+)', re.U)

def global_sent_key(r: dict) -> str:
    tpath = r.get("text_sentence_feature_path", "") or r.get("audio_sentence_feature_path", "")
    if tpath:
        m = _SENT_RE.search(str(tpath))
        if m:
            return f"{m.group(1)}::{m.group(2)}-{m.group(3)}"
    audio_path = r.get("original_audio_path", r.get("audio_feature_path", ""))
    stem = Path(audio_path).stem
    s_on = r.get("sentence_onset_in_audio_s", r.get("sent_onset_in_audio_s"))
    s_off = r.get("sentence_offset_in_audio_s", r.get("sent_offset_in_audio_s"))
    if (s_on is not None) and (s_off is not None) and stem:
        return f"{stem}::{int(round(float(s_on)*1000))}-{int(round(float(s_off)*1000))}"
    sid = r.get("sentence_id")
    if sid is not None and stem:
        return f"{stem}::SID@{sid}"
    return f"UNK::{hash(json.dumps(r, sort_keys=True, ensure_ascii=False)) & 0xffffffff:x}"

def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"MEG array must be 2D, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

# ----------------- Load encoder / ckpt -----------------
def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "records" / "config.json"
    if not p.exists(): return {}
    cfg = json.loads(p.read_text("utf-8"))
    return cfg.get("model_cfg", cfg.get("enc_cfg", {}))

def best_ckpt(run_dir: Path) -> Path:
    t = run_dir / "records" / "best_checkpoint.txt"
    s = t.read_text("utf-8").strip().splitlines()[0]
    p = Path(s) if s.startswith("/") else (run_dir / s)
    assert p.exists(), f"best ckpt not found: {p}"
    return p

def load_encoder(run_dir: Path, device: str):
    cfg = load_cfg_from_records(run_dir)
    ckpt_path = best_ckpt(run_dir)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    # 不强行覆盖 out_timesteps，保持与训练一致（Stage-G）
    if not cfg:
        hp = ckpt.get("hyper_parameters", {})
        cfg = hp.get("model_cfg", hp.get("enc_cfg", {}))

    enc = UltimateMEGEncoder(**cfg).to(device).eval()
    state = ckpt.get("state_dict", ckpt)
    enc_state = {(k.replace("model.", "", 1) if k.startswith("model.") else k): v for k, v in state.items()}
    miss, unexp = enc.load_state_dict(enc_state, strict=False)
    if miss:  log(f"[{run_dir.name}] missing keys: {len(miss)}")
    if unexp: log(f"[{run_dir.name}] unexpected keys: {len(unexp)}")
    return enc, ckpt

def build_meg2text_from_ckpt(ckpt: dict, g_dim: int, text_dim: int, device: str) -> Optional[nn.Module]:
    sd = ckpt.get("state_dict", ckpt)
    W = None
    for k, v in sd.items():
        if k.endswith("meg_to_text.weight") and isinstance(v, torch.Tensor):
            W = v.float(); break
    if W is not None:
        proj = nn.Linear(W.size(1), W.size(0), bias=False)
        with torch.no_grad():
            proj.weight.copy_(W)
        proj.to(device).eval()
        for p in proj.parameters(): p.requires_grad = False
        log(f"[DimAlign] loaded meg_to_text.weight: {W.size(1)} -> {W.size(0)}")
        if W.size(1) != g_dim or W.size(0) != text_dim:
            log(f"[WARN] ckpt meg_to_text shape {W.size(1)}->{W.size(0)} but g_dim={g_dim}, text_dim={text_dim}")
        return proj
    # 无权重：要求 g_dim==text_dim
    if g_dim != text_dim:
        raise RuntimeError(f"g_dim({g_dim}) != text_dim({text_dim}) and ckpt has no meg_to_text.weight.")
    log(f"[DimAlign] g_dim == text_dim == {text_dim}, use Identity.")
    return None  # Identity

def get_logit_scale(ckpt: dict, tau_arg: float = 0.0) -> float:
    # 优先 ckpt 的 learnable logit_scale
    sd = ckpt.get("state_dict", {})
    ls = None
    for k in ("logit_scale", "model.logit_scale"):
        if k in sd and torch.is_tensor(sd[k]):
            ls = float(sd[k].mean().item()); break
    if ls is not None:
        scale = float(np.exp(ls))
        return max(1e-3, min(1e3, scale))
    # 退化到 tau
    tau = float(tau_arg)
    if tau > 0:
        return float(1.0 / tau)
    return 1.0

# ----------------- Pools / queries -----------------
def build_text_pool(rows: List[dict]) -> Tuple[List[str], Dict[str,str]]:
    groups: DefaultDict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[global_sent_key(r)].append(r)
    uniq_keys, key2txt = [], {}
    for k, lst in groups.items():
        p = ""
        for r in lst:
            q = r.get("text_sentence_feature_path", "")
            if q and Path(q).exists():
                p = q; break
        if p:
            uniq_keys.append(k); key2txt[k] = p
    uniq_keys.sort()
    return uniq_keys, key2txt

def filter_query_rows(rows: List[dict], subj_map: Dict[str,int], valid_keys: set) -> List[dict]:
    out = []
    for r in rows:
        if not Path(r.get("sensor_coordinates_path","")).exists(): continue
        p = r.get("meg_sentence_full_path")
        if not (p and Path(p).exists()): continue
        k = global_sent_key(r)
        if k not in valid_keys: continue
        rr = dict(r)
        rr["_subject_idx"] = subj_map.get(str(r.get("subject_id")), 0)
        rr["_sent_key"] = k
        out.append(rr)
    return out

# ----------------- Batch encode sentence MEG (no resampling; pad+mask) -----------------
@torch.no_grad()
def encode_sentence_batch(enc, batch_rows: List[dict], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    in_ch = int(getattr(enc, "in_channels", 208))
    xs, locs, sidx = [], [], []
    T_list = []
    for r in batch_rows:
        x_np = ensure_meg_CxT(np.load(r["meg_sentence_full_path"]))
        C, T = x_np.shape if x_np.shape[0] <= x_np.shape[1] else x_np.T.shape
        # 对通道对齐到 in_ch（仅裁/零填，不动 T）
        x_t = torch.from_numpy(ensure_meg_CxT(x_np)).float()    # [C,T]
        if x_t.size(0) > in_ch:
            x_t = x_t[:in_ch, :]
        elif x_t.size(0) < in_ch:
            pad = torch.zeros(in_ch - x_t.size(0), x_t.size(1), dtype=x_t.dtype)
            x_t = torch.cat([x_t, pad], dim=0)
        xs.append(x_t)
        # 传感器
        l_np = np.load(r["sensor_coordinates_path"])
        l = torch.from_numpy(l_np).float()
        if l.dim() == 1 and l.numel() == in_ch*3:
            l = l.view(in_ch, 3)
        if l.dim() == 2 and l.size(0) != in_ch:
            if l.size(0) > in_ch: l = l[:in_ch]
            else:
                padl = torch.zeros(in_ch - l.size(0), l.size(1), dtype=l.dtype)
                l = torch.cat([l, padl], dim=0)
        locs.append(l)
        sidx.append(int(r.get("_subject_idx", 0)))
        T_list.append(x_t.size(1))

    B = len(xs)
    T_max = max(T_list)
    # 右侧 pad 到同长，并构建 mask（pad 位置 True）
    x_pad = torch.zeros(B, in_ch, T_max, dtype=torch.float32)
    m_pad = torch.zeros(B, T_max, dtype=torch.bool)
    for i, x in enumerate(xs):
        T = x.size(1)
        x_pad[i, :, :T] = x
        if T < T_max:
            m_pad[i, T:] = True
    locs_t = torch.stack(locs, 0)
    sidx_t = torch.tensor(sidx, dtype=torch.long)

    x_pad   = x_pad.to(device)
    m_pad   = m_pad.to(device)
    locs_t  = locs_t.to(device)
    sidx_t  = sidx_t.to(device)

    # 调 enc 的 global 分支
    _, g = enc._global_from_sentence(
        meg_sent_full=x_pad,
        meg_sent_full_mask=m_pad,
        sensor_locs=locs_t,
        subj_idx=sidx_t,
    )  # [B, d_meg]
    return g, m_pad  # mask 仅用于调试/一致性，后续不再使用

# ----------------- Scoring / metrics -----------------
def apply_l2_side(g: torch.Tensor, h: torch.Tensor, side: str) -> Tuple[torch.Tensor, torch.Tensor]:
    s = side.lower()
    if s in ("both","meg"): g = F.normalize(g, dim=-1)
    if s in ("both","text"): h = F.normalize(h, dim=-1)
    return g, h

def compute_metrics(ranks: List[int], topk: List[int]) -> Dict[str,Any]:
    n = len(ranks)
    recalls = {str(k): float(sum(1 for r in ranks if r<=k))/n for k in topk}
    mrr = float(sum(1.0/r for r in ranks))/n
    import numpy as np
    return {
        "num_queries": n,
        "recall_at": recalls,
        "mrr": mrr,
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser("Sentence-level retrieval (GLOBAL, Stage-G style)")
    ap.add_argument("--train_manifest", required=True, type=str)
    ap.add_argument("--test_manifest",  required=True, type=str)
    ap.add_argument("--sent_run_dir",   required=True, type=str)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp",    type=str, default="bf16", choices=["off","bf16","fp16"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--topk",   type=str, default="1,5,10")
    ap.add_argument("--save_json_dir", type=str, default="")
    ap.add_argument("--l2_side", type=str, default="both", choices=["both","meg","text","none"])
    ap.add_argument("--tau", type=float, default=0.0, help="fallback temp if ckpt has no logit_scale")
    args = ap.parse_args()

    if torch.cuda.is_available():
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    device = args.device
    topk = [int(x) for x in args.topk.split(",")]

    # data & subject map
    test_rows_raw  = read_jsonl(Path(args.test_manifest))
    train_rows_raw = read_jsonl(Path(args.train_manifest))
    subj_ids = sorted({str(r["subject_id"]) for r in train_rows_raw if "subject_id" in r})
    subj_map = {sid:i for i,sid in enumerate(subj_ids)}

    # text pool (dedupe by global sentence key)
    uniq_keys, key2txt = build_text_pool(test_rows_raw)
    assert uniq_keys, "No unique sentences with available text vectors."
    log(f"[Info] unique sentences (TEXT pool): {len(uniq_keys)}")
    key2idx = {k:i for i,k in enumerate(uniq_keys)}

    # load text vectors
    text_vecs = []
    for k in uniq_keys:
        v = np.load(key2txt[k]).reshape(-1).astype(np.float32, copy=False)
        text_vecs.append(torch.from_numpy(v))
    text_pool = torch.stack(text_vecs, 0).to(device)   # [U, text_dim]
    text_dim = int(text_pool.size(1))

    # queries (per-sample full sentence MEG)
    query_rows = filter_query_rows(test_rows_raw, subj_map, set(uniq_keys))
    assert query_rows, "No valid per-sample sentence queries after filtering."
    # representative sample per sentence (first occurrence)
    rep_indices: Dict[str, int] = {}
    for i, r in enumerate(query_rows):
        k = r["_sent_key"]
        if k not in rep_indices:
            rep_indices[k] = i
    rep_list = sorted(rep_indices.values())

    # encoder & ckpt
    enc, ckpt = load_encoder(Path(args.sent_run_dir), device)

    # encode queries in batches (no resampling; pad+mask)
    B = args.batch_size
    g_list: List[torch.Tensor] = []
    for i in tqdm(range(0, len(query_rows), B), desc="Encoding sentence MEG (GLOBAL)"):
        batch_rows = query_rows[i:i+B]
        g, _ = encode_sentence_batch(enc, batch_rows, device)  # [b, d_meg]
        g_list.append(g)
    g_all = torch.cat(g_list, dim=0)  # [Q, d_meg]
    g_dim = int(g_all.size(1))

    # dim align: MEG -> text_dim using ckpt's meg_to_text if needed
    proj = build_meg2text_from_ckpt(ckpt, g_dim, text_dim, device)
    if isinstance(proj, nn.Module):
        g_all = proj(g_all)

    # CLIP scoring: L2 side + temperature
    scale = get_logit_scale(ckpt, tau_arg=args.tau)
    # optional L2 per side
    g_all, text_pool_n = apply_l2_side(g_all, text_pool, args.l2_side)
    logits = (g_all.float() @ text_pool_n.float().t()) * float(scale)  # [Q,U]

    # GT per-sample
    gt_all = torch.tensor([key2idx[r["_sent_key"]] for r in query_rows], dtype=torch.long, device=device)

    # per-sample metrics
    ranks_all: List[int] = []
    for i in range(logits.size(0)):
        s = logits[i]
        gt = gt_all[i].item()
        rank = int((s > s[gt]).sum().item()) + 1
        ranks_all.append(rank)
    metrics_per_sample = compute_metrics(ranks_all, topk)

    # per-sentence-unique (first sample of each sentence)
    ranks_rep: List[int] = []
    for i in rep_list:
        s = logits[i]
        gt = gt_all[i].item()
        rank = int((s > s[gt]).sum().item()) + 1
        ranks_rep.append(rank)
    metrics_per_sentuniq = compute_metrics(ranks_rep, topk)

    # output
    summary = {
        "settings": {
            "sent_run_dir": args.sent_run_dir,
            "device": args.device,
            "amp": args.amp,
            "batch_size": args.batch_size,
            "text_dim": text_dim,
            "g_dim_after_proj": int(g_all.size(1)),
            "l2_side": args.l2_side,
            "temperature_scale": scale,
            "query": "per-sample sentence MEG via GLOBAL",
            "pool": "dedup TEXT sentence vectors",
            "time_handling": "no-resample; batch right-pad + mask",
        },
        "sentence_eval": {
            "per_sample": metrics_per_sample,
            "per_sentence_unique": metrics_per_sentuniq
        }
    }
    print("\n==== SENTENCE RESULTS (GLOBAL, Stage-G style) ====")
    print(json.dumps(summary, indent=2))

    out_dir = Path(args.save_json_dir) if args.save_json_dir else (Path(args.sent_run_dir)/"results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sent_eval_textpool.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[Saved] {out_path}")

if __name__ == "__main__":
    import numpy as np  # needed for get_logit_scale
    main()
