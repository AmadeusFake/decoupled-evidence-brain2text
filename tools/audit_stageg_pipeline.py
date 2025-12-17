#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/audit_stageg_pipeline.py

目的：一次性审查 Stage-G(句级 TPP, 实体聚合) 训练/数据/模型是否对齐。
特性：
- 读取 train/val/test manifests，汇总规模与被试数量（通过 SubjectRegistry）
- 从 datamodule 抽取一个/多个 val batch，跑：
  1) 槽位对齐（student vs teacher）余弦热图 + Hungarian/贪心排列
  2) scorer 灵敏度曲线（softmax α ∈ {6,12,24}, length_norm ∈ {sqrt, none}, agg ∈ {softmax, max}）
  3) 实体维度规模统计与 ce_topM 建议（可通过 --ce_topM_cap 控制上限）
  4) logits 统计（pos-neg margin、分位数）
  5) 最小化“单步梯度探针”（是否可反传、哪些模块几乎无梯度）
- 支持加载已训练权重（--student_ckpt），默认随机初始化也能跑，但对齐结论仅在加载权重时有意义
- 新增：
  * --subject_mapping_path  从训练 run 里加载被试映射，避免 SubjectLayers 映射错位
  * --slot_perm              对学生句级 token 做固定重排（逗号分隔整数）
  * --ce_topM_cap            建议/探针中的 topM 上限可调

运行示例见本文档底部。
"""

from __future__ import annotations
import os, sys, math, json, argparse, time, csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# matplotlib 后端（无显示环境也能画图）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== 依赖项目内模块 ====
from models.meg_encoder_audio_T import UltimateMEGEncoderTPP
from train.meg_utils import (
    SubjectRegistry, MEGDataModule,
    batch_retrieval_metrics_entity_dedup,
    groupby_aggregate_logits_by_entity,
)

# ----------------- 通用工具 -----------------
def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_jsonl_count(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f: n += 1
    return n

def parse_slot_perm(s: str, L: int) -> Optional[List[int]]:
    if not s: return None
    s = s.strip()
    if s.lower() in ("none", "no", "false"): return None
    arr = [int(x) for x in s.split(",") if x.strip() != ""]
    if len(arr) != L or set(arr) != set(range(L)):
        raise ValueError(f"--slot_perm 长度或内容非法（需要 0..{L-1} 的一个排列）：{arr}")
    return arr

# ----------------- scorer（与训练版对齐的最小实现） -----------------
class ColBERTLikeScorer(nn.Module):
    """
    s_ij = sum_l Agg_m <q_l, k_m>, Agg ∈ {max, softmax(α)}
    - 对 q,k 先 L2 normalize
    - length_norm: none | Lq | sqrt
    - τ 固定为 0.07（审计用途，不涉及可学习/调度）
    - 支持按 O 维分块（chunk_O）以节省显存
    """
    def __init__(self, length_norm: str = "sqrt", agg: str = "softmax",
                 agg_alpha: float = 12.0, chunk_O: int = 0, tau: float = 0.07):
        super().__init__()
        self.length_norm = str(length_norm)
        self.agg = str(agg)
        self.agg_alpha = float(agg_alpha)
        self.chunk_O = int(chunk_O)
        self.logit_scale = math.exp(-math.log(max(1e-6, tau)))  # 1/tau

    def _aggregate(self, S: torch.Tensor,
                   q_mask: Optional[torch.Tensor],
                   k_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # S:[B,O,Lq,Lk]; mask True=pad
        if (q_mask is not None) or (k_mask is not None):
            B, O, Lq, Lk = S.shape
            qm = torch.zeros(B, Lq, dtype=torch.bool, device=S.device) if q_mask is None else q_mask
            km = torch.zeros(O, Lk, dtype=torch.bool, device=S.device) if k_mask is None else k_mask
            S = S.masked_fill(qm.view(B,1,Lq,1) | km.view(1,O,1,Lk), float("-inf"))

        if self.agg == "max":
            s = S.max(dim=-1).values.sum(dim=-1)  # [B,O]
        elif self.agg == "softmax":
            a = self.agg_alpha
            s = torch.logsumexp(a * S, dim=-1) / max(1e-6, a)
            s = s.sum(dim=-1)  # [B,O]
        else:
            raise ValueError(f"agg={self.agg}")
        s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
        return s

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                q_mask: Optional[torch.Tensor] = None,
                k_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q:[B,Lq,D]; k:[O,Lk,D]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        B, Lq, _ = q.shape
        O, Lk, _ = k.shape

        if self.chunk_O <= 0 or self.chunk_O >= O:
            S = torch.einsum("bid,ojd->boij", q, k)
            s = self._aggregate(S, q_mask, k_mask)
        else:
            outs = []
            step = self.chunk_O
            for o0 in range(0, O, step):
                o1 = min(o0 + step, O)
                kb = k[o0:o1]
                S = torch.einsum("bid,ojd->boij", q, kb)
                km = None if k_mask is None else k_mask[o0:o1]
                outs.append(self._aggregate(S, q_mask, km))
            s = torch.cat(outs, dim=1)

        # 长度归一
        if self.length_norm == "Lq":
            s = s / (float(Lq) + 1e-6)
        elif self.length_norm == "sqrt":
            s = s / (float(Lq * Lk) ** 0.5 + 1e-6)

        # 温度缩放
        s = s * self.logit_scale
        return s

# ----------------- Hungarian（优先用 SciPy，备选贪心） -----------------
def greedy_perm_for_max(sim: torch.Tensor) -> Tuple[List[int], float]:
    # sim:[L,L], 选择使总和最大的双射，贪心版（小规模 L=15 用作兜底）
    L = sim.size(0)
    used_r = set(); used_c = set()
    pairs = []
    vals = []
    sim_np = sim.detach().cpu().numpy()
    while len(pairs) < L:
        best = (-1e9, -1, -1)
        for i in range(L):
            if i in used_r: continue
            for j in range(L):
                if j in used_c: continue
                v = sim_np[i, j]
                if v > best[0]:
                    best = (v, i, j)
        v, i, j = best
        pairs.append((i, j))
        vals.append(v)
        used_r.add(i); used_c.add(j)
    perm = [-1]*L
    for i, j in pairs: perm[i] = j
    return perm, float(sum(vals) / L)

def hungarian_for_max(sim: torch.Tensor) -> Tuple[List[int], float, str]:
    """
    返回 (perm, mean_score, method)；perm 使得 sum(sim[i,perm[i]]) 最大。
    优先使用 SciPy；失败则回退贪心。
    """
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        S = sim.detach().cpu().numpy()
        # 最大化 => 最小化(-S)
        row_ind, col_ind = linear_sum_assignment(-S)
        perm = [-1] * S.shape[0]
        for r, c in zip(row_ind, col_ind):
            perm[r] = int(c)
        mean_sc = float(S[row_ind, col_ind].mean())
        return perm, mean_sc, "scipy"
    except Exception:
        perm, mean_sc = greedy_perm_for_max(sim)
        return perm, mean_sc, "greedy"

# ----------------- 权重加载 -----------------
def load_student_from_ckpt(enc: nn.Module, ckpt_path: str) -> Dict:
    """
    支持 Lightning .ckpt 或纯 state_dict；尝试前缀：
    'enc.', 'model.enc.', 'lit.enc.', 'encoder.', ''，不行则用 full dict(strict=False)
    返回 {'missing':[], 'unexpected':[], 'used_prefix': str}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd_full = ckpt.get("state_dict", ckpt)
    prefixes = ["enc.", "model.enc.", "lit.enc.", "encoder.", ""]
    last_err = None
    for pref in prefixes:
        sub = {k[len(pref):]: v for k, v in sd_full.items() if k.startswith(pref)}
        if not sub: continue
        try:
            missing, unexpected = enc.load_state_dict(sub, strict=False)
            return {"missing": list(missing), "unexpected": list(unexpected), "used_prefix": pref}
        except Exception as e:
            last_err = e
            continue
    try:
        missing, unexpected = enc.load_state_dict(sd_full, strict=False)
        return {"missing": list(missing), "unexpected": list(unexpected), "used_prefix": "<full>"}
    except Exception as e:
        raise RuntimeError(f"Load ckpt failed: {ckpt_path}; last_err={last_err or e}")

# ----------------- 统计与可视化 -----------------
@torch.no_grad()
def slot_alignment_probe(out_dir: Path,
                         g_tok: torch.Tensor, h_tok: torch.Tensor) -> Dict:
    """
    g_tok/h_tok: [B, L, D]（已规范化）
    计算：对每个槽位 l，取 batch 配对 cos(q[:,l], k[:,l]) 的均值 => diag；
         以及 LxL 的槽位间均值矩阵，然后跑 Hungarian/贪心。
    """
    B, L, D = g_tok.shape
    g = F.normalize(g_tok, dim=-1)
    h = F.normalize(h_tok, dim=-1)

    # diag mean
    diag = F.cosine_similarity(g, h, dim=-1).mean(dim=0)  # [L]
    diag_mean = float(diag.mean().item())

    # LxL 矩阵
    G = g.transpose(0,1).contiguous().view(L, B, D)  # [L,B,D]
    H = h.transpose(0,1).contiguous().view(L, B, D)
    g_exp = F.normalize(G, dim=-1)[:, None, :, :]   # [L,1,B,D]
    h_exp = F.normalize(H, dim=-1)[None, :, :, :]   # [1,L,B,D]
    sim = (g_exp * h_exp).sum(dim=-1).mean(dim=-1)  # [L,L]

    perm, hung_mean, method = hungarian_for_max(sim)
    # 保存热图
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(sim.detach().cpu().numpy(), interpolation="nearest", aspect="auto")
    ax.set_title("Slot Cosine (student vs teacher)")
    ax.set_xlabel("teacher slot"); ax.set_ylabel("student slot")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ensure_dir(out_dir)
    fig.savefig(out_dir / "cossim_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    ret = {
        "diag_mean": diag_mean,
        "hungarian_mean": float(hung_mean),
        "hungarian_perm": list(map(int, perm)),
        "method": method,
    }
    save_json(out_dir / "hungarian.json", ret)
    return ret

def scorer_eval_configs() -> List[Tuple[str, Dict]]:
    cfgs = []
    for a in [6.0, 12.0, 24.0]:
        cfgs.append((f"softmax_a{a:.1f}_lnsqrt", dict(agg="softmax", agg_alpha=a, length_norm="sqrt")))
        cfgs.append((f"softmax_a{a:.1f}_lnnone", dict(agg="softmax", agg_alpha=a, length_norm="none")))
    cfgs.append(("max_a0.0_lnsqrt", dict(agg="max", agg_alpha=0.0, length_norm="sqrt")))
    return cfgs

@torch.no_grad()
def scorer_sensitivity_probe(g_tok: torch.Tensor, h_tok: torch.Tensor,
                             h_mask: Optional[torch.Tensor],
                             global_keys: List[str],
                             out_path: Path) -> Dict:
    out = {}
    for name, cfg in scorer_eval_configs():
        scorer = ColBERTLikeScorer(length_norm=cfg["length_norm"],
                                   agg=cfg["agg"], agg_alpha=cfg["agg_alpha"], chunk_O=0, tau=0.07).to(g_tok.device)
        logits_raw = scorer(g_tok, h_tok, q_mask=None, k_mask=h_mask)
        metrics = batch_retrieval_metrics_entity_dedup(
            logits_raw, global_keys, ks=(1,5,10), agg="logsumexp"
        )
        out[name] = {k: float(v) for k, v in metrics.items()}
    save_json(out_path, out); return out

def logits_diagnostics(logits_raw: torch.Tensor, labels_col: torch.Tensor) -> Dict:
    B, O = logits_raw.shape
    idx = torch.arange(B, device=logits_raw.device)
    pos = logits_raw[idx, labels_col]                         # [B]
    mask = torch.ones_like(logits_raw, dtype=torch.bool)
    mask[idx, labels_col] = False
    neg = logits_raw[mask].view(B, O-1)                       # [B, O-1]
    neg_top = neg.max(dim=1).values                           # [B]
    margin = (pos - neg_top).mean().item()
    return {
        "pos_mean": float(pos.mean().item()),
        "neg_top_mean": float(neg_top.mean().item()),
        "pos_minus_negTop_mean": float(margin),
        "pos_q25_q50_q75": [float(x) for x in pos.quantile(torch.tensor([0.25,0.5,0.75], device=pos.device)).tolist()],
        "negTop_q25_q50_q75": [float(x) for x in neg_top.quantile(torch.tensor([0.25,0.5,0.75], device=pos.device)).tolist()],
    }

def ce_with_topM(s_full: torch.Tensor, labels_col: torch.Tensor, topM: int, margin: float) -> torch.Tensor:
    B, O = s_full.shape
    arange = torch.arange(B, device=s_full.device)
    pos = s_full[arange, labels_col].unsqueeze(1)  # [B,1]
    if margin > 0: pos = pos - margin
    if topM <= 0 or topM >= O - 1:
        s_copy = s_full.clone()
        s_copy[arange, labels_col] = pos.squeeze(1)
        return F.cross_entropy(s_copy, labels_col)
    neg = s_full.clone()
    neg[arange, labels_col] = -float("inf")
    topk_vals, _ = neg.topk(topM, dim=1)          # [B,M]
    sub = torch.cat([pos, topk_vals], dim=1)      # [B,1+M]
    target = torch.zeros(B, dtype=torch.long, device=s_full.device)
    return F.cross_entropy(sub, target)

def write_grads_csv(out_path: Path, model: nn.Module):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "grad_norm"])
        for n, p in model.named_parameters():
            if p.grad is None:
                g = 0.0
            else:
                g = float(p.grad.detach().data.norm(p=2).item())
            w.writerow([n, f"{g:.6e}"])

# ----------------- 主流程 -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--subject_namespace", default="")
    ap.add_argument("--subject_mapping_path", default="",
                    help="指向训练 run 的 records/subject_mapping.json；若提供且存在，将直接加载该映射。")
    # dataloader
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--sentences_per_batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action="store_true")
    # encoder cfg
    ap.add_argument("--in_channels", type=int, default=208)
    ap.add_argument("--spatial_channels", type=int, default=270)
    ap.add_argument("--fourier_k", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=320)
    ap.add_argument("--backbone_depth", type=int, default=5)
    ap.add_argument("--subject_layer_pos", type=str, choices=["early","late","none"], default="early")
    ap.add_argument("--audio_dim", type=int, default=2048)
    ap.add_argument("--tpp_slots", type=int, default=15)
    ap.add_argument("--slot_perm", type=str, default="",
                    help="可选，对学生句级 token 做固定重排；逗号分隔的 0..L-1 排列，如 '0,1,13,5,...'")
    # scorer default
    ap.add_argument("--length_norm", type=str, default="sqrt", choices=["none","Lq","sqrt"])
    ap.add_argument("--agg", type=str, default="softmax", choices=["max","softmax"])
    ap.add_argument("--agg_alpha", type=float, default=12.0)
    ap.add_argument("--ce_topM", type=int, default=128)  # 梯度探针时的 topM（保持原行为）
    ap.add_argument("--ce_topM_cap", type=int, default=512,
                    help="建议/统计中的 topM 上限（替代原脚本中的固定 128 上限）")
    ap.add_argument("--margin", type=float, default=0.05)
    # ckpt & out
    ap.add_argument("--student_ckpt", default="", help="path to .ckpt / state_dict for student encoder")
    ap.add_argument("--out_dir", default="runs/audit")
    ap.add_argument("--sample_batches", type=int, default=1, help="number of val batches to sample (>=1)")
    return ap.parse_args()

def build_dm(args, registry: SubjectRegistry) -> MEGDataModule:
    common = dict(
        train_manifest=str(Path(args.train_manifest)),
        val_manifest=str(Path(args.val_manifest)),
        test_manifest=str(Path(args.test_manifest)),
        registry=registry,
        ns_train=args.subject_namespace,
        ns_val=args.subject_namespace,
        ns_test=args.subject_namespace,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=False,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    dm = MEGDataModule(**common,
                       context_mode="sentence",
                       sentence_fast_io=True,
                       group_by_sentence=True,
                       sentences_per_batch=args.sentences_per_batch,
                       windows_per_sentence=1,
                       key_mode="audio")
    return dm

def entity_batch_stats_with_cap(logits_raw: torch.Tensor,
                                global_keys: List[str],
                                cap: int,
                                agg: str = "logsumexp") -> Dict:
    agg_logits, uniq_entities, query_ent_col = groupby_aggregate_logits_by_entity(
        logits_raw, global_keys, agg=agg, row_global_sentence_key=global_keys
    )
    B, Ne = agg_logits.shape
    rec = {
        "Ne": int(Ne),
        "B": int(B),
        "ce_topM_now": int(min(cap, max(1, B-1))),
        "ce_topM_recommended_min": int(min(cap, max(1, Ne-1))),
        "ce_topM_recommended_80pct": int(min(cap, max(1, int(0.8 * (Ne-1))))),
    }
    return rec

def main():
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = ensure_dir(Path(args.out_dir) / f"audit_{now_ts()}")

    # 1) Registry & datamodule（优先从 subject_mapping_path 复用训练时映射）
    train_p, val_p, test_p = Path(args.train_manifest), Path(args.val_manifest), Path(args.test_manifest)
    mapping_src = "built_from_manifests"
    if args.subject_mapping_path and Path(args.subject_mapping_path).exists():
        registry = SubjectRegistry.load(Path(args.subject_mapping_path))
        mapping_src = f"loaded:{args.subject_mapping_path}"
    else:
        registry = SubjectRegistry.build_from_manifests([(train_p, args.subject_namespace),
                                                         (val_p,   args.subject_namespace),
                                                         (test_p,  args.subject_namespace)])
    save_json(root / "manifest_summary.json", {
        "train": {"path": str(train_p), "n": read_jsonl_count(train_p)},
        "val":   {"path": str(val_p),   "n": read_jsonl_count(val_p)},
        "test":  {"path": str(test_p),  "n": read_jsonl_count(test_p)},
        "num_subjects_total": int(registry.num_subjects),
        "subject_mapping_source": mapping_src,
    })
    dm = build_dm(args, registry)
    dm.prepare_data(); dm.setup("fit")
    vloader = dm.val_dataloader()

    # 2) Student encoder
    enc_cfg: Dict = dict(
        in_channels=args.in_channels,
        n_subjects=registry.num_subjects,
        spatial_channels=args.spatial_channels,
        fourier_k=args.fourier_k,
        d_model=args.d_model,
        backbone_depth=args.backbone_depth,
        subject_layer_pos=args.subject_layer_pos,
        tpp_slots=args.tpp_slots,
        audio_dim=args.audio_dim,
        out_channels=1024,
        out_timesteps=None,
    )
    enc = UltimateMEGEncoderTPP(**enc_cfg).to(device)
    load_report = None
    if args.student_ckpt:
        load_report = load_student_from_ckpt(enc, args.student_ckpt)
        save_json(root / "student_ckpt_load_report.json", load_report or {"note": "no-ckpt"})

    # 3) 拉一个/几个 batch 审查
    batch_iter = iter(vloader)
    n_batches = max(1, int(args.sample_batches))

    batch = next(batch_iter)
    B = batch["meg_sent_full"].size(0)

    meg_full = batch["meg_sent_full"].to(device, non_blocking=True)
    meg_mask = batch.get("meg_sent_full_mask", None)
    meg_mask = meg_mask.to(device, non_blocking=True) if meg_mask is not None else None
    locs = batch["sensor_locs"].to(device, non_blocking=True)
    subj = batch["subject_idx"].to(device, non_blocking=True)
    g_tok = enc.encode_sentence_tokens(meg_full, meg_mask, locs, subj, normalize=True)  # [B,L,D]

    # 可选：对学生 token 做固定重排（验证“槽位错序”假设）
    L_slots = g_tok.size(1)
    slot_perm = parse_slot_perm(args.slot_perm, L_slots) if args.slot_perm else None
    if slot_perm is not None:
        idx = torch.tensor(slot_perm, device=g_tok.device, dtype=torch.long)
        g_tok = g_tok.index_select(dim=1, index=idx)

    # teacher tokens / mask
    if ("audio_tpp" not in batch) or ("audio_tpp_mask" not in batch):
        raise RuntimeError("batch 缺少 audio_tpp/audio_tpp_mask；请确保 sentence_fast_io=True。")
    h_tok = batch["audio_tpp"].to(device, non_blocking=True)          # [B,L,D]
    h_mask = batch["audio_tpp_mask"].to(device, non_blocking=True)    # [B,L] (True=pad)
    h_tok = F.normalize(h_tok, dim=-1)

    shapes = {
        "student_tokens": list(g_tok.shape),
        "teacher_tokens": list(h_tok.shape),
        "student_ckpt_loaded": bool(args.student_ckpt),
        "student_ckpt_report": load_report or {},
        "slot_perm_used": slot_perm if slot_perm is not None else [],
    }
    save_json(root / "batch_shapes.json", shapes)

    # 4) 槽位对齐
    slot_dir = ensure_dir(root / "slot_alignment")
    slot_info = slot_alignment_probe(slot_dir, g_tok, h_tok)

    # 5) scorer 灵敏度
    sens = scorer_sensitivity_probe(g_tok, h_tok, h_mask, batch["global_sentence_key"], root / "scorer_sensitivity.json")

    # 6) 实体维度统计 + ce_topM 建议（带 cap）
    scorer_def = ColBERTLikeScorer(length_norm=args.length_norm, agg=args.agg, agg_alpha=args.agg_alpha, chunk_O=0, tau=0.07).to(device)
    logits_raw = scorer_def(g_tok, h_tok, q_mask=None, k_mask=h_mask)  # [B,B]
    ent_stat = entity_batch_stats_with_cap(logits_raw, batch["global_sentence_key"], cap=int(args.ce_topM_cap), agg="logsumexp")
    save_json(root / "entity_batch.json", ent_stat)

    # 7) logits 统计（实体聚合后）
    agg_logits, uniq_entities, labels_col = groupby_aggregate_logits_by_entity(
        logits_raw, batch["global_sentence_key"], agg="logsumexp", row_global_sentence_key=batch["global_sentence_key"]
    )
    logi_stat = logits_diagnostics(agg_logits, labels_col)
    save_json(root / "logits_stats.json", logi_stat)

    # 8) 单步梯度探针（确认梯度是否有路 & 模块是否有效更新）
    enc.zero_grad(set_to_none=True)
    topM_probe = min(int(args.ce_topM_cap), max(1, agg_logits.size(1)-1))  # 用 cap
    loss = ce_with_topM(agg_logits, labels_col, topM=topM_probe, margin=float(args.margin))
    loss.backward()
    write_grads_csv(root / "grads.csv", enc)

    # 9) 汇总入口
    save_json(root / "index.json", {
        "manifest_summary": str((root / "manifest_summary.json").as_posix()),
        "batch_shapes": str((root / "batch_shapes.json").as_posix()),
        "slot_alignment": {
            "dir": str(slot_dir.as_posix()),
            "diag_mean": slot_info["diag_mean"],
            "hungarian_mean": slot_info["hungarian_mean"],
            "perm": slot_info["hungarian_perm"],
            "method": slot_info["method"],
            "png": str((slot_dir / "cossim_heatmap.png").as_posix()),
        },
        "scorer_sensitivity": str((root / "scorer_sensitivity.json").as_posix()),
        "entity_batch": str((root / "entity_batch.json").as_posix()),
        "logits_stats": str((root / "logits_stats.json").as_posix()),
        "grads_csv": str((root / "grads.csv").as_posix()),
        "student_ckpt_load_report": str((root / "student_ckpt_load_report.json").as_posix()) if args.student_ckpt else "",
    })

    # 控制台小结
    print(json.dumps({
        "slot_alignment": slot_info,
        "scorer_sensitivity_keys": list(sens.keys()),
        "entity_batch": ent_stat,
        "logits_stats": logi_stat,
        "paths": {
            "root": str(root),
            "manifest_summary": str(root / "manifest_summary.json"),
            "batch_shapes": str(root / "batch_shapes.json"),
            "slot_heatmap_png": str(slot_dir / "cossim_heatmap.png"),
            "scorer_sensitivity": str(root / "scorer_sensitivity.json"),
            "entity_batch": str(root / "entity_batch.json"),
            "grads_csv": str(root / "grads.csv"),
        }
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
