#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
viz_rerank_analysis.py — Visualizing how GCB reshapes logits (not features)
and producing a paper-ready t-SNE view of the *fixed* feature space.

Outputs:
  (1) cosine distance vs rank (base vs post) per-query
  (2) global GT rank shift histogram (post - base)
  (3) sentence-length–wise R@1 curves (base vs GCB + ΔR@1) with CIs
  (4) optional global logits distribution: pos/neg × base/post
  (5) t-SNE: features fixed, only ranking changes

Key flags:
  * --viz_logits_distribution_only
  * --skip_global_logits_distribution
  * --tsne_only / --heatmap_only
  * --skip_metrics
"""

import argparse, json, random, importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

SEED = 42
_rng = np.random.default_rng(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# ========================= Global Matplotlib style =========================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 19,
    "legend.fontsize": 13,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "lines.linewidth": 2.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ========================= Utils =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def to_numpy(t: torch.Tensor) -> np.ndarray:
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    return t.detach().to(torch.float32).cpu().numpy()

def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = x.norm(dim=1, p=2).clamp_min_(eps)
    return x / n.unsqueeze(1)

def savefig_pdf_svg(path_with_ext: Path, dpi: int = 200):
    stem = path_with_ext.with_suffix("")
    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(svg_path, dpi=dpi, bbox_inches="tight")
    return pdf_path, svg_path

# Wilson score interval for Bernoulli mean
def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p_hat = k / n
    z = 1.96  # ~95%
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    half = z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return float(lo), float(hi)

# ========================= PCA (fallback for t-SNE) =========================
def pca_rows_from_gram(X: np.ndarray, n: int = 2):
    X = X.astype(np.float64, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    M, D = X.shape
    G = (X @ X.T) / max(1.0, D - 1.0)
    w, U = np.linalg.eigh(G)
    idx = np.argsort(w)[::-1][:n]
    w = np.maximum(w[idx], 1e-12)
    U = U[:, idx]
    Z = U * np.sqrt(w)
    return Z.astype(np.float32)

# ========================= t-SNE =========================
def tsne_embed(X: np.ndarray, perplexity: float = 30.0, n_iter: int = 1000,
               metric: str = "cosine", lr: float = 200.0, random_state: int = 42) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=float(perplexity),
            n_iter=int(n_iter),
            learning_rate=float(lr),
            metric=metric,
            init="pca",
            random_state=random_state,
            verbose=0,
            n_jobs=1,
        )
        Z = tsne.fit_transform(X.astype(np.float32, copy=False))
    except Exception as e:
        print(f"[WARN] t-SNE unavailable ({e}); fallback to PCA(2).")
        Z = pca_rows_from_gram(X, n=2)
    return Z

# ========================= Plot helpers =========================
C_BG = "#D9D9D9"
C_BASE = "#2F6FED"
C_POST = "#E66100"
C_GT = "#111111"
C_GRID = "#BBBBBB"
C_NEG_BASE = "#F2CF5B"
C_POS_POST = "#1B9E77"
C_NEG_POST = "#D62728"

def _beautify_axes(ax):
    ax.grid(True, linestyle="--", alpha=0.22, linewidth=0.9, color=C_GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --------- simple 1D KDE for logits distribution ----------
def _kde_1d(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if x.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    std = np.std(x)
    if std <= 0:
        std = 1.0
    bw = 1.06 * std * x.size ** (-1.0 / 5.0)
    bw = max(bw, 1e-3)
    diffs = (grid[:, None] - x[None, :]) / bw
    kern = np.exp(-0.5 * diffs ** 2) / (np.sqrt(2.0 * np.pi) * bw)
    dens = kern.mean(axis=1)
    return dens.astype(np.float64)

def plot_global_logit_distributions(
    base_pos: np.ndarray,
    base_neg: np.ndarray,
    post_pos: np.ndarray,
    post_neg: np.ndarray,
    out_path: Path,
):
    all_vals = np.concatenate([base_pos, base_neg, post_pos, post_neg], axis=0)
    lo = float(np.percentile(all_vals, 0.5))
    hi = float(np.percentile(all_vals, 99.5))
    xs = np.linspace(lo, hi, 400, dtype=np.float64)

    d_bp = _kde_1d(base_pos, xs)
    d_bn = _kde_1d(base_neg, xs)
    d_pp = _kde_1d(post_pos, xs)
    d_pn = _kde_1d(post_neg, xs)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    ax.fill_between(xs, d_bp, color=C_BASE, alpha=0.25, label="Positive (base)")
    ax.fill_between(xs, d_bn, color=C_NEG_BASE, alpha=0.35, label="Negative (base)")

    ax.plot(xs, d_pp, color=C_POS_POST, linewidth=2.8, label="Positive (post)")
    ax.plot(xs, d_pn, color=C_NEG_POST, linewidth=2.8, label="Negative (post)")

    ax.set_title("Logits distribution: pos / neg (base vs post)", fontweight="bold")
    ax.set_xlabel("logit (confidence)", fontweight="bold")
    ax.set_ylabel("density", fontweight="bold")
    _beautify_axes(ax)
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    savefig_pdf_svg(out_path)
    plt.close(fig)

def plot_query_heatmaps(vec_base: np.ndarray, vec_post: np.ndarray, sent_ids: np.ndarray,
                        out_path: Path, title="Query logits by sentence-buckets"):
    order = np.lexsort((-vec_base, sent_ids))
    vb = vec_base[order]
    vp = vec_post[order]
    ss = sent_ids[order]
    boost = vp - vb

    fig, axs = plt.subplots(3, 1, figsize=(12.8, 6.8), sharex=True)

    axs[0].imshow(vb[None, :], aspect="auto", cmap="viridis")
    axs[0].set_ylabel("base", fontweight="bold")

    axs[1].imshow(vp[None, :], aspect="auto", cmap="viridis")
    axs[1].set_ylabel("post", fontweight="bold")

    axs[2].imshow(boost[None, :], aspect="auto", cmap="coolwarm")
    axs[2].set_ylabel("post-base", fontweight="bold")
    axs[2].set_xlabel("candidates sorted by [sentence,id]", fontweight="bold")

    change = np.where(np.diff(ss) != 0)[0]
    for ax in axs:
        for c in change:
            ax.axvline(c + 0.5, color="w", alpha=0.28, linewidth=1.0)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()

    savefig_pdf_svg(out_path)
    plt.close(fig)

def plot_tsne_grayscale(
    Z: np.ndarray,
    idx_roles: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "t-SNE of feature space (features fixed; GCB changes ranking)",
    subtitle: str = "",
):
    fig = plt.figure(figsize=(7.6, 6.9))
    ax = plt.gca()

    bg = idx_roles.get("bg", np.array([], dtype=int))
    if bg.size:
        ax.scatter(Z[bg, 0], Z[bg, 1], s=22, c=C_BG, marker="o",
                   linewidths=0, alpha=0.85, label="background")

    base = idx_roles.get("base_topk", np.array([], dtype=int))
    if base.size:
        ax.scatter(Z[base, 0], Z[base, 1], s=56, c=C_BASE, marker="o",
                   alpha=0.88, label="TopK (base)")

    post = idx_roles.get("post_topk", np.array([], dtype=int))
    if post.size:
        ax.scatter(Z[post, 0], Z[post, 1], s=62, c=C_POST, marker="^",
                   alpha=0.88, label="TopK (post)")

    gt = idx_roles.get("gt", None)
    if gt is not None:
        ax.scatter(Z[gt, 0], Z[gt, 1], s=220, c=C_GT, marker="*",
                   edgecolors=C_GT, linewidths=1.2, label="GT")

    q = idx_roles.get("query", None)
    if q is not None:
        ax.scatter(Z[q, 0], Z[q, 1], s=220, c=C_GT, marker="x",
                   linewidths=3.2, label="Query")

    t1b = idx_roles.get("top1_base", None)
    if t1b is not None:
        ax.scatter(Z[t1b, 0], Z[t1b, 1], s=300, facecolors="none",
                   edgecolors=C_BASE, marker="o", linewidths=3.2)
        ax.annotate("Top1(base)", (Z[t1b, 0], Z[t1b, 1]),
                    xytext=(10, -10), textcoords="offset points",
                    fontsize=13, color=C_BASE, fontweight="bold")

    t1p = idx_roles.get("top1_post", None)
    if t1p is not None:
        ax.scatter(Z[t1p, 0], Z[t1p, 1], s=320, facecolors="none",
                   edgecolors=C_POST, marker="^", linewidths=3.2)
        ax.annotate("Top1(post)", (Z[t1p, 0], Z[t1p, 1]),
                    xytext=(10, 12), textcoords="offset points",
                    fontsize=13, color=C_POST, fontweight="bold")

    ax.set_title(title + (("\n" + subtitle) if subtitle else ""), fontweight="bold")
    ax.set_xlabel("t-SNE Dim-1", fontweight="bold")
    ax.set_ylabel("t-SNE Dim-2", fontweight="bold")
    _beautify_axes(ax)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="best", frameon=False)

    plt.tight_layout()
    savefig_pdf_svg(out_path)
    plt.close(fig)

def plot_rank_vs_distance(
    dist_full: np.ndarray,
    rank_base_all: np.ndarray,
    rank_post_all: np.ndarray,
    idx_show: np.ndarray,
    gt_cid: int,
    top1_base: int,
    top1_post: int,
    out_path: Path,
    title: str,
    subtitle: str = "",
):
    x = dist_full[idx_show]
    yb = rank_base_all[idx_show]
    yp = rank_post_all[idx_show]

    fig, ax = plt.subplots(figsize=(8.2, 6.7))

    ax.scatter(x, yb, s=52, marker="o", c=C_BASE, alpha=0.70, linewidths=0,
               label="base rank")
    ax.scatter(x, yp, s=58, marker="^", c=C_POST, alpha=0.70, linewidths=0,
               label="post rank")

    def _mark(cid, text, edge_color):
        if cid in idx_show:
            j = np.where(idx_show == cid)[0][0]
            ax.scatter([x[j]], [yb[j]], s=260, marker="o", facecolors="none",
                       edgecolors=edge_color, linewidths=3.2)
            ax.annotate(text, (x[j], yb[j]), xytext=(10, 10),
                        textcoords="offset points", fontsize=13,
                        fontweight="bold", color=edge_color)

    _mark(gt_cid, "GT", C_GT)
    _mark(top1_base, "Top1(base)", C_BASE)
    _mark(top1_post, "Top1(post)", C_POST)

    ax.set_title(title + (("\n" + subtitle) if subtitle else ""), fontweight="bold")
    ax.set_xlabel("cosine distance to query  (1 - cos)", fontweight="bold")
    ax.set_ylabel("rank (1 is best)", fontweight="bold")
    ax.set_ylim(bottom=1)
    ax.invert_yaxis()
    _beautify_axes(ax)
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    savefig_pdf_svg(out_path)
    plt.close(fig)

def plot_rank_shift_histogram(
    r_base_list: np.ndarray,
    r_post_list: np.ndarray,
    out_path: Path,
    title: str,
):
    shift = r_post_list - r_base_list

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    clip = np.clip(shift, -200, 200)
    ax.hist(clip, bins=61, color="#7A7A7A", alpha=0.85)

    ax.axvline(0, linestyle="--", linewidth=2.8, alpha=0.7, color=C_GT)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("GT rank shift = (post - base)    [<0 means improved]", fontweight="bold")
    ax.set_ylabel("count", fontweight="bold")
    _beautify_axes(ax)
    ax.grid(axis="y", linestyle="--", alpha=0.22, linewidth=0.9, color=C_GRID)

    m = float(np.median(shift))
    mu = float(np.mean(shift))
    ax.text(0.02, 0.98, f"mean={mu:.2f}\nmedian={m:.2f}\nN={shift.size}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13, fontweight="bold", color=C_GT)

    plt.tight_layout()
    savefig_pdf_svg(out_path)
    plt.close(fig)

def plot_r1_vs_length(
    lengths_unique: np.ndarray,
    n_per_len: np.ndarray,
    r1_base: np.ndarray,
    r1_post: np.ndarray,
    r1_base_lo: np.ndarray,
    r1_base_hi: np.ndarray,
    r1_post_lo: np.ndarray,
    r1_post_hi: np.ndarray,
    delta: np.ndarray,
    delta_lo: np.ndarray,
    delta_hi: np.ndarray,
    out_path: Path,
    title: str,
):
    """
    Sentence length vs R@1 (base & post) + ΔR@1, all with CI bands.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.6, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.4]},
    )

    # 上面：base / post R@1 曲线 + 阴影 CI
    ax1.fill_between(
        lengths_unique, r1_base_lo, r1_base_hi,
        color=C_BASE, alpha=0.18, linewidth=0,
    )
    ax1.fill_between(
        lengths_unique, r1_post_lo, r1_post_hi,
        color=C_POST, alpha=0.18, linewidth=0,
    )
    ax1.plot(
        lengths_unique, r1_base, "-o",
        color=C_BASE, label="R@1 (base)", markersize=5,
    )
    ax1.plot(
        lengths_unique, r1_post, "-^",
        color=C_POST, label="R@1 (GCB)", markersize=5,
    )
    ax1.set_ylabel("R@1", fontweight="bold")
    ax1.set_title(title, fontweight="bold")
    ax1.set_ylim(
        bottom=0.0,
        top=min(1.0, max(1.0, float(np.nanmax(r1_post_hi)) + 0.05)),
    )
    _beautify_axes(ax1)
    ax1.legend(loc="lower right", frameon=False)

    # 下面：ΔR@1 曲线 + CI 阴影
    ax2.fill_between(
        lengths_unique, delta_lo, delta_hi,
        color=C_POST, alpha=0.18, linewidth=0,
    )
    ax2.axhline(0.0, linestyle="--", color=C_GT, linewidth=2.0, alpha=0.7)
    ax2.plot(
        lengths_unique, delta, "-s",
        color=C_POST, markersize=5,
    )
    ax2.set_xlabel("sentence length (number of windows)", fontweight="bold")
    ax2.set_ylabel("ΔR@1 (GCB - base)", fontweight="bold")
    _beautify_axes(ax2)

    plt.tight_layout()
    savefig_pdf_svg(out_path)
    plt.close(fig)

# ========================= Main =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_py", required=True, type=str,
                    help="Path to eval/retrieval_window_vote.py")
    ap.add_argument("--test_manifest", required=True, type=str)
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--ckpt_path", type=str, default="")
    ap.add_argument("--use_best_ckpt", action="store_true")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", type=str, default="bf16",
                    choices=["off", "bf16", "fp16", "16-mixed"])
    ap.add_argument("--use_ckpt_logit_scale", action="store_true")

    # Stages
    ap.add_argument("--no_qccp", action="store_true")
    ap.add_argument("--no_windowvote", action="store_true")
    ap.add_argument("--no_gcb", action="store_true")
    ap.add_argument("--gcb_topk", type=int, default=128)
    ap.add_argument("--gcb_q", type=float, default=0.95)
    ap.add_argument("--gcb_top_m", type=int, default=3)
    ap.add_argument("--gcb_norm", type=str, default="bucket_sqrt",
                    choices=["bucket_sqrt"])
    ap.add_argument("--gcb_topS", type=int, default=3)
    ap.add_argument("--gcb_gamma", type=float, default=0.7)

    ap.add_argument("--out_dir", type=str, default="./viz_tsne_out")
    ap.add_argument("--feature_topk", type=int, default=40,
                    help="TopK for base/post overlays")
    ap.add_argument("--bg_limit", type=int, default=100,
                    help="random background negatives")
    ap.add_argument("--pick_query_mode", type=str, default="improved",
                    choices=["improved", "random", "hard"])
    ap.add_argument("--min_windows", type=int, default=10,
                    help="prefer sentences with >= this many windows")
    ap.add_argument("--n_example_queries", type=int, default=3)

    ap.add_argument("--select_post_top1_only", action="store_true")
    ap.add_argument("--select_base_wrong_only", action="store_true")

    # t-SNE params
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_n_iter", type=int, default=1000)
    ap.add_argument("--tsne_lr", type=float, default=200.0)
    ap.add_argument("--tsne_metric", type=str, default="cosine",
                    choices=["cosine", "euclidean"])

    # Modes / decoupling flags
    ap.add_argument("--viz_logits_distribution_only", action="store_true",
                    help="Only compute and plot global logits distribution "
                         "(base vs post), then exit.")
    ap.add_argument("--skip_global_logits_distribution", action="store_true",
                    help="Skip expensive global pos/neg logits "
                         "distribution collection/plot.")
    ap.add_argument("--tsne_only", action="store_true",
                    help="Only run per-query t-SNE "
                         "(skip heatmaps + skip global logits dist + skip metrics).")
    ap.add_argument("--heatmap_only", action="store_true",
                    help="Only run per-query heatmaps "
                         "(skip t-SNE + skip global logits dist + skip metrics).")
    ap.add_argument("--skip_metrics", action="store_true",
                    help="Skip global metric computation (Pass 2).")

    # Quantitative plot toggles（兼容旧脚本，虽然现在只用 rank_vs_distance）
    ap.add_argument("--no_rank_distance_plot", action="store_true",
                    help="Disable per-query cosine-distance vs rank plot.")
    ap.add_argument("--no_rank_logit_plot", action="store_true",
                    help="(deprecated) kept for CLI compatibility.")
    ap.add_argument("--no_top1_distance_plot", action="store_true",
                    help="(deprecated) kept for CLI compatibility.")
    ap.add_argument("--no_rank_shift_hist", action="store_true",
                    help="Disable global GT rank shift histogram.")

    args = ap.parse_args()

    if args.tsne_only and args.heatmap_only:
        raise SystemExit(
            "[ERROR] --tsne_only and --heatmap_only cannot be used together."
        )

    if args.tsne_only or args.heatmap_only:
        args.skip_global_logits_distribution = True
        args.skip_metrics = True

    # dynamic import eval script
    spec = importlib.util.spec_from_file_location("rvote", args.eval_py)
    rv = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, f"Cannot load module from {args.eval_py}"
    spec.loader.exec_module(rv)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = args.device
    test_rows = rv.read_jsonl(Path(args.test_manifest))
    A, pool_ids, candidate_rows = rv.load_audio_pool_unique(
        test_rows, device=device, dtype=torch.float32
    )
    canon2idx, alias2idx, cand_sent_idx = rv.build_sentence_index_with_alias(
        candidate_rows
    )
    cand_sent_idx = torch.tensor(cand_sent_idx, dtype=torch.long, device=device)

    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[rv.content_id_of(r)] for r in test_rows]

    run_dir = Path(args.run_dir)
    subj_map = rv.read_subject_mapping_from_records(run_dir)

    # model
    if args.use_best_ckpt or args.ckpt_path:
        ckpt_path = rv.choose_ckpt_path(args) if args.use_best_ckpt \
                    else Path(args.ckpt_path)
    else:
        ckpt_path = rv.choose_ckpt_path(
            argparse.Namespace(use_best_ckpt=True,
                               run_dir=args.run_dir,
                               ckpt_path="")
        )
    model, meta = rv.load_model_from_ckpt(ckpt_path, run_dir, device=device)
    scale = meta.get("logit_scale_exp", None) if args.use_ckpt_logit_scale else None

    # group by sentence
    def sent_key_for_group(r: dict) -> Tuple[str, str]:
        als = rv.sentence_aliases(r)
        return als[0] if als else ("unknown", rv.content_id_of(r))

    sent2idx: Dict[Tuple[str, str], List[int]] = {}
    for i, r in enumerate(test_rows):
        k = sent_key_for_group(r)
        sent2idx.setdefault(k, []).append(i)
    groups = list(sent2idx.values())

    qid_to_group = {}
    qid_to_len = {}
    for gi, qids in enumerate(groups):
        L = len(qids)
        for q in qids:
            qid_to_group[q] = gi
            qid_to_len[q] = L

    # AMP
    amp = args.amp.lower()
    if amp == "bf16":
        autocast_dtype = torch.bfloat16
    elif "16" in amp or "fp16" in amp:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    @torch.no_grad()
    def encode_group(q_indices: List[int]):
        rows = [test_rows[i] for i in q_indices]
        if autocast_dtype is None:
            Y = rv.encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        else:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                Y = rv.encode_meg_batch(model, rows, device=device,
                                        subj_map=subj_map)

        base = rv.compute_logits_clip(Y, A, scale=scale).to(torch.float32)
        post = base.clone()

        if not args.no_qccp:
            times = rv.window_centers(test_rows, q_indices).to(device=base.device)
            post = rv.qccp_rerank_group(
                post, times_b=times,
                hops=1, alpha=0.6, topk=128, q_quantile=0.9
            )
        if not args.no_windowvote:
            post = rv.window_vote_rerank(
                post, cand_sent_idx_o=cand_sent_idx,
                topk_window=128, q_quantile=0.95,
                sent_top_m=3, sent_topS=3,
                sent_norm="bucket_sqrt", gamma=0.7
            )
        if not args.no_gcb:
            buckets = rv._precompute_sentence_buckets(cand_sent_idx)
            post = rv.gcb_apply_to_group(
                post, cand_sent_idx, buckets,
                topk=args.gcb_topk, q_quantile=args.gcb_q,
                top_m=args.gcb_top_m, sent_norm=args.gcb_norm,
                topS=args.gcb_topS, gamma=args.gcb_gamma
            )
        return base, post, Y

    # ---------- Pass 1: scan ranks, optionally collect global logits ----------
    delta_rank_pairs = []

    run_global_logits = bool(args.viz_logits_distribution_only) or \
                        (not args.skip_global_logits_distribution)
    if run_global_logits:
        all_base_pos: List[float] = []
        all_base_neg: List[float] = []
        all_post_pos: List[float] = []
        all_post_neg: List[float] = []
        max_neg_per_query = 2000
    else:
        all_base_pos = all_base_neg = all_post_pos = all_post_neg = None
        max_neg_per_query = 0

    for g in tqdm(range(len(groups)), desc="Pass 1: scan ranks"):
        q_idx = groups[g]
        base, post, _ = encode_group(q_idx)
        for j_local, qid in enumerate(q_idx):
            gti = gt_index[qid]
            r_base = int((base[j_local] > base[j_local, gti]).sum().item()) + 1
            r_post = int((post[j_local] > post[j_local, gti]).sum().item()) + 1
            delta_rank_pairs.append(
                {"qid": qid, "dr": (r_post - r_base),
                 "r_base": r_base, "r_post": r_post}
            )

            if run_global_logits:
                vb = to_numpy(base[j_local])
                vp = to_numpy(post[j_local])
                all_base_pos.append(float(vb[gti]))
                all_post_pos.append(float(vp[gti]))

                neg_indices = np.arange(vb.shape[0], dtype=np.int64)
                neg_indices = neg_indices[neg_indices != gti]
                if neg_indices.size > max_neg_per_query:
                    sel = _rng.choice(
                        neg_indices, size=max_neg_per_query, replace=False
                    )
                else:
                    sel = neg_indices
                all_base_neg.extend(vb[sel].astype(np.float32).tolist())
                all_post_neg.extend(vp[sel].astype(np.float32).tolist())

    # ---------- Global logits distribution (optional) ----------
    if run_global_logits:
        all_base_pos_arr = np.asarray(all_base_pos, dtype=np.float32)
        all_base_neg_arr = np.asarray(all_base_neg, dtype=np.float32)
        all_post_pos_arr = np.asarray(all_post_pos, dtype=np.float32)
        all_post_neg_arr = np.asarray(all_post_neg, dtype=np.float32)

        np.savez(
            out_dir / "global_logits_distribution_data.npz",
            base_pos=all_base_pos_arr,
            base_neg=all_base_neg_arr,
            post_pos=all_post_pos_arr,
            post_neg=all_post_neg_arr,
        )
        print("[OK] Wrote global logits distribution data to:",
              (out_dir / "global_logits_distribution_data.npz").as_posix())

        plot_global_logit_distributions(
            base_pos=all_base_pos_arr,
            base_neg=all_base_neg_arr,
            post_pos=all_post_pos_arr,
            post_neg=all_post_neg_arr,
            out_path=out_dir / "global_logits_distribution.pdf",
        )
    else:
        print("[INFO] Skipping global logits distribution "
              "(--skip_global_logits_distribution).")

    # ---------- Global GT rank-shift histogram + R@1 vs length ----------
    if not args.no_rank_shift_hist:
        r_base_arr = np.asarray(
            [d["r_base"] for d in delta_rank_pairs], dtype=np.int32
        )
        r_post_arr = np.asarray(
            [d["r_post"] for d in delta_rank_pairs], dtype=np.int32
        )
        plot_rank_shift_histogram(
            r_base_arr, r_post_arr,
            out_path=out_dir / "gt_rank_shift_hist.pdf",
            title="GT rank shift histogram (post - base)",
        )
        np.savez(
            out_dir / "gt_rank_shift_data.npz",
            r_base=r_base_arr, r_post=r_post_arr,
            shift=(r_post_arr - r_base_arr),
        )

        # sentence-level R@1 vs length (+ CIs)
        lengths = []
        succ_b = []
        succ_p = []
        for rec in delta_rank_pairs:
            qid = rec["qid"]
            L = qid_to_len.get(qid, 0)
            if L <= 0:
                continue
            lengths.append(L)
            succ_b.append(int(rec["r_base"] == 1))
            succ_p.append(int(rec["r_post"] == 1))

        if len(lengths) > 0:
            lengths = np.asarray(lengths, dtype=np.int32)
            succ_b = np.asarray(succ_b, dtype=np.int32)
            succ_p = np.asarray(succ_p, dtype=np.int32)

            uniq_L = np.unique(lengths)
            n_list = []
            r1_base = []
            r1_post = []
            r1_base_lo = []
            r1_base_hi = []
            r1_post_lo = []
            r1_post_hi = []
            delta = []
            delta_lo = []
            delta_hi = []

            B = 2000  # bootstrap rounds for ΔR@1

            for L in uniq_L:
                m = (lengths == L)
                n = int(m.sum())
                n_list.append(n)
                if n == 0:
                    r1_base.append(np.nan)
                    r1_post.append(np.nan)
                    r1_base_lo.append(np.nan)
                    r1_base_hi.append(np.nan)
                    r1_post_lo.append(np.nan)
                    r1_post_hi.append(np.nan)
                    delta.append(np.nan)
                    delta_lo.append(np.nan)
                    delta_hi.append(np.nan)
                    continue

                sb = succ_b[m].astype(np.float64)
                sp = succ_p[m].astype(np.float64)
                k_b = int(sb.sum())
                k_p = int(sp.sum())
                rb = float(sb.mean())
                rp = float(sp.mean())
                r1_base.append(rb)
                r1_post.append(rp)

                lo_b, hi_b = wilson_interval(k_b, n)
                lo_p, hi_p = wilson_interval(k_p, n)
                r1_base_lo.append(lo_b)
                r1_base_hi.append(hi_b)
                r1_post_lo.append(lo_p)
                r1_post_hi.append(hi_p)

                d = sp - sb  # paired difference in { -1,0,1 }
                delta_mean = float(d.mean())
                delta.append(delta_mean)

                if n > 1:
                    boots = np.empty(B, dtype=np.float64)
                    for b in range(B):
                        idx = _rng.integers(0, n, size=n)
                        boots[b] = d[idx].mean()
                    lo_d, hi_d = np.percentile(boots, [2.5, 97.5])
                else:
                    lo_d = hi_d = delta_mean

                delta_lo.append(float(lo_d))
                delta_hi.append(float(hi_d))

            uniq_L = uniq_L.astype(np.float32)
            n_arr = np.asarray(n_list, dtype=np.int32)
            r1_base = np.asarray(r1_base, dtype=np.float32)
            r1_post = np.asarray(r1_post, dtype=np.float32)
            r1_base_lo = np.asarray(r1_base_lo, dtype=np.float32)
            r1_base_hi = np.asarray(r1_base_hi, dtype=np.float32)
            r1_post_lo = np.asarray(r1_post_lo, dtype=np.float32)
            r1_post_hi = np.asarray(r1_post_hi, dtype=np.float32)
            delta = np.asarray(delta, dtype=np.float32)
            delta_lo = np.asarray(delta_lo, dtype=np.float32)
            delta_hi = np.asarray(delta_hi, dtype=np.float32)

            plot_r1_vs_length(
                lengths_unique=uniq_L,
                n_per_len=n_arr,
                r1_base=r1_base,
                r1_post=r1_post,
                r1_base_lo=r1_base_lo,
                r1_base_hi=r1_base_hi,
                r1_post_lo=r1_post_lo,
                r1_post_hi=r1_post_hi,
                delta=delta,
                delta_lo=delta_lo,
                delta_hi=delta_hi,
                out_path=out_dir / "r1_vs_sentlen.pdf",
                title="Sentence-level R@1 vs length (base vs GCB)",
            )

            np.savez(
                out_dir / "r1_vs_sentlen_data.npz",
                length=uniq_L,
                n=n_arr,
                r1_base=r1_base,
                r1_post=r1_post,
                delta=delta,
                r1_base_lo=r1_base_lo,
                r1_base_hi=r1_base_hi,
                r1_post_lo=r1_post_lo,
                r1_post_hi=r1_post_hi,
                delta_lo=delta_lo,
                delta_hi=delta_hi,
            )

    if args.viz_logits_distribution_only:
        print("[INFO] --viz_logits_distribution_only is set; exiting now.")
        return

    # ---------- Pass 2: compute metrics (optional) ----------
    if not args.skip_metrics:
        topk_list = [1, 5, 10]
        recalls_base = {k: 0 for k in topk_list}
        recalls_post = {k: 0 for k in topk_list}
        mrr_b = 0.0
        mrr_p = 0.0
        for g in tqdm(range(len(groups)), desc="Pass 2: compute metrics"):
            q_idx = groups[g]
            base, post, _ = encode_group(q_idx)
            for j_local, qid in enumerate(q_idx):
                gti = gt_index[qid]
                r_base = int((base[j_local] > base[j_local, gti]).sum().item()) + 1
                r_post = int((post[j_local] > post[j_local, gti]).sum().item()) + 1
                mrr_b += 1.0 / r_base
                mrr_p += 1.0 / r_post
                for k in topk_list:
                    recalls_base[k] += int(r_base <= k)
                    recalls_post[k] += int(r_post <= k)
        Nq = len(test_rows)
        save_json(
            {
                "base": {
                    "recall_at": {
                        str(k): recalls_base[k] / Nq for k in topk_list
                    },
                    "mrr": mrr_b / Nq,
                },
                "post": {
                    "recall_at": {
                        str(k): recalls_post[k] / Nq for k in topk_list
                    },
                    "mrr": mrr_p / Nq,
                },
                "flags": {
                    "qccp": not args.no_qccp,
                    "windowvote": not args.no_windowvote,
                    "gcb": not args.no_gcb,
                    "use_ckpt_logit_scale": bool(args.use_ckpt_logit_scale),
                },
            },
            out_dir / "metrics_before_after.json",
        )
    else:
        print("[INFO] Skipping metrics (--skip_metrics).")

    # ---------- Select example queries ----------
    candidates = []
    for rec in delta_rank_pairs:
        qid, dr, rb, rp = rec["qid"], rec["dr"], rec["r_base"], rec["r_post"]
        if qid_to_len.get(qid, 0) < args.min_windows:
            continue
        if args.select_post_top1_only and rp != 1:
            continue
        if args.select_base_wrong_only and rb == 1:
            continue
        improved_flag = (dr < 0)
        candidates.append(
            (qid, improved_flag, dr, rb, rp, qid_to_len.get(qid, 0))
        )

    candidates.sort(key=lambda t: (-int(t[1]), -t[5], t[2], -t[3]))
    example_queries = [t[0] for t in candidates][:args.n_example_queries]

    if len(example_queries) < args.n_example_queries:
        loosen = []
        for rec in delta_rank_pairs:
            qid, dr, rb, rp = rec["qid"], rec["dr"], rec["r_base"], rec["r_post"]
            if args.select_post_top1_only and rp != 1:
                continue
            if args.select_base_wrong_only and rb == 1:
                continue
            improved_flag = (dr < 0)
            loosen.append(
                (qid, improved_flag, dr, rb, rp, qid_to_len.get(qid, 0))
            )
        loosen.sort(key=lambda t: (-int(t[1]), -t[5], t[2], -t[3]))
        for qid, *_ in loosen:
            if qid not in example_queries:
                example_queries.append(qid)
                if len(example_queries) >= args.n_example_queries:
                    break

    if len(example_queries) == 0:
        print("[WARN] No example queries selected. "
              "Try relaxing --min_windows or selection flags.")
        return

    # ---------- Per-query figs ----------
    do_heatmap = (not args.tsne_only)
    do_tsne = (not args.heatmap_only)

    cand_sent_idx_np = to_numpy(cand_sent_idx)

    for qid in tqdm(example_queries, desc="Per-query figs"):
        gi = qid_to_group[qid]
        q_idx = groups[gi]
        base, post, Y = encode_group(q_idx)
        j_local = q_idx.index(qid)

        vb = to_numpy(base[j_local])
        vp = to_numpy(post[j_local])
        gti = int(gt_index[qid])

        r_base = int((base[j_local] > base[j_local, gti]).sum().item()) + 1
        r_post = int((post[j_local] > post[j_local, gti]).sum().item()) + 1
        top1_base = int(np.argmax(vb))
        top1_post = int(np.argmax(vp))

        order_b = np.argsort(-vb)
        order_p = np.argsort(-vp)
        rank_base_all = np.empty_like(order_b)
        rank_post_all = np.empty_like(order_p)
        rank_base_all[order_b] = np.arange(1, vb.shape[0] + 1)
        rank_post_all[order_p] = np.arange(1, vp.shape[0] + 1)

        if do_heatmap:
            heatmap_pdf = out_dir / f"q{qid:06d}_logits_by_bucket.pdf"
            plot_query_heatmaps(
                vb, vp, cand_sent_idx_np,
                out_path=heatmap_pdf,
                title=f"Query {qid} logits by sentence-buckets",
            )
            np.savez(
                out_dir / f"q{qid:06d}_heatmap_data.npz",
                vec_base=vb, vec_post=vp, sent_ids=cand_sent_idx_np,
            )

        K = int(max(1, min(args.feature_topk, vb.shape[0])))
        topk_base = np.argpartition(-vb, K - 1)[:K]
        topk_post = np.argpartition(-vp, K - 1)[:K]
        core = set(map(int, topk_base.tolist())) | \
               set(map(int, topk_post.tolist())) | \
               {int(gti), int(top1_base), int(top1_post)}
        all_idx = np.arange(vb.shape[0], dtype=int)
        mask = np.ones(vb.shape[0], dtype=bool)
        mask[list(core)] = False
        bg_pool = all_idx[mask]
        if bg_pool.size > 0 and args.bg_limit > 0:
            n_take = min(int(args.bg_limit), bg_pool.size)
            bg_idx = _rng.choice(bg_pool, size=n_take, replace=False)
        else:
            bg_idx = np.array([], dtype=int)

        idx_all = np.concatenate(
            [bg_idx, np.array(sorted(core), dtype=int)], axis=0
        )

        with torch.no_grad():
            sub = A.index_select(
                0, torch.from_numpy(idx_all).to(A.device)
            ).to(torch.float32)
            sub = sub.reshape(sub.size(0), -1)
            sub = l2_normalize_rows(sub).cpu().numpy().astype(np.float32)

            Yq = Y[j_local:j_local + 1].to(torch.float32).reshape(1, -1)
            Yq = l2_normalize_rows(Yq).cpu().numpy().astype(np.float32)

        cos = (sub @ Yq.T).reshape(-1)
        dist_sub = 1.0 - cos

        dist_full = np.full((vb.shape[0],), np.nan, dtype=np.float32)
        dist_full[idx_all] = dist_sub.astype(np.float32)

        if not args.no_rank_distance_plot:
            plot_rank_vs_distance(
                dist_full=dist_full,
                rank_base_all=rank_base_all,
                rank_post_all=rank_post_all,
                idx_show=idx_all,
                gt_cid=gti,
                top1_base=top1_base,
                top1_post=top1_post,
                out_path=out_dir / f"q{qid:06d}_rank_vs_cosdist.pdf",
                title="Rigorous: cosine distance vs rank (base vs post)",
                subtitle=f"Query {qid} — GT rank base={r_base}, post={r_post}",
            )

        # 保存严谨数据方便后面自己重新画
        np.savez(
            out_dir / f"q{qid:06d}_distance_rank_data.npz",
            idx_show=idx_all.astype(np.int64),
            dist_sub=dist_sub.astype(np.float32),
            top1_base=np.int64(top1_base),
            top1_post=np.int64(top1_post),
            gt_idx=np.int64(gti),
            rank_base_all=rank_base_all.astype(np.int32),
            rank_post_all=rank_post_all.astype(np.int32),
            vb=vb.astype(np.float32),
            vp=vp.astype(np.float32),
        )

        if do_tsne:
            X_tsne = np.vstack([sub, Yq])
            Z = tsne_embed(
                X_tsne,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_n_iter,
                metric=args.tsne_metric,
                lr=args.tsne_lr,
                random_state=SEED,
            )

            Nbg = bg_idx.size
            sorted_core = np.array(sorted(core), dtype=int)
            pos_in_core = {cid: i for i, cid in enumerate(sorted_core)}
            Ncore = sorted_core.size

            roles = {
                "bg": np.arange(0, Nbg, dtype=int),
                "base_topk": (
                    Nbg
                    + np.array(
                        [pos_in_core[i] for i in topk_base
                         if i in pos_in_core],
                        dtype=int,
                    )
                ),
                "post_topk": (
                    Nbg
                    + np.array(
                        [pos_in_core[i] for i in topk_post
                         if i in pos_in_core],
                        dtype=int,
                    )
                ),
                "gt": int(Nbg + pos_in_core[gti]),
                "query": int(Nbg + Ncore),
            }
            if top1_base in pos_in_core:
                roles["top1_base"] = int(Nbg + pos_in_core[top1_base])
            if top1_post in pos_in_core:
                roles["top1_post"] = int(Nbg + pos_in_core[top1_post])

            plot_tsne_grayscale(
                Z=Z,
                idx_roles=roles,
                out_path=out_dir / f"q{qid:06d}_tsne.pdf",
                title="t-SNE (intuition only): features fixed; GCB changes ranking",
                subtitle=f"Query {qid} — GT rank: base={r_base}, post={r_post}",
            )

            # 保存完整的角色索引，方便后续用 Excel 画图
            np.savez(
                out_dir / f"q{qid:06d}_tsne_data.npz",
                Z=Z.astype(np.float32),
                idx_bg=roles["bg"].astype(np.int64),
                idx_base_topk=roles["base_topk"].astype(np.int64),
                idx_post_topk=roles["post_topk"].astype(np.int64),
                idx_gt=np.int64(roles["gt"]),
                idx_query=np.int64(roles["query"]),
                idx_top1_base=np.int64(roles["top1_base"])
                    if "top1_base" in roles else np.int64(-1),
                idx_top1_post=np.int64(roles["top1_post"])
                    if "top1_post" in roles else np.int64(-1),
            )

        save_json(
            {
                "query_index": int(qid),
                "gt_pool_index": int(gti),
                "gt_content_id": str(pool_ids[gti]),
                "base_top1_pool_index": int(top1_base),
                "post_top1_pool_index": int(top1_post),
                "gt_rank_base": int(r_base),
                "gt_rank_post": int(r_post),
                "note": "Embeddings are fixed; post changes ranking via "
                        "contextual prior in logit space.",
            },
            out_dir / f"q{qid:06d}_summary.json",
        )

    print("[OK] Done. Outputs in:", out_dir.as_posix())

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
