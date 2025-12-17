#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global negative analysis for Whisper → E5 sentence embeddings.

Compute global negative similarity statistics:
- Positive: self-similarity (sanity only, always 1.0)
- Negatives: all cross-sentence pairs (deduplicated) + top-1 nearest negative

Outputs:
  summary.json
  top1_negative.tsv
  hist_top1_negative.png
  hist_all_negative_sampled.png
  all_negative_sampled.tsv (optional)
"""

import argparse, json
from pathlib import Path
from typing import List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- I/O -------------------- #

def load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------- Embeddings -------------------- #

def load_embeddings(index_jsonl: Path):
    rows = load_jsonl(index_jsonl)
    sids, texts, vecs = [], [], []
    miss = 0

    for r in rows:
        sid = str(r.get("sentence_id") or r.get("sentence_key") or "")
        if not sid:
            continue

        vpath = r.get("text_sentence_feature_path") or r.get("feature_path") or ""
        if not vpath:
            continue

        vf = Path(vpath)
        if not vf.exists():
            miss += 1
            continue

        try:
            v = np.load(vf.as_posix(), mmap_mode="r")
            if v.ndim != 1:
                continue
            vecs.append(v.astype(np.float32))
            sids.append(sid)
            texts.append(r.get("text_used", r.get("text_raw", "")) or "")
        except Exception:
            continue

    if len(vecs) == 0:
        return [], [], np.zeros((0, 0), dtype=np.float32)

    X = np.stack(vecs, axis=0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)  # L2 normalize
    return sids, texts, X

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    return X @ X.T  # cosine similarity (L2-normalized)

# -------------------- Stats / plots -------------------- #

def plot_hist(vals: np.ndarray, title: str, out_png: Path, bins: int = 60):
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(v, bins=bins)
    plt.title(title)
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png.as_posix(), dpi=150)
    plt.close()

def stats_of(x: np.ndarray):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "std":  float(np.std(x)),
        "p50":  float(np.percentile(x, 50)),
        "p90":  float(np.percentile(x, 90)),
        "p95":  float(np.percentile(x, 95)),
        "p99":  float(np.percentile(x, 99)),
        "min":  float(np.min(x)),
        "max":  float(np.max(x)),
        "n":    int(x.size),
    }

# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index_jsonl",
        required=True,
        help=".../precomputed_text_sentence_features/*_whisper/sent_index/{split}.jsonl",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--pair_samples",
        type=int,
        default=50000,
        help="Max number of negative pairs to sample",
    )
    ap.add_argument(
        "--dump_sampled_pairs_tsv",
        action="store_true",
        help="Write all_negative_sampled.tsv",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sids, texts, X = load_embeddings(Path(args.index_jsonl))
    if X.shape[0] < 2:
        print(f"[WARN] Not enough sentences (<2): {args.index_jsonl}")
        return

    N, D = X.shape

    S = cosine_sim_matrix(X)
    np.fill_diagonal(S, -1.0)  # exclude self

    # Top-1 negative per sentence
    top1_idx = np.argmax(S, axis=1)
    top1_sim = S[np.arange(N), top1_idx]

    # All negative pairs (upper triangle)
    iu, ju = np.triu_indices(N, k=1)
    all_sims = S[iu, ju]

    rng = np.random.default_rng(1234)
    if all_sims.size > args.pair_samples:
        sel = rng.choice(all_sims.size, size=args.pair_samples, replace=False)
        sample_pairs = np.stack([iu[sel], ju[sel]], axis=1)
        all_sims_sampled = all_sims[sel]
    else:
        sample_pairs = np.stack([iu, ju], axis=1)
        all_sims_sampled = all_sims

    # Plots
    plot_hist(top1_sim, "Top-1 negative similarity (global)", out_dir / "hist_top1_negative.png")
    plot_hist(all_sims_sampled, "All negative pairs (sampled)", out_dir / "hist_all_negative_sampled.png")

    # Top-1 negative table
    with open(out_dir / "top1_negative.tsv", "w", encoding="utf-8") as f:
        f.write("i\tsid_i\tsim_top1_neg\tj\tsid_j\ttext_i\ttext_j\n")
        for i in range(N):
            j = int(top1_idx[i])
            ti = texts[i].replace("\t", " ").strip()
            tj = texts[j].replace("\t", " ").strip()
            f.write(
                f"{i}\t{sids[i]}\t{float(top1_sim[i]):.6f}\t"
                f"{j}\t{sids[j]}\t{ti}\t{tj}\n"
            )

    # Optional sampled negative pairs
    if args.dump_sampled_pairs_tsv:
        with open(out_dir / "all_negative_sampled.tsv", "w", encoding="utf-8") as f:
            f.write("i\tj\tsim\tsid_i\tsid_j\ttext_i\ttext_j\n")
            for (i, j), s in zip(sample_pairs, all_sims_sampled):
                ti = texts[int(i)].replace("\t", " ").strip()
                tj = texts[int(j)].replace("\t", " ").strip()
                f.write(
                    f"{int(i)}\t{int(j)}\t{float(s):.6f}\t"
                    f"{sids[int(i)]}\t{sids[int(j)]}\t{ti}\t{tj}\n"
                )

    summary = {
        "N": int(N),
        "D": int(D),
        "top1_negative": stats_of(top1_sim),
        "all_negative_sampled": stats_of(all_sims_sampled),
        "positive_self_similarity": 1.0,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] N={N}, D={D}, "
        f"top1_neg_mean={summary['top1_negative'].get('mean', float('nan')):.3f}, "
        f"p95={summary['top1_negative'].get('p95', float('nan')):.3f} -> {out_dir}"
    )

if __name__ == "__main__":
    main()
