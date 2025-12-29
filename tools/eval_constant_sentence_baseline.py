#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_sentence_level_stats_fast.py

Purpose
-------
1. Sentence profiling:
   - Generate distribution plots (SVG / PDF)
   - Export sentence-level statistics to CSV.

2. Ultra-fast metric computation:
   - GPU-accelerated batch computation for BERTScore
     (e.g. 568 x 568 sentence pairs in a single run).
   - Compute WER / CER and identify the "centroid sentence".

Acceleration strategy
---------------------
- Avoid Python for-loops invoking GPU kernels.
- Construct a full (N x N) sentence-pair list and process it
  in large batches to fully utilize CUDA cores.

Dependencies
------------
- torch (required, CUDA strongly recommended)
- bert_score, jiwer, matplotlib, pandas
"""

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# ------------------------- Dependency checks -------------------------
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from jiwer import (
        wer as jiwer_wer,
        cer as jiwer_cer,
        Compose,
        ToLowerCase,
        RemovePunctuation,
        RemoveMultipleSpaces,
        Strip,
    )
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False

try:
    from bert_score import BERTScorer
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- Helper functions -------------------------

_TEXT_KEYS = [
    "transcript",
    "text",
    "sentence_text",
    "transcript_text",
    "global_segment_text",
]

def extract_text(r: dict) -> str:
    """
    Extract the first non-empty text field from a manifest row.
    """
    for k in _TEXT_KEYS:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def get_sentence_key(r: dict) -> str:
    """
    Resolve sentence identity for grouping.
    """
    return str(r.get("sentence_id") or r.get("content_id") or "unknown")

def build_normalizer(lowercase: bool, remove_punct: bool):
    """
    Build a JiWER normalization pipeline.
    """
    if not _HAS_JIWER:
        return lambda s: s
    steps = []
    if lowercase:
        steps.append(ToLowerCase())
    if remove_punct:
        steps.append(RemovePunctuation())
    steps += [RemoveMultipleSpaces(), Strip()]
    return Compose(steps)

# ------------------------- Plotting utilities -------------------------

def generate_plots(df: pd.DataFrame, out_dir: Path):
    """
    Generate sentence-level distribution plots:
    - Duration histogram
    - Window-count histogram
    - Duration vs. window-count scatter
    """
    valid_df = df.dropna(subset=["duration_s"])
    if len(valid_df) == 0:
        return

    durs = valid_df["duration_s"].values
    wins = valid_df["num_windows"].values

    # 1. Duration histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(durs, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of Sentence Durations")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "hist_duration.svg", bbox_inches="tight")
    fig.savefig(out_dir / "hist_duration.pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Window-count histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(wins.min(), wins.max() + 2) - 0.5
    ax.hist(wins, bins=bins, color="salmon", edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of Windows per Sentence")
    ax.set_xlabel("Number of Windows")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "hist_windows.svg", bbox_inches="tight")
    fig.savefig(out_dir / "hist_windows.pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. Scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(durs, wins, alpha=0.6, s=30)
    ax.set_title("Duration vs. Windows")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Window Count")
    fig.savefig(out_dir / "scatter_dur_win.svg", bbox_inches="tight")
    fig.savefig(out_dir / "scatter_dur_win.pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------------- Main program -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore_model", default="roberta-large")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for GPU inference (larger is faster if memory allows)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read manifest
    print("[1] Reading manifest...")
    with open(args.manifest, "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    groups: Dict[str, List[dict]] = {}
    for r in rows:
        groups.setdefault(get_sentence_key(r), []).append(r)

    print(f" -> Found {len(groups)} unique sentence groups.")

    # 2. Collect statistics and unique sentence texts
    stats_list = []
    unique_texts = []
    unique_ids = []

    for sid, items in groups.items():
        txt = extract_text(items[0])

        # Estimate sentence duration from available timestamps
        onsets = [
            x.get("global_segment_onset_in_audio_s")
            for x in items
            if x.get("global_segment_onset_in_audio_s") is not None
        ]
        offsets = [
            x.get("global_segment_offset_in_audio_s")
            for x in items
            if x.get("global_segment_offset_in_audio_s") is not None
        ]

        # Fallback to local window timestamps if needed
        if not onsets:
            onsets = [x.get("local_window_onset_in_audio_s", 0) for x in items]
        if not offsets:
            offsets = [x.get("local_window_offset_in_audio_s", 0) for x in items]

        dur = float("nan")
        if onsets and offsets:
            dur = max(offsets) - min(onsets)

        stats_list.append(
            {
                "sentence_id": sid,
                "text": txt,
                "duration_s": dur,
                "num_windows": len(items),
            }
        )

        # Only sentences with text participate in cross-sentence evaluation
        if txt:
            unique_texts.append(txt)
            unique_ids.append(sid)

    df_stats = pd.DataFrame(stats_list)
    df_stats.to_csv(out_dir / "sentence_stats.csv", index=False)
    generate_plots(df_stats, out_dir)
    print(" -> Stats and plots saved.")

    # 3. Fast all-vs-all evaluation
    print(f"[2] Starting batch evaluation on {len(unique_texts)} unique sentences...")

    normalizer = build_normalizer(args.lowercase, args.remove_punct)
    normalized_refs = [normalizer(t) for t in unique_texts]

    # ---------------------- BERTScore (GPU batch mode) ----------------------
    bert_f1_scores = None
    special_bert_f1 = None

    if args.bertscore and _HAS_BERTSCORE:
        device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        print(
            f" -> Loading BERTScore model on {device} "
            f"(batch size={args.batch_size})..."
        )

        scorer = BERTScorer(
            model_type=args.bertscore_model,
            lang="en",
            rescale_with_baseline=False,
            device=device,
        )

        N = len(unique_texts)

        # Baseline: "I don't know" vs all sentences
        special_cand = "I don't know"
        special_preds = [special_cand] * N
        print(" -> Computing BERTScore for 'I don't know'...")
        _, _, F_special = scorer.score(
            special_preds, unique_texts, batch_size=args.batch_size
        )
        special_bert_f1 = F_special.mean().item()

        # Full N x N BERTScore matrix
        print(f" -> Computing BERTScore matrix ({N}x{N} = {N*N} pairs)...")

        flat_preds = []
        for t in unique_texts:
            flat_preds.extend([t] * N)
        flat_refs = unique_texts * N

        start_time = time.time()
        _, _, F_matrix = scorer.score(
            flat_preds, flat_refs, batch_size=args.batch_size
        )
        end_time = time.time()
        print(f"    Done in {end_time - start_time:.2f}s!")

        F_matrix = F_matrix.view(N, N)
        bert_f1_scores = F_matrix.mean(dim=1).cpu().numpy()

        del scorer
        if _HAS_TORCH:
            torch.cuda.empty_cache()

    # ---------------------- WER / CER / BLEU ----------------------
    print(" -> Computing WER / CER / BLEU...")

    special_norm = normalizer("I don't know")
    special_wer = (
        jiwer_wer(normalized_refs, [special_norm] * len(normalized_refs))
        if _HAS_JIWER
        else None
    )

    pd.DataFrame(
        [
            {
                "candidate_text": "I don't know",
                "type": "special",
                "WER": special_wer,
                "BERTScore_F1": special_bert_f1,
            }
        ]
    ).to_csv(out_dir / "special_baseline_idontknow.csv", index=False)

    final_rows = []
    for i, txt in enumerate(unique_texts):
        norm_c = normalized_refs[i]
        preds_batch = [norm_c] * len(normalized_refs)

        w = jiwer_wer(normalized_refs, preds_batch) if _HAS_JIWER else None
        c = jiwer_cer(normalized_refs, preds_batch) if _HAS_JIWER else None

        b = None
        if _HAS_SACREBLEU:
            b = sacrebleu.corpus_bleu(
                [txt] * len(unique_texts), [unique_texts]
            ).score

        bf = bert_f1_scores[i] if bert_f1_scores is not None else None

        final_rows.append(
            {
                "sentence_id": unique_ids[i],
                "candidate_text": txt,
                "WER": w,
                "CER": c,
                "BLEU": b,
                "BERTScore_F1": bf,
            }
        )

    df_res = pd.DataFrame(final_rows).sort_values("WER")
    df_res.to_csv(out_dir / "all_sentences_as_baseline.csv", index=False)

    print("\n[DONE] Results summary:")
    if not df_res.empty:
        best = df_res.iloc[0]
        print(f"Best centroid sentence: '{best['candidate_text']}'")
        print(
            f"WER: {best['WER']:.4f} | "
            f"BERT-F1: {best['BERTScore_F1']:.4f}"
        )

    print(f"Files saved to: {out_dir}")

if __name__ == "__main__":
    main()
