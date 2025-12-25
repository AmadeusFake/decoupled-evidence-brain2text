#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paired_bootstrap_ci.py

Sentence-level paired bootstrap evaluation for *single-split* settings
(e.g. zero-shot decoding, non k-fold experiments).

This script is intentionally simpler than the k-fold variant:
- Unit of resampling: sentence (paired across systems via `sentence_key`)
- No fold stratification
- Typical use cases:
    * Zero-shot evaluation
    * Single held-out test set
    * Cross-dataset transfer without cross-validation

Inputs
------
- Multiple run directories, each containing:
      results/retrieval_final_min/sentence_metrics_{label}.tsv
  (or sentence_metrics.tsv as fallback)

Metrics
-------
- R@1, R@5, R@10, MRR, MedR

Outputs
-------
- Console summary
- LaTeX table snippet with:
    * Absolute scores
    * 95% paired bootstrap CI (Δ vs baseline)
    * Two-sided bootstrap p-values
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------

METRICS = ["R1", "R5", "R10", "MRR", "MedR"]

# ---------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------

def _ensure_rr_dir(run_dir: Path) -> Path:
    """
    Normalize input path to results/retrieval_final_min.

    Accepts either:
    - the retrieval directory itself, or
    - a higher-level run directory containing results/retrieval_final_min
    """
    if (run_dir / "sentence_metrics.tsv").exists() or any(run_dir.glob("sentence_metrics_*.tsv")):
        return run_dir

    cand = run_dir / "results" / "retrieval_final_min"
    return cand if cand.exists() else run_dir


def _resolve_sentence_tsv(rr_dir: Path, label: str) -> Path:
    """
    Resolve sentence-level metrics TSV with preference order:
      1) sentence_metrics_{label}.tsv
      2) sentence_metrics.tsv
      3) newest matching file under rr_dir (recursive fallback)
    """
    p1 = rr_dir / f"sentence_metrics_{label}.tsv"
    if p1.exists():
        return p1

    p2 = rr_dir / "sentence_metrics.tsv"
    if p2.exists():
        return p2

    # Recursive fallback: newest match wins
    candidates = sorted(
        rr_dir.glob(f"**/sentence_metrics_{label}.tsv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"[{label}] sentence metrics not found under: {rr_dir}")


def load_sentence_table_for_label(run_dir: Path, label: str) -> pd.DataFrame:
    """
    Load sentence-level metrics for a given system label.
    """
    rr_dir = _ensure_rr_dir(run_dir)
    tsv = _resolve_sentence_tsv(rr_dir, label)

    df = pd.read_csv(tsv, sep="\t")

    required = {"sentence_key", "R1", "R5", "R10", "MRR", "MedR"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"[{label}] Bad TSV schema at {tsv}; "
            f"missing columns: {required - set(df.columns)}"
        )
    return df


# ---------------------------------------------------------------------
# Pairing and aggregation
# ---------------------------------------------------------------------

def intersect_sentences(
    dfs: Dict[str, pd.DataFrame]
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Intersect sentence sets across all systems and align them.

    Pairing key: sentence_key
    """
    sets = [set(df["sentence_key"]) for df in dfs.values()]
    common = set.intersection(*sets) if sets else set()

    if not common:
        raise RuntimeError(
            "No common sentence_key across runs; paired bootstrap is invalid."
        )

    aligned = {}
    for k, df in dfs.items():
        aligned[k] = (
            df[df["sentence_key"].isin(common)]
            .set_index("sentence_key")
            .sort_index()
        )

    return sorted(common), aligned


def agg_point_estimates(df: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate sentence-level metrics into global point estimates.

    If `n_windows` is present:
    - R@k / MRR: window-weighted micro-average
    - MedR: sentence-level median (exact per-window MedR not recoverable)

    Otherwise:
    - Sentence-level macro averages
    """
    if "n_windows" in df.columns:
        w = df["n_windows"].astype(float).to_numpy()
        if w.sum() > 0:
            return {
                "R1":  float(np.average(df["R1"],  weights=w)),
                "R5":  float(np.average(df["R5"],  weights=w)),
                "R10": float(np.average(df["R10"], weights=w)),
                "MRR": float(np.average(df["MRR"], weights=w)),
                "MedR": float(df["MedR"].median()),
            }

    # Fallback: unweighted macro averages
    return {
        "R1":  float(df["R1"].mean()),
        "R5":  float(df["R5"].mean()),
        "R10": float(df["R10"].mean()),
        "MRR": float(df["MRR"].mean()),
        "MedR": float(df["MedR"].median()),
    }


# ---------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------

def paired_bootstrap(
    idx: List[str],
    dfs: Dict[str, pd.DataFrame],
    baseline: str,
    *,
    B: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Paired bootstrap over sentences.

    Resampling unit: sentence
    Null hypothesis: Δ(metric) = 0 vs baseline
    """
    rng = np.random.default_rng(seed)
    res: Dict[str, Dict] = {}

    base_df = dfs[baseline]

    # Point estimates
    for lab, df in dfs.items():
        res.setdefault(lab, {})["point"] = agg_point_estimates(df.loc[idx])

    # Bootstrap deltas
    for lab, df in dfs.items():
        if lab == baseline:
            continue

        delta = {m: [] for m in METRICS}
        n = len(idx)

        for _ in range(B):
            samp = rng.choice(n, size=n, replace=True)
            I = [idx[i] for i in samp]

            b_pt = agg_point_estimates(base_df.loc[I])
            k_pt = agg_point_estimates(df.loc[I])

            for m in METRICS:
                delta[m].append(k_pt[m] - b_pt[m])

        comp = {}
        for m in METRICS:
            arr = np.asarray(delta[m], dtype=float)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            p_two = 2 * min((arr <= 0).mean(), (arr >= 0).mean())

            comp[m] = {
                "d_mean": float(arr.mean()),
                "d_med":  float(np.median(arr)),
                "ci_lo":  float(lo),
                "ci_hi":  float(hi),
                "p":      float(p_two),
            }

        res[lab]["vs_" + baseline] = comp

    return res


# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def _fmt_p(pv: float, B: int) -> str:
    """
    Pretty-print bootstrap p-values.

    If pv == 0, report lower bound based on bootstrap resolution.
    """
    if pv <= 0.0:
        lb = 1.0 / (B + 1)
        exp = int(np.floor(np.log10(lb)))
        base = lb / (10 ** exp)
        return f"<{base:.1f}×10^{{{exp}}}"

    if pv < 1e-4:
        exp = int(np.floor(np.log10(pv)))
        base = pv / (10 ** exp)
        return f"{base:.1f}×10^{{{exp}}}"

    return f"{pv:.4f}"


def _fmt_entry_abs(
    point: float,
    abs_lo: float,
    abs_hi: float,
    rel_gain_pct: float,
    p: float,
    B: int,
    higher_is_better: bool = True,
) -> str:
    arrow = "↑" if (rel_gain_pct >= 0 if higher_is_better else rel_gain_pct <= 0) else "↓"
    return (
        f"{point:.2f} [{abs_lo:.2f}–{abs_hi:.2f}] "
        f"({arrow}{abs(rel_gain_pct):.1f}%) ; p={_fmt_p(p, B)}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+", required=True,
        help="Pairs LABEL=RUN_DIR (e.g. baseline=/path gcb=/path)"
    )
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tex", type=str, default="")
    args = ap.parse_args()

    # Parse run map
    label2dir: Dict[str, Path] = {}
    for item in args.runs:
        if "=" not in item:
            print(f"Bad --runs item: {item}", file=sys.stderr)
            sys.exit(2)
        lab, path = item.split("=", 1)
        label2dir[lab] = Path(path)

    # Load sentence tables
    dfs = {}
    resolved = {}
    for lab, p in label2dir.items():
        rr = _ensure_rr_dir(p)
        tsv = _resolve_sentence_tsv(rr, lab)
        resolved[lab] = tsv
        dfs[lab] = pd.read_csv(tsv, sep="\t")

    print("\n=== Resolved TSV paths ===")
    for k, v in resolved.items():
        print(f"{k}: {v.as_posix()}")

    # Pair sentences
    idx, dfs_aligned = intersect_sentences(dfs)

    # Bootstrap
    stats = paired_bootstrap(
        idx, dfs_aligned,
        baseline=args.baseline,
        B=args.B,
        seed=args.seed,
    )

    # Summary
    print("\n=== Point estimates ===")
    for lab in label2dir:
        pe = stats[lab]["point"]
        print(
            f"[{lab}] "
            f"R@1={pe['R1']:.4f} R@5={pe['R5']:.4f} "
            f"R@10={pe['R10']:.4f} MRR={pe['MRR']:.4f} MedR={pe['MedR']:.2f}"
        )

    # Build LaTeX table
    baseline_pe = stats[args.baseline]["point"]
    labels = list(label2dir.keys())

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Retrieval under zero-shot decoding with sentence-level paired bootstrap (95\\% CI) and p-values vs. baseline.}",
        "\\label{tab:retrieval_zeroshot_bootstrap}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{R@1} & \\textbf{R@5} & "
        "\\textbf{R@10} & \\textbf{MRR} & \\textbf{MedR} \\\\",
        "\\midrule",
    ]

    def _rel(base, now):
        return 100.0 * (now - base) / max(1e-12, base)

    for lab in labels:
        pe = stats[lab]["point"]

        if lab == args.baseline:
            lines.append(
                f"{lab} & {pe['R1']:.2f} & {pe['R5']:.2f} & "
                f"{pe['R10']:.2f} & {pe['MRR']:.2f} & {pe['MedR']:.1f} \\\\"
            )
            continue

        comp = stats[lab]["vs_" + args.baseline]

        r1  = _fmt_entry_abs(
            pe["R1"],
            baseline_pe["R1"]  + comp["R1"]["ci_lo"],
            baseline_pe["R1"]  + comp["R1"]["ci_hi"],
            _rel(baseline_pe["R1"], pe["R1"]),
            comp["R1"]["p"], args.B
        )
        r5  = _fmt_entry_abs(
            pe["R5"],
            baseline_pe["R5"]  + comp["R5"]["ci_lo"],
            baseline_pe["R5"]  + comp["R5"]["ci_hi"],
            _rel(baseline_pe["R5"], pe["R5"]),
            comp["R5"]["p"], args.B
        )
        r10 = _fmt_entry_abs(
            pe["R10"],
            baseline_pe["R10"] + comp["R10"]["ci_lo"],
            baseline_pe["R10"] + comp["R10"]["ci_hi"],
            _rel(baseline_pe["R10"], pe["R10"]),
            comp["R10"]["p"], args.B
        )
        mrr = _fmt_entry_abs(
            pe["MRR"],
            baseline_pe["MRR"] + comp["MRR"]["ci_lo"],
            baseline_pe["MRR"] + comp["MRR"]["ci_hi"],
            _rel(baseline_pe["MRR"], pe["MRR"]),
            comp["MRR"]["p"], args.B
        )

        # MedR: lower is better
        med_rel = -_rel(baseline_pe["MedR"], pe["MedR"])
        med = (
            f"{pe['MedR']:.1f} "
            f"[{baseline_pe['MedR'] + comp['MedR']['ci_lo']:.1f}–"
            f"{baseline_pe['MedR'] + comp['MedR']['ci_hi']:.1f}] "
            f"(↓{abs(med_rel):.1f}%) ; p={_fmt_p(comp['MedR']['p'], args.B)}"
        )

        lines.append(f"{lab} & {r1} & {r5} & {r10} & {mrr} & {med} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    tex = "\n".join(lines)
    print("\n=== LaTeX Table Snippet ===\n")
    print(tex)

    if args.out_tex:
        Path(args.out_tex).write_text(tex, encoding="utf-8")
        print(f"[INFO] LaTeX table written to: {args.out_tex}")


if __name__ == "__main__":
    main()
