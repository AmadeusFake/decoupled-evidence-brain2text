#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paired_bootstrap_ci_kfold.py

Sentence-level paired bootstrap evaluation across k-fold runs.

Overview
--------
- Input:
    * A list of run directories (one per fold, ordered as fold0, fold1, ...).
    * For each run_dir, metrics are searched under:
          run_dir/results/retrieval_final_min/
      with the following priority:
          1) sentence_metrics_<label>_fold<F>.tsv
          2) sentence_metrics_<label>.tsv
          3) sentence_metrics.tsv   (fallback)

- Comparison:
    * Multiple system labels (e.g., baseline, vote_only, gcb_only, gcb_vote)
    * Sentence-level pairing across methods
    * Stratified bootstrap by fold (fold-preserving resampling)

- Output:
    * Point estimates
    * 95% confidence intervals (paired Δ vs. baseline)
    * Two-sided paired bootstrap p-values
    * LaTeX table ready for inclusion in the paper
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Metrics (protocol-level constants)
# ---------------------------------------------------------------------

METRICS = ["R1", "R5", "R10", "MRR", "MedR"]

# ---------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------

def _ensure_rr_dir(run_dir: Path) -> Path:
    """
    Normalize input to .../results/retrieval_final_min if needed.
    """
    if (run_dir / "sentence_metrics.tsv").exists() or any(run_dir.glob("sentence_metrics_*.tsv")):
        return run_dir
    cand = run_dir / "results" / "retrieval_final_min"
    return cand if cand.exists() else run_dir

def _find_sentence_tsv(rr_dir: Path, label: str, fold: int) -> Path:
    """
    Resolve sentence-level metrics TSV with the following priority:
      1) sentence_metrics_<label>_fold<F>.tsv
      2) sentence_metrics_<label>.tsv
      3) sentence_metrics.tsv
    Falls back to recursive search if needed.
    """
    prefs = [
        rr_dir / f"sentence_metrics_{label}_fold{fold}.tsv",
        rr_dir / f"sentence_metrics_{label}.tsv",
        rr_dir / "sentence_metrics.tsv",
    ]
    for p in prefs:
        if p.exists():
            return p

    # Recursive fallback (latest mtime wins)
    for pattern in (
        f"sentence_metrics_{label}_fold{fold}.tsv",
        f"sentence_metrics_{label}.tsv",
        "sentence_metrics.tsv",
    ):
        cand = sorted(
            rr_dir.rglob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if cand:
            return cand[0]

    raise FileNotFoundError(
        f"[fold={fold}, label={label}] sentence metrics not found under {rr_dir}"
    )

def load_label_fold_table(
    run_dir: Path, label: str, fold: int
) -> Tuple[pd.DataFrame, Path]:
    """
    Load sentence-level metrics for a given (label, fold).
    """
    rr = _ensure_rr_dir(run_dir)
    p = _find_sentence_tsv(rr, label, fold)

    df = pd.read_csv(p, sep="\t")
    required = {"sentence_key", "R1", "R5", "R10", "MRR", "MedR"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"[{label}, fold={fold}] Bad TSV schema at {p}; "
            f"missing {required - set(df.columns)}"
        )

    df["__fold__"] = fold
    return df, p

# ---------------------------------------------------------------------
# Data collection and alignment
# ---------------------------------------------------------------------

def collect_all(
    run_dirs: List[Path],
    labels: List[str],
) -> Tuple[Dict[str, pd.DataFrame], Dict[Tuple[int, str], Path]]:
    """
    Collect and align sentence-level metrics across folds and labels.

    Returns
    -------
    dfs[label]:
        DataFrame indexed by uid = "fold|sentence_key",
        aligned by intersection across all labels.
    resolved[(fold, label)]:
        Resolved TSV path (for debugging / provenance).
    """
    resolved: Dict[Tuple[int, str], Path] = {}
    out: Dict[str, pd.DataFrame] = {}

    for lab in labels:
        parts = []
        for f, rd in enumerate(run_dirs):
            df, p = load_label_fold_table(rd, lab, fold=f)
            resolved[(f, lab)] = p
            parts.append(df)

        cat = pd.concat(parts, axis=0, ignore_index=True)
        cat["uid"] = cat["__fold__"].astype(str) + "|" + cat["sentence_key"].astype(str)
        out[lab] = cat.set_index("uid").sort_index()

    # Sentence-level pairing across all labels
    common = set.intersection(*[set(df.index) for df in out.values()])
    if not common:
        raise RuntimeError(
            "No common fold|sentence entries across labels; "
            "paired bootstrap cannot be performed."
        )

    common_sorted = sorted(common)
    out = {lab: df.loc[common_sorted].copy() for lab, df in out.items()}
    return out, resolved

# ---------------------------------------------------------------------
# Point estimates and bootstrap
# ---------------------------------------------------------------------

def agg_point_estimates(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute point estimates.

    Protocol:
    - If 'n_windows' is present:
        * R@k and MRR: window-weighted micro-average
        * MedR: sentence-level median
    - Otherwise:
        * Sentence-level macro averages
    """
    if "n_windows" in df.columns:
        w = df["n_windows"].to_numpy(dtype=float)
        if w.sum() > 0:
            return {
                "R1":  float(np.average(df["R1"],  weights=w)),
                "R5":  float(np.average(df["R5"],  weights=w)),
                "R10": float(np.average(df["R10"], weights=w)),
                "MRR": float(np.average(df["MRR"], weights=w)),
                "MedR": float(df["MedR"].median()),
            }

    # Fallback: macro
    return {
        "R1":  float(df["R1"].mean()),
        "R5":  float(df["R5"].mean()),
        "R10": float(df["R10"].mean()),
        "MRR": float(df["MRR"].mean()),
        "MedR": float(df["MedR"].median()),
    }

def point_estimate(df: pd.DataFrame) -> Dict[str, float]:
    return agg_point_estimates(df)

def stratified_bootstrap(
    idx_by_fold: Dict[int, List[str]],
    dfs: Dict[str, pd.DataFrame],
    baseline: str,
    B: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Paired, fold-stratified bootstrap.

    Resampling is performed *within each fold* and concatenated,
    preserving the fold structure.
    """
    rng = np.random.default_rng(seed)

    res = {lab: {"point": point_estimate(dfs[lab])} for lab in dfs}
    base_df = dfs[baseline]

    # Sanity check: exact pairing
    for lab, df in dfs.items():
        assert set(df.index) == set(base_df.index), "Indices must align for pairing"

    fold_keys = sorted(idx_by_fold.keys())

    for lab, df in dfs.items():
        if lab == baseline:
            continue

        deltas = {m: [] for m in METRICS}

        for _ in range(B):
            sample_uids = []
            for f in fold_keys:
                keys = idx_by_fold[f]
                picks = rng.choice(len(keys), size=len(keys), replace=True)
                sample_uids.extend([keys[i] for i in picks])

            df_b = base_df.loc[sample_uids]
            df_k = df.loc[sample_uids]

            pe_b = point_estimate(df_b)
            pe_k = point_estimate(df_k)

            for m in METRICS:
                deltas[m].append(pe_k[m] - pe_b[m])

        comp = {}
        for m in METRICS:
            arr = np.asarray(deltas[m], dtype=float)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            p_two = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
            comp[m] = {
                "d_mean": float(arr.mean()),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "p": float(p_two),
            }

        res[lab]["vs_" + baseline] = comp

    return res

# ---------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------

def _fmt_p(pv: float, B: int) -> str:
    """
    Pretty-print p-values.
    If p=0, report a lower bound based on bootstrap resolution.
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
) -> str:
    """
    Higher-is-better metric formatting:
    absolute CI + relative gain vs. baseline + p-value.
    """
    arrow = "↑" if rel_gain_pct >= 0 else "↓"
    return (
        f"{point:.2f} [{abs_lo:.2f}–{abs_hi:.2f}] "
        f"({arrow}{abs(rel_gain_pct):.1f}%) ; p={_fmt_p(p, B)}"
    )

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="run directories (one per fold, ordered)")
    ap.add_argument("--labels", nargs="+",
                    default=["baseline", "vote_only", "gcb_only", "gcb_vote"])
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tex", default="")
    ap.add_argument("--caption", default=(
        "Retrieval under session isolation (k-fold) with sentence-level paired "
        "bootstrap (95\\% CI) and p-values vs. baseline."
    ))
    ap.add_argument("--table_label", default="tab:retrieval_isolation_bootstrap")
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs]
    dfs, resolved = collect_all(run_dirs, args.labels)

    # Debug: resolved TSV paths
    print("\n=== Resolved TSV paths (fold,label) ===")
    for (f, lab), p in sorted(resolved.items()):
        print(f"fold{f},{lab}: {p.as_posix()}")

    # Fold stratification from baseline
    idx_by_fold: Dict[int, List[str]] = {}
    for uid in dfs[args.baseline].index:
        f = int(uid.split("|", 1)[0])
        idx_by_fold.setdefault(f, []).append(uid)

    stats = stratified_bootstrap(
        idx_by_fold, dfs,
        baseline=args.baseline,
        B=args.B, seed=args.seed
    )

    base_pe = stats[args.baseline]["point"]

    # -----------------------------------------------------------------
    # LaTeX table
    # -----------------------------------------------------------------

    lines: List[str] = []
    lines += [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{args.caption}}}",
        f"\\label{{{args.table_label}}}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Configuration} & \\textbf{R@1} & \\textbf{R@5} & "
        "\\textbf{R@10} & \\textbf{MRR} & \\textbf{MedR} \\\\",
        "\\midrule",
    ]

    def _rel(base: float, now: float) -> float:
        return 100.0 * (now - base) / max(1e-12, base)

    for lab in args.labels:
        pe = stats[lab]["point"]

        if lab == args.baseline:
            lines.append(
                f"{lab} & {pe['R1']:.2f} & {pe['R5']:.2f} & {pe['R10']:.2f} & "
                f"{pe['MRR']:.2f} & {pe['MedR']:.1f} \\\\"
            )
            continue

        comp = stats[lab]["vs_" + args.baseline]

        r1  = _fmt_entry_abs(pe["R1"],
                             base_pe["R1"]  + comp["R1"]["ci_lo"],
                             base_pe["R1"]  + comp["R1"]["ci_hi"],
                             _rel(base_pe["R1"], pe["R1"]),
                             comp["R1"]["p"], args.B)
        r5  = _fmt_entry_abs(pe["R5"],
                             base_pe["R5"]  + comp["R5"]["ci_lo"],
                             base_pe["R5"]  + comp["R5"]["ci_hi"],
                             _rel(base_pe["R5"], pe["R5"]),
                             comp["R5"]["p"], args.B)
        r10 = _fmt_entry_abs(pe["R10"],
                             base_pe["R10"] + comp["R10"]["ci_lo"],
                             base_pe["R10"] + comp["R10"]["ci_hi"],
                             _rel(base_pe["R10"], pe["R10"]),
                             comp["R10"]["p"], args.B)
        mrr = _fmt_entry_abs(pe["MRR"],
                             base_pe["MRR"] + comp["MRR"]["ci_lo"],
                             base_pe["MRR"] + comp["MRR"]["ci_hi"],
                             _rel(base_pe["MRR"], pe["MRR"]),
                             comp["MRR"]["p"], args.B)

        # MedR: lower is better
        med_rel = -_rel(base_pe["MedR"], pe["MedR"])
        med_abs_lo = base_pe["MedR"] + comp["MedR"]["ci_lo"]
        med_abs_hi = base_pe["MedR"] + comp["MedR"]["ci_hi"]
        med = (
            f"{pe['MedR']:.1f} [{med_abs_lo:.1f}–{med_abs_hi:.1f}] "
            f"(↓{abs(med_rel):.1f}%) ; p={_fmt_p(comp['MedR']['p'], args.B)}"
        )

        lines.append(f"{lab} & {r1} & {r5} & {r10} & {mrr} & {med} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]

    tex = "\n".join(lines)
    print("\n=== LaTeX Table ===\n")
    print(tex)

    if args.out_tex:
        out_tex = Path(args.out_tex)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(tex, encoding="utf-8")
        print(f"[INFO] LaTeX table written to: {out_tex}")

if __name__ == "__main__":
    main()
