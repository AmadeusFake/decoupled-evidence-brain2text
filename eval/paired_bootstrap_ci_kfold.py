#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paired_bootstrap_ci_kfold.py

- Inputs: a list of run_dirs (one per fold). For each run_dir, we look under:
      run_dir/results/retrieval_final_min/
  for files like:
      sentence_metrics_<label>.tsv        (推荐方式)
      sentence_metrics_<label>_fold<F>.tsv  (也支持)
      sentence_metrics.tsv                (最后兜底)
  同时会在终端打印每个 (fold,label) 实际使用的文件路径，方便 debug。

- Labels compared: e.g., baseline, vote_only, gcb_only, gcb_vote
- Unit: sentence (paired across methods), with stratified bootstrap over folds
- Output: LaTeX table with point estimates + 95% CI (Δ vs baseline) + paired bootstrap p-values
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

METRICS = ["R1","R5","R10","MRR","MedR"]

# ---------------- Path helpers ----------------

def _ensure_rr_dir(run_dir: Path) -> Path:
    """Normalize to .../results/retrieval_final_min if needed."""
    # 如果传进来的就是 retrieval_final_min，就直接用；
    # 否则尝试附加 results/retrieval_final_min。
    if (run_dir / "sentence_metrics.tsv").exists() or any(run_dir.glob("sentence_metrics_*.tsv")):
        return run_dir
    cand = run_dir / "results" / "retrieval_final_min"
    return cand if cand.exists() else run_dir

def _find_sentence_tsv(rr_dir: Path, label: str, fold: int) -> Path:
    """Prefer label+fold, then label-only, then generic; search recursively if needed."""
    prefs = [
        rr_dir / f"sentence_metrics_{label}_fold{fold}.tsv",
        rr_dir / f"sentence_metrics_{label}.tsv",
        rr_dir / "sentence_metrics.tsv",
    ]
    for p in prefs:
        if p.exists():
            return p

    # recursive fallbacks
    cand = list(rr_dir.rglob(f"sentence_metrics_{label}_fold{fold}.tsv"))
    if cand:
        return sorted(cand, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    cand = list(rr_dir.rglob(f"sentence_metrics_{label}.tsv"))
    if cand:
        return sorted(cand, key=lambda x: x.stat().st_mtime, reverse=True)[0]
    cand = list(rr_dir.rglob("sentence_metrics.tsv"))
    if cand:
        return sorted(cand, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(f"[fold={fold}, label={label}] sentence metrics not found under: {rr_dir}")

def load_label_fold_table(run_dir: Path, label: str, fold: int) -> Tuple[pd.DataFrame, Path]:
    rr = _ensure_rr_dir(run_dir)
    p = _find_sentence_tsv(rr, label, fold)
    df = pd.read_csv(p, sep="\t")
    need = {"sentence_key","R1","R5","R10","MRR","MedR"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"[{label}, fold={fold}] Bad TSV schema at {p}; missing {need - set(df.columns)}")
    df["__fold__"] = fold
    return df, p

# ---------------- Collect & align ----------------

def collect_all(run_dirs: List[Path], labels: List[str]) -> Tuple[Dict[str, pd.DataFrame], Dict[Tuple[int, str], Path]]:
    """
    Return:
      dfs[label]: DataFrame indexed by 'uid'="fold|sentence_key", aligned by intersection across labels.
      resolved[(fold,label)]: resolved file path for debug printing.
    """
    resolved: Dict[Tuple[int,str], Path] = {}
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

    # align by intersection of uids across labels
    common = set.intersection(*[set(df.index) for df in out.values()])
    if not common:
        raise RuntimeError("No common fold|sentence entries across labels; cannot do paired bootstrap.")
    common_sorted = sorted(common)
    out = {lab: df.loc[common_sorted].copy() for lab, df in out.items()}
    return out, resolved

# ---------------- Estimates & bootstrap ----------------
def agg_point_estimates(df: pd.DataFrame) -> Dict[str,float]:
    """
    与 paired_bootstrap_ci.py 保持一致：
    - 若存在 n_windows：R@k / MRR 使用 window 数量加权的 micro-average
    - MedR 仍然用句级 median
    """
    if "n_windows" in df.columns:
        w = df["n_windows"].to_numpy().astype(float)
        w_sum = w.sum()
        if w_sum <= 0:
            return {
                "R1":  float(df["R1"].mean()),
                "R5":  float(df["R5"].mean()),
                "R10": float(df["R10"].mean()),
                "MRR": float(df["MRR"].mean()),
                "MedR": float(df["MedR"].median()),
            }

        R1  = np.average(df["R1"],  weights=w)
        R5  = np.average(df["R5"],  weights=w)
        R10 = np.average(df["R10"], weights=w)
        MRR = np.average(df["MRR"], weights=w)
        MedR = float(df["MedR"].median())

        return {
            "R1":  float(R1),
            "R5":  float(R5),
            "R10": float(R10),
            "MRR": float(MRR),
            "MedR": MedR,
        }

    # 兜底：没有 n_windows 就退回句级 macro
    return {
        "R1":  float(df["R1"].mean()),
        "R5":  float(df["R5"].mean()),
        "R10": float(df["R10"].mean()),
        "MRR": float(df["MRR"].mean()),
        "MedR": float(df["MedR"].median()),
    }

def point_estimate(df: pd.DataFrame) -> Dict[str,float]:
    return agg_point_estimates(df)
    
def stratified_bootstrap(idx_by_fold: Dict[int, List[str]], dfs: Dict[str, pd.DataFrame],
                         baseline: str, B: int = 10000, seed: int = 42) -> Dict[str, Dict]:
    rng = np.random.default_rng(seed)
    res = {lab: {"point": point_estimate(dfs[lab])} for lab in dfs.keys()}
    base_df = dfs[baseline]

    # pairing sanity
    for lab, df in dfs.items():
        assert set(df.index) == set(base_df.index), "indices must align for pairing"

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
            arr = np.array(deltas[m], dtype=float)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            p_two = 2 * min((arr <= 0).mean(), (arr >= 0).mean())  # two-sided
            comp[m] = {
                "d_mean": float(arr.mean()),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "p": float(p_two),
            }
        res[lab]["vs_" + baseline] = comp
    return res

# ---------------- Formatting ----------------

def _fmt_p(pv: float, B: int) -> str:
    """Show p=0 as a lower bound based on bootstrap resolution: <1/(B+1)."""
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

def _fmt_entry_abs(point: float, abs_lo: float, abs_hi: float,
                   rel_gain_pct: float, p: float, B: int) -> str:
    """Higher-is-better metric：arrow 表示“比 baseline 高多少 %”"""
    arrow = "↑" if rel_gain_pct >= 0 else "↓"
    return f"{point:.2f} [{abs_lo:.2f}–{abs_hi:.2f}] ({arrow}{abs(rel_gain_pct):.1f}%) ; p={_fmt_p(p, B)}"

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="list of run_dirs, one per fold, order=fold0..foldK (baseline training dirs)")
    ap.add_argument("--labels", nargs="+", default=["baseline","vote_only","gcb_only","gcb_vote"])
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tex", type=str, default="")
    ap.add_argument("--caption", type=str,
                    default="Retrieval under session isolation (k-fold) with sentence-level paired bootstrap (95\\% CI) and p-values vs. baseline.")
    ap.add_argument("--table_label", type=str, default="tab:retrieval_isolation_bootstrap")
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs]
    dfs, resolved = collect_all(run_dirs, args.labels)

    # print resolved map for debugging
    print("\n=== Resolved TSV paths (fold,label) ===")
    for (f, lab), p in sorted(resolved.items()):
        print(f"fold{f},{lab}: {p.as_posix()}")

    # build fold stratification from baseline index
    idx_by_fold: Dict[int, List[str]] = {}
    for uid in dfs[args.baseline].index:
        f = int(uid.split("|", 1)[0])
        idx_by_fold.setdefault(f, []).append(uid)

    stats = stratified_bootstrap(idx_by_fold, dfs, baseline=args.baseline, B=args.B, seed=args.seed)
    base_pe = stats[args.baseline]["point"]

    # -- LaTeX --
    lines: List[str] = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{args.caption}}}")
    lines.append(f"\\label{{{args.table_label}}}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.1}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Configuration} & \\textbf{R@1} & \\textbf{R@5} & "
                 "\\textbf{R@10} & \\textbf{MRR} & \\textbf{MedR} \\\\")
    lines.append("\\midrule")

    def _rel(base: float, now: float) -> float:
        return 100.0 * (now - base) / max(1e-12, base)

    for lab in args.labels:
        pe = stats[lab]["point"]
        if lab == args.baseline:
            lines.append(
                f"{lab} & {pe['R1']:.2f} & {pe['R5']:.2f} & {pe['R10']:.2f} & "
                f"{pe['MRR']:.2f} & {pe['MedR']:.1f} \\\\"
            )
        else:
            comp = stats[lab]["vs_" + args.baseline]

            # absolute CIs = baseline_point + ΔCI (更直观)
            r1_abs_lo  = base_pe["R1"]  + comp["R1"]["ci_lo"];  r1_abs_hi  = base_pe["R1"]  + comp["R1"]["ci_hi"]
            r5_abs_lo  = base_pe["R5"]  + comp["R5"]["ci_lo"];  r5_abs_hi  = base_pe["R5"]  + comp["R5"]["ci_hi"]
            r10_abs_lo = base_pe["R10"] + comp["R10"]["ci_lo"]; r10_abs_hi = base_pe["R10"] + comp["R10"]["ci_hi"]
            mrr_abs_lo = base_pe["MRR"] + comp["MRR"]["ci_lo"]; mrr_abs_hi = base_pe["MRR"] + comp["MRR"]["ci_hi"]

            r1  = _fmt_entry_abs(pe["R1"],  r1_abs_lo,  r1_abs_hi,  _rel(base_pe["R1"],  pe["R1"]),  comp["R1"]["p"],  args.B)
            r5  = _fmt_entry_abs(pe["R5"],  r5_abs_lo,  r5_abs_hi,  _rel(base_pe["R5"],  pe["R5"]),  comp["R5"]["p"],  args.B)
            r10 = _fmt_entry_abs(pe["R10"], r10_abs_lo, r10_abs_hi, _rel(base_pe["R10"], pe["R10"]), comp["R10"]["p"], args.B)
            mrr = _fmt_entry_abs(pe["MRR"], mrr_abs_lo, mrr_abs_hi, _rel(base_pe["MRR"], pe["MRR"]), comp["MRR"]["p"], args.B)

            # MedR: smaller is better → improvement = 相对下降百分比（始终用 ↓）
            med_rel = -_rel(base_pe["MedR"], pe["MedR"])  # 正值 = MedR 降低
            med_abs_lo = base_pe["MedR"] + comp["MedR"]["ci_lo"]
            med_abs_hi = base_pe["MedR"] + comp["MedR"]["ci_hi"]
            med = f"{pe['MedR']:.1f} [{med_abs_lo:.1f}–{med_abs_hi:.1f}] (↓{abs(med_rel):.1f}%) ; p={_fmt_p(comp['MedR']['p'], args.B)}"

            lines.append(f"{lab} & {r1} & {r5} & {r10} & {mrr} & {med} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
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
