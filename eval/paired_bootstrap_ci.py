#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paired_bootstrap_ci.py
- Inputs: multiple run dirs; each must contain results/retrieval_final_min/sentence_metrics_{label}.tsv
  (or sentence_metrics.tsv as fallback)
- Unit of resampling: sentence (paired across methods via `sentence_key`)
- Metrics: R@1, R@5, R@10, MRR, MedR
- Outputs: text summary + LaTeX table snippet with 95% CI (for Δ vs. baseline) and paired bootstrap p-values

Usage example:
  python eval/paired_bootstrap_ci.py \
    --runs baseline=/path/.../results/retrieval_final_min \
           vote_only=/path/.../results/retrieval_final_min \
           gcb_only=/path/.../results/retrieval_final_min \
           gcb_vote=/path/.../results/retrieval_final_min \
    --baseline baseline \
    --B 10000 \
    --out_tex /path/to/bootstrap_table.tex
"""

import argparse, sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

METRICS = ["R1","R5","R10","MRR","MedR"]

# ---------- Path resolving ----------

def _ensure_rr_dir(run_dir: Path) -> Path:
    """
    Accept either the directory that already IS results/retrieval_final_min,
    or a higher run dir. Normalize to .../results/retrieval_final_min
    """
    if (run_dir / "sentence_metrics.tsv").exists() or any(run_dir.glob("sentence_metrics_*.tsv")):
        return run_dir
    cand = run_dir / "results" / "retrieval_final_min"
    return cand if cand.exists() else run_dir  # last fallback; later checks will fail loudly if missing

def _resolve_sentence_tsv(rr_dir: Path, label: str) -> Path:
    """
    Prefer label-specific TSV, fallback to unlabeled TSV.
    """
    p1 = rr_dir / f"sentence_metrics_{label}.tsv"
    if p1.exists():
        return p1
    p2 = rr_dir / "sentence_metrics.tsv"
    if p2.exists():
        return p2
    # As a last resort: try to find the newest subdir containing the label file
    candidates = sorted(rr_dir.glob(f"**/sentence_metrics_{label}.tsv"), key=lambda x: x.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"[{label}] sentence metrics not found under: {rr_dir}")

def load_sentence_table_for_label(run_dir: Path, label: str) -> pd.DataFrame:
    rr = _ensure_rr_dir(run_dir)
    p = _resolve_sentence_tsv(rr, label)
    df = pd.read_csv(p, sep="\t")
    # basic sanity
    need_cols = {"sentence_key","R1","R5","R10","MRR","MedR"}
    if not need_cols.issubset(set(df.columns)):
        raise RuntimeError(f"[{label}] Bad TSV schema at {p} — missing columns in {need_cols - set(df.columns)}")
    return df

# ---------- Pairing & aggregation ----------

def intersect_sentences(dfs: Dict[str,pd.DataFrame]) -> Tuple[List[str], Dict[str,pd.DataFrame]]:
    sets = [set(df["sentence_key"].tolist()) for df in dfs.values()]
    common = set.intersection(*sets) if sets else set()
    if not common:
        raise RuntimeError("No common sentences across runs; cannot do paired bootstrap.")
    out = {}
    for k, df in dfs.items():
        out[k] = df[df["sentence_key"].isin(common)].set_index("sentence_key").sort_index()
    idx = sorted(list(common))
    return idx, out

def agg_point_estimates(df: pd.DataFrame) -> Dict[str,float]:
    """
    sentence_metrics.tsv -> window-level (micro) metrics:
    - 用 n_windows 作为权重，把每个 sentence 行还原成 per-window 统计
    """
    if "n_windows" in df.columns:
        w = df["n_windows"].to_numpy().astype(float)
        w_sum = w.sum()
        if w_sum <= 0:
            # 极端情况就退回原来的简单双均值
            return {
                "R1":  float(df["R1"].mean()),
                "R5":  float(df["R5"].mean()),
                "R10": float(df["R10"].mean()),
                "MRR": float(df["MRR"].mean()),
                "MedR": float(df["MedR"].median()),
            }

        # 这些就是按 window 数加权的 micro-average，
        # 等价于直接在所有窗口上算 hit@k / MRR
        R1  = np.average(df["R1"],  weights=w)   # sum(R1_s * n_s) / sum(n_s)
        R5  = np.average(df["R5"],  weights=w)
        R10 = np.average(df["R10"], weights=w)
        MRR = np.average(df["MRR"], weights=w)

        # MedR 严格的 per-window median 需要原始 ranks，没法从汇总完全还原；
        # 这里先保留句级 median（或者做一个加权 median 近似，都不会影响 R@k / MRR）。
        MedR = float(df["MedR"].median())

        return {
            "R1":  float(R1),
            "R5":  float(R5),
            "R10": float(R10),
            "MRR": float(MRR),
            "MedR": MedR,
        }

    # 没有 n_windows 时退回旧行为（以防其它任务重用这个脚本）
    out = {
        "R1":  float(df["R1"].mean()),
        "R5":  float(df["R5"].mean()),
        "R10": float(df["R10"].mean()),
        "MRR": float(df["MRR"].mean()),
        "MedR": float(df["MedR"].median()),
    }
    return out


def paired_bootstrap(idx: List[str], dfs: Dict[str,pd.DataFrame], base_key: str,
                     B: int = 10000, seed: int = 42) -> Dict[str, Dict[str, Dict[str,float]]]:
    rng = np.random.default_rng(seed)
    res = {}
    base = dfs[base_key]

    # point estimates
    for key, df in dfs.items():
        pe = agg_point_estimates(df.loc[idx])
        res.setdefault(key, {})["point"] = pe

    # deltas vs. baseline
    for key, df in dfs.items():
        if key == base_key: 
            continue
        delta = {m: [] for m in METRICS}
        n = len(idx)
        for _ in range(B):
            samp_idx = rng.choice(n, size=n, replace=True)
            I = [idx[i] for i in samp_idx]
            b_pt = agg_point_estimates(base.loc[I])
            k_pt = agg_point_estimates(df.loc[I])
            for m in METRICS:
                delta[m].append(k_pt[m] - b_pt[m])

        out = {}
        for m in METRICS:
            arr = np.array(delta[m], dtype=float)
            lo, hi = np.quantile(arr, [0.025, 0.975])
            # two-sided p under H0: delta=0
            p_left  = (arr <= 0).mean()
            p_right = (arr >= 0).mean()
            p_two = 2 * min(p_left, p_right)
            out[m] = {
                "d_mean": float(arr.mean()),
                "d_med": float(np.median(arr)),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "p": float(p_two)
            }
        res[key]["vs_"+base_key] = out
    return res

# ---------- formatting ----------

def _fmt_p(pv: float, B: int) -> str:
    """
    If pv==0 (no sign flips), show it as a lower bound based on bootstrap resolution: <1/(B+1)
    Otherwise: scientific if <1e-4, else 4-decimal fixed.
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

def _fmt_entry_abs(point: float, abs_lo: float, abs_hi: float,
                   rel_gain_pct: float, p: float, B: int, higher_is_better: bool=True) -> str:
    arrow = "↑" if (rel_gain_pct >= 0 if higher_is_better else rel_gain_pct <= 0) else "↓"
    return f"{point:.2f} [{abs_lo:.2f}–{abs_hi:.2f}] ({arrow}{abs(rel_gain_pct):.1f}%) ; p={_fmt_p(p, B)}"

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Pairs like LABEL=RUN_DIR ... e.g., baseline=/path/ ... gcb_only=/path/ ...")
    ap.add_argument("--baseline", required=True, help="Label name of baseline")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_tex", type=str, default="")
    args = ap.parse_args()

    # parse run map
    label2dir: Dict[str,Path] = {}
    for item in args.runs:
        if "=" not in item:
            print(f"Bad --runs item: {item}", file=sys.stderr); sys.exit(2)
        lab, path = item.split("=",1)
        label2dir[lab] = Path(path)

    # load per-label sentence tables
    dfs = {}
    resolved = {}
    for lab, p in label2dir.items():
        rr_dir = _ensure_rr_dir(p)
        tsv = _resolve_sentence_tsv(rr_dir, lab)
        resolved[lab] = tsv
        dfs[lab] = pd.read_csv(tsv, sep="\t")

    print("\n=== Resolved TSV paths ===")
    for k,v in resolved.items():
        print(f"{k}: {v.as_posix()}")

    # pair by sentence_key
    idx, dfs2 = intersect_sentences(dfs)

    # run bootstrap
    stats = paired_bootstrap(idx, dfs2, base_key=args.baseline, B=args.B, seed=args.seed)

    # summary
    print("\n=== Point estimates (sentence-averaged) ===")
    for lab in label2dir:
        pe = stats[lab]["point"]
        print(f"[{lab}] R@1={pe['R1']:.4f} R@5={pe['R5']:.4f} R@10={pe['R10']:.4f} MRR={pe['MRR']:.4f} MedR={pe['MedR']:.2f}")

    # build LaTeX
    baseline_pe = stats[args.baseline]["point"]
    labels = list(label2dir.keys())

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Retrieval under zero-shot decoding with sentence-level paired bootstrap (95\\% CI) and p-values vs. baseline.}")
    lines.append("\\label{tab:retrieval_zeroshot_bootstrap}")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.1}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Configuration} & \\textbf{R@1} & \\textbf{R@5} & \\textbf{R@10} & \\textbf{MRR} & \\textbf{MedR} \\\\")
    lines.append("\\midrule")

    def rel(base, now): 
        return 100.0*(now-base)/max(1e-12, base)

    for lab in labels:
        pe = stats[lab]["point"]
        if lab == args.baseline:
            row = f"{lab} & {pe['R1']:.2f} & {pe['R5']:.2f} & {pe['R10']:.2f} & {pe['MRR']:.2f} & {pe['MedR']:.1f} \\\\"
            lines.append(row)
        else:
            comp = stats[lab]["vs_"+args.baseline]

            # absolute CI bounds are baseline_point + [Δ_lo, Δ_hi]
            r1_abs_lo  = baseline_pe["R1"]  + comp["R1"]["ci_lo"]
            r1_abs_hi  = baseline_pe["R1"]  + comp["R1"]["ci_hi"]
            r5_abs_lo  = baseline_pe["R5"]  + comp["R5"]["ci_lo"]
            r5_abs_hi  = baseline_pe["R5"]  + comp["R5"]["ci_hi"]
            r10_abs_lo = baseline_pe["R10"] + comp["R10"]["ci_lo"]
            r10_abs_hi = baseline_pe["R10"] + comp["R10"]["ci_hi"]
            mrr_abs_lo = baseline_pe["MRR"] + comp["MRR"]["ci_lo"]
            mrr_abs_hi = baseline_pe["MRR"] + comp["MRR"]["ci_hi"]

            r1  = _fmt_entry_abs(pe["R1"],  r1_abs_lo,  r1_abs_hi,  rel(baseline_pe["R1"],  pe["R1"]),  comp["R1"]["p"],  args.B, True)
            r5  = _fmt_entry_abs(pe["R5"],  r5_abs_lo,  r5_abs_hi,  rel(baseline_pe["R5"],  pe["R5"]),  comp["R5"]["p"],  args.B, True)
            r10 = _fmt_entry_abs(pe["R10"], r10_abs_lo, r10_abs_hi, rel(baseline_pe["R10"], pe["R10"]), comp["R10"]["p"], args.B, True)
            mrr = _fmt_entry_abs(pe["MRR"], mrr_abs_lo, mrr_abs_hi, rel(baseline_pe["MRR"], pe["MRR"]), comp["MRR"]["p"], args.B, True)

            # MedR: smaller is better → 相对“下降”为正向改进
            med_rel = -100.0*(pe["MedR"] - baseline_pe["MedR"]) / max(1e-12, baseline_pe["MedR"])
            med_abs_lo = baseline_pe["MedR"] + comp["MedR"]["ci_lo"]
            med_abs_hi = baseline_pe["MedR"] + comp["MedR"]["ci_hi"]
            med_entry = f"{pe['MedR']:.1f} [{med_abs_lo:.1f}–{med_abs_hi:.1f}] ({('↑' if med_rel>=0 else '↓')}{abs(med_rel):.1f}%) ; p={_fmt_p(comp['MedR']['p'], args.B)}"

            row = f"{lab} & {r1} & {r5} & {r10} & {mrr} & {med_entry} \\\\"
            lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    print("\n=== LaTeX Table Snippet ===\n")
    print(tex)
    if args.out_tex:
        Path(args.out_tex).write_text(tex, encoding="utf-8")
        print(f"\n[INFO] LaTeX table written to: {args.out_tex}")

if __name__ == "__main__":
    main()
