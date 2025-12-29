#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock benchmark for retrieval scorer + GCB using your exact implementation:

    from eval.retrieval_window_vote import
        compute_logits_clip, _precompute_sentence_buckets, gcb_apply_to_group

What it measures (per configuration):
  1) scorer-only time: compute_logits_clip(Y, A)
  2) scorer + GCB time: compute_logits_clip(Y, A) + gcb_apply_to_group(...)
  3) GCB overhead = (2) - (1)

It generates mock data:
  - queries:    Y in shape [B, D, T]
  - candidates: A in shape [O, D, T]
  - sentence ids for candidates: cand_sent_idx_o in shape [O]
      with a simple grouping rule: sent_id = idx // sent_size

Outputs:
  - Console summary
  - Optional CSV for paper-ready tables/plots

Typical usage:
  python bench_mock_complexity.py --device cuda --dtype bf16 --B 8 16 32 --O 1464 5000 --T 360 --D 1024 --csv_out bench.csv
"""

from __future__ import annotations
import os
import sys
import time
import math
import argparse
import platform
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch


# -----------------------------------------------------------------------------
# Import from your repo (exactly as requested).
# If "eval" is not discoverable, we add cwd to sys.path and retry.
# -----------------------------------------------------------------------------
def import_eval_funcs():
    try:
        from eval.retrieval_window_vote import (
            compute_logits_clip,
            _precompute_sentence_buckets,
            gcb_apply_to_group,
        )
        return compute_logits_clip, _precompute_sentence_buckets, gcb_apply_to_group
    except ModuleNotFoundError:
        # fallback: add current working directory
        sys.path.insert(0, os.getcwd())
        from eval.retrieval_window_vote import (
            compute_logits_clip,
            _precompute_sentence_buckets,
            gcb_apply_to_group,
        )
        return compute_logits_clip, _precompute_sentence_buckets, gcb_apply_to_group


compute_logits_clip, _precompute_sentence_buckets, gcb_apply_to_group = import_eval_funcs()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _resolve_device(device: str) -> str:
    device = device.lower()
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device.startswith("cuda:"):
        if torch.cuda.is_available():
            return device
        return "cpu"
    return device


def _resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    s = dtype_str.lower()
    if s in ("fp32", "float32"):
        return torch.float32
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _safe_randn(shape: Tuple[int, ...], device: str, dtype: torch.dtype) -> torch.Tensor:
    # On some CPU builds, fp16 might be problematic; we still allow it,
    # but if it errors, fall back to fp32 for generation.
    try:
        return torch.randn(*shape, device=device, dtype=dtype)
    except Exception:
        return torch.randn(*shape, device=device, dtype=torch.float32).to(dtype=dtype)


def _percentile(sorted_vals: List[float], q: float) -> float:
    # q in [0,1]
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w


def _fmt_ms(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if x < 1:
        return f"{x:.3f}"
    if x < 10:
        return f"{x:.3f}"
    return f"{x:.2f}"


def approx_scorer_flops(B: int, O: int, D: int, T: int) -> float:
    """
    Rough FLOPs for compute_logits_clip:
      inv_norms: A.norm over (D*T) per candidate
        - squares + adds ~ 2*(D*T) (very rough)
      einsum dot-product per (B,O): ~2*(D*T)
    Total ~ O * 2*(D*T) + B*O*2*(D*T)

    This is an approximation used only for "order-of-magnitude" context.
    """
    dt = D * T
    flops_norm = O * 2.0 * dt
    flops_dot = B * O * 2.0 * dt
    return flops_norm + flops_dot


@dataclass
class BenchResult:
    device: str
    dtype: str
    B: int
    O: int
    D: int
    T: int
    sent_size: int

    # GCB params
    gcb_topk: int
    gcb_q: float
    gcb_top_m: int
    gcb_topS: int
    gcb_gamma: float
    gcb_sent_norm: str

    # timing stats
    scorer_ms_med: float
    scorer_ms_p10: float
    scorer_ms_p90: float

    total_ms_med: float
    total_ms_p10: float
    total_ms_p90: float

    overhead_ms_med: float
    overhead_pct_med: float

    # approx flops (scorer only)
    scorer_flops: float
    scorer_gflops_achieved: float


def time_op(
    fn,
    device: str,
    warmup: int,
    iters: int,
) -> float:
    # warmup
    for _ in range(warmup):
        _ = fn()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    _sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) / iters


def run_one_config(
    *,
    device: str,
    dtype: torch.dtype,
    dtype_str: str,
    B: int,
    O: int,
    D: int,
    T: int,
    sent_size: int,
    # gcb
    gcb_topk: int,
    gcb_q: float,
    gcb_top_m: int,
    gcb_topS: int,
    gcb_gamma: float,
    gcb_sent_norm: str,
    # timing
    warmup: int,
    iters: int,
    repeats: int,
    seed: int,
) -> BenchResult:

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # mock data
    Y = _safe_randn((B, D, T), device=device, dtype=dtype)
    A = _safe_randn((O, D, T), device=device, dtype=dtype)

    # sentence ids for candidates
    cand_sent_idx_o = (torch.arange(O, device=device) // max(1, int(sent_size))).to(torch.long)

    # buckets (your helper)
    buckets = _precompute_sentence_buckets(cand_sent_idx_o)

    def scorer_only():
        return compute_logits_clip(Y, A, scale=None)

    def scorer_plus_gcb():
        logits = compute_logits_clip(Y, A, scale=None)
        logits2 = gcb_apply_to_group(
            logits,
            cand_sent_idx_o,
            buckets,
            topk=gcb_topk,
            q_quantile=gcb_q,
            top_m=gcb_top_m,
            sent_norm=gcb_sent_norm,
            topS=gcb_topS,
            gamma=gcb_gamma,
        )
        return logits2

    scorer_times = []
    total_times = []

    for r in range(repeats):
        # change seed slightly each repeat (optional)
        torch.manual_seed(seed + r + 1)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed + r + 1)

        scorer_sec = time_op(scorer_only, device=device, warmup=warmup, iters=iters)
        total_sec = time_op(scorer_plus_gcb, device=device, warmup=warmup, iters=iters)

        scorer_times.append(scorer_sec * 1000.0)
        total_times.append(total_sec * 1000.0)

    scorer_times.sort()
    total_times.sort()

    scorer_med = _percentile(scorer_times, 0.5)
    scorer_p10 = _percentile(scorer_times, 0.1)
    scorer_p90 = _percentile(scorer_times, 0.9)

    total_med = _percentile(total_times, 0.5)
    total_p10 = _percentile(total_times, 0.1)
    total_p90 = _percentile(total_times, 0.9)

    overhead_med = total_med - scorer_med
    overhead_pct = (overhead_med / max(1e-9, scorer_med)) * 100.0

    flops = approx_scorer_flops(B, O, D, T)  # FLOPs per forward
    gflops_achieved = (flops / 1e9) / max(1e-12, scorer_med / 1000.0)

    return BenchResult(
        device=device,
        dtype=dtype_str,
        B=B, O=O, D=D, T=T, sent_size=sent_size,
        gcb_topk=gcb_topk,
        gcb_q=gcb_q,
        gcb_top_m=gcb_top_m,
        gcb_topS=gcb_topS,
        gcb_gamma=gcb_gamma,
        gcb_sent_norm=gcb_sent_norm,
        scorer_ms_med=scorer_med,
        scorer_ms_p10=scorer_p10,
        scorer_ms_p90=scorer_p90,
        total_ms_med=total_med,
        total_ms_p10=total_p10,
        total_ms_p90=total_p90,
        overhead_ms_med=overhead_med,
        overhead_pct_med=overhead_pct,
        scorer_flops=flops,
        scorer_gflops_achieved=gflops_achieved,
    )


def to_csv(rows: List[BenchResult], path: str):
    import csv
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def print_env(device: str):
    print("=" * 80)
    print("ENV")
    print(f"  python:   {sys.version.split()[0]}")
    print(f"  torch:    {torch.__version__}")
    print(f"  platform: {platform.platform()}")
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":")[1])
            except Exception:
                idx = 0
        prop = torch.cuda.get_device_properties(idx)
        print(f"  cuda:     {torch.version.cuda}")
        print(f"  gpu:      {prop.name}  (sm={prop.major}.{prop.minor}, mem={prop.total_memory/1e9:.1f} GB)")
    else:
        print("  device:   CPU")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda", help="cuda / cuda:0 / cpu")
    ap.add_argument("--dtype", type=str, default="bf16", help="bf16 / fp16 / fp32")
    ap.add_argument("--seed", type=int, default=0)

    # shapes (grid)
    ap.add_argument("--B", type=int, nargs="+", default=[8, 16, 32], help="query batch sizes")
    ap.add_argument("--O", type=int, nargs="+", default=[1464, 5000], help="candidate pool sizes")
    ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--T", type=int, default=360)
    ap.add_argument("--sent_size", type=int, nargs="+", default=[8, 16], help="windows per sentence in mock grouping")

    # timing
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=7)

    # gcb params (grid-able if you pass multiple values)
    ap.add_argument("--gcb_topk", type=int, nargs="+", default=[128])
    ap.add_argument("--gcb_q", type=float, nargs="+", default=[0.95])
    ap.add_argument("--gcb_top_m", type=int, nargs="+", default=[3])
    ap.add_argument("--gcb_topS", type=int, nargs="+", default=[3])
    ap.add_argument("--gcb_gamma", type=float, nargs="+", default=[0.7])
    ap.add_argument("--gcb_sent_norm", type=str, default="bucket_sqrt")

    # output
    ap.add_argument("--csv_out", type=str, default="", help="optional path to write CSV")
    args = ap.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device=device)
    dtype_str = args.dtype.lower()

    print_env(device)

    rows: List[BenchResult] = []

    # Nested grid over configurations
    for B in args.B:
        for O in args.O:
            for sent_size in args.sent_size:
                for topk in args.gcb_topk:
                    for q in args.gcb_q:
                        for top_m in args.gcb_top_m:
                            for topS in args.gcb_topS:
                                for gamma in args.gcb_gamma:

                                    r = run_one_config(
                                        device=device,
                                        dtype=dtype,
                                        dtype_str=dtype_str,
                                        B=int(B), O=int(O), D=int(args.D), T=int(args.T),
                                        sent_size=int(sent_size),
                                        gcb_topk=int(topk),
                                        gcb_q=float(q),
                                        gcb_top_m=int(top_m),
                                        gcb_topS=int(topS),
                                        gcb_gamma=float(gamma),
                                        gcb_sent_norm=str(args.gcb_sent_norm),
                                        warmup=int(args.warmup),
                                        iters=int(args.iters),
                                        repeats=int(args.repeats),
                                        seed=int(args.seed),
                                    )
                                    rows.append(r)

                                    # Console line
                                    print(
                                        f"[B={r.B:>3} O={r.O:>6} sent={r.sent_size:>3} | "
                                        f"topk={r.gcb_topk:>4} q={r.gcb_q:.2f} m={r.gcb_top_m} S={r.gcb_topS} g={r.gcb_gamma:.2f}]  "
                                        f"scorer { _fmt_ms(r.scorer_ms_med) } ms (p10 { _fmt_ms(r.scorer_ms_p10) }, p90 { _fmt_ms(r.scorer_ms_p90) })  |  "
                                        f"+GCB { _fmt_ms(r.total_ms_med) } ms  |  "
                                        f"over +{ _fmt_ms(r.overhead_ms_med) } ms ({r.overhead_pct_med:.1f}%)  |  "
                                        f"~{r.scorer_gflops_achieved:.1f} GFLOPs/s"
                                    )

    if args.csv_out:
        to_csv(rows, args.csv_out)
        print("=" * 80)
        print(f"CSV written: {args.csv_out}")
        print("=" * 80)

    # A compact "best candidates" hint: sort by overhead pct (lowest)
    if rows:
        best = sorted(rows, key=lambda x: x.overhead_pct_med)[:5]
        print("\nTop-5 lowest GCB overhead (%):")
        for r in best:
            print(
                f"  B={r.B:>3} O={r.O:>6} sent={r.sent_size:>3}  "
                f"scorer={_fmt_ms(r.scorer_ms_med)}ms  total={_fmt_ms(r.total_ms_med)}ms  "
                f"over={_fmt_ms(r.overhead_ms_med)}ms ({r.overhead_pct_med:.1f}%)"
            )


if __name__ == "__main__":
    main()
