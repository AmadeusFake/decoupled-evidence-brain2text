#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize sweep runs (robust)

- Automatically search for an epoch-level JSONL under both the run root and run_dir/records/.
  Multiple common filenames are supported.
- Scan the JSONL line-by-line and compute best_val_mrr / best_val_top1 / best_val_top10 ourselves
  (no dependency on training-time "best_*" being logged).
- Use the `monitor` field from config_init.json (mrr/top1/top10) to decide `best_metric` for sorting.
- Compatible with current field names:
  val_self_mrr / val_self_top1 / val_self_top10 / train_loss / val_loss.
"""

from __future__ import annotations
import json, csv, argparse, math
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple

# Candidate filenames to look for (search in run root and run_dir/records/)
CANDIDATE_JSONL = (
    "epoch_metrics.jsonl",
    "metrics_epoch.jsonl",
    "epochs.jsonl",
    "metrics.jsonl",
    "metric_epoch.jsonl",
)

def _to_float(x) -> Optional[float]:
    try:
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        pass
    return None

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path or not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue

def _find_epoch_file(run_dir: Path) -> Optional[Path]:
    # 1) search in run root
    for name in CANDIDATE_JSONL:
        p = run_dir / name
        if p.exists():
            return p
    # 2) search in records/
    rec = run_dir / "records"
    for name in CANDIDATE_JSONL:
        p = rec / name
        if p.exists():
            return p
    # 3) fallback: pick the first "*.jsonl" that is parseable under run root / records
    for base in (run_dir, rec):
        for p in sorted(base.glob("*.jsonl")):
            for _ in _iter_jsonl(p):
                return p
    return None

def _load_config(run_dir: Path) -> Dict[str, Any]:
    # Prefer records/config_init.json; fallback to run_root/config_init.json
    for p in (run_dir / "records" / "config_init.json", run_dir / "config_init.json"):
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                args = data.get("args", {})
                return {
                    "experiment_name": args.get("experiment_name"),
                    "monitor": (args.get("monitor") or "mrr").lower(),
                    "warmup_ratio": args.get("warmup_ratio"),
                    "fe_lr_mult": args.get("fe_lr_mult"),
                    "freeze_backbone_epochs": args.get("freeze_backbone_epochs"),
                    "tau": args.get("tau"),
                    "learnable_temp": bool(args.get("learnable_temp", False)),
                    "amp": args.get("amp"),
                    "lr": args.get("lr"),
                    "weight_decay": args.get("weight_decay"),
                    "batch_size": args.get("batch_size"),
                    "d_model": args.get("d_model"),
                    "backbone_type": args.get("backbone_type"),
                    "backbone_depth": args.get("backbone_depth"),
                    "tpp_slots": args.get("tpp_slots"),
                }
            except Exception:
                pass
    return {"monitor": "mrr"}

def _scan_metrics(epoch_file: Path) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Returns:
      last_vals: last available values {val_mrr, val_top1, val_top10, val_loss, train_loss, epoch}
      bests:     {'best_val_mrr', 'best_val_top1', 'best_val_top10'}
    """
    last_epoch = None
    last_vals: Dict[str, Any] = {}
    best = {"best_val_mrr": float("-inf"), "best_val_top1": float("-inf"), "best_val_top10": float("-inf")}

    for row in _iter_jsonl(epoch_file):
        ep = row.get("epoch")
        if ep is None:
            continue

        # Field-name compatibility
        cur_mrr   = _to_float(row.get("val_self_mrr")   or row.get("val_mrr"))
        cur_top1  = _to_float(row.get("val_self_top1")  or row.get("val_top1"))
        cur_top10 = _to_float(row.get("val_self_top10") or row.get("val_top10"))
        cur_vloss = _to_float(row.get("val_loss"))
        cur_tloss = _to_float(row.get("train_loss"))

        # Update bests
        if cur_mrr   is not None:  best["best_val_mrr"]   = max(best["best_val_mrr"],   cur_mrr)
        if cur_top1  is not None:  best["best_val_top1"]  = max(best["best_val_top1"],  cur_top1)
        if cur_top10 is not None:  best["best_val_top10"] = max(best["best_val_top10"], cur_top10)

        # Track the "last row"
        last_epoch = ep
        last_vals = {
            "epoch": ep,
            "val_mrr": cur_mrr,
            "val_top1": cur_top1,
            "val_top10": cur_top10,
            "val_loss": cur_vloss,
            "train_loss": cur_tloss,
        }

        # Some training scripts may already log best_val_* fields.
        # If present, take max with our running best (keeps an upper envelope; does not reduce our computed best).
        for k in ("best_val_mrr", "best_val_top1", "best_val_top10"):
            v = _to_float(row.get(k))
            if v is not None:
                best[k] = max(best[k], v)

    # Convert missing bests to None
    for k in list(best.keys()):
        if best[k] == float("-inf"):
            best[k] = None
    return last_vals, best

def summarize_runs(runs_dir: Path, prefix: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in sorted(runs_dir.iterdir()):
        if not run.is_dir():
            continue
        if prefix and not run.name.startswith(prefix):
            continue

        epoch_file = _find_epoch_file(run)
        if not epoch_file:
            continue

        cfg = _load_config(run)
        last, bests = _scan_metrics(epoch_file)

        # Pick best_metric according to the monitor field
        mon = (cfg.get("monitor") or "mrr").lower()
        if mon not in ("mrr", "top1", "top10"):
            mon = "mrr"
        mon_key = {"mrr": "best_val_mrr", "top1": "best_val_top1", "top10": "best_val_top10"}[mon]
        best_metric = bests.get(mon_key)

        # Emit one summary row
        rows.append({
            "run_dir": run.name,
            "epoch_file": str(epoch_file.relative_to(run)),
            "experiment_name": cfg.get("experiment_name"),
            "monitor": mon,
            "best_metric": best_metric,
            "last_epoch": last.get("epoch"),
            "val_mrr": last.get("val_mrr"),
            "val_top1": last.get("val_top1"),
            "val_top10": last.get("val_top10"),
            "val_loss": last.get("val_loss"),
            "train_loss": last.get("train_loss"),
            "best_val_mrr": bests.get("best_val_mrr"),
            "best_val_top1": bests.get("best_val_top1"),
            "best_val_top10": bests.get("best_val_top10"),
            # Hyperparameters (for quick comparison)
            "warmup_ratio": cfg.get("warmup_ratio"),
            "fe_lr_mult": cfg.get("fe_lr_mult"),
            "freeze_backbone_epochs": cfg.get("freeze_backbone_epochs"),
            "tau": cfg.get("tau"),
            "learnable_temp": cfg.get("learnable_temp"),
            "lr": cfg.get("lr"),
            "weight_decay": cfg.get("weight_decay"),
            "batch_size": cfg.get("batch_size"),
            "d_model": cfg.get("d_model"),
            "backbone_type": cfg.get("backbone_type"),
            "backbone_depth": cfg.get("backbone_depth"),
            "tpp_slots": cfg.get("tpp_slots"),
        })

    # Sort by best_metric (descending); None goes last
    rows.sort(key=lambda r: (-1 if r["best_metric"] is None else 0, r["best_metric"] or -1), reverse=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--out-csv", type=str, default="sweep_summary.csv")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    rows = summarize_runs(runs_dir, args.prefix or None)

    if not rows:
        print("[WARN] No runs found to summarize. Check --runs-dir or --prefix.")
        return

    out_csv = runs_dir / args.out_csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote: {out_csv}  (total {len(rows)} runs)\n")
    print(f"Top-{args.topk}: (monitor={rows[0]['monitor']}, best_metric DESC)")
    for i, r in enumerate(rows[:args.topk], 1):
        lt = 1 if r.get("learnable_temp") else 0
        print(f"{i:>2}. {r['run_dir']} | best={r['best_metric']} "
              f"| last val: mrr={r['val_mrr']} top1={r['val_top1']} top10={r['val_top10']} "
              f"| warmup={r['warmup_ratio']} fe={r['fe_lr_mult']} fr={r['freeze_backbone_epochs']} "
              f"| tau={r['tau']} lt={lt}")

if __name__ == "__main__":
    main()
