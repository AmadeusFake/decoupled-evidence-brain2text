#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/verify_sensor_coords_fast.py

Fast consistency checker:
- Group by (subject_id, session_id); load each coordinate file once (mmap).
- Verify that all rows in a session share the same sensor_coordinates_path.
- Spot-check a small number of MEG windows (default: 2) for channel count match (mmap),
  avoiding full-table per-row I/O.
- Aggregate issues and optionally export a CSV report.

Usage:
python tools/verify_sensor_coords_fast.py \
  --manifest_dir /PATH/final_splits_sentence_fully_preprocessed \
  --report_csv /PATH/reports/sensor_fast_check.csv \
  --spot_check_per_session 2 \
  --num_workers 8
"""

import argparse, json, os
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np


# -------------------- IO -------------------- #

def load_jsonl_rows(p: Path):
    rows = []
    with open(p, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# -------------------- Grouping -------------------- #

def group_by_session(rows):
    """Group rows by (subject_id, session_id)."""
    sess = defaultdict(list)
    for r in rows:
        sid = str(r.get("subject_id", ""))
        ses = str(r.get("session_id", ""))
        sess[(sid, ses)].append({
            "meg": r.get("meg_win_path", ""),
            "coord": r.get("sensor_coordinates_path", ""),
            "wid": r.get("window_id", ""),
        })
    return sess


# -------------------- NPY helpers -------------------- #

def npy_shape(path: str):
    """Read array shape via mmap without loading data."""
    arr = np.load(path, mmap_mode="r")
    return arr.shape


# -------------------- Per-session check -------------------- #

def _check_one_session(args):
    (sid, ses), items, spot = args
    issues = []

    # 1) Coordinate path consistency
    coord_paths = {it["coord"] for it in items if it["coord"]}
    if not coord_paths:
        issues.append((sid, ses, "MISSING_COORD_PATH", "", ""))
        return sid, ses, issues, 0
    if len(coord_paths) != 1:
        issues.append((
            sid,
            ses,
            f"COORD_PATH_INCONSISTENT({len(coord_paths)})",
            ";".join(sorted(coord_paths)),
            "",
        ))
        # Continue with one path for further checks

    coord_path = sorted(coord_paths)[0]
    if not Path(coord_path).exists():
        issues.append((sid, ses, "COORD_FILE_NOT_FOUND", coord_path, ""))
        return sid, ses, issues, 0

    # 2) Load coordinate shape
    try:
        Ccoords = npy_shape(coord_path)[0]  # expected [C, 3]
    except Exception as e:
        issues.append((sid, ses, f"COORD_LOAD_FAIL:{e}", coord_path, ""))
        return sid, ses, issues, 0

    # 3) Spot-check MEG window channel count
    checked = 0
    for it in items[:spot]:
        mp = it["meg"]
        if not mp or not Path(mp).exists():
            issues.append((sid, ses, "MEG_WIN_MISSING", coord_path, mp))
            continue
        try:
            shp = npy_shape(mp)  # [C, T] or [T, C]
        except Exception as e:
            issues.append((sid, ses, f"MEG_LOAD_FAIL:{e}", coord_path, mp))
            continue
        if len(shp) != 2:
            issues.append((sid, ses, f"MEG_SHAPE_INVALID:{shp}", coord_path, mp))
            continue

        Cmeg = min(shp[0], shp[1])  # channel dimension estimate
        if Cmeg != Ccoords:
            issues.append((
                sid,
                ses,
                f"CHANNEL_MISMATCH:coords={Ccoords},meg={Cmeg}",
                coord_path,
                mp,
            ))
        checked += 1

    return sid, ses, issues, checked


# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--report_csv", default="")
    ap.add_argument("--spot_check_per_session", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=min(8, cpu_count()))
    args = ap.parse_args()

    md = Path(args.manifest_dir)
    rows = []
    for name in ("train", "valid", "test"):
        p = md / f"{name}.jsonl"
        if p.exists():
            rows.extend(load_jsonl_rows(p))
        else:
            print(f"[WARN] {p} not found; skipping split")

    sess = group_by_session(rows)
    tasks = [
        ((sid, ses), items, args.spot_check_per_session)
        for (sid, ses), items in sess.items()
    ]

    issues_all = []
    checked_cnt = 0

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for sid, ses, issues, ck in pool.imap_unordered(_check_one_session, tasks):
                issues_all.extend(issues)
                checked_cnt += ck
    else:
        for t in tasks:
            sid, ses, issues, ck = _check_one_session(t)
            issues_all.extend(issues)
            checked_cnt += ck

    print("====================================================")
    print(f"Total sessions: {len(sess)}")
    print(f"Spot-checked MEG windows: {checked_cnt}")
    print(f"Issues found: {len(issues_all)}")
    if issues_all:
        for it in issues_all[:20]:
            print("ISSUE:", it)
    print("====================================================")

    if args.report_csv:
        outp = Path(args.report_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            f.write("subject,session,issue,coord_path,meg_path\n")
            for sid, ses, issue, cpath, mpath in issues_all:
                f.write(f"{sid},{ses},{issue},{cpath},{mpath}\n")
        print(f"[REPORT WRITTEN] {outp}")


if __name__ == "__main__":
    main()
