#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
verify_slr_kfold.py — Sanity checker for SLR v4 k-fold splits.

Per-fold checks:
  1) No recording overlap across splits
     (recording = subject_id + session_id)
  2) No window_id overlap across splits
  3) Every subject appears in TRAIN (hard constraint)
  4) Covered-test-only: contents(test) ⊆ contents(train)
  5) Recording-count ratios ≈ target (± tolerance)
  6) Summary statistics per split:
     rows, unique subjects, recordings, contents

Cross-fold checks:
  A) Union of TEST subjects across folds
  B) Subjects with only one recording
     (cannot be placed into TEST when require_subject_in_train is enforced)
  C) Whether all subjects with ≥2 recordings appear in TEST
     across folds (expected)

Usage:
  python verify_slr_kfold.py \
    --root /path/to/resplit_sentence_a025_SLR_kfold \
    --target_ratios 0.7,0.1,0.2
"""

import argparse, json, os, glob, math, sys
from collections import defaultdict


# -------------------- IO -------------------- #

def read_jsonl(p):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


# -------------------- Keys -------------------- #

def rec_key(r):
    """Recording identifier: subject_id + session_id."""
    sid = str(r.get("subject_id"))
    ses = str(r.get("session_id"))
    if sid is not None and ses is not None:
        return f"{sid}_{ses}"

    # Fallback: parse window_id like "01_0_..."
    wid = str(r.get("window_id", ""))
    parts = wid.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    raise ValueError("Cannot derive recording key")

def content_id(r):
    """Content identifier (audio path + local window time)."""
    if r.get("content_id"):
        return str(r["content_id"])
    a = r["original_audio_path"]
    on = float(r["local_window_onset_in_audio_s"])
    off = float(r["local_window_offset_in_audio_s"])
    return f"{a}::{on:.3f}-{off:.3f}"


# -------------------- Uniques -------------------- #

def uniq_subjects(rows):
    return sorted({str(r["subject_id"]) for r in rows})

def uniq_recordings(rows):
    return sorted({rec_key(r) for r in rows})

def uniq_contents(rows):
    return sorted({content_id(r) for r in rows})

def uniq_window_ids(rows):
    return sorted({str(r.get("window_id", "")) for r in rows})


# -------------------- Fold loading -------------------- #

def load_fold(fold_dir):
    paths = {s: os.path.join(fold_dir, f"{s}.jsonl") for s in ("train", "valid", "test")}
    for s, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}")
    data = {s: read_jsonl(paths[s]) for s in ("train", "valid", "test")}
    return data


# -------------------- Per-fold checks -------------------- #

def check_fold(fold_dir, target_ratios=None, ratio_tolerance=1):
    print(f"\n== Check {fold_dir} ==")
    data = load_fold(fold_dir)

    stats = {}
    for s in ("train", "valid", "test"):
        rows = data[s]
        stats[s] = {
            "rows": len(rows),
            "subjects": uniq_subjects(rows),
            "n_subjects": len(uniq_subjects(rows)),
            "recordings": uniq_recordings(rows),
            "n_recordings": len(uniq_recordings(rows)),
            "contents": uniq_contents(rows),
            "n_contents": len(uniq_contents(rows)),
            "window_ids": uniq_window_ids(rows),
            "n_window_ids": len(uniq_window_ids(rows)),
        }
        print(
            f"[{s}] rows={stats[s]['rows']:,}  "
            f"unique_subjects={stats[s]['n_subjects']}  "
            f"unique_recordings={stats[s]['n_recordings']}  "
            f"unique_contents={stats[s]['n_contents']}"
        )

    # 1) Recording overlap
    recs = {s: set(stats[s]["recordings"]) for s in ("train", "valid", "test")}
    assert recs["train"].isdisjoint(recs["valid"]), "Recording leak: train∩valid != ∅"
    assert recs["train"].isdisjoint(recs["test"]),  "Recording leak: train∩test  != ∅"
    assert recs["valid"].isdisjoint(recs["test"]),  "Recording leak: valid∩test  != ∅"
    print("[OK] No recording overlap across splits.")

    # 2) window_id overlap
    wins = {s: set(stats[s]["window_ids"]) for s in ("train", "valid", "test")}
    assert wins["train"].isdisjoint(wins["valid"]), "window_id leak: train∩valid != ∅"
    assert wins["train"].isdisjoint(wins["test"]),  "window_id leak: train∩test  != ∅"
    assert wins["valid"].isdisjoint(wins["test"]),  "window_id leak: valid∩test  != ∅"
    print("[OK] No window_id overlap across splits.")

    # 3) Every subject appears in TRAIN
    subs_all = sorted(
        set().union(*[set(stats[s]["subjects"]) for s in ("train", "valid", "test")])
    )
    missing_in_train = [sid for sid in subs_all if sid not in set(stats["train"]["subjects"])]
    assert not missing_in_train, f"Subjects not in TRAIN: {missing_in_train}"
    print("[OK] Every subject appears in TRAIN.")

    # 4) Covered test only
    cont_train = set(stats["train"]["contents"])
    cont_test = set(stats["test"]["contents"])
    leak = sorted(list(cont_test - cont_train))
    assert not leak, f"Found {len(leak)} test contents not covered by train."
    print("[OK] contents(test) ⊆ contents(train).")

    # 5) Recording-count ratio check (± tolerance)
    if target_ratios:
        n_total = sum(stats[s]["n_recordings"] for s in ("train", "valid", "test"))
        tgt = [target_ratios[0] * n_total,
               target_ratios[1] * n_total,
               target_ratios[2] * n_total]
        floor = [math.floor(x) for x in tgt]

        # Largest-remainder rounding
        rem = n_total - sum(floor)
        fracs = sorted(
            enumerate([tgt[i] - floor[i] for i in range(3)]),
            key=lambda x: x[1],
            reverse=True,
        )
        for i in range(rem):
            floor[fracs[i][0]] += 1

        got = [
            stats["train"]["n_recordings"],
            stats["valid"]["n_recordings"],
            stats["test"]["n_recordings"],
        ]
        diff = [abs(got[i] - floor[i]) for i in range(3)]
        ok = all(d <= ratio_tolerance for d in diff)

        msg = (
            f"[ratio] recordings  target≈{tuple(floor)}  "
            f"got={tuple(got)}  diff={tuple(diff)}  tol=±{ratio_tolerance}"
        )
        print(msg)
        assert ok, "Recording count ratios deviate beyond tolerance."

    return {
        "subjects_all": subs_all,
        "subjects_train": set(stats["train"]["subjects"]),
        "subjects_valid": set(stats["valid"]["subjects"]),
        "subjects_test":  set(stats["test"]["subjects"]),
        "recordings_all": set().union(
            *[set(stats[s]["recordings"]) for s in ("train", "valid", "test")]
        ),
    }


# -------------------- Cross-fold helpers -------------------- #

def per_subject_recording_counts(fold_dir):
    data = load_fold(fold_dir)
    recs_by_subj = defaultdict(set)
    for s in ("train", "valid", "test"):
        for r in data[s]:
            sid = str(r["subject_id"])
            recs_by_subj[sid].add(rec_key(r))
    return {sid: len(recs) for sid, recs in recs_by_subj.items()}


# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root directory containing fold*/{train,valid,test}.jsonl")
    ap.add_argument("--target_ratios", default="0.7,0.1,0.2")
    ap.add_argument("--ratio_tolerance", type=int, default=1)
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.target_ratios.split(","))
    folds = sorted(
        [d for d in glob.glob(os.path.join(args.root, "fold*")) if os.path.isdir(d)]
    )
    if not folds:
        print(f"[ERROR] No folds found under {args.root}")
        sys.exit(2)

    print(f"[verify] root={args.root}  folds={len(folds)}  target_ratios={ratios}")

    # Dataset-intrinsic subject recording counts
    subj_rec_counts = per_subject_recording_counts(folds[0])

    union_test_subjects = set()
    all_subjects = None

    for fd in folds:
        agg = check_fold(fd, target_ratios=ratios, ratio_tolerance=args.ratio_tolerance)
        if all_subjects is None:
            all_subjects = set(agg["subjects_all"])
        union_test_subjects |= set(agg["subjects_test"])

    # Cross-fold summary
    n_total_subjects = len(all_subjects or [])
    subjects_ge2 = sorted([sid for sid, c in subj_rec_counts.items() if c >= 2])
    subjects_eq1 = sorted([sid for sid, c in subj_rec_counts.items() if c == 1])

    print("\n==== Cross-fold summary ====")
    print(f"Subjects total: {n_total_subjects}")
    print(f"Subjects with ≥2 recordings: {len(subjects_ge2)}")
    print(
        f"Subjects with 1 recording : {len(subjects_eq1)}  "
        f"-> cannot be placed in TEST with require_subject_in_train"
    )
    print(f"Union of TEST subjects across folds: {len(union_test_subjects)}")

    miss_from_union = sorted(list(set(subjects_ge2) - union_test_subjects))
    if miss_from_union:
        print(
            "[WARN] Subjects (≥2 rec) that never appeared in TEST across folds: "
            f"{miss_from_union}"
        )
    else:
        print("[OK] All subjects with ≥2 recordings appear in TEST across folds.")

    print("\n[PASS] All folds satisfy SLR constraints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
