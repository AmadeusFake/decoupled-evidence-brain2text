#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_candidate_pool.py

Report content-unique 3s window counts per split (candidate pool),
and efficiently detect cross-split temporal overlap with TEST windows.
"""

import argparse, json
from pathlib import Path
from collections import defaultdict

def read_jsonl(p: Path):
    with open(p, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def content_id_of(r: dict) -> str:
    """Return content-level unique ID for a window."""
    if "content_id" in r and r["content_id"]:
        return r["content_id"]
    return (
        f"{r['original_audio_path']}::"
        f"{float(r['local_window_onset_in_audio_s']):.3f}-"
        f"{float(r['local_window_offset_in_audio_s']):.3f}"
    )

def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))

def build_audio_buckets(rows):
    """Group windows by audio path and sort by onset."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["original_audio_path"]].append((
            float(r["local_window_onset_in_audio_s"]),
            float(r["local_window_offset_in_audio_s"])
        ))
    for k in buckets:
        buckets[k].sort(key=lambda x: x[0])
    return buckets

def count_cross_overlap(split_rows, test_buckets):
    """Count windows overlapping with TEST windows on the same audio."""
    cnt = 0
    for audio_path, intervals in test_buckets.items():
        if not intervals:
            continue
        t_idx = 0
        t_len = len(intervals)

        # Windows from this split on the same audio
        s_rows = [(
            float(r["local_window_onset_in_audio_s"]),
            float(r["local_window_offset_in_audio_s"])
        ) for r in split_rows if r["original_audio_path"] == audio_path]

        for a0, a1 in s_rows:
            while t_idx < t_len and intervals[t_idx][1] < a0:
                t_idx += 1
            j = t_idx
            while j < t_len and intervals[j][0] <= a1:
                if overlap(a0, a1, intervals[j][0], intervals[j][1]) > 0:
                    cnt += 1
                    break
                j += 1
    return cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        required=True,
        help="Directory containing train.jsonl / valid.jsonl / test.jsonl",
    )
    args = ap.parse_args()

    base = Path(args.dir)
    rows = {
        split: list(read_jsonl(base / f"{split}.jsonl"))
        for split in ("train", "valid", "test")
    }

    for split, lst in rows.items():
        uniq = set(content_id_of(r) for r in lst)
        print(f"[{split}] rows={len(lst):,} | unique_content={len(uniq):,}")

    # Cross-split overlap check (train/valid vs test)
    test_buckets = build_audio_buckets(rows["test"])
    for split in ("train", "valid"):
        cnt = count_cross_overlap(rows[split], test_buckets)
        print(f"[{split}] cross-overlap with TEST windows: {cnt:,}")

if __name__ == "__main__":
    main()
