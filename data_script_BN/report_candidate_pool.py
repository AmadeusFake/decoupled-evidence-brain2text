#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_candidate_pool.py (optimized)

This script reports the size of the content-level candidate pool and checks
cross-split temporal overlap of 3-second audio windows.

Functions:
- Count content-unique 3s windows per split (train / valid / test)
- Efficiently detect cross-split overlap using audio-level bucketing and sorting
"""

import argparse, json
from pathlib import Path
from collections import defaultdict


def read_jsonl(p: Path):
    """Yield rows from a JSONL file."""
    with open(p, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def content_id_of(r: dict) -> str:
    """
    Build a content-level identifier for a window.

    Preference:
    - Use explicit `content_id` if provided
    - Otherwise derive from audio path and onset/offset times
    """
    if "content_id" in r and r["content_id"]:
        return r["content_id"]
    return (
        f"{r['original_audio_path']}::"
        f"{float(r['local_window_onset_in_audio_s']):.3f}-"
        f"{float(r['local_window_offset_in_audio_s']):.3f}"
    )


def overlap(a0, a1, b0, b1):
    """Return the temporal overlap duration between two intervals."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_audio_buckets(rows):
    """
    Bucket windows by audio path and sort each bucket by onset time.
    """
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["original_audio_path"]].append(
            (
                float(r["local_window_onset_in_audio_s"]),
                float(r["local_window_offset_in_audio_s"]),
            )
        )
    for k in buckets:
        buckets[k].sort(key=lambda x: x[0])
    return buckets


def count_cross_overlap(split_rows, test_buckets):
    """
    Efficiently detect temporal overlap between a split and test windows.

    Uses two-pointer scanning within each audio bucket.
    """
    cnt = 0
    for audio_path, intervals in test_buckets.items():
        if not intervals:
            continue

        t_idx = 0
        t_len = len(intervals)

        # Extract windows from the current split for this audio file
        s_rows = [
            (
                float(r["local_window_onset_in_audio_s"]),
                float(r["local_window_offset_in_audio_s"]),
            )
            for r in split_rows
            if r["original_audio_path"] == audio_path
        ]

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

    # Report content-unique candidate pool size per split
    for split, lst in rows.items():
        uniq = set(content_id_of(r) for r in lst)
        print(f"[{split}] rows={len(lst):,} | unique_content={len(uniq):,}")

    # Efficient cross-split overlap check against test set
    test_buckets = build_audio_buckets(rows["test"])
    for split in ("train", "valid"):
        cnt = count_cross_overlap(rows[split], test_buckets)
        print(f"[{split}] cross-overlap with TEST windows: {cnt:,}")


if __name__ == "__main__":
    main()
