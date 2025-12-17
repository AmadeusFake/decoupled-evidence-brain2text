#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_candidate_pool_mous.py

Candidate pool size and content-leakage report for the MOUS dataset.

For each split (train / valid / test), this script reports:
- total_rows:
    Total number of 3-second windows (including all subjects).
- unique_window_contents:
    Number of unique 3-second window contents after removing subject identity.
    Defined by (original_audio_path + local window onset/offset).
    This corresponds to the *true candidate pool size*.
- unique_sentence_contents:
    Number of unique sentence-level contents (≈10 s segments).
    Defined by (original_audio_path + global segment onset/offset).

Additionally:
- avg_windows_per_sentence:
    Average number of 3-second windows per sentence
    (= total_rows / unique_sentence_contents).

Leakage check:
- Detect whether the same sentence_content_id appears in multiple splits.
  If so, those entries are printed (indicating a content split violation).
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict


def read_jsonl(p: Path):
    """Read a JSONL file line by line and yield dict records."""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def audio_window_id(r: dict) -> str:
    """
    Content-level ID for a 3-second window (subject-agnostic):

    - original_audio_path
    - local_window_onset_in_audio_s
    - local_window_offset_in_audio_s
    """
    return (
        f"{r['original_audio_path']}::"
        f"{float(r['local_window_onset_in_audio_s']):.3f}-"
        f"{float(r['local_window_offset_in_audio_s']):.3f}"
    )


def sentence_content_id(r: dict) -> str:
    """
    Sentence-level content ID (~10 s segment).

    For MOUS:
      - Same stimulus sentence ⇒ identical
        original_audio_path + global_segment_[onset,offset]_in_audio_s.

    If sentence-level fields are missing (legacy files),
    return "NA" and exclude from leakage checks.
    """
    try:
        g0 = float(r["global_segment_onset_in_audio_s"])
        g1 = float(r["global_segment_offset_in_audio_s"])
        return f"{r['original_audio_path']}::{g0:.3f}-{g1:.3f}"
    except KeyError:
        return "NA"


def main():
    ap = argparse.ArgumentParser(
        description="MOUS candidate pool size and content leakage report"
    )
    ap.add_argument(
        "--dir",
        required=True,
        help="Directory containing train.jsonl / valid.jsonl / test.jsonl",
    )
    args = ap.parse_args()

    base = Path(args.dir)
    splits = {}
    for split in ("train", "valid", "test"):
        p = base / f"{split}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        splits[split] = list(read_jsonl(p))

    print("========== Per-split statistics (MOUS) ==========")
    per_split_sentence_ids = {}  # split -> set(sentence_content_id)
    global_sentence_to_splits = defaultdict(set)

    for split, rows in splits.items():
        n_rows = len(rows)  # total 3 s windows (including subjects)
        audio_ids = set()
        sent_ids = set()
        subj_counter = Counter()

        for r in rows:
            audio_ids.add(audio_window_id(r))   # unique 3 s window contents
            sid = sentence_content_id(r)        # sentence-level content
            if sid != "NA":
                sent_ids.add(sid)
            subj_counter[r.get("subject_id", "NA")] += 1

        per_split_sentence_ids[split] = sent_ids
        for sid in sent_ids:
            global_sentence_to_splits[sid].add(split)

        unique_audio_windows = len(audio_ids)
        unique_sentence_contents = len(sent_ids)
        avg_win_per_sent = (
            n_rows / unique_sentence_contents
            if unique_sentence_contents > 0 else 0.0
        )

        print(
            f"[{split}] "
            f"total_rows(all 3s windows)={n_rows:,} | "
            f"unique_window_contents(candidate 3s windows)={unique_audio_windows:,} | "
            f"unique_sentence_contents(10s sentences)={unique_sentence_contents:,} | "
            f"avg_windows_per_sentence={avg_win_per_sent:.2f} | "
            f"subjects={len(subj_counter)}"
        )

        # Print top 5 subjects by number of rows
        top_subj = subj_counter.most_common(5)
        if top_subj:
            print(f"    Top subjects in {split}:")
            for sub, cnt in top_subj:
                print(f"        subject {sub}: {cnt} rows")
        print()

    print("========== Sentence-level split consistency check ==========")
    leaks = [
        (sid, sorted(list(sps)))
        for sid, sps in global_sentence_to_splits.items()
        if len(sps) > 1
    ]

    if not leaks:
        print(
            "✅ No sentence-level content appears in multiple splits. "
            "The content-based 7/1/2 split is clean."
        )
    else:
        print(
            "⚠️ The following sentence contents appear in multiple splits "
            "(potential content leakage):"
        )
        for sid, sps in leaks[:50]:  # limit output length
            print(f"  sentence_content_id={sid} -> splits={sps}")
        if len(leaks) > 50:
            print(f"... {len(leaks) - 50} additional cases not shown")

    print("\nMOUS candidate pool report finished ✅")


if __name__ == "__main__":
    main()
