#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create content-level train/valid/test splits with word-anchored 3s windows.

Key points (paper-aligned):
- Split by content (sentence / word_list segment), not by window
- 3s windows are word-anchored: [-0.5s, +2.5s]
- Candidate pool = all content-unique 3s windows in TEST
- Remove cross-split temporal overlap (keep TEST, drop train/valid)
"""

import argparse, logging, json, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

MEG_LAG_S = 0.150
WINDOW_SIZE_S = 3.0
ANCHOR_PRE_S = 0.5
ANCHOR_POST_S = 2.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------- IO -------------------- #

def write_jsonlines(p: Path, items: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

# -------------------- Content keys -------------------- #

def _content_key(seg: dict) -> str:
    """Content unit used for splitting (sentence / word_list segment)."""
    return f"{seg['story_id']}_{seg['segment_idx_in_file']}"

def _make_sentence_id(seg: dict) -> str:
    """Global sentence identifier."""
    return f"{seg['subject_id']}_{seg['session_id']}_{seg['story_id']}_{seg['segment_idx_in_file']}"

def _unique_window_key(audio_path: str, start_audio: float, stop_audio: float) -> str:
    """Content-unique window key (subject-independent)."""
    return f"{audio_path}::{start_audio:.3f}-{stop_audio:.3f}"

# -------------------- Loading -------------------- #

def load_meta(meta_manifest_path: Path, data_type: str) -> List[dict]:
    items = []
    with open(meta_manifest_path, "r") as f:
        for line in f:
            seg = json.loads(line)
            if seg.get("type") == data_type:
                items.append(seg)
    return items

# -------------------- Splitting -------------------- #

def split_by_content(
    unique_content_keys: List[str],
    split_ratios: Tuple[float, float, float],
    seed: int = 42,
):
    random.seed(seed)
    keys = unique_content_keys[:]
    random.shuffle(keys)
    n = len(keys)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])
    return {
        "train": set(keys[:n_train]),
        "valid": set(keys[n_train:n_train + n_val]),
        "test":  set(keys[n_train + n_val:]),
    }

# -------------------- Window enumeration -------------------- #

def enumerate_anchored_windows(seg: dict, bids_root: Path) -> List[dict]:
    """Enumerate 3s word-anchored windows for one content segment."""
    out = []
    audio_path = str(bids_root / seg["original_audio_path"])
    sent_id = _make_sentence_id(seg)

    gl_fif_on  = seg["segment_onset_in_fif_s"]
    gl_fif_off = seg["segment_offset_in_fif_s"]
    gl_au_on   = seg["segment_onset_in_audio_s"]

    words = seg.get("words_timing", [])
    for wi, w in enumerate(words):
        start_rel = float(w.get("start", 0.0)) - ANCHOR_PRE_S
        stop_rel  = start_rel + WINDOW_SIZE_S

        local_on_audio  = gl_au_on + start_rel
        local_off_audio = gl_au_on + stop_rel

        local_on_fif  = gl_fif_on + start_rel + MEG_LAG_S
        local_off_fif = local_on_fif + WINDOW_SIZE_S

        content_id = _unique_window_key(audio_path, local_on_audio, local_off_audio)

        out.append({
            "content_id": content_id,
            "sentence_id": sent_id,
            "anchor_word_idx": wi,
            "global_segment_text": seg.get("text", ""),

            "original_audio_path": audio_path,
            "original_fif_path": seg["original_fif_path"],
            "sensor_coordinates_path": seg.get("sensor_coordinates_path", ""),
            "adjacency_matrix_path": seg.get("adjacency_matrix_path", ""),

            "local_window_onset_in_audio_s": round(local_on_audio, 5),
            "local_window_offset_in_audio_s": round(local_off_audio, 5),
            "local_window_onset_in_fif_s": round(local_on_fif, 5),
            "local_window_offset_in_fif_s": round(local_off_fif, 5),
            "local_window_duration_s": WINDOW_SIZE_S,

            "global_segment_onset_in_fif_s": round(gl_fif_on + MEG_LAG_S, 5),
            "global_segment_offset_in_fif_s": round(gl_fif_off + MEG_LAG_S, 5),
            "global_segment_onset_in_audio_s": seg["segment_onset_in_audio_s"],
            "global_segment_offset_in_audio_s": seg["segment_offset_in_audio_s"],

            "subject_id": seg["subject_id"],
            "session_id": seg["session_id"],

            "window_id": (
                f"{seg['subject_id']}_{seg['session_id']}_"
                f"{seg['story_id']}_{seg['segment_idx_in_file']}_w{wi}"
            ),
        })
    return out

# -------------------- Overlap removal -------------------- #

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def remove_cross_split_overlaps(per_split_rows: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Remove train/valid windows overlapping TEST on the same audio."""
    test_intervals = defaultdict(list)
    for r in per_split_rows["test"]:
        test_intervals[r["original_audio_path"]].append(
            (r["local_window_onset_in_audio_s"], r["local_window_offset_in_audio_s"])
        )

    def is_conflict(row):
        intervals = test_intervals.get(row["original_audio_path"])
        if not intervals:
            return False
        a0, a1 = row["local_window_onset_in_audio_s"], row["local_window_offset_in_audio_s"]
        for b0, b1 in intervals:
            if _overlap(a0, a1, b0, b1) > 0:
                return True
        return False

    removed = 0
    cleaned = {"test": per_split_rows["test"], "train": [], "valid": []}
    for split in ("train", "valid"):
        for r in per_split_rows[split]:
            if is_conflict(r):
                removed += 1
                continue
            cleaned[split].append(r)

    logging.info(f"[no-cross-overlap] removed {removed} windows")
    return cleaned

# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser(description="Create paper-aligned anchored window splits.")
    ap.add_argument("--meta_manifest_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--bids_root_dir", required=True)
    ap.add_argument("--data_type", required=True, choices=["sentence", "word_list", "both"])
    ap.add_argument("--split_ratios", default="0.7,0.1,0.2")
    ap.add_argument("--random_seed", type=int, default=42)
    args = ap.parse_args()

    bids_root = Path(args.bids_root_dir)
    out_dir = Path(args.output_dir)
    ratios = tuple(float(x) for x in args.split_ratios.split(","))

    types = ["sentence", "word_list"] if args.data_type == "both" else [args.data_type]
    for dt in types:
        logging.info(f"[{dt}] loading meta")
        segs = load_meta(Path(args.meta_manifest_path), dt)
        logging.info(f"[{dt}] loaded {len(segs):,} segments")

        content_map = defaultdict(list)
        for s in segs:
            content_map[_content_key(s)].append(s)

        split_map = split_by_content(list(content_map.keys()), ratios, seed=args.random_seed)
        logging.info(
            f"[{dt}] split sizes: "
            f"train={len(split_map['train'])}, "
            f"valid={len(split_map['valid'])}, "
            f"test={len(split_map['test'])}"
        )

        per_split_rows = {"train": [], "valid": [], "test": []}
        for split_name, keyset in split_map.items():
            for ck in tqdm(keyset, desc=f"[{dt}] words->windows"):
                for seg in content_map[ck]:
                    per_split_rows[split_name].extend(
                        enumerate_anchored_windows(seg, bids_root)
                    )

        unique_content_ids = set(r["content_id"] for r in per_split_rows["test"])
        logging.info(
            f"[{dt}] unique TEST content windows (candidate pool): "
            f"{len(unique_content_ids):,}"
        )

        per_split_rows = remove_cross_split_overlaps(per_split_rows)

        subdir = out_dir / f"final_splits_{dt}"
        subdir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            p = subdir / f"{split}.jsonl"
            write_jsonlines(p, per_split_rows[split])
            logging.info(f"[{dt}] saved {len(per_split_rows[split]):,} rows -> {p}")

        logging.info(f"[{dt}] done")

if __name__ == "__main__":
    main()
