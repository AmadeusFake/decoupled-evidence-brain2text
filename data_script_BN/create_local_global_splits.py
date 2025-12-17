#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_local_global_splits.py (Brennan EEG • sentence-only)

- Split train/val/test by content (sentence-level split)
- 3s window: each word as anchor, interval [-0.5, +2.5] seconds (audio timeline)
- FIF (EEG) timeline: globally shifted by +EEG_LAG_S (default 0.150s)
- Candidate pool = content-unique 3s audio windows in TEST
- Remove time overlap across splits (keep test, drop train/valid)
"""
import argparse, logging, json, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

EEG_LAG_S = 0.150   # EEG response lag relative to audio (seconds)
WINDOW_SIZE_S = 3.0
ANCHOR_PRE_S = 0.5
ANCHOR_POST_S = 2.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def write_jsonlines(p: Path, items: List[dict]):
    # Write list of dicts to JSONL file
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def _content_key(seg: dict) -> str:
    # Sentence-level content key: story_id + sentence_id
    return f"{seg['story_id']}_{int(seg['segment_idx_in_file'])}"

def _make_sentence_id(seg: dict) -> str:
    # Unique sentence identifier
    return f"{seg['subject_id']}_{seg['session_id']}_{seg['story_id']}_{seg['segment_idx_in_file']}"

def _unique_window_key(audio_path: str, start_audio: float, stop_audio: float) -> str:
    # Unique key for an audio window
    return f"{audio_path}::{start_audio:.3f}-{stop_audio:.3f}"

def load_meta(meta_manifest_path: Path) -> List[dict]:
    # Load sentence-only segments from meta manifest
    items = []
    with open(meta_manifest_path, "r") as f:
        for line in f:
            seg = json.loads(line)
            if seg.get("type") == "sentence":
                items.append(seg)
    return items

def split_by_content(unique_content_keys: List[str], split_ratios: Tuple[float,float,float], seed=42):
    # Random content-level split
    random.seed(seed)
    keys = unique_content_keys[:]
    random.shuffle(keys)
    n = len(keys)
    n_train = int(n * split_ratios[0])
    n_val   = int(n * split_ratios[1])
    train = set(keys[:n_train])
    valid = set(keys[n_train:n_train+n_val])
    test  = set(keys[n_train+n_val:])
    return {"train": train, "valid": valid, "test": test}

def enumerate_anchored_windows(seg: dict, bids_root: Path) -> List[dict]:
    # Generate 3s windows anchored at each word
    out = []
    audio_path = str(bids_root / seg["original_audio_path"])
    sent_id = _make_sentence_id(seg)

    gl_fif_on = seg["segment_onset_in_fif_s"]
    gl_fif_off= seg["segment_offset_in_fif_s"]
    gl_au_on  = seg["segment_onset_in_audio_s"]

    words = seg.get("words_timing", [])
    for wi, w in enumerate(words):
        start_rel = float(w.get("start", 0.0)) - ANCHOR_PRE_S
        stop_rel  = start_rel + WINDOW_SIZE_S

        # Map to global audio timeline
        local_on_audio = gl_au_on + start_rel
        local_off_audio= gl_au_on + stop_rel

        # Map to EEG (FIF) timeline and apply lag
        local_on_fif = gl_fif_on + start_rel + EEG_LAG_S
        local_off_fif= local_on_fif + WINDOW_SIZE_S

        content_id = _unique_window_key(audio_path, local_on_audio, local_off_audio)

        out.append({
            "content_id": content_id,
            "sentence_id": sent_id,
            "anchor_word_idx": wi,
            "global_segment_text": seg.get("text",""),

            "original_audio_path": audio_path,
            "original_fif_path": seg["original_fif_path"],
            "sensor_coordinates_path": seg.get("sensor_coordinates_path",""),
            "adjacency_matrix_path": seg.get("adjacency_matrix_path",""),

            "local_window_onset_in_audio_s": round(local_on_audio, 5),
            "local_window_offset_in_audio_s": round(local_off_audio, 5),
            "local_window_onset_in_fif_s": round(local_on_fif, 5),
            "local_window_offset_in_fif_s": round(local_off_fif, 5),
            "local_window_duration_s": WINDOW_SIZE_S,

            "global_segment_onset_in_fif_s": round(gl_fif_on + EEG_LAG_S, 5),
            "global_segment_offset_in_fif_s": round(gl_fif_off + EEG_LAG_S, 5),
            "global_segment_onset_in_audio_s": seg["segment_onset_in_audio_s"],
            "global_segment_offset_in_audio_s": seg["segment_offset_in_audio_s"],

            "subject_id": seg["subject_id"],
            "session_id": seg["session_id"],

            "window_id": f"{seg['subject_id']}_{seg['session_id']}_{seg['story_id']}_{seg['segment_idx_in_file']}_w{wi}"
        })
    return out

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    # Compute overlap duration between two intervals
    return max(0.0, min(a1,b1) - max(a0,b0))

def remove_cross_split_overlaps(per_split_rows: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    # Remove audio time overlap between train/valid and test
    test_intervals = defaultdict(list)
    for r in per_split_rows["test"]:
        test_intervals[r["original_audio_path"]].append(
            (r["local_window_onset_in_audio_s"], r["local_window_offset_in_audio_s"])
        )

    def is_conflict(row):
        candidates = test_intervals.get(row["original_audio_path"])
        if not candidates:
            return False
        a0, a1 = row["local_window_onset_in_audio_s"], row["local_window_offset_in_audio_s"]
        for b0,b1 in candidates:
            if _overlap(a0,a1,b0,b1) > 0:
                return True
        return False

    removed = 0
    cleaned = {"test": per_split_rows["test"], "train": [], "valid": []}
    for split in ("train","valid"):
        for r in per_split_rows[split]:
            if is_conflict(r):
                removed += 1
                continue
            cleaned[split].append(r)

    logging.info(f"[no-cross-overlap] removed {removed} conflicting windows across splits")
    return cleaned

def main():
    ap = argparse.ArgumentParser(description="Create anchored 3s window splits (Brennan EEG, sentence-only).")
    ap.add_argument("--meta_manifest_path", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--bids_root_dir", required=True, type=str)
    ap.add_argument("--split_ratios", type=str, default="0.7,0.1,0.2", help="train,val,test")
    ap.add_argument("--random_seed", type=int, default=42)
    args = ap.parse_args()

    bids_root = Path(args.bids_root_dir)
    out_dir = Path(args.output_dir)
    ratios = tuple(float(x) for x in args.split_ratios.split(","))

    logging.info("Loading meta (sentence-only) ...")
    segs = load_meta(Path(args.meta_manifest_path))
    logging.info(f"Loaded {len(segs):,} sentence segments")

    content_map = defaultdict(list)
    for s in segs:
        content_map[_content_key(s)].append(s)
    content_keys = list(content_map.keys())
    split_map = split_by_content(content_keys, ratios, seed=args.random_seed)
    logging.info(
        f"[sentence] content split sizes: "
        f"train={len(split_map['train'])}, "
        f"valid={len(split_map['valid'])}, "
        f"test={len(split_map['test'])}"
    )

    per_split_rows = {"train":[], "valid":[], "test":[]}
    logging.info("[sentence] Enumerating anchored 3s windows from words...")
    for split_name, keyset in split_map.items():
        for ck in tqdm(keyset, desc="[sentence] words->windows"):
            for seg in content_map[ck]:
                rows = enumerate_anchored_windows(seg, bids_root)
                per_split_rows[split_name].extend(rows)

    unique_content_ids = set(r["content_id"] for r in per_split_rows["test"])
    logging.info(
        f"[sentence] Unique content windows in TEST (candidate pool): "
        f"{len(unique_content_ids):,}"
    )

    per_split_rows = remove_cross_split_overlaps(per_split_rows)

    subdir = out_dir / "final_splits_sentence"
    subdir.mkdir(parents=True, exist_ok=True)
    for split in ("train","valid","test"):
        p = subdir / f"{split}.jsonl"
        write_jsonlines(p, per_split_rows[split])
        logging.info(f"[sentence] Saved {len(per_split_rows[split]):,} rows -> {p}")

    logging.info("[sentence] Done.")

if __name__ == "__main__":
    main()
