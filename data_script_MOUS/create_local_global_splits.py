#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_local_global_splits.py (paper-aligned, final)

- Split data into train/valid/test by content (sentence / word_list segment).
- For evaluation windows, use word-anchored 3s windows [-0.5, +2.5] seconds
  as described in the paper.
- Candidate pool = all content-unique 3s audio windows in TEST
  (subject-independent).
- Optional: --subject_allowlist to restrict meta to a subset of subjects.
"""
# Source: :contentReference[oaicite:1]{index=1}

import argparse
import logging
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from tqdm import tqdm

MEG_LAG_S = 0.150   # MEG-to-audio alignment lag used in the paper (FIF timing)
WINDOW_SIZE_S = 3.0
ANCHOR_PRE_S = 0.5   # window start offset (-0.5 s)
ANCHOR_POST_S = 2.5  # window end offset (+2.5 s)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def write_jsonlines(p: Path, items: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _content_key(seg: dict) -> str:
    """
    Content-level split unit = audio segment.

    Defined by:
      (original_audio_path,
       segment_onset_in_audio_s,
       segment_offset_in_audio_s)

    Rationale:
    - MOUS: identical audio segments across subjects map to the same key,
      ensuring content-level splitting.
    - Gwilliams: each sentence already corresponds to a unique audio segment.
    """
    audio = seg["original_audio_path"]
    g0 = float(seg["segment_onset_in_audio_s"])
    g1 = float(seg["segment_offset_in_audio_s"])
    return f"{audio}::{g0:.3f}-{g1:.3f}"


def _make_sentence_id(seg: dict) -> str:
    return (
        f"{seg['subject_id']}_{seg['session_id']}_"
        f"{seg['story_id']}_{seg['segment_idx_in_file']}"
    )


def _unique_window_key(audio_path: str, start_audio: float, stop_audio: float) -> str:
    """Content-unique window identifier (subject-independent)."""
    return f"{audio_path}::{start_audio:.3f}-{stop_audio:.3f}"


def load_subject_allowlist(path: Optional[Path]) -> Optional[Set[str]]:
    """
    Load subject allowlist from a text file (one ID per line).

    Accepts:
      - "A2002"
      - "sub-A2002"

    Returned IDs are normalized to "A2002".
    """
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"subject_allowlist not found: {path}")

    allow: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            sid = line.strip()
            if not sid:
                continue
            if sid.startswith("sub-"):
                sid = sid.replace("sub-", "")
            allow.add(sid)

    logging.info(f"[splits] Loaded subject_allowlist with {len(allow)} IDs from {path}")
    return allow


def load_meta(
    meta_manifest_path: Path,
    data_type: str,
    allow: Optional[Set[str]] = None,
) -> List[dict]:
    items = []
    with open(meta_manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            seg = json.loads(line)
            if seg.get("type") != data_type:
                continue

            sid = seg.get("subject_id") or ""
            if sid.startswith("sub-"):
                sid = sid.replace("sub-", "")

            if allow is not None and sid not in allow:
                continue

            items.append(seg)
    return items


def split_by_content(
    unique_content_keys: List[str],
    split_ratios: Tuple[float, float, float],
    seed: int = 42,
):
    """
    Split by content key (sentence / word_list segment) using fixed ratios.
    Each content key appears in exactly one split.
    """
    random.seed(seed)
    keys = unique_content_keys[:]
    random.shuffle(keys)

    n = len(keys)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train = set(keys[:n_train])
    valid = set(keys[n_train : n_train + n_val])
    test = set(keys[n_train + n_val :])

    return {"train": train, "valid": valid, "test": test}


def enumerate_anchored_windows(seg: dict, bids_root: Path) -> List[dict]:
    """
    Enumerate word-anchored 3s windows for one sentence / segment.

    - Gwilliams: true word-level timing is available.
    - MOUS: a single pseudo-word exists (start=0, duration=sentence),
      resulting in one 3s window per segment.
    """
    out = []
    audio_path = str(bids_root / seg["original_audio_path"])
    sent_id = _make_sentence_id(seg)

    gl_fif_on = seg["segment_onset_in_fif_s"]
    gl_fif_off = seg["segment_offset_in_fif_s"]
    gl_au_on = seg["segment_onset_in_audio_s"]

    words = seg.get("words_timing", [])
    for wi, w in enumerate(words):
        start_rel = float(w.get("start", 0.0)) - ANCHOR_PRE_S
        stop_rel = start_rel + WINDOW_SIZE_S

        local_on_audio = gl_au_on + start_rel
        local_off_audio = gl_au_on + stop_rel

        local_on_fif = gl_fif_on + start_rel + MEG_LAG_S
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
            "story_id": seg.get("story_id", ""),
            "segment_idx_in_file": seg.get("segment_idx_in_file", -1),
            "content_key": (
                f"{seg.get('story_id', '')}_{seg.get('segment_idx_in_file', -1)}"
            ),

            "window_id": (
                f"{seg['subject_id']}_{seg['session_id']}_"
                f"{seg['story_id']}_{seg['segment_idx_in_file']}_w{wi}"
            ),
        })
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def remove_cross_split_overlaps(
    per_split_rows: Dict[str, List[dict]]
) -> Dict[str, List[dict]]:
    """
    Rule for long continuous stories (Gwilliams):
    - If windows overlap in time on the same audio file across splits,
      keep TEST windows and drop conflicting TRAIN/VALID windows.
    """
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
        for b0, b1 in candidates:
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

    logging.info(f"[no-cross-overlap] removed {removed} conflicting windows across splits")
    return cleaned


def main():
    ap = argparse.ArgumentParser(
        description="Create word-anchored 3s window splits aligned with the paper."
    )
    ap.add_argument("--meta_manifest_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--bids_root_dir", required=True)
    ap.add_argument("--data_type", required=True, choices=["sentence", "word_list", "both"])
    ap.add_argument("--split_ratios", default="0.7,0.1,0.2", help="train,valid,test")
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument(
        "--disable_cross_overlap_pruning",
        action="store_true",
        help=(
            "If set, skip cross-split temporal overlap pruning "
            "(recommended for trial-based datasets such as MOUS)."
        ),
    )
    ap.add_argument(
        "--subject_allowlist",
        default=None,
        help="Optional subject allowlist txt (one ID per line).",
    )
    args = ap.parse_args()

    bids_root = Path(args.bids_root_dir)
    out_dir = Path(args.output_dir)
    ratios = tuple(float(x) for x in args.split_ratios.split(","))

    allow = (
        load_subject_allowlist(Path(args.subject_allowlist))
        if args.subject_allowlist
        else None
    )

    types = ["sentence", "word_list"] if args.data_type == "both" else [args.data_type]
    for dt in types:
        logging.info(f"Loading meta for data_type='{dt}' ...")
        segs = load_meta(Path(args.meta_manifest_path), dt, allow=allow)
        logging.info(f"Loaded {len(segs):,} segments for data_type='{dt}'")

        if not segs:
            logging.warning(f"[{dt}] No segments after filtering; skipping.")
            continue

        content_map = defaultdict(list)
        for s in segs:
            content_map[_content_key(s)].append(s)

        content_keys = list(content_map.keys())
        split_map = split_by_content(content_keys, ratios, seed=args.random_seed)
        logging.info(
            f"[{dt}] content split sizes: "
            f"train={len(split_map['train'])}, "
            f"valid={len(split_map['valid'])}, "
            f"test={len(split_map['test'])}"
        )

        per_split_rows: Dict[str, List[dict]] = {"train": [], "valid": [], "test": []}
        logging.info(f"[{dt}] Enumerating anchored 3s windows...")
        for split_name, keyset in split_map.items():
            for ck in tqdm(keyset, desc=f"[{dt}] words->windows ({split_name})"):
                for seg in content_map[ck]:
                    rows = enumerate_anchored_windows(seg, bids_root)
                    per_split_rows[split_name].extend(rows)

        unique_content_ids = {r["content_id"] for r in per_split_rows["test"]}
        logging.info(
            f"[{dt}] Unique content windows in TEST (candidate pool): "
            f"{len(unique_content_ids):,}"
        )

        if not args.disable_cross_overlap_pruning:
            per_split_rows = remove_cross_split_overlaps(per_split_rows)
        else:
            logging.info(f"[{dt}] Cross-split overlap pruning disabled.")

        subdir = out_dir / f"final_splits_{dt}"
        subdir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            p = subdir / f"{split}.jsonl"
            write_jsonlines(p, per_split_rows[split])
            logging.info(f"[{dt}] Saved {len(per_split_rows[split]):,} rows -> {p}")

        logging.info(f"[{dt}] Done.")


if __name__ == "__main__":
    main()
