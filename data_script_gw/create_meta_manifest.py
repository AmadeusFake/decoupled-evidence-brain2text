#!/usr/bin/env python
# ==============================================================================
# create_meta_manifest.py (v3.0, BIDS-native)
#
# Build a sentence-level metadata blueprint directly from raw BIDS data.
#
# - Scan BIDS directories for events.tsv files
# - Reconstruct sentences / word_list segments from word-level events
# - Compute global timing in MEG (FIF) and audio
# - IMPORTANT: `original_fif_path` always points to the raw BIDS MEG file
#   (.con or .fif), never to preprocessed data
# ==============================================================================

import ast
import logging
import re
from pathlib import Path
from multiprocessing import Pool
import os
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse

import pandas as pd
from tqdm import tqdm
import jsonlines

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------- IO -------------------- #

def write_jsonlines(p: Path, items: List[Dict]):
    """Write a list of dicts to a JSONL file."""
    p.parent.mkdir(parents=True, exist_ok=True)
    if items:
        with jsonlines.open(p, "w") as w:
            w.write_all(items)

# -------------------- Core extraction -------------------- #

def extract_sentence_metadata(task_args: Tuple[str, dict]) -> List[Dict]:
    """Extract sentence / word_list metadata from a single events.tsv file."""
    events_tsv_path_str, cfg = task_args
    events_tsv_path = Path(events_tsv_path_str)

    try:
        subj_id = re.search(r"sub-([A-Za-z0-9]+)", str(events_tsv_path)).group(1)
        ses_id  = re.search(r"ses-([A-Za-z0-9]+)", str(events_tsv_path)).group(1)
    except AttributeError:
        logging.warning(f"Cannot parse subject/session from {events_tsv_path.name}")
        return []

    # Resolve raw BIDS MEG file (.con preferred, fallback to .fif)
    base_name = events_tsv_path.name.replace("_events.tsv", "")
    original_meg_path = events_tsv_path.parent / f"{base_name}_meg.con"
    if not original_meg_path.exists():
        original_meg_path = events_tsv_path.parent / f"{base_name}_meg.fif"
        if not original_meg_path.exists():
            logging.warning(
                f"Raw BIDS MEG file (.con or .fif) not found for base: {base_name}"
            )
            return []

    sensor_coords_path = events_tsv_path.parent / "sensor_coordinates.npy"
    adjacency_matrix_path = events_tsv_path.parent / "adjacency_matrix.npy"
    sensor_coords_path_str = str(sensor_coords_path) if sensor_coords_path.exists() else ""
    adjacency_matrix_path_str = str(adjacency_matrix_path) if adjacency_matrix_path.exists() else ""

    if not sensor_coords_path_str:
        logging.warning(f"Sensor coordinates missing: {sensor_coords_path}")

    try:
        df = pd.read_csv(events_tsv_path, delimiter="\t")
    except Exception as e:
        logging.error(f"Failed to read TSV {events_tsv_path}: {e}")
        return []

    # Collect word-level events
    all_events = []
    for _, row in df.iterrows():
        try:
            if not isinstance(row.get("trial_type"), str):
                continue
            trial_type = ast.literal_eval(row["trial_type"])
            if isinstance(trial_type, dict) and trial_type.get("kind") == "word":
                if pd.notna(row.get("onset")) and pd.notna(row.get("duration")):
                    trial_type["onset_global"] = row["onset"]
                    trial_type["duration_global"] = row["duration"]
                    all_events.append(trial_type)
        except Exception:
            continue

    if not all_events:
        return []

    # Group words into sentence / word_list segments
    grouped_events = defaultdict(list)
    for word in all_events:
        if word.get("sequence_id") is not None and word.get("condition") in ["sentence", "word_list"]:
            grouped_events[(word["condition"], word["sequence_id"])].append(word)

    segments = []
    for (condition, _), words_in_group in grouped_events.items():
        words_sorted = sorted(words_in_group, key=lambda w: w.get("start", float("inf")))
        if words_sorted:
            segments.append({"type": condition, "words": words_sorted})

    segment_metadata_list = []
    for i, segment in enumerate(segments):
        words = segment["words"]
        first_word, last_word = words[0], words[-1]

        audio_rel_path = first_word.get("sound")
        if not audio_rel_path:
            continue

        onset_fif_s = first_word["onset_global"]
        offset_fif_s = last_word["onset_global"] + last_word["duration_global"]
        onset_audio_s = first_word.get("start", 0)
        offset_audio_s = last_word.get("start", 0) + last_word.get("duration", 0)

        # Make word timings relative to sentence onset
        corrected_words = [
            {**w, "start": round(w.get("start", 0) - onset_audio_s, 3)}
            for w in words
        ]

        segment_metadata_list.append({
            "subject_id": subj_id,
            "session_id": ses_id,
            "type": segment["type"],
            "story_id": first_word.get("story_uid", "unknown"),
            "segment_idx_in_file": i,
            "text": " ".join(w["word"] for w in words),
            "words_timing": corrected_words,

            "original_fif_path": str(original_meg_path),
            "original_audio_path": audio_rel_path,
            "sensor_coordinates_path": sensor_coords_path_str,
            "adjacency_matrix_path": adjacency_matrix_path_str,

            "segment_duration_s": round(offset_fif_s - onset_fif_s, 3),
            "segment_onset_in_fif_s": round(onset_fif_s, 5),
            "segment_offset_in_fif_s": round(offset_fif_s, 5),
            "segment_onset_in_audio_s": round(onset_audio_s, 5),
            "segment_offset_in_audio_s": round(offset_audio_s, 5),
        })

    return segment_metadata_list

# -------------------- main -------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Create sentence-level metadata blueprint from raw BIDS data."
    )
    parser.add_argument("--input_dir", required=True, help="Root BIDS dataset directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for meta_manifest.jsonl")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(os.cpu_count(), 32),
        help="Number of parallel workers",
    )
    args = parser.parse_args()

    cfg = vars(args)
    tasks = [
        (str(p), cfg)
        for p in Path(cfg["input_dir"]).glob("sub-*/ses-*/meg/*_events.tsv")
    ]
    if not tasks:
        logging.error(f"No '*_events.tsv' files found under {cfg['input_dir']}")
        return

    all_segments: List[Dict] = []
    logging.info(f"Extracting metadata from {len(tasks)} files...")

    with Pool(cfg["num_workers"]) as pool:
        for result in tqdm(
            pool.imap_unordered(extract_sentence_metadata, tasks),
            total=len(tasks),
            desc="Processing TSV files",
        ):
            all_segments.extend(result)

    output_path = Path(cfg["output_dir"]) / "meta_manifest.jsonl"
    write_jsonlines(output_path, all_segments)
    logging.info(
        f"Created meta-manifest with {len(all_segments)} segments -> {output_path}"
    )

if __name__ == "__main__":
    main()
