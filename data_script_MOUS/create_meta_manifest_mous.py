#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_meta_manifest_mous.py — MOUS (Schoffelen 2019)
Content-driven meta manifest generation with Whisper word-level alignments.

Inputs:
  1) MOUS BIDS root: --mous_root
     (e.g. /mimer/.../MOUS_raw)
  2) Whisper word alignments: --align_jsonl
     (e.g. data_mous_local_global/mous_whisper_word_alignments.jsonl)
  3) Output directory: --out_dir
  4) (Optional) subject allowlist: --subject_allowlist
     (one subject per line, e.g. A2002 or sub-A2002)

Behavior:
  - Scan <mous_root>/sub-*/meg/*task-auditory*events.tsv
  - If subject_allowlist is provided, keep only matching subjects
  - Traverse each events.tsv in time order:
      * Picture ZINNEN   → current_block_type = "sentence"
      * Picture WOORDEN  → current_block_type = "word_list"
      * FIX / blank / QUESTION do not change block type
      * Sound + current_block_type in {"sentence","word_list"}:
          - Parse seq_id from value
          - Use Whisper seq2words[seq_id] to populate words_timing / text / duration
          - Use Sound.onset as MEG-aligned segment onset

Output:
  - <out_dir>/meta_manifest_mous.jsonl
    Each line is a JSON dict with fields:
      subject_id, session_id, type ("sentence"/"word_list"),
      story_id (e.g. "001"),
      segment_idx_in_file,
      text,
      words_timing: [{word, start, duration}, ...],
      original_fif_path,
      original_audio_path: "stimuli/audio_files/EQ_Ramp_Int2_Int1LPFXXX.wav",
      segment_duration_s,
      segment_onset_in_fif_s, segment_offset_in_fif_s,
      segment_onset_in_audio_s, segment_offset_in_audio_s,
      global_segment_onset_in_audio_s, global_segment_offset_in_audio_s

Contract:
  - If a language Sound event fails to parse seq_id,
    or seq_id is missing from Whisper alignments,
    a RuntimeError is raised (no fallback).
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import jsonlines
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ---------------- Whisper alignment loading ---------------- #

def load_whisper_alignments(path: Path) -> Dict[int, List[Dict]]:
    """
    Load word-level Whisper alignments from a jsonl file.

    Returns:
      seq2words: {sequence_id: [{word, start, duration}, ...], ...}
    """
    if not path.exists():
        raise FileNotFoundError(f"Whisper alignment file not found: {path}")

    seq2raw: Dict[int, List[Dict]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "sequence_id" not in rec:
                raise RuntimeError(f"Missing 'sequence_id' in Whisper record: {rec}")
            sid = int(rec["sequence_id"])
            seq2raw[sid].append(rec)

    seq2words: Dict[int, List[Dict]] = {}
    for sid, rows in seq2raw.items():
        rows = sorted(rows, key=lambda r: float(r.get("start", 0.0)))
        words = []
        for r in rows:
            w = r.get("word", "")
            if not isinstance(w, str):
                continue
            w = w.strip()
            if not w:
                continue
            start = float(r.get("start", 0.0))
            end = float(r.get("end", start))
            dur = max(0.0, end - start)
            words.append({
                "word": w,
                "start": round(start, 3),
                "duration": round(dur, 3),
            })
        if not words:
            raise RuntimeError(f"sequence_id={sid} has empty word list in {path}")
        seq2words[sid] = words

    logging.info(f"Loaded Whisper alignments for {len(seq2words)} sequence IDs from {path}")
    return seq2words


# ---------------- Generic utilities ---------------- #

def write_jsonlines(p: Path, items: List[Dict]) -> None:
    """Write a jsonl file."""
    p.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        return
    with jsonlines.open(p, "w") as w:
        w.write_all(items)


def _find_bids_root_from_events(events_tsv_path: Path) -> Path:
    """
    Given:
      <bids_root>/sub-A2002/meg/sub-A2002_task-auditory_events.tsv
    Return:
      bids_root = parents[2]
    """
    return events_tsv_path.parents[2]


def _parse_subject_id_from_events_path(events_tsv_path: Path) -> str:
    """
    sub-A2002_task-auditory_events.tsv → subject_id = "A2002"
    """
    name = events_tsv_path.name
    subj_tag = name.split("_")[0]
    return subj_tag.replace("sub-", "")


def load_subject_allowlist(path: Optional[Path]) -> Optional[Set[str]]:
    """
    Load subject allowlist from a text file (one ID per line).

    Accepted formats:
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

    logging.info(f"Loaded subject_allowlist with {len(allow)} IDs from {path}")
    return allow


# ---------------- Content-driven event parsing ---------------- #

def _parse_seq_id_from_sound_value(events_tsv_path: Path, onset: float, value: str) -> int:
    """
    Parse sequence_id from a Sound event value.

    Priority:
      1) Match "...File 891.wav"  → seq_id = 891
      2) Fallback: trailing integer → seq_id = 891
    """
    v = str(value)

    m = re.search(r"[Ff]ile\s+(\d+)\.wav", v)
    if m:
        return int(m.group(1))

    m2 = re.search(r"(\d+)\s*$", v)
    if m2:
        return int(m2.group(1))

    raise RuntimeError(
        f"{events_tsv_path}: cannot parse sequence_id from Sound value='{v}' (onset={onset})"
    )


def extract_segments_from_events(
    events_tsv_path: Path,
    seq2words: Dict[int, List[Dict]],
) -> List[Dict]:
    """
    Extract all content-driven language segments from a single
    *_task-auditory*_events.tsv.

    Strategy:
      - Iterate events by onset (ascending).
      - Maintain current_block_type:
          * Picture ZINNEN   → 'sentence'
          * Picture WOORDEN  → 'word_list'
          * Other Picture events do not change the block type
      - When type == 'Sound' and block_type in {'sentence','word_list'}:
          * Parse seq_id from value
          * Use seq2words[seq_id] for words_timing / text / duration
          * Use Sound.onset as MEG-aligned onset
    """
    logging.info(f"Parsing events: {events_tsv_path}")

    try:
        df = pd.read_csv(events_tsv_path, sep="\t")
    except Exception as e:
        raise RuntimeError(f"Failed to read {events_tsv_path}: {e}")

    required_cols = {"onset", "duration", "type", "value"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(
            f"{events_tsv_path} missing required columns {required_cols}; "
            f"found={list(df.columns)}"
        )

    df = df.copy()
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df = df.dropna(subset=["onset"])
    df = df.sort_values("onset").reset_index(drop=True)

    # --- subject / session / MEG path resolution ---
    name = events_tsv_path.name
    subj_tag = name.split("_")[0]
    subject_id = subj_tag.replace("sub-", "")
    session_id = "ses-1"  # MOUS has no explicit sessions

    meg_dir = events_tsv_path.parent
    events_stem = events_tsv_path.stem
    base_stem = events_stem.replace("_events", "_meg")

    cand_fif = meg_dir / f"{base_stem}.fif"
    cand_ds = meg_dir / f"{base_stem}.ds"

    meg_candidates = []
    if cand_fif.exists():
        meg_candidates.append(cand_fif)
    if cand_ds.exists():
        meg_candidates.append(cand_ds)

    if not meg_candidates:
        raise RuntimeError(
            f"{events_tsv_path}: cannot find matching MEG file "
            f"(tried {cand_fif.name} / {cand_ds.name})"
        )

    original_meg_path = meg_candidates[0].as_posix()
    _ = _find_bids_root_from_events(events_tsv_path)  # interface placeholder

    segments: List[Dict] = []
    seg_idx = 0
    current_block_type = None  # 'sentence' / 'word_list' / None

    for _, row in df.iterrows():
        onset = float(row["onset"])
        ev_type = str(row["type"])
        value = row["value"]

        if ev_type == "Picture":
            val_upper = str(value).upper()
            if "ZINNEN" in val_upper:
                current_block_type = "sentence"
            elif "WOORDEN" in val_upper:
                current_block_type = "word_list"
            else:
                continue
            continue

        if ev_type != "Sound":
            continue

        if current_block_type not in ("sentence", "word_list"):
            continue

        seq_id = _parse_seq_id_from_sound_value(events_tsv_path, onset, str(value))
        if seq_id not in seq2words:
            raise RuntimeError(
                f"{events_tsv_path}: seq_id={seq_id} not found in Whisper alignments."
            )

        words_timing = seq2words[seq_id]
        if not words_timing:
            raise RuntimeError(
                f"{events_tsv_path}: seq_id={seq_id} has empty Whisper word list."
            )

        starts = [w["start"] for w in words_timing]
        stops = [w["start"] + w["duration"] for w in words_timing]
        seg_audio_onset = float(min(starts))
        seg_audio_offset = float(max(stops))
        seg_audio_dur = seg_audio_offset - seg_audio_onset

        text = " ".join(w["word"] for w in words_timing)

        seg_fif_onset = onset
        seg_fif_offset = onset + seg_audio_dur

        audio_rel = f"stimuli/audio_files/EQ_Ramp_Int2_Int1LPF{seq_id:03d}.wav"
        story_id = f"{seq_id:03d}"

        segment = {
            "subject_id": subject_id,
            "session_id": session_id,
            "type": current_block_type,
            "story_id": story_id,
            "segment_idx_in_file": seg_idx,
            "text": text,
            "words_timing": words_timing,
            "original_fif_path": original_meg_path,
            "original_audio_path": audio_rel,
            "segment_duration_s": round(seg_audio_dur, 3),
            "segment_onset_in_fif_s": round(seg_fif_onset, 5),
            "segment_offset_in_fif_s": round(seg_fif_offset, 5),
            "segment_onset_in_audio_s": round(seg_audio_onset, 5),
            "segment_offset_in_audio_s": round(seg_audio_offset, 5),
            "global_segment_onset_in_audio_s": round(seg_audio_onset, 5),
            "global_segment_offset_in_audio_s": round(seg_audio_offset, 5),
        }

        segments.append(segment)
        seg_idx += 1

    logging.info(f"{events_tsv_path.name}: extracted {len(segments)} segments.")
    return segments


# ---------------- Entry point ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Stage 1 for MOUS: content-driven, Whisper-anchored meta manifest."
    )
    ap.add_argument("--mous_root", required=True, type=str)
    ap.add_argument("--align_jsonl", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument(
        "--subject_allowlist",
        type=str,
        default=None,
        help="Optional subject allowlist (one ID per line).",
    )
    args = ap.parse_args()

    mous_root = Path(args.mous_root)
    align_path = Path(args.align_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allow = load_subject_allowlist(Path(args.subject_allowlist)) if args.subject_allowlist else None
    seq2words = load_whisper_alignments(align_path)

    events_files = sorted(mous_root.glob("sub-*/meg/*task-auditory*events.tsv"))
    if not events_files:
        raise RuntimeError(f"No *task-auditory*_events.tsv found under {mous_root}/sub-*/meg/")

    if allow is not None:
        filtered = []
        for p in events_files:
            sid = _parse_subject_id_from_events_path(p)
            if sid in allow:
                filtered.append(p)
        logging.info(
            f"Filtered events by allowlist: raw={len(events_files)}, kept={len(filtered)}"
        )
        events_files = filtered

    if not events_files:
        raise RuntimeError("No events.tsv left after applying subject_allowlist.")

    logging.info(f"Found {len(events_files)} events.tsv files to process.")

    all_segments: List[Dict] = []
    for ev in events_files:
        segs = extract_segments_from_events(ev, seq2words)
        all_segments.extend(segs)

    if not all_segments:
        raise RuntimeError("No segments extracted from any events.tsv.")

    output_path = out_dir / "meta_manifest_mous.jsonl"
    write_jsonlines(output_path, all_segments)
    logging.info(f"Done. Wrote {len(all_segments)} segments -> {output_path}")


if __name__ == "__main__":
    main()
