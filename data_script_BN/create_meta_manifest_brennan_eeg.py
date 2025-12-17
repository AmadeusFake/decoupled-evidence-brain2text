#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create meta manifest for Brennan EEG (sentence-only, robust to variable trial counts).

- Join proc.trl with AliceChapterOne-EEG.csv using `Order`
- Build sentence-level segments only
- Time conventions:
    * words_timing.start  : relative to audio sentence onset
    * segment_onset_in_fif_s = first word EEG start (absolute EEG time)
    * segment_offset_in_fif_s = last word EEG end   (absolute EEG time)
"""

import argparse, logging, re
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.io import loadmat
import jsonlines

SFREQ = 500.0
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _safe_varnames(varnames) -> List[str]:
    # Ensure variable names are returned as a list
    return list(varnames)


def _read_proc_mat(proc_mat_path: Path) -> pd.DataFrame:
    # Load proc.mat and extract trial-level metadata
    proc = loadmat(
        proc_mat_path,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True
    )["proc"]

    trl = proc["trl"]
    varnames = _safe_varnames(proc["varnames"])

    if trl.ndim != 2:
        trl = np.atleast_2d(trl)

    if len(varnames) != trl.shape[1]:
        prefix = ["start_sample", "stop_sample", "offset"]
        assert len(prefix) + len(varnames) == trl.shape[1], \
            f"varnames({len(varnames)}) + 3 != trl_cols({trl.shape[1]}) @ {proc_mat_path}"
        columns = prefix + varnames
    else:
        columns = varnames

    meta = pd.DataFrame(trl, columns=[f"_{c}" for c in columns])

    # Locate Order column (case-insensitive)
    order_col = None
    for cand in ["_Order", "_order", "_ORDER"]:
        if cand in meta.columns:
            order_col = cand
            break
    if order_col is None:
        raise RuntimeError(f"`Order` column not found in {proc_mat_path}")

    if "_start_sample" in meta.columns:
        meta["_start_sample"] = meta["_start_sample"].astype(float)

    meta["_Order"] = meta[order_col].astype(int)
    return meta


def _subject_id_from_path(mat_path: Path) -> str:
    # Extract subject ID from file name
    m = re.search(r"(S\d{2})\.mat$", mat_path.name, flags=re.I)
    return m.group(1).upper() if m else mat_path.stem


def _resolve_raw_mat(eeg_root: Path, sid: str) -> Path:
    # Resolve raw EEG .mat path
    p1 = eeg_root / f"{sid}.mat"
    p2 = eeg_root / "proc" / f"{sid}.mat"
    return p1 if p1.exists() else (p2 if p2.exists() else p1)


def _build_sentence_segments_for_subject(
    eeg_root: Path, sid: str, story_df: pd.DataFrame, meta_df: pd.DataFrame
) -> List[Dict]:
    # Build sentence-level segments for one subject

    use_cols = ["Word", "Segment", "onset", "offset", "Order", "Position", "Sentence"]
    story = story_df[use_cols].copy()
    story["Order"] = story["Order"].astype(int)
    story["duration"] = story["offset"] - story["onset"]

    ev = meta_df.merge(story, left_on="_Order", right_on="Order", how="inner")
    if ev.empty:
        logging.warning(f"{sid}: join result is empty; skip.")
        return []

    if "_start_sample" not in ev.columns:
        raise RuntimeError(f"{sid}: `_start_sample` missing in proc.trl")

    # EEG absolute time (seconds)
    ev["eeg_start_sec"] = ev["_start_sample"] / SFREQ
    ev["eeg_end_sec"] = ev["eeg_start_sec"] + ev["duration"].astype(float)

    def pack_sentence(df_g: pd.DataFrame, sent_id_in_file: int) -> Dict:
        # Pack one sentence into a segment dict
        df_g = df_g.sort_values("onset")

        # Audio timeline
        first_onset = float(df_g["onset"].iloc[0])
        last_offset = float(df_g["offset"].iloc[-1])

        # EEG absolute timeline
        first_eeg_sta = float(df_g["eeg_start_sec"].iloc[0])
        last_eeg_end = float(df_g["eeg_end_sec"].iloc[-1])

        # Word-level timing (relative to audio onset)
        words = []
        for _, r in df_g.iterrows():
            words.append({
                "word": r["Word"],
                "start": float(r["onset"]) - first_onset,
                "duration": float(r["duration"]),
                "sound": f"audio/DownTheRabbitHoleFinal_SoundFile{int(r['Segment'])}.wav",
                "word_id": int(r["Position"]),
                "sequence_id": int(r["Sentence"]),
                "segment": int(r["Segment"]),
                "order": int(r["Order"]),
            })

        # Use absolute EEG time (no audio offset subtraction)
        return {
            "subject_id": sid,
            "session_id": "01",
            "type": "sentence",
            "story_id": "alice_ch1",
            "segment_idx_in_file": int(sent_id_in_file),

            "text": " ".join([w["word"] for w in words]),
            "words_timing": words,

            "original_fif_path": str(_resolve_raw_mat(eeg_root, sid)),
            "original_audio_path": f"audio/DownTheRabbitHoleFinal_SoundFile{int(df_g['Segment'].iloc[0])}.wav",

            "segment_onset_in_fif_s": round(first_eeg_sta, 5),
            "segment_offset_in_fif_s": round(last_eeg_end, 5),

            "segment_onset_in_audio_s": round(first_onset, 5),
            "segment_offset_in_audio_s": round(last_offset, 5),

            "segment_duration_s": round((last_eeg_end - first_eeg_sta), 3),
        }

    out: List[Dict] = []
    for sent_id, df_g in ev.groupby("Sentence"):
        out.append(pack_sentence(df_g, int(sent_id)))
    return out


def _write_jsonlines(p: Path, items: List[Dict]):
    # Write segments to JSONL
    p.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(p, "w") as w:
        w.write_all(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--brennan_root",
        required=True,
        type=str,
        help="Contains audio/, proc/, Sxx.mat, AliceChapterOne-EEG.csv"
    )
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument(
        "--keep_subjects_file",
        required=False,
        type=str,
        help="Subject IDs to keep, one per line; default uses all proc/*.mat"
    )
    args = ap.parse_args()

    root = Path(args.brennan_root)
    out_dir = Path(args.output_dir)

    csv_path = root / "AliceChapterOne-EEG.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    story_df = pd.read_csv(csv_path)
    need = {"Word","Segment","onset","offset","Order","Position","Sentence"}
    miss = need - set(story_df.columns)
    if miss:
        raise RuntimeError(f"AliceChapterOne-EEG.csv missing columns: {miss}")

    if args.keep_subjects_file:
        keep = [
            ln.strip()
            for ln in Path(args.keep_subjects_file).read_text().splitlines()
            if ln.strip()
        ]
        subjects = [sid for sid in keep if (root / "proc" / f"{sid}.mat").exists()]
    else:
        subjects = sorted([p.stem for p in (root / "proc").glob("S*.mat")])

    logging.info(f"Subjects to process: {subjects}")

    all_segments: List[Dict] = []
    for sid in subjects:
        proc_mat = root / "proc" / f"{sid}.mat"
        try:
            meta_df = _read_proc_mat(proc_mat)
            segs = _build_sentence_segments_for_subject(root, sid, story_df, meta_df)
            all_segments.extend(segs)
        except Exception as e:
            logging.error(f"{sid}: failed with {e}")

    out_path = out_dir / "meta_manifest.jsonl"
    _write_jsonlines(out_path, all_segments)
    logging.info(f"Wrote {len(all_segments)} segments -> {out_path}")


if __name__ == "__main__":
    main()
