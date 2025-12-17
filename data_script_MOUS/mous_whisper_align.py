#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mous_whisper_align.py

Run Whisper on all MOUS sentence audio files and export word-level timestamps.

Input:
  <MOUS_ROOT>/stimuli/audio_files/EQ_Ramp_Int2_Int1LPFXXX.wav

Output:
  <OUT_DIR>/mous_whisper_word_alignments.jsonl

Each JSONL record:
{
  "sequence_id": 123,
  "audio_path": ".../EQ_Ramp_Int2_Int1LPF123.wav",
  "word": "voorbeeld",
  "start": 0.53,
  "end": 0.82,
  "segment_start": 0.40,
  "segment_end": 1.10
}
"""

import argparse
import json
import logging
import re
from pathlib import Path

import whisper
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mous_root",
        required=True,
        help="MOUS BIDS root directory (e.g., /mimer/.../MOUS_raw)",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for Whisper alignments",
    )
    ap.add_argument(
        "--model",
        default="large-v2",
        help="Whisper model name (large-v3 / large-v2 / medium / small ...)",
    )
    ap.add_argument(
        "--language",
        default="nl",
        help="Audio language code (MOUS uses Dutch: nl)",
    )
    args = ap.parse_args()

    mous_root = Path(args.mous_root)
    audio_root = mous_root / "stimuli" / "audio_files"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mous_whisper_word_alignments.jsonl"

    wav_files = sorted(audio_root.glob("EQ_Ramp_Int2_Int1LPF*.wav"))
    if not wav_files:
        logging.error(f"No wav files found under {audio_root}")
        return

    logging.info(f"Found {len(wav_files)} wav files under {audio_root}")
    logging.info(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)

    with out_path.open("w", encoding="utf-8") as f_out:
        for wav in tqdm(wav_files, desc="Whisper align (MOUS)"):
            stem = wav.stem  # EQ_Ramp_Int2_Int1LPF123
            m = re.search(r"(\d+)$", stem)
            if not m:
                logging.warning(f"Cannot parse sequence_id from {stem}; skipping.")
                continue
            seq_id = int(m.group(1))

            # Enable word-level timestamps
            result = model.transcribe(
                str(wav),
                language=args.language,
                word_timestamps=True,
                verbose=False,
            )

            segments = result.get("segments", [])
            if not segments:
                logging.warning(f"No segments returned by Whisper for {wav}")
                continue

            for seg in segments:
                seg_start = float(seg.get("start", 0.0))
                seg_end = float(seg.get("end", seg_start))

                # Some models provide seg["words"] with finer timestamps
                words = seg.get("words")
                if not words:
                    # Fallback: treat the entire segment as a single token
                    words = [{
                        "word": seg.get("text", "").strip(),
                        "start": seg_start,
                        "end": seg_end,
                    }]

                for w in words:
                    word = w.get("word", "").strip()
                    if not word:
                        continue
                    start = float(w.get("start", seg_start))
                    end = float(w.get("end", seg_end))
                    rec = {
                        "sequence_id": seq_id,
                        "audio_path": str(wav),
                        "word": word,
                        "start": start,
                        "end": end,
                        "segment_start": seg_start,
                        "segment_end": seg_end,
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logging.info(f"Done. Alignments written to {out_path}")


if __name__ == "__main__":
    main()
