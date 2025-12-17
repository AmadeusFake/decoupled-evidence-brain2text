#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a global, de-duplicated sentence table from window-level manifests.

Sentence key:
  sentence_key = "{audio_stem}::{on_ms}-{off_ms}"
  where on/off are rounded milliseconds from segment onset/offset.

Each output row contains:
  sentence_id, sentence_key, original_audio_path,
  segment_onset_in_audio_s, segment_offset_in_audio_s,
  text (if available), members (window_ids)
"""

import argparse, json, logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build_sent_table_dedup")

# -------------------- IO -------------------- #

def read_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(p: Path, rows: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------------- Utils -------------------- #

def _ms(x: float) -> int:
    return int(round(float(x) * 1000.0))

def make_sentence_key(row: dict) -> str:
    """Derive sentence_key from audio stem and segment timestamps."""
    ap = row.get("original_audio_path") or row.get("audio_path") or ""
    stem = Path(ap).stem

    on  = row.get("global_segment_onset_in_audio_s", row.get("segment_onset_in_audio_s"))
    off = row.get("global_segment_offset_in_audio_s", row.get("segment_offset_in_audio_s"))
    on  = 0.0 if on  is None else float(on)
    off = 0.0 if off is None else float(off)

    return f"{stem}::{_ms(on)}-{_ms(off)}"

# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_window_manifest_dir",
        required=True,
        help="Directory containing train/valid/test window manifests",
    )
    ap.add_argument(
        "--output_sentence_table_dir",
        required=True,
        help="Output directory for de-duplicated sentence tables",
    )
    args = ap.parse_args()

    in_dir  = Path(args.input_window_manifest_dir)
    out_dir = Path(args.output_sentence_table_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sp in ("train", "valid", "test"):
        src = in_dir / f"{sp}.jsonl"
        rows = read_jsonl(src)
        if not rows:
            log.warning(f"[{sp}] empty or missing: {src}")
            continue

        groups: Dict[str, dict] = {}
        for r in rows:
            sk = make_sentence_key(r)
            g = groups.get(sk)
            if g is None:
                g = {
                    "sentence_id": sk,
                    "sentence_key": sk,
                    "original_audio_path": r.get("original_audio_path", r.get("audio_path", "")),
                    "segment_onset_in_audio_s": float(
                        r.get(
                            "global_segment_onset_in_audio_s",
                            r.get("segment_onset_in_audio_s", 0.0),
                        )
                    ),
                    "segment_offset_in_audio_s": float(
                        r.get(
                            "global_segment_offset_in_audio_s",
                            r.get("segment_offset_in_audio_s", 0.0),
                        )
                    ),
                    "text": r.get("global_segment_text", r.get("text", "")),
                    "members": [],
                }
                groups[sk] = g

            wid = r.get("window_id")
            if wid not in (None, ""):
                g["members"].append(str(wid))

        for k in groups:
            groups[k]["members"] = sorted(set(groups[k]["members"]))

        outp = out_dir / f"{sp}.jsonl"
        write_jsonl(outp, list(groups.values()))
        log.info(
            f"[{sp}] windows={len(rows):,} -> unique_sentences={len(groups):,} -> {outp}"
        )

    log.info("Done building de-duplicated sentence tables.")

if __name__ == "__main__":
    main()
