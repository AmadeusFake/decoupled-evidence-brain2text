#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attach sentence-level TXT(E5) features to window manifests using exact key match.

- Exact match only (no tolerance, no fuzzy logic)
- 0.0 timestamps are treated as valid
- Key format: <audio_stem_lower>::<on_ms>-<off_ms>
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("attach_text_features_exact")

_KEY_PAT = re.compile(r"^(?P<base>.+?)::(?P<on>\d+)-(?P<off>\d+)$")

# --------------------- IO --------------------- #

def load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(p: Path, rows: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --------------------- Utils --------------------- #

def norm_base_audio(p: str) -> str:
    """Lowercase audio stem without extension."""
    b = Path(p).name
    if "." in b:
        b = ".".join(b.split(".")[:-1])
    return b.lower()

def parse_key(k: str) -> Optional[Tuple[str, int, int]]:
    if not isinstance(k, str):
        return None
    m = _KEY_PAT.match(k.strip())
    if not m:
        return None
    try:
        return m.group("base").lower(), int(m.group("on")), int(m.group("off"))
    except Exception:
        return None

def to_ms(x) -> int:
    try:
        return int(round(float(x) * 1000.0))
    except Exception:
        return -1

def _pick_float(d: dict, keys: List[str]) -> Optional[float]:
    """Return first non-None float value (0.0 is valid)."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return None

def derive_key_from_window(w: dict) -> Tuple[str, int, int]:
    """
    Derive (base, on_ms, off_ms) from a window row.
    Falls back to content_id if needed.
    """
    ap = w.get("original_audio_path", "")
    if (not ap) and isinstance(w.get("content_id"), str) and "::" in w["content_id"]:
        ap = w["content_id"].split("::", 1)[0]
    base = norm_base_audio(ap) if ap else ""

    on = _pick_float(w, [
        "global_segment_onset_in_audio_s",
        "segment_onset_in_audio_s",
        "onset_audio_s",
    ])
    off = _pick_float(w, [
        "global_segment_offset_in_audio_s",
        "segment_offset_in_audio_s",
        "offset_audio_s",
    ])

    if (on is None or off is None) and isinstance(w.get("content_id"), str) and "::" in w["content_id"]:
        try:
            so, eo = w["content_id"].split("::", 1)[1].split("-", 1)
            if on  is None: on  = float(so)
            if off is None: off = float(eo)
        except Exception:
            pass

    if on is None or off is None:
        return base, -1, -1

    return base, to_ms(on), to_ms(off)

def build_sent_index(idx_path: Path, feature_field: str):
    """
    Build exact key -> feature path map from sentence index.
    """
    rows = load_jsonl(idx_path)
    exact_map: Dict[str, str] = {}
    bases = set()

    def pick_key(d: dict) -> Optional[str]:
        for k in ("sentence_key", "sentence_id", "content_id", "key", "id"):
            v = d.get(k)
            if isinstance(v, str) and "::" in v:
                return v
        fp = d.get(feature_field, "")
        if isinstance(fp, str) and fp:
            m = _KEY_PAT.search(Path(fp).stem)
            if m:
                return f"{m.group('base')}::{m.group('on')}-{m.group('off')}"
        return None

    used = 0
    for j in rows:
        feat = j.get(feature_field, "")
        if not isinstance(feat, str) or not feat:
            continue
        k = pick_key(j)
        if not k:
            continue
        parsed = parse_key(k)
        if not parsed:
            continue
        base, on, off = parsed
        exact_map[f"{base}::{on}-{off}"] = feat
        bases.add(base)
        used += 1

    log.info(f"[index] {idx_path.name}: rows={len(rows)}, usable={used}, bases={len(bases)}")
    return exact_map, bases

# --------------------- Core --------------------- #

def process_split(
    split: str,
    in_win_dir: Path,
    in_sent_dir: Path,
    out_dir: Path,
    feature_field: str,
    out_field: str,
    emit_tsv: bool,
):
    win_p = in_win_dir / f"{split}.jsonl"
    idx_p = in_sent_dir / f"{split}.jsonl"
    if not win_p.exists() or not idx_p.exists():
        log.warning(f"[{split}] missing input -> skip")
        return

    exact_map, _ = build_sent_index(idx_p, feature_field)
    rows = load_jsonl(win_p)

    out_rows: List[dict] = []
    hits_exact = 0
    miss = 0

    map_lines: List[tuple] = []
    miss_lines: List[tuple] = []
    miss_by_base: Dict[str, int] = {}

    for r in rows:
        base, on, off = derive_key_from_window(r)
        wanted_key = f"{base}::{on}-{off}"
        feat_path = ""

        if base and on >= 0 and off >= 0:
            fp = exact_map.get(wanted_key, "")
            if isinstance(fp, str) and fp:
                feat_path = fp
                hits_exact += 1
                if emit_tsv:
                    map_lines.append((
                        r.get("window_id",""), base, on, off,
                        wanted_key, wanted_key, 0, 0, feat_path, "exact"
                    ))

        if not feat_path:
            miss += 1
            miss_by_base[base] = miss_by_base.get(base, 0) + 1
            if emit_tsv:
                miss_lines.append((
                    r.get("window_id",""), base, on, off, wanted_key,
                    r.get("sentence_id",""),
                    r.get("content_id",""),
                    r.get("original_audio_path",""),
                ))

        r[out_field] = feat_path
        out_rows.append(r)

    out_p = out_dir / f"{split}.jsonl"
    write_jsonl(out_p, out_rows)
    log.info(f"[{split}] -> {out_p} | total={len(rows):,} exact={hits_exact:,} miss={miss:,}")

    if emit_tsv:
        tsv_map = out_dir / f"{split}_windows_to_sentence.tsv"
        with open(tsv_map, "w", encoding="utf-8") as f:
            f.write("window_id\tbase\ton_ms\toff_ms\twanted_key\tmatched_key\td_on_ms\td_off_ms\tfeature_path\tmatch_type\n")
            for row in map_lines:
                f.write("\t".join(str(x) for x in row) + "\n")

        if miss_lines:
            tsv_miss = out_dir / f"{split}_miss.tsv"
            with open(tsv_miss, "w", encoding="utf-8") as f:
                f.write("window_id\tbase\ton_ms\toff_ms\twanted_key\tsentence_id\tcontent_id\toriginal_audio_path\n")
                for row in miss_lines:
                    f.write("\t".join(str(x) for x in row) + "\n")

            top = sorted(miss_by_base.items(), key=lambda kv: kv[1], reverse=True)[:5]
            if top:
                log.info(f"[{split}] top miss bases: " + ", ".join(f"{b}:{n}" for b, n in top))

# --------------------- main --------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_window_manifest_dir", required=True)
    ap.add_argument("--input_sentence_manifest_dir", required=True)
    ap.add_argument("--output_window_manifest_dir", required=True)
    ap.add_argument("--feature_field", default="text_sentence_feature_path")
    ap.add_argument("--output_feature_field", default="text_sentence_feature_path")
    ap.add_argument("--emit_mapping_tsv", action="store_true")
    args = ap.parse_args()

    in_win_dir  = Path(args.input_window_manifest_dir)
    in_sent_dir = Path(args.input_sentence_manifest_dir)
    out_dir     = Path(args.output_window_manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        process_split(
            split,
            in_win_dir, in_sent_dir, out_dir,
            args.feature_field, args.output_feature_field,
            args.emit_mapping_tsv,
        )

    log.info("Done (exact match, zero tolerance).")

if __name__ == "__main__":
    main()
