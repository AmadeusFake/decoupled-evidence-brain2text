#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export sentence-level audio clips (.wav) and text (.txt) for QC or release.

Features:
- Case-insensitive audio lookup via recursive index
- Stable cutting with optional head/tail padding (default: +0.25s tail)
- Text sources:
    * stageT  : use existing sentence index (text_used)
    * whisper : transcribe each clip with Whisper
    * auto    : stageT first, Whisper fallback
Requires: ffmpeg; optional openai-whisper
"""

import argparse, json, logging, subprocess
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("qc_export_wav_txt_v2")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

# -------------------- IO -------------------- #

def load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p, "r", encoding="utf-8") if l.strip()]

# -------------------- Audio utils -------------------- #

def ffmpeg_cut(
    ffmpeg_bin: str,
    in_audio: Path,
    out_wav: Path,
    onset_s: float,
    offset_s: float,
    head_pad_s: float = 0.0,
    tail_pad_s: float = 0.25,
):
    """Cut a mono 16kHz wav clip with optional padding."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, float(onset_s) - max(0.0, head_pad_s))
    dur = max(
        0.05,
        float(offset_s) - float(onset_s)
        + max(0.0, tail_pad_s)
        + max(0.0, head_pad_s),
    )
    cmd = [
        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", in_audio.as_posix(),
        "-ac", "1", "-ar", "16000", "-vn",
        out_wav.as_posix(),
    ]
    subprocess.run(cmd, check=True)

def build_audio_index(roots: List[Path]) -> Dict[str, Path]:
    """Index audio files under roots (case-insensitive)."""
    idx: Dict[str, Path] = {}
    for root in roots:
        if not root or not root.exists():
            continue
        root = root.resolve()
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                idx[p.name.lower()] = p
                try:
                    idx[p.relative_to(root).as_posix().lower()] = p
                except Exception:
                    pass
    log.info(f"audio index built from {len(roots)} roots, entries={len(idx):,}")
    return idx

def resolve_audio_path(raw_path: str, idx: Dict[str, Path], roots: List[Path]) -> Optional[Path]:
    """Resolve audio path via direct hit, name, or relative path."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.exists() and p.suffix.lower() in AUDIO_EXTS:
        return p.resolve()

    name = p.name.lower()
    if name in idx:
        return idx[name]

    if p.suffix == "":
        stem = p.stem.lower()
        for ext in AUDIO_EXTS:
            key = f"{stem}{ext}"
            if key in idx:
                return idx[key]

    for r in roots:
        try:
            rel = p.relative_to(r).as_posix().lower()
            if rel in idx:
                return idx[rel]
        except Exception:
            pass
    return None

# -------------------- Whisper (optional) -------------------- #

def load_whisper(model_name: str, device: str):
    try:
        import whisper
    except Exception as e:
        raise RuntimeError("openai-whisper required: pip install -U openai-whisper") from e
    return whisper.load_model(model_name, device=device)

def transcribe_clip(whisper_model, wav_path: Path, language: Optional[str] = "en") -> str:
    """Short-clip transcription (no timestamps, low temperature)."""
    result = whisper_model.transcribe(
        wav_path.as_posix(),
        language=language,
        temperature=0.0,
        word_timestamps=False,
        verbose=False,
    )
    return (result.get("text", "") or "").strip()

# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence_table_dir", required=True, help="sent_table_dedup/{split}.jsonl")
    ap.add_argument("--stageT_index_dir", type=str, default="", help="Optional StageT index (text_used)")
    ap.add_argument("--out_dir", required=True, help="Output root (split subdirs)")
    ap.add_argument("--audio_roots", nargs="*", default=[], help="Audio root directories")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--head_pad_s", type=float, default=0.00)
    ap.add_argument("--tail_pad_s", type=float, default=0.25)
    ap.add_argument("--txt_source", choices=["auto", "stageT", "whisper"], default="auto")
    ap.add_argument("--whisper_model", default="large-v3")
    ap.add_argument("--whisper_device", default="cuda")
    ap.add_argument("--whisper_language", default="en")
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Build audio roots from args + sentence tables
    roots = [Path(x).resolve() for x in args.audio_roots]
    for sp in ("train", "valid", "test"):
        for r in load_jsonl(Path(args.sentence_table_dir) / f"{sp}.jsonl"):
            apath = r.get("original_audio_path", r.get("audio_path", ""))
            if apath:
                parent = Path(apath).parent
                if parent.exists():
                    rp = parent.resolve()
                    if rp not in roots:
                        roots.append(rp)

    audio_index = build_audio_index(roots)

    # Load StageT index (optional)
    id2idx: Dict[str, dict] = {}
    if args.stageT_index_dir:
        for sp in ("train", "valid", "test"):
            p = Path(args.stageT_index_dir) / f"{sp}.jsonl"
            for j in load_jsonl(p):
                sid = str(j.get("sentence_id", j.get("sentence_key", "")))
                id2idx[sid] = j

    # Whisper setup (optional)
    whisper_model = None
    use_whisper = args.txt_source in ("whisper", "auto")
    if use_whisper:
        whisper_model = load_whisper(args.whisper_model, args.whisper_device)
        log.info(f"Whisper loaded: {args.whisper_model} on {args.whisper_device}")

    for sp in ("train", "valid", "test"):
        srows = load_jsonl(Path(args.sentence_table_dir) / f"{sp}.jsonl")
        if not srows:
            log.warning(f"[{sp}] missing sentence table; skip.")
            continue

        clip_dir = out_root / sp / "audio"
        clip_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = out_root / f"qc_{sp}.tsv"

        with open(tsv_path, "w", encoding="utf-8") as tf:
            tf.write(
                "idx\tsentence_id\tsentence_key\tresolved_audio\t"
                "onset_s\toffset_s\twav_rel\ttxt_rel\tsource\n"
            )

            miss_audio = miss_txt = 0
            for i, r in enumerate(srows, 1):
                sid = str(r.get("sentence_id", r.get("sentence_key", "")))
                sk  = r.get("sentence_key", "")
                ap  = r.get("original_audio_path", r.get("audio_path", ""))
                on  = float(r.get("segment_onset_in_audio_s", r.get("onset_audio_s", 0.0)))
                off = float(r.get("segment_offset_in_audio_s", r.get("offset_audio_s", 0.0)))

                in_audio = resolve_audio_path(ap, audio_index, roots)
                if in_audio is None:
                    miss_audio += 1
                    log.warning(f"[{sp}] audio not found: {ap}")
                    tf.write(f"{i}\t{sid}\t{sk}\tNOT_FOUND\t{on:.3f}\t{off:.3f}\t\t\t\n")
                    continue

                wav_out = clip_dir / f"{sid}.wav"
                try:
                    ffmpeg_cut(
                        args.ffmpeg, in_audio, wav_out,
                        on, off, args.head_pad_s, args.tail_pad_s
                    )
                except Exception as e:
                    miss_audio += 1
                    log.warning(f"[{sp}] ffmpeg failed for {in_audio}: {e}")
                    tf.write(f"{i}\t{sid}\t{sk}\t{in_audio}\t{on:.3f}\t{off:.3f}\t\t\t\n")
                    continue

                txt_out = clip_dir / f"{sid}.txt"
                src = ""
                text_used = ""

                if args.txt_source in ("auto", "stageT"):
                    jj = id2idx.get(sid, {})
                    text_used = (jj.get("text_used", "") or "").strip()
                    if text_used:
                        src = "stageT"

                if args.txt_source == "whisper" or (args.txt_source == "auto" and not text_used):
                    if whisper_model is None:
                        raise RuntimeError("Whisper required but not loaded.")
                    try:
                        text_used = transcribe_clip(
                            whisper_model, wav_out, args.whisper_language
                        )
                        src = "whisper"
                    except Exception as e:
                        miss_txt += 1
                        log.warning(f"[{sp}] whisper failed for {wav_out}: {e}")
                        text_used = ""
                        src = ""

                if text_used:
                    with open(txt_out, "w", encoding="utf-8") as ftxt:
                        ftxt.write(text_used + "\n")
                    wav_rel = wav_out.relative_to(out_root).as_posix()
                    txt_rel = txt_out.relative_to(out_root).as_posix()
                else:
                    miss_txt += 1
                    wav_rel = wav_out.relative_to(out_root).as_posix()
                    txt_rel = ""

                tf.write(
                    f"{i}\t{sid}\t{sk}\t{in_audio}\t{on:.3f}\t{off:.3f}\t"
                    f"{wav_rel}\t{txt_rel}\t{src}\n"
                )

        log.info(
            f"[{sp}] clips & txt -> {clip_dir} ; tsv -> {tsv_path} | "
            f"miss_audio={miss_audio}, miss_txt={miss_txt}"
        )

    log.info("Export done.")

if __name__ == "__main__":
    main()
