#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 3.5 — Add sentence-level full MEG (incremental, non-destructive)

Purpose:
- Build sentence-level MEG arrays [C, T_sentence] on top of completed Stage-3 outputs.
- Processing is aligned with Stage-3:
  same sampling rate, baseline correction, RobustScaler (Q25/Q75) and clamping.
- Existing window .npy files are never modified.
- Only new files {sentence_id}_sent.npy are added, and their paths are written
  to the manifest key: `meg_sentence_full_path`.

Features:
- Resume / idempotent: skip existing outputs if verified.
- Optional recomputation with --recompute_existing.
- RobustScaler parameters are reused from Stage-3 if available;
  otherwise fitted from window-level .npy files of the same recording.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

try:
    import mne
except Exception as e:
    raise RuntimeError(
        "Stage 3.5 requires mne to read raw MEG (FIF/CON). Please install mne."
    ) from e

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("stage3_5_sentence_full")


# --------------------------- I/O utils --------------------------- #
def load_split(manifest_dir: Path, split: str) -> List[Dict]:
    p = manifest_dir / f"{split}.jsonl"
    if not p.exists():
        logger.warning(f"[{split}] manifest missing: {p}")
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_manifest(rows: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def atomic_save_npy(path: Path, arr: np.ndarray):
    """Atomically write .npy to avoid partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def verify_meg_npy(path: Path, expected_T: Optional[int]) -> bool:
    try:
        if not path.exists():
            return False
        arr = np.load(path.as_posix(), mmap_mode="r")
        if arr.ndim != 2:
            return False
        if expected_T is not None and arr.shape[1] != expected_T:
            return False
        return True
    except Exception:
        return False


# --------------------------- raw / span utils --------------------------- #
def read_raw(path: Path) -> mne.io.BaseRaw:
    if path.suffix.lower() == ".con":
        return mne.io.read_raw_kit(path, preload=True, verbose=False)
    return mne.io.read_raw_fif(path, preload=True, verbose=False)

def resolve_rec_path(row: Dict) -> Optional[Path]:
    """Resolve recording path from known manifest keys."""
    for k in (
        "original_fif_path",
        "original_meg_path",
        "fif_path",
        "raw_path",
        "recording_path",
    ):
        if k in row and row[k]:
            p = Path(row[k])
            if p.exists():
                return p
    return None

def sentence_key(row: Dict) -> Optional[str]:
    sid = row.get("sentence_id", None)
    return str(sid) if sid is not None else None

def sentence_span_T(row: Dict, sfreq: float) -> Optional[Tuple[int, int]]:
    """Return [start, end) sample indices for a sentence."""
    if (
        "global_segment_onset_in_fif_s" not in row
        or "global_segment_offset_in_fif_s" not in row
    ):
        return None
    on = float(row["global_segment_onset_in_fif_s"])
    off = float(row["global_segment_offset_in_fif_s"])
    a = int(round(on * sfreq))
    b = int(round(off * sfreq))
    if b <= a:
        return None
    return a, b


# --------------------------- RobustScaler (aligned with Stage-3) --------------------------- #
def fit_robust_from_windows(
    win_paths: List[Path],
    fit_max_windows: int,
    q_low: float = 0.25,
    q_high: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-channel Q25/Q75 from window-level MEG arrays."""
    if not win_paths:
        raise RuntimeError("No window .npy files available to fit RobustScaler.")

    paths = (
        win_paths
        if fit_max_windows <= 0 or len(win_paths) <= fit_max_windows
        else win_paths[:fit_max_windows]
    )

    arrs = []
    for p in paths:
        try:
            x = np.load(p.as_posix()).astype(np.float32)
            if x.ndim == 2:
                arrs.append(x)
        except Exception:
            continue

    if not arrs:
        raise RuntimeError("Failed to load any valid window arrays.")

    cat = np.concatenate(arrs, axis=1).astype(np.float32, copy=False)
    q25 = np.quantile(cat, q_low, axis=1)
    q75 = np.quantile(cat, q_high, axis=1)
    center = ((q25 + q75) * 0.5).astype(np.float32)
    scale = ((q75 - q25) * 0.5).astype(np.float32)
    scale[scale == 0.0] = 1.0
    return center, scale

def apply_robust_and_clamp(
    seg: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    std_clamp: float,
) -> np.ndarray:
    """Apply robust normalization and clamp."""
    y = (seg - center[:, None]) / scale[:, None]
    np.clip(y, -std_clamp, std_clamp, out=y)
    return y.astype(np.float32)


# --------------------------- per-recording processing --------------------------- #
def process_recording_for_sentences(
    rec_path: Path,
    rec_rows: List[Dict],
    out_sentence_root: Path,
    target_sfreq: float,
    baseline_end_s: float,
    std_clamp: float,
    fit_max_windows: int,
    resume: bool,
    verify_existing: bool,
    recompute_existing: bool,
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Generate sentence-level MEG arrays for one recording and
    write `meg_sentence_full_path` back to corresponding rows.
    """
    stats = dict(
        sent_total=0,
        sent_saved=0,
        sent_skipped_exist=0,
        sent_skipped_oob=0,
        win_refs=0,
    )
    updated_rows: List[Dict] = []

    # Recording-specific output directory
    rec_dir = out_sentence_root / rec_path.stem
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Collect window .npy files for fallback RobustScaler fitting
    win_paths = []
    for r in rec_rows:
        p = Path(r.get("meg_win_path", ""))
        if p.exists():
            win_paths.append(p)
    stats["win_refs"] = len(win_paths)

    # Try to reuse Stage-3 robust center/scale
    center = scale = None
    if win_paths:
        probe_dir = win_paths[0].parent
        cpath = probe_dir / f"{rec_path.stem}_robust_center.npy"
        spath = probe_dir / f"{rec_path.stem}_robust_scale.npy"
        if cpath.exists() and spath.exists():
            try:
                center = np.load(cpath.as_posix()).astype(np.float32)
                scale = np.load(spath.as_posix()).astype(np.float32)
            except Exception:
                center = scale = None

    if center is None or scale is None:
        logger.info(
            f"[{rec_path.stem}] robust params not found; fitting from windows ({len(win_paths)} refs)"
        )
        center, scale = fit_robust_from_windows(win_paths, fit_max_windows)

    C = center.shape[0]

    # Build sentence spans
    sent_span: Dict[str, Tuple[int, int]] = {}
    for r in rec_rows:
        sid = sentence_key(r)
        if not sid or sid in sent_span:
            continue
        span = sentence_span_T(r, target_sfreq)
        if span:
            sent_span[sid] = span

    if not sent_span:
        logger.warning(f"[{rec_path.stem}] no valid sentence spans; skip.")
        return list(rec_rows), stats

    # Load and preprocess raw MEG
    try:
        raw = read_raw(rec_path)
        raw.pick(picks="meg", exclude="bads")
        raw.resample(target_sfreq, npad="auto", verbose=False)
        picks = mne.pick_types(raw.info, meg=True, exclude="bads")
        data = raw.get_data(picks=picks)
        if data.shape[0] != C:
            minC = min(C, data.shape[0])
            data = data[:minC, :]
            center = center[:minC]
            scale = scale[:minC]
            C = minC
    except Exception as e:
        logger.error(f"Failed to read raw MEG for {rec_path}: {e}", exc_info=True)
        return list(rec_rows), stats

    # Generate sentence-level arrays
    for sid, (s0, s1) in sent_span.items():
        stats["sent_total"] += 1
        outp = rec_dir / f"{sid}_sent.npy"

        if (not recompute_existing) and verify_meg_npy(outp, expected_T=None):
            stats["sent_skipped_exist"] += 1
        else:
            if s0 < 0 or s1 > data.shape[1] or (s1 - s0) <= 0:
                stats["sent_skipped_oob"] += 1
            else:
                seg = data[:C, s0:s1].astype(np.float32, copy=True)

                # Baseline correction (first baseline_end_s seconds)
                T_bl = int(round(baseline_end_s * target_sfreq))
                if T_bl > 0:
                    b_end = min(T_bl, seg.shape[1])
                    seg -= seg[:, :b_end].mean(axis=1, keepdims=True).astype(np.float32)

                seg_n = apply_robust_and_clamp(seg, center, scale, std_clamp)
                atomic_save_npy(outp, seg_n)

        # Write path back to all rows of this sentence
        for r in rec_rows:
            if sentence_key(r) == sid:
                r2 = dict(r)
                r2["meg_sentence_full_path"] = outp.as_posix()
                updated_rows.append(r2)

    updated_rows.sort(key=lambda x: x.get("window_id", ""))
    return updated_rows, stats


# --------------------------- main --------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_manifest_dir",
        required=True,
        help="Stage-3 manifest directory (train/valid/test.jsonl)",
    )
    ap.add_argument(
        "--output_sentence_dir",
        required=True,
        help="Output root for sentence-level MEG .npy files",
    )
    ap.add_argument(
        "--output_manifest_dir",
        required=True,
        help="Output directory for updated manifests",
    )
    ap.add_argument("--target_sfreq", type=float, default=120.0)
    ap.add_argument("--baseline_end_s", type=float, default=0.5)
    ap.add_argument("--std_clamp", type=float, default=20.0)
    ap.add_argument("--fit_max_windows_per_recording", type=int, default=200)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--verify_existing", action="store_true", default=True)
    ap.add_argument("--recompute_existing", action="store_true", default=False)
    args = ap.parse_args()

    in_dir = Path(args.input_manifest_dir)
    out_sent_dir = Path(args.output_sentence_dir)
    out_dir = Path(args.output_manifest_dir)
    out_sent_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        rows = load_split(in_dir, split)
        if not rows:
            logger.info(f"[{split}] empty or missing; skip.")
            continue
        logger.info(f"[{split}] loaded {len(rows):,} rows")

        # Group rows by recording
        groups: Dict[Path, List[Dict]] = defaultdict(list)
        for r in rows:
            rp = resolve_rec_path(r)
            if rp is None:
                groups[Path("__NO_RAW__")].append(r)
            else:
                groups[rp].append(r)

        updated_all: List[Dict] = []
        total_stats = dict(
            sent_total=0,
            sent_saved=0,
            sent_skipped_exist=0,
            sent_skipped_oob=0,
            win_refs=0,
        )

        for i, (rec_path, rec_rows) in enumerate(groups.items(), start=1):
            if rec_path.name == "__NO_RAW__":
                logger.warning(
                    f"[{split}] {len(rec_rows)} rows missing raw path; kept unchanged."
                )
                updated_all.extend(rec_rows)
                continue

            logger.info(
                f"[{split}] ({i}/{len(groups)}) {rec_path.name} with {len(rec_rows)} windows"
            )
            upd, st = process_recording_for_sentences(
                rec_path=rec_path,
                rec_rows=rec_rows,
                out_sentence_root=out_sent_dir,
                target_sfreq=args.target_sfreq,
                baseline_end_s=args.baseline_end_s,
                std_clamp=args.std_clamp,
                fit_max_windows=args.fit_max_windows_per_recording,
                resume=args.resume,
                verify_existing=args.verify_existing,
                recompute_existing=args.recompute_existing,
            )
            updated_all.extend(upd)
            for k in total_stats:
                total_stats[k] += st.get(k, 0)

        updated_all.sort(key=lambda x: x.get("window_id", ""))
        out_path = out_dir / f"{split}.jsonl"
        save_manifest(updated_all, out_path)

        logger.info(
            f"[{split}] sentences total={total_stats['sent_total']:,} | "
            f"skipped_exist={total_stats['sent_skipped_exist']:,} | "
            f"skipped_oob={total_stats['sent_skipped_oob']:,} | "
            f"window_refs={total_stats['win_refs']:,} | "
            f"final_rows={len(updated_all):,} -> {out_path.as_posix()}"
        )

    logger.info("Stage 3.5 done. New manifests include 'meg_sentence_full_path'.")


if __name__ == "__main__":
    main()
