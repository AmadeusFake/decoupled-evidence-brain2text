#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_eeg_to_npy.py — Brennan EEG preprocessing (Stage-3 compatible)

This script preprocesses Brennan EEG recordings into fixed-length NumPy windows,
aligned with the Stage-3 MEG pipeline structure.

Pipeline overview:
- Load Brennan .mat files (raw_path) into MNE Raw (EEG + EOG + AUD), then keep EEG only
- Resample signals to target_sfreq (default: 120 Hz)
- Apply per-window baseline subtraction
- Fit a per-recording RobustScaler (Q25 / Q75) and clamp to ±std_clamp
- Save outputs:
    * EEG window: out_eeg_dir/{window_id}_win.npy              shape = [C, T]
    * 2D sensor coordinates: out_eeg_dir/{recording}_sensor_coordinates.npy
    * Robust center / scale: _robust_center.npy / _robust_scale.npy

Design notes:
- Parameterization and logging style are consistent with the Stage-3 MEG pipeline
- Supports resume, verification, and fast-reuse of existing outputs
"""

import argparse, json, logging, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import mne
from scipy.io import loadmat

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("stage3_eeg_robust")

# -------------------- Manifest utilities -------------------- #

def load_split(manifest_dir: Path, split: str) -> List[Dict]:
    """Load a JSONL manifest split."""
    p = manifest_dir / f"{split}.jsonl"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def save_manifest(rows: List[Dict], out_path: Path):
    """Write rows to a JSONL manifest file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -------------------- Atomic I/O helpers -------------------- #

def atomic_save_npy(path: Path, arr: np.ndarray):
    """Atomically save a NumPy array to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def verify_eeg_npy(path: Path, expected_T: Optional[int]) -> bool:
    """Verify that an EEG window file exists and has the expected shape."""
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

# -------------------- Brennan RAW loading -------------------- #

def _read_brennan_raw(mat_path: Path) -> mne.io.BaseRaw:
    """
    Load a Brennan Sxx.mat file into an MNE Raw object.

    This function follows the original pipeline logic and guarantees
    a total of 62 channels by appending an auxiliary AUD channel if needed.
    """
    mat = loadmat(
        mat_path,
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True,
    )
    rawm = mat["raw"]
    sfreq = float(rawm["hdr"]["Fs"])
    labels = list(rawm["hdr"]["label"])

    # Append auxiliary AUD channel if missing
    if len(labels) == 61:
        labels += ["AUD"]

    ch_types = ["eeg"] * 60 + ["eog", "misc"]
    info = mne.create_info(labels, sfreq, ch_types, verbose=False)

    data = rawm["trial"]
    if data.shape[0] == 61:
        data = np.vstack([data, np.zeros_like(data[0])])

    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)

    # Attempt to set a standard montage; fallback handled later
    try:
        raw.set_montage(mne.channels.make_standard_montage("easycap-M10"))
    except Exception:
        pass

    return raw

# -------------------- Sensor coordinates -------------------- #

def extract_coords_2d(raw: mne.io.BaseRaw, picks: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract normalized 2D EEG sensor coordinates.

    Priority:
    1) Use MNE layout if available
    2) Fallback to channel loc[:2]
    """
    try:
        layout = mne.channels.find_layout(raw.info, ch_type="eeg")
        if layout is not None and getattr(layout, "pos", None) is not None:
            pos2d = layout.pos[:len(picks), :2]
            mn = pos2d.min(axis=0)
            mx = pos2d.max(axis=0)
            span = np.maximum(mx - mn, 1e-6)
            xy01 = (pos2d - mn) / span
            return np.concatenate(
                [xy01, np.zeros((xy01.shape[0], 1))], axis=1
            ).astype(np.float32)
    except Exception:
        pass

    xs, ys = [], []
    for pi in picks:
        ch = raw.info["chs"][int(pi)]
        loc = ch.get("loc", None)
        if loc is None or len(loc) < 2:
            return None
        xs.append(loc[0])
        ys.append(loc[1])

    xy = np.stack([xs, ys], axis=1)
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    xy01 = (xy - mn) / span
    return np.concatenate(
        [xy01, np.zeros((xy01.shape[0], 1))], axis=1
    ).astype(np.float32)

# -------------------- Window helpers -------------------- #

def expected_T(row: Dict, sfreq: float) -> Optional[int]:
    """Compute expected temporal length for a window."""
    dur = row.get("local_window_duration_s", None)
    if dur is None and "local_window_onset_in_fif_s" in row and "local_window_offset_in_fif_s" in row:
        dur = float(row["local_window_offset_in_fif_s"]) - float(row["local_window_onset_in_fif_s"])
    if dur is None:
        return None
    return int(round(float(dur) * sfreq))

# -------------------- Robust scaling -------------------- #

def fit_robust(
    wins: List[np.ndarray],
    fit_max: int,
    seed: int,
    ql: float = 0.25,
    qh: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a per-channel robust center and scale."""
    rng = np.random.default_rng(seed)
    sel = wins if fit_max <= 0 or len(wins) <= fit_max else [
        wins[i] for i in rng.choice(len(wins), size=fit_max, replace=False)
    ]
    cat = np.concatenate(sel, axis=1).astype(np.float32)
    q25 = np.quantile(cat, ql, axis=1)
    q75 = np.quantile(cat, qh, axis=1)
    center = (q25 + q75) * 0.5
    scale = (q75 - q25) * 0.5
    scale[scale == 0.0] = 1.0
    return center.astype(np.float32), scale.astype(np.float32)

def apply_robust(
    seg: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    std_clamp: float,
) -> np.ndarray:
    """Apply robust normalization and clamp."""
    y = (seg - center[:, None]) / scale[:, None]
    np.clip(y, -std_clamp, std_clamp, out=y)
    return y.astype(np.float32)

# -------------------- Path resolution -------------------- #

def resolve_raw_path(row: Dict) -> Optional[Path]:
    """Resolve the raw EEG file path from a manifest row."""
    for k in ("raw_path", "original_fif_path", "original_meg_path", "fif_path"):
        p = row.get(k, "")
        if p:
            pp = Path(p)
            if pp.exists():
                return pp
    return None

# -------------------- Per-recording processing -------------------- #

def process_one_recording(
    rec_path: Path,
    rows_all: List[Dict],
    out_dir: Path,
    target_sfreq: float,
    baseline_end_s: float,
    fit_max_windows: int,
    std_clamp: float,
    seed: int,
    resume: bool,
    verify_existing: bool,
    recompute_existing: bool,
):
    """
    Process all windows belonging to a single recording.
    """
    stats = dict(
        total=0,
        saved=0,
        skipped_oob=0,
        skipped_no_coords=0,
        skipped_already=0,
        reused=0,
    )
    updated: List[Dict] = []

    rec_stem = rec_path.stem
    center_p = out_dir / f"{rec_stem}_robust_center.npy"
    scale_p = out_dir / f"{rec_stem}_robust_scale.npy"
    coord_p = out_dir / f"{rec_stem}_sensor_coordinates.npy"

    # FAST-RESUME: reuse everything if coordinates and all windows exist
    if resume and not recompute_existing and coord_p.exists():
        ok = True
        tmp = []
        for s in rows_all:
            outp = out_dir / f"{s['window_id']}_win.npy"
            expT = expected_T(s, target_sfreq)
            if not verify_eeg_npy(outp, expT):
                ok = False
                break
            s2 = dict(s)
            s2["meg_win_path"] = outp.as_posix()
            s2["sensor_coordinates_path"] = coord_p.as_posix()
            tmp.append(s2)
        if ok:
            updated.extend(tmp)
            stats["reused"] += len(tmp)
            return updated, stats

    # Load RAW EEG
    raw = _read_brennan_raw(rec_path)
    raw.pick(picks="eeg", exclude="bads")
    raw.resample(target_sfreq, npad="auto", verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if picks.size == 0:
        logger.warning(f"No EEG channels after pick: {rec_path}")
        return [], stats

    # Sensor coordinates
    if coord_p.exists():
        coords = np.load(coord_p.as_posix())
    else:
        coords = extract_coords_2d(raw, picks)
        if coords is None:
            stats["skipped_no_coords"] += len(rows_all)
            logger.warning(f"Cannot extract EEG sensor coordinates for {rec_path}, skip.")
            return [], stats
        atomic_save_npy(coord_p, coords)

    data = raw.get_data(picks=picks)
    _, N = data.shape
    T_bl = int(round(baseline_end_s * target_sfreq))

    # Pre-reuse individual windows
    rows_to_process = []
    if resume and not recompute_existing and verify_existing:
        for s in rows_all:
            outp = out_dir / f"{s['window_id']}_win.npy"
            expT = expected_T(s, target_sfreq)
            if verify_eeg_npy(outp, expT):
                s2 = dict(s)
                s2["meg_win_path"] = outp.as_posix()
                s2["sensor_coordinates_path"] = coord_p.as_posix()
                updated.append(s2)
                stats["reused"] += 1
            else:
                rows_to_process.append(s)
    else:
        rows_to_process = list(rows_all)

    # Fit robust parameters per recording
    if center_p.exists() and scale_p.exists():
        center = np.load(center_p.as_posix())
        scale = np.load(scale_p.as_posix())
    else:
        fit_wins = []
        for s in rows_all:
            onset = float(s["local_window_onset_in_fif_s"])
            dur = float(
                s.get(
                    "local_window_duration_s",
                    s["local_window_offset_in_fif_s"] - s["local_window_onset_in_fif_s"],
                )
            )
            start = int(round(onset * target_sfreq))
            length = int(round(dur * target_sfreq))
            end = start + length
            if start < 0 or end > N or length <= 0:
                stats["skipped_oob"] += 1
                continue
            seg = data[:, start:end].astype(np.float32, copy=True)
            if T_bl > 0:
                b_end = min(T_bl, seg.shape[1])
                seg -= seg[:, :b_end].mean(axis=1, keepdims=True).astype(np.float32)
            fit_wins.append(seg)

        if not fit_wins:
            return updated, stats

        center, scale = fit_robust(fit_wins, fit_max_windows, seed)
        atomic_save_npy(center_p, center)
        atomic_save_npy(scale_p, scale)

    # Process each window
    for s in rows_to_process:
        stats["total"] += 1
        onset = float(s["local_window_onset_in_fif_s"])
        dur = float(
            s.get(
                "local_window_duration_s",
                s["local_window_offset_in_fif_s"] - s["local_window_onset_in_fif_s"],
            )
        )
        start = int(round(onset * target_sfreq))
        length = int(round(dur * target_sfreq))
        end = start + length
        outp = out_dir / f"{s['window_id']}_win.npy"

        if not recompute_existing and verify_eeg_npy(outp, length):
            s2 = dict(s)
            s2["meg_win_path"] = outp.as_posix()
            s2["sensor_coordinates_path"] = coord_p.as_posix()
            updated.append(s2)
            stats["skipped_already"] += 1
            continue

        if start < 0 or end > N or length <= 0:
            stats["skipped_oob"] += 1
            continue

        seg = data[:, start:end].astype(np.float32, copy=True)
        if T_bl > 0:
            b_end = min(T_bl, seg.shape[1])
            seg -= seg[:, :b_end].mean(axis=1, keepdims=True).astype(np.float32)

        seg_n = apply_robust(seg, center, scale, std_clamp)
        atomic_save_npy(outp, seg_n)

        s2 = dict(s)
        s2["meg_win_path"] = outp.as_posix()
        s2["sensor_coordinates_path"] = coord_p.as_posix()
        updated.append(s2)
        stats["saved"] += 1

    return updated, stats

# -------------------- Main entry -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_manifest_dir", required=True, type=str)
    ap.add_argument("--output_eeg_dir", required=True, type=str)
    ap.add_argument("--output_manifest_dir", required=True, type=str)
    ap.add_argument("--target_sfreq", type=float, default=120.0)
    ap.add_argument("--baseline_end_s", type=float, default=0.5)
    ap.add_argument("--std_clamp", type=float, default=20.0)
    ap.add_argument("--fit_max_windows_per_recording", type=int, default=200)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--verify_existing", action="store_true", default=True)
    ap.add_argument("--recompute_existing", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    in_dir = Path(args.input_manifest_dir)
    out_eeg_dir = Path(args.output_eeg_dir)
    out_eeg_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_manifest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        in_path = in_dir / f"{split}.jsonl"
        if not in_path.exists():
            logger.info(f"[{split}] manifest missing, skip.")
            continue

        rows = load_split(in_dir, split)
        logger.info(f"[{split}] loaded {len(rows):,} rows from {in_path.name}")

        # Group windows by recording (raw_path)
        groups: Dict[Path, List[Dict]] = defaultdict(list)
        for r in rows:
            rp = resolve_raw_path(r)
            if rp is None:
                logger.warning(f"Missing raw_path in row: {r.get('window_id','')}")
                continue
            groups[rp].append(r)

        logger.info(f"[{split}] grouped into {len(groups):,} recordings")

        updated_all: List[Dict] = []
        total = dict(
            total=0,
            saved=0,
            skipped_oob=0,
            skipped_no_coords=0,
            skipped_already=0,
            reused=0,
        )

        for i, (rec_path, rec_rows) in enumerate(groups.items(), start=1):
            logger.info(
                f"[{split}] ({i}/{len(groups)}) {rec_path.name} with {len(rec_rows)} windows"
            )
            upd, st = process_one_recording(
                rec_path,
                rec_rows,
                out_eeg_dir,
                args.target_sfreq,
                args.baseline_end_s,
                args.fit_max_windows_per_recording,
                args.std_clamp,
                args.seed + i,
                args.resume,
                args.verify_existing,
                args.recompute_existing,
            )
            updated_all.extend(upd)
            for k in total:
                total[k] += st.get(k, 0)

        updated_all.sort(key=lambda x: x.get("window_id", ""))
        out_path = out_dir / f"{split}.jsonl"
        save_manifest(updated_all, out_path)

        logger.info(
            f"[{split}] saved={total['saved']:,} | reused={total['reused']:,} | "
            f"skipped_already={total['skipped_already']:,} | "
            f"skipped_oob={total['skipped_oob']:,} | "
            f"skipped_no_coords={total['skipped_no_coords']:,} | "
            f"total_seen={total['total']:,}"
        )
        logger.info(f"[{split}] wrote {len(updated_all):,} rows -> {out_path.as_posix()}")

    logger.info("Stage-3 EEG preprocessing (RobustScaler + FAST-RESUME) finished.")

if __name__ == "__main__":
    main()
