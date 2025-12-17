#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 3 (MOUS | paper-aligned + resumable) — RobustScaler + unified 273 head-MEG channels

Key design:
- Keep head MEG only: mne.pick_types(meg=True, ref_meg=False)
- Compute the intersection of head-MEG channels shared by all recordings
  (typically 273), and enforce a fixed channel order
- Fit RobustScaler (Q25/Q75) per recording; normalize and clamp to ±20
- Sensor coordinates must be extractable (2D); otherwise skip the recording
- FAST RESUME: if all windows and coordinates exist with valid shapes,
  skip raw loading and reuse results

Outputs:
- out_meg_dir/{window_id}_win.npy                  [C=273, T] float32
- out_meg_dir/common_head_meg_channels.json        unified channel list
- out_meg_dir/{recording_stem}_sensor_coordinates.npy  [273, 3]
- out_meg_dir/{recording_stem}_robust_center.npy
- out_meg_dir/{recording_stem}_robust_scale.npy
- out_manifest_dir/{split}.jsonl (successful samples only)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import mne

# Silence most MNE logs; keep only errors
mne.set_log_level("ERROR")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("stage3_mous_robust_273ch")

# --------------------------- I/O utilities --------------------------- #

def load_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def atomic_save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_meg_npy(
    path: Path,
    expected_T: Optional[int],
    expected_C: Optional[int],
) -> bool:
    try:
        if not path.exists():
            return False
        arr = np.load(path.as_posix(), mmap_mode="r")
        if arr.ndim != 2:
            return False
        if expected_C is not None and arr.shape[0] != expected_C:
            return False
        if expected_T is not None and arr.shape[1] != expected_T:
            return False
        return True
    except Exception:
        return False

# --------------------------- raw reading --------------------------- #

def read_raw(path: Path) -> mne.io.BaseRaw:
    suffix = path.suffix.lower()
    if suffix == ".con":
        return mne.io.read_raw_kit(path, preload=True, verbose="ERROR")
    if suffix == ".ds" or (path.is_dir() and path.name.endswith(".ds")):
        return mne.io.read_raw_ctf(path, preload=True, verbose="ERROR")
    return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")


def resolve_rec_path(row: Dict) -> Optional[Path]:
    for k in ["original_fif_path", "original_meg_path", "fif_path", "raw_path"]:
        if k in row and row[k]:
            p = Path(row[k])
            if p.exists():
                return p
    return None


def expected_T_for_row(row: Dict, sfreq: float) -> Optional[int]:
    dur = None
    if "local_window_duration_s" in row:
        dur = float(row["local_window_duration_s"])
    elif "local_window_offset_in_fif_s" in row and "local_window_onset_in_fif_s" in row:
        dur = float(row["local_window_offset_in_fif_s"]) - float(row["local_window_onset_in_fif_s"])
    elif "local_window_onset_in_fif_s" in row and "local_window_duration_in_fif_s" in row:
        dur = float(row["local_window_duration_in_fif_s"])
    if dur is None:
        return None
    return int(round(dur * sfreq))

# --------------------------- channel unification (273ch) --------------------------- #

def get_head_meg_ch_names_fast(ds_path: Path) -> List[str]:
    """
    Load info only (preload=False) to obtain channel names.
    Head MEG = pick_types(meg=True, ref_meg=False).
    """
    if ds_path.is_dir() and ds_path.name.endswith(".ds"):
        raw = mne.io.read_raw_ctf(ds_path, preload=False, verbose="ERROR")
    else:
        raw = mne.io.read_raw_fif(ds_path, preload=False, verbose="ERROR")

    info = raw.info
    head_picks = mne.pick_types(
        info,
        meg=True,
        ref_meg=False,
        eeg=False,
        stim=False,
        eog=False,
        misc=False,
        exclude="bads",
    )
    return [info["ch_names"][i] for i in head_picks]


def compute_common_head_meg_channels(
    all_rec_paths: List[Path],
    save_to: Path,
) -> List[str]:
    """
    Compute the intersection of head-MEG channels shared by all recordings.
    The output order follows the channel order of the first recording.
    """
    assert all_rec_paths, "No recordings provided to compute common channels."

    first_names = get_head_meg_ch_names_fast(all_rec_paths[0])
    common = set(first_names)

    for rp in all_rec_paths[1:]:
        try:
            names = set(get_head_meg_ch_names_fast(rp))
            common &= names
        except Exception as e:
            logger.warning(f"[COMMON-CH] Skip {rp} due to read error: {e}")

    common_ordered = [n for n in first_names if n in common]
    logger.info(f"[COMMON-CH] common head-MEG channels = {len(common_ordered)}")

    save_to.parent.mkdir(parents=True, exist_ok=True)
    save_to.write_text(json.dumps(common_ordered, indent=2), encoding="utf-8")
    logger.info(f"[COMMON-CH] written to {save_to.as_posix()}")
    return common_ordered


def load_or_compute_common_channels(
    input_manifest_dir: Path,
    out_meg_dir: Path,
) -> List[str]:
    """
    Load common_head_meg_channels.json if it exists;
    otherwise compute it from all train/valid/test recordings.
    """
    ch_json = out_meg_dir / "common_head_meg_channels.json"
    if ch_json.exists():
        names = json.loads(ch_json.read_text(encoding="utf-8"))
        logger.info(f"[COMMON-CH] loaded {len(names)} channels from disk")
        return names

    rec_paths = []
    for split in ("train", "valid", "test"):
        rows = load_jsonl(input_manifest_dir / f"{split}.jsonl")
        for r in rows:
            rp = resolve_rec_path(r)
            if rp is not None:
                rec_paths.append(rp)

    seen = set()
    uniq = []
    for p in rec_paths:
        ps = p.as_posix()
        if ps not in seen:
            seen.add(ps)
            uniq.append(p)

    logger.info(f"[COMMON-CH] computing intersection from {len(uniq)} recordings")
    return compute_common_head_meg_channels(uniq, ch_json)

# --------------------------- sensor coordinates --------------------------- #

def extract_coords_from_info(raw: mne.io.BaseRaw) -> Optional[np.ndarray]:
    """
    Assume raw has already been picked to the unified channel set.
    Extract 2D sensor coordinates from info['chs'][i]['loc'],
    normalize to [0, 1], and return [C, 3] with z = 0.
    """
    xs, ys = [], []
    for ch in raw.info["chs"]:
        loc = ch.get("loc", None)
        if loc is None or len(loc) < 2:
            return None
        xs.append(loc[0])
        ys.append(loc[1])

    xy = np.stack([np.array(xs), np.array(ys)], axis=1)
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    xy01 = (xy - mn) / span

    return np.concatenate(
        [xy01, np.zeros((xy01.shape[0], 1))],
        axis=1,
    ).astype(np.float32)


def extract_coords_with_layout(raw: mne.io.BaseRaw) -> Optional[np.ndarray]:
    try:
        layout = mne.channels.find_layout(raw.info, ch_type="meg")
    except Exception:
        return None
    if layout is None or layout.pos is None:
        return None

    pos2d = layout.pos[:, :2]
    if pos2d.shape[0] != len(raw.ch_names):
        return None

    mn = pos2d.min(axis=0)
    mx = pos2d.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    xy01 = (pos2d - mn) / span

    return np.concatenate(
        [xy01, np.zeros((xy01.shape[0], 1))],
        axis=1,
    ).astype(np.float32)

# --------------------------- RobustScaler --------------------------- #

def fit_robust_from_windows(
    wins: List[np.ndarray],
    fit_max_windows: int,
    seed: int,
    q_low: float = 0.25,
    q_high: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    if not wins:
        raise RuntimeError("No windows provided for RobustScaler fitting.")

    rng = np.random.default_rng(seed)
    sel = wins
    if fit_max_windows > 0 and len(wins) > fit_max_windows:
        idx = rng.choice(len(wins), size=fit_max_windows, replace=False)
        sel = [wins[i] for i in idx]

    cat = np.concatenate(sel, axis=1).astype(np.float32, copy=False)
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
    y = (seg - center[:, None]) / scale[:, None]
    np.clip(y, -std_clamp, std_clamp, out=y)
    return y.astype(np.float32)

# --------------------------- FAST RESUME --------------------------- #

def fast_resume_check(
    rec_rows: List[Dict],
    out_meg_dir: Path,
    target_sfreq: float,
    coord_path: Path,
    expected_C: int,
) -> Tuple[bool, List[Dict], Dict[str, int]]:

    stats = dict(
        total=0,
        saved=0,
        skipped_oob=0,
        skipped_no_coords=0,
        skipped_already=0,
        reused=0,
    )

    if not coord_path.exists():
        return False, [], stats

    updated = []
    for s in rec_rows:
        outp = out_meg_dir / f"{s['window_id']}_win.npy"
        expT = expected_T_for_row(s, target_sfreq)
        if not verify_meg_npy(outp, expT, expected_C):
            return False, [], stats

        s2 = dict(s)
        s2["meg_win_path"] = outp.as_posix()
        s2["sensor_coordinates_path"] = coord_path.as_posix()
        updated.append(s2)
        stats["reused"] += 1

    return True, updated, stats

# --------------------------- per-recording processing --------------------------- #

def process_one_recording(
    rec_path: Path,
    rec_rows: List[Dict],
    out_meg_dir: Path,
    target_sfreq: float,
    baseline_end_s: float,
    fit_max_windows: int,
    std_clamp: float,
    seed: int,
    resume: bool,
    verify_existing: bool,
    recompute_existing: bool,
    common_ch_names: List[str],
) -> Tuple[List[Dict], Dict[str, int]]:

    expected_C = len(common_ch_names)
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
    center_path = out_meg_dir / f"{rec_stem}_robust_center.npy"
    scale_path = out_meg_dir / f"{rec_stem}_robust_scale.npy"
    coord_path = out_meg_dir / f"{rec_stem}_sensor_coordinates.npy"

    # FAST RESUME
    if resume and not recompute_existing:
        ok, reused_rows, st = fast_resume_check(
            rec_rows, out_meg_dir, target_sfreq, coord_path, expected_C
        )
        if ok:
            logger.info(f"[FAST-RESUME] {rec_path.name}: reused {len(reused_rows)} windows")
            updated.extend(reused_rows)
            for k in stats:
                stats[k] += st.get(k, 0)
            return updated, stats

    try:
        logger.info(f"[REC] start {rec_path.name}, n_windows={len(rec_rows)}")
        raw = read_raw(rec_path)

        # Keep head MEG only, then enforce unified channel order
        head_picks = mne.pick_types(
            raw.info,
            meg=True,
            ref_meg=False,
            eeg=False,
            stim=False,
            eog=False,
            misc=False,
            exclude="bads",
        )
        head_names = [raw.info["ch_names"][i] for i in head_picks]

        if not all(n in head_names for n in common_ch_names):
            missing = [n for n in common_ch_names if n not in head_names]
            logger.warning(
                f"{rec_path.name}: missing {len(missing)} common channels; skip. "
                f"Example: {missing[:5]}"
            )
            return [], stats

        raw.pick_channels(common_ch_names, ordered=True)
        raw.resample(target_sfreq, npad="auto", verbose="ERROR")

        if len(raw.ch_names) != expected_C:
            logger.warning(
                f"{rec_path.name}: channel count mismatch after pick "
                f"(C={len(raw.ch_names)} != {expected_C}); skip"
            )
            return [], stats

        # Sensor coordinates
        if coord_path.exists():
            coords = np.load(coord_path.as_posix())
            if coords.shape[0] != expected_C:
                coord_path.unlink(missing_ok=True)
                coords = None
        else:
            coords = None

        if coords is None:
            coords = extract_coords_from_info(raw)
            if coords is None:
                coords = extract_coords_with_layout(raw)
            if coords is None:
                stats["skipped_no_coords"] += len(rec_rows)
                logger.warning(
                    f"Cannot extract sensor coordinates for {rec_path.name}; skip recording"
                )
                return [], stats
            atomic_save_npy(coord_path, coords)

        data = raw.get_data()  # [C, N]
        C, N = data.shape
        T_bl = int(round(baseline_end_s * target_sfreq))

        # Reuse existing windows if allowed
        rows_to_process = []
        if resume and not recompute_existing and verify_existing:
            for s in rec_rows:
                outp = out_meg_dir / f"{s['window_id']}_win.npy"
                expT = expected_T_for_row(s, target_sfreq)
                if verify_meg_npy(outp, expT, expected_C):
                    s2 = dict(s)
                    s2["meg_win_path"] = outp.as_posix()
                    s2["sensor_coordinates_path"] = coord_path.as_posix()
                    updated.append(s2)
                    stats["reused"] += 1
                else:
                    rows_to_process.append(s)
        else:
            rows_to_process = list(rec_rows)

        # Load or fit robust parameters
        if center_path.exists() and scale_path.exists():
            center = np.load(center_path.as_posix())
            scale = np.load(scale_path.as_posix())
            logger.info(f"[REC] {rec_path.name}: loaded existing robust parameters")
        else:
            logger.info(f"[REC] {rec_path.name}: fitting robust parameters")
            win_for_fit: List[np.ndarray] = []

            for s in rec_rows:
                stats["total"] += 1
                onset = float(s["local_window_onset_in_fif_s"])
                dur = float(
                    s.get(
                        "local_window_duration_s",
                        float(s["local_window_offset_in_fif_s"])
                        - float(s["local_window_onset_in_fif_s"]),
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
                    seg -= seg[:, :b_end].mean(axis=1, keepdims=True)

                win_for_fit.append(seg)

            if not win_for_fit:
                logger.warning(f"{rec_path.name}: no valid windows for robust fit; skip")
                return updated, stats

            center, scale = fit_robust_from_windows(
                win_for_fit, fit_max_windows, seed
            )
            atomic_save_npy(center_path, center)
            atomic_save_npy(scale_path, scale)
            logger.info(f"[REC] {rec_path.name}: robust parameters saved")

        # Process windows
        for idx, s in enumerate(rows_to_process, start=1):
            stats["total"] += 1
            onset = float(s["local_window_onset_in_fif_s"])
            dur = float(
                s.get(
                    "local_window_duration_s",
                    float(s["local_window_offset_in_fif_s"])
                    - float(s["local_window_onset_in_fif_s"]),
                )
            )
            start = int(round(onset * target_sfreq))
            length = int(round(dur * target_sfreq))
            end = start + length
            outp = out_meg_dir / f"{s['window_id']}_win.npy"

            if not recompute_existing and verify_meg_npy(outp, length, expected_C):
                s2 = dict(s)
                s2["meg_win_path"] = outp.as_posix()
                s2["sensor_coordinates_path"] = coord_path.as_posix()
                updated.append(s2)
                stats["skipped_already"] += 1
                continue

            if start < 0 or end > N or length <= 0:
                stats["skipped_oob"] += 1
                continue

            seg = data[:, start:end].astype(np.float32, copy=True)
            if T_bl > 0:
                b_end = min(T_bl, seg.shape[1])
                seg -= seg[:, :b_end].mean(axis=1, keepdims=True)

            seg_n = apply_robust_and_clamp(seg, center, scale, std_clamp)
            atomic_save_npy(outp, seg_n)

            s2 = dict(s)
            s2["meg_win_path"] = outp.as_posix()
            s2["sensor_coordinates_path"] = coord_path.as_posix()
            updated.append(s2)
            stats["saved"] += 1

            if idx % 50 == 0 or idx == len(rows_to_process):
                logger.info(
                    f"[REC] {rec_path.name}: processed {idx}/{len(rows_to_process)} "
                    f"(saved={stats['saved']}, reused={stats['reused']})"
                )

        logger.info(
            f"[REC] done {rec_path.name}: saved={stats['saved']}, "
            f"reused={stats['reused']}, skipped_oob={stats['skipped_oob']}, "
            f"skipped_no_coords={stats['skipped_no_coords']}"
        )

    except Exception as e:
        logger.error(f"Recording failed: {rec_path} | {e}", exc_info=True)

    return updated, stats

# --------------------------- main --------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_manifest_dir", required=True, type=str)
    ap.add_argument("--output_meg_dir", required=True, type=str)
    ap.add_argument("--output_manifest_dir", required=True, type=str)
    ap.add_argument("--target_sfreq", type=float, default=120.0)
    ap.add_argument("--baseline_end_s", type=float, default=0.5)
    ap.add_argument("--std_clamp", type=float, default=20.0)
    ap.add_argument("--fit_max_windows_per_recording", type=int, default=200)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--verify_existing", action="store_true", default=True)
    ap.add_argument("--recompute_existing", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of recordings processed in parallel (processes)"
    )
    args = ap.parse_args()

    in_dir = Path(args.input_manifest_dir)
    out_meg_dir = Path(args.output_meg_dir)
    out_meg_dir.mkdir(parents=True, exist_ok=True)
    out_manifest_dir = Path(args.output_manifest_dir)
    out_manifest_dir.mkdir(parents=True, exist_ok=True)

    # Unify channels first (typically C=273)
    common_ch_names = load_or_compute_common_channels(in_dir, out_meg_dir)
    logger.info(f"[COMMON-CH] using C={len(common_ch_names)} head-MEG channels")

    for split in ("train", "valid", "test"):
        in_path = in_dir / f"{split}.jsonl"
        if not in_path.exists():
            logger.info(f"[{split}] manifest missing; skip")
            continue

        rows = load_jsonl(in_path)
        logger.info(f"[{split}] loaded {len(rows):,} rows")

        groups: Dict[Path, List[Dict]] = defaultdict(list)
        for r in rows:
            rp = resolve_rec_path(r)
            if rp is None:
                logger.warning(f"Missing original MEG path for window_id={r.get('window_id','')}")
                continue
            groups[rp].append(r)

        logger.info(f"[{split}] grouped into {len(groups):,} recordings")

        tasks = []
        for i, (rec_path, rec_rows) in enumerate(groups.items(), start=1):
            tasks.append(
                (
                    rec_path,
                    rec_rows,
                    out_meg_dir,
                    args.target_sfreq,
                    args.baseline_end_s,
                    args.fit_max_windows_per_recording,
                    args.std_clamp,
                    args.seed + i,
                    args.resume,
                    args.verify_existing,
                    args.recompute_existing,
                    common_ch_names,
                )
            )

        updated_all: List[Dict] = []
        total_stats = dict(
            total=0,
            saved=0,
            skipped_oob=0,
            skipped_no_coords=0,
            skipped_already=0,
            reused=0,
        )

        if args.num_workers > 1 and len(tasks) > 1:
            logger.info(
                f"[{split}] multiprocessing enabled: "
                f"num_workers={args.num_workers}, n_recordings={len(tasks)}"
            )
            with mp.Pool(processes=args.num_workers) as pool:
                results = pool.starmap(process_one_recording, tasks)
        else:
            logger.info(f"[{split}] running sequentially over {len(tasks)} recordings")
            results = [process_one_recording(*t) for t in tasks]

        for upd, st in results:
            updated_all.extend(upd)
            for k in total_stats:
                total_stats[k] += st.get(k, 0)

        updated_all.sort(key=lambda x: x.get("window_id", ""))
        out_path = out_manifest_dir / f"{split}.jsonl"
        save_jsonl(updated_all, out_path)

        logger.info(
            f"[{split}] saved={total_stats['saved']:,} | reused={total_stats['reused']:,} | "
            f"skipped_already={total_stats['skipped_already']:,} | "
            f"skipped_oob={total_stats['skipped_oob']:,} | "
            f"skipped_no_coords={total_stats['skipped_no_coords']:,} | "
            f"total_seen={total_stats['total']:,}"
        )
        logger.info(f"[{split}] wrote {len(updated_all):,} rows -> {out_path.as_posix()}")

    logger.info("Stage-3 (MOUS RobustScaler + unified 273ch head-MEG) finished")


if __name__ == "__main__":
    main()
