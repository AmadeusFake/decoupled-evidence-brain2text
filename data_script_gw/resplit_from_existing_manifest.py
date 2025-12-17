#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
resplit_from_existing_manifest_rotate.py (v5)

Recording-level split with subject rotation coverage.

Goals (all enforced):
1) Recording isolation: a recording (subject_id + session_id) appears in only one split.
2) Subject coverage: every subject appears in TRAIN at least once (optional).
3) Content coverage: contents in TEST must appear in TRAIN (optional).
4) Target split ratios by number of recordings (≈0.7 / 0.1 / 0.2, tolerance ±1).
5) Rotation coverage:
   - Subjects with ≥2 recordings are eligible.
   - For K folds, fold f forces one recording per subject into TEST.
   - Across folds, all eligible subjects are evaluated at least once.

Design note:
This differs from zero-shot cross-subject setups (e.g., Défossez):
- Sentence/content overlap across splits is allowed.
- MEG recordings are strictly isolated.
- TEST content is constrained to be covered by TRAIN.
"""

import argparse, json, logging, math, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------- I/O ---------------- #

def read_jsonl(p: Path) -> List[dict]:
    out = []
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out

def write_jsonl(p: Path, rows: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------------- Keys ---------------- #

def recording_key_of(r: dict) -> str:
    """Recording identifier: subject_id + session_id."""
    sid = r.get("subject_id")
    ses = r.get("session_id")
    if sid is not None and ses is not None:
        return f"{sid}_{ses}"
    wid = str(r.get("window_id", ""))
    parts = wid.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    raise ValueError("Cannot derive recording key.")

def content_id_of(r: dict) -> str:
    """Content-level identifier (audio path + time)."""
    if r.get("content_id"):
        return r["content_id"]
    a = r["original_audio_path"]
    on = float(r["local_window_onset_in_audio_s"])
    off = float(r["local_window_offset_in_audio_s"])
    return f"{a}::{on:.3f}-{off:.3f}"

# ------------- Helpers ------------- #

def largest_remainder_targets(
    n_total: int,
    ratios: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """Allocate targets with largest-remainder rounding."""
    quotas = [ratios[0] * n_total, ratios[1] * n_total, ratios[2] * n_total]
    floors = [math.floor(q) for q in quotas]
    rem = n_total - sum(floors)
    fracs = sorted(
        ((i, quotas[i] - floors[i]) for i in range(3)),
        key=lambda x: x[1],
        reverse=True,
    )
    for i in range(rem):
        floors[fracs[i][0]] += 1
    return tuple(floors)  # train, valid, test

def initial_maps(rows: List[dict], seed: int):
    """Build recording/content/subject lookup tables."""
    rng = random.Random(seed)

    sent_to_recs: Dict[str, Set[str]] = defaultdict(set)
    rec_to_rows: Dict[str, List[dict]] = defaultdict(list)
    rec_to_contents: Dict[str, Set[str]] = defaultdict(set)
    rec_to_subject: Dict[str, str] = {}

    for r in rows:
        rk = recording_key_of(r)
        ck = content_id_of(r)
        sid = str(r.get("subject_id"))
        sent_to_recs[ck].add(rk)
        rec_to_rows[rk].append(r)
        rec_to_contents[rk].add(ck)
        rec_to_subject[rk] = sid

    all_recordings = list(rec_to_rows.keys())
    rng.shuffle(all_recordings)

    subj_to_recs: Dict[str, List[str]] = defaultdict(list)
    for rk, sid in rec_to_subject.items():
        subj_to_recs[sid].append(rk)

    return (
        all_recordings,
        sent_to_recs,
        rec_to_rows,
        rec_to_contents,
        rec_to_subject,
        subj_to_recs,
    )

def build_rotation_group(
    subj_to_recs: Dict[str, List[str]],
    num_folds: int,
    seed: int,
) -> List[List[str]]:
    """Build K rotation groups from subjects with ≥2 recordings."""
    rng = random.Random(seed)
    eligible = [s for s, rs in subj_to_recs.items() if len(rs) >= 2]
    eligible.sort()
    rng.shuffle(eligible)

    groups = [[] for _ in range(num_folds)]
    for i, s in enumerate(eligible):
        groups[i % num_folds].append(s)

    logging.info(
        f"[rotation] eligible_subjects={len(eligible)} groups={[len(g) for g in groups]}"
    )
    return groups

# ------------- Core splitting logic ------------- #

def split_with_targets(
    rows: List[dict],
    ratios: Tuple[float, float, float],
    seed: int,
    min_valid: int,
    min_test: int,
    require_subject_in_train: bool,
    enforce_test_covered_only: bool,
    rotation_fold_index: int | None,
    rotation_num_folds: int | None,
):
    """
    Perform recording-level splitting with optional subject rotation
    and content coverage constraints.
    """
    rng = random.Random(seed)
    (
        all_recordings,
        sent_to_recs,
        rec_to_rows,
        rec_to_contents,
        rec_to_subject,
        subj_to_recs,
    ) = initial_maps(rows, seed)

    n_total = len(all_recordings)

    # Rotation: subjects forced into TEST for this fold
    forced_test_subjects: Set[str] = set()
    pinned_test_recs: Set[str] = set()

    if rotation_num_folds is not None and rotation_fold_index is not None:
        groups = build_rotation_group(subj_to_recs, rotation_num_folds, seed)
        if 0 <= rotation_fold_index < len(groups):
            forced_test_subjects = set(groups[rotation_fold_index])
            logging.info(
                f"[rotation] fold={rotation_fold_index}/{rotation_num_folds} "
                f"forced_test_subjects={sorted(forced_test_subjects)}"
            )

    # Step 0: content coverage placement
    rec_split: Dict[str, str] = {}
    train, valid, test = set(), set(), set()

    cross_contents = [ck for ck, rs in sent_to_recs.items() if len(rs) >= 2]
    cross_contents.sort(key=lambda ck: len(sent_to_recs[ck]))

    for ck in cross_contents:
        recs = list(sent_to_recs[ck])
        rng.shuffle(recs)

        r_train = None
        for rk in recs:
            if rk in rec_split:
                continue
            if rec_to_subject[rk] not in forced_test_subjects:
                r_train = rk
                break
        if r_train is None:
            r_train = next((rk for rk in recs if rk not in rec_split), None)

        r_test = None
        for rk in recs:
            if rk in rec_split or rk == r_train:
                continue
            if rec_to_subject[rk] in forced_test_subjects:
                r_test = rk
                break
        if r_test is None:
            r_test = next(
                (rk for rk in recs if rk not in rec_split and rk != r_train), None
            )

        if r_train is not None:
            rec_split[r_train] = "train"
            train.add(r_train)
        if r_test is not None:
            rec_split[r_test] = "test"
            test.add(r_test)

    # Step 1: ensure each subject appears in TRAIN
    if require_subject_in_train:
        for sid, rs in subj_to_recs.items():
            if any(rk in train for rk in rs):
                continue
            r_move = next((rk for rk in rs if rk in test), None)
            src = "test" if r_move else None
            if r_move is None:
                r_move = rs[0]
                src = rec_split.get(r_move)
            if src == "test":
                test.remove(r_move)
            elif src == "valid":
                valid.remove(r_move)
            rec_split[r_move] = "train"
            train.add(r_move)
            logging.info(f"[subject-train] move {r_move} {src or 'unassigned'} -> train")

    # Step 2: rotation constraint (force one test recording per subject)
    def subject_has_test(sid: str) -> bool:
        return any(rk in test for rk in subj_to_recs[sid])

    for sid in sorted(forced_test_subjects):
        rs = subj_to_recs[sid]
        if len(rs) < 2 or subject_has_test(sid):
            continue

        cand = next((rk for rk in sorted(rs) if rk in train or rk not in rec_split), rs[0])
        other_train = next(
            (rk for rk in rs if rk != cand and (rk in train or rk not in rec_split)),
            None,
        )
        if other_train is None:
            logging.info(
                f"[subject-test-rotate] subject {sid}: cannot satisfy constraints."
            )
            continue

        src = rec_split.get(cand)
        if src == "train":
            train.remove(cand)
        elif src == "valid":
            valid.remove(cand)

        rec_split[cand] = "test"
        test.add(cand)
        pinned_test_recs.add(cand)
        logging.info(f"[subject-test-rotate] pin {cand} {src or 'unassigned'} -> test")

    # Step 3: target recording counts
    tgt_train, tgt_valid, tgt_test = largest_remainder_targets(n_total, ratios)

    auto_min_v = (
        min_valid
        if min_valid >= 0
        else max(1 if n_total >= 3 else 0, math.ceil(n_total * ratios[1]))
    )
    auto_min_t = (
        min_test
        if min_test >= 0
        else max(1 if n_total >= 2 else 0, math.ceil(n_total * ratios[2]))
    )

    tgt_valid = max(tgt_valid, auto_min_v)
    tgt_test = max(tgt_test, auto_min_t)
    if tgt_train + tgt_valid + tgt_test != n_total:
        tgt_train = max(0, n_total - tgt_valid - tgt_test)

    # Step 4: assign remaining recordings
    remaining = [rk for rk in all_recordings if rk not in rec_split]
    rng.shuffle(remaining)

    need = {
        "train": max(0, tgt_train - len(train)),
        "valid": max(0, tgt_valid - len(valid)),
        "test": max(0, tgt_test - len(test)),
    }

    def pick_need():
        return max(need.items(), key=lambda x: x[1])[0]

    for rk in remaining:
        side = pick_need()
        rec_split[rk] = side
        if side == "train":
            train.add(rk)
        elif side == "valid":
            valid.add(rk)
        else:
            test.add(rk)
        need[side] = max(0, need[side] - 1)

    # Step 5: adjust to targets (avoid pinned test recordings)
    def can_move_from(split_name: str, rk: str) -> bool:
        if split_name == "test" and rk in pinned_test_recs:
            return False
        if require_subject_in_train and split_name == "train":
            sid = rec_to_subject[rk]
            if not any(x in train for x in subj_to_recs[sid] if x != rk):
                return False
        return True

    def move(rk: str, src: str, dst: str):
        if src == "train":
            train.remove(rk)
        elif src == "valid":
            valid.remove(rk)
        else:
            test.remove(rk)
        if dst == "train":
            train.add(rk)
        elif dst == "valid":
            valid.add(rk)
        else:
            test.add(rk)
        rec_split[rk] = dst
        logging.info(f"[shrink] move {rk} {src} -> {dst}")

    while len(test) > tgt_test:
        movable = [rk for rk in list(test) if can_move_from("test", rk)]
        if not movable:
            break
        move(movable[0], "test", "train" if len(train) < tgt_train else "valid")

    while len(valid) > tgt_valid:
        rk = next(iter(valid))
        move(rk, "valid", "train" if len(train) < tgt_train else "test")

    while len(train) > tgt_train:
        movable = [rk for rk in list(train) if can_move_from("train", rk)]
        if not movable:
            break
        move(movable[0], "train", "test" if len(test) < tgt_test else "valid")

    # Output rows
    out_rows = {"train": [], "valid": [], "test": []}
    for rk, sp in rec_split.items():
        out_rows[sp].extend(rec_to_rows[rk])

    for sp in ("train", "valid", "test"):
        out_rows[sp].sort(key=lambda x: x.get("window_id", ""))

    # Optional: enforce test content ⊆ train content
    if enforce_test_covered_only:
        train_contents = {content_id_of(r) for r in out_rows["train"]}
        before = len(out_rows["test"])
        out_rows["test"] = [
            r for r in out_rows["test"] if content_id_of(r) in train_contents
        ]
        logging.info(
            f"[enforce_covered_test_only] test rows {before} -> {len(out_rows['test'])}"
        )

    # Reporting & checks
    def uniq_subjects(lst): return sorted({str(r.get("subject_id")) for r in lst})
    def uniq_recordings(lst): return sorted({recording_key_of(r) for r in lst})
    def uniq_contents(lst): return len({content_id_of(r) for r in lst})

    tr, va, te = out_rows["train"], out_rows["valid"], out_rows["test"]

    logging.info(
        f"[train] rows={len(tr):,} unique_subjects={len(uniq_subjects(tr))} "
        f"unique_recordings={len(uniq_recordings(tr))} unique_contents={uniq_contents(tr)}"
    )
    logging.info(
        f"[valid] rows={len(va):,} unique_subjects={len(uniq_subjects(va))} "
        f"unique_recordings={len(uniq_recordings(va))} unique_contents={uniq_contents(va)}"
    )
    logging.info(
        f"[test]  rows={len(te):,} unique_subjects={len(uniq_subjects(te))} "
        f"unique_recordings={len(uniq_recordings(te))} unique_contents={uniq_contents(te)}"
    )

    rec_sets = {k: set(uniq_recordings(out_rows[k])) for k in ("train", "valid", "test")}
    assert (
        rec_sets["train"].isdisjoint(rec_sets["valid"])
        and rec_sets["train"].isdisjoint(rec_sets["test"])
        and rec_sets["valid"].isdisjoint(rec_sets["test"])
    )
    logging.info("[OK] No recording overlap across splits.")

    train_contents = {content_id_of(r) for r in tr}
    test_contents = {content_id_of(r) for r in te}
    assert test_contents.issubset(train_contents)
    logging.info("[OK] contents(test) ⊆ contents(train).")

    if require_subject_in_train:
        train_subjects = set(uniq_subjects(tr))
        all_subjects = set(uniq_subjects(tr + va + te))
        assert all_subjects.issubset(train_subjects)
        logging.info("[OK] Every subject appears in TRAIN.")

    ntr, nva, nte = len(rec_sets["train"]), len(rec_sets["valid"]), len(rec_sets["test"])
    tgt = largest_remainder_targets(ntr + nva + nte, ratios)
    diff = (ntr - tgt[0], nva - tgt[1], nte - tgt[2])
    logging.info(
        f"[ratio] recordings target≈{tgt} got={(ntr, nva, nte)} diff={diff} tol=±1"
    )

    cross_set = [ck for ck, rs in sent_to_recs.items() if len(rs) >= 2]
    covered = sum(
        1 for ck in cross_set if ck in train_contents and ck in test_contents
    )
    logging.info(
        f"[coverage] contents with ≥2 recordings: {len(cross_set)} | "
        f"covered(train&test)={covered}"
    )

    return out_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_manifest_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split_ratios", default="0.7,0.1,0.2")
    ap.add_argument("--random_seed", type=int, default=42)
    ap.add_argument("--min_valid_recordings", type=int, default=-1)
    ap.add_argument("--min_test_recordings", type=int, default=-1)
    ap.add_argument("--require_subject_in_train", action="store_true", default=False)
    ap.add_argument("--enforce_covered_test_only", action="store_true", default=False)
    ap.add_argument("--rotation_num_folds", type=int, default=None)
    ap.add_argument("--rotation_fold_index", type=int, default=None)

    args = ap.parse_args()
    in_dir = Path(args.input_manifest_dir)
    out_dir = Path(args.output_dir)
    ratios = tuple(float(x) for x in args.split_ratios.split(","))

    # Merge and de-duplicate rows
    rows = []
    for sp in ("train", "valid", "test"):
        rows += read_jsonl(in_dir / f"{sp}.jsonl")

    uniq = {}
    for r in rows:
        wid = r.get("window_id") or (
            f"{r.get('subject_id','?')}_{r.get('session_id','?')}_"
            f"{r['original_audio_path']}@{r['local_window_onset_in_audio_s']}"
        )
        uniq[wid] = r
    rows = list(uniq.values())

    logging.info(f"Merged unique rows: {len(rows):,}")

    out_rows = split_with_targets(
        rows,
        ratios,
        args.random_seed,
        args.min_valid_recordings,
        args.min_test_recordings,
        require_subject_in_train=args.require_subject_in_train,
        enforce_test_covered_only=args.enforce_covered_test_only,
        rotation_fold_index=args.rotation_fold_index,
        rotation_num_folds=args.rotation_num_folds,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "valid", "test"):
        write_jsonl(out_dir / f"{sp}.jsonl", out_rows[sp])
        logging.info(f"Saved {len(out_rows[sp]):,} rows -> {out_dir / f'{sp}.jsonl'}")

if __name__ == "__main__":
    main()
