#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 2 (FAST + RESUMABLE, paper-aligned with layer flags)

- Model: default facebook/wav2vec2-large-xlsr-53 (configurable)
- Features:
    * Select layers via --w2v_layers:
        - "last4" (default)
        - "14-18"
        - "14,15,16,17,18"
    * Layer aggregation via --w2v_agg:
        - mean   (default, output dim = hidden_size = 1024)
        - concat (output dim = 1024 * num_layers)
- Time: linear interpolation to T=360 (120 Hz × 3 s), performed on GPU
- I/O: DataLoader parallel audio loading with prefetching;
        GPU inference overlaps with CPU I/O
- Resume support:
    * Verify existing .npy files with expected shape [D, 360]
    * Valid files are skipped automatically
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("stage2_layerflags")

SR = 16_000
TARGET_T = 360          # 120 Hz × 3 s
DEFAULT_WIN_SECONDS = 3.0

# -------------------- Utilities -------------------- #

def safe_name_from_id(s: str) -> str:
    """Sanitize string for filesystem-safe filenames."""
    return "".join([c if c.isalnum() or c in "-_." else "_" for c in s])


def content_id_for_sample(audio_path: Path, onset_s: float, offset_s: float) -> str:
    """Content-unique identifier for an audio window."""
    return f"{audio_path.stem}_{onset_s:.3f}_{offset_s:.3f}"


def load_split(manifest_dir: Path, split: str) -> List[Dict]:
    p = manifest_dir / f"{split}.jsonl"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_audio_index(roots: List[Path]) -> Dict[str, Path]:
    """
    Case-insensitive audio path index.
    Indexes full paths, relative paths, and basenames.
    """
    index: Dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            low = p.as_posix().lower()
            index[low] = p
            try:
                rel = p.relative_to(root).as_posix().lower()
                index[rel] = p
            except Exception:
                pass
            index[p.name.lower()] = p
    return index


def guess_audio_roots(samples: List[Dict]) -> List[Path]:
    """Heuristically infer audio root directories from manifests."""
    roots = set()
    for s in samples:
        a = s.get("original_audio_path", "")
        if not a:
            continue
        p = Path(a)
        cands = [
            p.parent,
            p.parent.parent / "stimuli" / "audio",
            p.parent.parent.parent / "stimuli" / "audio",
        ]
        for c in cands:
            if c.exists():
                roots.add(c.resolve())
    return sorted(list(roots))


def atomic_save_npy(path: Path, arr: np.ndarray):
    """
    Atomic .npy write:
    - Write to *.npy.tmp
    - fsync
    - atomic replace
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_npy_shape(path: Path, expect: Tuple[int, int]) -> bool:
    """Check that .npy exists and has the expected shape."""
    try:
        if not path.exists():
            return False
        arr = np.load(path.as_posix(), mmap_mode="r")
        return arr.shape == expect
    except Exception:
        return False


# -------------------- Layer selection -------------------- #

def parse_layers_count(spec: str) -> int:
    s = spec.strip().lower()
    m = re.match(r"last-?(\d+)$", s)
    if m:
        return int(m.group(1))
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return abs(int(b) - int(a)) + 1
    return len([t for t in s.split(",") if t.strip()])


def build_layer_indices(spec: str, hs_len: int) -> List[int]:
    """
    Parse layer spec into hidden_states indices (0-based).
    Supports:
      - last4 / last-5
      - 14-18
      - 14,15,16,17,18
    """
    s = spec.strip().lower()
    m = re.match(r"last-?(\d+)$", s)
    if m:
        k = int(m.group(1))
        if k <= 0 or k > hs_len:
            raise ValueError(f"Invalid last-k spec: {spec}")
        return list(range(hs_len - k, hs_len))

    if "-" in s and "," not in s:
        a, b = s.split("-")
        ia, ib = sorted([int(a), int(b)])
        if ib >= hs_len:
            raise IndexError(f"Layer index {ib} out of range.")
        return list(range(ia, ib + 1))

    out: List[int] = []
    for t in s.split(","):
        if not t.strip():
            continue
        it = int(t)
        if it < 0 or it >= hs_len:
            raise IndexError(f"Layer index {it} out of range.")
        out.append(it)

    if not out:
        raise ValueError("Empty layer spec.")
    return out


# -------------------- Dataset -------------------- #

@dataclass
class Row:
    original_audio_path: str
    onset_s: float
    offset_s: float
    window_id: str


class AudioWinDataset(Dataset):
    def __init__(self, rows: List[Dict], audio_index: Dict[str, Path]):
        self.rows = [
            Row(
                original_audio_path=r.get("original_audio_path", ""),
                onset_s=float(r["local_window_onset_in_audio_s"]),
                offset_s=float(r["local_window_offset_in_audio_s"]),
                window_id=r["window_id"],
            )
            for r in rows
        ]
        self.audio_index = audio_index

    def __len__(self):
        return len(self.rows)

    def _resolve_path(self, orig: str) -> Optional[Path]:
        p = Path(orig)
        for k in (p.as_posix().lower(), p.name.lower()):
            if k in self.audio_index:
                return self.audio_index[k]
        return p if p.exists() else None

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        rp = self._resolve_path(r.original_audio_path)

        desired = int(round((r.offset_s - r.onset_s) * SR))
        if desired <= 0:
            desired = int(round(DEFAULT_WIN_SECONDS * SR))
        start = int(round(r.onset_s * SR))

        meta = {
            "window_id": r.window_id,
            "original_audio_path": r.original_audio_path,
            "local_window_onset_in_audio_s": r.onset_s,
            "local_window_offset_in_audio_s": r.offset_s,
            "_missing": False,
        }

        if rp is None:
            return {"wave": torch.zeros(desired), "meta": {**meta, "_missing": True}}

        try:
            wav, sr = torchaudio.load(rp)
        except Exception as e:
            logger.error(f"torchaudio.load failed: {rp} ({e}); zero-filled.")
            return {"wave": torch.zeros(desired), "meta": {**meta, "_missing": True}}

        wav = wav.mean(dim=0) if wav.dim() == 2 else wav
        wav = wav.to(torch.float32)

        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)

        end = min(start + desired, wav.numel())
        start = max(start, 0)

        seg = torch.zeros(desired)
        if end > start:
            seg[: end - start] = wav[start:end]

        return {"wave": seg, "meta": meta}


def collate_batch(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    waves = torch.stack([s["wave"] for s in samples], dim=0)
    metas = [s["meta"] for s in samples]
    return {"waves": waves, "metas": metas}


# -------------------- Model forward -------------------- #

def load_w2v2(model_name: str, device: str) -> Wav2Vec2Model:
    model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    return model


@torch.inference_mode()
def wav2vec_forward(
    model: Wav2Vec2Model,
    waves: torch.Tensor,
    device: str,
    use_bf16: bool,
    layer_spec: str,
    agg: str,
) -> torch.Tensor:
    """
    Input:  waves [B, T] (16 kHz)
    Output: features [B, D, 360]
    """
    waves = waves.to(device, non_blocking=True)
    waves = (waves - waves.mean(dim=1, keepdim=True)) / waves.std(dim=1, keepdim=True).clamp_min(1e-6)

    def _forward(inp):
        out = model(inp, output_hidden_states=True)
        hs = out.hidden_states
        idx = build_layer_indices(layer_spec, len(hs))
        sel = [hs[i] for i in idx]
        x = torch.stack(sel).mean(0) if agg == "mean" else torch.cat(sel, dim=-1)
        x = x.transpose(1, 2)
        return F.interpolate(x, size=TARGET_T, mode="linear", align_corners=False).float()

    if use_bf16 and torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return _forward(waves)
    return _forward(waves)


# -------------------- Resume utilities -------------------- #

def load_done_set(
    out_manifest_dir: Path,
    split: str,
    verify_existing: bool,
    expected_shape: Tuple[int, int],
) -> Dict[str, str]:
    out_path = out_manifest_dir / f"{split}.jsonl"
    done: Dict[str, str] = {}
    if not out_path.exists():
        return done

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            wid = row.get("window_id")
            feat = row.get("audio_feature_path")
            if not wid or not feat:
                continue
            if not verify_existing or verify_npy_shape(Path(feat), expected_shape):
                done[wid] = feat

    logger.info(f"[resume] {split}: {len(done):,} completed samples found.")
    return done


# -------------------- Main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_manifest_dir", required=True)
    ap.add_argument("--output_feature_dir", required=True)
    ap.add_argument("--output_manifest_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    ap.add_argument("--save_dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--verify_existing", action="store_true", default=True)
    ap.add_argument("--recompute_existing", action="store_true", default=False)
    ap.add_argument("--w2v_model", default="facebook/wav2vec2-large-xlsr-53")
    ap.add_argument("--w2v_layers", default="last4")
    ap.add_argument("--w2v_agg", choices=["mean", "concat"], default="mean")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    in_dir = Path(args.input_manifest_dir)
    feat_dir = Path(args.output_feature_dir); feat_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_manifest_dir); out_dir.mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(in_dir, s) for s in ("train", "valid", "test")}
    all_samples = [r for s in splits.values() for r in s]

    audio_roots = guess_audio_roots(all_samples)
    audio_index = build_audio_index(audio_roots)

    model = load_w2v2(args.w2v_model, args.device)
    use_bf16 = (args.amp == "bf16")
    save_as_fp16 = (args.save_dtype == "float16")

    hidden_size = int(getattr(model.config, "hidden_size", 1024))
    n_sel_layers = parse_layers_count(args.w2v_layers)
    out_feat_dim = hidden_size if args.w2v_agg == "mean" else hidden_size * n_sel_layers
    expected_shape = (out_feat_dim, TARGET_T)

    logger.info(f"Feature dim = {out_feat_dim} (agg={args.w2v_agg}, layers={args.w2v_layers})")

    for split in ("train", "valid", "test"):
        rows = splits[split]
        if not rows:
            continue

        done_set = {}
        if args.resume and not args.recompute_existing:
            done_set = load_done_set(out_dir, split, args.verify_existing, expected_shape)

        to_process = [r for r in rows if r["window_id"] not in done_set]

        ds = AudioWinDataset(to_process, audio_index)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            collate_fn=collate_batch,
        )

        out_path = out_dir / f"{split}.jsonl"
        id2row = {r["window_id"]: r for r in rows}

        with open(out_path, "a", encoding="utf-8") as fout:
            for batch in tqdm(dl, desc=f"[{split}]"):
                feats = wav2vec_forward(
                    model,
                    batch["waves"],
                    args.device,
                    use_bf16,
                    args.w2v_layers,
                    args.w2v_agg,
                ).cpu()

                for i, m in enumerate(batch["metas"]):
                    if m.get("_missing"):
                        continue
                    rp = Path(m["original_audio_path"])
                    cid = content_id_for_sample(
                        rp,
                        m["local_window_onset_in_audio_s"],
                        m["local_window_offset_in_audio_s"],
                    )
                    outp = feat_dir / f"{safe_name_from_id(cid)}.npy"
                    arr = feats[i].numpy()
                    if save_as_fp16:
                        arr = arr.astype(np.float16)
                    atomic_save_npy(outp, arr)

                    r2 = dict(id2row[m["window_id"]])
                    r2["audio_feature_path"] = outp.as_posix()
                    fout.write(json.dumps(r2, ensure_ascii=False) + "\n")

    logger.info("Stage-2 finished.")


if __name__ == "__main__":
    main()
