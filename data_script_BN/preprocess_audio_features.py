#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-2 Audio Feature Preprocessing (Brennan, sentence-level)

Extract Wav2Vec2 features for each 3s audio window and interpolate to 360 frames.
Supports explicit audio root, resume, and shape verification.
"""

import argparse, json, logging, os, re
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
logger = logging.getLogger("stage2_sentence_only")

SR = 16_000
TARGET_T = 360
DEFAULT_WIN_SECONDS = 3.0

# -------------------- Utilities -------------------- #

def safe_name_from_id(s: str) -> str:
    """Make a string safe for filenames."""
    return "".join([c if c.isalnum() or c in "-_." else "_" for c in s])

def content_id_for_sample(audio_path: Path, onset_s: float, offset_s: float) -> str:
    """Content-level ID from audio path and window time."""
    return f"{audio_path.stem}_{onset_s:.3f}_{offset_s:.3f}"

def load_split(manifest_dir: Path, split: str) -> List[Dict]:
    """Load a JSONL split."""
    p = manifest_dir / f"{split}.jsonl"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def build_audio_index(root: Optional[Path]) -> Dict[str, Path]:
    """Index all .wav files under audio_root for fast lookup."""
    index: Dict[str, Path] = {}
    if root is None or not root.exists():
        return index
    for p in root.rglob("*.wav"):
        index[p.as_posix().lower()] = p
        index[p.name.lower()] = p
        try:
            index[p.relative_to(root).as_posix().lower()] = p
        except Exception:
            pass
    return index

def atomic_save_npy(path: Path, arr: np.ndarray):
    """Atomic npy save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def verify_npy_shape(path: Path, expect: Tuple[int, int]) -> bool:
    """Check npy exists and matches expected shape."""
    try:
        if not path.exists():
            return False
        arr = np.load(path.as_posix(), mmap_mode="r")
        return arr.shape == expect
    except Exception:
        return False

# -------------------- Layer selection -------------------- #

def parse_layers_count(spec: str) -> int:
    """Return number of layers specified."""
    s = spec.strip().lower()
    m = re.match(r"last-?(\d+)$", s)
    if m:
        return int(m.group(1))
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return abs(int(b) - int(a)) + 1
    return len([t for t in s.split(",") if t.strip()])

def build_layer_indices(spec: str, hs_len: int) -> List[int]:
    """Parse layer spec into explicit indices."""
    s = spec.strip().lower()
    m = re.match(r"last-?(\d+)$", s)
    if m:
        k = int(m.group(1))
        if k <= 0 or k > hs_len:
            raise ValueError(f"last{k} invalid for hs_len={hs_len}")
        return list(range(hs_len - k, hs_len))
    if "-" in s and "," not in s:
        a, b = s.split("-")
        ia, ib = sorted((int(a), int(b)))
        if ib >= hs_len:
            raise IndexError(f"Layer {ib} out of range")
        return list(range(ia, ib + 1))
    out = []
    for t in s.split(","):
        if not t.strip():
            continue
        it = int(t)
        if it < 0 or it >= hs_len:
            raise IndexError(f"Layer {it} out of range")
        out.append(it)
    if not out:
        raise ValueError("Empty layer spec")
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
                original_audio_path=s.get("original_audio_path", ""),
                onset_s=float(s["local_window_onset_in_audio_s"]),
                offset_s=float(s["local_window_offset_in_audio_s"]),
                window_id=s["window_id"],
            )
            for s in rows
        ]
        self.audio_index = audio_index

    def __len__(self) -> int:
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

        meta = dict(
            window_id=r.window_id,
            original_audio_path=r.original_audio_path,
            local_window_onset_in_audio_s=r.onset_s,
            local_window_offset_in_audio_s=r.offset_s,
            _missing=False,
        )

        if rp is None:
            meta["_missing"] = True
            return {"wave": torch.zeros(desired), "meta": meta}

        try:
            wav, sr = torchaudio.load(rp)
        except Exception as e:
            logging.error(f"torchaudio.load failed: {rp} ({e}); fill zeros.")
            meta["_missing"] = True
            return {"wave": torch.zeros(desired), "meta": meta}

        wav = wav.mean(0) if wav.dim() == 2 else wav
        wav = wav.to(torch.float32)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)

        end = min(start + desired, wav.numel())
        seg = torch.zeros(desired, dtype=torch.float32)
        if end > start:
            seg[: end - start] = wav[start:end]

        return {"wave": seg, "meta": meta}

def collate_batch(samples: List[Dict]) -> Dict:
    return {
        "waves": torch.stack([s["wave"] for s in samples]),
        "metas": [s["meta"] for s in samples],
    }

# -------------------- Model -------------------- #

def load_w2v2(model_name: str, device: str) -> Wav2Vec2Model:
    model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True)
    return model.eval().to(device)

@torch.inference_mode()
def wav2vec_forward(model, waves, device, use_bf16, layer_spec, agg):
    waves = waves.to(device, non_blocking=True)
    waves = (waves - waves.mean(1, keepdim=True)) / waves.std(1, keepdim=True).clamp_min(1e-6)

    def _run(x):
        hs = model(x).hidden_states
        sel = [hs[i] for i in build_layer_indices(layer_spec, len(hs))]
        x = torch.stack(sel).mean(0) if agg == "mean" else torch.cat(sel, -1)
        x = F.interpolate(x.transpose(1, 2), TARGET_T, mode="linear", align_corners=False)
        return x.float()

    if use_bf16 and torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return _run(waves)
    return _run(waves)

# -------------------- Resume -------------------- #

def load_done_set(out_manifest_dir, split, verify_existing, expected_shape):
    out_path = out_manifest_dir / f"{split}.jsonl"
    done = {}
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            wid, feat = row.get("window_id"), row.get("audio_feature_path")
            if wid and feat and (not verify_existing or verify_npy_shape(Path(feat), expected_shape)):
                done[wid] = feat
    logger.info(f"[resume] {split}: found {len(done):,} completed rows.")
    return done

# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_manifest_dir", required=True)
    ap.add_argument("--output_feature_dir", required=True)
    ap.add_argument("--output_manifest_dir", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", choices=["off", "bf16"], default="bf16")
    ap.add_argument("--save_dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--verify_existing", action="store_true", default=True)
    ap.add_argument("--recompute_existing", action="store_true", default=False)
    ap.add_argument("--w2v_model", default="facebook/wav2vec2-large-xlsr-53")
    ap.add_argument("--w2v_layers", default="14,15,16,17,18")
    ap.add_argument("--w2v_agg", choices=["mean", "concat"], default="mean")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    in_dir = Path(args.input_manifest_dir)
    feat_dir = Path(args.output_feature_dir); feat_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_manifest_dir); out_dir.mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(in_dir, s) for s in ("train", "valid", "test")}
    model = load_w2v2(args.w2v_model, args.device)

    hidden = int(getattr(model.config, "hidden_size", 1024))
    n_layers = parse_layers_count(args.w2v_layers)
    out_dim = hidden if args.w2v_agg == "mean" else hidden * n_layers
    expected_shape = (out_dim, TARGET_T)

    audio_index = build_audio_index(Path(args.audio_root))

    for split, rows in splits.items():
        if not rows:
            continue

        done = {}
        if args.resume and not args.recompute_existing:
            done = load_done_set(out_dir, split, args.verify_existing, expected_shape)

        to_process = []
        id2row = {r["window_id"]: r for r in rows}
        for r in rows:
            wid = r["window_id"]
            if args.recompute_existing or wid not in done:
                to_process.append(r)

        logger.info(f"[{split}] skip={len(done):,} | process={len(to_process):,}")
        out_path = out_dir / f"{split}.jsonl"

        ds = AudioWinDataset(to_process, audio_index)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        prefetch_factor=args.prefetch_factor, collate_fn=collate_batch)

        with open(out_path, "a", encoding="utf-8") as fout:
            for batch in tqdm(dl):
                feats = wav2vec_forward(
                    model, batch["waves"], args.device,
                    args.amp == "bf16", args.w2v_layers, args.w2v_agg
                ).cpu()

                for i, m in enumerate(batch["metas"]):
                    if m.get("_missing"):
                        continue
                    cid = content_id_for_sample(
                        Path(m["original_audio_path"]),
                        m["local_window_onset_in_audio_s"],
                        m["local_window_offset_in_audio_s"],
                    )
                    outp = feat_dir / f"{safe_name_from_id(cid)}.npy"
                    arr = feats[i].numpy().astype(
                        np.float16 if args.save_dtype == "float16" else np.float32
                    )
                    atomic_save_npy(outp, arr)

                    r2 = dict(id2row[m["window_id"]])
                    r2["audio_feature_path"] = outp.as_posix()
                    fout.write(json.dumps(r2, ensure_ascii=False) + "\n")

        logger.info(f"[{split}] done -> {out_path}")

    logger.info("Stage-2 sentence-only finished.")

if __name__ == "__main__":
    main()
