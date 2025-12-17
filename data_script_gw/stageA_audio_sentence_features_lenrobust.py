#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
StageA (length-robust, discriminative) — TOKEN export (deterministic, stable)

Key settings:
- AMP disabled by default (amp=off); features saved in float32
- All randomness removed: time_drop_p=0.0, ensemble=1
- Fixed random seeds for full reproducibility
- Output filenames include `_fp32_det` tag when using the deterministic setup
- Output tokens are 2D arrays: [L, Dt]
  * Default TPP levels = 1,2,4,8
  * Dt = 2 * D when using mean|std pooling
"""

import argparse, json, logging, os, re, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("audio_sent_lenrobust_tokens")
SR = 16_000

# ---------- IO utilities ----------

def safe_name_from_id(s: str) -> str:
    return "".join([c if (c.isalnum() or c in "-_.") else "_" for c in s])

def content_id_for_sentence(audio_path: Path, onset_s: float, offset_s: float) -> str:
    return f"{audio_path.stem}_{onset_s:.3f}_{offset_s:.3f}"

def atomic_save_npy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def verify_feat_shape(path: Path, expect_vec_dim: int, expect_L: int, expect_Dt: int) -> bool:
    """
    Backward-compatible shape check:
    - legacy: 1D array with length == expect_vec_dim
    - current: 2D array with shape == (expect_L, expect_Dt)
    """
    try:
        if not path.exists():
            return False
        x = np.load(path.as_posix(), mmap_mode="r")
        if x.ndim == 1 and x.shape[0] == expect_vec_dim:
            return True
        if x.ndim == 2 and x.shape[0] == expect_L and x.shape[1] == expect_Dt:
            return True
        return False
    except Exception:
        return False

def load_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l:
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    return out

# ---------- Sentence table ----------

@dataclass
class SentRow:
    sentence_id: str
    original_audio_path: str
    onset_audio_s: float
    offset_audio_s: float
    window_ids: List[str]

def _pick(d: Dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default

def parse_sentence_row(d: Dict) -> Optional[SentRow]:
    sid = str(_pick(d, ["sentence_id", "sentence_uid", "sent_id", "segment_uid", "segment_id"]))
    ap  = _pick(d, ["original_audio_path", "audio_path"])
    on  = _pick(d, ["segment_onset_in_audio_s", "onset_audio_s", "segment_onset_s", "onset_s"])
    off = _pick(d, ["segment_offset_in_audio_s", "offset_audio_s", "segment_offset_s", "offset_s"])
    if not sid or not ap or on is None or off is None:
        return None
    wins = _pick(d, ["window_ids", "members", "windows"], default=[]) or []
    try:
        on = float(on)
        off = float(off)
    except Exception:
        return None
    return SentRow(sid, ap, on, off, list(wins))

def build_audio_index(roots: List[Path]) -> Dict[str, Path]:
    idx = {}
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            idx[p.as_posix().lower()] = p
            idx[p.name.lower()] = p
            try:
                idx[p.relative_to(root).as_posix().lower()] = p
            except Exception:
                pass
    return idx

def guess_audio_roots(sent_rows: List[Dict]) -> List[Path]:
    roots = set()
    for r in sent_rows:
        a = r.get("original_audio_path") or r.get("audio_path") or ""
        if not a:
            continue
        p = Path(a)
        for c in (
            p.parent,
            p.parent.parent / "stimuli" / "audio",
            p.parent.parent.parent / "stimuli" / "audio",
        ):
            if c.exists():
                roots.add(c.resolve())
    return sorted(list(roots))

# ---------- Dataset ----------

class SentenceAudioDataset(Dataset):
    def __init__(self, rows: List[SentRow], audio_index: Dict[str, Path], max_seconds: Optional[float] = None):
        self.rows = rows
        self.audio_index = audio_index
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.rows)

    def _resolve(self, orig: str) -> Optional[Path]:
        p = Path(orig)
        for k in (p.as_posix().lower(), p.name.lower()):
            if k in self.audio_index:
                return self.audio_index[k]
        return p if p.exists() else None

    def __getitem__(self, i: int):
        r = self.rows[i]
        rp = self._resolve(r.original_audio_path)

        desired = int(round((r.offset_audio_s - r.onset_audio_s) * SR))
        if desired <= 0:
            desired = int(0.5 * SR)
        if self.max_seconds is not None:
            desired = min(desired, int(round(self.max_seconds * SR)))

        start = int(round(r.onset_audio_s * SR))
        meta = dict(
            sentence_id=r.sentence_id,
            original_audio_path=r.original_audio_path,
            onset=r.onset_audio_s,
            offset=r.offset_audio_s,
            window_ids=r.window_ids,
        )

        if rp is None:
            return {"wave": torch.zeros(desired), "len": 0, "meta": meta}

        try:
            wav, sr = torchaudio.load(rp)
        except Exception:
            return {"wave": torch.zeros(desired), "len": 0, "meta": meta}

        if wav.dim() == 2:
            wav = wav.mean(0)
        wav = wav.to(torch.float32)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)

        s0 = max(0, start)
        s1 = min(s0 + desired, wav.numel())
        seg = torch.zeros(desired, dtype=torch.float32)
        if s1 > s0:
            seg[: s1 - s0] = wav[s0:s1]
        eff = int(s1 - s0)
        return {"wave": seg, "len": eff, "meta": meta}

def pad_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    T = max([b["wave"].numel() for b in batch]) if batch else 0
    waves, masks, metas = [], [], []
    for b in batch:
        w = b["wave"]
        if w.numel() < T:
            w = torch.cat([w, torch.zeros(T - w.numel())], 0)
        m = torch.zeros(T, dtype=torch.bool)
        if b["len"] > 0:
            m[: b["len"]] = True
        waves.append(w)
        masks.append(m)
        metas.append(b["meta"])
    return {"waves": torch.stack(waves, 0), "mask": torch.stack(masks, 0), "metas": metas}

# ---------- Model ----------

def parse_layers(spec: str, hs_len: int) -> List[int]:
    s = spec.strip().lower()
    m = re.match(r"last-?(\d+)$", s)
    if m:
        k = int(m.group(1))
        if k <= 0 or k > hs_len:
            raise ValueError("Invalid last-k specification")
        return list(range(hs_len - k, hs_len))
    if "-" in s and "," not in s:
        a, b = s.split("-")
        a, b = int(a), int(b)
        if a > b:
            a, b = b, a
        if b >= hs_len:
            raise ValueError("Layer index out of range")
        return list(range(a, b + 1))
    idx = []
    for t in s.split(","):
        t = t.strip()
        if not t:
            continue
        it = int(t)
        if it < 0 or it >= hs_len:
            raise ValueError("Layer index out of range")
        idx.append(it)
    if not idx:
        raise ValueError("Empty layer list")
    return idx

def load_w2v2(name: str, device: str) -> Wav2Vec2Model:
    m = Wav2Vec2Model.from_pretrained(name, output_hidden_states=True)
    m.eval().to(device)
    return m

def zscore_per_sample_pcm(waves: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    waves = waves.clone()
    valid = mask.sum(dim=1).clamp(min=1).float()
    mean = (waves * mask).sum(dim=1) / valid
    waves = waves - mean[:, None]
    var = ((waves ** 2) * mask).sum(dim=1) / valid
    std = torch.sqrt(var).clamp(min=1e-6)
    return (waves / std[:, None]) * mask

@torch.inference_mode()
def forward_w2v2_hidden(
    model: Wav2Vec2Model,
    waves: torch.Tensor,
    attn_mask: torch.Tensor,
    device: str,
    use_bf16: bool,
    layer_spec: str,
    layer_agg: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _ = waves.shape
    lengths = attn_mask.sum(dim=1).to(torch.long).cpu()
    waves = waves.to(device, non_blocking=True)
    am = attn_mask.to(device, non_blocking=True).to(torch.long)

    def _once(inp, mask_inp):
        out = model(inp, attention_mask=mask_inp, output_hidden_states=True)
        hs = out.hidden_states
        idx = parse_layers(layer_spec, hs_len=len(hs))
        sel = [hs[i] for i in idx]
        x = torch.stack(sel, 0).mean(0) if layer_agg == "mean" else torch.cat(sel, dim=-1)
        with torch.no_grad():
            Ths = model._get_feat_extract_output_lengths(lengths).to(torch.long)
        Th = x.shape[1]
        mask_h = torch.zeros(B, Th, dtype=torch.bool, device=x.device)
        for i in range(B):
            t = int(min(Th, max(0, Ths[i].item())))
            if t > 0:
                mask_h[i, :t] = True
        return x, mask_h

    if use_bf16 and torch.cuda.is_available():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            x, mask_h = _once(waves, am)
    else:
        x, mask_h = _once(waves, am)
    return x.float(), mask_h

# ---------- Token pooling ----------

def _drop_time(x_valid: torch.Tensor, drop_p: float) -> torch.Tensor:
    if drop_p <= 0 or x_valid.shape[0] == 0:
        return x_valid
    keep = torch.rand(x_valid.shape[0], device=x_valid.device) > drop_p
    return x_valid[keep] if keep.any() else x_valid

def _adaptive_pool_bins(x_valid: torch.Tensor, out_bins: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_valid.numel() == 0:
        D = x_valid.shape[-1]
        z = torch.zeros(out_bins, D, device=x_valid.device)
        return z, z
    xt = x_valid.transpose(0, 1).unsqueeze(0)  # [1, D, T]
    m1 = F.adaptive_avg_pool1d(xt, out_bins).squeeze(0).transpose(0, 1)
    m2 = F.adaptive_avg_pool1d(xt ** 2, out_bins).squeeze(0).transpose(0, 1)
    return m1, m2

def pool_tokens(
    feats: torch.Tensor,
    mask_h: torch.Tensor,
    *,
    levels: Optional[List[int]] = None,
    bins: Optional[int] = None,
    with_std: bool = True,
    time_drop_p: float = 0.0,
    ensemble: int = 1,
) -> torch.Tensor:
    """
    Produce token sequences:
    - TPP: levels=[1,2,4,...] → L = sum(levels)
    - avg_bins: bins=K → L = K
    Returns [B, L, Dt], where Dt = D or 2D (mean|std).
    """
    assert (levels is not None) ^ (bins is not None)
    B, Th, D = feats.shape
    outs = []
    for i in range(B):
        x = feats[i][mask_h[i]]  # [T_valid, D]
        toks_en = []
        for _ in range(max(1, ensemble)):
            xi = _drop_time(x, time_drop_p)
            parts = []
            if levels is not None:
                for L in levels:
                    m1, m2 = _adaptive_pool_bins(xi, L)
                    if with_std:
                        std = torch.sqrt((m2 - m1 * m1).clamp(min=1e-12))
                        parts.append(torch.cat([m1, std], dim=-1))
                    else:
                        parts.append(m1)
            else:
                m1, m2 = _adaptive_pool_bins(xi, bins)
                if with_std:
                    std = torch.sqrt((m2 - m1 * m1).clamp(min=1e-12))
                    parts.append(torch.cat([m1, std], dim=-1))
                else:
                    parts.append(m1)
            toks_en.append(torch.cat(parts, dim=0))
        toks = torch.stack(toks_en, 0).mean(0)
        toks = torch.nn.functional.normalize(toks, p=2, dim=-1)
        outs.append(toks)
    return torch.stack(outs, 0)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentence_table_dir", required=True)
    ap.add_argument("--output_feature_dir", required=True)
    ap.add_argument("--output_sentence_dir", required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", choices=["off", "bf16"], default="off")
    ap.add_argument("--save_dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--max_seconds", type=float, default=None)

    ap.add_argument("--w2v_model", default="facebook/wav2vec2-large-xlsr-53")
    ap.add_argument("--w2v_layers", default="14,15,16,17,18")
    ap.add_argument("--w2v_agg", choices=["mean", "concat"], default="mean")

    ap.add_argument("--pooling", choices=["tpp", "avg_bins"], default="tpp")
    ap.add_argument("--tpp_levels", default="1,2,4,8")
    ap.add_argument("--bins", type=int, default=32)
    ap.add_argument("--with_std", action="store_true", default=True)
    ap.add_argument("--time_drop_p", type=float, default=0.0)
    ap.add_argument("--ensemble", type=int, default=1)

    # Legacy flags (kept for compatibility)
    ap.add_argument("--power_norm", choices=["none", "sqrt", "log1p"], default="sqrt")
    ap.add_argument("--unit_norm", action="store_true", default=True)
    ap.add_argument("--dataset_norm_npz", default="")
    ap.add_argument("--bin_center_alpha", type=float, default=0.25)

    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--verify_existing", action="store_true", default=True)
    ap.add_argument("--recompute_existing", action="store_true", default=False)
    ap.add_argument("--write_sentence_key", action="store_true", default=True)
    args = ap.parse_args()

    # Deterministic seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Performance settings (AMP off; TF32 allowed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out_feat_dir = Path(args.output_feature_dir); out_feat_dir.mkdir(parents=True, exist_ok=True)
    out_sent_dir = Path(args.output_sentence_dir); out_sent_dir.mkdir(parents=True, exist_ok=True)

    # Load sentence tables
    splits = {}
    for sp in ("train", "valid", "test"):
        p = Path(args.sentence_table_dir) / f"{sp}.jsonl"
        rows = [s for r in load_jsonl(p) if (s := parse_sentence_row(r))]
        splits[sp] = rows
    if sum(len(v) for v in splits.values()) == 0:
        log.error("No sentence rows found.")
        return

    # Audio index
    all_rows = [r for lst in splits.values() for r in lst]
    audio_roots = guess_audio_roots([{"original_audio_path": r.original_audio_path} for r in all_rows])
    audio_index = build_audio_index(audio_roots)

    # Model
    model = load_w2v2(args.w2v_model, args.device)
    hidden_size = int(getattr(model.config, "hidden_size", 1024))
    D = hidden_size if args.w2v_agg == "mean" else hidden_size

    # Token shape
    if args.pooling == "tpp":
        levels = [int(x) for x in args.tpp_levels.split(",") if x.strip()]
        total_bins = sum(levels)
    else:
        levels = None
        total_bins = args.bins

    per_token_dim = (2 * D if args.with_std else D)
    flattened_dim = per_token_dim * total_bins
    log.info(f"Tokens: L={total_bins}, Dt={per_token_dim} (flattened_dim={flattened_dim})")

    use_bf16 = (args.amp == "bf16")
    save_fp16 = (args.save_dtype == "float16")

    def _make_sentence_key(m):
        ap = m["original_audio_path"]
        stem = Path(ap).stem
        on = float(m["onset"]); off = float(m["offset"])
        on_ms = int(round(on * 1000.0))
        off_ms = int(round(off * 1000.0))
        return f"{stem}::{on_ms}-{off_ms}"

    for sp in ("train", "valid", "test"):
        rows = splits[sp]
        if not rows:
            log.info(f"{sp}: empty, skip.")
            continue

        done_path = out_sent_dir / f"{sp}.jsonl"
        done_map = {}
        if args.resume and done_path.exists():
            for r in load_jsonl(done_path):
                sid = str(r.get("sentence_id") or r.get("sentence_uid") or r.get("sent_id") or "")
                feat = r.get("audio_sentence_feature_path", "")
                if sid and feat and (
                    not args.verify_existing
                    or verify_feat_shape(Path(feat), flattened_dim, total_bins, per_token_dim)
                ):
                    done_map[sid] = feat
            log.info(f"[{sp}] resume found {len(done_map):,} completed")

        todo = []
        for r in rows:
            if args.recompute_existing:
                todo.append(r)
                continue
            if r.sentence_id in done_map and (
                not args.verify_existing
                or verify_feat_shape(Path(done_map[r.sentence_id]), flattened_dim, total_bins, per_token_dim)
            ):
                continue
            todo.append(r)

        log.info(f"{sp}: total={len(rows):,} | to_process={len(todo):,}")

        ds = SentenceAudioDataset(todo, audio_index, max_seconds=args.max_seconds)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=(args.num_workers > 0),
            collate_fn=pad_collate,
        )

        # Output mode tag (explicitly marking deterministic setup)
        mode_tag = (f"TPP_{args.tpp_levels.replace(',', '-')}" if args.pooling == "tpp" else f"AB{args.bins}")
        if args.with_std:
            mode_tag += "S"
        if args.time_drop_p > 0:
            mode_tag += f"_drop{int(args.time_drop_p * 100)}"
        if args.ensemble > 1:
            mode_tag += f"_ens{args.ensemble}"
        if (args.amp == "off" and args.time_drop_p == 0.0 and args.ensemble == 1 and args.save_dtype == "float32"):
            mode_tag += "_fp32_det"

        with open(done_path, "a", encoding="utf-8") as fout:
            for batch in tqdm(dl, desc=f"{sp}: audio-sent"):
                waves = batch["waves"]
                mask = batch["mask"]
                metas = batch["metas"]

                waves = zscore_per_sample_pcm(waves, mask)

                feats, mask_h = forward_w2v2_hidden(
                    model=model,
                    waves=waves,
                    attn_mask=mask,
                    device=args.device,
                    use_bf16=use_bf16,
                    layer_spec=args.w2v_layers,
                    layer_agg=args.w2v_agg,
                )

                # Token generation: [B, L, Dt]
                if args.pooling == "tpp":
                    levels = [int(x) for x in args.tpp_levels.split(",") if x.strip()]
                    tokens = pool_tokens(
                        feats,
                        mask_h,
                        levels=levels,
                        with_std=args.with_std,
                        time_drop_p=args.time_drop_p,
                        ensemble=args.ensemble,
                    )
                else:
                    tokens = pool_tokens(
                        feats,
                        mask_h,
                        bins=args.bins,
                        with_std=args.with_std,
                        time_drop_p=args.time_drop_p,
                        ensemble=args.ensemble,
                    )

                # Write outputs (2D tokens)
                for i, m in enumerate(metas):
                    rp = Path(m["original_audio_path"])
                    fn = (
                        f"{safe_name_from_id(content_id_for_sentence(rp, float(m['onset']), float(m['offset'])))}"
                        f"_sentAUDIO_{mode_tag}.npy"
                    )
                    outp = out_feat_dir / fn
                    arr = tokens[i].cpu().numpy()
                    if save_fp16:
                        arr = arr.astype(np.float16)
                    atomic_save_npy(outp, arr)

                    r2 = dict(
                        sentence_id=m["sentence_id"],
                        original_audio_path=m["original_audio_path"],
                        onset_audio_s=m["onset"],
                        offset_audio_s=m["offset"],
                        audio_sentence_feature_path=outp.as_posix(),
                    )
                    if args.write_sentence_key:
                        r2["sentence_key"] = _make_sentence_key(m)
                    if isinstance(m.get("window_ids"), list):
                        r2["window_ids"] = m["window_ids"]
                    fout.write(json.dumps(r2, ensure_ascii=False) + "\n")

        log.info(f"[{sp}] done -> {done_path.as_posix()}")

    log.info("All splits done.")

if __name__ == "__main__":
    main()
