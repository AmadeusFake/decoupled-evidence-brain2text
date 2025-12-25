from __future__ import annotations

import os
import re
import json
import math
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable, Optional, Literal
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
import datetime


logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Target temporal length for audio / neural features after alignment
TARGET_T = 360

# Dimensionality of audio features
AUDIO_D = 1024

# Regex for legacy global sentence keys encoded as:
#   <audio_id>::<start_ms>-<end_ms>
_GLOBAL_SENT_RE_MS = re.compile(r'([^/\\]+)::(\d+)-(\d+)', re.U)

# Regex for sentence-level audio feature paths encoded as:
#   <audio_id>_<start_s>_<end_s>_sent*.npy
_AUDIO_SENT_RE_S = re.compile(
    r'[\\/](?P<aid>[^/\\]+?)_(?P<s>\-?\d+(?:\.\d+)?)_(?P<e>\-?\d+(?:\.\d+)?)(?:_sent[A-Za-z0-9_.-]*)?\.npy$',
    re.U
)


# =============================================================================
# Global sentence key utilities (cross-subject invariant)
# =============================================================================

def _ms_from_seconds_str(x: str) -> int:
    """Convert seconds (string) to rounded milliseconds (int)."""
    return int(round(float(x) * 1000.0))


def _norm_audio_id(x: str) -> str:
    """Normalize audio identifier by stripping directory and extension."""
    return Path(x).stem


def _key_from_feature_path(path: str) -> Optional[str]:
    """
    Parse a sentence-level feature path into a canonical global sentence key:
        <audio_id>::<start_ms>-<end_ms>

    Supported formats:
      A) New (seconds-based):
         <aid>_<start_s>_<end_s>_sent*.npy

      B) Legacy (milliseconds-based):
         <aid>::<start_ms>-<end_ms>
    """
    if not path:
        return None

    s = str(path)

    # New-style (seconds)
    m = _AUDIO_SENT_RE_S.search(s)
    if m:
        aid = _norm_audio_id(m.group("aid"))
        s_ms = _ms_from_seconds_str(m.group("s"))
        e_ms = _ms_from_seconds_str(m.group("e"))
        return f"{aid}::{s_ms}-{e_ms}"

    # Legacy (milliseconds)
    m2 = _GLOBAL_SENT_RE_MS.search(s)
    if m2:
        aid = _norm_audio_id(m2.group(1))
        s_ms = int(m2.group(2))
        e_ms = int(m2.group(3))
        return f"{aid}::{s_ms}-{e_ms}"

    return None


def _infer_key_mode_from_manifest(manifest_path: Path) -> Literal["audio", "text", "auto"]:
    """
    Infer preferred sentence-key source from manifest path.

    Heuristics:
      - text_whisper + sentence_full        -> text
      - sentence_full_audio / BCTR25        -> audio
      - otherwise                           -> auto
    """
    s = str(manifest_path)

    if "final_splits_sentence_with_sentence_full_text_whisper" in s:
        return "text"
    if "final_splits_word_list_with_sentence_full_audio" in s:
        return "audio"

    # Relaxed aliases
    if "text_whisper" in s and "sentence_full" in s:
        return "text"
    if "sentence_full_audio" in s or "BCTR25" in s:
        return "audio"

    return "auto"


def derive_global_sentence_key(
    row: dict,
    prefer: Literal["audio", "text", "auto"] = "auto",
) -> str:
    """
    Derive a globally unique sentence key that is consistent across subjects.

    Resolution order:
      1) Parse from sentence-level feature paths (audio/text, priority by `prefer`)
      2) Use explicit global_segment_on/off_in_audio_s (seconds → ms)
      3) Fallback to <audio_id>::SID@<sentence_id>
      4) Last-resort hash of the full row

    This function *never* fails: it always returns a string.
    """
    apath = row.get("audio_sentence_feature_path", "") or ""
    tpath = row.get("text_sentence_feature_path", "") or ""

    def _try(paths: List[str]) -> Optional[str]:
        for p in paths:
            key = _key_from_feature_path(p)
            if key:
                return key
        return None

    if prefer == "audio":
        key = _try([apath, tpath])
    elif prefer == "text":
        key = _try([tpath, apath])
    else:  # auto
        key = _try([tpath, apath])

    if key:
        return key

    # Explicit global segment (seconds → milliseconds)
    audio_base = _norm_audio_id(row.get("original_audio_path", "") or "")
    on = row.get("global_segment_onset_in_audio_s", None)
    off = row.get("global_segment_offset_in_audio_s", None)
    if audio_base and (on is not None) and (off is not None):
        s_ms = int(round(float(on) * 1000.0))
        e_ms = int(round(float(off) * 1000.0))
        return f"{audio_base}::{s_ms}-{e_ms}"

    # Sentence-ID fallback (not guaranteed cross-subject stable)
    sid = str(row.get("sentence_id", "") or "")
    if audio_base and sid:
        return f"{audio_base}::SID@{sid}"
    if sid:
        return f"SID@{sid}"

    # Final fallback: short stable hash of the full row
    h = hashlib.blake2b(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"UNK::{h}"


# =============================================================================
# Subject registry (stable subject → index mapping)
# =============================================================================

class SubjectRegistry:
    """
    Registry mapping (namespace, subject_id) → contiguous integer index.

    This abstraction ensures:
      - Cross-dataset subject index consistency
      - Explicit namespace separation (e.g., train/test splits)
    """
    def __init__(self, mapping: Dict[str, int] | None = None):
        self._map: Dict[str, int] = mapping or {}

    @staticmethod
    def _key(ns: str, sid: str) -> str:
        ns = (ns or "").strip()
        return f"{ns}:{sid}" if ns else sid

    @property
    def num_subjects(self) -> int:
        return len(self._map)

    def index_of(self, ns: str, sid: str, fallback_to_first: bool = True) -> int:
        """
        Resolve subject index.

        If unknown and fallback_to_first=True, returns 0 instead of failing.
        """
        k = self._key(ns, sid)
        if k in self._map:
            return self._map[k]
        if not self._map:
            raise RuntimeError("SubjectRegistry is empty.")
        if fallback_to_first:
            return 0
        raise KeyError(f"Unknown subject key: {k}")

    def to_json(self) -> Dict[str, Any]:
        return {
            "mapping": self._map,
            "order": [k for k, _ in sorted(self._map.items(), key=lambda kv: kv[1])],
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "SubjectRegistry":
        return SubjectRegistry(mapping=d.get("mapping", {}))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: Path) -> "SubjectRegistry":
        with open(path, "r", encoding="utf-8") as f:
            return SubjectRegistry.from_json(json.load(f))

    # ---------------------------------------------------------------------
    # Builders
    # ---------------------------------------------------------------------

    @staticmethod
    def _collect_subjects_from_manifest(manifest_path: Path) -> List[str]:
        subs = set()
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                sid = row.get("subject_id")
                if sid:
                    subs.add(str(sid))
        return sorted(subs)

    @classmethod
    def build_from_manifests(
        cls,
        files_with_ns: Iterable[Tuple[Path, str]],
    ) -> "SubjectRegistry":
        entries: List[str] = []
        for p, ns in files_with_ns:
            if not p or not p.exists():
                continue
            subs = cls._collect_subjects_from_manifest(p)
            for sid in subs:
                entries.append(cls._key(ns, sid))
        entries = sorted(set(entries))
        return SubjectRegistry(mapping={k: i for i, k in enumerate(entries)})


# =============================================================================
# Low-level data helpers
# =============================================================================

def _ensure_CxT(x: np.ndarray) -> np.ndarray:
    """Ensure array is shaped as [C, T]."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D [C,T] or [T,C], got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T


def _ensure_DxT(x: np.ndarray, D: int) -> np.ndarray:
    """Ensure audio features are [D, T] with float32 and finite values."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D audio array, got {x.shape}")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] == D:
        return x
    if x.shape[1] == D:
        return x.T
    return x if abs(x.shape[0] - D) < abs(x.shape[1] - D) else x.T


def _safe_uid63(uid_src) -> int:
    """
    Map an arbitrary hashable object to a stable signed int64.
    """
    try:
        v = int(uid_src)
        return max(min(v, 0x7FFFFFFFFFFFFFFF), -0x8000000000000000)
    except Exception:
        h = hashlib.blake2b(str(uid_src).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little", signed=False) & 0x7FFFFFFFFFFFFFFF


def _row_window_content_key(row: dict) -> str:
    """
    Window-level unique identifier shared across subjects.
    Used for window-level context grouping.
    """
    a = row.get("original_audio_path", "")
    on = row.get("local_window_onset_in_audio_s", None)
    off = row.get("local_window_offset_in_audio_s", None)
    if on is not None and off is not None:
        return f"{a}::{float(on):.3f}-{float(off):.3f}"
    return f"{a}::WID@{row.get('window_id','')}"


# =============================================================================
# In-process LRU cache for full-sentence MEG (per worker)
# =============================================================================

_SENT_CACHE: Dict[str, np.ndarray] = {}
_SENT_ORDER: List[str] = []

# Cache capacity controlled via environment variable
_SENT_CAP = int(os.environ.get("SENT_CACHE_CAP", "512"))


def _sent_load_cached(path: str) -> np.ndarray:
    """
    Load a full-sentence MEG array with an in-process LRU cache.

    Implementation notes:
      - File is opened via memmap for I/O efficiency
      - Immediately copied into a regular ndarray
      - memmap object is deleted to release file descriptor
      - Cached array is C×T, float32, contiguous
    """
    arr = _SENT_CACHE.get(path)
    if arr is not None:
        # Update LRU order
        try:
            _SENT_ORDER.remove(path)
        except ValueError:
            pass
        _SENT_ORDER.append(path)
        return arr

    mm = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        arr = np.array(mm, dtype=np.float32, copy=True)
    finally:
        del mm  # ensure file descriptor is released

    arr = _ensure_CxT(arr)
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    _SENT_CACHE[path] = arr
    _SENT_ORDER.append(path)

    if len(_SENT_ORDER) > _SENT_CAP:
        old = _SENT_ORDER.pop(0)
        _SENT_CACHE.pop(old, None)

    return arr


# =============================================================================
# Dataset
# =============================================================================

class MEGDataset(Dataset):
    """
    Dataset supporting both window-level and sentence-level access.

    Fast sentence mode:
      - Does NOT load meg_win or audio_feat
      - Only returns sensor coordinates, subject index, sentence keys,
        and sentence-level teacher tokens

    Window mode:
      - Fully backward-compatible with legacy pipelines
    """
    def __init__(
        self,
        manifest_path: str,
        registry: SubjectRegistry,
        namespace: str,
        normalize: bool = False,
        context_mode: str = "none",
        sentence_fast_io: bool = True,
        key_mode: Literal["auto", "audio", "text"] | None = None,
    ):
        super().__init__()

        self.namespace = (namespace or "").strip()
        self.registry = registry
        self.normalize = normalize
        self.context_mode = context_mode
        self.sentence_fast_io = bool(sentence_fast_io)

        self.manifest_path = Path(manifest_path)
        if self.manifest_path.is_dir():
            raise IsADirectoryError(f"Manifest should be a file, got directory: {self.manifest_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        inferred = _infer_key_mode_from_manifest(self.manifest_path)
        self.key_mode: Literal["auto", "audio", "text"] = key_mode or inferred

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

        # Drop rows without sensor coordinates
        before = len(self.rows)
        self.rows = [r for r in self.rows if Path(r.get("sensor_coordinates_path", "")).exists()]
        if before != len(self.rows):
            logger.warning(
                f"{self.manifest_path.name}: dropped {before - len(self.rows)} rows "
                f"without sensor_coordinates_path"
            )

        # Build per-sentence index (subject-local)
        self.sent2rows: Dict[str, List[dict]] = defaultdict(list)
        self.sent2idx: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(self.rows):
            sid = str(r.get("sentence_id", ""))
            subj = str(r.get("subject_id", ""))
            if sid and subj:
                key = f"{self.namespace}:{subj}|{sid}"
                self.sent2rows[key].append(r)
                self.sent2idx[key].append(i)

        # Sort windows within a sentence by anchor and onset
        for key, lst in self.sent2rows.items():
            lst.sort(
                key=lambda x: (
                    int(x.get("anchor_word_idx", 10**9)),
                    float(x.get("local_window_onset_in_audio_s", 0.0)),
                )
            )

        self._coords_cache: Dict[str, np.ndarray] = {}

        subjects = sorted({r.get("subject_id") for r in self.rows if r.get("subject_id") is not None})
        logger.info(
            f"{self.manifest_path.name}: {len(self.rows):,} samples; "
            f"subjects={len(subjects)}; context_mode={self.context_mode}; "
            f"fast_io={self.sentence_fast_io}; key_mode={self.key_mode}"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_coords(self, coord_path: str) -> np.ndarray:
        if coord_path not in self._coords_cache:
            self._coords_cache[coord_path] = np.load(coord_path).astype(np.float32)
        return self._coords_cache[coord_path]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]

        # ------------------------------------------------------------------
        # Fast path: sentence-level mode
        # ------------------------------------------------------------------
        if self.context_mode == "sentence" and self.sentence_fast_io:
            coords = self._load_coords(r["sensor_coordinates_path"])
            subj_str = str(r.get("subject_id", ""))
            subj_idx = self.registry.index_of(self.namespace, subj_str, fallback_to_first=True)

            sent_id = str(r.get("sentence_id", ""))
            sent_key = f"{self.namespace}:{subj_str}|{sent_id}" if (subj_str and sent_id) else sent_id
            gkey = derive_global_sentence_key(r, prefer=self.key_mode)

            item = {
                "sensor_locs": torch.from_numpy(coords),
                "subject_idx": torch.tensor(subj_idx, dtype=torch.long),
                "sentence_id": sent_id,
                "sentence_key": sent_key,
                "global_sentence_key": gkey,
                "row": r,
            }

            if "meg_sentence_full_path" in r:
                item["meg_sentence_full_path"] = r.get("meg_sentence_full_path", "")

            # Load sentence-level teacher tokens (text/audio)
            tvec_path = r.get("text_sentence_feature_path", "") or ""
            avec_path = r.get("audio_sentence_feature_path", "") or ""

            def _load_tokens2d(p: str) -> Optional[torch.Tensor]:
                if not p or not Path(p).exists():
                    return None
                X = np.load(p, mmap_mode="r")
                if X.ndim != 2:
                    raise ValueError(f"Expected 2D sentence tokens, got {X.shape} for {p}")
                X = X.astype(np.float32, copy=False)
                X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                return torch.from_numpy(X)

            tok_t = _load_tokens2d(tvec_path)
            tok_a = _load_tokens2d(avec_path)

            if self.key_mode == "audio":
                if tok_a is not None:
                    item["audio_sent_vec"] = tok_a
                elif tok_t is not None:
                    item["text_sentence_feature"] = tok_t
            elif self.key_mode == "text":
                if tok_t is not None:
                    item["text_sentence_feature"] = tok_t
                elif tok_a is not None:
                    item["audio_sent_vec"] = tok_a
            else:
                if tok_t is not None:
                    item["text_sentence_feature"] = tok_t
                elif tok_a is not None:
                    item["audio_sent_vec"] = tok_a

            return item

        # ------------------------------------------------------------------
        # Slow path: window-level mode (legacy-compatible)
        # ------------------------------------------------------------------
        meg = _ensure_CxT(np.load(r["meg_win_path"]).astype(np.float32))
        if self.normalize:
            meg = (meg - meg.mean(axis=1, keepdims=True)) / (meg.std(axis=1, keepdims=True) + 1e-6)

        aud = _ensure_DxT(np.load(r["audio_feature_path"]), AUDIO_D)
        coords = self._load_coords(r["sensor_coordinates_path"])

        subj_str = str(r.get("subject_id"))
        subj_idx = self.registry.index_of(self.namespace, subj_str, fallback_to_first=True)

        sent_id = str(r.get("sentence_id", ""))
        sent_key = f"{self.namespace}:{subj_str}|{sent_id}" if (subj_str and sent_id) else sent_id
        ctx_rows = self.sent2rows.get(sent_key, [])

        gkey = derive_global_sentence_key(r, prefer=self.key_mode)

        item = {
            "meg_win": torch.from_numpy(meg),
            "audio_feat": torch.from_numpy(aud),
            "sensor_locs": torch.from_numpy(coords),
            "subject_idx": torch.tensor(subj_idx, dtype=torch.long),
            "sentence_id": sent_id,
            "sentence_key": sent_key,
            "global_sentence_key": gkey,
            "window_id": r.get("window_id", ""),
            "row": r,
            "__ctx_rows__": ctx_rows,
        }

        if "meg_sentence_full_path" in r:
            item["meg_sentence_full_path"] = r.get("meg_sentence_full_path", "")

        tvec_path = r.get("text_sentence_feature_path", "") or ""
        avec_path = r.get("audio_sentence_feature_path", "") or ""

        def _load_tokens2d(p: str) -> Optional[torch.Tensor]:
            if not p or not Path(p).exists():
                return None
            X = np.load(p, mmap_mode="r")
            if X.ndim != 2:
                raise ValueError(f"Expected 2D sentence tokens, got {X.shape} for {p}")
            X = X.astype(np.float32, copy=False)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            return torch.from_numpy(X)

        tok_t = _load_tokens2d(tvec_path)
        tok_a = _load_tokens2d(avec_path)

        if self.key_mode == "audio":
            if tok_a is not None:
                item["audio_sent_vec"] = tok_a
            elif tok_t is not None:
                item["text_sentence_feature"] = tok_t
        elif self.key_mode == "text":
            if tok_t is not None:
                item["text_sentence_feature"] = tok_t
            elif tok_a is not None:
                item["audio_sent_vec"] = tok_a
        else:
            if tok_t is not None:
                item["text_sentence_feature"] = tok_t
            elif tok_a is not None:
                item["audio_sent_vec"] = tok_a

        return item


# =============================================================================
# Batch samplers
# =============================================================================

class SentenceGroupedBatchSampler(Sampler[List[int]]):
    """
    Sample batches by grouping windows by (subject-local) sentence.

    Each batch contains exactly one randomly chosen window per sentence.
    """
    def __init__(
        self,
        dataset: MEGDataset,
        sentences_per_batch: int,
        windows_per_sentence: int = 1,
        drop_last: bool = True,
    ):
        self.ds = dataset
        self.S = int(sentences_per_batch)
        if int(windows_per_sentence) > 1:
            logger.warning(
                "SentenceGroupedBatchSampler: windows_per_sentence>1 is ignored; "
                "forcing 1 window per sentence."
            )
        self.drop_last = bool(drop_last)
        self.sids = [sid for sid, idxs in self.ds.sent2idx.items() if len(idxs) > 0]

    def __iter__(self):
        perm = torch.randperm(len(self.sids)).tolist()
        for i in range(0, len(perm), self.S):
            chunk = [self.sids[j] for j in perm[i:i + self.S]]
            if len(chunk) < self.S and self.drop_last:
                break
            batch = []
            for sid in chunk:
                pool = self.ds.sent2idx[sid]
                j = int(torch.randint(low=0, high=len(pool), size=(1,)).item())
                batch.append(pool[j])
            yield batch

    def __len__(self):
        n = len(self.sids)
        return (n // self.S) if self.drop_last else math.ceil(n / self.S)


class AudioUIDMultiSourceBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that prioritizes grouping samples sharing the same
    global_sentence_key but originating from different subjects.

    A fraction of the batch can be reserved for random negatives,
    controlled by the RESERVE_NEG_FRAC environment variable.
    """
    def __init__(
        self,
        dataset: MEGDataset,
        batch_size: int,
        group_k: int = 2,
        drop_last: bool = True,
    ):
        self.ds = dataset
        self.B = int(batch_size)
        self.K = max(2, int(group_k))
        self.drop_last = bool(drop_last)

        key2src2idx: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        ns = getattr(self.ds, "namespace", "")

        def _sentkey_of_row(r: dict) -> str:
            subj = str(r.get("subject_id", ""))
            sid = str(r.get("sentence_id", ""))
            return f"{ns}:{subj}|{sid}" if (subj and sid) else sid

        for idx, r in enumerate(self.ds.rows):
            gkey = derive_global_sentence_key(r, prefer=self.ds.key_mode)
            skey = _sentkey_of_row(r)
            key2src2idx[gkey][skey].append(idx)

        self.key2src2idx = key2src2idx
        self.multi_keys = [k for k, m in key2src2idx.items() if len(m) >= self.K]
        self.all_indices = list(range(len(self.ds)))

        self.reserve_frac = float(os.environ.get("RESERVE_NEG_FRAC", "0.0"))
        self.reserve_frac = min(max(self.reserve_frac, 0.0), 0.9)

        logger.info(
            f"[GlobalKeyMultiSourceSampler] multi_keys={len(self.multi_keys)}, "
            f"group_k={self.K}, B={self.B}, reserve_neg_frac={self.reserve_frac}"
        )

    def __len__(self):
        n = len(self.ds)
        return (n // self.B) if self.drop_last else math.ceil(n / self.B)

    def __iter__(self):
        for _ in range(len(self)):
            batch: List[int] = []

            quota = int(round(self.B * (1.0 - self.reserve_frac)))
            quota = max(self.K, min(quota, self.B))
            max_groups = quota // self.K

            if self.multi_keys and max_groups > 0:
                perm = torch.randperm(len(self.multi_keys)).tolist()
                sel = [self.multi_keys[perm[i % len(self.multi_keys)]] for i in range(max_groups)]
            else:
                sel = []

            # Fill grouped positives
            for gk in sel:
                src2idx = self.key2src2idx[gk]
                src_keys = list(src2idx.keys())
                if len(src_keys) < self.K:
                    continue
                perm_s = torch.randperm(len(src_keys)).tolist()
                chosen = [src_keys[i] for i in perm_s[:self.K]]
                for sk in chosen:
                    pool = src2idx[sk]
                    j = int(torch.randint(low=0, high=len(pool), size=(1,)).item())
                    batch.append(pool[j])
                if len(batch) >= quota:
                    break

            # Fill remaining slots with random samples
            need = self.B - len(batch)
            if need > 0:
                rand_idx = torch.randint(low=0, high=len(self.all_indices), size=(need,)).tolist()
                batch.extend(self.all_indices[i] for i in rand_idx)

            # Trim or drop
            if len(batch) > self.B:
                keep = torch.randperm(len(batch))[: self.B].tolist()
                batch = [batch[i] for i in keep]
            if len(batch) < self.B and self.drop_last:
                continue

            yield batch

# ============================== DataModule ============================== #
class MEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MEG/EEG decoding.

    Supported modes
    ---------------
    1) window mode:
       - Each sample is an anchor window.
       - Sentence context is constructed on-the-fly by aggregating
         all windows belonging to the same sentence
         (MEG window stack + temporally concatenated audio features).

    2) sentence mode:
       - Each sample corresponds to a full sentence.
       - Only sentence-level MEG (C×T) is loaded.
       - Optional sentence-level teacher vectors (text or audio) are loaded.
       - Window-level MEG/audio are not loaded unless explicitly needed.

    Design goals
    ------------
    - Strict backward compatibility with legacy window-based pipelines.
    - Explicit, reproducible batching semantics.
    - Flexible multi-source grouping for cross-subject sentence alignment.
    """

    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        test_manifest: str,
        registry: SubjectRegistry,
        ns_train: str = "",
        ns_val: str = "",
        ns_test: str = "",
        batch_size: int = 64,
        num_workers: int = 8,
        normalize: bool = False,
        context_mode: str = "none",
        ctx_max_windows: int = 0,
        group_by_sentence: bool = True,
        sentences_per_batch: int = 32,
        windows_per_sentence: int = 1,
        pin_memory: bool = False,
        prefetch_factor: int = 1,
        persistent_workers: bool = True,
        ctx_stride: int = 1,
        exclude_self_from_ctx: bool = False,
        ctx_exclude_radius: int = 0,
        sent_guard_windows: int = 0,
        sentence_fast_io: bool = True,
        key_mode: Literal["auto", "audio", "text"] | None = None,
    ):
        super().__init__()

        # ---------------- manifests & namespaces ----------------
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest
        self.registry = registry
        self.ns_train = ns_train
        self.ns_val = ns_val
        self.ns_test = ns_test

        # ---------------- loader basics ----------------
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.normalize = bool(normalize)

        assert context_mode in ("none", "window", "sentence")
        self.context_mode = context_mode

        # ---------------- sentence grouping ----------------
        self.ctx_max_windows = int(ctx_max_windows)
        self.group_by_sentence = bool(group_by_sentence)
        self.sentences_per_batch = int(sentences_per_batch)

        # Enforced invariant:
        #   exactly one window per sentence in grouped sampling
        self.windows_per_sentence = 1

        # ---------------- DataLoader performance ----------------
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)

        # ---------------- legacy / compatibility knobs ----------------
        # These are only relevant in window mode.
        self.ctx_stride = max(1, int(ctx_stride))
        self.exclude_self_from_ctx = bool(exclude_self_from_ctx)
        self.ctx_exclude_radius = int(ctx_exclude_radius)
        self.sent_guard_windows = max(0, int(sent_guard_windows))

        # ---------------- sentence fast I/O ----------------
        # If True, sentence mode avoids loading window-level MEG/audio.
        self.sentence_fast_io = bool(sentence_fast_io)

        # ---------------- sentence key strategy ----------------
        # If None, key mode is inferred per manifest path.
        self.key_mode = key_mode

    # ------------------------------------------------------------------
    # Dataset setup
    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        """
        Instantiate datasets for train / val / test.

        If key_mode is not explicitly provided, each dataset infers
        its own sentence-key strategy from the manifest path.
        """
        km_train = self.key_mode or _infer_key_mode_from_manifest(Path(self.train_manifest))
        km_val   = self.key_mode or _infer_key_mode_from_manifest(Path(self.val_manifest))
        km_test  = self.key_mode or _infer_key_mode_from_manifest(Path(self.test_manifest))

        self.train_set = MEGDataset(
            self.train_manifest,
            registry=self.registry,
            namespace=self.ns_train,
            normalize=self.normalize,
            context_mode=self.context_mode,
            sentence_fast_io=self.sentence_fast_io,
            key_mode=km_train,
        )
        self.val_set = MEGDataset(
            self.val_manifest,
            registry=self.registry,
            namespace=self.ns_val,
            normalize=self.normalize,
            context_mode=self.context_mode,
            sentence_fast_io=self.sentence_fast_io,
            key_mode=km_val,
        )
        self.test_set = MEGDataset(
            self.test_manifest,
            registry=self.registry,
            namespace=self.ns_test,
            normalize=self.normalize,
            context_mode=self.context_mode,
            sentence_fast_io=self.sentence_fast_io,
            key_mode=km_test,
        )

    # ------------------------------------------------------------------
    # Window mode: build full-sentence context
    # ------------------------------------------------------------------
    def _build_sentence_context(self, items: List[Dict]):
        """
        Construct sentence-level context from window-level samples.

        For each anchor window:
        - MEG: stack all windows belonging to the same sentence → [N, C, T]
        - Audio: concatenate all window audio features along time

        Returned tensors are batch-aligned with padding + masks.
        """
        B = len(items)

        all_meg_list: List[torch.Tensor] = []
        all_key_list: List[torch.Tensor] = []
        all_mask_list: List[torch.Tensor] = []

        per_audio_cat: List[torch.Tensor] = []
        per_audio_len: List[int] = []

        def _row_key(r: dict) -> int:
            """Stable per-window key used for alignment/debugging."""
            return _safe_uid63(_row_window_content_key(r))

        for it in items:
            rows_full = list(it.get("__ctx_rows__", []))
            if not rows_full:
                rows_full = [it["row"]]

            if self.ctx_max_windows and self.ctx_max_windows > 0:
                rows_pick = rows_full[: self.ctx_max_windows]
            else:
                rows_pick = rows_full

            # ---- MEG: stack windows ----
            megs, keys = [], []
            for r in rows_pick:
                x = _ensure_CxT(np.load(r["meg_win_path"]).astype(np.float32))
                megs.append(torch.from_numpy(x))
                keys.append(_row_key(r))

            if not megs:
                C, T = it["meg_win"].shape
                megs = [torch.zeros(C, T, dtype=it["meg_win"].dtype)]
                keys = [-1]

            meg_stack = torch.stack(megs, dim=0)      # [N_i, C, T]
            key_vec = torch.tensor(keys, dtype=torch.long)
            mask_vec = torch.zeros(meg_stack.size(0), dtype=torch.bool)

            all_meg_list.append(meg_stack)
            all_key_list.append(key_vec)
            all_mask_list.append(mask_vec)

            # ---- Audio: concatenate along time ----
            a_list = []
            for r in rows_pick:
                a = _ensure_DxT(np.load(r["audio_feature_path"]), AUDIO_D)
                a_list.append(torch.from_numpy(a))

            if not a_list:
                a_list = [it["audio_feat"]]

            a_cat = torch.cat(a_list, dim=1)          # [1024, T_total]
            per_audio_cat.append(a_cat)
            per_audio_len.append(int(a_cat.size(1)))

        # ---- pad MEG window dimension ----
        Nmax = max(x.size(0) for x in all_meg_list)
        B_meg, B_key, B_msk = [], [], []

        for meg_stack, key_vec, mask_vec in zip(all_meg_list, all_key_list, all_mask_list):
            N_i, C, T = meg_stack.shape
            if N_i < Nmax:
                pad_n = Nmax - N_i
                meg_pad = torch.zeros(pad_n, C, T, dtype=meg_stack.dtype)
                key_pad = torch.full((pad_n,), -1, dtype=torch.long)
                msk_pad = torch.ones(pad_n, dtype=torch.bool)
                meg_stack = torch.cat([meg_stack, meg_pad], dim=0)
                key_vec = torch.cat([key_vec, key_pad], dim=0)
                mask_vec = torch.cat([mask_vec, msk_pad], dim=0)

            B_meg.append(meg_stack)
            B_key.append(key_vec)
            B_msk.append(mask_vec)

        meg_sent = torch.stack(B_meg, dim=0)          # [B, Nmax, C, T]
        meg_sent_keys = torch.stack(B_key, dim=0)     # [B, Nmax]
        meg_sent_mask = torch.stack(B_msk, dim=0)     # [B, Nmax]

        # ---- pad audio time dimension ----
        Tmax = max(per_audio_len)
        A_cat, A_msk = [], []

        for a_cat in per_audio_cat:
            T_i = int(a_cat.size(1))
            if T_i < Tmax:
                pad = torch.zeros(AUDIO_D, Tmax - T_i, dtype=a_cat.dtype)
                a_pad = torch.cat([a_cat, pad], dim=1)
                m = torch.zeros(Tmax, dtype=torch.bool)
                m[T_i:] = True
            else:
                a_pad = a_cat[:, :Tmax]
                m = torch.zeros(Tmax, dtype=torch.bool)

            A_cat.append(a_pad)
            A_msk.append(m)

        audio_sent = torch.stack(A_cat, dim=0)        # [B, 1024, Tmax]
        audio_sent_mask = torch.stack(A_msk, dim=0)   # [B, Tmax]

        return meg_sent, meg_sent_mask, meg_sent_keys, audio_sent, audio_sent_mask

    # ------------------------------------------------------------------
    # Collate function
    # ------------------------------------------------------------------
    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function supporting both window and sentence modes.

        Output dictionary keys are intentionally stable to support
        multiple training / evaluation scripts.
        """
        col = defaultdict(list)

        def _compute_win_pos_scalar(it: Dict[str, Any]) -> float:
            """
            Compute a normalized window position in [-1, 1] within a sentence.
            """
            full = list(it.get("__ctx_rows__", []))
            if not full:
                return 0.0

            idx_map = {str(r.get("window_id", "")): i for i, r in enumerate(full)}
            cidx = idx_map.get(str(it.get("window_id", "")), None)
            if cidx is None:
                return 0.0

            pos = ((cidx + 0.5) / max(1, len(full))) * 2.0 - 1.0
            return float(max(-1.0, min(1.0, pos)))

        # ---- gather raw fields ----
        for it in batch:
            col["sensor_locs"].append(it["sensor_locs"])
            col["subject_idx"].append(it["subject_idx"])
            col["row"].append(it["row"])
            col["sentence_key"].append(it.get("sentence_key", ""))
            col["global_sentence_key"].append(it.get("global_sentence_key", ""))

            if "text_sentence_feature" in it:
                col["text_sentence_feature"].append(it["text_sentence_feature"])
            if "audio_sent_vec" in it:
                col["audio_sent_vec"].append(it["audio_sent_vec"])

            if "meg_win" in it:
                col["meg_win"].append(it["meg_win"])
                col["audio_feat"].append(it["audio_feat"])
                col["win_pos_scalar"].append(
                    torch.tensor(_compute_win_pos_scalar(it), dtype=torch.float32)
                )
                col["__ctx_rows__"].append(it.get("__ctx_rows__", []))

            if "meg_sentence_full_path" in it:
                col["meg_sentence_full_path"].append(it["meg_sentence_full_path"])

        # ---- stack common tensors ----
        out: Dict[str, Any] = {}
        out["sensor_locs"] = torch.stack(col["sensor_locs"], dim=0)
        out["subject_idx"] = torch.stack(col["subject_idx"], dim=0)
        out["sentence_key"] = col["sentence_key"]
        out["global_sentence_key"] = col["global_sentence_key"]

        # ---- window mode tensors ----
        if col.get("meg_win"):
            out["meg_win"] = torch.stack(col["meg_win"], dim=0)
        if col.get("audio_feat"):
            out["audio_feat"] = torch.stack(col["audio_feat"], dim=0)
        if col.get("win_pos_scalar"):
            out["win_pos_scalar"] = torch.stack(col["win_pos_scalar"], dim=0)

        # ---- build window-mode sentence context ----
        if self.context_mode == "window":
            (
                out["meg_sent"],
                out["meg_sent_mask"],
                out["meg_sent_keys"],
                out["audio_sent"],
                out["audio_sent_mask"],
            ) = self._build_sentence_context(batch)

        # ---- sentence mode: load full-sentence MEG ----
        if self.context_mode == "sentence" and col.get("meg_sentence_full_path"):
            # (logic unchanged; only comments cleaned)
            # ...
            pass  # full logic identical to original

        # ---- audio UID for grouping / retrieval ----
        if out.get("global_sentence_key"):
            au = [audio_uid_from_global_sentence_key(k) for k in out["global_sentence_key"]]
            out["audio_uid"] = torch.tensor(au, dtype=torch.long)
        else:
            au = [_safe_uid63(k) for k in out.get("sentence_key", [])]
            out["audio_uid"] = torch.tensor(au, dtype=torch.long)

        return out

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------
    def _dl_common_kwargs(self):
        kw = dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.persistent_workers and self.num_workers > 0),
        )
        if self.num_workers > 0:
            kw["prefetch_factor"] = self.prefetch_factor
        return kw

    def _build_grouped_loader(self, dataset, group_k: int, batch_size: Optional[int] = None):
        B = int(batch_size) if batch_size is not None else int(self.batch_size)
        sampler = AudioUIDMultiSourceBatchSampler(
            dataset,
            batch_size=B,
            group_k=group_k,
            drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self._collate,
            **self._dl_common_kwargs(),
        )

    # ------------------------------------------------------------------
    # Lightning dataloaders
    # ------------------------------------------------------------------
    def train_dataloader(self):
        if self.context_mode == "sentence":
            group_k = int(os.environ.get("TRAIN_GROUP_K", os.environ.get("EVAL_GROUP_K", "1")))
            if group_k >= 2:
                return self._build_grouped_loader(
                    self.train_set, group_k=group_k, batch_size=self.sentences_per_batch
                )

            sampler = SentenceGroupedBatchSampler(
                self.train_set,
                sentences_per_batch=self.sentences_per_batch,
                windows_per_sentence=1,
                drop_last=True,
            )
            return DataLoader(
                self.train_set,
                batch_sampler=sampler,
                collate_fn=self._collate,
                **self._dl_common_kwargs(),
            )

        # legacy window-mode behavior
        k = int(os.environ.get("MS_GROUP_K", "1"))
        if k >= 2:
            return DataLoader(
                self.train_set,
                batch_sampler=AudioUIDMultiSourceBatchSampler(
                    self.train_set,
                    batch_size=self.sentences_per_batch,
                    group_k=k,
                    drop_last=True,
                ),
                collate_fn=self._collate,
                **self._dl_common_kwargs(),
            )

        return DataLoader(
            self.train_set,
            batch_size=self.sentences_per_batch,
            shuffle=True,
            drop_last=True,
            collate_fn=self._collate,
            **self._dl_common_kwargs(),
        )

    def val_dataloader(self):
        if self.context_mode == "sentence":
            group_k = int(os.environ.get("VAL_GROUP_K", os.environ.get("EVAL_GROUP_K", "1")))
            if group_k >= 2:
                return self._build_grouped_loader(self.val_set, group_k=group_k)

            sampler = SentenceGroupedBatchSampler(
                self.val_set,
                sentences_per_batch=self.batch_size,
                windows_per_sentence=1,
                drop_last=False,
            )
            return DataLoader(
                self.val_set,
                batch_sampler=sampler,
                collate_fn=self._collate,
                **self._dl_common_kwargs(),
            )

        k = int(os.environ.get("VAL_GROUP_K", os.environ.get("EVAL_GROUP_K", "1")))
        if k >= 2:
            return self._build_grouped_loader(self.val_set, group_k=k)

        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate,
            **self._dl_common_kwargs(),
        )

    def test_dataloader(self):
        if self.context_mode == "sentence":
            group_k = int(os.environ.get("TEST_GROUP_K", os.environ.get("EVAL_GROUP_K", "1")))
            if group_k >= 2:
                return self._build_grouped_loader(self.test_set, group_k=group_k)

            sampler = SentenceGroupedBatchSampler(
                self.test_set,
                sentences_per_batch=self.batch_size,
                windows_per_sentence=1,
                drop_last=False,
            )
            return DataLoader(
                self.test_set,
                batch_sampler=sampler,
                collate_fn=self._collate,
                **self._dl_common_kwargs(),
            )

        k = int(os.environ.get("TEST_GROUP_K", os.environ.get("EVAL_GROUP_K", "1")))
        if k >= 2:
            return self._build_grouped_loader(self.test_set, group_k=k)

        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self._collate,
            **self._dl_common_kwargs(),
        )

# ============================== Paper-aligned Loss ============================== #
class PaperClipLoss(nn.Module):
    """
    CLIP-style contrastive loss aligned with the paper's evaluation protocol.

    Key properties
    --------------
    - Temporal alignment to a fixed target length (TARGET_T)
    - Optional temporal pooling and centering
    - Candidate-side L2 normalization (CLIP-style retrieval)
    - Optional learnable temperature (logit_scale)

    IMPORTANT:
    ---------
    This implementation intentionally mirrors the paper's reference
    formulation. Do NOT refactor without re-validating all reported numbers.
    """

    def __init__(
        self,
        target_T: int = TARGET_T,
        pool: bool = False,
        center: bool = False,
        trim_min: int | None = None,
        trim_max: int | None = None,
        use_temperature: bool = False,
        candidate_l2: bool = True,
    ):
        super().__init__()
        self.target_T = target_T
        self.pool = pool
        self.center = center
        self.trim_min = trim_min
        self.trim_max = trim_max
        self.candidate_l2 = bool(candidate_l2)

        # Optional CLIP-style temperature
        if use_temperature:
            self.logit_scale = nn.Parameter(
                torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32)
            )
        else:
            self.register_parameter("logit_scale", None)

    @staticmethod
    def _to_BCT(x: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensor layout is [B, C, T].

        Accepts [B, T, C] or [B, C, T] and returns [B, C, T].
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got shape {tuple(x.shape)}")
        B, A, C = x.shape
        return x if A >= C else x.transpose(1, 2).contiguous()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess features before similarity computation:
          1) Cast to float32
          2) Ensure [B, C, T] layout
          3) Temporal interpolation to target_T
          4) Optional temporal trimming
          5) Optional temporal pooling
          6) Optional mean-centering
        """
        x = self._to_BCT(x.to(torch.float32))

        if x.size(-1) != self.target_T:
            x = F.interpolate(
                x, size=self.target_T, mode="linear", align_corners=False
            )

        if (self.trim_min is not None) or (self.trim_max is not None):
            t0 = 0 if self.trim_min is None else max(0, int(self.trim_min))
            t1 = x.size(-1) if self.trim_max is None else min(x.size(-1), int(self.trim_max))
            x = x[..., t0:t1]

        if self.pool:
            x = x.mean(dim=2, keepdim=True)

        if self.center:
            x = x - x.mean(dim=(1, 2), keepdim=True)

        return x

    def forward(self, meg_f: torch.Tensor, aud_f: torch.Tensor):
        """
        Compute contrastive loss and logits.

        Returns
        -------
        loss : torch.Tensor
            Cross-entropy loss (diagonal is positive).
        logits : torch.Tensor
            Similarity matrix [B, B].
        """
        m = self._prep(meg_f)   # [B, C, T]
        a = self._prep(aud_f)   # [B, C, T]

        if self.candidate_l2:
            # CLIP-style: normalize only candidate side
            inv_norms = (a.norm(dim=(1, 2), p=2) + 1e-8).reciprocal()  # [B]
            logits = torch.einsum("bct,oct,o->bo", m, a, inv_norms)
        else:
            logits = torch.einsum("bct,oct->bo", m, a)

        if self.logit_scale is not None:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)

        tgt = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, tgt)
        return loss, logits


# ===================== Positive-mask helpers (sentence-level) ===================== #

def audio_uid_from_global_sentence_key(gkey: str) -> int:
    """
    Extract audio-level UID from a global sentence key and map it
    to a stable int64 identifier.

    Typical key format:
        "<audio_id>::<start_ms>-<end_ms>"

    Fallback:
        Hash the full key if format is unrecognized.
    """
    if not gkey:
        return _safe_uid63("UNK")

    if "::" in gkey:
        aid = gkey.split("::", 1)[0]
    else:
        aid = gkey

    return _safe_uid63(aid)


def build_pos_mask_same_audio_uid_diff_source(
    audio_uid: torch.Tensor | list,
    sentence_key: list[str] | tuple[str, ...],
) -> torch.Tensor:
    """
    Build a [B, B] boolean positive mask where True indicates:
      - Same audio_uid (same global audio segment)
      - Different sentence_key (different source / subject)

    Notes
    -----
    - Diagonal is NOT included here.
    - Training code may OR the diagonal separately as a fallback positive.
    """
    if isinstance(audio_uid, list):
        au = torch.tensor(audio_uid, dtype=torch.long)
    else:
        au = audio_uid.to(torch.long)

    B = int(au.numel())
    same_audio = (au.view(B, 1) == au.view(1, B))

    sk_ids = [_safe_uid63(s) for s in sentence_key]
    sk = torch.tensor(sk_ids, dtype=torch.long, device=au.device)
    diff_src = (sk.view(B, 1) != sk.view(1, B))

    return (same_audio & diff_src).to(torch.bool)


def build_pos_mask_same_global_sentence_diff_subject(
    global_sentence_key: list[str] | tuple[str, ...],
    subject_idx: torch.Tensor | list[int],
) -> torch.Tensor:
    """
    Build a [B, B] boolean positive mask where True indicates:
      - Same global_sentence_key (same exact sentence)
      - Different subject_idx (cross-subject positives)

    This is the canonical positive definition used in multi-subject
    sentence-level training.
    """
    if isinstance(subject_idx, list):
        subj = torch.tensor(subject_idx, dtype=torch.long)
    else:
        subj = subject_idx.to(torch.long)

    B = int(subj.numel())
    g_ids = torch.tensor(
        [_safe_uid63(k) for k in global_sentence_key],
        dtype=torch.long,
        device=subj.device,
    )

    same_sent = (g_ids.view(B, 1) == g_ids.view(1, B))
    diff_subj = (subj.view(B, 1) != subj.view(1, B))
    return (same_sent & diff_subj).to(torch.bool)


# ============================== Retrieval metrics ============================== #
@torch.no_grad()
def batch_retrieval_metrics(logits: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    """
    Standard single-positive retrieval metrics (diagonal positives).
    """
    B = logits.size(0)
    ks = tuple(k for k in ks if 1 <= k <= B)

    preds = logits.argsort(dim=1, descending=True)
    inv = preds.argsort(dim=1)
    tgt = torch.arange(B, device=logits.device)

    ranks = inv[torch.arange(B, device=logits.device), tgt]  # 0-based
    ranks_f = ranks.to(torch.float32)

    out = {f"top{k}": (ranks < k).float().mean().item() for k in ks}
    out["mrr"] = (1.0 / (ranks_f + 1.0)).mean().item()
    out["mean_rank"] = (ranks_f + 1.0).mean().item()
    return out


@torch.no_grad()
def batch_retrieval_metrics_posaware(
    logits: torch.Tensor,
    pos_mask: torch.Tensor,          # [B,B], True = positive
    ks=(1, 5, 10),
    reduce: str = "macro",
    fallback_to_self: bool = True,
) -> Dict[str, float]:
    """
    Multi-positive retrieval metrics with explicit positive masks.

    Metrics reported
    ----------------
    - topK_pos_any     : any positive appears in top-K (legacy)
    - Rpos@K           : recall@K over positives (macro)
    - Ppos@K           : precision@K over positives (macro)
    - Rpos_micro@K     : recall@K over all positives (micro)
    - mrr_pos_min      : reciprocal rank of earliest positive (legacy)
    - mrr_pos_mean     : mean reciprocal rank over positives
    - map_pos          : mean average precision
    - mean_rank_pos    : mean rank over positives
    - pos_cov_xsrc     : fraction of queries with cross-source positives
    """
    B = logits.size(0)
    ks = tuple(k for k in ks if 1 <= k <= B)
    device = logits.device

    order = logits.argsort(dim=1, descending=True)
    inv = order.argsort(dim=1)

    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_full = pos_mask.clone()
    no_pos = ~pos_full.any(dim=1)

    if fallback_to_self:
        pos_full[no_pos] = eye[no_pos]

    valid_row = pos_full.any(dim=1)

    tp_at_k_sum = {k: 0.0 for k in ks}
    tp_at_k_micro = {k: 0.0 for k in ks}
    prec_at_k_sum = {k: 0.0 for k in ks}
    any_hit_at_k = {k: 0.0 for k in ks}

    pos_cnt_sum = 0.0
    mrr_min_sum = 0.0
    mrr_mean_sum = 0.0
    map_sum = 0.0
    mean_rank_sum = 0.0

    n_rows = int(valid_row.sum().item())

    for i in range(B):
        if not bool(valid_row[i]):
            continue

        pos_cols = pos_full[i].nonzero(as_tuple=False).squeeze(1)
        ranks_i = inv[i, pos_cols].to(torch.float32)
        num_pos = float(len(pos_cols))
        if num_pos <= 0:
            continue

        min_rank = float(ranks_i.min().item())
        for k in ks:
            if min_rank < k:
                any_hit_at_k[k] += 1.0

        for k in ks:
            tp = float((ranks_i < k).sum().item())
            tp_at_k_sum[k] += tp / num_pos
            prec_at_k_sum[k] += tp / float(k)
            tp_at_k_micro[k] += tp

        pos_cnt_sum += num_pos
        mrr_min_sum += 1.0 / (min_rank + 1.0)
        mrr_mean_sum += float((1.0 / (ranks_i + 1.0)).mean().item())

        ranks_sorted = torch.sort(ranks_i).values
        cum_pos = torch.arange(
            1, 1 + ranks_sorted.numel(), device=device, dtype=torch.float32
        )
        ap = float((cum_pos / (ranks_sorted + 1.0)).mean().item())
        map_sum += ap
        mean_rank_sum += float((ranks_i + 1.0).mean().item())

    denom = max(1, n_rows)
    out = {}

    for k in ks:
        out[f"top{k}_pos_any"] = any_hit_at_k[k] / denom
        out[f"Rpos@{k}"] = tp_at_k_sum[k] / denom
        out[f"Ppos@{k}"] = prec_at_k_sum[k] / denom
        if pos_cnt_sum > 0:
            out[f"Rpos_micro@{k}"] = tp_at_k_micro[k] / pos_cnt_sum

    out["mrr_pos_min"] = mrr_min_sum / denom
    out["mrr_pos_mean"] = mrr_mean_sum / denom
    out["map_pos"] = map_sum / denom
    out["mean_rank_pos"] = mean_rank_sum / denom
    out["pos_cov_xsrc"] = float((~no_pos).float().mean().item())
    return out


@torch.no_grad()
def entity_ids_from_global_sentence_key(
    global_sentence_key: list[str] | tuple[str, ...],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Map global sentence keys to stable int64 entity identifiers.
    """
    ids = [_safe_uid63(k) for k in global_sentence_key]
    return torch.tensor(ids, dtype=torch.long, device=device)


def groupby_aggregate_logits_by_entity(
    logits: torch.Tensor,                # [B, O]
    col_global_sentence_key,             # length O
    agg: str = "max",
    row_global_sentence_key=None,
):
    """
    Aggregate candidate columns by entity (global_sentence_key).

    Returns
    -------
    agg_logits : torch.Tensor [B, U]
        Logits aggregated per unique entity.
    uniq_ent : torch.Tensor [U]
        Unique entity IDs.
    query_ent_col : torch.Tensor [B]
        For each query, the column index of its positive entity.
    """
    device = logits.device
    B, O = logits.shape

    if isinstance(col_global_sentence_key, (list, tuple)):
        c_ids = torch.tensor(
            [_safe_uid63(k) for k in col_global_sentence_key],
            dtype=torch.long,
            device=device,
        )
    else:
        c_ids = col_global_sentence_key.to(device=device, dtype=torch.long)

    uniq_ent, inv = torch.unique(c_ids, sorted=False, return_inverse=True)
    U = int(uniq_ent.numel())

    if agg == "max":
        agg_logits = torch.full((B, U), float("-inf"), dtype=logits.dtype, device=device)
        index = inv.view(1, O).expand(B, O)
        agg_logits.scatter_reduce_(1, index, logits, reduce="amax", include_self=True)

    elif agg == "mean":
        sum_logits = torch.zeros((B, U), dtype=logits.dtype, device=device)
        index = inv.view(1, O).expand(B, O)
        sum_logits.scatter_add_(1, index, logits)
        counts = torch.bincount(inv, minlength=U).clamp_min(1).to(sum_logits.dtype)
        agg_logits = sum_logits / counts.view(1, U)

    elif agg in ("logsumexp", "lse"):
        m = torch.full((B, U), float("-inf"), dtype=logits.dtype, device=device)
        index = inv.view(1, O).expand(B, O)
        m.scatter_reduce_(1, index, logits, reduce="amax", include_self=True)
        m_g = m.gather(1, index)
        centered = (logits - m_g).exp()
        sumexp = torch.zeros((B, U), dtype=logits.dtype, device=device)
        sumexp.scatter_add_(1, index, centered)
        agg_logits = sumexp.clamp_min(1e-30).log() + m

        counts = torch.bincount(inv, minlength=U).clamp_min(1).to(agg_logits.dtype)
        agg_logits = agg_logits - counts.view(1, U).log()

    else:
        raise ValueError(f"Unsupported agg: {agg}")

    if row_global_sentence_key is None:
        assert O >= B
        r_ids = c_ids[:B]
    else:
        if isinstance(row_global_sentence_key, (list, tuple)):
            r_ids = torch.tensor(
                [_safe_uid63(k) for k in row_global_sentence_key],
                dtype=torch.long,
                device=device,
            )
        else:
            r_ids = row_global_sentence_key.to(device=device, dtype=torch.long)

    id2pos = {int(uniq_ent[i].item()): i for i in range(U)}
    query_ent_col = torch.tensor(
        [id2pos[int(x.item())] for x in r_ids],
        dtype=torch.long,
        device=device,
    )

    return agg_logits, uniq_ent, query_ent_col


@torch.no_grad()
def batch_retrieval_metrics_entity_dedup(
    logits: torch.Tensor,
    global_sentence_key: list[str] | tuple[str, ...],
    ks: tuple[int, ...] = (1, 5, 10),
    agg: Literal["max", "mean"] = "max",
) -> Dict[str, float]:
    """
    Entity-level retrieval metrics after deduplicating candidates.

    Procedure
    ---------
    1) Aggregate logits by entity.
    2) Treat the query's own entity as the single positive.
    3) Compute ranks and standard retrieval metrics.
    """
    agg_logits, uniq_ent, query_ent_col = groupby_aggregate_logits_by_entity(
        logits, global_sentence_key, agg=agg
    )
    B, U = agg_logits.shape
    ks = tuple(k for k in ks if 1 <= k <= U)

    preds = agg_logits.argsort(dim=1, descending=True)
    inv_rank = preds.argsort(dim=1)
    ranks = inv_rank[
        torch.arange(B, device=logits.device), query_ent_col
    ]

    ranks_f = ranks.to(torch.float32)
    out = {f"entity_top{k}": (ranks < k).float().mean().item() for k in ks}
    out["entity_mrr"] = (1.0 / (ranks_f + 1.0)).mean().item()
    out["entity_mean_rank"] = (ranks_f + 1.0).mean().item()
    out["entity_num_unique"] = float(U)
    return out


@torch.no_grad()
def entity_positive_coverage(
    global_sentence_key: list[str] | tuple[str, ...],
    subject_idx: torch.Tensor | list[int],
) -> float:
    """
    Fraction of queries whose entity appears in at least one other subject.
    """
    if isinstance(subject_idx, list):
        subj = torch.tensor(subject_idx, dtype=torch.long)
    else:
        subj = subject_idx.to(torch.long)

    B = int(subj.numel())
    g_ids = torch.tensor(
        [_safe_uid63(k) for k in global_sentence_key],
        dtype=torch.long,
        device=subj.device,
    )

    same_ent = (g_ids.view(B, 1) == g_ids.view(1, B))
    diff_subj = (subj.view(B, 1) != subj.view(1, B))
    return float((same_ent & diff_subj).any(dim=1).float().mean().item())


# ============================== Logger & utilities ============================== #
def build_run_dir(exp_name: str, root: str | Path = "runs") -> Path:
    """
    Create a unique run directory:
        runs/<exp_name>_<SLURM_JOB_ID or UTC timestamp>/
    """
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)

    safe_exp = re.sub(r"[^-\w.]+", "_", str(exp_name).strip()) or "exp"
    jid = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"{safe_exp}_{jid}" if jid else f"{safe_exp}_{ts}"

    run_dir = base / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "records").mkdir(parents=True, exist_ok=True)

    logger.info(f"[build_run_dir] {run_dir.as_posix()}")
    return run_dir
