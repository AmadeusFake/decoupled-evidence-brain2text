#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train/meg_utils.py  —  Global(sentence) 加速版 + 键策略（text/audio）自动/可控
================================================
改动摘要（仅在 context_mode="sentence" 时生效）：
1) FAST I/O：Dataset 在 sentence 模式下不再读取每个“锚窗”的 meg_win/audio_feat，
   仅加载：sensor coords、subject_idx、整句 MEG 路径、句向量（Whisper→E5 文本或 BCTR25 音频）。
2) 进程内 LRU 缓存：整句 MEG .npy 通过 _sent_load_cached() 复用（每个 DataLoader worker 各自一份）。
3) collate_fn：sentence 分支只按整句拼批并对齐；meg_win/audio_feat 字段在 sentence 模式下不再必需。
4) 采样：支持两种 sentence 模式采样：
   - 默认：SentenceGroupedBatchSampler（一个 batch = 若干句，每句取一个代表索引）。
   - 分组：AudioUIDMultiSourceBatchSampler（env: TRAIN_GROUP_K>=2 时），保证同句跨被试至少 K。
5) 新增“全局句键”策略：
   - 自动根据 manifest 路径判别优先从“文本句向量路径（text_whisper）”或“音频句向量路径（sentence_full_audio/BCTR25）”解析键；
   - 也可在 Dataset 初始化时手动指定 key_mode="text"/"audio"/"auto"。
   键格式统一归一到：<audio_id>::<start_ms>-<end_ms>（毫秒），跨被试稳定。
6) teacher 句向量选择：当 key_mode="audio" 时优先用音频句向量；key_mode="text" 时优先用文本句向量；均可回退。
7) ★ 新增正样本掩码（pos-mask）函数：
   build_pos_mask_same_global_sentence_diff_subject(global_sentence_key, subject_idx)
   —— “同句且跨被试”为正；训练脚本可直接使用它来替换旧的基于 audio_uid 的正样本逻辑。

注意：window 模式（local 流复现）逻辑完全未改。
"""

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

# ===== constants =====
TARGET_T = 360     # 3s @ 120Hz（单窗口时间步）
AUDIO_D  = 1024    # 预提取音频表征维度（如 1024-d conv/W2V 投影）

# 老格式（毫秒）： "...::<start_ms>-<end_ms>"
_GLOBAL_SENT_RE_MS = re.compile(r'([^/\\]+)::(\d+)-(\d+)', re.U)

# 新格式（秒 + 下划线）： ".../<audio_id>_<start_s>_<end_s>_sent*.npy"
# 兼容 sentAUDIO/sentTEXT/E5 等后缀
_AUDIO_SENT_RE_S = re.compile(
    r'[\\/](?P<aid>[^/\\]+?)_(?P<s>\-?\d+(?:\.\d+)?)_(?P<e>\-?\d+(?:\.\d+)?)(?:_sent[A-Za-z0-9_.-]*)?\.npy$',
    re.U
)


# ======================= 句子全局唯一键（跨被试一致） ======================= #
def _ms_from_seconds_str(x: str) -> int:
    return int(round(float(x) * 1000.0))


def _norm_audio_id(x: str) -> str:
    return Path(x).stem


def _key_from_feature_path(path: str) -> Optional[str]:
    """
    从句向量路径（文本或音频）解析为统一键：<aid>::<start_ms>-<end_ms>
    兼容两类命名：
      A) 新：<aid>_<start_s>_<end_s>_sent*.npy （秒）
      B) 旧：<aid>::<start_ms>-<end_ms>      （毫秒）
    """
    if not path:
        return None
    s = str(path)
    m = _AUDIO_SENT_RE_S.search(s)
    if m:
        aid = _norm_audio_id(m.group("aid"))
        s_ms = _ms_from_seconds_str(m.group("s"))
        e_ms = _ms_from_seconds_str(m.group("e"))
        return f"{aid}::{s_ms}-{e_ms}"
    m2 = _GLOBAL_SENT_RE_MS.search(s)
    if m2:
        aid = _norm_audio_id(m2.group(1))
        s_ms = int(m2.group(2))
        e_ms = int(m2.group(3))
        return f"{aid}::{s_ms}-{e_ms}"
    return None


def _infer_key_mode_from_manifest(manifest_path: Path) -> Literal["audio", "text", "auto"]:
    """
    根据 manifest 路径推断句键优先策略：
      - 包含 "final_splits_sentence_with_sentence_full_text_whisper" → text
      - 包含 "final_splits_word_list_with_sentence_full_audio"      → audio
      - 其它 → auto
    """
    s = str(manifest_path)
    if "final_splits_sentence_with_sentence_full_text_whisper" in s:
        return "text"
    if "final_splits_word_list_with_sentence_full_audio" in s:
        return "audio"
    # 兜底一些宽松别名
    if "text_whisper" in s and "sentence_full" in s:
        return "text"
    if "sentence_full_audio" in s or "BCTR25" in s:
        return "audio"
    return "auto"


def derive_global_sentence_key(row: dict, prefer: Literal["audio", "text", "auto"] = "auto") -> str:
    """
    生成“句子唯一键”，优先保证跨被试一致。
    解析优先级受 `prefer` 影响：
      prefer="audio"：优先从 audio_sentence_feature_path 解析；
      prefer="text" ：优先从 text_sentence_feature_path  解析；
      prefer="auto" ：谁能解析出来就用谁（先 text 再 audio）。
    解析失败再退化：
      1) 用显式 global_segment_on/off_in_audio_s（秒）→ 毫秒
      2) <audio_base>::SID@<sentence_id>
      3) 哈希兜底
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

    # 2) 用显式全局起止（秒转毫秒）
    audio_base = _norm_audio_id(row.get("original_audio_path", "") or "")
    on = row.get("global_segment_onset_in_audio_s", None)
    off = row.get("global_segment_offset_in_audio_s", None)
    if (on is not None) and (off is not None) and audio_base:
        s_ms = int(round(float(on) * 1000.0))
        e_ms = int(round(float(off) * 1000.0))
        return f"{audio_base}::{s_ms}-{e_ms}"

    # 3) 用 SID 兜底（不一定跨被试稳定，但不至于崩）
    sid = str(row.get("sentence_id", "") or "")
    if audio_base and sid:
        return f"{audio_base}::SID@{sid}"
    if sid:
        return f"SID@{sid}"

    # 4) 最后一层兜底：对整行做短哈希
    h = hashlib.blake2b(
        json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"),
        digest_size=8
    ).hexdigest()
    return f"UNK::{h}"


# ============================== Subject Registry ============================== #
class SubjectRegistry:
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
        k = self._key(ns, sid)
        if k in self._map:
            return self._map[k]
        if not self._map:
            raise RuntimeError("SubjectRegistry is empty; cannot map subject.")
        if fallback_to_first:
            return 0
        raise KeyError(f"Unknown subject key: {k}")

    def to_json(self) -> Dict[str, Any]:
        return {"mapping": self._map, "order": [k for k, _ in sorted(self._map.items(), key=lambda kv: kv[1])]}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "SubjectRegistry":
        mp = d.get("mapping", {})
        return SubjectRegistry(mapping=mp)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(path: Path) -> "SubjectRegistry":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return SubjectRegistry.from_json(d)

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
    def build_from_manifests(cls, files_with_ns: Iterable[Tuple[Path, str]]) -> "SubjectRegistry":
        entries: List[str] = []
        for p, ns in files_with_ns:
            if not p or not p.exists():
                continue
            subs = cls._collect_subjects_from_manifest(p)
            for sid in subs:
                entries.append(cls._key(ns, sid))
        entries = sorted(set(entries))
        mapping = {k: i for i, k in enumerate(entries)}
        return SubjectRegistry(mapping=mapping)


# ============================== Data helpers ============================== #
def _ensure_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expect 2D [C,T] or [T,C], got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T


def _ensure_DxT(x: np.ndarray, D: int) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D audio array, got {x.shape}")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    x = np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[0] == D:
        return x
    if x.shape[1] == D:
        return x.T
    return x if abs(x.shape[0] - D) < abs(x.shape[1] - D) else x.T


def _safe_uid63(uid_src) -> int:
    """将任意可哈希对象稳定映射为 int64。"""
    try:
        v = int(uid_src)
        if v >  0x7FFFFFFFFFFFFFFF: v = v & 0x7FFFFFFFFFFFFFFF
        if v < -0x8000000000000000: v = -0x8000000000000000
        return v
    except Exception:
        h = hashlib.blake2b(str(uid_src).encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "little", signed=False) & 0x7FFFFFFFFFFFFFFF
        return v


def _row_window_content_key(row: dict) -> str:
    """窗口级唯一标识（跨受试者相同，用于 window 模式上下文拼接）"""
    a = row.get("original_audio_path", "")
    on = row.get("local_window_onset_in_audio_s", None)
    off= row.get("local_window_offset_in_audio_s", None)
    if on is not None and off is not None:
        return f"{a}::{float(on):.3f}-{float(off):.3f}"
    return f"{a}::WID@{row.get('window_id','')}"


# ---- tiny in-process LRU for full-sentence MEG (per worker) ----
_SENT_CACHE: dict[str, np.ndarray] = {}
_SENT_ORDER: list[str] = []
# 从环境变量读取，默认 512；可用 SENT_CACHE_CAP=256/512/1024 动态调节
import os as _os
_SENT_CAP = int(_os.environ.get("SENT_CACHE_CAP", "512"))

def _sent_load_cached(path: str) -> np.ndarray:
    """
    读取整句 MEG .npy -> 返回 float32、C×T、连续内存的普通 ndarray（非 memmap）。
    采用 memmap 打开以获得 I/O 吞吐，但立刻复制为常规数组并删除 memmap，保证不占用文件描述符。
    结果放入进程内 LRU 缓存，后续复用不会再次打开文件。
    """
    # 命中缓存：更新为最新使用
    arr = _SENT_CACHE.get(path)
    if arr is not None:
        try:
            _SENT_ORDER.remove(path)
        except ValueError:
            pass
        _SENT_ORDER.append(path)
        return arr

    # 用 memmap 打开 → 立刻复制为普通 ndarray 并关闭 FD
    mm = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        arr = np.array(mm, dtype=np.float32, copy=True)  # ★ 强制复制，memmap FD 可被释放
    finally:
        del mm  # ★ 关键：删除 memmap 对象，释放文件描述符

    # 统一为 C×T、连续内存
    arr = _ensure_CxT(arr)
    arr = np.ascontiguousarray(arr, dtype=np.float32)

    # 写入 LRU 缓存
    _SENT_CACHE[path] = arr
    _SENT_ORDER.append(path)
    if len(_SENT_ORDER) > _SENT_CAP:
        old = _SENT_ORDER.pop(0)
        _SENT_CACHE.pop(old, None)

    return arr


# ============================== Dataset ============================== #
class MEGDataset(Dataset):
    """
    关键字段（__getitem__ 输出）：
    - sentence 模式（FAST I/O 开）：只返回 sensor_locs / subject_idx / 全局句键 / 整句路径 / teacher 向量等
      —— 不再读取 meg_win/audio_feat。
    - window 模式：与旧版一致。
    """
    def __init__(self, manifest_path: str, registry: SubjectRegistry, namespace: str,
                 normalize: bool = False, context_mode: str = "none",
                 sentence_fast_io: bool = True,
                 key_mode: Literal["auto", "audio", "text"] | None = None):
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

        # 键策略：优先从 manifest 路径中推断；可显式覆盖
        inferred = _infer_key_mode_from_manifest(self.manifest_path)
        self.key_mode: Literal["auto", "audio", "text"] = key_mode or inferred

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

        before = len(self.rows)
        self.rows = [s for s in self.rows if Path(s.get("sensor_coordinates_path", "")).exists()]
        drop = before - len(self.rows)
        if drop > 0:
            logger.warning(f"{self.manifest_path.name}: dropped {drop} rows without sensor_coordinates_path")

        # 唯一句键：{ns}:{subject}|{sentence_id} —— 仅用于“句内”上下文组织/采样
        self.sent2rows: Dict[str, List[dict]] = defaultdict(list)
        self.sent2idx: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(self.rows):
            sid = str(r.get("sentence_id", ""))
            subj = str(r.get("subject_id", ""))
            if sid and subj:
                key = f"{self.namespace}:{subj}|{sid}"
                self.sent2rows[key].append(r)
                self.sent2idx[key].append(i)
        # 按时间排序
        for key, lst in self.sent2rows.items():
            lst.sort(
                key=lambda x: (
                    int(x.get("anchor_word_idx", 10**9)),
                    float(x.get("local_window_onset_in_audio_s", 0.0)),
                )
            )

        self._coords_cache: Dict[str, np.ndarray] = {}

        seen = sorted({s.get("subject_id") for s in self.rows if s.get("subject_id") is not None})
        logger.info(f"{self.manifest_path.name}: {len(self.rows):,} samples; subjects={len(seen)}; "
                    f"context_mode={self.context_mode}; fast_io={self.sentence_fast_io}; key_mode={self.key_mode}")

    def __len__(self):
        return len(self.rows)

    def _load_coords(self, coord_path: str) -> np.ndarray:
        if coord_path in self._coords_cache:
            return self._coords_cache[coord_path]
        arr = np.load(coord_path).astype(np.float32)
        self._coords_cache[coord_path] = arr
        return arr

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]

        # ---------- FAST PATH for sentence-mode ----------
        if self.context_mode == "sentence" and self.sentence_fast_io:
            coords = self._load_coords(r["sensor_coordinates_path"])
            subj_str = str(r.get("subject_id", ""))
            subj_idx = self.registry.index_of(self.namespace, subj_str, fallback_to_first=True)

            sent_id  = str(r.get("sentence_id", ""))
            sent_key = f"{self.namespace}:{subj_str}|{sent_id}" if (subj_str and sent_id) else sent_id
            gkey     = derive_global_sentence_key(r, prefer=self.key_mode)

            item = {
                "sensor_locs": torch.from_numpy(coords),           # [C,3]
                "subject_idx": torch.tensor(subj_idx, dtype=torch.long),
                "sentence_id": sent_id,
                "sentence_key": sent_key,
                "global_sentence_key": gkey,
                "row": r,
            }

            # 整句 MEG（优先使用）
            if "meg_sentence_full_path" in r:
                item["meg_sentence_full_path"] = r.get("meg_sentence_full_path", "")

            # teacher 句 tokens：按 key_mode 优先顺序选择，另一种作为回退
            tvec_path = r.get("text_sentence_feature_path", "") or ""
            avec_path = r.get("audio_sentence_feature_path", "") or ""

            def _load_tokens2d(p: str) -> Optional[torch.Tensor]:
                if not p or not Path(p).exists():
                    return None
                X = np.load(p, mmap_mode="r")
                if X.ndim != 2:
                    raise ValueError(f"expect 2D sentence tokens, got {X.shape} for {p}")
                X = X.astype(np.float32, copy=False)
                # 每 token L2 归一化，更稳
                n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
                X = X / n
                return torch.from_numpy(X)  # [L, Dt]

            tok_t = _load_tokens2d(tvec_path)  # 文本句 tokens [Lt,Dt] 或 None
            tok_a = _load_tokens2d(avec_path)  # 音频句 tokens [La,Dt] 或 None

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
            else:  # auto（默认偏文本以保持兼容）
                if tok_t is not None:
                    item["text_sentence_feature"] = tok_t
                elif tok_a is not None:
                    item["audio_sent_vec"] = tok_a

            return item
        # ---------- /FAST PATH ----------

        # --- 原慢路径（window 模式 / fallback） ---
        # MEG window
        meg = _ensure_CxT(np.load(r["meg_win_path"]).astype(np.float32))
        if self.normalize:
            m = meg.mean(axis=1, keepdims=True)
            sd = meg.std(axis=1, keepdims=True) + 1e-6
            meg = (meg - m) / sd

        # audio window features
        aud = _ensure_DxT(np.load(r["audio_feature_path"]), AUDIO_D)

        # sensor coordinates
        coords = self._load_coords(r["sensor_coordinates_path"])

        # subject index
        subj_str = str(r.get("subject_id"))
        subj_idx = self.registry.index_of(self.namespace, subj_str, fallback_to_first=True)

        # 句内 key & 行集合
        sent_id = str(r.get("sentence_id", ""))
        sent_key = f"{self.namespace}:{subj_str}|{sent_id}" if (subj_str and sent_id) else sent_id
        ctx_rows = self.sent2rows.get(sent_key, [])

        # 全局句键
        gkey = derive_global_sentence_key(r, prefer=self.key_mode)

        item = {
            "meg_win": torch.from_numpy(meg),         # [C,T]
            "audio_feat": torch.from_numpy(aud),      # [D,T]
            "sensor_locs": torch.from_numpy(coords),  # [C,3]
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

        # 句 tokens：按 key_mode 优先
        tvec_path = r.get("text_sentence_feature_path", "") or ""
        avec_path = r.get("audio_sentence_feature_path", "") or ""

        def _load_tokens2d(p: str) -> Optional[torch.Tensor]:
            if not p or not Path(p).exists():
                return None
            X = np.load(p, mmap_mode="r")
            if X.ndim != 2:
                raise ValueError(f"expect 2D sentence tokens, got {X.shape} for {p}")
            X = X.astype(np.float32, copy=False)
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X = X / n
            return torch.from_numpy(X)  # [L, Dt]

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
        else:  # auto
            if tok_t is not None:
                item["text_sentence_feature"] = tok_t
            elif tok_a is not None:
                item["audio_sent_vec"] = tok_a

        return item



# ========= BatchSampler（按句分组） ========= #
class SentenceGroupedBatchSampler(Sampler[List[int]]):
    """按句（被试内）分组采一个窗口索引作为“一句”的代表。用于 sentence / window 模式。"""
    def __init__(self, dataset: MEGDataset, sentences_per_batch: int, windows_per_sentence: int = 1, drop_last: bool = True):
        self.ds = dataset
        self.S = int(sentences_per_batch)
        if int(windows_per_sentence) > 1:
            logger.warning("SentenceGroupedBatchSampler: windows_per_sentence>1 会产生重复样本，已强制为 1。")
        self.drop_last = bool(drop_last)
        self.sids = [sid for sid, idxs in self.ds.sent2idx.items() if len(idxs) > 0]

    def __iter__(self):
        # 使用全局 RNG（由 PyTorch/Lightning 负责播种），避免每个 epoch 完全相同
        perm = torch.randperm(len(self.sids)).tolist()
        for i in range(0, len(perm), self.S):
            sid_chunk = [self.sids[j] for j in perm[i:i + self.S]]
            if len(sid_chunk) < self.S and self.drop_last:
                break
            batch_indices: List[int] = []
            for sid in sid_chunk:
                pool = self.ds.sent2idx[sid]
                j = int(torch.randint(low=0, high=len(pool), size=(1,)).item())
                batch_indices.append(pool[j])
            yield batch_indices

    def __len__(self):
        num_batches = len(self.sids) // self.S
        return num_batches if self.drop_last else math.ceil(len(self.sids) / self.S)

class AudioUIDMultiSourceBatchSampler(Sampler[List[int]]):
    """
    每个 batch 里优先放入若干组“同 global_sentence_key、不同来源(被试) ”的正组；
    同时预留一部分槽位给随机样本以增加负样本多样性（由 RESERVE_NEG_FRAC 控制）。
    """
    def __init__(self, dataset: "MEGDataset", batch_size: int, group_k: int = 2, drop_last: bool = True):
        self.ds = dataset
        self.B = int(batch_size)
        self.K = max(2, int(group_k))
        self.drop_last = bool(drop_last)

        key2src2idx: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        ns = getattr(self.ds, "namespace", "")

        def _sentkey_of_row(r: dict) -> str:
            subj = str(r.get("subject_id", ""))
            sid  = str(r.get("sentence_id", ""))
            return f"{ns}:{subj}|{sid}" if (subj and sid) else sid

        for idx, r in enumerate(self.ds.rows):
            gkey = derive_global_sentence_key(r, prefer=self.ds.key_mode)
            skey = _sentkey_of_row(r)
            key2src2idx[gkey][skey].append(idx)

        self.key2src2idx = key2src2idx
        self.multi_keys = [k for k, m in key2src2idx.items() if len(m) >= self.K]
        self.all_indices = list(range(len(self.ds)))

        # 预留比例（0~0.9），例如 0.25 表示保留 25% 槽位给随机负样本
        import os
        self.reserve_frac = float(os.environ.get("RESERVE_NEG_FRAC", "0.0"))
        self.reserve_frac = min(max(self.reserve_frac, 0.0), 0.9)

        logger.info(f"[GlobalKeyMultiSourceSampler] multi_keys={len(self.multi_keys)}, "
                    f"group_k={self.K}, B={self.B}, reserve_neg_frac={self.reserve_frac}")

    def __len__(self):
        n = len(self.ds)
        return (n // self.B) if self.drop_last else math.ceil(n / self.B)

    def __iter__(self):
        num_batches = len(self)
        for _ in range(num_batches):
            batch: List[int] = []

            # 可用于“成组”的配额（其余为随机槽位）
            quota = int(round(self.B * (1.0 - self.reserve_frac)))
            quota = max(self.K, min(quota, self.B))
            max_groups = quota // self.K

            if self.multi_keys and max_groups > 0:
                perm = torch.randperm(len(self.multi_keys)).tolist()
                sel = [self.multi_keys[perm[i % len(self.multi_keys)]] for i in range(max_groups)]
            else:
                sel = []

            # 先填充成组的正样本
            for gk in sel:
                src2idx = self.key2src2idx[gk]
                src_keys = list(src2idx.keys())
                if len(src_keys) < self.K:
                    continue
                perm_s = torch.randperm(len(src_keys)).tolist()
                chosen_src = [src_keys[i] for i in perm_s[:self.K]]
                for sk in chosen_src:
                    pool = src2idx[sk]
                    j = int(torch.randint(low=0, high=len(pool), size=(1,)).item())
                    batch.append(pool[j])
                if len(batch) >= quota:
                    break

            # 再用随机样本补齐到 B
            need = self.B - len(batch)
            if need > 0:
                fill_idx = torch.randint(low=0, high=len(self.all_indices), size=(need,)).tolist()
                batch.extend(self.all_indices[i] for i in fill_idx)

            # 多了就随机裁掉；不够且 drop_last 则跳过
            if len(batch) > self.B:
                keep = torch.randperm(len(batch))[: self.B].tolist()
                batch = [batch[i] for i in keep]
            if len(batch) < self.B and self.drop_last:
                continue

            yield batch

# ============================== DataModule ============================== #
class MEGDataModule(pl.LightningDataModule):
    """
    window 模式：为每个锚窗构造整句上下文（MEG 窗口序列 + Audio 拼接）
    sentence 模式：只读整句 MEG（CxT）和可选的句级向量（Whisper→E5 或 audio）
    """
    def __init__(self,
                 train_manifest: str, val_manifest: str, test_manifest: str,
                 registry: SubjectRegistry,
                 ns_train: str = "", ns_val: str = "", ns_test: str = "",
                 batch_size: int = 64, num_workers: int = 8, normalize: bool = False,
                 context_mode: str = "none",
                 ctx_max_windows: int = 0,
                 group_by_sentence: bool = True, sentences_per_batch: int = 32, windows_per_sentence: int = 1,
                 pin_memory: bool = False, prefetch_factor: int = 1, persistent_workers: bool = True,
                 ctx_stride: int = 1,
                 exclude_self_from_ctx: bool = False,
                 ctx_exclude_radius: int = 0,
                 sent_guard_windows: int = 0,
                 sentence_fast_io: bool = True,
                 key_mode: Literal["auto", "audio", "text"] | None = None):
        super().__init__()
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.test_manifest = test_manifest
        self.registry = registry
        self.ns_train = ns_train
        self.ns_val = ns_val
        self.ns_test = ns_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        assert context_mode in ("none", "window", "sentence")
        self.context_mode = context_mode
        self.ctx_max_windows = int(ctx_max_windows)
        self.group_by_sentence = bool(group_by_sentence)
        self.sentences_per_batch = int(sentences_per_batch)
        self.windows_per_sentence = 1  # 强制为 1
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.persistent_workers = bool(persistent_workers)

        # 兼容参数（用于 window 模式）：
        self.ctx_stride = max(1, int(ctx_stride))
        self.exclude_self_from_ctx = bool(exclude_self_from_ctx)
        self.ctx_exclude_radius = int(ctx_exclude_radius)
        self.sent_guard_windows = max(0, int(sent_guard_windows))

        # FAST I/O for sentence
        self.sentence_fast_io = bool(sentence_fast_io)

        # 键策略（可统一指定；未指定则各 Dataset 按 manifest 自行推断）
        self.key_mode = key_mode

    # ---------------- setup ----------------
    def setup(self, stage: str | None = None):
        # 若未显式指定 key_mode，则各自按 manifest 路径推断
        km_train = self.key_mode or _infer_key_mode_from_manifest(Path(self.train_manifest))
        km_val   = self.key_mode or _infer_key_mode_from_manifest(Path(self.val_manifest))
        km_test  = self.key_mode or _infer_key_mode_from_manifest(Path(self.test_manifest))

        self.train_set = MEGDataset(self.train_manifest, registry=self.registry, namespace=self.ns_train,
                                    normalize=self.normalize, context_mode=self.context_mode,
                                    sentence_fast_io=self.sentence_fast_io, key_mode=km_train)
        self.val_set = MEGDataset(self.val_manifest, registry=self.registry, namespace=self.ns_val,
                                  normalize=self.normalize, context_mode=self.context_mode,
                                  sentence_fast_io=self.sentence_fast_io, key_mode=km_val)
        self.test_set = MEGDataset(self.test_manifest, registry=self.registry, namespace=self.ns_test,
                                   normalize=self.normalize, context_mode=self.context_mode,
                                   sentence_fast_io=self.sentence_fast_io, key_mode=km_test)

    # -------- window 模式：整句上下文（MEG 全部窗口 + Audio 拼接） --------
    def _build_sentence_context(self, items: List[Dict]):
        B = len(items)
        all_meg_list: List[torch.Tensor] = []
        all_key_list: List[torch.Tensor] = []
        all_mask_list: List[torch.Tensor] = []

        per_audio_cat: List[torch.Tensor] = []
        per_audio_len: List[int] = []

        def _row_key(r: dict) -> int:
            return _safe_uid63(_row_window_content_key(r))

        for it in items:
            rows_full = list(it.get("__ctx_rows__", []))  # 已按时间排序
            if len(rows_full) == 0:
                rows_full = [it["row"]]

            if self.ctx_max_windows is None or self.ctx_max_windows <= 0:
                rows_pick = rows_full
            else:
                rows_pick = rows_full[: self.ctx_max_windows]

            # --- MEG 侧：堆叠 [N,C,T] ---
            megs, keys = [], []
            for r in rows_pick:
                x = _ensure_CxT(np.load(r["meg_win_path"]).astype(np.float32))
                megs.append(torch.from_numpy(x))
                keys.append(_row_key(r))
            N_i = max(1, len(megs))
            C, T = (megs[0].shape if len(megs) > 0 else it["meg_win"].shape)
            if len(megs) == 0:
                megs = [torch.zeros(C, T, dtype=it["meg_win"].dtype)]
                keys = [-1]
            meg_stack = torch.stack(megs, dim=0)          # [N_i,C,T]
            key_vec   = torch.tensor(keys, dtype=torch.long)
            mask_vec  = torch.zeros(N_i, dtype=torch.bool)

            all_meg_list.append(meg_stack)
            all_key_list.append(key_vec)
            all_mask_list.append(mask_vec)

            # --- Audio 侧：沿时间拼接 [1024, sum(Tw)] ---
            a_list = []
            for r in rows_pick:
                a = _ensure_DxT(np.load(r["audio_feature_path"]), AUDIO_D)
                a_list.append(torch.from_numpy(a))
            if len(a_list) == 0:
                a_list = [it["audio_feat"]]
            a_cat = torch.cat(a_list, dim=1)  # [1024, T_total]
            per_audio_cat.append(a_cat)
            per_audio_len.append(int(a_cat.size(1)))

        # 对齐 batch 维度（MEG 的 N，Audio 的 T_total）
        Nmax = max(arr.size(0) for arr in all_meg_list)
        B_meg, B_key, B_msk = [], [], []
        for meg_stack, key_vec, mask_vec in zip(all_meg_list, all_key_list, all_mask_list):
            N_i, C, T = meg_stack.shape
            if N_i < Nmax:
                pad_n = Nmax - N_i
                meg_pad = torch.zeros(pad_n, C, T, dtype=meg_stack.dtype)
                key_pad = torch.full((pad_n,), -1, dtype=torch.long)
                msk_pad = torch.ones(pad_n, dtype=torch.bool)  # True=pad
                meg_stack = torch.cat([meg_stack, meg_pad], dim=0)
                key_vec   = torch.cat([key_vec, key_pad], dim=0)
                mask_vec  = torch.cat([mask_vec, msk_pad], dim=0)
            B_meg.append(meg_stack)
            B_key.append(key_vec)
            B_msk.append(mask_vec)
        meg_sent = torch.stack(B_meg, dim=0)            # [B,Nmax,C,T]
        meg_sent_keys = torch.stack(B_key, dim=0)       # [B,Nmax]
        meg_sent_mask = torch.stack(B_msk, dim=0)       # [B,Nmax]

        Tmax = max(per_audio_len) if len(per_audio_len) > 0 else items[0]["audio_feat"].size(1)
        A_cat, A_msk = [], []
        for a_cat in per_audio_cat:
            T_i = int(a_cat.size(1))
            if T_i < Tmax:
                pad = torch.zeros(AUDIO_D, Tmax - T_i, dtype=a_cat.dtype)
                a_pad = torch.cat([a_cat, pad], dim=1)
                m = torch.zeros(Tmax, dtype=torch.bool); m[T_i:] = True
            else:
                a_pad = a_cat[:, :Tmax]
                m = torch.zeros(Tmax, dtype=torch.bool)
            A_cat.append(a_pad)
            A_msk.append(m)
        audio_sent = torch.stack(A_cat, dim=0)          # [B,1024,Tmax]
        audio_sent_mask = torch.stack(A_msk, dim=0)     # [B,Tmax]

        return meg_sent, meg_sent_mask, meg_sent_keys, audio_sent, audio_sent_mask

    # ---------------- collate ----------------
    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        col = defaultdict(list)

        def _compute_win_pos_scalar(it: Dict[str, Any]) -> float:
            full = list(it.get("__ctx_rows__", []))
            c_wid = str(it.get("window_id", ""))
            if not full:
                return 0.0
            idx_map = {str(r.get("window_id", "")): i for i, r in enumerate(full)}
            cidx = idx_map.get(c_wid, None)
            if cidx is None:
                return 0.0
            pos = ((cidx + 0.5) / max(1, len(full))) * 2.0 - 1.0
            return float(max(-1.0, min(1.0, pos)))

        # -------- collect --------
        for it in batch:
            # 公共
            col["sensor_locs"].append(it["sensor_locs"])
            col["subject_idx"].append(it["subject_idx"])
            col["row"].append(it["row"])
            col["sentence_key"].append(it.get("sentence_key", ""))
            col["global_sentence_key"].append(it.get("global_sentence_key", ""))

            # 句 tokens（可能来自 text/audio，优先级在 __getitem__ 已处理）
            if "text_sentence_feature" in it:
                col["text_sentence_feature"].append(it["text_sentence_feature"])  # [L,Dt]
            if "audio_sent_vec" in it:
                col["audio_sent_vec"].append(it["audio_sent_vec"])                # [L,Dt]

            # window 模式额外字段
            if "meg_win" in it:
                col["meg_win"].append(it["meg_win"])
                col["audio_feat"].append(it["audio_feat"])
                col["win_pos_scalar"].append(torch.tensor(_compute_win_pos_scalar(it), dtype=torch.float32))
                col["__ctx_rows__"].append(it.get("__ctx_rows__", []))

            # sentence 模式：整句 MEG 路径
            if "meg_sentence_full_path" in it:
                col["meg_sentence_full_path"].append(it["meg_sentence_full_path"])

        out: Dict[str, Any] = {}

        # -------- stack commons --------
        loc = torch.stack(col["sensor_locs"], dim=0)        # [B,C,3]
        sid = torch.stack(col["subject_idx"], dim=0)        # [B]
        out["sensor_locs"] = loc
        out["subject_idx"] = sid
        out["sentence_key"] = col["sentence_key"]
        out["global_sentence_key"] = col["global_sentence_key"]

        # window 模式：堆叠可选
        if "meg_win" in col and col["meg_win"]:
            out["meg_win"] = torch.stack(col["meg_win"], dim=0)
        if "audio_feat" in col and col["audio_feat"]:
            out["audio_feat"] = torch.stack(col["audio_feat"], dim=0)
        if "win_pos_scalar" in col and col["win_pos_scalar"]:
            out["win_pos_scalar"] = torch.stack(col["win_pos_scalar"], dim=0)

        # window 模式：构建整句上下文（MEG 全部窗口 + Audio 拼接）
        if self.context_mode == "window":
            meg_sent, meg_mask, meg_keys, audio_sent, audio_sent_mask = self._build_sentence_context(batch)
            out["meg_sent"] = meg_sent
            out["meg_sent_mask"] = meg_mask
            out["meg_sent_keys"] = meg_keys
            out["audio_sent"] = audio_sent
            out["audio_sent_mask"] = audio_sent_mask
        else:
            # 非 window 模式给占位
            B = sid.size(0)
            out["meg_sent_relpos"] = torch.full((B, 1), -1, dtype=torch.long)

        # -------- sentence 模式：加载整句 MEG & pad 对齐 --------
        if self.context_mode == "sentence" and ("meg_sentence_full_path" in col) and col["meg_sentence_full_path"]:
            uniq_sids: List[str] = []
            sid2u: Dict[str, int] = {}
            uniq_full: List[torch.Tensor] = []
            uniq_Ts: List[int] = []
            sample2u: List[int] = []

            for i, it in enumerate(batch):
                s_str = it.get("sentence_key", "")
                if s_str not in sid2u:
                    sid2u[s_str] = len(uniq_sids)
                    uniq_sids.append(s_str)
                    if "meg_sentence_full_path" in it:
                        x_np = _sent_load_cached(it["meg_sentence_full_path"])
                        x = _ensure_CxT(x_np)
                    else:
                        if "meg_win" in it:
                            x = _ensure_CxT(it["meg_win"].numpy().astype(np.float32))
                        else:
                            raise RuntimeError("sentence mode sample missing meg_sentence_full_path and meg_win.")
                    uniq_full.append(torch.from_numpy(x.copy()))
                    uniq_Ts.append(int(x.shape[1]))
                sample2u.append(sid2u[s_str])

            # 通道对齐 & 时间 pad
            C_target = int(loc.shape[1])  # [B,C,3] -> C
            T_max = max(uniq_Ts) if len(uniq_Ts) > 0 else (out["meg_win"].shape[2] if "meg_win" in out else 1)
            XU, MU = [], []
            for x in uniq_full:
                Ti = int(x.shape[1])
                # 通道对齐
                if x.shape[0] != C_target:
                    if x.shape[0] > C_target:
                        x = x[:C_target, :]
                    else:
                        padC = torch.zeros(C_target - x.shape[0], x.shape[1], dtype=x.dtype)
                        x = torch.cat([x, padC], dim=0)
                # 时间对齐：尾部复制 pad（mask=True 表示 pad）
                if Ti < T_max:
                    pad = x[:, -1:].repeat(1, T_max - Ti)
                    x = torch.cat([x, pad], dim=1)
                    m = torch.zeros(T_max, dtype=torch.bool); m[Ti:] = True
                else:
                    m = torch.zeros(T_max, dtype=torch.bool)
                XU.append(x); MU.append(m)

            sent_unique_full = torch.stack(XU, dim=0)            # [U,C,Tmax]
            sent_unique_mask = torch.stack(MU, dim=0)            # [U,Tmax]
            sample2sent      = torch.tensor(sample2u, dtype=torch.long)

            out["meg_sent_full"] = sent_unique_full[sample2sent]     # [B,C,Tmax]
            out["meg_sent_full_mask"] = sent_unique_mask[sample2sent]# [B,Tmax]

            # -------- 句级 tokens（矩阵）打包：优先 text，其次 audio；两者都可缺省 --------
            tok_list = None
            if "text_sentence_feature" in col and col["text_sentence_feature"]:
                tok_list = col["text_sentence_feature"]          # 每项是 [L,Dt] 的张量
            elif "audio_sent_vec" in col and col["audio_sent_vec"]:
                tok_list = col["audio_sent_vec"]                 # 每项是 [L,Dt] 的张量

            if tok_list is not None:
                # 统计 Lmax, Dt；对齐 batch
                Ls = [int(x.size(0)) for x in tok_list]
                Dt = int(tok_list[0].size(1))
                Lmax = max(Ls)
                pads, masks = [], []
                for X in tok_list:
                    L = int(X.size(0))
                    if L < Lmax:
                        pad = torch.zeros(Lmax - L, Dt, dtype=X.dtype)
                        Xpad = torch.cat([X, pad], dim=0)
                        m = torch.zeros(Lmax, dtype=torch.bool); m[L:] = True
                    else:
                        Xpad = X[:Lmax]
                        m = torch.zeros(Lmax, dtype=torch.bool)
                    pads.append(Xpad)
                    masks.append(m)
                tokens = torch.stack(pads,  dim=0)   # [B, Lmax, Dt]
                tmask  = torch.stack(masks, dim=0)   # [B, Lmax] True=pad

                # 导出标准键（训练脚本会优先找这些）
                out["audio_tpp"] = tokens
                out["audio_tpp_mask"] = tmask

                # 兼容旧评估：对有效 token 取均值并 L2 归一化，给一个句向量
                valid_counts = (~tmask).sum(dim=1, keepdim=True).clamp(min=1).to(tokens.dtype)  # [B,1]
                sent_vec = (tokens.masked_fill(tmask.unsqueeze(-1), 0.0).sum(dim=1) / valid_counts)
                sent_vec = F.normalize(sent_vec, dim=-1)
                out["teacher_sentence_vec"] = sent_vec
                out["audio_sentence_feature"] = sent_vec  # 兼容旧字段名

                # 如需也保留原 tokens（区分来源），可解除下列注释：
                # if "audio_sent_vec" in col and col["audio_sent_vec"]:
                #     out["audio_sent_tokens"] = tokens  # 或分别暴露
                # if "text_sentence_feature" in col and col["text_sentence_feature"]:
                #     out["text_sent_tokens"] = tokens

        # -------- audio_uid（基于全局句键） --------
        if "global_sentence_key" in out and isinstance(out["global_sentence_key"], list) and len(out["global_sentence_key"]) > 0:
            au_list = [audio_uid_from_global_sentence_key(k) for k in out["global_sentence_key"]]
            out["audio_uid"] = torch.tensor(au_list, dtype=torch.long)
        else:
            sk_list = out.get("sentence_key", [])
            au_list = [_safe_uid63(k) for k in sk_list]
            out["audio_uid"] = torch.tensor(au_list, dtype=torch.long)

        return out

    # ---------------- dataloaders ----------------
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

    def train_dataloader(self):
        # sentence 模式：支持 multi-source 分组（env: TRAIN_GROUP_K / EVAL_GROUP_K）
        if self.context_mode == "sentence":
            group_k = int(os.environ.get("TRAIN_GROUP_K", os.environ.get("EVAL_GROUP_K", "1")))
            if group_k >= 2:
                # 分组时，batch 大小用 sentences_per_batch 更直观
                return self._build_grouped_loader(self.train_set, group_k=group_k, batch_size=self.sentences_per_batch)
            # 否则：每句采 1 个代表
            sampler = SentenceGroupedBatchSampler(
                self.train_set,
                sentences_per_batch=self.sentences_per_batch,
                windows_per_sentence=1,
                drop_last=True
            )
            return DataLoader(
                self.train_set,
                batch_sampler=sampler,
                collate_fn=self._collate,
                **self._dl_common_kwargs(),
            )

        # window 模式（保留旧逻辑）
        k = int(os.environ.get("MS_GROUP_K", "1"))
        if k >= 2:
            return DataLoader(
                self.train_set,
                batch_sampler=AudioUIDMultiSourceBatchSampler(
                    self.train_set, batch_size=self.sentences_per_batch, group_k=k, drop_last=True
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
        # sentence 模式：评估也支持 multi-source 分组
        if self.context_mode == "sentence":
            group_k = int(os.environ.get("VAL_GROUP_K", os.environ.get("EVAL_GROUP_K", "1")))
            if group_k >= 2:
                return self._build_grouped_loader(self.val_set, group_k=group_k)
            sampler = SentenceGroupedBatchSampler(
                self.val_set,
                sentences_per_batch=self.batch_size,
                windows_per_sentence=1,
                drop_last=False
            )
            return DataLoader(
                self.val_set,
                batch_sampler=sampler,
                collate_fn=self._collate,
                **self._dl_common_kwargs(),
            )
        # 其它模式沿用旧逻辑
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
                drop_last=False
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


# ============================== Paper-aligned Loss（保留） ============================== #
class PaperClipLoss(nn.Module):
    def __init__(self,
                 target_T: int = TARGET_T,
                 pool: bool = False,
                 center: bool = False,
                 trim_min: int | None = None,
                 trim_max: int | None = None,
                 use_temperature: bool = False,
                 candidate_l2: bool = True):
        super().__init__()
        self.target_T = target_T
        self.pool = pool
        self.center = center
        self.trim_min = trim_min
        self.trim_max = trim_max
        self.candidate_l2 = bool(candidate_l2)
        if use_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))
        else:
            self.register_parameter("logit_scale", None)

    @staticmethod
    def _to_BCT(x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expect 3D, got {tuple(x.shape)}")
        B, A, C = x.shape
        return x if A >= C else x.transpose(1, 2).contiguous()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_BCT(x.to(torch.float32))
        if x.size(-1) != self.target_T:
            x = torch.nn.functional.interpolate(x, size=self.target_T, mode="linear", align_corners=False)
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
        m = self._prep(meg_f)  # [B,D,T]
        a = self._prep(aud_f)  # [B,D,T]
        if self.candidate_l2:
            inv_norms = (a.norm(dim=(1, 2), p=2) + 1e-8).reciprocal()  # [B]
            logits = torch.einsum("bct,oct,o->bo", m, a, inv_norms)
        else:
            logits = torch.einsum("bct,oct->bo", m, a)
        if self.logit_scale is not None:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)
        tgt = torch.arange(logits.size(0), device=logits.device)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        return loss, logits

# ======== Pos-mask helpers for sentence-level retrieval ========

def audio_uid_from_global_sentence_key(gkey: str) -> int:
    """
    从全局句键抽取 audio_id，并稳定映射为 int64：
      典型 gkey: "<audio_id>::<start_ms>-<end_ms>"
    若不匹配，则退化为对 gkey 的哈希。
    """
    if not gkey:
        return _safe_uid63("UNK")
    if "::" in gkey:
        aid = gkey.split("::", 1)[0]
    else:
        # 尽量保留 audio_id 前缀；不确定格式时直接哈希
        aid = gkey
    return _safe_uid63(aid)


def build_pos_mask_same_audio_uid_diff_source(
    audio_uid: torch.Tensor | list,
    sentence_key: list[str] | tuple[str, ...]
) -> torch.Tensor:
    """
    返回 [B,B] 的 bool 矩阵，True 表示正样本：
      - audio_uid 相同（同一全局音频片段）
      - 且 sentence_key 不同（跨来源/被试）
    注意：训练脚本会另外把对角线 OR 进去，不需要在这里加对角。
    """
    if isinstance(audio_uid, list):
        au = torch.tensor(audio_uid, dtype=torch.long)
    else:
        au = audio_uid.to(torch.long)
    B = int(au.numel())

    # same-audio 矩阵
    same_audio = (au.view(B, 1) == au.view(1, B))

    # 把 sentence_key 映射为 int，便于广播比较
    sk_ids = []
    for s in sentence_key:
        sk_ids.append(_safe_uid63(s))
    sk = torch.tensor(sk_ids, dtype=torch.long, device=au.device)
    diff_src = (sk.view(B, 1) != sk.view(1, B))

    pos_mask = same_audio & diff_src
    return pos_mask.to(torch.bool)


def build_pos_mask_same_global_sentence_diff_subject(
    global_sentence_key: list[str] | tuple[str, ...],
    subject_idx: torch.Tensor | list[int],
) -> torch.Tensor:
    """
    返回 [B,B] 的 bool 矩阵，True 表示正样本：
      - global_sentence_key 完全相同（同一个音频的同一句）
      - 且 subject_idx 不同（跨被试）
    训练脚本可直接使用本函数，之后再 OR 上对角线作为兜底正样本。
    """
    if isinstance(subject_idx, list):
        subj = torch.tensor(subject_idx, dtype=torch.long)
    else:
        subj = subject_idx.to(torch.long)
    B = int(subj.numel())

    # 把字符串键哈希成 int，避免 Python 层面逐一比较
    g_ids = torch.tensor([_safe_uid63(k) for k in global_sentence_key], dtype=torch.long, device=subj.device)

    same_sent = (g_ids.view(B, 1) == g_ids.view(1, B))
    diff_subj = (subj.view(B, 1) != subj.view(1, B))
    return (same_sent & diff_subj).to(torch.bool)


# ============================== Retrieval metrics ============================== #
@torch.no_grad()
def batch_retrieval_metrics(logits: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
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
    pos_mask: torch.Tensor,              # [B,B]，True=正样本（通常“同句且跨来源”）
    ks=(1, 5, 10),
    reduce: str = "macro",               # "macro"（默认，按query平均）或 "micro"（跨全体正样本）
    fallback_to_self: bool = True        # 某行没有跨来源正样本时，是否回退把对角当作正样本
) -> Dict[str, float]:
    """
    兼容旧口径的同时，新增“all-positives”统计：
      - topK_pos_any:   是否“任一正样本落入topK”（旧口径）
      - Rpos@K:         (#落入topK的正样本) / (#正样本)         —— k 限定召回
      - Ppos@K:         (#落入topK的正样本) / K                 —— k 限定精度
      - MRR_pos_min:    1 / (最早命中正样本的rank)               —— 旧口径
      - MRR_pos_mean:   正样本的 (1/rank) 的均值                 —— 新
      - MAP_pos:        平均精度（对所有正样本位置做AP）          —— 新
      - mean_rank_pos:  正样本rank的均值                         —— 新
    """
    B = logits.size(0)
    ks = tuple(k for k in ks if 1 <= k <= B)
    device = logits.device

    # 排序与rank
    order = logits.argsort(dim=1, descending=True)    # [B,B]
    inv   = order.argsort(dim=1)                      # inv[i,j] = 排名(0-based)

    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos_full = pos_mask.clone()
    no_pos = ~pos_full.any(dim=1)
    if fallback_to_self:
        pos_full[no_pos] = eye[no_pos]               # 退化为对角
    valid_row = pos_full.any(dim=1)                  # 有正样本的行（应用fallback后几乎全True）

    # 逐行统计
    tp_at_k_sum, pos_cnt_sum, tp_at_k_micro = {k:0.0 for k in ks}, 0.0, {k:0.0 for k in ks}
    prec_at_k_sum = {k:0.0 for k in ks}
    any_hit_at_k = {k:0.0 for k in ks}
    mrr_min_sum = 0.0
    mrr_mean_sum = 0.0
    map_sum = 0.0
    mean_rank_sum = 0.0
    n_rows = int(valid_row.sum().item())

    for i in range(B):
        if not bool(valid_row[i]): 
            continue
        pos_cols = pos_full[i].nonzero(as_tuple=False).squeeze(1)      # 正样本列索引
        ranks_i  = inv[i, pos_cols].to(torch.float32)                  # 0-based ranks of positives
        num_pos  = float(len(pos_cols))
        if num_pos <= 0:
            continue

        # any-hit（旧口径）
        min_rank = float(ranks_i.min().item())
        for k in ks:
            if min_rank < k:
                any_hit_at_k[k] += 1.0

        # all-positives：Recall@K 与 Precision@K
        for k in ks:
            tp = float((ranks_i < k).sum().item())            # topK 内的正样本数
            tp_at_k_sum[k] += (tp / num_pos)                  # 每行召回（宏平均）
            prec_at_k_sum[k] += (tp / float(k))               # 每行精度（宏平均）
            tp_at_k_micro[k] += tp                            # 微平均分子
        pos_cnt_sum += num_pos

        # MRR（min 与 mean）
        mrr_min_sum  += 1.0 / (min_rank + 1.0)
        mrr_mean_sum += float((1.0 / (ranks_i + 1.0)).mean().item())

        # MAP（标准平均精度）
        # 利用正样本rank闭式：AP = (1/|P|) * sum_{r in ranks_pos} (#pos with rank<=r) / (r+1)
        ranks_sorted = torch.sort(ranks_i).values
        # 位置 m 的累计正样本数恰好为 m+1
        cum_pos = torch.arange(1, 1 + ranks_sorted.numel(), device=device, dtype=torch.float32)
        ap = float((cum_pos / (ranks_sorted + 1.0)).mean().item())
        map_sum += ap

        # mean rank
        mean_rank_sum += float((ranks_i + 1.0).mean().item())

    out = {}
    denom = max(1, n_rows)
    for k in ks:
        out[f"top{k}_pos_any"] = any_hit_at_k[k] / denom           # 旧：任一正样落入topK
        out[f"Rpos@{k}"]       = tp_at_k_sum[k] / denom            # 新：k限定召回（宏）
        out[f"Ppos@{k}"]       = prec_at_k_sum[k] / denom          # 新：k限定精度（宏）
        # 微平均的 k 限定召回（跨全体正样本）
        if pos_cnt_sum > 0:
            out[f"Rpos_micro@{k}"] = tp_at_k_micro[k] / pos_cnt_sum

    out["mrr_pos_min"]   = mrr_min_sum  / denom                    # 旧：最早命中的RR
    out["mrr_pos_mean"]  = mrr_mean_sum / denom                    # 新：所有正样本RR的均值
    out["map_pos"]       = map_sum / denom                         # 新：平均精度
    out["mean_rank_pos"] = mean_rank_sum / denom                   # 新：所有正样本rank的均值
    out["pos_cov_xsrc"]  = float((~no_pos).float().mean().item())  # 覆盖率（和你原来的等价）

    return out

@torch.no_grad()
def entity_ids_from_global_sentence_key(
    global_sentence_key: List[str] | Tuple[str, ...],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    ids = [_safe_uid63(k) for k in global_sentence_key]
    return torch.tensor(ids, dtype=torch.long, device=device)


def groupby_aggregate_logits_by_entity(
    logits: torch.Tensor,                       # [B,O]
    col_global_sentence_key,                    # 长度 O：全局句键 或 LongTensor[O]
    agg: str = "max",
    row_global_sentence_key=None,
):
    """
    将列按“实体”（由 global_sentence_key 唯一化）分组做聚合，返回：
      - agg_logits: [B,U]，U 为 batch 内去重后的实体数
      - uniq_ent:   [U]，每个实体的 long id
      - query_ent_col: [B]，每个 query 的“正类实体”列号（在 [0..U-1]）
    支持聚合：
      - "max"： 组内取最大值
      - "mean"：组内求均值
      - "logsumexp"/"lse"：log( sum(exp(·)) )，数值稳定版（训练推荐）
    """
    device = logits.device
    B, O = logits.shape

    # 列实体 id -> long 向量 [O]
    if isinstance(col_global_sentence_key, (list, tuple)):
        c_ids = torch.tensor([_safe_uid63(k) for k in col_global_sentence_key],
                             dtype=torch.long, device=device)
    else:
        c_ids = col_global_sentence_key.to(device=device, dtype=torch.long)

    # 唯一实体 + 反向索引 inv[o] ∈ [0..U-1]
    uniq_ent, inv = torch.unique(c_ids, sorted=False, return_inverse=True)  # [U], [O]
    U = int(uniq_ent.numel())

    # --------- 聚合 ----------
    if agg == "max":
        agg_logits = torch.full((B, U), float("-inf"), dtype=logits.dtype, device=device)
        index = inv.view(1, O).expand(B, O)
        agg_logits.scatter_reduce_(dim=1, index=index, src=logits, reduce="amax", include_self=True)

    elif agg == "mean":
        sum_logits = torch.zeros((B, U), dtype=logits.dtype, device=device)
        index = inv.view(1, O).expand(B, O)
        sum_logits.scatter_add_(dim=1, index=index, src=logits)
        counts = torch.bincount(inv, minlength=U).clamp_min(1).to(sum_logits.dtype)  # [U]
        agg_logits = sum_logits / counts.view(1, U)

    elif agg in ("logsumexp", "lse"):
        # 1) 每组的 max（数值中心），scatter_reduce_ 比循环快
        m = torch.full((B, U), float("-inf"), dtype=logits.dtype, device=device)
        index = inv.view(1, O).expand(B, O)
        m.scatter_reduce_(dim=1, index=index, src=logits, reduce="amax", include_self=True)  # [B,U]

        # 2) 对中心化后的 exp 做组内求和
        m_g = m.gather(1, index)                 # [B,O]：列 o 所属组的 max
        centered = (logits - m_g).exp()          # [B,O] 数值稳定的 exp
        sumexp = torch.zeros((B, U), dtype=logits.dtype, device=device)
        sumexp.scatter_add_(dim=1, index=index, src=centered)  # [B,U]

        # 3) logsumexp = log(sumexp) + m
        agg_logits = sumexp.clamp_min(1e-30).log() + m         # [B,U]

        # —— 如需“按组大小归一”（可选）：解注释下一段
        counts = torch.bincount(inv, minlength=U).clamp_min(1).to(agg_logits.dtype)  # [U]
        agg_logits = agg_logits - counts.view(1, U).log()  # 近似把“加和”变为“平均”

    else:
        raise ValueError(f"Unsupported agg: {agg}")

    # 行（query）实体 id
    if row_global_sentence_key is None:
        assert O >= B, "row_global_sentence_key 未提供时，要求 O>=B"
        r_ids = c_ids[:B]
    else:
        if isinstance(row_global_sentence_key, (list, tuple)):
            r_ids = torch.tensor([_safe_uid63(k) for k in row_global_sentence_key],
                                 dtype=torch.long, device=device)
        else:
            r_ids = row_global_sentence_key.to(device=device, dtype=torch.long)

    # 把每个 query 的实体 id 映射到 uniq_ent 中的位置，得到金标列号 [B]
    id2pos = {int(uniq_ent[i].item()): i for i in range(U)}
    query_ent_col = torch.tensor([id2pos[int(x.item())] for x in r_ids],
                                 dtype=torch.long, device=device)

    return agg_logits, uniq_ent, query_ent_col



@torch.no_grad()
def batch_retrieval_metrics_entity_dedup(
    logits: torch.Tensor,                               # [B,B]，原始行=查询，列=候选（含重复实体）
    global_sentence_key: List[str] | Tuple[str, ...],   # 与列/行同顺序（同一个 batch）
    ks: Tuple[int, ...] = (1, 5, 10),
    agg: Literal["max", "mean"] = "max",
) -> Dict[str, float]:
    """
    在“去重后的实体库”上评估：
      1) 先把同实体的候选列做聚合（默认 group-max）得到 [B,U]。
      2) 对每个 query i，正类就是“与 i 相同实体”的聚合列（唯一一列）。
      3) 计算该正类列的 rank → MRR / mean_rank / top-k 命中率。

    备注：
      - 这与你离线的“去重 Recall@k（entity-level）”一致；
      - 不再区分是否跨被试，因为实体本身已去重（包含了跨被试的合并）。
    """
    agg_logits, uniq_ent, query_ent_col = groupby_aggregate_logits_by_entity(
        logits, global_sentence_key, agg=agg
    )  # [B,U], [U], [B]
    B, U = agg_logits.shape
    ks = tuple(k for k in ks if 1 <= k <= U)

    # 排名
    preds = agg_logits.argsort(dim=1, descending=True)   # [B,U]
    inv_rank = preds.argsort(dim=1)                      # inv_rank[i, j] = rank of column j for query i
    ranks = inv_rank[torch.arange(B, device=logits.device), query_ent_col]  # [B] 0-based

    ranks_f = ranks.to(torch.float32)
    out = {f"entity_top{k}": float((ranks < k).float().mean().item()) for k in ks}
    out["entity_mrr"] = float((1.0 / (ranks_f + 1.0)).mean().item())
    out["entity_mean_rank"] = float((ranks_f + 1.0).mean().item())
    out["entity_num_unique"] = float(U)
    return out
@torch.no_grad()
def entity_positive_coverage(
    global_sentence_key: List[str] | Tuple[str, ...],
    subject_idx: torch.Tensor | List[int]
) -> float:
    """
    返回一个标量：batch 中有多少比例的 query，其实体在“其它被试”中也出现过。
    这与多正样本训练的有效性直接相关。
    """
    if isinstance(subject_idx, list):
        subj = torch.tensor(subject_idx, dtype=torch.long)
    else:
        subj = subject_idx.to(torch.long)

    B = int(subj.numel())
    g_ids = torch.tensor([_safe_uid63(k) for k in global_sentence_key], dtype=torch.long, device=subj.device)
    same_ent = (g_ids.view(B, 1) == g_ids.view(1, B))
    diff_subj = (subj.view(B, 1) != subj.view(1, B))
    has_cross_subject_pos = (same_ent & diff_subj).any(dim=1).float().mean().item()
    return float(has_cross_subject_pos)


# ============================== Logger & Utils ============================== #
def build_run_dir(exp_name: str, root: str | Path = "runs") -> Path:
    """
    创建唯一 run 目录：
      runs/<exp_name>_<SLURM_JOB_ID|UTC时间>/
    并建好 checkpoints / records 子目录。
    """
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)

    # 清理一下名字（避免奇怪字符）
    safe_exp = re.sub(r"[^-\w.]+", "_", str(exp_name).strip()) or "exp"

    jid = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    ts  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"{safe_exp}_{jid}" if jid else f"{safe_exp}_{ts}"

    run_dir = base / name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "records").mkdir(parents=True, exist_ok=True)

    logger.info(f"[build_run_dir] {run_dir.as_posix()}")
    return run_dir


def _safe_float(x):
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None

class MetricLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.step_log = run_dir / "metrics_step.jsonl"
        self.epoch_log = run_dir / "metrics_epoch.jsonl"
        self._step_records: List[dict] = []
        self._epoch_records: List[dict] = []
        self._open_files()

    def _open_files(self):
        self._fs = open(self.step_log, "a", encoding="utf-8")
        self._fe = open(self.epoch_log, "a", encoding="utf-8")

    def close(self):
        try:
            self._fs.close()
            self._fe.close()
        except Exception:
            pass

    def write_step(self, phase: str, step: int, epoch: int, loss: float, lr: float | None,
                   metrics: Dict[str, float] | None = None):
        rec = {
            "time": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "phase": phase,
            "global_step": int(step),
            "epoch": int(epoch),
            "loss": _safe_float(loss),
            "lr": _safe_float(lr) if lr is not None else None,
        }
        if metrics:
            for k, v in metrics.items():
                rec[k] = _safe_float(v)
        self._fs.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fs.flush()
        self._step_records.append(rec)

    def write_epoch(self, epoch: int, train_loss: float | None, val_loss: float | None,
                    best_val_top10: float | None, train_metrics: Dict[str, float] | None,
                    val_metrics: Dict[str, float] | None):
        rec = {
            "time": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "epoch": int(epoch),
            "train_loss": None if train_loss is None else _safe_float(train_loss),
            "val_loss":   None if val_loss   is None else _safe_float(val_loss),
            "best_val_top10": None if best_val_top10 is None else _safe_float(best_val_top10),
        }
        if train_metrics:
            for k, v in train_metrics.items():
                rec[f"train_{k}"] = _safe_float(v)
        if val_metrics:
            for k, v in val_metrics.items():
                rec[f"val_{k}"] = _safe_float(v)
        self._fe.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fe.flush()
        self._epoch_records.append(rec)

    @pl.utilities.rank_zero_only
    def export_tables_and_plots(self):
        import csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        step_csv = self.run_dir / "metrics_step.csv"
        epoch_csv = self.run_dir / "metrics_epoch.csv"

        # ---- CSV 导出 ----
        if self._step_records:
            keys = sorted({k for r in self._step_records for k in r.keys()})
            with open(step_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self._step_records)
        if self._epoch_records:
            keys = sorted({k for k in self._epoch_records[0].keys()})
            with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self._epoch_records)

        # ---- 可选：XLSX 导出 ----
        try:
            import pandas as pd
            step_xlsx = self.run_dir / "metrics_step.xlsx"
            epoch_xlsx = self.run_dir / "metrics_epoch.xlsx"
            if self._step_records:
                pd.DataFrame(self._step_records).to_excel(step_xlsx, index=False)
            if self._epoch_records:
                pd.DataFrame(self._epoch_records).to_excel(epoch_xlsx, index=False)
        except Exception as e:
            logger.warning(f"XLSX export skipped: {e}")

        # ---- 图表：loss vs epoch ----
        try:
            if self._epoch_records:
                epochs = [r["epoch"] for r in self._epoch_records]
                tr = [r.get("train_loss") for r in self._epoch_records]
                va = [r.get("val_loss") for r in self._epoch_records]
                if any(x is not None for x in tr+va):
                    plt.figure(figsize=(8, 5))
                    if tr and any(x is not None for x in tr): plt.plot(epochs, tr, label="train_loss")
                    if va and any(x is not None for x in va): plt.plot(epochs, va, label="val_loss")
                    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs Epoch")
                    plt.legend(); plt.tight_layout()
                    plt.savefig(self.run_dir / "loss_vs_epoch.png"); plt.close()
        except Exception as e:
            logger.warning(f"Plotting loss skipped: {e}")

        # ---- 图表：mrr_pos vs epoch ----
        try:
            if self._epoch_records:
                epochs = [r["epoch"] for r in self._epoch_records]
                tr_mrr = [r.get("train_mrr_pos") for r in self._epoch_records]
                va_mrr = [r.get("val_mrr_pos")   for r in self._epoch_records]
                if any(x is not None for x in tr_mrr+va_mrr):
                    plt.figure(figsize=(8, 5))
                    if any(x is not None for x in tr_mrr): plt.plot(epochs, tr_mrr, label="train_mRR_pos")
                    if any(x is not None for x in va_mrr): plt.plot(epochs, va_mrr, label="val_mRR_pos")
                    # 标注 best_val_top10（若存在）
                    bests = [r.get("best_val_top10") for r in self._epoch_records]
                    if any(x is not None for x in bests):
                        last_best = next((b for b in reversed(bests) if b is not None), None)
                        if last_best is not None:
                            plt.axhline(y=last_best, linestyle="--", alpha=0.3, label=f"best_val_top10={last_best:.4f}")
                    plt.xlabel("epoch"); plt.ylabel("mrr_pos"); plt.title("MRR_pos vs Epoch")
                    plt.legend(); plt.tight_layout()
                    plt.savefig(self.run_dir / "mrr_pos_vs_epoch.png"); plt.close()
        except Exception as e:
            logger.warning(f"Plotting mrr_pos skipped: {e}")

        # ---- 图表：step 级曲线（如有） ----
        try:
            if self._step_records:
                steps = [r["global_step"] for r in self._step_records if "mrr_pos" in r]
                mrrs  = [r.get("mrr_pos") for r in self._step_records if "mrr_pos" in r]
                if steps and any(x is not None for x in mrrs):
                    plt.figure(figsize=(8, 5))
                    plt.plot(steps, mrrs, label="mrr_pos (step)")
                    plt.xlabel("global_step"); plt.ylabel("mrr_pos"); plt.title("MRR_pos vs Step")
                    plt.legend(); plt.tight_layout()
                    plt.savefig(self.run_dir / "mrr_pos_vs_step.png"); plt.close()

                steps2 = [r["global_step"] for r in self._step_records if "top10_pos" in r]
                top10s = [r.get("top10_pos") for r in self._step_records if "top10_pos" in r]
                if steps2 and any(x is not None for x in top10s):
                    plt.figure(figsize=(8, 5))
                    plt.plot(steps2, top10s, label="top10_pos (step)")
                    plt.xlabel("global_step"); plt.ylabel("top10_pos"); plt.title("Top-10_pos vs Step")
                    plt.legend(); plt.tight_layout()
                    plt.savefig(self.run_dir / "top10_pos_vs_step.png"); plt.close()
        except Exception as e:
            logger.warning(f"Plotting step curves skipped: {e}")


def map_amp_to_precision(amp: str) -> str | int:
    amp = amp.lower()
    if amp in ("off", "none", "32", "fp32"): return 32
    elif amp in ("fp16", "16", "16-mixed", "half"): return "16-mixed"
    elif amp in ("bf16", "bfloat16", "bf16-mixed"): return "bf16-mixed"
    else: raise ValueError(f"Unsupported --amp: {amp}")


@pl.utilities.rank_zero_only
def save_records(run_dir: Path, cfg: dict, best_ckpt_path: str | None,
                 subject_map_path: Optional[str] = None, registry: Optional[SubjectRegistry] = None):
    rec_dir = run_dir / "records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    with open(rec_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    if best_ckpt_path:
        with open(rec_dir / "best_checkpoint.txt", "w", encoding="utf-8") as f:
            f.write(best_ckpt_path + "\n")
    if subject_map_path:
        with open(rec_dir / "subject_mapping_path.txt", "w", encoding="utf-8") as f:
            f.write(subject_map_path + "\n")
    if registry is not None:
        with open(rec_dir / "subject_mapping_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(registry.to_json(), f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote records to: {rec_dir.as_posix()}")

