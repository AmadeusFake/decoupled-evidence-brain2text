#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script (paper-aligned + clean metrics/logging + subject registry)

Paper / reference-implementation alignment (used in this script)
---------------------------------------------------------------
- One-way contrastive objective (brain -> audio), in-batch negatives on the same GPU only.
- Linearly interpolate both modalities to T=360; **no temporal pooling** (keep time dimension).
- Scoring: **only candidates (audio side) are L2-normalized over (C,T)**; estimates (MEG side) are NOT normalized.
  logits[b, o] = < MEG_b , AUDIO_o / ||AUDIO_o||_2 >
- No temperature by default; enable learnable temperature via --loss_temp.
- Optional toggles (off by default): center (mean removal), trim (time cropping).

Engineering improvements (does not change training logic)
---------------------------------------------------------
1) SubjectRegistry: stable subject indexing (supports multi-dataset namespaces).
2) Cleaner logging & retrieval metrics (Top-1/5/10, MRR, MeanRank).
3) Export CSV/XLSX/PNG; print an epoch-level console table.
"""

import os
import sys
import json
import math
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable, Optional
from collections import defaultdict
import inspect

# ---- Matplotlib headless backend (HPC-friendly) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities import rank_zero_only

# Model (UltimateMEGEncoder)
from models.meg_encoder_ExpDilated import UltimateMEGEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")

TARGET_T = 360  # 120 Hz * 3 s


# ============================== Subject Registry ============================== #
class SubjectRegistry:
    """
    Stable "subject -> index" registry with optional per-manifest namespaces.

    - Composite key: "<namespace>:<subject_id>" (or just subject_id if namespace is empty).
    - Build: collect the union of subject_id across manifests (with namespaces), sort, then assign stable indices.
    - Persistence: save/load JSON so that training and downstream evaluation use exactly the same mapping.
    """
    def __init__(self, mapping: Dict[str, int] | None = None):
        self._map: Dict[str, int] = mapping or {}
        self._order: List[str] = [k for k, _ in sorted(self._map.items(), key=lambda kv: kv[1])]

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
        # Unknown subject: do not expand the model; optionally fall back to index 0 (minimal impact in practice).
        if fallback_to_first:
            return 0
        raise KeyError(f"Unknown subject key: {k}")

    def subjects(self) -> List[str]:
        return list(self._order)

    def to_json(self) -> Dict[str, Any]:
        return {"mapping": self._map, "order": self._order}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "SubjectRegistry":
        mp = d.get("mapping", {})
        reg = SubjectRegistry(mapping=mp)
        order = d.get("order")
        if order:
            reg._order = order
        else:
            reg._order = [k for k, _ in sorted(mp.items(), key=lambda kv: kv[1])]
        return reg

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
        reg = SubjectRegistry(mapping=mapping)
        return reg


# ============================== Data ============================== #
def _ensure_CxT(x: np.ndarray) -> np.ndarray:
    """Ensure a 2D array is in channel×time layout. Heuristic: if rows <= cols treat as [C,T], else transpose."""
    if x.ndim != 2:
        raise ValueError(f"Expect 2D [C,T] or [T,C], got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T


class MEGDataset(Dataset):
    """
    Reads a jsonl manifest (Stage-3 fully_preprocessed):
    - meg_win_path: normalized + clamped [C,T] float32
    - audio_feature_path: [1024,T] float32 (Stage-2 already interpolated to T=360; if not 360, we still handle it here/in loss)
    - sensor_coordinates_path: [C,3] float32, x,y in [0,1], z=0
    - subject_id: subject identifier (combined with namespace in SubjectRegistry to form a stable key)
    """
    def __init__(self, manifest_path: str, registry: SubjectRegistry, namespace: str, normalize: bool = False):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.is_dir():
            raise IsADirectoryError(f"Manifest should be a file, got a directory: {self.manifest_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(l) for l in f if l.strip()]

        before = len(self.samples)
        self.samples = [s for s in self.samples if Path(s.get("sensor_coordinates_path", "")).exists()]
        drop = before - len(self.samples)
        if drop > 0:
            logger.warning(f"{self.manifest_path.name}: dropped {drop} rows without sensor_coordinates_path")

        self.registry = registry
        self.namespace = (namespace or "").strip()
        self.normalize = normalize
        self._coords_cache: Dict[str, np.ndarray] = {}

        seen = sorted({s.get("subject_id") for s in self.samples if s.get("subject_id") is not None})
        logger.info(f"{self.manifest_path.name} loaded: {len(self.samples):,} samples; subjects in this split={len(seen)} (namespace='{self.namespace}')")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        meg = np.load(s["meg_win_path"]).astype(np.float32)
        meg = _ensure_CxT(meg)
        if self.normalize:
            # Ablation-only; default pipeline does NOT re-standardize (Stage-3 already robust-normalized + clamped).
            m = meg.mean(axis=1, keepdims=True)
            sd = meg.std(axis=1, keepdims=True) + 1e-6
            meg = (meg - m) / sd

        aud = np.load(s["audio_feature_path"]).astype(np.float32)
        aud = _ensure_CxT(aud)

        coord_path = s.get("sensor_coordinates_path", "")
        if not coord_path or not Path(coord_path).exists():
            raise RuntimeError("Missing sensor_coordinates_path; re-run Stage-3 to generate coordinates.")
        if coord_path in self._coords_cache:
            coords = self._coords_cache[coord_path]
        else:
            coords = np.load(coord_path).astype(np.float32)  # [C,3]
            self._coords_cache[coord_path] = coords

        sid_str = str(s.get("subject_id"))
        subj_idx = self.registry.index_of(self.namespace, sid_str, fallback_to_first=True)

        item = {
            "meg_win": torch.from_numpy(meg),         # [C,T]
            "audio_feature": torch.from_numpy(aud),   # [T,1024] or [1024,T] (corrected inside loss)
            "sensor_locs": torch.from_numpy(coords),  # [C,3]
            "subject_idx": torch.tensor(subj_idx, dtype=torch.long),
            "key": s.get("window_id", s.get("audio_key", "")),
        }
        return item


class MEGDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_manifest: str, val_manifest: str, test_manifest: str,
                 registry: SubjectRegistry,
                 ns_train: str = "", ns_val: str = "", ns_test: str = "",
                 batch_size: int = 64, num_workers: int = 8, normalize: bool = False):
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

    def setup(self, stage: str | None = None):
        self.train_set = MEGDataset(self.train_manifest, registry=self.registry, namespace=self.ns_train, normalize=self.normalize)
        self.val_set   = MEGDataset(self.val_manifest,   registry=self.registry, namespace=self.ns_val,   normalize=self.normalize)
        self.test_set  = MEGDataset(self.test_manifest,  registry=self.registry, namespace=self.ns_test,  normalize=self.normalize)

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        col = defaultdict(list)
        for b in batch:
            for k, v in b.items():
                col[k].append(v)

        meg = torch.stack(col["meg_win"], dim=0)       # [B,C,T]
        aud = torch.stack(col["audio_feature"], dim=0) # [B,?,?] (corrected inside loss to [B,D,T])
        loc = torch.stack(col["sensor_locs"], dim=0)   # [B,C,3]
        sid = torch.stack(col["subject_idx"], dim=0)   # [B]
        keys = col["key"]

        return {"meg_win": meg, "audio": aud, "sensor_locs": loc, "subject_idx": sid, "keys": keys}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True,
                          collate_fn=self._collate)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, drop_last=False,
                          collate_fn=self._collate)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, drop_last=False,
                          collate_fn=self._collate)


# ============================== Paper-aligned Loss ============================== #
class PaperClipLoss(nn.Module):
    """
    Paper-aligned contrastive loss (one-way: brain -> audio)
    - In-batch negatives on the same GPU.
    - Linear interpolation to T=360 before scoring.
    - **Only candidates (audio) are L2-normalized** over (C,T); MEG estimates are NOT normalized.
    - **No temporal pooling** (keep time dimension).
    - Optional: center / trim (off by default).
    - No temperature by default; enable learnable temperature via --loss_temp.
    """
    def __init__(
        self,
        target_T: int = TARGET_T,
        pool: bool = False,          # Keep False (no pooling)
        center: bool = False,        # Off by default (enable only for ablations)
        trim_min: int | None = None, # No trimming by default
        trim_max: int | None = None,
        use_temperature: bool = False,
        init_temp: float = 0.07,
    ):
        super().__init__()
        self.target_T = target_T
        self.pool = pool
        self.center = center
        self.trim_min = trim_min
        self.trim_max = trim_max
        if use_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp), dtype=torch.float32))
        else:
            self.register_parameter("logit_scale", None)

    @staticmethod
    def _to_BCT(x: torch.Tensor) -> torch.Tensor:
        # Accept [B,D,T] or [B,T,D]; return [B,C(=D),T].
        if x.dim() != 3:
            raise ValueError(f"Expect 3D, got {tuple(x.shape)}")
        B, A, C = x.shape
        return x if A >= C else x.transpose(1, 2).contiguous()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to [B,C,T] and interpolate to target_T; optional trim/pool/center.
        x = self._to_BCT(x.to(torch.float32))
        if x.size(-1) != self.target_T:
            x = F.interpolate(x, size=self.target_T, mode="linear", align_corners=False)
        # Optional trimming (time indices).
        if (self.trim_min is not None) or (self.trim_max is not None):
            t0 = 0 if self.trim_min is None else max(0, int(self.trim_min))
            t1 = x.size(-1) if self.trim_max is None else min(x.size(-1), int(self.trim_max))
            x = x[..., t0:t1]
        if self.pool:
            x = x.mean(dim=2, keepdim=True)  # Not recommended: discards temporal information.
        if self.center:
            x = x - x.mean(dim=(1, 2), keepdim=True)
        return x

    def forward(self, meg_f: torch.Tensor, aud_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Preprocess (interpolate / optional trim / optional centering)
        m = self._prep(meg_f)  # estimates:  [B,C,T']
        a = self._prep(aud_f)  # candidates: [B,C,T']

        # 2) L2-normalize candidates only, over (C,T)
        inv_norms = (a.norm(dim=(1, 2), p=2) + 1e-8).reciprocal()  # [B]

        # 3) Scoring: einsum form matching the reference implementation
        #    logits[b, o] = sum_{c,t} m[b,c,t] * ( a[o,c,t] / ||a[o]|| )
        logits = torch.einsum("bct,oct,o->bo", m, a, inv_norms)

        # 4) Optional temperature (off by default; CLIP-style when enabled)
        if self.logit_scale is not None:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)

        tgt = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, tgt)
        return loss, logits


# ============================== Metrics utils ============================== #
@torch.no_grad()
def batch_retrieval_metrics(logits: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    """
    Compute retrieval metrics from in-batch logits ([B,B], rows=MEG queries, cols=audio candidates).
    Returns: {'top1','top5','top10','mrr','mean_rank'}.
    """
    B = logits.size(0)
    device = logits.device
    ks = tuple(int(k) for k in ks if k >= 1)
    ks = tuple(k for k in ks if k <= B)
    preds = logits.argsort(dim=1, descending=True)           # [B,B]
    inv = preds.argsort(dim=1)                               # [B,B] inverse permutation
    tgt = torch.arange(B, device=device)
    ranks = inv[torch.arange(B, device=device), tgt]         # [B] 0-based
    ranks_float = ranks.to(torch.float32)
    out = {}
    for k in ks:
        out[f"top{k}"] = (ranks < k).float().mean().item()
    out["mrr"] = (1.0 / (ranks_float + 1.0)).mean().item()
    out["mean_rank"] = (ranks_float + 1.0).mean().item()
    return out


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x*100:.2f}%"


# ============================== Rich Logger (JSONL/CSV/XLSX/PNG) ============================== #
class MetricLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.step_log = self.run_dir / "metrics_step.jsonl"
        self.epoch_log = self.run_dir / "metrics_epoch.jsonl"
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
            "time": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "global_step": int(step),
            "epoch": int(epoch),
            "loss": float(loss),
            "lr": float(lr) if lr is not None else None,
        }
        if metrics:
            for k, v in metrics.items():
                rec[k] = float(v)
        self._fs.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fs.flush()
        self._step_records.append(rec)

    def write_epoch(self,
                    epoch: int,
                    train_loss: float | None,
                    val_loss: float | None,
                    best_val: float | None,
                    train_metrics: Dict[str, float] | None,
                    val_metrics: Dict[str, float] | None):
        rec = {
            "time": datetime.now(timezone.utc).isoformat(),
            "epoch": int(epoch),
            "train_loss": None if train_loss is None else float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "best_val_loss": None if best_val is None else float(best_val),
        }
        if train_metrics:
            for k, v in train_metrics.items(): rec[f"train_{k}"] = float(v)
        if val_metrics:
            for k, v in val_metrics.items(): rec[f"val_{k}"] = float(v)
        self._fe.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fe.flush()
        self._epoch_records.append(rec)

    @rank_zero_only
    def export_tables_and_plots(self):
        """Aggregate JSONL into CSV/XLSX, and save PNG curves (loss, lr, Top1)."""
        import csv
        # CSV export
        step_csv = self.run_dir / "metrics_step.csv"
        epoch_csv = self.run_dir / "metrics_epoch.csv"
        if self._step_records:
            keys = sorted({k for r in self._step_records for k in r.keys()})
            with open(step_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(self._step_records)
        if self._epoch_records:
            keys = sorted({k for r in self._epoch_records for k in r.keys()})
            with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(self._epoch_records)

        # XLSX export (optional dependency; skip if unavailable)
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

        # Plots: loss (epoch) & lr (step) & Top-1 (epoch)
        try:
            epochs = [r["epoch"] for r in self._epoch_records]
            tr = [r.get("train_loss") for r in self._epoch_records]
            va = [r.get("val_loss") for r in self._epoch_records]
            if epochs:
                plt.figure(figsize=(8, 5))
                if tr and any(x is not None for x in tr): plt.plot(range(len(tr)), tr, label="train")
                if va and any(x is not None for x in va): plt.plot(range(len(va)), va, label="val")
                plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss vs Epoch")
                plt.legend(); plt.tight_layout()
                plt.savefig(self.run_dir / "loss_vs_epoch.png"); plt.close()

            st_lr = [(r["global_step"], r["lr"]) for r in self._step_records if r["phase"] == "train_step" and r.get("lr") is not None]
            if st_lr:
                x = [s for s, _ in st_lr]; y = [l for _, l in st_lr]
                plt.figure(figsize=(8, 4))
                plt.plot(x, y)
                plt.xlabel("global_step"); plt.ylabel("learning_rate"); plt.title("Learning Rate Schedule")
                plt.tight_layout(); plt.savefig(self.run_dir / "lr_vs_step.png"); plt.close()

            t1_tr = [r.get("train_top1") for r in self._epoch_records]
            t1_va = [r.get("val_top1") for r in self._epoch_records]
            if epochs and (any(x is not None for x in t1_tr) or any(x is not None for x in t1_va)):
                plt.figure(figsize=(8, 5))
                if t1_tr and any(x is not None for x in t1_tr):
                    plt.plot(range(len(t1_tr)), [x*100 if x is not None else None for x in t1_tr], label="train Top-1")
                if t1_va and any(x is not None for x in t1_va):
                    plt.plot(range(len(t1_va)), [x*100 if x is not None else None for x in t1_va], label="val Top-1")
                plt.xlabel("epoch"); plt.ylabel("Top-1 (%)"); plt.title("Top-1 vs Epoch")
                plt.legend(); plt.tight_layout()
                plt.savefig(self.run_dir / "top1_vs_epoch.png"); plt.close()
        except Exception as e:
            logger.warning(f"Plotting skipped: {e}")


# ============================== Lightning Module ============================== #
class MEGLitModule(pl.LightningModule):
    def __init__(self, model_cfg: dict, lr: float = 3e-4, weight_decay: float = 0.0,
                 warmup_ratio: float = 0.1, max_epochs: int = 100, optimizer_name: str = "adamw",
                 metric_logger: MetricLogger | None = None,
                 loss_use_l2: bool = False, loss_use_temp: bool = False,
                 metrics_every_n_steps: int = 50):
        super().__init__()
        self.save_hyperparameters(ignore=["metric_logger"])

        # Compatibility with new/old encoder:
        # - New versions do not have out_timesteps
        # - Old versions accept out_timesteps; set None to enforce "no temporal pooling"
        enc_sig = inspect.signature(UltimateMEGEncoder).parameters
        enc_cfg = dict(model_cfg)
        if "out_timesteps" in enc_sig:
            enc_cfg["out_timesteps"] = None  # strictly paper-aligned: no temporal pooling during training
        self.model = UltimateMEGEncoder(**enc_cfg)

        # Paper-aligned defaults: one-way; candidate-only L2; no temperature; no pooling
        if loss_use_l2:
            logger.warning("`--loss_l2` 将被忽略：论文路径仅做候选端（语音）L2 归一化，已内置。")
        self.loss_fn = PaperClipLoss(
            target_T=TARGET_T,
            pool=False,
            center=False,
            trim_min=None,
            trim_max=None,
            use_temperature=bool(loss_use_temp)
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_epochs = max_epochs
        self.optimizer_name = optimizer_name
        self.metric_logger = metric_logger
        self.metrics_every_n_steps = max(1, int(metrics_every_n_steps))

        # Cache per-epoch losses & metrics
        self._train_epoch_losses: List[float] = []
        self._val_epoch_losses: List[float] = []
        self._best_val: float | None = None

        # Accumulate epoch-level retrieval metrics
        self._train_metric_sums: Dict[str, float] = defaultdict(float)
        self._train_metric_count: int = 0
        self._val_metric_sums: Dict[str, float] = defaultdict(float)
        self._val_metric_count: int = 0

    # -------------- helpers -------------- #
    def forward(self, batch):
        # Local baseline only: do not pass any context-related arguments.
        return self.model(
            meg_win=batch["meg_win"],
            sensor_locs=batch["sensor_locs"],
            subj_idx=batch["subject_idx"],
        )

    def _assert_T(self, y: torch.Tensor):
        # Safety check: only require feature dim=1024; time dim is interpolated to 360 inside the loss.
        assert y.dim() == 3 and y.size(1) == 1024, \
            f"Encoder must output [B,1024,T], got {tuple(y.shape)}"

    def _accumulate_metrics(self, store: Dict[str, float], metrics: Dict[str, float], counter_name: str):
        for k, v in metrics.items():
            store[k] += float(v)
        if counter_name == "train":
            self._train_metric_count += 1
        else:
            self._val_metric_count += 1

    def _avg_metrics(self, store: Dict[str, float], count: int) -> Dict[str, float]:
        if count == 0:
            return {}
        return {k: (v / count) for k, v in store.items()}

    # -------------- epoch hooks -------------- #
    def on_train_epoch_start(self):
        self._train_epoch_losses = []
        self._train_metric_sums = defaultdict(float)
        self._train_metric_count = 0

    def on_validation_epoch_start(self):
        self._val_epoch_losses = []
        self._val_metric_sums = defaultdict(float)
        self._val_metric_count = 0

    # -------------- steps -------------- #
    def training_step(self, batch, batch_idx):
        meg_feat = self.forward(batch)              # [B,1024,T]
        self._assert_T(meg_feat)
        loss, logits = self.loss_fn(meg_feat, batch["audio"])
        self._train_epoch_losses.append(loss.detach().item())

        # Per-batch retrieval metrics (accumulate each step; write step logs at a chosen frequency)
        metrics = batch_retrieval_metrics(logits.detach(), ks=(1, 5, 10))
        self._accumulate_metrics(self._train_metric_sums, metrics, counter_name="train")

        # Step-level JSONL logging (subsampled)
        if self.metric_logger is not None:
            if (int(self.global_step) % self.metrics_every_n_steps) == 0:
                lr = None
                try:
                    if self.trainer is not None and self.trainer.optimizers:
                        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                except Exception:
                    pass
                self.metric_logger.write_step("train_step", int(self.global_step), int(self.current_epoch),
                                              float(loss.detach().item()), lr, metrics)

        # Lightning logging (progress bar)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch["meg_win"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        meg_feat = self.forward(batch)
        self._assert_T(meg_feat)
        loss, logits = self.loss_fn(meg_feat, batch["audio"])
        self._val_epoch_losses.append(loss.detach().item())

        metrics = batch_retrieval_metrics(logits.detach(), ks=(1, 5, 10))
        self._accumulate_metrics(self._val_metric_sums, metrics, counter_name="val")

        if self.metric_logger is not None:
            if (int(self.global_step) % self.metrics_every_n_steps) == 0:
                self.metric_logger.write_step("val_step", int(self.global_step), int(self.current_epoch),
                                              float(loss.detach().item()), None, metrics)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=batch["meg_win"].size(0))
        return {"val_loss": loss, "metrics": metrics}

    def test_step(self, batch, batch_idx):
        meg_feat = self.forward(batch)
        self._assert_T(meg_feat)
        loss, logits = self.loss_fn(meg_feat, batch["audio"])
        metrics = batch_retrieval_metrics(logits.detach(), ks=(1, 5, 10))
        if self.metric_logger is not None:
            self.metric_logger.write_step("test_step", int(self.global_step), int(self.current_epoch),
                                          float(loss.detach().item()), None, metrics)
        self.log("test/loss", loss, on_epoch=True, batch_size=batch["meg_win"].size(0))
        return {"test_loss": loss, "metrics": metrics}

    # -------------- epoch end -------------- #
    @staticmethod
    def _fmt4(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    def on_validation_epoch_end(self):
        # Epoch means
        tr = np.mean(self._train_epoch_losses).item() if self._train_epoch_losses else None
        va = np.mean(self._val_epoch_losses).item() if self._val_epoch_losses else None
        if va is not None:
            if self._best_val is None or va < self._best_val:
                self._best_val = va

        tr_metrics = self._avg_metrics(self._train_metric_sums, self._train_metric_count)
        va_metrics = self._avg_metrics(self._val_metric_sums, self._val_metric_count)

        # Epoch JSONL
        if self.metric_logger is not None:
            self.metric_logger.write_epoch(
                epoch=int(self.current_epoch),
                train_loss=tr, val_loss=va, best_val=self._best_val,
                train_metrics=tr_metrics, val_metrics=va_metrics
            )

        # Console summary table
        hdr = f"[Epoch {int(self.current_epoch):03d}]"
        line1 = f"{hdr} train_loss={self._fmt4(tr)} | val_loss={self._fmt4(va)} | best_val={self._fmt4(self._best_val)}"
        def mget(d, k): return None if not d else d.get(k)
        line2 = (
            f"Train: Top1={_fmt_pct(mget(tr_metrics,'top1'))}  Top5={_fmt_pct(mget(tr_metrics,'top5'))}  "
            f"Top10={_fmt_pct(mget(tr_metrics,'top10'))}  MRR={self._fmt4(mget(tr_metrics,'mrr'))}  "
            f"MeanRank={self._fmt4(mget(tr_metrics,'mean_rank'))}"
        )
        line3 = (
            f"Valid: Top1={_fmt_pct(mget(va_metrics,'top1'))}  Top5={_fmt_pct(mget(va_metrics,'top5'))}  "
            f"Top10={_fmt_pct(mget(va_metrics,'top10'))}  MRR={self._fmt4(mget(va_metrics,'mrr'))}  "
            f"MeanRank={self._fmt4(mget(va_metrics,'mean_rank'))}"
        )
        logger.info(line1); logger.info(line2); logger.info(line3)

    # -------------- optim -------------- #
    def configure_optimizers(self):
        # Optimizer selection
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Cosine schedule with warmup
        def lr_lambda(step: int):
            max_steps = max(1, self.trainer.estimated_stepping_batches)
            warmup = int(self.warmup_ratio * max_steps)
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, max_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ============================== Utils ============================== #
def map_amp_to_precision(amp: str) -> str | int:
    """Map command-line --amp to Lightning 'precision'."""
    amp = amp.lower()
    if amp in ("off", "none", "32", "fp32"):
        return 32  # FP32
    elif amp in ("fp16", "16", "16-mixed", "half"):
        return "16-mixed"
    elif amp in ("bf16", "bfloat16", "bf16-mixed"):
        return "bf16-mixed"
    else:
        raise ValueError(f"Unsupported --amp: {amp}")

def build_run_dir(experiment_name: str, default_root: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    jobid = os.environ.get("SLURM_JOB_ID")
    suffix = f"{experiment_name}_{ts}" + (f"_job{jobid}" if jobid else "")
    run_dir = Path(default_root) / suffix
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    return run_dir

@rank_zero_only
def save_records(run_dir: Path, cfg: dict, best_ckpt_path: str | None, subject_map_path: Optional[str] = None, registry: Optional[SubjectRegistry] = None):
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


# ============================== Main ============================== #
def parse_args():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--val_manifest", required=True, type=str)
    p.add_argument("--test_manifest", required=True, type=str)

    # Multi-dataset namespaces (optional; default empty)
    p.add_argument("--subject_namespace_train", type=str, default="", help="给 train manifest 中的 subject 加命名空间前缀")
    p.add_argument("--subject_namespace_val",   type=str, default="", help="给 val manifest 中的 subject 加命名空间前缀")
    p.add_argument("--subject_namespace_test",  type=str, default="", help="给 test manifest 中的 subject 加命名空间前缀")

    # Persistence / reproduction: subject mapping
    p.add_argument("--subject_mapping_path", type=str, default="", help="若存在则加载；否则用 train/val/test 联合集合构建并保存到此路径")

    # Batch / parallelism / precision
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--gpus", type=int, default=1, help="与 --devices 等价；保留兼容 sbatch")
    p.add_argument("--devices", type=int, default=None, help="优先于 --gpus；不传则用 --gpus")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", type=str, default="off", choices=["off", "bf16", "16-mixed", "fp16", "32"])

    # Optimization & schedule
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--gradient_clip_val", type=float, default=0.0)
    p.add_argument("--early_stop_patience", type=int, default=15)

    # LR linear scaling reference
    p.add_argument("--base_bsz_for_lr", type=int, default=256,
                   help="若与论文等效，设置为 256；实际 lr 会按 effective_bsz/base_bsz 线性缩放")

    # Model
    p.add_argument("--in_channels", type=int, default=270)
    p.add_argument("--n_subjects", type=int, default=None, help="若不指定，自动由 registry 推断")
    p.add_argument("--spatial_channels", type=int, default=270)
    p.add_argument("--fourier_k", type=int, default=32)
    p.add_argument("--d_model", type=int, default=320)
    p.add_argument("--out_channels", type=int, default=1024)
    p.add_argument("--backbone_depth", type=int, default=5)
    p.add_argument("--backbone_type", type=str, default="cnn", choices=["cnn", "conformer"])
    p.add_argument("--subject_layer_pos", type=str, choices=["early", "late", "none"], default="early")
    p.add_argument("--spatial_dropout_p", type=float, default=0.0)
    p.add_argument("--spatial_dropout_radius", type=float, default=0.2)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default_root_dir", type=str, default="runs")
    p.add_argument("--experiment_name", type=str, default="exp")
    p.add_argument("--normalize", action="store_true", help="若指定，则在加载时再做 z-score（默认关闭）")

    # Loss toggles
    p.add_argument("--loss_l2", action="store_true", help="（已忽略）若指定也会 WARNING：论文路径仅候选端 L2 归一化")
    p.add_argument("--loss_temp", action="store_true", help="使用可学习温度（默认否）")

    # Step-level logging frequency
    p.add_argument("--metrics_every_n_steps", type=int, default=50)

    # Unused placeholder (kept for CLI compatibility)
    p.add_argument("--steps_per_epoch", type=int, default=0, help="占位参数，当前不启用")

    return p.parse_args()


def main():
    args = parse_args()

    # Recommend enabling TF32 / high matmul precision on A100 for better throughput.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    pl.seed_everything(args.seed, workers=True)

    # Device / precision
    precision = map_amp_to_precision(args.amp)
    devices = args.devices if args.devices is not None else args.gpus
    devices = max(1, int(devices))

    # Run directory & metric logger
    run_dir = build_run_dir(args.experiment_name, args.default_root_dir)
    metric_logger = MetricLogger(run_dir)
    records_dir = run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    # ========= Subject Registry: load or build ========= #
    train_p = Path(args.train_manifest)
    val_p   = Path(args.val_manifest)
    test_p  = Path(args.test_manifest)
    ns_tr, ns_va, ns_te = args.subject_namespace_train, args.subject_namespace_val, args.subject_namespace_test

    map_path = Path(args.subject_mapping_path) if args.subject_mapping_path else (run_dir / "records" / "subject_mapping.json")

    if args.subject_mapping_path and Path(args.subject_mapping_path).exists():
        registry = SubjectRegistry.load(Path(args.subject_mapping_path))
        logger.info(f"Loaded subject mapping from: {args.subject_mapping_path} (num_subjects={registry.num_subjects})")
    else:
        registry = SubjectRegistry.build_from_manifests([
            (train_p, ns_tr),
            (val_p,   ns_va),
            (test_p,  ns_te),
        ])
        registry.save(map_path)
        logger.info(f"Built & saved subject mapping to: {map_path.as_posix()} (num_subjects={registry.num_subjects})")

    # ========= DataModule ========= #
    dm = MEGDataModule(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        registry=registry,
        ns_train=ns_tr, ns_val=ns_va, ns_test=ns_te,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize
    )
    dm.setup(None)

    # n_subjects defaults to registry size (union across train/val/test)
    n_subjects = args.n_subjects if args.n_subjects is not None else max(1, registry.num_subjects)

    # Model config (strict baseline: do not include any context-related params)
    model_cfg = dict(
        in_channels=args.in_channels,
        n_subjects=n_subjects,
        spatial_channels=args.spatial_channels,
        fourier_k=args.fourier_k,
        d_model=args.d_model,
        out_channels=args.out_channels,
        backbone_depth=args.backbone_depth,
        subject_layer_pos=args.subject_layer_pos,
        spatial_dropout_p=args.spatial_dropout_p,
        spatial_dropout_radius=args.spatial_dropout_radius,
        backbone_type=args.backbone_type,
    )

    # Effective batch size & LR linear scaling
    effective_bsz = args.batch_size * args.accumulate_grad_batches * devices
    scaled_lr = args.lr * (effective_bsz / max(1, args.base_bsz_for_lr))
    logger.info(f"train.jsonl loaded: {len(dm.train_set):,} samples")
    logger.info(f"valid.jsonl loaded: {len(dm.val_set):,} samples")
    logger.info(f"test.jsonl  loaded: {len(dm.test_set):,} samples")
    logger.info(f"SubjectRegistry size (union train/val/test) = {registry.num_subjects}")
    logger.info(f"Effective batch size = {args.batch_size} × {args.accumulate_grad_batches} × {devices} = {effective_bsz}")
    logger.info(f"LR scaled from {args.lr} -> {scaled_lr} (base_bsz_for_lr={args.base_bsz_for_lr})")

    lit = MEGLitModule(
        model_cfg=model_cfg,
        lr=scaled_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_epochs=args.max_epochs,
        optimizer_name=args.optimizer,
        metric_logger=metric_logger,
        loss_use_l2=args.loss_use_l2 if hasattr(args, "loss_use_l2") else False,  # CLI compatibility
        loss_use_temp=args.loss_temp,
        metrics_every_n_steps=args.metrics_every_n_steps
    )

    # Save config snapshot (for reproducibility)
    cfg_to_save = {
        "args": vars(args),
        "model_cfg": model_cfg,
        "effective_bsz": effective_bsz,
        "scaled_lr": scaled_lr,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "subject_mapping_path": str(map_path.resolve().as_posix()),
        "subject_namespaces": {"train": ns_tr, "val": ns_va, "test": ns_te},
    }
    save_records(run_dir, cfg_to_save, best_ckpt_path=None, subject_map_path=str(map_path), registry=registry)

    # Callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="meg-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    es_cb = EarlyStopping(monitor="val/loss", mode="min", patience=args.early_stop_patience, verbose=True)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices,
        precision=precision,
        default_root_dir=str(run_dir),
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=50,
        deterministic=False,
        gradient_clip_val=args.gradient_clip_val,
    )

    # Train & Test
    try:
        trainer.fit(lit, datamodule=dm)
    finally:
        # Always export any accumulated tables/plots, even if training is interrupted.
        metric_logger.export_tables_and_plots()

    # Write best checkpoint path
    best_path = ckpt_cb.best_model_path
    if not best_path:
        last = (run_dir / "checkpoints" / "last.ckpt")
        best_path = str(last) if last.exists() else ""
    save_records(run_dir, cfg_to_save, best_ckpt_path=best_path, subject_map_path=str(map_path), registry=registry)

    # Test using best checkpoint
    trainer.test(lit, datamodule=dm, ckpt_path="best")
    metric_logger.export_tables_and_plots()
    metric_logger.close()

    logger.info(f"Run directory: {run_dir.resolve().as_posix()}")
    logger.info(f"Subject mapping file: {Path(map_path).resolve().as_posix()}")


if __name__ == "__main__":
    main()
