#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EEG Training script (MEG-style encoder)

- 数据：仍然是 EEGDataModule（从 eeg_win_path / sensor_coordinates_path 读取）。
- 模型：复用 models.meg_encoder.UltimateMEGEncoder（局部分支，本质是 MEG encoder 风格）。
- Loss：PaperClipLoss，使用 Fixed Temperature（init_temp=0.07 -> scale≈14.3）。
- 去掉 Matplotlib，改用 CSV 记录，避免 HPC 上崩溃/卡死。
"""

import os, json, math, argparse, logging, inspect, csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities import rank_zero_only

# 直接复用 MEG-style UltimateMEGEncoder
from models.meg_encoder import UltimateMEGEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_eeg")

TARGET_T = 360  # 120 Hz * 3 s


# ============================== Subject Registry ============================== #
class SubjectRegistry:
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
        reg._order = order if order else [k for k, _ in sorted(mp.items(), key=lambda kv: kv[1])]
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
        return SubjectRegistry(mapping=mapping)


# ============================== Data ============================== #
def _ensure_brain_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expect 2D [C,T] or [T,C], got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

def _ensure_audio_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Audio must be 2D, got {x.shape}")
    if x.shape[0] == 1024:
        return x
    elif x.shape[1] == 1024:
        return x.T
    return x if x.shape[0] < x.shape[1] else x.T

class EEGDataset(Dataset):
    def __init__(self, manifest_path: str, registry: SubjectRegistry, namespace: str, normalize: bool = False):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists() or self.manifest_path.is_dir():
            raise FileNotFoundError(f"Manifest not found/file is dir: {self.manifest_path}")

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
        logger.info(f"{self.manifest_path.name} loaded: {len(self.samples):,} samples; "
                    f"subjects={len(seen)} (ns='{self.namespace}')")

    @staticmethod
    def _get_brain_path(s: dict) -> str:
        return s.get("meg_win_path") or s.get("eeg_win_path") or s.get("brain_win_path") or ""

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        brain_path = self._get_brain_path(s)
        if not brain_path:
            raise KeyError("Missing meg_win_path/eeg_win_path/brain_win_path in sample.")
        
        brain = np.load(brain_path).astype(np.float32)
        brain = _ensure_brain_CxT(brain)

        if self.normalize:
            m = brain.mean(axis=1, keepdims=True)
            sd = np.maximum(brain.std(axis=1, keepdims=True), 1e-6)
            brain = (brain - m) / sd

        aud = np.load(s["audio_feature_path"]).astype(np.float32)
        aud = _ensure_audio_CxT(aud)

        coord_path = s.get("sensor_coordinates_path", "")
        if not coord_path or not Path(coord_path).exists():
            raise RuntimeError("Missing sensor_coordinates_path.")
        if coord_path in self._coords_cache:
            coords = self._coords_cache[coord_path]
        else:
            coords = np.load(coord_path).astype(np.float32)
            if coords.ndim == 2 and coords.shape[1] == 2:
                coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)], axis=1)
            elif coords.ndim != 2 or coords.shape[1] not in (2, 3):
                raise RuntimeError(f"Unexpected sensor_locs shape: {coords.shape}")
            self._coords_cache[coord_path] = coords

        sid_str = str(s.get("subject_id"))
        subj_idx = self.registry.index_of(self.namespace, sid_str, fallback_to_first=True)

        return {
            "meg_win": torch.from_numpy(brain),        # [C,T]
            "audio_feature": torch.from_numpy(aud),    # [1024,T]
            "sensor_locs": torch.from_numpy(coords),   # [C,3]
            "subject_idx": torch.tensor(subj_idx, dtype=torch.long),
            "key": s.get("window_id", s.get("audio_key", "")),
        }


class EEGDataModule(pl.LightningDataModule):
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
        self.train_set = EEGDataset(self.train_manifest, registry=self.registry, namespace=self.ns_train, normalize=self.normalize)
        self.val_set   = EEGDataset(self.val_manifest,   registry=self.registry, namespace=self.ns_val,   normalize=self.normalize)
        self.test_set  = EEGDataset(self.test_manifest,  registry=self.registry, namespace=self.ns_test,  normalize=self.normalize)

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        col = defaultdict(list)
        for b in batch:
            for k, v in b.items():
                col[k].append(v)

        meg = torch.stack(col["meg_win"], dim=0)        # [B,C,T]
        aud = torch.stack(col["audio_feature"], dim=0)  # [B,1024,T]
        loc = torch.stack(col["sensor_locs"], dim=0)    # [B,C,3]
        sid = torch.stack(col["subject_idx"], dim=0)    # [B]
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


# ============================== Paper-aligned Loss (FIXED) ============================== #
class PaperClipLoss(nn.Module):
    def __init__(self, target_T: int = TARGET_T, pool: bool = False,
                 center: bool = False, trim_min: int | None = None, trim_max: int | None = None,
                 use_temperature: bool = False, init_temp: float = 0.07):
        super().__init__()
        self.target_T = target_T
        self.pool = pool
        self.center = center
        self.trim_min = trim_min
        self.trim_max = trim_max
        
        # init_temp = 0.07 => scale ~14.3
        val = math.log(1.0 / init_temp)
        if use_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(val, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(val, dtype=torch.float32))

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.target_T:
            return F.interpolate(x, size=self.target_T, mode='linear', align_corners=False)
        return x

    def forward(self, meg_f: torch.Tensor, aud_f: torch.Tensor):
        if meg_f.dim() != 3: meg_f = meg_f.unsqueeze(-1)
        if aud_f.dim() != 3: aud_f = aud_f.unsqueeze(-1)
             
        m = self._resize(meg_f)
        a = self._resize(aud_f)
        
        m_flat = m.reshape(m.size(0), -1)
        a_flat = a.reshape(a.size(0), -1)
        
        m_norm = F.normalize(m_flat, p=2, dim=1, eps=1e-6)
        a_norm = F.normalize(a_flat, p=2, dim=1, eps=1e-6)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = torch.matmul(m_norm, a_norm.T) * scale
            
        tgt = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, tgt)
        return loss, logits


# ============================== Metrics utils ============================== #
@torch.no_grad()
def batch_retrieval_metrics(logits: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    B = logits.size(0); device = logits.device
    ks = tuple(int(k) for k in ks if 1 <= k <= B)
    preds = logits.argsort(dim=1, descending=True)
    inv = preds.argsort(dim=1)
    tgt = torch.arange(B, device=device)
    ranks = inv[torch.arange(B, device=device), tgt]
    ranks_float = ranks.to(torch.float32)
    out = {f"top{k}": (ranks < k).float().mean().item() for k in ks}
    out["mrr"] = (1.0 / (ranks_float + 1.0)).mean().item()
    out["mean_rank"] = (ranks_float + 1.0).mean().item()
    return out

def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x*100:.2f}%"


# ============================== Logger (No Matplotlib) ============================== #
class MetricLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.step_log = self.run_dir / "metrics_step.jsonl"
        self.epoch_log = self.run_dir / "metrics_epoch.jsonl"
        self._step_records: List[dict] = []
        self._epoch_records: List[dict] = []
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
        rec = {"time": datetime.now(timezone.utc).isoformat(), "phase": phase,
               "global_step": int(step), "epoch": int(epoch),
               "loss": float(loss), "lr": float(lr) if lr is not None else None}
        if metrics:
            rec.update({k: float(v) for k, v in metrics.items()})
        try:
            self._fs.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fs.flush()
            self._step_records.append(rec)
        except Exception:
            pass

    def write_epoch(self, epoch: int, train_loss: float | None, val_loss: float | None,
                    best_val: float | None, train_metrics: Dict[str, float] | None,
                    val_metrics: Dict[str, float] | None):
        rec = {"time": datetime.now(timezone.utc).isoformat(), "epoch": int(epoch),
               "train_loss": None if train_loss is None else float(train_loss),
               "val_loss": None if val_loss is None else float(val_loss),
               "best_val_loss": None if best_val is None else float(best_val)}
        if train_metrics:
            rec.update({f"train_{k}": float(v) for k, v in train_metrics.items()})
        if val_metrics:
            rec.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        try:
            self._fe.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fe.flush()
            self._epoch_records.append(rec)
        except Exception:
            pass

    @rank_zero_only
    def export_tables(self):
        step_csv = self.run_dir / "metrics_step.csv"
        epoch_csv = self.run_dir / "metrics_epoch.csv"
        try:
            if self._step_records:
                keys = sorted({k for r in self._step_records for k in r.keys()})
                with open(step_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(self._step_records)
            if self._epoch_records:
                keys = sorted({k for r in self._epoch_records for k in r.keys()})
                with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(self._epoch_records)
        except Exception as e:
            logger.warning(f"CSV export failed: {e}")


# ============================== Lightning Module ============================== #
class EEGLitModule(pl.LightningModule):
    def __init__(self, model_cfg: dict, lr: float = 3e-4, weight_decay: float = 0.0,
                 warmup_ratio: float = 0.1, max_epochs: int = 100, optimizer_name: str = "adamw",
                 metric_logger: MetricLogger | None = None,
                 loss_use_temp: bool = False, metrics_every_n_steps: int = 50):
        super().__init__()
        self.save_hyperparameters(ignore=["metric_logger"])

        enc_sig = inspect.signature(UltimateMEGEncoder).parameters
        enc_cfg = dict(model_cfg)
        if "out_timesteps" in enc_sig and enc_cfg.get("out_timesteps") is None:
            pass

        # 使用 MEG-style UltimateMEGEncoder
        self.model = UltimateMEGEncoder(**enc_cfg)

        self.loss_fn = PaperClipLoss(target_T=TARGET_T, pool=False, center=False,
                                     trim_min=None, trim_max=None, use_temperature=bool(loss_use_temp))
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_epochs = max_epochs
        self.optimizer_name = optimizer_name
        self.metric_logger = metric_logger
        self.metrics_every_n_steps = max(1, int(metrics_every_n_steps))

        self._train_epoch_losses: List[float] = []
        self._val_epoch_losses: List[float] = []
        self._best_val: float | None = None
        self._train_metric_sums: Dict[str, float] = defaultdict(float); self._train_metric_count: int = 0
        self._val_metric_sums: Dict[str, float] = defaultdict(float);   self._val_metric_count: int = 0

    def forward(self, batch):
        # 仅使用 local 分支：返回 [B, out_channels, T]
        return self.model(meg_win=batch["meg_win"], sensor_locs=batch["sensor_locs"], subj_idx=batch["subject_idx"])

    def _assert_T(self, y: torch.Tensor):
        assert y.dim() == 3 and y.size(1) == 1024, f"Encoder must output [B,1024,T], got {tuple(y.shape)}"

    def _acc(self, store: Dict[str, float], metrics: Dict[str, float], which: str):
        for k, v in metrics.items(): store[k] += float(v)
        if which == "train": self._train_metric_count += 1
        else: self._val_metric_count += 1

    def _avg(self, store: Dict[str, float], cnt: int) -> Dict[str, float]:
        return {} if cnt == 0 else {k: (v / cnt) for k, v in store.items()}

    def training_step(self, batch, batch_idx):
        meg_feat = self.forward(batch)
        self._assert_T(meg_feat)
        
        loss, logits = self.loss_fn(meg_feat, batch["audio"])
        
        self._train_epoch_losses.append(loss.detach().item())
        metrics = batch_retrieval_metrics(logits.detach(), ks=(1,5,10))
        self._acc(self._train_metric_sums, metrics, "train")
        
        if self.trainer.is_global_zero and self.metric_logger is not None:
            if (int(self.global_step) % self.metrics_every_n_steps) == 0:
                lr = None
                try:
                    if self.trainer.optimizers:
                        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                except Exception:
                    pass
                self.metric_logger.write_step(
                    "train_step", int(self.global_step), int(self.current_epoch),
                    float(loss.detach().item()), lr, metrics
                )
        
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=batch["meg_win"].size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        meg_feat = self.forward(batch)
        self._assert_T(meg_feat)
        
        loss, logits = self.loss_fn(meg_feat, batch["audio"])
        
        self._val_epoch_losses.append(loss.detach().item())
        metrics = batch_retrieval_metrics(logits.detach(), ks=(1,5,10))
        self._acc(self._val_metric_sums, metrics, "val")
        
        if self.trainer.is_global_zero and self.metric_logger is not None:
            if (batch_idx % self.metrics_every_n_steps) == 0:
                self.metric_logger.write_step(
                    "val_step", int(self.global_step), int(self.current_epoch),
                    float(loss.detach().item()), None, metrics
                )
        
        self.log("val/loss", loss, prog_bar=True, on_epoch=True,
                 batch_size=batch["meg_win"].size(0))
        return {"val_loss": loss, "metrics": metrics}

    def test_step(self, batch, batch_idx):
        meg_feat = self.forward(batch)
        self._assert_T(meg_feat)
        
        loss, logits = self.loss_fn(meg_feat, batch["audio"])
        metrics = batch_retrieval_metrics(logits.detach(), ks=(1,5,10))
        
        if self.trainer.is_global_zero and self.metric_logger is not None:
            self.metric_logger.write_step(
                "test_step", int(self.global_step), int(self.current_epoch),
                float(loss.detach().item()), None, metrics
            )
        
        self.log("test/loss", loss, on_epoch=True,
                 batch_size=batch["meg_win"].size(0))
        return {"test_loss": loss, "metrics": metrics}

    @staticmethod
    def _fmt4(x: float | None) -> str: return "n/a" if x is None else f"{x:.4f}"

    def on_validation_epoch_end(self):
        tr = np.mean(self._train_epoch_losses).item() if self._train_epoch_losses else None
        va = np.mean(self._val_epoch_losses).item() if self._val_epoch_losses else None
        
        if va is not None and (self._best_val is None or va < self._best_val): 
            self._best_val = va
        
        trm = self._avg(self._train_metric_sums, self._train_metric_count)
        vam = self._avg(self._val_metric_sums, self._val_metric_count)
        
        if self.trainer.is_global_zero:
            if self.metric_logger is not None:
                self.metric_logger.write_epoch(int(self.current_epoch), tr, va, self._best_val, trm, vam)
                
            def mget(d,k): return None if not d else d.get(k)
            
            logger.info(
                f"[Epoch {int(self.current_epoch):03d}] "
                f"train_loss={self._fmt4(tr)} | val_loss={self._fmt4(va)} | best_val={self._fmt4(self._best_val)}"
            )
            logger.info(
                f"Train: Top1={_fmt_pct(mget(trm,'top1'))}  "
                f"Top5={_fmt_pct(mget(trm,'top5'))}  "
                f"Top10={_fmt_pct(mget(trm,'top10'))}  "
                f"MRR={self._fmt4(mget(trm,'mrr'))}  "
                f"MeanRank={self._fmt4(mget(trm,'mean_rank'))}"
            )
            logger.info(
                f"Valid: Top1={_fmt_pct(mget(vam,'top1'))}  "
                f"Top5={_fmt_pct(mget(vam,'top5'))}  "
                f"Top10={_fmt_pct(mget(vam,'top10'))}  "
                f"MRR={self._fmt4(mget(vam,'mrr'))}  "
                f"MeanRank={self._fmt4(mget(vam,'mean_rank'))}"
            )
        
        self._train_epoch_losses.clear()
        self._val_epoch_losses.clear()
        self._train_metric_sums.clear(); self._train_metric_count = 0
        self._val_metric_sums.clear();   self._val_metric_count = 0

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def lr_lambda(step: int):
            max_steps = max(1, self.trainer.estimated_stepping_batches)
            warmup = int(self.warmup_ratio * max_steps)
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, max_steps - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


# ============================== Utils & Main ============================== #
def map_amp_to_precision(amp: str) -> str | int:
    amp = amp.lower()
    if amp in ("off","none","32","fp32"): return 32
    elif amp in ("fp16","16","16-mixed","half"): return "16-mixed"
    elif amp in ("bf16","bfloat16","bf16-mixed"): return "bf16-mixed"
    else: raise ValueError(f"Unsupported --amp: {amp}")

def build_run_dir(experiment_name: str, default_root: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    jobid = os.environ.get("SLURM_JOB_ID")
    suffix = f"{experiment_name}_{ts}" + (f"_job{jobid}" if jobid else "")
    run_dir = Path(default_root) / suffix
    (run_dir / "records").mkdir(parents=True, exist_ok=True)
    return run_dir

@rank_zero_only
def save_records(run_dir: Path, cfg: dict, best_ckpt_path: str | None,
                 subject_map_path: Optional[str] = None,
                 registry: Optional[SubjectRegistry] = None):
    rec_dir = run_dir / "records"; rec_dir.mkdir(parents=True, exist_ok=True)
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

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--val_manifest",   required=True, type=str)
    p.add_argument("--test_manifest",  required=True, type=str)
    # Namespace
    p.add_argument("--subject_namespace_train", type=str, default="")
    p.add_argument("--subject_namespace_val",   type=str, default="")
    p.add_argument("--subject_namespace_test",  type=str, default="")
    # Mapping
    p.add_argument("--subject_mapping_path", type=str, default="")
    
    # Training Params
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--devices", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", type=str, default="off", choices=["off","bf16","16-mixed","fp16","32"])
    
    # Optimization
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam","adamw"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--gradient_clip_val", type=float, default=0.0)
    p.add_argument("--early_stop_patience", type=int, default=15)
    p.add_argument("--base_bsz_for_lr", type=int, default=256)
    
    # Model General
    p.add_argument("--in_channels", type=int, default=60)
    p.add_argument("--n_subjects", type=int, default=None)
    p.add_argument("--out_channels", type=int, default=1024)
    
    # MEG-style Encoder 参数（与 MEG train.train 对齐）
    p.add_argument("--spatial_channels", type=int, default=270)
    p.add_argument("--fourier_k", type=int, default=32)
    p.add_argument("--d_model", type=int, default=320)
    p.add_argument("--backbone_depth", type=int, default=5)
    p.add_argument("--backbone_type", type=str, default="cnn",
                   choices=["cnn", "conformer"])
    p.add_argument("--dropout", type=float, default=0.1)
    
    p.add_argument("--use_subjects", action="store_true")
    p.add_argument("--subject_layer_pos", type=str, default="early",
                   choices=["early","late","none"])
    p.add_argument("--spatial_dropout_p", type=float, default=0.0)
    p.add_argument("--spatial_dropout_radius", type=float, default=0.2)

    # 其他杂项
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default_root_dir", type=str, default="runs")
    p.add_argument("--experiment_name", type=str, default="eeg_exp")
    p.add_argument("--normalize", action="store_true")
    # Loss Options
    p.add_argument("--loss_temp", action="store_true", help="Use learnable temperature")
    # Logging
    p.add_argument("--metrics_every_n_steps", type=int, default=50)
    
    return p.parse_args()

def main():
    args = parse_args()
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    pl.seed_everything(args.seed, workers=True)

    precision = map_amp_to_precision(args.amp)
    devices = args.devices if args.devices is not None else args.gpus
    devices = max(1, int(devices))

    run_dir = build_run_dir(args.experiment_name, args.default_root_dir)
    metric_logger = MetricLogger(run_dir)
    records_dir = run_dir / "records"; records_dir.mkdir(parents=True, exist_ok=True)

    # SubjectRegistry
    train_p, val_p, test_p = Path(args.train_manifest), Path(args.val_manifest), Path(args.test_manifest)
    ns_tr, ns_va, ns_te = args.subject_namespace_train, args.subject_namespace_val, args.subject_namespace_test
    map_path = Path(args.subject_mapping_path) if args.subject_mapping_path else (run_dir / "records" / "subject_mapping.json")
    if args.subject_mapping_path and Path(args.subject_mapping_path).exists():
        registry = SubjectRegistry.load(Path(args.subject_mapping_path))
        logger.info(f"Loaded subject mapping: {args.subject_mapping_path} (num_subjects={registry.num_subjects})")
    else:
        registry = SubjectRegistry.build_from_manifests([(train_p, ns_tr), (val_p, ns_va), (test_p, ns_te)])
        registry.save(map_path)
        logger.info(f"Built & saved subject mapping to: {map_path.as_posix()} (num_subjects={registry.num_subjects})")

    # DataModule
    dm = EEGDataModule(args.train_manifest, args.val_manifest, args.test_manifest,
                       registry=registry, ns_train=ns_tr, ns_val=ns_va, ns_test=ns_te,
                       batch_size=args.batch_size, num_workers=args.num_workers, normalize=args.normalize)
    dm.setup(None)

    # 自动 override in_channels
    sample = dm.train_set[0] if len(dm.train_set) > 0 else dm.val_set[0]
    auto_C = int(sample["meg_win"].shape[0])

    if auto_C != args.in_channels:
        logger.warning(
            f"[Sanity] Detected EEG channels={auto_C}, overriding "
            f"in_channels={args.in_channels} -> {auto_C}"
        )
        args.in_channels = auto_C

    n_subjects = args.n_subjects if args.n_subjects is not None else max(1, registry.num_subjects)

    # 配置给 MEG-style UltimateMEGEncoder（只用 local 分支，context_mode='none'）
    model_cfg = dict(
        in_channels=args.in_channels,
        n_subjects=n_subjects,
        spatial_channels=args.spatial_channels,
        fourier_k=args.fourier_k,
        d_model=args.d_model,
        text_dim=None,
        out_channels=args.out_channels,
        backbone_depth=args.backbone_depth,
        backbone_type=args.backbone_type,
        subject_layer_pos=args.subject_layer_pos,
        use_subjects=args.use_subjects,
        spatial_dropout_p=args.spatial_dropout_p,
        spatial_dropout_radius=args.spatial_dropout_radius,
        nhead=8,
        dropout=args.dropout,
        out_timesteps=None,
        readout_dropout=0.0,
        context_mode="none",
        context_memory_len=0,
        mem_enc_layers=0,
        mem_enc_heads=8,
        mem_dropout_p=0.0,
        freeze_ctx_local=True,
        detach_context=False,
        global_frontend="separate_full",
        warm_start_global=False,
        slot_agg="mean",
        ctx_token_mbatch=64,
        pre_down_tcap=0,
        token_pool_max_T=0,
    )

    effective_bsz = args.batch_size * args.accumulate_grad_batches * devices
    scaled_lr = args.lr * (effective_bsz / max(1, args.base_bsz_for_lr))
    logger.info(f"train: {len(dm.train_set):,} | valid: {len(dm.val_set):,} | test: {len(dm.test_set):,}")
    logger.info(f"SubjectRegistry (union) = {registry.num_subjects}")
    logger.info(f"Effective BSZ = {args.batch_size} × {args.accumulate_grad_batches} × {devices} = {effective_bsz}")
    logger.info(f"LR scaled from {args.lr} -> {scaled_lr} (base_bsz_for_lr={args.base_bsz_for_lr})")

    lit = EEGLitModule(
        model_cfg=model_cfg,
        lr=scaled_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_epochs=args.max_epochs,
        optimizer_name=args.optimizer,
        metric_logger=metric_logger,
        loss_use_temp=args.loss_temp,
        metrics_every_n_steps=args.metrics_every_n_steps,
    )

    cfg_to_save = {
        "args": vars(args),
        "model_cfg": model_cfg,
        "effective_bsz": effective_bsz,
        "scaled_lr": scaled_lr,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "subject_mapping_path": str(map_path.resolve().as_posix()),
        "subject_namespaces": {"train": ns_tr, "val": ns_va, "test": ns_te},
    }
    save_records(run_dir, cfg_to_save, best_ckpt_path=None,
                 subject_map_path=str(map_path), registry=registry)

    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="eeg-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss", mode="min",
        save_top_k=-1, save_last=True,
    )
    es_cb = EarlyStopping(
        monitor="val/loss", mode="min",
        patience=args.early_stop_patience, verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices, precision=precision,
        default_root_dir=str(run_dir),
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ckpt_cb, es_cb],
        log_every_n_steps=50,
        deterministic=False,
        gradient_clip_val=args.gradient_clip_val,
    )

    try:
        trainer.fit(lit, datamodule=dm)
    finally:
        if metric_logger:
            metric_logger.export_tables()

    best_path = ckpt_cb.best_model_path
    if not best_path:
        last = (run_dir / "checkpoints" / "last.ckpt")
        best_path = str(last) if last.exists() else ""
    save_records(run_dir, cfg_to_save, best_ckpt_path=best_path,
                 subject_map_path=str(map_path), registry=registry)

    trainer.test(lit, datamodule=dm, ckpt_path="best")
    if metric_logger:
        metric_logger.export_tables()
        metric_logger.close()
    
    logger.info(f"Run directory: {run_dir.resolve().as_posix()}")
    logger.info(f"Subject mapping file: {Path(map_path).resolve().as_posix()}")

if __name__ == "__main__":
    main()
