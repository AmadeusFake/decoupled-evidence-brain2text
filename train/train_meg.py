#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script (paper-aligned + clean metrics/logging + subject registry)

论文/官方实现对齐要点（本脚本所用）：
- 单向对比（脑->语音），同 GPU 批内作为负样本
- 两端线性插值到 T=360；**不做时间池化**（保留时序）
- 打分方式：**仅 candidates（语音端）按 (C,T) 做 L2 归一化**，estimates（MEG 端）不归一化；
  logits[b, o] = < MEG_b , AUDIO_o / ||AUDIO_o||_2 >
- 默认**无温度**；可通过 --loss_temp 打开可学习温度
- 可选开关（默认关闭）：center（均值移除）、trim（时间裁切）

其它工程增强（不改训练逻辑）：
1) SubjectRegistry：稳定被试索引（支持多数据集命名空间）
2) 更清爽的日志与指标（Top-1/5/10、MRR、MeanRank）
3) 导出 CSV/XLSX/PNG；控制台 epoch 表格
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

# ---- Matplotlib 无显示后端（HPC 友好） ----
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

# 你的模型（UltimateMEGEncoder）——这里用新的 SimpleConv 版本
from models.meg_encoder_ExpDilated import UltimateMEGEncoder
# 如果你是覆盖了原来的 meg_encoder.py，请改成：
# from models.meg_encoder import UltimateMEGEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train")

TARGET_T = 360  # 120 Hz * 3 s


# ============================== Subject Registry ============================== #
class SubjectRegistry:
    """
    稳定的“被试 -> 索引”注册表，支持多数据集命名空间。
    - 组合键： "<namespace>:<subject_id>"（namespace 为空则直接 subject_id）
    - 构建：从若干 manifest（及各自 namespace）收集 subject_id 的并集，排序后稳定编号
    - 持久化：save/load 到 JSON，复现实验、后续检索/评测时保证一致
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
        # 未登记的被试：不扩容模型，回退到第一个索引（影响极小）
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
    """将 2D 矩阵调整为“通道×时间”布局：MEG 通常 C≈270, T≈360，Heuristic：行数<=列数视为 [C,T]。"""
    if x.ndim != 2:
        raise ValueError(f"Expect 2D [C,T] or [T,C], got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T


class MEGDataset(Dataset):
    """
    读取 jsonl manifest（Stage-3 fully_preprocessed）：
    - meg_win_path: 归一化+clamp 后的 [C,T] float32
    - audio_feature_path: [1024,T] float32（Stage-2 已插值到 T=360；若不是360，这里与 loss 内部会插值）
    - sensor_coordinates_path: [C,3] float32，x,y ∈ [0,1]，z=0
    - subject_id: 标识被试（与 SubjectRegistry 的 namespace 组合成稳定键）
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
            # 仅用于 ablation；论文路径默认不二次标准化（Stage-3 已 robust+clamp）
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
            "audio_feature": torch.from_numpy(aud),   # [T,1024] or [1024,T]，loss 内处理
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
        aud = torch.stack(col["audio_feature"], dim=0) # [B,?,?]（loss 内纠正为 [B,D,T]）
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
    论文/官方对齐版对比损失（单向：脑->语音）
    - 仅同 GPU 批内为负样本
    - 对比前按线性插值到 T=360
    - **仅 candidates（语音端）做 L2 归一化**（按 (C,T)），MEG 端不归一化
    - **不做时间池化**（保留时序）；可选：center/trim（默认关闭）
    - 默认无温度；如需可通过 --loss_temp 打开（learnable）
    """
    def __init__(
        self,
        target_T: int = TARGET_T,
        pool: bool = False,          # 保持 False（不池化）
        center: bool = False,        # 默认关闭（可复现实验时再开）
        trim_min: int | None = None, # 默认不裁剪
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
        # 接受 [B,D,T] 或 [B,T,D]；返回 [B,C(=D),T]
        if x.dim() != 3:
            raise ValueError(f"Expect 3D, got {tuple(x.shape)}")
        B, A, C = x.shape
        return x if A >= C else x.transpose(1, 2).contiguous()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        # 统一到 [B,C,T] 并插值到 target_T；可选裁剪/池化/均值移除
        x = self._to_BCT(x.to(torch.float32))
        if x.size(-1) != self.target_T:
            x = F.interpolate(x, size=self.target_T, mode="linear", align_corners=False)
        # 裁剪（样本点索引）
        if (self.trim_min is not None) or (self.trim_max is not None):
            t0 = 0 if self.trim_min is None else max(0, int(self.trim_min))
            t1 = x.size(-1) if self.trim_max is None else min(x.size(-1), int(self.trim_max))
            x = x[..., t0:t1]
        if self.pool:
            x = x.mean(dim=2, keepdim=True)  # 不建议开：会丢时序
        if self.center:
            x = x - x.mean(dim=(1, 2), keepdim=True)
        return x

    def forward(self, meg_f: torch.Tensor, aud_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) 预处理（插值/可选裁剪/可选中心化）
        m = self._prep(meg_f)  # estimates:  [B,C,T']
        a = self._prep(aud_f)  # candidates: [B,C,T']

        # 2) 仅 candidates 做 L2 归一化（按 (C,T)）
        inv_norms = (a.norm(dim=(1, 2), p=2) + 1e-8).reciprocal()  # [B]

        # 3) 打分：与官方 ClipLoss 等价的 einsum 写法
        #    logits[b, o] = sum_{c,t} m[b,c,t] * ( a[o,c,t] / ||a[o]|| )
        logits = torch.einsum("bct,oct,o->bo", m, a, inv_norms)

        # 4) 可选温度（默认关闭；开启时与常见 CLIP 一致）
        if self.logit_scale is not None:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)

        tgt = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, tgt)
        return loss, logits


# ============================== Metrics utils ============================== #
@torch.no_grad()
def batch_retrieval_metrics(logits: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    """
    基于 batch 内 logits（[B,B]，行=MEG，列=Audio）计算检索指标。
    返回：{'top1':..., 'top5':..., 'top10':..., 'mrr':..., 'mean_rank':...}
    """
    B = logits.size(0)
    device = logits.device
    ks = tuple(int(k) for k in ks if k >= 1)
    ks = tuple(k for k in ks if k <= B)
    preds = logits.argsort(dim=1, descending=True)           # [B,B]
    inv = preds.argsort(dim=1)                               # [B,B] 逆置换
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
        """把 JSONL 汇总为 CSV/XLSX，并画 PNG 曲线（loss、lr、Top-1）"""
        import csv
        # CSV
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

        # XLSX（若缺依赖，自动跳过）
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

        # 画图：loss（epoch） & lr（step） & Top-1（epoch）
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

        # —— 兼容新/旧 Encoder：新版没有 out_timesteps；旧版若有则传 None 以保持“不池化” —— #
        enc_sig = inspect.signature(UltimateMEGEncoder).parameters
        enc_cfg = dict(model_cfg)
        if "out_timesteps" in enc_sig:
            enc_cfg["out_timesteps"] = None  # 训练阶段严格按论文：不做时间池化
        self.model = UltimateMEGEncoder(**enc_cfg)

        # 论文对齐：单向、候选端 L2 归一化、默认无温度、不池化
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

        # 缓存当 epoch 的 loss & metrics
        self._train_epoch_losses: List[float] = []
        self._val_epoch_losses: List[float] = []
        self._best_val: float | None = None

        # 累积 epoch 级检索指标
        self._train_metric_sums: Dict[str, float] = defaultdict(float)
        self._train_metric_count: int = 0
        self._val_metric_sums: Dict[str, float] = defaultdict(float)
        self._val_metric_count: int = 0

    # -------------- helpers -------------- #
    def forward(self, batch):
        # 仅本地基线：不传任何上下文相关参数
        return self.model(
            meg_win=batch["meg_win"],
            sensor_locs=batch["sensor_locs"],
            subj_idx=batch["subject_idx"],
        )

    def _assert_T(self, y: torch.Tensor):
        # 保险：只要求特征维=1024；时间维由损失内部插值到 360
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

        # 批级检索指标（每步累积；按频率写 step 日志）
        metrics = batch_retrieval_metrics(logits.detach(), ks=(1, 5, 10))
        self._accumulate_metrics(self._train_metric_sums, metrics, counter_name="train")

        # step 级 JSONL（抽样写）
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

        # Lightning 日志（用于进度条）
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
        # 均值
        tr = np.mean(self._train_epoch_losses).item() if self._train_epoch_losses else None
        va = np.mean(self._val_epoch_losses).item() if self._val_epoch_losses else None
        if va is not None:
            if self._best_val is None or va < self._best_val:
                self._best_val = va

        tr_metrics = self._avg_metrics(self._train_metric_sums, self._train_metric_count)
        va_metrics = self._avg_metrics(self._val_metric_sums, self._val_metric_count)

        # JSONL（epoch）
        if self.metric_logger is not None:
            self.metric_logger.write_epoch(
                epoch=int(self.current_epoch),
                train_loss=tr, val_loss=va, best_val=self._best_val,
                train_metrics=tr_metrics, val_metrics=va_metrics
            )

        # 控制台对齐表格
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
        # 选择优化器
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 余弦+warmup 学习率
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
    """把命令行参数 --amp 映射到 Lightning 的 precision 选项"""
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

    # 数据
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--val_manifest", required=True, type=str)
    p.add_argument("--test_manifest", required=True, type=str)

    # 多数据集命名空间（可选；默认空串）
    p.add_argument("--subject_namespace_train", type=str, default="", help="给 train manifest 中的 subject 加命名空间前缀")
    p.add_argument("--subject_namespace_val",   type=str, default="", help="给 val manifest 中的 subject 加命名空间前缀")
    p.add_argument("--subject_namespace_test",  type=str, default="", help="给 test manifest 中的 subject 加命名空间前缀")

    # 持久化/复现：被试映射
    p.add_argument("--subject_mapping_path", type=str, default="", help="若存在则加载；否则用 train/val/test 联合集合构建并保存到此路径")

    # 批量/并行/精度
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--gpus", type=int, default=1, help="与 --devices 等价；保留兼容 sbatch")
    p.add_argument("--devices", type=int, default=None, help="优先于 --gpus；不传则用 --gpus")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", type=str, default="off", choices=["off", "bf16", "16-mixed", "fp16", "32"])

    # 优化 & 日程
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--gradient_clip_val", type=float, default=0.0)
    p.add_argument("--early_stop_patience", type=int, default=15)

    # LR 线性缩放参考
    p.add_argument("--base_bsz_for_lr", type=int, default=256,
                   help="若与论文等效，设置为 256；实际 lr 会按 effective_bsz/base_bsz 线性缩放")

    # ================= 模型（对齐 BrainMagick simpleconv） =================
    p.add_argument("--in_channels", type=int, default=270)
    p.add_argument("--n_subjects", type=int, default=None, help="若不指定，自动由 registry 推断")
    p.add_argument("--spatial_channels", type=int, default=270)
    p.add_argument("--fourier_k", type=int, default=32)
    p.add_argument("--d_model", type=int, default=320)
    p.add_argument("--out_channels", type=int, default=1024)

    # SimpleConv 主干参数（对齐 config.yaml.simpleconv）
    p.add_argument("--backbone_depth", type=int, default=10, help="simpleconv.depth = 10")
    p.add_argument("--backbone_kernel", type=int, default=3, help="simpleconv.kernel_size = 3")
    p.add_argument("--dilation_period", type=int, default=5, help="simpleconv.dilation_period = 5 （1,2,4,8,16 周期）")
    p.add_argument("--glu_mult", type=int, default=2, help="simpleconv.glu = 2")
    p.add_argument("--backbone_dropout", type=float, default=0.0, help="simpleconv.dropout = 0.0")

    # 旧参数保留（虽然在新 Encoder 中多数不会用到）
    p.add_argument("--backbone_type", type=str, default="cnn", choices=["cnn", "conformer"])
    p.add_argument("--subject_layer_pos", type=str, choices=["early", "late", "none"], default="early")
    p.add_argument("--spatial_dropout_p", type=float, default=0.0)
    p.add_argument("--spatial_dropout_radius", type=float, default=0.2)

    # 其它
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default_root_dir", type=str, default="runs")
    p.add_argument("--experiment_name", type=str, default="exp")
    p.add_argument("--normalize", action="store_true", help="若指定，则在加载时再做 z-score（默认关闭）")

    # Loss 可选开关
    p.add_argument("--loss_l2", action="store_true", help="（已忽略）若指定也会 WARNING：论文路径仅候选端 L2 归一化")
    p.add_argument("--loss_temp", action="store_true", help="使用可学习温度（默认否）")

    # 步级日志频率
    p.add_argument("--metrics_every_n_steps", type=int, default=50)

    # 兼容你命令行的冗余参数（当前不使用）
    p.add_argument("--steps_per_epoch", type=int, default=0, help="占位参数，当前不启用")

    return p.parse_args()


def main():
    args = parse_args()

    # 建议打开 A100 的 TF32/高精度 matmul（性能更好）
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    pl.seed_everything(args.seed, workers=True)

    # 设备/精度
    precision = map_amp_to_precision(args.amp)
    devices = args.devices if args.devices is not None else args.gpus
    devices = max(1, int(devices))

    # 运行目录 & 记录器
    run_dir = build_run_dir(args.experiment_name, args.default_root_dir)
    metric_logger = MetricLogger(run_dir)
    records_dir = run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    # ========= Subject Registry：加载或构建 ========= #
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

    # 推断 n_subjects：默认采用 registry 大小（覆盖 train/val/test 的并集）
    n_subjects = args.n_subjects if args.n_subjects is not None else max(1, registry.num_subjects)

    # ================= 模型 cfg =================
    model_cfg = dict(
        in_channels=args.in_channels,
        n_subjects=n_subjects,
        spatial_channels=args.spatial_channels,
        fourier_k=args.fourier_k,
        d_model=args.d_model,
        out_channels=args.out_channels,

        # SimpleConv 主干
        backbone_depth=args.backbone_depth,
        backbone_kernel=args.backbone_kernel,
        dilation_period=args.dilation_period,
        glu_mult=args.glu_mult,
        dropout=args.backbone_dropout,

        # 被试/空间 dropout
        subject_layer_pos=args.subject_layer_pos,
        spatial_dropout_p=args.spatial_dropout_p,
        spatial_dropout_radius=args.spatial_dropout_radius,

        # 兼容而已，在新 Encoder 中会被忽略
        backbone_type=args.backbone_type,
    )

    # 有效 batch & LR 线性缩放
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
        loss_use_l2=args.loss_use_l2 if hasattr(args, "loss_use_l2") else False,  # 兼容
        loss_use_temp=args.loss_temp,
        metrics_every_n_steps=args.metrics_every_n_steps
    )

    # 保存配置（供复现实验使用）
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
        # 不管是否中断，都导出当前已有的表格与图
        metric_logger.export_tables_and_plots()

    # 写 best ckpt 路径
    best_path = ckpt_cb.best_model_path
    if not best_path:
        last = (run_dir / "checkpoints" / "last.ckpt")
        best_path = str(last) if last.exists() else ""
    save_records(run_dir, cfg_to_save, best_ckpt_path=best_path, subject_map_path=str(map_path), registry=registry)

    # Test（使用 best ckpt）
    trainer.test(lit, datamodule=dm, ckpt_path="best")
    metric_logger.export_tables_and_plots()
    metric_logger.close()

    logger.info(f"Run directory: {run_dir.resolve().as_posix()}")
    logger.info(f"Subject mapping file: {Path(map_path).resolve().as_posix()}")


if __name__ == "__main__":
    main()
