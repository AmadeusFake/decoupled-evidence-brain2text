#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train UltimateMEGEncoder (+ Contextual Re-ranking bias) — no MSM dependency.

稳定方案（分数加权）：
  base logits = q_raw · L2(a) * exp(logit_scale)
  ctx  bias   = σ(α) * (ctx_proj · audio_bank^T)    # audio_bank=整批候选
  logits_final = base + bias

改动要点（2025-09-13）：
- 新增：可训练上下文分支（按 epoch 打开），不再强制 no_grad。
- 新增：bias 相关控制项（起用 epoch、对角模式、行内中心化、上限裁剪、bank reduce=rms/mean）。
- 新增：logit_scale 的 LR 可乘一个倍数（默认 0.1），避免其干扰 bias。
- 调整：ContextReRankHead 的 alpha 初始值移到模型文件（-3.0）。
"""

from __future__ import annotations
import math, argparse, logging, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.meg_encoder import UltimateMEGEncoder, ContextualReRanker
from train.meg_utils import (
    SubjectRegistry, MEGDataModule,
    MetricLogger, batch_retrieval_metrics,
    build_run_dir, save_records, map_amp_to_precision,
)

logger = logging.getLogger("train_fused")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ------------------------ 只加载“局部路径”做 warm-start ------------------------
def _choose_mapped_state_dict(
    raw_state: dict,
    target_state: dict,
    strip_prefixes: Tuple[str, ...] = ("model.", "module.", "encoder.", "local.", "model.local."),
    allow_prefixes: Optional[Tuple[str, ...]] = None,
) -> Tuple[str, dict, float]:
    if "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    target_keys = list(target_state.keys())
    cands = []
    for pfx in ("",) + strip_prefixes:
        mapped = {}
        for k, v in raw_state.items():
            nk = k[len(pfx):] if (pfx and k.startswith(pfx)) else (k if not pfx else None)
            if nk is None:
                continue
            if (allow_prefixes is None) or any(nk.startswith(ap) for ap in allow_prefixes):
                mapped[nk] = v
        matched = 0
        for k, v in mapped.items():
            if k in target_state and hasattr(v, "shape") and hasattr(target_state[k], "shape"):
                if tuple(v.shape) == tuple(target_state[k].shape):
                    matched += 1
        frac = matched / max(1, len(target_keys))
        cands.append((pfx, mapped, frac))
    cands.sort(key=lambda x: x[2], reverse=True)
    return cands[0]

def _smart_load_local_only(module: torch.nn.Module, ckpt_path: str) -> float:
    sd = torch.load(ckpt_path, map_location="cpu")
    tgt = module.state_dict()
    pfx, mapped, frac = _choose_mapped_state_dict(
        sd, tgt,
        allow_prefixes=("spatial.", "pre_S", "to_d", "subjS", "subjD", "backbone.", "tail.", "proj.", "out_pool")
    )
    missing, unexpected = module.load_state_dict(mapped, strict=False)
    ok = len(tgt) - len(missing)
    logger.info(f"[warm local] prefix='{pfx or '<none>'}' matched={ok}/{len(tgt)} ({frac:.1%}); unexpected={len(unexpected)}")
    return float(frac)


# ------------------------ Lightning 模块 ------------------------
class LitModule(pl.LightningModule):
    def __init__(self,
                 enc_cfg: Dict,
                 local_ckpt: str = "",
                 freeze_local: bool = True,
                 freeze_memory: bool = False,
                 freeze_reranker: bool = False,
                 # 优化
                 lr: float = 3e-4, weight_decay: float = 0.01,
                 warmup_ratio: float = 0.1, max_epochs: int = 20,
                 optimizer_name: str = "adamw",
                 logit_scale_lr_mult: float = 0.1,
                 # bias / reranker 选项
                 bias_mode: str = "full",                 # full | diag
                 bias_rowwise_center: bool = False,
                 bias_start_epoch: int = 2,               # 该 epoch 起才叠加 bias
                 bias_cap: float = 0.5,                   # |bias| 上限；<=0 关闭裁剪
                 rerank_bank_reduce: str = "rms",         # mean | rms
                 rerank_dropout: float = 0.0,
                 mem_train_start_epoch: int = 2,          # 该 epoch 起允许训练记忆分支
                 metric_logger: Optional[MetricLogger] = None,
                 metrics_every_n_steps: int = 200):
        super().__init__()
        self.save_hyperparameters(ignore=["metric_logger"])

        # 编码器 + 上下文重排器
        self.enc = UltimateMEGEncoder(**enc_cfg)
        self.reranker = ContextualReRanker(d_model=enc_cfg["d_model"], out_dim=enc_cfg["out_channels"],
                                           dropout=rerank_dropout)

        # 温度（只在 logits 上）
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07), dtype=torch.float32))

        # warm-start：local-only
        if local_ckpt and Path(local_ckpt).exists():
            _smart_load_local_only(self.enc, local_ckpt)
        elif local_ckpt:
            logger.warning(f"[warm local] not found: {local_ckpt}")

        # 冻结策略（默认只训 reranker + 温度）
        if freeze_local:
            for n in ["spatial", "pre_S", "to_d", "subjS", "subjD", "backbone", "tail", "proj", "out_pool"]:
                if hasattr(self.enc, n):
                    for p in getattr(self.enc, n).parameters(): p.requires_grad_(False)
        if freeze_memory:
            for n in ["token_pool", "mem_enc", "mem_norm", "rpe", "mem_pos_mlp"]:
                if hasattr(self.enc, n):
                    for p in getattr(self.enc, n).parameters(): p.requires_grad_(False)
        if freeze_reranker:
            for p in self.reranker.parameters(): p.requires_grad_(False)

        self.metric_logger = metric_logger
        self.metrics_every_n_steps = int(metrics_every_n_steps)

        # 运行期常量
        self.eps = 1e-8

    # ------------ 候选端向量（不依赖 MSM；只用你提供的 audio_feat） ------------
    @torch.no_grad()
    def _audio_vec(self, batch: dict, device: torch.device) -> torch.Tensor:
        if "audio_feat" not in batch:
            raise KeyError("batch 缺少 'audio_feat'。请提供 [B, D] 或 [B, D, T]。")
        a = batch["audio_feat"].to(device, non_blocking=True)
        if a.ndim == 2:
            a = a.unsqueeze(-1)  # -> [B, D, 1]
        elif a.ndim != 3:
            raise ValueError(f"audio_feat 维度不支持：{tuple(a.shape)}")
        return a  # [B, D, T]

    def _reduce_audio_for_bias(self, v_a_bdt: torch.Tensor) -> torch.Tensor:
        """根据配置把音频特征降到 [B, D] 作为 rerank bank。"""
        method = self.hparams.rerank_bank_reduce.lower()
        if method == "rms":
            x2 = v_a_bdt.pow(2).mean(dim=2)
            out = torch.sqrt(x2 + self.eps)
        elif method == "mean":
            out = v_a_bdt.mean(dim=2)
        else:
            raise ValueError(f"Unsupported rerank_bank_reduce: {method}")
        return F.normalize(out, p=2, dim=1)

    # ------------ 构造 logits（base + bias） ------------
    def _forward_logits(self, batch: dict) -> torch.Tensor:
        device = self.device

        # 1) 基线 query（局部）
        y_local = self.enc(
            batch["meg_win"].to(device, non_blocking=True),
            batch["sensor_locs"].to(device, non_blocking=True),
            batch["subject_idx"].to(device, non_blocking=True),
        )  # [B, out_dim, Tq or 1]
        v_q = y_local  # [B, D, Tq?]

        # 2) 候选端：保持时域，不做 mean 池化（与 eval/paper 对齐）
        v_a = self._audio_vec(batch, device=device)      # [B, D, Ta]
        v_a_flat = v_a.flatten(1)                        # [B, D*Ta]
        v_a_n = F.normalize(v_a_flat, p=2, dim=1)        # [B, D*Ta]

        v_q_flat = v_q.flatten(1)                        # [B, D*Tq]
        scale = self.logit_scale.exp().clamp(min=1/0.5, max=1/0.03)
        base = (v_q_flat @ v_a_n.t()) * scale            # [B, B]

        # 3) 上下文偏置（按 batch 候选做分数加权）
        use_ctx = ("meg_sent" in batch) and ("meg_sent_relpos" in batch)
        if use_ctx:
            # bank（供 bias 头）使用更稳的 RMS 聚合/或 mean
            va_for_bias = self._reduce_audio_for_bias(v_a)           # [B, D]
            audio_bank = va_for_bias.unsqueeze(0).expand(va_for_bias.size(0), -1, -1)  # [B,B,D]

            anchor_idx = batch["meg_sent_relpos"].abs().argmin(dim=1).to(device)  # [B]
            train_mem_flag = (self.current_epoch >= int(self.hparams.mem_train_start_epoch))
            bias = self.reranker(
                encoder=self.enc,
                meg_sent=batch["meg_sent"], sensor_locs=batch["sensor_locs"], subj_idx=batch["subject_idx"],
                anchor_idx=anchor_idx, audio_bank=audio_bank,
                meg_sent_relpos=batch["meg_sent_relpos"], meg_sent_keys=batch.get("meg_sent_keys"),
                train_mem=bool(train_mem_flag),
            )  # [B, B]

            # 行内中心化（可选）
            if bool(self.hparams.bias_rowwise_center):
                bias = bias - bias.mean(dim=1, keepdim=True)

            # 只在指定 epoch 之后叠加偏置
            if int(self.current_epoch) >= int(self.hparams.bias_start_epoch):
                # 对角模式（可选）
                if str(self.hparams.bias_mode).lower() == "diag":
                    B = bias.size(0)
                    bias = bias * torch.eye(B, device=bias.device, dtype=bias.dtype)
                # 裁剪上限（可选）
                if float(self.hparams.bias_cap) > 0:
                    cap = float(self.hparams.bias_cap)
                    bias = bias.clamp(min=-cap, max=cap)
                logits = base + bias
            else:
                logits = base
        else:
            logits = base

        return logits

    # ------------ 训练/验证步骤 ------------
    def _step(self, batch, split: str):
        logits = self._forward_logits(batch)
        tgt = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, tgt)

        metrics = batch_retrieval_metrics(logits.detach(), ks=(1,5,10))
        bs = logits.size(0)

        self.log(f"{split}/loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        for k, v in metrics.items():
            self.log(f"{split}/{k}", v, prog_bar=(split == "val" and k in {"top1","top5","top10"}), on_epoch=True, batch_size=bs)

        if (self.metric_logger is not None) and (split == "train") and (int(self.global_step) % self.metrics_every_n_steps == 0):
            extra = {**metrics, "temp": float(1.0 / self.logit_scale.exp().item())}
            cur_lr = None
            try: cur_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            except Exception: pass
            self.metric_logger.write_step("train_step", int(self.global_step), int(self.current_epoch),
                                          float(loss.detach().item()), cur_lr, extra)
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._step(batch, "val")
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, _ = self._step(batch, "test")
        return {"test_loss": loss}

    # ------------ 优化器与调度 ------------
    def configure_optimizers(self):
        params = []
        # 温度：降低 LR 影响
        ls_lr = float(self.hparams.lr) * float(self.hparams.logit_scale_lr_mult)
        params.append({"params": [self.logit_scale], "lr": ls_lr, "weight_decay": 0.0})
        # reranker 主体
        if not self.hparams.freeze_reranker:
            alpha_params, other_params = [], []
            for n, p in self.reranker.named_parameters():
                (alpha_params if n.endswith("alpha") else other_params).append(p)
            if other_params:
                params.append({"params": other_params, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay})
            if alpha_params:
                params.append({"params": alpha_params, "lr": self.hparams.lr * 2.0, "weight_decay": 0.0})
        # enc 可训练部分（若你解冻）
        trainable_enc = [p for p in self.enc.parameters() if p.requires_grad]
        if trainable_enc:
            params.append({"params": trainable_enc, "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay})

        opt = torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=(0.9,0.999), eps=1e-8) \
              if self.hparams.optimizer_name.lower() == "adamw" else torch.optim.Adam(params, lr=self.hparams.lr)

        def lr_lambda(step: int):
            max_steps = max(1, self.trainer.estimated_stepping_batches)
            warm = int(self.hparams.warmup_ratio * max_steps)
            if step < warm: return float(step + 1) / max(1, warm)
            prog = float(step - warm) / max(1, max_steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}


# ------------------------ Epoch 级别 JSON 记录回调 ------------------------
class EpochJSONLogger(pl.Callback):
    """把每个 epoch 的聚合指标写到 <run_dir>/metrics_epoch.jsonl（jsonlines）。"""
    def __init__(self, out_path: Path, monitor: str = "val/top10", mode: str = "max"):
        super().__init__()
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.best = -math.inf if self.mode == "max" else math.inf

    @staticmethod
    def _to_float_map(metric_dict):
        out = {}
        for k, v in metric_dict.items():
            if isinstance(v, torch.Tensor):
                try:
                    out[k] = float(v.detach().cpu().item())
                except Exception:
                    continue
            elif isinstance(v, (float, int)):
                out[k] = float(v)
        return out

    def _append(self, payload: dict):
        payload["time"] = datetime.now(timezone.utc).isoformat()
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def on_train_epoch_end(self, trainer, pl_module):
        logs = self._to_float_map(trainer.callback_metrics)
        payload = {
            "epoch": int(trainer.current_epoch),
            "split": "train",
            **{k: v for k, v in logs.items() if k.startswith("train/")}
        }
        if payload.keys() - {"epoch", "split"}:
            self._append(payload)

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = self._to_float_map(trainer.callback_metrics)
        cur = logs.get(self.monitor)
        improved = None
        if cur is not None:
            if self.mode == "max":
                improved = cur > self.best
                if improved: self.best = cur
            else:
                improved = cur < self.best
                if improved: self.best = cur

        payload = {
            "epoch": int(trainer.current_epoch),
            "split": "val",
            **{k: v for k, v in logs.items() if k.startswith("val/")},
            "monitor": self.monitor,
            "current_monitor": cur,
            "best_so_far": self.best if math.isfinite(self.best) else None,
            "improved": bool(improved) if improved is not None else None,
        }
        # 附带当前 best ckpt 路径
        try:
            best_ckpt = None
            for cb in trainer.callbacks:
                if isinstance(cb, ModelCheckpoint) and getattr(cb, "best_model_path", ""):
                    best_ckpt = cb.best_model_path
                    break
            if best_ckpt: payload["best_checkpoint_path"] = best_ckpt
        except Exception:
            pass
        self._append(payload)


# ------------------------ DataModule 装配 ------------------------
def _add_dm_args(p: argparse.ArgumentParser):
    p.add_argument("--batch_size", type=int, default=192)  # sampler 会接管，留占位
    p.add_argument("--accumulate_grad_batches", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--persistent_workers", action="store_true")

    # window 模式参数（DataModule 使用）
    p.add_argument("--ctx_max_windows", type=int, default=16)
    p.add_argument("--ctx_stride", type=int, default=1)
    p.add_argument("--exclude_self_from_ctx", action="store_true")
    p.add_argument("--ctx_exclude_radius", type=int, default=2)

    p.add_argument("--sentences_per_batch", type=int, default=22)
    p.add_argument("--windows_per_sentence", type=int, default=5)


def parse_args():
    p = argparse.ArgumentParser()

    # 数据
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--val_manifest", required=True, type=str)
    p.add_argument("--test_manifest", required=True, type=str)

    # 被试映射
    p.add_argument("--subject_mapping_path", type=str, default="")
    p.add_argument("--subject_namespace", type=str, default="")

    # 训练/并行/精度
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--devices", type=int, default=None)
    p.add_argument("--amp", type=str, default="bf16", choices=["off","bf16","fp16","16-mixed","32"])
    p.add_argument("--tf32", type=str, default="on", choices=["auto","on","off"])
    p.add_argument("--compile", type=str, default="none", choices=["none","default","reduce-overhead","max-autotune"])

    # 优化 & 日程
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam","adamw"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--metrics_every_n_steps", type=int, default=200)
    p.add_argument("--early_stop_metric", type=str, default="mrr", choices=["top10", "mrr", "mean_rank"])

    # 模式固定为 window（需要 meg_sent）
    p.add_argument("--mode", type=str, default="msm_window", choices=["msm_window"])

    # 模型（传进 UltimateMEGEncoder）
    p.add_argument("--in_channels", type=int, default=270)
    p.add_argument("--spatial_channels", type=int, default=270)
    p.add_argument("--fourier_k", type=int, default=32)
    p.add_argument("--d_model", type=int, default=320)
    p.add_argument("--out_channels", type=int, default=1024)
    p.add_argument("--backbone_depth", type=int, default=5)
    p.add_argument("--backbone_type", type=str, default="cnn", choices=["cnn","conformer"])
    p.add_argument("--subject_layer_pos", type=str, choices=["early","late","none"], default="early")
    p.add_argument("--mem_enc_layers", type=int, default=3)
    p.add_argument("--mem_enc_heads", type=int, default=8)
    p.add_argument("--mem_dropout_p", type=float, default=0.08)
    p.add_argument("--context_memory_len", type=int, default=64)

    # 其它编码器细节
    p.add_argument("--rpe_max_rel", type=int, default=32)
    p.add_argument("--window_token_agg", type=str, default="mean", choices=["mean","asp"])
    p.add_argument("--asp_hidden", type=int, default=64)
    p.add_argument("--ctx_token_mbatch", type=int, default=48)
    p.add_argument("--detach_context", action="store_true")
    p.add_argument("--out_timesteps", type=int, default=1)

    # 权重加载
    p.add_argument("--local_ckpt", type=str, default="", help="none-baseline ckpt（只加载局部路径）")

    # 冻结选项
    p.add_argument("--freeze_local", action="store_true")
    p.add_argument("--freeze_memory", action="store_true")
    p.add_argument("--freeze_reranker", action="store_true")

    # rerank / bias / lr 控制
    p.add_argument("--bias_mode", type=str, default="full", choices=["full", "diag"])
    p.add_argument("--bias_rowwise_center", action="store_true")
    p.add_argument("--bias_start_epoch", type=int, default=2)
    p.add_argument("--bias_cap", type=float, default=0.5)
    p.add_argument("--rerank_bank_reduce", type=str, default="rms", choices=["mean","rms"])
    p.add_argument("--rerank_dropout", type=float, default=0.0)
    p.add_argument("--mem_train_start_epoch", type=int, default=2)
    p.add_argument("--logit_scale_lr_mult", type=float, default=0.1)

    # 其它
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default_root_dir", type=str, default="runs")
    p.add_argument("--experiment_name", type=str, default="MSM_CtxReRank_L64_MEAN")

    _add_dm_args(p)
    return p.parse_args()


def _build_dm(args, registry: SubjectRegistry) -> MEGDataModule:
    common = dict(
        train_manifest=str(Path(args.train_manifest)),
        val_manifest=str(Path(args.val_manifest)),
        test_manifest=str(Path(args.test_manifest)),
        registry=registry,
        ns_train=args.subject_namespace, ns_val=args.subject_namespace, ns_test=args.subject_namespace,
        batch_size=args.batch_size, num_workers=args.num_workers, normalize=False,
        pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    return MEGDataModule(
        **common,
        context_mode="window",
        ctx_max_windows=args.ctx_max_windows,
        ctx_stride=args.ctx_stride,
        group_by_sentence=True,
        sentences_per_batch=args.sentences_per_batch,
        windows_per_sentence=args.windows_per_sentence,
        exclude_self_from_ctx=args.exclude_self_from_ctx,
        ctx_exclude_radius=args.ctx_exclude_radius,
    )


def main():
    args = parse_args()

    # Precision & CUDA
    if args.tf32 == "on" or (args.tf32 == "auto" and args.amp in {"bf16","fp16","16-mixed","32"}):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    elif args.tf32 == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try: torch.set_float32_matmul_precision("highest")
        except Exception: pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    pl.seed_everything(args.seed, workers=True)
    precision = map_amp_to_precision(args.amp)
    devices = args.devices if args.devices is not None else args.gpus
    devices = max(1, int(devices))

    # run dir & logger
    exp_tag = f"{args.experiment_name}_{args.mode}"
    run_dir = build_run_dir(exp_tag, args.default_root_dir)
    metric_logger = MetricLogger(run_dir)
    (run_dir / "records").mkdir(parents=True, exist_ok=True)

    # Subject registry
    train_p, val_p, test_p = Path(args.train_manifest), Path(args.val_manifest), Path(args.test_manifest)
    map_path = Path(args.subject_mapping_path) if args.subject_mapping_path else (run_dir / "records" / "subject_mapping.json")

    if args.subject_mapping_path and map_path.exists():
        registry = SubjectRegistry.load(map_path)
        logger.info(f"[SubjectRegistry] Loaded: {map_path}  (#subjects={registry.num_subjects})")
    else:
        registry = SubjectRegistry.build_from_manifests([(train_p, args.subject_namespace),
                                                         (val_p,   args.subject_namespace),
                                                         (test_p,  args.subject_namespace)])
        registry.save(map_path)
        logger.info(f"[SubjectRegistry] Built & saved: {map_path}  (#subjects={registry.num_subjects})")

    # DataModule
    dm = _build_dm(args, registry)

    # Model cfg（传给 UltimateMEGEncoder）
    model_cfg: Dict = dict(
        in_channels=args.in_channels,
        n_subjects=registry.num_subjects,
        spatial_channels=args.spatial_channels,
        fourier_k=args.fourier_k,
        d_model=args.d_model,
        out_channels=args.out_channels,
        backbone_depth=args.backbone_depth,
        backbone_type=args.backbone_type,
        subject_layer_pos=args.subject_layer_pos,
        context_mode="window",
        context_memory_len=args.context_memory_len,
        mem_enc_layers=args.mem_enc_layers,
        mem_enc_heads=args.mem_enc_heads,
        mem_dropout_p=args.mem_dropout_p,
        freeze_ctx_local=True,
        detach_context=bool(args.detach_context),
        out_timesteps=args.out_timesteps,
        rpe_max_rel=args.rpe_max_rel,
        use_near_suppress=False,
        window_token_agg=args.window_token_agg,
        asp_hidden=args.asp_hidden,
        ctx_token_mbatch=args.ctx_token_mbatch,
    )

    # 记录 init 配置
    init_cfg = {
        "args": vars(args),
        "enc_cfg": model_cfg,
        "devices": devices,
        "precision": precision,
        "mode": args.mode,
        "subject_map_path": str(map_path.as_posix()),
        "similarity": "CLIP-style (q raw · L2(a, time-preserving)) + score-weighted context bias (batch-wise)",
        "local_ckpt": args.local_ckpt,
    }
    try:
        with open(run_dir / "records" / "config_init.json", "w", encoding="utf-8") as f:
            json.dump(init_cfg, f, indent=2, ensure_ascii=False)
        logger.info(f"[INIT] config written → { (run_dir / 'records' / 'config_init.json').as_posix() }")
    except Exception as e:
        logger.warning(f"[INIT] write config_init.json failed: {e}")

    lit = LitModule(
        enc_cfg=model_cfg,
        local_ckpt=args.local_ckpt,
        freeze_local=args.freeze_local,
        freeze_memory=args.freeze_memory,
        freeze_reranker=args.freeze_reranker,
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, max_epochs=args.max_epochs,
        optimizer_name=args.optimizer,
        logit_scale_lr_mult=args.logit_scale_lr_mult,
        # bias / rerank
        bias_mode=args.bias_mode,
        bias_rowwise_center=bool(args.bias_rowwise_center),
        bias_start_epoch=int(args.bias_start_epoch),
        bias_cap=float(args.bias_cap),
        rerank_bank_reduce=args.rerank_bank_reduce,
        rerank_dropout=float(args.rerank_dropout),
        mem_train_start_epoch=int(args.mem_train_start_epoch),
        metric_logger=metric_logger,
        metrics_every_n_steps=args.metrics_every_n_steps,
    )

    # 监控/早停
    monitor_map = {"top10": ("val/top10", "max"),
                   "mrr": ("val/mrr", "max"),
                   "mean_rank": ("val/mean_rank", "min")}
    mon, mode = monitor_map[args.early_stop_metric]

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename=f"fused-{{epoch:03d}}-{{{mon.replace('/','_')}:.4f}}",
        monitor=mon, mode=mode,
        save_top_k=3, save_last=True, auto_insert_metric_name=False,
    )
    early_cb = EarlyStopping(monitor=mon, mode=mode, patience=args.early_stop_patience, verbose=True)

    # === epoch 级 JSON 写入回调 ===
    epoch_json_path = run_dir / "metrics_epoch.jsonl"
    epoch_json_cb = EpochJSONLogger(out_path=epoch_json_path, monitor=mon, mode=mode)
    print(f"[LOG] epoch metrics → {epoch_json_path.as_posix()}", flush=True)

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=devices,
        max_epochs=args.max_epochs,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ckpt_cb, early_cb, epoch_json_cb],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=args.metrics_every_n_steps,
    )

    logger.info(f"[Run] dir={run_dir} | monitor={mon}({mode}) | freeze_local={args.freeze_local} | freeze_memory={args.freeze_memory}")
    trainer.fit(lit, datamodule=dm)

    best_ckpt_path = ckpt_cb.best_model_path if hasattr(ckpt_cb, "best_model_path") else None
    save_records(
        run_dir,
        cfg={
            "args": vars(args),
            "enc_cfg": model_cfg,
            "mode": args.mode,
            "devices": devices,
            "precision": precision,
            "similarity": "CLIP-style + score-weighted context bias (time-preserving audio)",
            "local_ckpt": args.local_ckpt,
        },
        best_ckpt_path=best_ckpt_path,
        subject_map_path=str((run_dir / "records" / "subject_mapping.json").as_posix()),
        registry=registry
    )

    try:
        metric_logger.export_tables_and_plots()
    except Exception as e:
        logger.warning(f"export plots failed: {e}")


if __name__ == "__main__":
    main()
