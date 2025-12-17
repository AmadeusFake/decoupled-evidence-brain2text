#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-G — Paper-aligned sample-wise contrast on TPP tokens
- Student (MEG→TPP): [B, L, 2048]，L=tpp_slots；不做 L2。
- Teacher (Audio TPP): [B, L, 2048]；只在 loss 内做整体 L2（按(C,T)）。
- “样本对齐”：把 L 当成时间长度 T，在损失里统一到 target_T（这里 = tpp_slots）。
- 不做 set-to-set；仅记录统一的一组检索指标（top1/top5/mrr/mean_rank）与 loss。
"""

from __future__ import annotations
import math, argparse, json, logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.meg_encoder_audio_T import UltimateMEGEncoderTPP
from train.meg_utils import (
    SubjectRegistry, MEGDataModule,
    MetricLogger, build_run_dir, save_records, map_amp_to_precision,
    batch_retrieval_metrics,     # 输入 [B,B] logits，输出 topk/mrr/mean_rank
)

logger = logging.getLogger("train_global_tpp_paper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

AUDIO_TPP_KEYS = ("audio_tpp","audio_sent_tpp","audio_sentence_tokens","audio_tpp_tokens","teacher_tpp","teacher_tokens")

def _get_first(batch: dict, keys: Tuple[str, ...]) -> Optional[torch.Tensor]:
    for k in keys:
        if k in batch:
            return batch[k]
    return None


# ============================== Paper-aligned Loss ============================== #
class PaperClipLoss(nn.Module):
    """
    论文口径对比（单向：MEG→Audio）
    - 仅 candidates（Audio）做 L2（按 (C,T)），Student 不做 L2；
    - 不做时间池化；对比前插值到 target_T（此处 = tpp_slots，等价于 no-op）；
    - 固定温度（tau）或可学习温度（可选）。
    """
    def __init__(self, target_T: int, use_temperature: bool = False, init_temp: float = 0.07,
                 pool: bool = False, center: bool = False, trim_min: int | None = None, trim_max: int | None = None):
        super().__init__()
        self.target_T = int(target_T)
        self.pool = bool(pool)
        self.center = bool(center)
        self.trim_min = trim_min
        self.trim_max = trim_max
        if use_temperature:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / max(1e-6, init_temp)), dtype=torch.float32))
        else:
            self.register_parameter("logit_scale", None)

    @staticmethod
    def _to_BCT(x: torch.Tensor) -> torch.Tensor:
        # 接受 [B,D,T] 或 [B,L,D]；统一到 [B,C=TOK_DIM,T=L]
        if x.dim() != 3:
            raise ValueError(f"Expect 3D, got {tuple(x.shape)}")
        B, A, C = x.shape
        return x if A >= C else x.transpose(1, 2).contiguous()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_BCT(x.to(torch.float32))                # [B,C,T]
        if x.size(-1) != self.target_T:
            x = F.interpolate(x, size=self.target_T, mode="linear", align_corners=False)
        if (self.trim_min is not None) or (self.trim_max is not None):
            t0 = 0 if self.trim_min is None else max(0, int(self.trim_min))
            t1 = x.size(-1) if self.trim_max is None else min(x.size(-1), int(self.trim_max))
            x = x[..., t0:t1]
        if self.pool:
            x = x.mean(dim=2, keepdim=True)
        if self.center:
            x = x - x.mean(dim=(1, 2), keepdim=True)
        return x

    def forward(self, meg_f: torch.Tensor, aud_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self._prep(meg_f)  # [B,C,T]
        a = self._prep(aud_f)  # [B,C,T]
        inv_norms = (a.norm(dim=(1, 2), p=2) + 1e-8).reciprocal()  # 仅 candidates 做 L2
        logits = torch.einsum("bct,oct,o->bo", m, a, inv_norms)
        if self.logit_scale is not None:
            logits = logits * self.logit_scale.exp().clamp(max=100.0)
        tgt = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, tgt)
        return loss, logits


# ============================== Lightning Module ============================== #
class GlobalTrainTPP(pl.LightningModule):
    def __init__(self,
                 enc_cfg: Dict,
                 lr: float = 1e-4, weight_decay: float = 0.03,
                 warmup_ratio: float = 0.10, max_epochs: int = 40,
                 optimizer_name: str = "adamw",
                 tau: float = 0.07,
                 learnable_temp: bool = False,
                 metric_logger: Optional[MetricLogger] = None,
                 metrics_every_n_steps: int = 300,
                 fe_lr_mult: float = 0.5,
                 freeze_backbone_epochs: int = 1,
                 target_T: int = 15):
        super().__init__()
        self.save_hyperparameters(ignore=["metric_logger"])
        self.metric_logger = metric_logger
        self.metrics_every_n_steps = int(metrics_every_n_steps)

        # Student encoder: [B,L,2048]，不做 L2
        self.enc = UltimateMEGEncoderTPP(**enc_cfg)

        # 论文口径损失：target_T = tpp_slots
        self.loss_fn = PaperClipLoss(
            target_T=int(target_T),
            use_temperature=bool(learnable_temp),
            init_temp=float(tau),
            pool=False, center=False,
        )

        self.teacher_dim = int(enc_cfg.get("audio_dim", 2048))
        self.target_slots = int(enc_cfg.get("tpp_slots", target_T))
        self._shape_logged = False
        self._best_val_top1 = float("-inf")
        self._frozen_until = int(max(0, freeze_backbone_epochs))

    # ------- optim ------- #
    def _split_decay_params(self, modules: List[nn.Module | None]):
        decay, no_decay = [], []
        for m in modules:
            if m is None: continue
            for n, p in m.named_parameters():
                if not p.requires_grad: continue
                if n.endswith(".bias") or "norm" in n.lower():
                    no_decay.append(p)
                else:
                    decay.append(p)
        return decay, no_decay

    def configure_optimizers(self):
        base_lr = float(self.hparams.lr); wd = float(self.hparams.weight_decay)
        fe_mods = [self.enc.spatial, self.enc.pre_S, self.enc.subjS, self.enc.subjD,
                   self.enc.to_d, self.enc.backbone, self.enc.tqasp, self.enc.tok_norm]
        decay, no_decay = self._split_decay_params(fe_mods)
        groups = []
        fe_lr = base_lr * float(self.hparams.fe_lr_mult)
        if decay:    groups.append({"params": decay,    "lr": fe_lr, "weight_decay": wd})
        if no_decay: groups.append({"params": no_decay, "lr": fe_lr, "weight_decay": 0.0})
        if getattr(self.loss_fn, "logit_scale", None) is not None:
            groups.append({"params": [self.loss_fn.logit_scale], "lr": base_lr, "weight_decay": 0.0})

        if self.hparams.optimizer_name.lower() == "adamw":
            opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=wd, betas=(0.9,0.999), eps=1e-8)
        else:
            opt = torch.optim.Adam(groups, lr=base_lr, weight_decay=wd)

        def lr_lambda(step: int):
            try:
                max_steps = int(self.trainer.estimated_stepping_batches)
            except Exception:
                max_steps = 1000
            warm = int(float(self.hparams.warmup_ratio) * max_steps)
            if step < warm: return float(step+1)/max(1,warm)
            prog = float(step-warm)/max(1,max_steps-warm)
            return 0.5*(1.0+math.cos(math.pi*prog))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    # ------- freeze/unfreeze ------- #
    def on_fit_start(self):
        logger.info(f"[Sanity] Expect teacher L={self.target_slots}, D={self.teacher_dim}")
        if self._frozen_until > 0:
            for p in self.enc.backbone.parameters(): p.requires_grad = False
            logger.info(f"[Freeze] backbone frozen for first {self._frozen_until} epochs")

    def on_train_epoch_start(self):
        if self.current_epoch == self._frozen_until and self._frozen_until > 0:
            for p in self.enc.backbone.parameters(): p.requires_grad = True
            logger.info("[Unfreeze] backbone unfrozen")

    # ------- encode & io ------- #
    def _encode_meg_tokens(self, batch: dict, device: torch.device) -> torch.Tensor:
        g = self.enc.encode_sentence_tokens(
            meg_sent_full=batch["meg_sent_full"].to(device, non_blocking=True),
            meg_sent_full_mask=batch.get("meg_sent_full_mask", None).to(device, non_blocking=True)
                if ("meg_sent_full_mask" in batch) else None,
            sensor_locs=batch["sensor_locs"].to(device, non_blocking=True),
            subj_idx=batch["subject_idx"].to(device, non_blocking=True),
            normalize=False,  # 论文口径：Student 不 L2
        )  # [B,L,2048]
        return g

    def _get_teacher_tokens(self, batch: dict, device: torch.device) -> torch.Tensor:
        h = _get_first(batch, AUDIO_TPP_KEYS)
        if h is None:
            raise RuntimeError("Need audio TPP tokens in batch (keys="
                               + ", ".join(AUDIO_TPP_KEYS) + ").")
        h = h.to(device, non_blocking=True)  # [B,L,2048]
        if h.dim() != 3 or h.size(-1) != self.teacher_dim:
            raise RuntimeError(f"Teacher tokens shape must be [B,L,{self.teacher_dim}], got {tuple(h.shape)}")
        return h  # 不在此处 L2，交由 loss 统一处理

    # ------- one step ------- #
    def _step(self, batch: dict, split: str):
        device = self.device
        g = self._encode_meg_tokens(batch, device)   # [B,L,2048]
        h = self._get_teacher_tokens(batch, device)  # [B,L,2048]

        if not self._shape_logged:
            logger.info(f"[Shapes] student={tuple(g.shape)} teacher={tuple(h.shape)}  (L=tpp_slots={self.target_slots})")
            self._shape_logged = True

        # [B,C,T] = [B,2048,L]
        g_ct = g.transpose(1, 2).contiguous()
        h_ct = h.transpose(1, 2).contiguous()

        loss, logits = self.loss_fn(g_ct, h_ct)  # [B,B]

        # metrics（统一）
        B = int(logits.size(0))
        m = batch_retrieval_metrics(logits.detach(), ks=(1,5))
        self.log(f"{split}/self_top1",      m["top1"],      on_epoch=True, prog_bar=(split=="val"), batch_size=B)
        self.log(f"{split}/self_top5",      m["top5"],      on_epoch=True, prog_bar=False,           batch_size=B)
        self.log(f"{split}/self_mrr",       m["mrr"],       on_epoch=True, prog_bar=False,           batch_size=B)
        self.log(f"{split}/self_mean_rank", m["mean_rank"], on_epoch=True, prog_bar=False,           batch_size=B)
        self.log(f"{split}/loss",           loss,           on_epoch=True, prog_bar=False,           batch_size=B)

        # step 级外部日志（关键指标）
        if (self.metric_logger is not None) and (split == "train") and \
           (int(self.global_step) % int(self.hparams.metrics_every_n_steps) == 0):
            cur_lr = None
            try: cur_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            except: pass
            temp_scale = None
            if getattr(self.loss_fn, "logit_scale", None) is not None:
                temp_scale = float(self.loss_fn.logit_scale.detach().exp().item())
            self.metric_logger.write_step(
                "train_step", int(self.global_step), int(self.current_epoch),
                float(loss.detach().item()), cur_lr,
                {"self_top1": float(m["top1"]), "self_top5": float(m["top5"]),
                 "self_mrr": float(m["mrr"]), "mean_rank": float(m["mean_rank"]),
                 **({"temp": temp_scale} if temp_scale is not None else {})}
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self._step(batch, "val")}

    def test_step(self, batch, batch_idx):
        return {"test_loss": self._step(batch, "test")}

    def on_validation_epoch_end(self):
        if self.metric_logger is None:
            return
        cm = self.trainer.callback_metrics

        def _to_float(x):
            try:
                if isinstance(x, torch.Tensor): return float(x.detach().cpu().item())
                return float(x)
            except Exception:
                return None

        def _collect(prefix: str) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for k, v in cm.items():
                if k.startswith(prefix + "/"):
                    f = _to_float(v)
                    if f is not None and math.isfinite(f):
                        out[k.split("/",1)[1]] = f
            return out

        train_metrics = _collect("train")
        val_metrics   = _collect("val")
        train_loss = train_metrics.get("loss")
        val_loss   = val_metrics.get("loss")

        cur_val_top1 = _to_float(cm.get("val/self_top1"))
        if cur_val_top1 is not None:
            self._best_val_top1 = max(self._best_val_top1, cur_val_top1)

        self.metric_logger.write_epoch(
            epoch=int(self.current_epoch),
            train_loss=train_loss, val_loss=val_loss,
            best_val_top10=self._best_val_top1,   # 字段名保留，含义为 best val/self_top1
            train_metrics=train_metrics, val_metrics=val_metrics
        )


# -------------------- Args & runner -------------------- #
def _add_dm_args(p: argparse.ArgumentParser):
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=1)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--sentences_per_batch", type=int, default=256)  # 等价 batch_size（按句）
    p.add_argument("--windows_per_sentence", type=int, default=1)   # 句级训练通常=1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--val_manifest", required=True, type=str)
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--subject_mapping_path", type=str, default="")
    p.add_argument("--subject_namespace", type=str, default="")

    # compute & run
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--amp", type=str, default="bf16", choices=["off","bf16","fp16","16-mixed","32"])
    p.add_argument("--tf32", type=str, default="on", choices=["auto","on","off"])
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam","adamw"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=3e-2)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--metrics_every_n_steps", type=int, default=300)

    # encoder config (TPP)
    p.add_argument("--in_channels", type=int, default=208)
    p.add_argument("--spatial_channels", type=int, default=270)
    p.add_argument("--fourier_k", type=int, default=32)
    p.add_argument("--d_model", type=int, default=320)
    p.add_argument("--backbone_type", type=str, default="cnn", choices=["cnn","conformer"])
    p.add_argument("--backbone_depth", type=int, default=5)
    p.add_argument("--subject_layer_pos", type=str, choices=["early","late","none"], default="early")
    p.add_argument("--tpp_slots", type=int, default=15)    # == target_T
    p.add_argument("--audio_dim", type=int, default=2048)

    # loss/temp
    p.add_argument("--tau", type=float, default=0.07)      # 固定温度
    p.add_argument("--learnable_temp", action="store_true")

    # lr mult & freeze
    p.add_argument("--fe_lr_mult", type=float, default=0.5)
    p.add_argument("--freeze_backbone_epochs", type=int, default=1)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default_root_dir", type=str, default="runs")
    p.add_argument("--experiment_name", type=str, default="StageG_TPP_PaperLoss_L15D2048")
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
        context_mode="sentence",
        sentence_fast_io=True,
        group_by_sentence=True,
        sentences_per_batch=args.sentences_per_batch,
        windows_per_sentence=args.windows_per_sentence,
        key_mode="audio",      # 只用音频 TPP tokens
    )

def main():
    args = parse_args()

    # TF32 / precision
    if args.tf32 == "on" or (args.tf32 == "auto" and args.amp in {"bf16","fp16","16-mixed","32"}):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except: pass
    elif args.tf32 == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try: torch.set_float32_matmul_precision("highest")
        except: pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    pl.seed_everything(args.seed, workers=True)
    precision = map_amp_to_precision(args.amp)
    devices = max(1, int(args.devices))

    run_dir = build_run_dir(args.experiment_name, args.default_root_dir)
    metric_logger = MetricLogger(run_dir)
    records_dir = run_dir / "records"; records_dir.mkdir(parents=True, exist_ok=True)

    # Subject registry
    train_p, val_p, test_p = Path(args.train_manifest), Path(args.val_manifest), Path(args.test_manifest)
    map_path = Path(args.subject_mapping_path) if args.subject_mapping_path else (records_dir / "subject_mapping.json")
    if args.subject_mapping_path and map_path.exists():
        registry = SubjectRegistry.load(map_path)
        logger.info(f"[SubjectRegistry] Loaded: {map_path} (#subjects={registry.num_subjects})")
    else:
        registry = SubjectRegistry.build_from_manifests([(train_p, args.subject_namespace),
                                                         (val_p,   args.subject_namespace),
                                                         (test_p,  args.subject_namespace)])
        registry.save(map_path)
        logger.info(f"[SubjectRegistry] Built & saved: {map_path} (#subjects={registry.num_subjects})")

    # Encoder cfg（不需要本地 out_timesteps，因此设为 None）
    enc_cfg: Dict = dict(
        in_channels=args.in_channels,
        n_subjects=registry.num_subjects,
        spatial_channels=args.spatial_channels,
        fourier_k=args.fourier_k,
        d_model=args.d_model,
        backbone_depth=args.backbone_depth,
        backbone_type=args.backbone_type,
        subject_layer_pos=args.subject_layer_pos,
        tpp_slots=args.tpp_slots,            # L
        audio_dim=args.audio_dim,            # 2048
        out_channels=1024,
        out_timesteps=None,
    )

    # Save config snapshot
    try:
        with open(records_dir / "config_init.json", "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "enc_cfg": enc_cfg,
                       "devices": devices, "precision": precision}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[INIT] write config_init.json failed: {e}")

    lit = GlobalTrainTPP(
        enc_cfg=enc_cfg,
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, max_epochs=args.max_epochs,
        optimizer_name=args.optimizer,
        metric_logger=metric_logger, metrics_every_n_steps=args.metrics_every_n_steps,
        fe_lr_mult=args.fe_lr_mult,
        tau=args.tau, learnable_temp=args.learnable_temp,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        target_T=args.tpp_slots,   # 关键：target_T == tpp_slots
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="stageG_paper-{epoch:03d}-{val_self_top1:.4f}",
        monitor="val/self_top1", mode="max",
        save_top_k=3, save_last=True, auto_insert_metric_name=False,
    )
    early_cb = EarlyStopping(monitor="val/self_top1", mode="max", patience=args.early_stop_patience, verbose=True)

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=devices,
        max_epochs=args.max_epochs,
        precision=precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ckpt_cb, early_cb],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        log_every_n_steps=args.metrics_every_n_steps,
    )

    logger.info(f"[Run] dir={run_dir} | L={args.tpp_slots}, D={args.audio_dim} | d_model={args.d_model}, "
                f"depth={args.backbone_depth} | tau={args.tau} | bs(sentences)={args.sentences_per_batch} "
                f"| learnable_temp={bool(args.learnable_temp)} | target_T={args.tpp_slots}")
    dm = _build_dm(args, registry)
    trainer.fit(lit, datamodule=dm)

    best_ckpt_path = ckpt_cb.best_model_path if hasattr(ckpt_cb, "best_model_path") else None
    save_records(
        run_dir,
        cfg={"args": vars(args), "enc_cfg": enc_cfg, "mode": "sentence", "backend": "TPP",
             "devices": devices, "precision": precision},
        best_ckpt_path=best_ckpt_path,
        subject_map_path=str((records_dir / "subject_mapping.json").as_posix()),
        registry=registry
    )

    try:
        metric_logger.export_tables_and_plots()
    except Exception as e:
        logger.warning(f"export plots failed: {e}")

if __name__ == "__main__":
    main()
