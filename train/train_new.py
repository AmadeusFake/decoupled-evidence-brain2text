#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Global training (window-only, sentence-level embedding):
- Freeze local encoder by default (unless --enable_unfreeze is given)
- Multi-positive InfoNCE between MEG sentence vector g and audio sentence vector h
- Clean, readable epoch logs (one line per epoch, VAL first, then TRAIN)
- Write both TSV and JSONL epoch logs under runs/<exp>/records/

Usage example (frozen local):
python -u -m train.train_global \
  --train_manifest ... --val_manifest ... --test_manifest ... \
  --ctx_max_windows 18 --sentences_per_batch 40 --windows_per_sentence 8 \
  --batch_size 256 --accumulate_grad_batches 2 \
  --devices 1 --amp bf16 --tf32 on \
  --default_root_dir runs --experiment_name StageG_Embed_winK12_fullctx

(Only if you WANT to unfreeze later)
  --enable_unfreeze --local_warmup_epochs 1 --unfreeze_last_n_blocks 2 --local_lr_mult 0.1
"""
from __future__ import annotations
import math, argparse, json, logging, os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.meg_encoder import UltimateMEGEncoder, AttentiveStatsPool1D
from train.meg_utils import (
    TARGET_T, AUDIO_D,
    SubjectRegistry, MEGDataModule,
    MetricLogger, batch_retrieval_metrics,
    build_run_dir, save_records, map_amp_to_precision,
    build_pos_mask_same_audio_uid,
)

logger = logging.getLogger("train_global")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------- Audio sentence head ----------------
class AudioSentenceHead(nn.Module):
    """ a[B,1024,T] -> h[B,D] """
    def __init__(self, out_dim: int):
        super().__init__()
        self.pool = AttentiveStatsPool1D(d_model=AUDIO_D, hidden=AUDIO_D, dropout=0.1)
        self.proj = nn.Linear(AUDIO_D, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, a_bdt: torch.Tensor) -> torch.Tensor:
        v = self.pool(a_bdt)             # [B,1024]
        v = self.norm(self.proj(v))      # [B,D]
        v = torch.nan_to_num(v)
        return F.normalize(v, dim=-1)


# ---------------- Lightning Module ----------------
class GlobalLit(pl.LightningModule):
    def __init__(self,
                 enc_cfg: Dict,
                 lr: float = 3e-4, weight_decay: float = 0.01,
                 warmup_ratio: float = 0.10, max_epochs: int = 20,
                 optimizer_name: str = "adamw",
                 temp_min: float = 0.03, temp_max: float = 0.5,
                 metric_logger: Optional[MetricLogger] = None,
                 metrics_every_n_steps: int = 200,
                 # warm start + (optional) unfreeze
                 encoder_ckpt_path: str = "",
                 enable_unfreeze: bool = False,
                 local_warmup_epochs: int = 1,
                 unfreeze_last_n_blocks: int = 2,
                 local_lr_mult: float = 0.1,
                 # logging files
                 epoch_tsv_path: Optional[Path] = None,
                 epoch_jsonl_path: Optional[Path] = None,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["metric_logger", "epoch_tsv_path", "epoch_jsonl_path"])

        # ---- Encoder (window mode) ----
        enc_cfg = dict(enc_cfg)
        enc_cfg["out_timesteps"] = int(TARGET_T)
        enc_cfg["detach_context"] = False
        self.enc = UltimateMEGEncoder(**enc_cfg)

        # Warm start (optional)
        if encoder_ckpt_path and Path(encoder_ckpt_path).is_file():
            self._load_encoder_ckpt_(encoder_ckpt_path)

        # Freeze local by default (your ask)
        self.enc.set_local_trainable(False)

        # ---- Audio head ----
        self.audio_head = AudioSentenceHead(out_dim=enc_cfg["d_model"])

        # Temperature (CLIP-style)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07), dtype=torch.float32))
        self._temp_min = float(temp_min)
        self._temp_max = float(temp_max)

        # logging helpers
        self.metric_logger = metric_logger
        self.metrics_every_n_steps = int(metrics_every_n_steps)
        self.epoch_tsv_path = Path(epoch_tsv_path) if epoch_tsv_path else None
        self.epoch_jsonl_path = Path(epoch_jsonl_path) if epoch_jsonl_path else None
        self._epoch_header_written = False

        # unfreeze plan
        self.enable_unfreeze = bool(enable_unfreeze)
        self.local_warmup_epochs = int(local_warmup_epochs)
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)
        self.local_lr_mult = float(local_lr_mult)
        self._has_unfroze = False

    # ---------- warm start ----------
    def _load_encoder_ckpt_(self, path: str):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:  # Lightning ckpt
            sd = {k.replace("enc.", "", 1): v for k, v in sd["state_dict"].items() if k.startswith("enc.")}
        missing, unexpected = self.enc.load_state_dict(sd, strict=False)
        logger.info(f"[WarmStart] loaded from {path}\n - missing: {len(missing)}\n - unexpected: {len(unexpected)}")

    # ---------- optimizer ----------
    def configure_optimizers(self):
        base_lr = float(self.hparams.lr)
        wd = float(self.hparams.weight_decay)

        # trainable modules (local frozen by default)
        params = []
        params += list(self.enc.mem_enc.parameters())
        params += list(self.enc.mem_norm.parameters())
        params += list(self.enc.token_pool.parameters())
        params += list(self.enc.readout.parameters())
        params += list(self.audio_head.parameters())
        params += [self.logit_scale]

        if self.hparams.optimizer_name.lower() == "adamw":
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=wd, betas=(0.9, 0.999), eps=1e-8)
        else:
            opt = torch.optim.Adam(params, lr=base_lr, weight_decay=wd)

        def lr_lambda(step: int):
            try:
                max_steps = int(self.trainer.estimated_stepping_batches)
            except Exception:
                max_steps = 1000
            warm = int(self.hparams.warmup_ratio * max_steps)
            if step < warm: return float(step + 1) / max(1, warm)
            prog = float(step - warm) / max(1, max_steps - warm)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    # ---------- contrastive loss (multi-positive) ----------
    def _info_nce_multi_pos(self, q: torch.Tensor, k: torch.Tensor, pos_mask: torch.Tensor):
        # L2 on both sides (already normalized in forward, but make safe)
        q = F.normalize(torch.nan_to_num(q), dim=1)
        k = F.normalize(torch.nan_to_num(k), dim=1)

        scale = self.logit_scale.exp().clamp(min=1.0/self._temp_max, max=1.0/self._temp_min)
        logits = (q @ k.t()) * scale  # [B,B]
        B = logits.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=logits.device)
        pos_any = pos_mask.any(dim=1, keepdim=True)
        pos_full = torch.where(pos_any, pos_mask, eye)
        logit_pos = torch.where(pos_full, logits, torch.full_like(logits, -1e9))
        log_pos = torch.logsumexp(logit_pos, dim=1)
        log_den = torch.logsumexp(logits, dim=1)
        loss = (-(log_pos - log_den)).mean()
        return loss, logits

    # ---------- shared train/val step ----------
    def _step(self, batch: dict, split: str):
        device = self.device

        # === MEG sentence vector g (window mode; fixed-K; key cache used only when local frozen) ===
        _mem, g = self.enc._global_from_window_seq(
            meg_sent=batch["meg_sent"].to(device, non_blocking=True),          # [B,N,C,T]
            sensor_locs=batch["sensor_locs"].to(device, non_blocking=True),
            subj_idx=batch["subject_idx"].to(device, non_blocking=True),
            anchor_idx=None, select_topk=None,
            meg_sent_keys=batch.get("meg_sent_keys", None)
        )  # g [B,D] already L2 & safe

        # === audio sentence vector h ===
        a = batch["audio_feat"].to(device, non_blocking=True)   # [B,1024,T]
        h = self.audio_head(a)                                  # [B,D], L2

        # === multi-positive InfoNCE ===
        pos_mask = build_pos_mask_same_audio_uid(batch["audio_uid"].to(device))  # [B,B]
        loss, logits = self._info_nce_multi_pos(g, h, pos_mask)

        # === retrieval metrics ===
        metrics = batch_retrieval_metrics(logits.detach(), ks=(1, 5, 10))
        bs = int(logits.size(0))

        # log to Lightning
        self.log(f"{split}/loss", loss, prog_bar=(split == "val"), on_epoch=True, batch_size=bs)
        for k, v in metrics.items():
            self.log(f"{split}/{k}", v, prog_bar=(split == "val" and k in {"top1", "top5", "top10", "mrr"}),
                     on_epoch=True, batch_size=bs)

        # step-level custom logger
        if (self.metric_logger is not None) and (split == "train") and (int(self.global_step) % int(self.hparams.metrics_every_n_steps) == 0):
            extra = {**metrics, "temp": float(1.0 / self.logit_scale.exp().item())}
            cur_lr = None
            try: cur_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
            except Exception: pass
            self.metric_logger.write_step("train_step", int(self.global_step), int(self.current_epoch),
                                          float(loss.detach().item()), cur_lr, extra)

        return loss, metrics

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self._step(batch, "val")[0]}

    def test_step(self, batch, batch_idx):
        return {"test_loss": self._step(batch, "test")[0]}

    # ---------- epoch-end hooks: pretty logging + optional unfreeze ----------
    def on_validation_epoch_end(self):
        # Optional: unfreeze after warmup epochs (only if enabled)
        if (not self._has_unfroze) and self.enable_unfreeze and (int(self.current_epoch) + 1 >= int(self.local_warmup_epochs)):
            self.enc.set_local_trainable(True, last_n_backbone_blocks=int(self.unfreeze_last_n_blocks))
            logger.info("[Unfreeze] local path enabled (last %d backbone blocks + to_d).", int(self.unfreeze_last_n_blocks))
            self._has_unfroze = True

        # Pretty epoch logging (VAL first, then TRAIN)
        cm = self.trainer.callback_metrics if self.trainer is not None else {}

        def _get(k: str, default=None):
            v = cm.get(k, default)
            try:
                return float(v.detach().item()) if isinstance(v, torch.Tensor) else (float(v) if v is not None else None)
            except Exception:
                return default

        ep = int(self.current_epoch)
        # VAL metrics
        v_mrr = _get("val/mrr", None)
        v_top1 = _get("val/top1", None)
        v_top5 = _get("val/top5", None)
        v_top10 = _get("val/top10", None)
        v_loss = _get("val/loss", None)
        # TRAIN metrics (epoch)
        t_mrr = _get("train/mrr", _get("train/mrr_epoch", None))
        t_top1 = _get("train/top1", _get("train/top1_epoch", None))
        t_top5 = _get("train/top5", _get("train/top5_epoch", None))
        t_top10 = _get("train/top10", _get("train/top10_epoch", None))
        t_loss = _get("train/loss", _get("train/loss_epoch", None))
        # extras
        temp = None
        try: temp = float(1.0 / self.logit_scale.exp().item())
        except Exception: pass
        lr = None
        try: lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])
        except Exception: pass

        # ---- console one-liner ----
        def _fmt(x):
            return "nan" if x is None else f"{x:.4f}"

        line = (
            f"[EP {ep:03d}] "
            f"VAL  mrr={_fmt(v_mrr)}  top1={_fmt(v_top1)}  top5={_fmt(v_top5)}  top10={_fmt(v_top10)}  loss={_fmt(v_loss)}  |  "
            f"TRAIN mrr={_fmt(t_mrr)}  top1={_fmt(t_top1)}  top5={_fmt(t_top5)}  top10={_fmt(t_top10)}  loss={_fmt(t_loss)}"
        )
        if (temp is not None) or (lr is not None):
            line += "  |  "
            if temp is not None: line += f"temp={_fmt(temp)}  "
            if lr is not None:   line += f"lr={_fmt(lr)}"
        logger.info(line)

        # ---- write TSV / JSONL ----
        if (self.epoch_tsv_path is not None):
            self.epoch_tsv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._epoch_header_written or not self.epoch_tsv_path.exists():
                with open(self.epoch_tsv_path, "w", encoding="utf-8") as f:
                    f.write("epoch\tval_mrr\tval_top1\tval_top5\tval_top10\tval_loss\ttrain_mrr\ttrain_top1\ttrain_top5\ttrain_top10\ttrain_loss\ttemp\tlr\n")
                self._epoch_header_written = True
            with open(self.epoch_tsv_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{ep}\t{_fmt(v_mrr)}\t{_fmt(v_top1)}\t{_fmt(v_top5)}\t{_fmt(v_top10)}\t{_fmt(v_loss)}\t"
                    f"{_fmt(t_mrr)}\t{_fmt(t_top1)}\t{_fmt(t_top5)}\t{_fmt(t_top10)}\t{_fmt(t_loss)}\t{_fmt(temp)}\t{_fmt(lr)}\n"
                )

        if (self.epoch_jsonl_path is not None):
            self.epoch_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "phase": "epoch",
                "epoch": ep,
                "val": {"mrr": v_mrr, "top1": v_top1, "top5": v_top5, "top10": v_top10, "loss": v_loss},
                "train": {"mrr": t_mrr, "top1": t_top1, "top5": t_top5, "top10": t_top10, "loss": t_loss},
                "temp": temp, "lr": lr,
            }
            try:
                with open(self.epoch_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                logger.warning(f"write epoch jsonl failed: {e}")


# ---------------- CLI / DataModule ----------------
def _add_dm_args(p: argparse.ArgumentParser):
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--accumulate_grad_batches", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--persistent_workers", action="store_true")

    # window 模式（固定 K，避免句长泄露；默认不剔除中心窗、半径=0）
    p.add_argument("--ctx_max_windows", type=int, default=12)     # 建议 8~18
    p.add_argument("--ctx_stride", type=int, default=1)
    p.add_argument("--exclude_self_from_ctx", action="store_true")
    p.add_argument("--ctx_exclude_radius", type=int, default=0)
    p.add_argument("--sentences_per_batch", type=int, default=32)
    p.add_argument("--windows_per_sentence", type=int, default=8)


def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_manifest", required=True, type=str)
    p.add_argument("--val_manifest", required=True, type=str)
    p.add_argument("--test_manifest", required=True, type=str)

    # subject mapping
    p.add_argument("--subject_mapping_path", type=str, default="")
    p.add_argument("--subject_namespace", type=str, default="")

    # accel
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--amp", type=str, default="bf16", choices=["off","bf16","fp16","16-mixed","32"])
    p.add_argument("--tf32", type=str, default="on", choices=["auto","on","off"])

    # opt / schedule
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adam","adamw"])
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--metrics_every_n_steps", type=int, default=200)

    # encoder cfg
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
    p.add_argument("--ctx_token_mbatch", type=int, default=32)  # for window token encoding chunk size

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default_root_dir", type=str, default="runs")
    p.add_argument("--experiment_name", type=str, default="StageG_Embed_winK")

    # warm start & (optional) unfreeze
    p.add_argument("--encoder_ckpt_path", type=str, default="")
    p.add_argument("--enable_unfreeze", action="store_true")       # <- 默认不解冻，除非显式打开
    p.add_argument("--local_warmup_epochs", type=int, default=1)
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=2)
    p.add_argument("--local_lr_mult", type=float, default=0.1)

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
        exclude_self_from_ctx=bool(args.exclude_self_from_ctx),
        ctx_exclude_radius=args.ctx_exclude_radius,
    )


def main():
    args = parse_args()

    # Precision & CUDA setup
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
    devices = max(1, int(args.devices))

    # run dir & loggers
    run_dir = build_run_dir(args.experiment_name, args.default_root_dir)
    metric_logger = MetricLogger(run_dir)
    records_dir = run_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    epoch_tsv = records_dir / "metrics_epoch.tsv"
    epoch_jsonl = records_dir / "metrics_epoch.jsonl"

    # Subject registry
    train_p, val_p, test_p = Path(args.train_manifest), Path(args.val_manifest), Path(args.test_manifest)
    map_path = Path(args.subject_mapping_path) if args.subject_mapping_path else (records_dir / "subject_mapping.json")

    if args.subject_mapping_path and map_path.exists():
        registry = SubjectRegistry.load(map_path)
        logger.info(f"[SubjectRegistry] Loaded: {map_path}  (#subjects={registry.num_subjects})")
    else:
        registry = SubjectRegistry.build_from_manifests([(train_p, args.subject_namespace),
                                                         (val_p,   args.subject_namespace),
                                                         (test_p,  args.subject_namespace)])
        registry.save(map_path)
        logger.info(f"[SubjectRegistry] Built & saved: {map_path}  (#subjects={registry.num_subjects})")

    # Encoder cfg (window-only)
    enc_cfg: Dict = dict(
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
        freeze_ctx_local=True,              # keep local frozen unless enable_unfreeze later
        detach_context=False,
        use_rpe=False, use_mem_pos=False, use_near_suppress=False,
        ctx_token_mbatch=args.ctx_token_mbatch,
        use_act_ckpt=True,                  # safe; only effective if local is trainable
    )

    # save initial config
    try:
        with open(records_dir / "config_init.json", "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "enc_cfg": enc_cfg, "devices": devices, "precision": precision}, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"[INIT] write config_init.json failed: {e}")

    lit = GlobalLit(
        enc_cfg=enc_cfg,
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, max_epochs=args.max_epochs,
        optimizer_name=args.optimizer,
        metric_logger=metric_logger, metrics_every_n_steps=args.metrics_every_n_steps,
        encoder_ckpt_path=args.encoder_ckpt_path,
        enable_unfreeze=bool(args.enable_unfreeze),
        local_warmup_epochs=args.local_warmup_epochs,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        local_lr_mult=args.local_lr_mult,
        epoch_tsv_path=epoch_tsv,
        epoch_jsonl_path=epoch_jsonl,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="global-{epoch:03d}-{val_mrr:.4f}",
        monitor="val/mrr", mode="max",
        save_top_k=3, save_last=True, auto_insert_metric_name=False,
    )
    early_cb = EarlyStopping(monitor="val/mrr", mode="max", patience=args.early_stop_patience, verbose=True)

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

    logger.info(f"[Run] dir={run_dir} | window-only | K={args.ctx_max_windows}")
    dm = _build_dm(args, registry)
    trainer.fit(lit, datamodule=dm)

    best_ckpt_path = ckpt_cb.best_model_path if hasattr(ckpt_cb, "best_model_path") else None
    save_records(
        run_dir,
        cfg={"args": vars(args), "enc_cfg": enc_cfg, "mode": "window", "devices": devices, "precision": precision},
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
