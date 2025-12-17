#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import math, argparse, logging, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.meg_encoder_audio_T import UltimateMEGEncoderTPP
from train.meg_utils import (
    SubjectRegistry, MEGDataModule,
    build_run_dir, MetricLogger, save_records, map_amp_to_precision
)

logger = logging.getLogger("stageG_sentence_tpp_pairwise")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ----------------- metric helpers -----------------
def _utc_now():
    return datetime.now(timezone.utc).isoformat()

def topk_inbatch_acc(logits: torch.Tensor, k: int = 1) -> float:
    B = logits.size(0)
    tgt = torch.arange(B, device=logits.device)
    k_eff = max(1, min(k, B))
    topk = logits.topk(k_eff, dim=1).indices
    hit = (topk == tgt.view(-1, 1)).any(dim=1).float().mean()
    return float(hit.item())

@torch.no_grad()
def compute_inbatch_metrics(logits: torch.Tensor) -> Dict[str, float]:
    B = logits.size(0)
    gt = logits.diag()
    ranks = (logits > gt.view(-1, 1)).sum(dim=1).add_(1).float()
    def _topk(k: int) -> float:
        k_eff = max(1, min(k, B))
        return float((ranks <= k_eff).float().mean().item())
    mrr = float((1.0 / ranks).mean().item())
    mean_rank = float(ranks.mean().item())
    median_rank = float(ranks.median().item())
    return {
        "acc@1": _topk(1),
        "acc@5": _topk(5),
        "acc@10": _topk(10),
        "mrr": mrr,
        "mean_rank": mean_rank,
        "median_rank": median_rank,
    }

def log_step_json(split: str, epoch: int, step: int, metrics: Dict[str, float]) -> None:
    payload = {"time": _utc_now(), "epoch": int(epoch), "step": int(step), "split": split}
    for k, v in metrics.items():
        payload[f"{split}/{k}"] = float(v)
    logger.info(json.dumps(payload, ensure_ascii=False))

# ----------------- scorer with softmax-pooling -----------------
class ColBERTPairwise(nn.Module):
    """
    s_ij = sum_l  Agg_m <q_l, k_m>,  Agg=max | softmax-pool(alpha)
    - norm_both: L2 normalize q,k
    - length_norm: none | Lq | sqrt
    - temp_mode: learnable | fixed | sched | none
      - learnable: logit_scale is Parameter, trainable
      - fixed:     logit_scale is buffer, constant
      - sched:     logit_scale is buffer, 每步 with copy_() 更新
    """
    def __init__(self, tau: float = 0.07, temp_mode: str = "learnable",
                 norm_both: bool = True, length_norm: str = "sqrt",
                 chunk_O: int = 0, agg: str = "softmax", agg_alpha: float = 12.0):
        super().__init__()
        self.norm_both = bool(norm_both)
        self.temp_mode = str(temp_mode)
        self.length_norm = str(length_norm)
        self.chunk_O = int(chunk_O)
        self.agg = str(agg)              # "max" | "softmax"
        self.agg_alpha = float(agg_alpha)

        if self.temp_mode == "learnable":
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / max(1e-6, tau))))
        elif self.temp_mode in ("fixed", "sched"):
            self.register_buffer("logit_scale", torch.log(torch.tensor(1.0 / max(1e-6, tau))), persistent=True)
        elif self.temp_mode == "none":
            self.logit_scale = None
        else:
            raise ValueError(f"temp_mode={temp_mode}")

    def _aggregate(self, S: torch.Tensor,
                   q_mask: Optional[torch.Tensor], k_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # S: [B,O,Lq,Lk]
        if (q_mask is not None) or (k_mask is not None):
            B, O, Lq, Lk = S.shape
            q_mask_b = torch.zeros(B, Lq, dtype=torch.bool, device=S.device) if q_mask is None else q_mask
            k_mask_b = torch.zeros(O, Lk, dtype=torch.bool, device=S.device) if k_mask is None else k_mask
            m = q_mask_b.view(B,1,Lq,1) | k_mask_b.view(1,O,1,Lk)
            S = S.masked_fill(m, float("-inf"))

        if self.agg == "max":
            s = S.max(dim=-1).values.sum(dim=-1)  # [B,O]
        elif self.agg == "softmax":
            a = self.agg_alpha
            s = torch.logsumexp(a * S, dim=-1) / max(1e-6, a)  # [B,O,Lq]
            s = s.sum(dim=-1)                                   # [B,O]
        else:
            raise ValueError(f"agg={self.agg}")
        return torch.where(torch.isfinite(s), s, torch.zeros_like(s))

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                q_mask: Optional[torch.Tensor] = None,
                k_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q:[B,Lq,D], k:[O,Lk,D] (训练时 O=B；离线全库时 O=pool_size)
        if self.norm_both:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        B, Lq, D = q.shape
        O, Lk, _ = k.shape

        if self.chunk_O <= 0 or self.chunk_O >= O:
            S = torch.einsum("bid,ojd->boij", q, k)  # [B,O,Lq,Lk]
            s = self._aggregate(S, q_mask, k_mask)   # [B,O]
        else:
            out = []
            step = int(self.chunk_O)
            for o0 in range(0, O, step):
                o1 = min(o0 + step, O)
                kb = k[o0:o1]
                S = torch.einsum("bid,ojd->boij", q, kb)  # [B,o,Lq,Lk]
                km = None if k_mask is None else k_mask[o0:o1]
                out.append(self._aggregate(S, q_mask, km))
            s = torch.cat(out, dim=1)  # [B,O]

        # 长度归一
        if self.length_norm == "Lq":
            s = s / (float(Lq) + 1e-6)
        elif self.length_norm == "sqrt":
            s = s / (float(Lq * Lk) ** 0.5 + 1e-6)

        # 温度缩放
        if self.logit_scale is not None:
            s = s * self.logit_scale.exp().clamp(max=100.0)
        return s

# ----------------- Lightning module -----------------
class LitStageGPairwise(pl.LightningModule):
    """
    句级学生：确定性 TPP；目标：Top-M 硬负采样的 CE（in-batch），但评测仍看全库（离线）。
    训练/验证：记录 acc@1/5/10、MRR、mean/median rank（批内 proxy）。
    """
    def __init__(self, enc_cfg: Dict, lr=1.5e-4, weight_decay=3e-2, warmup_ratio=0.15,
                 tau=0.07, tau_start=0.20, tau_end=0.07,
                 temp_mode="learnable", length_norm="sqrt",
                 agg="softmax", agg_alpha=12.0,
                 ce_topM: int = 128, margin: float = 0.05,
                 score_chunk_o: int = 0, metric_logger: Optional[MetricLogger] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["metric_logger"])
        self.metric_logger = metric_logger
        self.enc = UltimateMEGEncoderTPP(**enc_cfg)
        self.scorer = ColBERTPairwise(
            tau=tau, temp_mode=temp_mode, length_norm=length_norm,
            chunk_O=score_chunk_o, norm_both=True, agg=agg, agg_alpha=agg_alpha
        )
        self._shape_logged = False
        self._epoch_train_loss: Optional[float] = None
        self._best_val_acc10: Optional[float] = None

    # ---- temperature schedule (sched mode) ----
    def _tau_at(self, step: int, max_steps: int) -> float:
        t = min(step / max(1, max_steps), 1.0)
        # cosine from tau_start -> tau_end
        return float(self.hparams.tau_end + 0.5 * (self.hparams.tau_start - self.hparams.tau_end) * (1 + math.cos(math.pi * t)))

    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.temp_mode == "sched":
            try:
                max_steps = int(self.trainer.estimated_stepping_batches)
            except Exception:
                max_steps = 1000
            tau = self._tau_at(self.global_step, max_steps)
            with torch.no_grad():
                self.scorer.logit_scale.copy_(torch.log(torch.tensor(1.0 / max(1e-6, tau), device=self.device)))
            self.log("train/tau_sched", tau, on_step=True, on_epoch=False, prog_bar=False, batch_size=batch["meg_sent_full"].size(0))

    # ---- optimizer / scheduler ----
    def configure_optimizers(self):
        params = [{"params": [p for p in self.enc.parameters() if p.requires_grad],
                   "lr": self.hparams.lr, "weight_decay": self.hparams.weight_decay}]
        # learnable 温度才训练
        if self.hparams.temp_mode == "learnable" and isinstance(getattr(self.scorer, "logit_scale", None), nn.Parameter):
            params.append({"params": [self.scorer.logit_scale], "lr": self.hparams.lr, "weight_decay": 0.0})
        opt = torch.optim.AdamW(params)

        def lr_lambda(step):
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

    # ---- encode ----
    def _encode_meg_tokens(self, batch):
        return self.enc.encode_sentence_tokens(
            meg_sent_full=batch["meg_sent_full"].to(self.device, non_blocking=True),
            meg_sent_full_mask=batch.get("meg_sent_full_mask", None).to(self.device, non_blocking=True)
                if ("meg_sent_full_mask" in batch) else None,
            sensor_locs=batch["sensor_locs"].to(self.device, non_blocking=True),
            subj_idx=batch["subject_idx"].to(self.device, non_blocking=True),
            normalize=False,
        )

    def _teacher_tokens(self, batch):
        if ("audio_tpp" in batch) and ("audio_tpp_mask" in batch):
            return batch["audio_tpp"], batch["audio_tpp_mask"]
        raise RuntimeError("需要 batch['audio_tpp'] 与 batch['audio_tpp_mask']（sentence_fast_io=True）。")

    # ---- CE with Top-M hard negatives + margin ----
    def _ce_with_topM(self, s_full: torch.Tensor, labels: torch.Tensor, topM: int, margin: float) -> torch.Tensor:
        # s_full: [B,O], labels: [B]
        if topM <= 0 or topM >= s_full.size(1) - 1:
            # margin on diagonal only
            if margin > 0:
                B = s_full.size(0)
                arange = torch.arange(B, device=s_full.device)
                s_full = s_full.clone()
                s_full[arange, labels] = s_full[arange, labels] - margin
            return F.cross_entropy(s_full, labels)

        B, O = s_full.shape
        arange = torch.arange(B, device=s_full.device)
        pos = s_full[arange, labels].unsqueeze(1)  # [B,1]
        if margin > 0:
            pos = pos - margin

        neg = s_full.clone()
        neg[arange, labels] = -float("inf")
        topk_vals, _ = neg.topk(topM, dim=1)       # [B,M]
        sub = torch.cat([pos, topk_vals], dim=1)   # [B,1+M]
        target = torch.zeros(B, dtype=torch.long, device=s_full.device)  # pos at index 0
        return F.cross_entropy(sub, target)

    # ---- one step ----
    def _step(self, batch, split: str, batch_idx: int):
        g = self._encode_meg_tokens(batch)        # [B,Lq,D]
        h, hmask = self._teacher_tokens(batch)    # [B,Lk,D], [B,Lk] (True=pad)
        if not self._shape_logged:
            logger.info(f"[shapes] student={tuple(g.shape)} teacher={tuple(h.shape)}")
            self._shape_logged = True

        logits = self.scorer(g, h, q_mask=None, k_mask=hmask)  # [B,B]
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)

        loss = self._ce_with_topM(
            s_full=logits, labels=labels,
            topM=int(self.hparams.ce_topM),
            margin=float(self.hparams.margin)
        )

        # 记录 tau（learnable/sched/fixed）
        ls = getattr(self.scorer, "logit_scale", None)
        if ls is not None:
            # buffer 或 parameter 都能 .exp()
            tau_now = 1.0 / float(ls.detach().exp().clamp(min=1e-6, max=1e6).item())
            self.log(f"{split}/tau", tau_now, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)

        # 批内指标
        met = compute_inbatch_metrics(logits.detach())
        self.log(f"{split}/acc@1_inbatch", met["acc@1"], on_step=True, on_epoch=True, prog_bar=(split=="train"), batch_size=B)
        self.log(f"{split}/acc@5_inbatch", met["acc@5"], on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log(f"{split}/acc@10_inbatch", met["acc@10"], on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log(f"{split}/mrr_inbatch", met["mrr"], on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log(f"{split}/mean_rank_inbatch", met["mean_rank"], on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        self.log(f"{split}/median_rank_inbatch", met["median_rank"], on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        log_step_json(split, int(self.current_epoch), int(batch_idx), met)

        if split == "train":
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        else:
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=B)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return {"val_loss": self._step(batch, "val", batch_idx)}

    def test_step(self, batch, batch_idx):
        return {"test_loss": self._step(batch, "test", batch_idx)}

    def on_train_epoch_end(self):
        cm = self.trainer.callback_metrics if self.trainer is not None else {}
        tr = cm.get("train/loss", None)
        self._epoch_train_loss = float(tr.item()) if torch.is_tensor(tr) else (float(tr) if tr is not None else None)

    def on_validation_epoch_end(self):
        cm = self.trainer.callback_metrics if self.trainer is not None else {}
        acc10 = cm.get("val/acc@10_inbatch", None)
        if acc10 is not None:
            v = float(acc10.item()) if torch.is_tensor(acc10) else float(acc10)
            self._best_val_acc10 = v if (self._best_val_acc10 is None) else max(self._best_val_acc10, v)

        if self.metric_logger is None:
            return
        rec = {k: (float(v.item()) if torch.is_tensor(v) else float(v))
               for k, v in cm.items() if isinstance(k, str) and (k.startswith("val/"))}
        self.metric_logger.write_epoch(epoch=int(self.current_epoch),
                                       train_loss=self._epoch_train_loss,
                                       val_loss=rec.get("val/loss"),
                                       best_val_top10=self._best_val_acc10,
                                       train_metrics=None,
                                       val_metrics=rec)
        self._epoch_train_loss = None

# ----------------- CLI / DataModule -----------------
def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--test_manifest", required=True)
    p.add_argument("--subject_mapping_path", default="")
    p.add_argument("--subject_namespace", default="")

    # sys
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16", "16-mixed", "32"])
    p.add_argument("--max_epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=1)
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--sentences_per_batch", type=int, default=256)

    # encoder（CNN-only）
    p.add_argument("--in_channels", type=int, default=208)
    p.add_argument("--spatial_channels", type=int, default=270)
    p.add_argument("--fourier_k", type=int, default=32)
    p.add_argument("--d_model", type=int, default=320)
    p.add_argument("--backbone_type", type=str, default="cnn", choices=["cnn"])
    p.add_argument("--backbone_depth", type=int, default=5)
    p.add_argument("--subject_layer_pos", type=str, choices=["early", "late", "none"], default="early")
    p.add_argument("--audio_dim", type=int, default=2048)
    p.add_argument("--tpp_slots", type=int, default=15)

    # scorer / loss
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=3e-2)
    p.add_argument("--warmup_ratio", type=float, default=0.15)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--tau_start", type=float, default=0.20)
    p.add_argument("--tau_end", type=float, default=0.07)
    p.add_argument("--temp_mode", type=str, default="sched", choices=["learnable", "fixed", "sched", "none"])
    p.add_argument("--length_norm", type=str, default="sqrt", choices=["none", "Lq", "sqrt"])
    p.add_argument("--score_chunk_o", type=int, default=0)
    p.add_argument("--agg", type=str, default="softmax", choices=["max", "softmax"])
    p.add_argument("--agg_alpha", type=float, default=12.0)
    p.add_argument("--ce_topM", type=int, default=128)
    p.add_argument("--margin", type=float, default=0.05)

    # logging
    p.add_argument("--default_root_dir", type=str, default="runs")
    p.add_argument("--experiment_name", type=str, default="StageG_TPP_PairwiseFast")
    return p.parse_args()

def _build_dm(args, registry: SubjectRegistry) -> MEGDataModule:
    common = dict(
        train_manifest=str(Path(args.train_manifest)),
        val_manifest=str(Path(args.val_manifest)),
        test_manifest=str(Path(args.test_manifest)),
        registry=registry,
        ns_train=args.subject_namespace, ns_val=args.subject_namespace, ns_test=args.subject_namespace,
        batch_size=args.batch_size, num_workers=args.num_workers, normalize=False,
        pin_memory=True, prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers,
    )
    return MEGDataModule(**common,
                         context_mode="sentence",
                         sentence_fast_io=True,
                         group_by_sentence=True,
                         sentences_per_batch=args.sentences_per_batch,
                         windows_per_sentence=1,
                         key_mode="audio")

# ----------------- main -----------------
def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    pl.seed_everything(42, workers=True)
    precision = map_amp_to_precision(args.amp)

    run_dir = build_run_dir(args.experiment_name, args.default_root_dir)
    metric_logger = MetricLogger(run_dir)
    records_dir = run_dir / "records"; records_dir.mkdir(parents=True, exist_ok=True)

    # Subject registry
    train_p, val_p, test_p = Path(args.train_manifest), Path(args.val_manifest), Path(args.test_manifest)
    map_path = Path(args.subject_mapping_path) if args.subject_mapping_path else (records_dir / "subject_mapping.json")
    if args.subject_mapping_path and map_path.exists():
        registry = SubjectRegistry.load(map_path)
    else:
        registry = SubjectRegistry.build_from_manifests([(train_p, args.subject_namespace),
                                                         (val_p,   args.subject_namespace),
                                                         (test_p,  args.subject_namespace)])
        registry.save(map_path)

    enc_cfg: Dict = dict(
        in_channels=args.in_channels,
        n_subjects=registry.num_subjects,
        spatial_channels=args.spatial_channels,
        fourier_k=args.fourier_k,
        d_model=args.d_model,
        backbone_depth=args.backbone_depth,
        subject_layer_pos=args.subject_layer_pos,
        tpp_slots=args.tpp_slots,
        audio_dim=args.audio_dim,
        out_channels=1024,
        out_timesteps=None,
    )

    lit = LitStageGPairwise(enc_cfg=enc_cfg,
                            lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
                            tau=args.tau, tau_start=args.tau_start, tau_end=args.tau_end,
                            temp_mode=args.temp_mode, length_norm=args.length_norm,
                            agg=args.agg, agg_alpha=args.agg_alpha,
                            ce_topM=args.ce_topM, margin=args.margin,
                            score_chunk_o=args.score_chunk_o,
                            metric_logger=metric_logger)

    ckpt_cb = ModelCheckpoint(
        dirpath=str((run_dir / "checkpoints")),
        filename="stageG_sentence_pairwise-{epoch:03d}-{val_acc10:.4f}",
        monitor="val/acc@10_inbatch",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        accelerator="gpu" if torch.cuda.is_available() else "auto",
        devices=max(1, args.devices),
        max_epochs=args.max_epochs,
        precision=precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=False,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        callbacks=[ckpt_cb],
    )

    dm = _build_dm(args, registry)
    trainer.fit(lit, datamodule=dm)

    # --- 写 best_checkpoint.txt（优先 best，其次 last） ---
    best_path = getattr(ckpt_cb, "best_model_path", "") or ""
    last_ckpt = best_path
    if not last_ckpt:
        try:
            cb = getattr(trainer, "checkpoint_callback", None)
            if cb is not None and getattr(cb, "last_model_path", ""):
                last_ckpt = cb.last_model_path
        except Exception:
            pass
    if not last_ckpt:
        last_ckpt = str((run_dir / "last.ckpt").as_posix())
        torch.save({"state_dict": lit.state_dict()}, last_ckpt)

    best_txt = run_dir / "records" / "best_checkpoint.txt"
    best_txt.parent.mkdir(parents=True, exist_ok=True)
    best_txt.write_text(last_ckpt + "\n", encoding="utf-8")

    save_records(
        run_dir,
        cfg={"args": vars(args), "enc_cfg": enc_cfg, "mode": "sentence", "backend": "TPP",
             "devices": max(1, args.devices), "precision": precision},
        best_ckpt_path=last_ckpt,
        subject_map_path=str((records_dir / "subject_mapping.json").as_posix()),
        registry=registry
    )

    try: metric_logger.export_tables_and_plots()
    except Exception as e: logger.warning(f"export plots failed: {e}")

if __name__ == "__main__":
    main()
