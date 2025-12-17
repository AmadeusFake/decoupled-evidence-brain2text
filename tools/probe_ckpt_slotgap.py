#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, torch
import torch.nn.functional as F
from pathlib import Path

from train.train_global_tpp_late import LitStageG   # 直接复用
from train.meg_utils import SubjectRegistry, MEGDataModule

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--subject_namespace", default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sentences_per_batch", type=int, default=64)
    args = ap.parse_args()

    # 一个 batch 即可
    reg = SubjectRegistry.build_from_manifests([
        (Path(args.train_manifest), args.subject_namespace),
        (Path(args.val_manifest),   args.subject_namespace),
        (Path(args.test_manifest),  args.subject_namespace),
    ])
    dm = MEGDataModule(
        train_manifest=args.train_manifest, val_manifest=args.val_manifest, test_manifest=args.test_manifest,
        registry=reg, ns_train=args.subject_namespace, ns_val=args.subject_namespace, ns_test=args.subject_namespace,
        batch_size=args.batch_size, num_workers=0, normalize=False,
        pin_memory=False, prefetch_factor=2, persistent_workers=False,
        context_mode="sentence", sentence_fast_io=True,
        group_by_sentence=True, sentences_per_batch=args.sentences_per_batch,
        windows_per_sentence=1, key_mode="audio"
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # 直接用 ckpt 里保存的超参重建（self.save_hyperparameters 已保存 enc_cfg）
        lit: LitStageG = LitStageG.load_from_checkpoint(
            args.ckpt, strict=False, map_location=device
        )
    except TypeError:
        # 某些旧 ckpt/PL 版本需要手动取 enc_cfg
        ckpt = torch.load(args.ckpt, map_location="cpu")
        enc_cfg = ckpt["hyper_parameters"]["enc_cfg"]
        lit: LitStageG = LitStageG.load_from_checkpoint(
            args.ckpt, enc_cfg=enc_cfg, strict=False, map_location=device
        )

    lit.eval().to(device)

    # mask 也要搬到同一设备
    mask = batch.get("meg_sent_full_mask", None)
    if mask is not None:
        mask = mask.to(device, non_blocking=True)

    g = lit.enc.encode_sentence_tokens(
        meg_sent_full=batch["meg_sent_full"].to(device, non_blocking=True),
        meg_sent_full_mask=mask,
        sensor_locs=batch["sensor_locs"].to(device, non_blocking=True),
        subj_idx=batch["subject_idx"].to(device, non_blocking=True),
        normalize=False,
    )

    lit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit.to(device)

    g = lit.enc.encode_sentence_tokens(
        meg_sent_full=batch["meg_sent_full"].to(device),
        meg_sent_full_mask=batch.get("meg_sent_full_mask"),
        sensor_locs=batch["sensor_locs"].to(device),
        subj_idx=batch["subject_idx"].to(device),
        normalize=False
    )  # [B,L,D]
    h = batch["audio_tpp"].to(device)  # [B,L,D]

    g_n = F.normalize(g, dim=-1); h_n = F.normalize(h, dim=-1)
    L = g.size(1); D = g.size(2)

    # L 个槽位各自的 B×B 相似度
    sims = torch.matmul(g_n.transpose(0,1), h_n.transpose(0,1).transpose(1,2))  # [L,B,B]
    pos = torch.diagonal(sims, 0, 1, 2)   # [L,B]
    neg = (sims.sum(dim=2) - torch.diagonal(sims, 0, 1, 2)) / (sims.size(2)-1 + 1e-9)

    print(f"[slot-level] pos={pos.mean().item():.4f}  neg={neg.mean().item():.4f}  gap={(pos.mean()-neg.mean()).item():+.4f}")
    names = ["L1","L2","L4","L8"]; sizes = [1,2,4,8]
    s=0
    for nm, k in zip(names, sizes):
        rng = slice(s, s+k); s+=k
        p = pos[rng].mean().item(); q = neg[rng].mean().item()
        print(f"  - {nm}: pos={p:.4f}  neg={q:.4f}  gap={p-q:+.4f}")

    # ColBERT 批内 logits（B×B）

    # g_n, h_n: [B, L, D]
    # 生成四维相似度：对每个 (i,j) 的样本对，计算 [L, L] 的槽位相似
    S4 = torch.einsum('bld,cmd->bclm', g_n, h_n)   # [B, B, Lq, Lk]
    S_colb = S4.max(dim=3).values.sum(dim=2)       # [B, B] 先对 key 槽位 max，再沿 query 槽位 sum

    diag = torch.diagonal(S_colb).mean().item()
    off_mean = ((S_colb.sum() - torch.diagonal(S_colb).sum())
                / (S_colb.numel() - S_colb.size(0))).item()
    print(f"[colbert] diag_mean={diag:.4f}  off_mean={off_mean:.4f}  gap={diag-off_mean:+.4f}")
if __name__ == "__main__":
    main()
