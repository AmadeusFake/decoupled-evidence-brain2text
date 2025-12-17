#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sentence Fast I/O 体检脚本（无 memmap）：
- 打印 Teacher（audio_tpp）分层可分性（L=1/2/4/8 聚合）的 diag/off/gap
- 学生（MEG encoder）与老师的槽位级相似度（pos/neg/gap）
- ColBERT 式批内配对分数矩阵 [B,B]（修复版）
- 可选：加载训练好的 ckpt 使用已训练学生；不传则随机学生
- 可选：线性最小二乘（Procrustes/Linear Probe）评估“线性可对齐性”
"""

import argparse, torch
import torch.nn.functional as F
from pathlib import Path

from train.meg_utils import SubjectRegistry, MEGDataModule
from models.meg_encoder_audio_T import UltimateMEGEncoderTPP

# 可选加载训练好的学生编码器
try:
    from train.train_global_tpp_late import LitStageG
    _HAS_LIT = True
except Exception:
    _HAS_LIT = False

def get_batch(args):
    reg = SubjectRegistry.build_from_manifests([
        (Path(args.train_manifest), args.subject_namespace),
        (Path(args.val_manifest),   args.subject_namespace),
        (Path(args.test_manifest),  args.subject_namespace),
    ])
    dm = MEGDataModule(
        train_manifest=args.train_manifest, val_manifest=args.val_manifest, test_manifest=args.test_manifest,
        registry=reg, ns_train=args.subject_namespace, ns_val=args.subject_namespace, ns_test=args.subject_namespace,
        batch_size=args.batch_size, num_workers=0, normalize=False, pin_memory=False,
        prefetch_factor=2, persistent_workers=False,
        context_mode="sentence", sentence_fast_io=True, group_by_sentence=True,
        sentences_per_batch=args.sentences_per_batch, windows_per_sentence=1, key_mode="audio"
    )
    dm.setup(stage="fit")
    return next(iter(dm.train_dataloader())), reg

def build_student(args, batch, device, ckpt=None):
    if (ckpt is not None) and _HAS_LIT:
        try:
            lit: LitStageG = LitStageG.load_from_checkpoint(ckpt, map_location=device)
            enc = lit.enc.eval().to(device)
            return enc
        except Exception as e:
            print(f"[WARN] 加载 ckpt 失败，回退到随机学生：{e}")

    # 随机学生
    enc = UltimateMEGEncoderTPP(
        in_channels=args.in_channels,
        n_subjects=int(batch["subject_idx"].max().item())+1,
        spatial_channels=args.spatial_channels,
        d_model=args.d_model,
        backbone_depth=args.backbone_depth,
        backbone_type=args.backbone_type,
        subject_layer_pos="early",
        audio_dim=args.audio_dim,
        tpp_levels=[1,2,4,8],
        tpp_slots=15
    ).eval().to(device)
    return enc

@torch.no_grad()
def run_probe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch, registry = get_batch(args)
    for k in ("meg_sent_full","meg_sent_full_mask","sensor_locs","subject_idx","audio_tpp","audio_tpp_mask"):
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)

    # teacher tokens（规范化）
    h  = F.normalize(batch["audio_tpp"], dim=-1)     # [B,L,D]
    ht = h                                           # teacher for diagnostics

    # 学生编码器（随机或 ckpt）
    enc = build_student(args, batch, device, ckpt=args.ckpt)

    # 学生 tokens
    g = enc.encode_sentence_tokens(
        meg_sent_full=batch["meg_sent_full"],
        meg_sent_full_mask=batch.get("meg_sent_full_mask"),
        sensor_locs=batch["sensor_locs"],
        subj_idx=batch["subject_idx"],
        normalize=True
    )  # [B,L,D]

    B,L,D = g.shape
    print(f"=== [SHAPES] ===  student={tuple(g.shape)}  teacher={tuple(h.shape)}  (n_subjects={registry.num_subjects})")

    # ---------- Teacher 分层自检（验证“粗尺度是否有信息”） ----------
    levels = [1,2,4,8]
    s = 0
    for Lv in levels:
        sl = slice(s, s+Lv); s += Lv
        ht_lv = ht[:, sl].mean(dim=1)     # [B, D]
        S_lv  = ht_lv @ ht_lv.T           # [B, B]
        dmean = torch.diagonal(S_lv, 0).mean().item()
        off   = ((S_lv.sum() - torch.diagonal(S_lv,0).sum()) / (S_lv.numel() - B)).item()
        print(f"[teacher L{Lv}] diag={dmean:.4f}  off={off:.4f}  gap={dmean-off:.4f}")

    # ---------- 槽位级相似度（原始、不做学习） ----------
    gA_n = F.normalize(g, dim=-1); hA_n = h  # h 已经 norm
    simsA = torch.matmul(gA_n.transpose(0,1), hA_n.transpose(0,1).transpose(1,2))  # [L,B,B]
    posA = torch.diagonal(simsA, 0, 1, 2)     # [L,B]
    negA = (simsA.sum(dim=2) - torch.diagonal(simsA,0,1,2)) / (simsA.size(2)-1 + 1e-9)
    pos_m = posA.mean().item(); neg_m = negA.mean().item()
    print(f"[slot-level/A] pos={pos_m:.4f}  neg={neg_m:.4f}  gap={pos_m-neg_m:+.4f}")
    s=0
    for nm,k in zip(("L1","L2","L4","L8"), (1,2,4,8)):
        sl = slice(s, s+k); s+=k
        print(f"  - {nm}: pos={posA[sl].mean().item():.4f}  neg={negA[sl].mean().item():.4f}  gap={(posA[sl].mean()-negA[sl].mean()).item():+.4f}")

    # ---------- ColBERT 批内配对分数 [B,B]（修复版） ----------
    # T[bq,bk,lq,lk] = <g[bq,lq], h[bk,lk]>
    T = torch.einsum("bld,cmd->bclm", gA_n, hA_n)  # [B, B, L, L]
    ScolA = T.amax(dim=3).sum(dim=2)              # [B, B]  ← 先 max_m 再对 l 累加
    diag = torch.diagonal(ScolA, 0).mean().item()
    offm = ((ScolA.sum() - torch.diagonal(ScolA, 0).sum()) / (ScolA.numel() - B)).item()
    print(f"[colbert/A] diag_mean={diag:.4f}  off_mean={offm:.4f}  gap={diag - offm:+.4f}")

    # ---------- 槽位矩阵：对角 vs 行内非对角最大（修复 .amax 用法） ----------
    # M0[lq, lk] = mean_b <g_hat[:,lq], h[:,lk]>。这里直接用 gA_n 与 hA_n 的相似度均值。
    M0 = torch.einsum("bld,bmd->lm", gA_n, hA_n) / float(B)  # [L, L]
    diag0   = torch.diag(M0).mean().item()
    # 把对角线抹掉（置 0），行内取最大值（非对角）；注意 amax 返回 Tensor，本身可 .mean()
    offmax0 = (M0 - torch.diag(torch.diag(M0))).amax(dim=1).mean().item()
    print(f"[slot-matrix/raw] diag_mean={diag0:.4f}  row_offdiag_max_mean={offmax0:.4f}  gap={diag0-offmax0:+.4f}")

    # ---------- 可选：线性最小二乘（Probe）看“线性可对齐性” ----------
    if args.linear_probe:
        X = gA_n.reshape(B*L, D)    # student
        Y = hA_n.reshape(B*L, D)    # teacher
        W, *_ = torch.linalg.lstsq(X, Y)  # [D, D]
        Y_hat = (X @ W).reshape(B, L, D)
        sim = (F.normalize(Y_hat, dim=-1) * hA_n).sum(dim=-1).mean().item()
        print(f"[linear-probe cos] mean={sim:.4f}")
        M = torch.einsum("bld,bmd->lm", F.normalize(Y_hat, dim=-1), hA_n) / float(B)
        diag_lp   = torch.diag(M).mean().item()
        offmax_lp = (M - torch.diag(torch.diag(M))).amax(dim=1).mean().item()
        print(f"[slot-matrix/after-lstsq A→A] diag={diag_lp:.4f}  row_offmax={offmax_lp:.4f}  gap={diag_lp-offmax_lp:+.4f}")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_manifest", required=True)
    ap.add_argument("--val_manifest", required=True)
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--subject_namespace", default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sentences_per_batch", type=int, default=64)
    ap.add_argument("--in_channels", type=int, default=208)
    ap.add_argument("--spatial_channels", type=int, default=270)
    ap.add_argument("--d_model", type=int, default=320)
    ap.add_argument("--backbone_type", choices=["cnn","conformer"], default="cnn")
    ap.add_argument("--backbone_depth", type=int, default=5)
    ap.add_argument("--audio_dim", type=int, default=2048)
    ap.add_argument("--ckpt", type=str, default="")          # 可选：加载训练好的学生
    ap.add_argument("--linear_probe", action="store_true")   # 可选：线性对齐评估
    return ap.parse_args()

def main():
    args = parse_args()
    try:
        run_probe(args)
    except Exception as e:
        print("=== FATAL ERROR ===")
        raise

if __name__ == "__main__":
    main()
