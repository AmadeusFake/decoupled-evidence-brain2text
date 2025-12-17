#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_sentence_level_stats_fast.py

功能：
1. 句子画像：生成 SVG/PDF 分布图和统计 CSV。
2. 极速指标计算：
   - 利用 GPU 矩阵化计算 (Batch Processing) 瞬间完成 568x568 次 BERTScore 推理。
   - 计算 WER/CER 并寻找“中心句”。

加速原理：
- 避免 Python for-loop 调用 GPU。
- 构建 (N*N) 的全量 Pair 列表，一次性喂给显卡，打满 CUDA 核心。

依赖：
- torch (必须且需要检测到 CUDA)
- bert_score, jiwer, matplotlib, pandas
"""
import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

# ------------------------- 依赖导入 -------------------------
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from jiwer import wer as jiwer_wer, cer as jiwer_cer, Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False

try:
    from bert_score import BERTScorer
    _HAS_BERTSCORE = True
except ImportError:
    _HAS_BERTSCORE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------- 辅助函数 -------------------------

_TEXT_KEYS = ["transcript", "text", "sentence_text", "transcript_text", "global_segment_text"]

def extract_text(r: dict) -> str:
    for k in _TEXT_KEYS:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def get_sentence_key(r: dict) -> str:
    return str(r.get("sentence_id") or r.get("content_id") or "unknown")

def build_normalizer(lowercase: bool, remove_punct: bool):
    if not _HAS_JIWER: return lambda s: s
    steps = []
    if lowercase: steps.append(ToLowerCase())
    if remove_punct: steps.append(RemovePunctuation())
    steps += [RemoveMultipleSpaces(), Strip()]
    return Compose(steps)

# ------------------------- 绘图函数 -------------------------
def generate_plots(df: pd.DataFrame, out_dir: Path):
    valid_df = df.dropna(subset=["duration_s"])
    if len(valid_df) == 0: return

    durs = valid_df["duration_s"].values
    wins = valid_df["num_windows"].values

    # 1. Duration Hist
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(durs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title("Distribution of Sentence Durations")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "hist_duration.svg", bbox_inches='tight')
    fig.savefig(out_dir / "hist_duration.pdf", bbox_inches='tight')
    plt.close(fig)

    # 2. Windows Hist
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.arange(wins.min(), wins.max() + 2) - 0.5
    ax.hist(wins, bins=bins, color='salmon', edgecolor='black', alpha=0.7)
    ax.set_title("Distribution of Windows per Sentence")
    ax.set_xlabel("Number of Windows")
    ax.set_ylabel("Count")
    fig.savefig(out_dir / "hist_windows.svg", bbox_inches='tight')
    fig.savefig(out_dir / "hist_windows.pdf", bbox_inches='tight')
    plt.close(fig)

    # 3. Scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(durs, wins, alpha=0.6, s=30)
    ax.set_title("Duration vs Windows")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Window Count")
    fig.savefig(out_dir / "scatter_dur_win.svg", bbox_inches='tight')
    fig.savefig(out_dir / "scatter_dur_win.pdf", bbox_inches='tight')
    plt.close(fig)

# ------------------------- 核心主程序 -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore_model", default="roberta-large")
    # 增加 batch size 参数，越大越快（只要不爆显存）
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for GPU inference")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取数据
    print("[1] Reading manifest...")
    with open(args.manifest, "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    groups = {}
    for r in rows:
        groups.setdefault(get_sentence_key(r), []).append(r)
    
    print(f" -> Found {len(groups)} unique sentence groups.")

    # 2. 提取统计 & 唯一文本列表
    stats_list = []
    unique_texts = []
    unique_ids = []

    for sid, items in groups.items():
        txt = extract_text(items[0])
        
        # 简单的时长计算
        onsets = [x.get("global_segment_onset_in_audio_s") for x in items if x.get("global_segment_onset_in_audio_s") is not None]
        offsets = [x.get("global_segment_offset_in_audio_s") for x in items if x.get("global_segment_offset_in_audio_s") is not None]
        
        # 兜底逻辑
        if not onsets: onsets = [x.get("local_window_onset_in_audio_s", 0) for x in items]
        if not offsets: offsets = [x.get("local_window_offset_in_audio_s", 0) for x in items]

        dur = float("nan")
        if onsets and offsets:
            dur = max(offsets) - min(onsets)

        stats_list.append({
            "sentence_id": sid,
            "text": txt,
            "duration_s": dur,
            "num_windows": len(items)
        })
        
        # 只有包含文本的句子才参与互评
        if txt:
            unique_texts.append(txt)
            unique_ids.append(sid)

    df_stats = pd.DataFrame(stats_list)
    df_stats.to_csv(out_dir / "sentence_stats.csv", index=False)
    generate_plots(df_stats, out_dir)
    print(" -> Stats and Plots saved.")

    # 3. 极速计算 (One vs All)
    print(f"[2] Starting Batch Evaluation on {len(unique_texts)} unique sentences...")
    
    normalizer = build_normalizer(args.lowercase, args.remove_punct)
    
    # 归一化所有文本（用于 WER/CER）
    normalized_refs = [normalizer(t) for t in unique_texts]
    
    # 准备结果容器
    # 结构: [{"sentence_id": ..., "text": ..., "WER": ..., ...}, ...]
    results = []

    # ---------------------- BERTScore (GPU Matrix Mode) ----------------------
    bert_f1_scores = None # 形状将是 (N, ) 的数组
    special_bert_f1 = None

    if args.bertscore and _HAS_BERTSCORE:
        device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        print(f" -> Loading BERTScore model on {device} (Batch Size: {args.batch_size})...")
        
        # 初始化 Scorer
        scorer = BERTScorer(model_type=args.bertscore_model, lang="en", rescale_with_baseline=False, device=device)

        # --- 策略：构造超级大列表一次性推理 ---
        N = len(unique_texts)
        
        # 1. "I don't know" vs All
        # 构造 N 个对: ("I don't know", ref1), ("I don't know", ref2)...
        special_cand = "I don't know"
        special_preds = [special_cand] * N
        
        print(" -> Computing BERTScore for 'I don't know'...")
        # P, R, F
        _, _, F_special = scorer.score(special_preds, unique_texts, batch_size=args.batch_size)
        special_bert_f1 = F_special.mean().item()

        # 2. All Candidates vs All Refs (N x N)
        # 这将产生 N*N 个对。对于 568 句，约 32万对。对于 3090/A100 显卡简直是小菜一碟。
        print(f" -> Computing BERTScore Matrix ({N}x{N} = {N*N} pairs)...")
        
        # 构造扁平化列表: 
        # Preds: [S1, S1, ..., S2, S2, ...] 
        # Refs:  [S1, S2, ..., S1, S2, ...]
        # 这样可以直接算完 reshape
        
        # 优化内存：我们实际上不需要物理复制 list 那么多次，
        # 但 bert_score API 需要 list 输入。
        # Python list of string 引用开销很小。
        flat_preds = []
        for t in unique_texts:
            flat_preds.extend([t] * N)
        
        flat_refs = unique_texts * N # 重复 N 次列表
        
        start_time = time.time()
        _, _, F_matrix = scorer.score(flat_preds, flat_refs, batch_size=args.batch_size)
        end_time = time.time()
        print(f"    Done in {end_time - start_time:.2f}s!")

        # Reshape: (N*N) -> (N, N) -> mean(dim=1) -> (N,)
        # F_matrix 是 tensor
        F_matrix = F_matrix.view(N, N)
        # 对每一行求平均（即：该句子作为 candidate，对应所有 ref 的平均分）
        bert_f1_scores = F_matrix.mean(dim=1).cpu().numpy()
        
        # 清理显存
        del scorer
        if _HAS_TORCH: torch.cuda.empty_cache()

    # ---------------------- WER/CER (CPU Loop) ----------------------
    # 虽然是 CPU，但 JiWER 很快，且不需要 N*N 的完全对比（有些库支持 broadcasting，但这里简单起见跑 N 次循环）
    # 对 500 个句子，跑 500 次 WER 计算是秒级的。
    
    print(" -> Computing WER/CER/BLEU...")
    
    # 1. Special "I don't know"
    special_norm = normalizer("I don't know")
    special_wer = jiwer_wer(normalized_refs, [special_norm] * len(normalized_refs)) if _HAS_JIWER else None
    special_res = {
        "candidate_text": "I don't know",
        "type": "special",
        "WER": special_wer,
        "BERTScore_F1": special_bert_f1
    }
    pd.DataFrame([special_res]).to_csv(out_dir / "special_baseline_idontknow.csv", index=False)
    print(f"    'I don't know' WER: {special_wer:.4f}")

    # 2. Matrix Loop for text metrics
    final_rows = []
    for i, txt in enumerate(unique_texts):
        # Normalized Candidate
        norm_c = normalized_refs[i] # 既然都是从 unique_texts 来的，直接取
        
        # WER: Candidate vs All Normalized Refs
        # jiwer 支持 list vs list，我们构造一个全是一样的 list
        preds_batch = [norm_c] * len(normalized_refs)
        
        w = jiwer_wer(normalized_refs, preds_batch) if _HAS_JIWER else None
        c = jiwer_cer(normalized_refs, preds_batch) if _HAS_JIWER else None
        
        # BLEU
        b = None
        if _HAS_SACREBLEU:
            # sacrebleu 计算 corpus bleu
            # sys: [txt, txt, ...], refs: [[ref1, ref2, ...]]
            # 注意：sacrebleu 期望 refs 是 list of list (多参考)，但在 corpus 模式下，
            # 我们把整个 unique_texts 当做验证集。
            # 这里的语境是：如果 predictions 全是 txt，相对于 references (unique_texts) 的分数。
            b = sacrebleu.corpus_bleu([txt]*len(unique_texts), [unique_texts]).score

        bf = bert_f1_scores[i] if bert_f1_scores is not None else None

        final_rows.append({
            "sentence_id": unique_ids[i],
            "candidate_text": txt,
            "WER": w,
            "CER": c,
            "BLEU": b,
            "BERTScore_F1": bf
        })

    # 保存结果
    df_res = pd.DataFrame(final_rows).sort_values("WER")
    df_res.to_csv(out_dir / "all_sentences_as_baseline.csv", index=False)

    print("\n[DONE] Results Summary:")
    if not df_res.empty:
        best = df_res.iloc[0]
        print(f"Best Centroid Sentence: '{best['candidate_text']}'")
        print(f"WER: {best['WER']:.4f} | BERT-F1: {best['BERTScore_F1']:.4f}")
    
    print(f"Files saved to: {out_dir}")

if __name__ == "__main__":
    main()