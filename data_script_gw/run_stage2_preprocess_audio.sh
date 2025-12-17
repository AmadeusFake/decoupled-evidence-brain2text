#!/bin/bash
set -euo pipefail

# -------------------- Paths -------------------- #
PROJECT_BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
SCRIPT_DIR="$PROJECT_BASE_DIR/data_script"
OUTPUT_DIR="$PROJECT_BASE_DIR/data_manifests_local_global"

PY="$SCRIPT_DIR/preprocess_audio_features.py"

# HF 缓存（首次会下载 W2V2 权重，以后直接命中缓存）
export HF_HOME="$PROJECT_BASE_DIR/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# -------------------- Perf knobs (A100/4090 友好) -------------------- #
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 限制 CPU 线程，避免与 DataLoader 抢核
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------------------- Tunables -------------------- #
# A100 建议 batch_size=192~256；3090/24GB 建议 128~160
BATCH_SIZE_SENT=192
BATCH_SIZE_WORD=192
NUM_WORKERS=8
PREFETCH=2
DEVICE=cuda
AMP=bf16           # 与脚本一致：bf16/ off
SAVE_DTYPE=float16 # 用 fp16 落盘以减小 IO（不影响训练端使用）

# 新增：Wav2Vec 层选择（与论文官方 YAML 一致）
W2V_LAYERS="14,15,16,17,18"
W2V_AGG="mean"  # mean 输出维=1024；如需拼接可设为 concat（则输出维=1024*层数）

run_one () {
  local data_type="$1"
  local in_dir="$OUTPUT_DIR/final_splits_${data_type}"
  local out_feat="$OUTPUT_DIR/precomputed_audio_features_${data_type}"
  local out_manifest="$OUTPUT_DIR/final_splits_${data_type}_precomputed"
  local bs="$2"

  echo "== Audio Stage2: ${data_type} (bs=${bs}, workers=${NUM_WORKERS}, prefetch=${PREFETCH}) =="
  python "$PY" \
    --input_manifest_dir "$in_dir" \
    --output_feature_dir "$out_feat" \
    --output_manifest_dir "$out_manifest" \
    --batch_size "$bs" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH" \
    --device "$DEVICE" \
    --amp "$AMP" \
    --save_dtype "$SAVE_DTYPE" \
    --w2v_layers "$W2V_LAYERS" \
    --w2v_agg "$W2V_AGG"
}

# 两个集合：sentence / word_list
run_one "sentence"  "$BATCH_SIZE_SENT"
run_one "word_list" "$BATCH_SIZE_WORD"

echo "All audio features done."
