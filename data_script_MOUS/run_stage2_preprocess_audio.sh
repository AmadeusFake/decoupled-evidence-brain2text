#!/bin/bash
set -euo pipefail

# -------------------- Paths -------------------- #
PROJECT_BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
SCRIPT_DIR="$PROJECT_BASE_DIR/data_script_MOUS"                  # 注意：这里用的是通用的 data_script
DATA_ROOT_MOUS="$PROJECT_BASE_DIR/data_mous_local_global"  # MOUS 的 manifests 根目录
PY="$SCRIPT_DIR/preprocess_audio_features.py"

# HF 缓存
export HF_HOME="$PROJECT_BASE_DIR/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# -------------------- Perf knobs -------------------- #
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# -------------------- Tunables -------------------- #
BATCH_SIZE_SENT=192
NUM_WORKERS=8
PREFETCH=2
DEVICE=cuda
AMP=bf16           # 和脚本里的选项一致：bf16 / off
SAVE_DTYPE=float16 # 减少 IO

# 和论文对齐的 Wav2Vec 层
W2V_MODEL="facebook/wav2vec2-large-xlsr-53"
W2V_LAYERS="14,15,16,17,18"
W2V_AGG="mean"  # 输出 1024 维

run_sentence () {
  local in_dir="$DATA_ROOT_MOUS/final_splits_sentence"              # 来自 MOUS Stage-1
  local out_feat="$DATA_ROOT_MOUS/precomputed_audio_features_sentence"
  local out_manifest="$DATA_ROOT_MOUS/final_splits_sentence_precomputed"

  echo "== [MOUS] Audio Stage2: sentence (bs=${BATCH_SIZE_SENT}) =="
  python "$PY" \
    --input_manifest_dir "$in_dir" \
    --output_feature_dir "$out_feat" \
    --output_manifest_dir "$out_manifest" \
    --batch_size "$BATCH_SIZE_SENT" \
    --num_workers "$NUM_WORKERS" \
    --prefetch_factor "$PREFETCH" \
    --device "$DEVICE" \
    --amp "$AMP" \
    --save_dtype "$SAVE_DTYPE" \
    --w2v_model "$W2V_MODEL" \
    --w2v_layers "$W2V_LAYERS" \
    --w2v_agg "$W2V_AGG" \
    --resume \
    --verify_existing
}

# MOUS 我们目前只做 sentence，不做 word_list
run_sentence

echo "All MOUS audio features done."
