#!/usr/bin/env bash
set -euo pipefail

PROJECT_BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
SCRIPT_DIR="$PROJECT_BASE_DIR/data_script_BN"                  # 使用 BN 版 Stage-2
OUTPUT_DIR="$PROJECT_BASE_DIR/EEGdata_manifests_local_global"
EEG_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/Brennan"  # 用于 --audio_root

PY="$SCRIPT_DIR/preprocess_audio_features.py"

export HF_HOME="$PROJECT_BASE_DIR/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

BATCH_SIZE_SENT=192
NUM_WORKERS=8
PREFETCH=2
DEVICE=cuda
AMP=off
SAVE_DTYPE=float32
W2V_LAYERS="14,15,16,17,18"
W2V_AGG="mean"

echo "== Audio Stage2: sentence-only (bs=${BATCH_SIZE_SENT}) =="
python "$PY" \
  --input_manifest_dir "$OUTPUT_DIR/final_splits_sentence" \
  --output_feature_dir "$OUTPUT_DIR/precomputed_audio_features_sentence" \
  --output_manifest_dir "$OUTPUT_DIR/final_splits_sentence_precomputed" \
  --audio_root "$EEG_ROOT/audio" \
  --batch_size "$BATCH_SIZE_SENT" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --save_dtype "$SAVE_DTYPE" \
  --w2v_layers "$W2V_LAYERS" \
  --w2v_agg "$W2V_AGG"

echo "Audio features done (sentence-only)."
