#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Directory containing preprocessing scripts
SCRIPT_DIR="${PROJECT_BASE_DIR}/data_script"

# Output directory produced by Stage-1 (manifests & splits)
OUTPUT_DIR="${PROJECT_BASE_DIR}/data_manifests_local_global"

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

PY_SCRIPT="${SCRIPT_DIR}/preprocess_audio_features.py"

###############################################################################
# HuggingFace cache (weights downloaded once, then reused)
###############################################################################

export HF_HOME="${PROJECT_BASE_DIR}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

###############################################################################
# Performance knobs (GPU-friendly defaults)
###############################################################################

# Enable TF32 where supported (safe for feature extraction)
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# Reduce CUDA memory fragmentation for large batches
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit CPU threads to avoid DataLoader contention
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

###############################################################################
# Tunable parameters
###############################################################################

# Recommended batch sizes:
#   A100 (40/80GB): 192–256
#   RTX 3090 / 24GB: 128–160
BATCH_SIZE_SENT=192
BATCH_SIZE_WORD=192

# Data loading
NUM_WORKERS=8
PREFETCH_FACTOR=2

# Device / precision
DEVICE="cuda"
AMP="bf16"            # bf16 or off (must match script support)
SAVE_DTYPE="float16"  # fp16 on disk to reduce I/O and storage

# Wav2Vec2 feature configuration (aligned with paper YAML)
W2V_LAYERS="14,15,16,17,18"
W2V_AGG="mean"        # mean -> 1024 dims; concat -> 1024 * num_layers

###############################################################################
# Helper: run audio feature extraction for one data type
###############################################################################

run_one () {
  local DATA_TYPE="$1"
  local BATCH_SIZE="$2"

  local INPUT_MANIFEST_DIR="${OUTPUT_DIR}/final_splits_${DATA_TYPE}"
  local OUTPUT_FEATURE_DIR="${OUTPUT_DIR}/precomputed_audio_features_${DATA_TYPE}"
  local OUTPUT_MANIFEST_DIR="${OUTPUT_DIR}/final_splits_${DATA_TYPE}_precomputed"

  echo "== Audio Stage 2: ${DATA_TYPE} | batch_size=${BATCH_SIZE} =="

  python "${PY_SCRIPT}" \
    --input_manifest_dir "${INPUT_MANIFEST_DIR}" \
    --output_feature_dir "${OUTPUT_FEATURE_DIR}" \
    --output_manifest_dir "${OUTPUT_MANIFEST_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --prefetch_factor "${PREFETCH_FACTOR}" \
    --device "${DEVICE}" \
    --amp "${AMP}" \
    --save_dtype "${SAVE_DTYPE}" \
    --w2v_layers "${W2V_LAYERS}" \
    --w2v_agg "${W2V_AGG}"
}

###############################################################################
# Run for sentence-level and word-list splits
###############################################################################

run_one "sentence"  "${BATCH_SIZE_SENT}"
run_one "word_list" "${BATCH_SIZE_WORD}"

###############################################################################
# Final report
###############################################################################

echo
echo "Audio feature extraction completed successfully ✅"
echo "Outputs written under:"
echo "- ${OUTPUT_DIR}/precomputed_audio_features_*"
