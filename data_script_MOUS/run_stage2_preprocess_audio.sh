#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Directory containing MOUS preprocessing scripts
# NOTE: Uses the shared audio preprocessing script
SCRIPT_DIR="${PROJECT_BASE_DIR}/data_script_MOUS"

# Root directory for MOUS manifests and splits
DATA_ROOT_MOUS="${PROJECT_BASE_DIR}/data_mous_local_global"

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

PY_SCRIPT="${SCRIPT_DIR}/preprocess_audio_features.py"

INPUT_MANIFEST_DIR="${DATA_ROOT_MOUS}/final_splits_sentence"
OUTPUT_FEATURE_DIR="${DATA_ROOT_MOUS}/precomputed_audio_features_sentence"
OUTPUT_MANIFEST_DIR="${DATA_ROOT_MOUS}/final_splits_sentence_precomputed"

###############################################################################
# HuggingFace cache (wav2vec2 weights)
###############################################################################

export HF_HOME="${PROJECT_BASE_DIR}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

###############################################################################
# Performance knobs
###############################################################################

# Enable TF32 where supported (safe for feature extraction)
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# Reduce CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Limit CPU threads to avoid DataLoader contention
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

###############################################################################
# Tunable parameters
###############################################################################

# Batch size (A100: 192–256; 3090/24GB: 128–160)
BATCH_SIZE_SENT=192

# Data loading
NUM_WORKERS=8
PREFETCH_FACTOR=2

# Device / precision
DEVICE="cuda"
AMP="bf16"            # bf16 or off (must match script support)
SAVE_DTYPE="float16"  # fp16 on disk to reduce I/O

# Wav2Vec2 configuration (aligned with paper)
W2V_MODEL="facebook/wav2vec2-large-xlsr-53"
W2V_LAYERS="14,15,16,17,18"
W2V_AGG="mean"        # mean -> 1024 dims

###############################################################################
# Stage 2: Audio feature extraction (MOUS, sentence-only)
###############################################################################

echo "== [MOUS] Audio Stage 2: sentence-level =="
echo "Batch size: ${BATCH_SIZE_SENT}"
echo "W2V model: ${W2V_MODEL}"
echo "Layers: ${W2V_LAYERS} | agg=${W2V_AGG}"

python "${PY_SCRIPT}" \
  --input_manifest_dir "${INPUT_MANIFEST_DIR}" \
  --output_feature_dir "${OUTPUT_FEATURE_DIR}" \
  --output_manifest_dir "${OUTPUT_MANIFEST_DIR}" \
  --batch_size "${BATCH_SIZE_SENT}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --device "${DEVICE}" \
  --amp "${AMP}" \
  --save_dtype "${SAVE_DTYPE}" \
  --w2v_model "${W2V_MODEL}" \
  --w2v_layers "${W2V_LAYERS}" \
  --w2v_agg "${W2V_AGG}" \
  --resume \
  --verify_existing

###############################################################################
# Final report
###############################################################################

echo
echo "MOUS audio feature extraction completed successfully ✅"
echo "Features written to:"
echo "- ${OUTPUT_FEATURE_DIR}"
echo "Updated manifests:"
echo "- ${OUTPUT_MANIFEST_DIR}"
