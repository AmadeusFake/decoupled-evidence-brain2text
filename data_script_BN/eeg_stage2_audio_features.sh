#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_BASE_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Directory containing preprocessing scripts (BN version, do not rename)
SCRIPT_DIR="${PROJECT_BASE_DIR}/data_script_BN"

# Root directory of Brennan EEG dataset
# Must contain subdirectory: audio/
EEG_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/Brennan"

# Output directory produced by Stage-1 (manifests & splits)
OUTPUT_DIR="${PROJECT_BASE_DIR}/EEGdata_manifests_local_global"

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

PY_SCRIPT="${SCRIPT_DIR}/preprocess_audio_features.py"

INPUT_MANIFEST_DIR="${OUTPUT_DIR}/final_splits_sentence"
OUTPUT_FEATURE_DIR="${OUTPUT_DIR}/precomputed_audio_features_sentence"
OUTPUT_MANIFEST_DIR="${OUTPUT_DIR}/final_splits_sentence_precomputed"

###############################################################################
# HuggingFace / cache configuration
###############################################################################

# Local cache to avoid repeated downloads on HPC
export HF_HOME="${PROJECT_BASE_DIR}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

###############################################################################
# Runtime / performance configuration
###############################################################################

# Enable TF32 where available (safe for audio feature extraction)
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

# Reduce CUDA fragmentation for large batches
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CPU thread limits
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

###############################################################################
# Audio feature extraction parameters
###############################################################################

# Batch size for sentence-level audio segments
BATCH_SIZE_SENT=192

# Data loading
NUM_WORKERS=8
PREFETCH_FACTOR=2

# Device / precision
DEVICE="cuda"
AMP="off"
SAVE_DTYPE="float32"

# Wav2Vec2 feature selection
# Layers follow wav2vec2-large convention
W2V_LAYERS="14,15,16,17,18"
W2V_AGG="mean"

###############################################################################
# Stage 2: Audio feature extraction (sentence-only)
###############################################################################

echo "== Stage 2: Audio feature extraction (sentence-only) =="
echo "Batch size: ${BATCH_SIZE_SENT}"
echo "W2V layers: ${W2V_LAYERS} (agg=${W2V_AGG})"

python "${PY_SCRIPT}" \
  --input_manifest_dir "${INPUT_MANIFEST_DIR}" \
  --output_feature_dir "${OUTPUT_FEATURE_DIR}" \
  --output_manifest_dir "${OUTPUT_MANIFEST_DIR}" \
  --audio_root "${EEG_ROOT}/audio" \
  --batch_size "${BATCH_SIZE_SENT}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --device "${DEVICE}" \
  --amp "${AMP}" \
  --save_dtype "${SAVE_DTYPE}" \
  --w2v_layers "${W2V_LAYERS}" \
  --w2v_agg "${W2V_AGG}"

echo
echo "Audio feature extraction completed successfully ✅"
echo "Features saved to: ${OUTPUT_FEATURE_DIR}"
echo "Updated manifests: ${OUTPUT_MANIFEST_DIR}"
