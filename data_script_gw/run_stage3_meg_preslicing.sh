#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Root directory containing manifests and precomputed audio features
DATA_ROOT="${PROJECT_ROOT}/data_manifests_local_global"

# MEG preprocessing script (Stage-3)
SCRIPT_PATH="${PROJECT_ROOT}/data_script/preprocess_meg_to_npy.py"

###############################################################################
# Temporary directories and caches
###############################################################################

# Avoid /tmp overflow on HPC systems
export TMPDIR="${PROJECT_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

# MNE cache for FIF loading, filtering, and resampling
export MNE_CACHE_DIR="${PROJECT_ROOT}/mne_cache"

mkdir -p "${TMPDIR}" "${MNE_CACHE_DIR}"

###############################################################################
# CPU threading limits
###############################################################################

# Prevent oversubscription in MNE / NumPy
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

###############################################################################
# Helper: process one data family (sentence / word_list)
###############################################################################

process_one () {
  local DATA_TYPE="$1"

  # Stage-2 output manifests (audio already precomputed)
  local INPUT_MANIFEST_DIR="${DATA_ROOT}/final_splits_${DATA_TYPE}_precomputed"

  # Output directories
  local OUTPUT_MEG_DIR="${DATA_ROOT}/precomputed_meg_windows/${DATA_TYPE}"
  local OUTPUT_MANIFEST_DIR="${DATA_ROOT}/final_splits_${DATA_TYPE}_fully_preprocessed"

  mkdir -p "${OUTPUT_MEG_DIR}" "${OUTPUT_MANIFEST_DIR}"

  echo "== MEG Stage 3: ${DATA_TYPE} =="

  python "${SCRIPT_PATH}" \
    --input_manifest_dir "${INPUT_MANIFEST_DIR}" \
    --output_meg_dir "${OUTPUT_MEG_DIR}" \
    --output_manifest_dir "${OUTPUT_MANIFEST_DIR}" \
    --num_workers 8 \
    --target_sfreq 120 \
    --baseline_end_s 0.3 \
    --std_clamp 20 \
    --fit_max_windows_per_recording 200 \
    --resume \
    --verify_existing
    # For a full rebuild, add: --recompute_existing
}

###############################################################################
# Run MEG preprocessing for sentence and word-list splits
###############################################################################

process_one "sentence"
process_one "word_list"

###############################################################################
# Final report
###############################################################################

echo
echo "MEG preprocessing completed successfully ✅"
echo "Final manifests:"
echo "- ${DATA_ROOT}/final_splits_sentence_fully_preprocessed/"
echo "- ${DATA_ROOT}/final_splits_word_list_fully_preprocessed/"
