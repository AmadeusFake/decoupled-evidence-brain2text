#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Root directory containing Stage-3 outputs
DATA_ROOT="${PROJECT_ROOT}/data_manifests_local_global"

# Stage-3.5 script: add sentence-level full MEG signals
SCRIPT_PATH="${PROJECT_ROOT}/data_script_gw/stage3_5_add_sentence_full.py"

###############################################################################
# Temporary directories and caches
###############################################################################

# Avoid /tmp overflow on HPC systems
export TMPDIR="${PROJECT_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

# MNE cache for loading and resampling
export MNE_CACHE_DIR="${PROJECT_ROOT}/mne_cache"

mkdir -p "${TMPDIR}" "${MNE_CACHE_DIR}"

###############################################################################
# CPU threading limits
###############################################################################

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

###############################################################################
# Helper: add sentence-full MEG signals for one data family
###############################################################################

process_one () {
  local DATA_TYPE="$1"

  # Input: Stage-3 fully preprocessed manifests
  local INPUT_MANIFEST_DIR="${DATA_ROOT}/final_splits_${DATA_TYPE}_fully_preprocessed"

  # Output: sentence-level full MEG signals
  local OUTPUT_SENTENCE_DIR="${DATA_ROOT}/precomputed_meg_sentence_full/${DATA_TYPE}"

  # Output: updated manifests with `meg_sentence_full_path`
  local OUTPUT_MANIFEST_DIR="${DATA_ROOT}/final_splits_${DATA_TYPE}_with_sentence_full"

  mkdir -p "${OUTPUT_SENTENCE_DIR}" "${OUTPUT_MANIFEST_DIR}"

  echo "== MEG Stage 3.5: add sentence-full signals | ${DATA_TYPE} =="

  python "${SCRIPT_PATH}" \
    --input_manifest_dir "${INPUT_MANIFEST_DIR}" \
    --output_sentence_dir "${OUTPUT_SENTENCE_DIR}" \
    --output_manifest_dir "${OUTPUT_MANIFEST_DIR}" \
    --target_sfreq 120 \
    --baseline_end_s 0.3 \
    --std_clamp 20 \
    --fit_max_windows_per_recording 200 \
    --resume \
    --verify_existing
    # For a full rebuild, add: --recompute_existing
}

###############################################################################
# Run Stage-3.5
###############################################################################

# Sentence-level is required; word_list is optional but supported
process_one "sentence"
process_one "word_list"

###############################################################################
# Final report
###############################################################################

echo
echo "Sentence-full MEG preprocessing completed successfully ✅"
echo "New manifests (with 'meg_sentence_full_path'):"
echo "- ${DATA_ROOT}/final_splits_sentence_with_sentence_full/"
echo "- ${DATA_ROOT}/final_splits_word_list_with_sentence_full/"

echo
echo "For training/evaluation, point --train/--val/--test manifests to the"
echo "*_with_sentence_full directories above."
