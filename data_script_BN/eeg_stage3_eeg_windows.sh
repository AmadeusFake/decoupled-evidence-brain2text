#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Root directory produced by Stage-1 and Stage-2
DATA_ROOT="${PROJECT_ROOT}/EEGdata_manifests_local_global"

# EEG preprocessing script (Stage-3, Brennan-adapted)
SCRIPT_PATH="${PROJECT_ROOT}/data_script_BN/preprocess_eeg_to_npy.py"

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

INPUT_MANIFEST_DIR="${DATA_ROOT}/final_splits_sentence_precomputed"
OUTPUT_EEG_DIR="${DATA_ROOT}/precomputed_eeg_windows/sentence"
OUTPUT_MANIFEST_DIR="${DATA_ROOT}/final_splits_sentence_fully_preprocessed"

###############################################################################
# Temporary directories and caches
###############################################################################

# Local temporary directory (important on HPC)
export TMPDIR="${PROJECT_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

# MNE cache for FIF / resampling intermediates
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
# Stage 3: EEG preprocessing and window extraction (sentence-only)
###############################################################################

echo "== Stage 3: EEG preprocessing and window slicing (sentence-only) =="

python "${SCRIPT_PATH}" \
  --input_manifest_dir  "${INPUT_MANIFEST_DIR}" \
  --output_eeg_dir      "${OUTPUT_EEG_DIR}" \
  --output_manifest_dir "${OUTPUT_MANIFEST_DIR}" \
  --target_sfreq 120 \
  --baseline_end_s 0.3 \
  --std_clamp 20 \
  --fit_max_windows_per_recording 200 \
  --resume \
  --verify_existing
  # For a full rebuild, add: --recompute_existing

echo
echo "EEG preprocessing completed successfully ✅"
echo "Final EEG windows directory:"
echo "- ${OUTPUT_EEG_DIR}"
echo "Final manifests:"
echo "- ${OUTPUT_MANIFEST_DIR}"
