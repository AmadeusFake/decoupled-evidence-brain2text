#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT="$PROJECT_ROOT/EEGdata_manifests_local_global"
SCRIPT_PATH="$PROJECT_ROOT/data_script_BN/preprocess_eeg_to_npy.py"  # 你现有的 EEG Stage-3 脚本（已适配 Brennan）

export TMPDIR="$PROJECT_ROOT/tmp"; export TEMP="$TMPDIR"; export TMP="$TMPDIR"
export MNE_CACHE_DIR="$PROJECT_ROOT/mne_cache"
mkdir -p "$TMPDIR" "$MNE_CACHE_DIR"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "== EEG Stage3: sentence-only =="
python "$SCRIPT_PATH" \
  --input_manifest_dir  "$DATA_ROOT/final_splits_sentence_precomputed" \
  --output_eeg_dir      "$DATA_ROOT/precomputed_eeg_windows/sentence" \
  --output_manifest_dir "$DATA_ROOT/final_splits_sentence_fully_preprocessed" \
  --target_sfreq 120 \
  --baseline_end_s 0.3 \
  --std_clamp 20 \
  --fit_max_windows_per_recording 200 \
  --resume \
  --verify_existing
  # 全量重算可加: --recompute_existing

echo "All EEG pre-slicing done (sentence-only)."
echo "Final manifests:"
echo "- $DATA_ROOT/final_splits_sentence_fully_preprocessed/"
