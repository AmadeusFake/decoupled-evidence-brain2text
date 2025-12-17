#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT="$PROJECT_ROOT/data_manifests_local_global"
SCRIPT_PATH="$PROJECT_ROOT/data_script/preprocess_meg_to_npy.py"

# 避免 /tmp 爆满 & MNE 缓存
export TMPDIR="$PROJECT_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export MNE_CACHE_DIR="$PROJECT_ROOT/mne_cache"
mkdir -p "$TMPDIR" "$MNE_CACHE_DIR"

# 限制 CPU 线程，避免 mne/numpy 争夺资源
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

process () {
  local data_type="$1"
  local in_dir="$DATA_ROOT/final_splits_${data_type}_precomputed"           # Stage-2 输出的 manifest
  local out_meg="$DATA_ROOT/precomputed_meg_windows/${data_type}"           # 保存 MEG 窗口/坐标/标尺
  local out_manifest="$DATA_ROOT/final_splits_${data_type}_fully_preprocessed"  # 新 manifest（含 meg_win_path/coords）

  mkdir -p "$out_meg" "$out_manifest"

  echo "== MEG Stage3: ${data_type} =="
  python "$SCRIPT_PATH" \
    --input_manifest_dir "$in_dir" \
    --output_meg_dir "$out_meg" \
    --output_manifest_dir "$out_manifest" \
    --num_workers 8 \
    --target_sfreq 120 \
    --baseline_end_s 0.3 \
    --std_clamp 20 \
    --fit_max_windows_per_recording 200 \
    --resume \
    --verify_existing
    # 如需全量重算，额外加： --recompute_existing
}

process "sentence"
process "word_list"

echo "All MEG pre-slicing done."
echo "Final manifests:"
echo "- $DATA_ROOT/final_splits_sentence_fully_preprocessed/"
echo "- $DATA_ROOT/final_splits_word_list_fully_preprocessed/"
