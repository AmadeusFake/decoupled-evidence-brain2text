#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT="$PROJECT_ROOT/data_manifests_local_global"
SCRIPT_PATH="$PROJECT_ROOT/data_script_gw/stage3_5_add_sentence_full.py"   # ← 上一步我给你的 3.5 脚本路径

# 避免 /tmp 爆满 & MNE 缓存
export TMPDIR="$PROJECT_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export MNE_CACHE_DIR="$PROJECT_ROOT/mne_cache"
mkdir -p "$TMPDIR" "$MNE_CACHE_DIR"

# 限制 CPU 线程
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

process () {
  local data_type="$1"

  # 输入：stage3 产物（含 meg_win_path / sensor_coordinates_path 等）
  local in_dir="$DATA_ROOT/final_splits_${data_type}_fully_preprocessed"

  # 输出：整句 MEG 存放目录（按 recording 分子目录），以及“带有 meg_sentence_full_path 的新 manifest”
  local out_sentence="$DATA_ROOT/precomputed_meg_sentence_full/${data_type}"
  local out_manifest="$DATA_ROOT/final_splits_${data_type}_with_sentence_full"

  mkdir -p "$out_sentence" "$out_manifest"

  echo "== MEG Stage3.5 (add sentence-full): ${data_type} =="
  python "$SCRIPT_PATH" \
    --input_manifest_dir "$in_dir" \
    --output_sentence_dir "$out_sentence" \
    --output_manifest_dir "$out_manifest" \
    --target_sfreq 120 \
    --baseline_end_s 0.3 \
    --std_clamp 20 \
    --fit_max_windows_per_recording 200 \
    --resume \
    --verify_existing
    # 如需全量重算整句，加： --recompute_existing
}

# 一般我们只需要 sentence；如果也想给 word_list 做整句（可选）就一起跑
process "sentence"
process "word_list"

echo "All sentence-full MEG done."
echo "New manifests (with 'meg_sentence_full_path'):"
echo "- $DATA_ROOT/final_splits_sentence_with_sentence_full/"
echo "- $DATA_ROOT/final_splits_word_list_with_sentence_full/"

echo
echo "训练/评测时，把 --train/--val/--test manifest 指到以上 *_with_sentence_full 目录即可。"
