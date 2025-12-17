#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT="$PROJECT_ROOT/data_manifests_local_global"
SCRIPT_DIR="$PROJECT_ROOT/data_script_gw"

PY_RESPLIT="$SCRIPT_DIR/resplit_from_existing_manifest.py"

# 输入：你现有的 final_splits（句子 / 单词列表）
IN_SENT="$DATA_ROOT/final_splits_sentence_with_sentence_full_audio_attn"
IN_WORD="$DATA_ROOT/final_splits_word_list_with_sentence_full_audio_attn"

# 输出：新 resplit 目录（避免覆盖旧结果）
OUT_SENT="$DATA_ROOT/resplit_sentence_SLR"
OUT_WORD="$DATA_ROOT/resplit_word_list_SLR"

# 比例（train,val,test），SLR 推荐 0.7,0.1,0.2 或 0.7,0.2,0.1
RATIOS="0.7,0.1,0.2"
SEED=42

# 最小配额：-1 表示自动 ceil(n * ratio)，且至少 1（当 n>=3）
MIN_VALID=-1
MIN_TEST=-1

# 开关：
#  - require_subject_in_train：每个被试至少一场在 Train（强约束，默认开）
#  - soft_cover_subjects_test：尽量每个被试 Test 也出现（软约束，尽力而为）
#  - enforce_covered_test_only：过滤掉 test 中训练未覆盖的句子（建议开，评测更纯净）
run_one () {
  local IN_DIR="$1"; local OUT_DIR="$2"; local NAME="$3"
  echo "== [SLR v4] Resplit ($NAME) ratios=$RATIOS =="
  python "$PY_RESPLIT" \
    --input_manifest_dir "$IN_DIR" \
    --output_dir        "$OUT_DIR" \
    --split_ratios "$RATIOS" \
    --random_seed $SEED \
    --min_valid_recordings $MIN_VALID \
    --min_test_recordings  $MIN_TEST \
    --require_subject_in_train \
    --soft_cover_subjects_test \
    --enforce_covered_test_only
}

run_one "$IN_SENT" "$OUT_SENT" "sentence"
run_one "$IN_WORD" "$OUT_WORD" "word_list"

echo
echo "✅ SLR v4 resplit done:"
echo "- $OUT_SENT/{train,valid,test}.jsonl"
echo "- $OUT_WORD/{train,valid,test}.jsonl"
