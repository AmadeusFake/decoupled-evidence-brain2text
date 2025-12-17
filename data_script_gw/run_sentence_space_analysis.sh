#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT="$PROJECT_ROOT/data_manifests_local_global"
SCRIPT_DIR="$PROJECT_ROOT/data_script_gw"

ANALYZER="$SCRIPT_DIR/analyze_sentence_space_e5.py"

# 输入：Whisper→E5 的索引
IDX_SENT="$DATA_ROOT/precomputed_text_sentence_features/sentence_whisper/sent_index"
IDX_WORD="$DATA_ROOT/precomputed_text_sentence_features/word_list_whisper/sent_index"

# 输出目录
OUT_SENT="$DATA_ROOT/analysis_text_sentence_global/sentence"
OUT_WORD="$DATA_ROOT/analysis_text_sentence_global/word_list"

# 环境
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true
if [[ -d "/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/whisper_venv" ]]; then
  source "/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/whisper_venv/bin/activate"
fi

mkdir -p "$OUT_SENT" "$OUT_WORD"

run_one () {
  local KIND="$1"    # sentence | word_list
  local IDX_DIR="$2" # .../sent_index
  local OUT_DIR="$3" # analysis out root

  for SPLIT in train valid test; do
    local INJ="$IDX_DIR/$SPLIT.jsonl"
    local ODIR="$OUT_DIR/$SPLIT"
    if [[ ! -f "$INJ" ]]; then
      echo "[WARN][$KIND/$SPLIT] missing: $INJ (skip)"
      continue
    fi
    mkdir -p "$ODIR"
    echo "[INFO][$KIND/$SPLIT] analyzing: $INJ"
    python "$ANALYZER" \
      --index_jsonl "$INJ" \
      --out_dir "$ODIR" \
      --pair_samples 50000 \
      --dump_sampled_pairs_tsv
  done
  echo "✅ $KIND 输出在: $OUT_DIR"
}

run_one "sentence"  "$IDX_SENT" "$OUT_SENT"
run_one "word_list" "$IDX_WORD" "$OUT_WORD"

echo "🎯 Global-negative sentence-space analysis done."
