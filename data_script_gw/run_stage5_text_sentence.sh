#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths
# =========================
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT="$PROJECT_ROOT/data_manifests_local_global"
SCRIPT_DIR="$PROJECT_ROOT/data_script_gw"

# Python scripts
BUILD_SENT_TABLE="$SCRIPT_DIR/build_sentence_table_from_window_manifest.py"
ATTACH_BACK="$SCRIPT_DIR/attach_sentence_features_to_windows.py"
EXPORT_QC="$SCRIPT_DIR/export_sentence_clips_txt.py"      # Whisper 导出 wav+txt
E5_FROM_QC="$SCRIPT_DIR/stageT_from_qc_txt_e5.py"         # 从 QC txt 用 E5 编码

# Input window manifests
WIN_SENT_DIR="$DATA_ROOT/final_splits_sentence_with_sentence_full"
WIN_WORD_DIR="$DATA_ROOT/final_splits_word_list_with_sentence_full"

# Dedup sentence tables
SENT_TABLE_SENT_DIR="$DATA_ROOT/sent_table_dedup/sentence"
SENT_TABLE_WORD_DIR="$DATA_ROOT/sent_table_dedup/word_list"

# QC outputs (wav + txt + tsv)
QC_DIR_SENT="$DATA_ROOT/qc_text_sentence_whisper/sentence"
QC_DIR_WORD="$DATA_ROOT/qc_text_sentence_whisper/word_list"

# Text features (E5) from QC txt
TEXT_FEAT_SENT_DIR="$DATA_ROOT/precomputed_text_sentence_features/sentence_whisper/npy"
TEXT_INDEX_SENT_DIR="$DATA_ROOT/precomputed_text_sentence_features/sentence_whisper/sent_index"

TEXT_FEAT_WORD_DIR="$DATA_ROOT/precomputed_text_sentence_features/word_list_whisper/npy"
TEXT_INDEX_WORD_DIR="$DATA_ROOT/precomputed_text_sentence_features/word_list_whisper/sent_index"

# Windows with text features attached
WIN_WITH_TEXT_SENT_DIR="$DATA_ROOT/final_splits_sentence_with_sentence_full_text_whisper"
WIN_WITH_TEXT_WORD_DIR="$DATA_ROOT/final_splits_word_list_with_sentence_full_text_whisper"

# Audio roots
AUDIO_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/download/stimuli/audio"

# =========================
# Environment
# =========================
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true
module load FFmpeg/6.0-GCCcore-12.3.0 || true

if [[ -d "/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/whisper_venv" ]]; then
  source "/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/whisper_venv/bin/activate"
fi

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$PROJECT_ROOT/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME"

mkdir -p "$PROJECT_ROOT/runs" \
         "$SENT_TABLE_SENT_DIR" "$SENT_TABLE_WORD_DIR" \
         "$QC_DIR_SENT" "$QC_DIR_WORD" \
         "$TEXT_FEAT_SENT_DIR" "$TEXT_INDEX_SENT_DIR" \
         "$TEXT_FEAT_WORD_DIR" "$TEXT_INDEX_WORD_DIR" \
         "$WIN_WITH_TEXT_SENT_DIR" "$WIN_WITH_TEXT_WORD_DIR"

# =========================
# Helper: run one split-kind (sentence / word_list)
# =========================
run_pipeline () {
  local KIND="$1"  # "sentence" or "word_list"

  if [[ "$KIND" == "sentence" ]]; then
    local WIN_DIR="$WIN_SENT_DIR"
    local TAB_DIR="$SENT_TABLE_SENT_DIR"
    local QC_DIR="$QC_DIR_SENT"
    local FEAT_DIR="$TEXT_FEAT_SENT_DIR"
    local IDX_DIR="$TEXT_INDEX_SENT_DIR"
    local WIN_OUT_DIR="$WIN_WITH_TEXT_SENT_DIR"
  else
    local WIN_DIR="$WIN_WORD_DIR"
    local TAB_DIR="$SENT_TABLE_WORD_DIR"
    local QC_DIR="$QC_DIR_WORD"
    local FEAT_DIR="$TEXT_FEAT_WORD_DIR"
    local IDX_DIR="$TEXT_INDEX_WORD_DIR"
    local WIN_OUT_DIR="$WIN_WITH_TEXT_WORD_DIR"
  fi

  echo
  echo "== [$KIND] Step 1: Build *dedup* sentence table from $WIN_DIR =="
  python "$BUILD_SENT_TABLE" \
    --input_window_manifest_dir "$WIN_DIR" \
    --output_sentence_table_dir "$TAB_DIR"

  echo
  echo "== [$KIND] Step 2: Export WAV+TXT via Whisper =="
  if ls "$QC_DIR"/train/audio/*.txt >/dev/null 2>&1; then
    echo "[INFO][$KIND] Whisper 已经处理过 ($QC_DIR)，跳过。"
  else
    python "$EXPORT_QC" \
      --sentence_table_dir "$TAB_DIR" \
      --out_dir "$QC_DIR" \
      --audio_roots "$AUDIO_ROOT" \
      --txt_source whisper \
      --whisper_model large-v3 \
      --whisper_device cuda \
      --whisper_language en \
      --head_pad_s 0.00 \
      --tail_pad_s 0.25
  fi

  echo
  echo "== [$KIND] Step 3: Build E5 vectors from QC TXT =="
  if [[ -f "$IDX_DIR/train/sent_index.jsonl" ]]; then
    echo "[INFO][$KIND] E5 向量已存在 ($IDX_DIR)，跳过。"
  else
    python "$E5_FROM_QC" \
      --qc_dir "$QC_DIR" \
      --out_index_dir "$IDX_DIR" \
      --out_feat_dir "$FEAT_DIR" \
      --e5_model "intfloat/e5-large-v2" \
      --device cuda \
      --batch_size 128
  fi

  echo
  echo "== [$KIND] Step 4: Attach TEXT features back to windows =="
  python "$ATTACH_BACK" \
    --input_window_manifest_dir "$WIN_DIR" \
    --input_sentence_manifest_dir "$IDX_DIR" \
    --output_window_manifest_dir "$WIN_OUT_DIR" \
    --feature_field "text_sentence_feature_path" \
    --output_feature_field "text_sentence_feature_path" \
    --emit_mapping_tsv

  echo
  echo "✅ [$KIND] Whisper→E5 pipeline done."
  echo "- $KIND 句子表:     $TAB_DIR/{train,valid,test}.jsonl"
  echo "- $KIND QC:         $QC_DIR/{train,valid,test}/audio/*.{wav,txt}"
  echo "- $KIND E5 向量:    $FEAT_DIR | 索引: $IDX_DIR/{split}.jsonl"
  echo "- $KIND 窗口回填:   $WIN_OUT_DIR/{split}.jsonl"
}

# =========================
# Run both kinds
# =========================
run_pipeline "sentence"
run_pipeline "word_list"

echo
echo "🎉 All done: Stage5 TEXT (Whisper→E5→Attach) 完成"
