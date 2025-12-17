#!/usr/bin/env bash
set -euo pipefail

# ====== 路径（按需检查/修改） ======
PROJECT_BASE="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
SCRIPT_DIR="$PROJECT_BASE/data_script_BN"    # Python脚本目录（BN 版）
EEG_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/Brennan"  # 含 audio/, proc/, Sxx.mat, AliceChapterOne-EEG.csv
SCORES_FILE="$EEG_ROOT/comprehension-scores.txt"

OUT_DIR="$PROJECT_BASE/EEGdata_manifests_local_global"
META="$OUT_DIR/meta_manifest.jsonl"

# ====== 基础检查 ======
echo "== 基础检查 =="
[[ -d "$EEG_ROOT/audio" ]] || { echo "缺少 $EEG_ROOT/audio 目录"; exit 2; }
[[ -d "$EEG_ROOT/proc"  ]] || { echo "缺少 $EEG_ROOT/proc 目录"; exit 2; }
[[ -f "$EEG_ROOT/AliceChapterOne-EEG.csv" ]] || { echo "缺少 $EEG_ROOT/AliceChapterOne-EEG.csv"; exit 2; }
[[ -f "$SCORES_FILE" ]] || { echo "缺少评分文件 $SCORES_FILE"; exit 2; }
[[ -f "$SCRIPT_DIR/create_meta_manifest_brennan_eeg.py" ]] || { echo "缺少 $SCRIPT_DIR/create_meta_manifest_brennan_eeg.py"; exit 2; }
[[ -f "$SCRIPT_DIR/create_local_global_splits.py" ]] || { echo "缺少 $SCRIPT_DIR/create_local_global_splits.py"; exit 2; }
[[ -f "$SCRIPT_DIR/report_candidate_pool.py" ]] || { echo "缺少 $SCRIPT_DIR/report_candidate_pool.py"; exit 2; }

mkdir -p "$OUT_DIR"

# ====== 从评分表自动生成 keep33.txt（只保留 use） ======
KEEP_TXT="$OUT_DIR/keep33.txt"
echo "== 生成 keep33.txt（只保留 use） =="
awk '
  BEGIN{IGNORECASE=1}
  /^S[0-9]+/ {
    sid=$1
    line=$0
    if (line ~ /exclude/) next
    if (line ~ /use/) { print sid }
  }
' "$SCORES_FILE" | sort -u > "$KEEP_TXT"

echo "保留被试列表："
cat "$KEEP_TXT"
KEEP_N=$(wc -l < "$KEEP_TXT" | tr -d ' ')
echo "数量: $KEEP_N"
if [ "$KEEP_N" -lt 1 ]; then
  echo "ERROR: 筛选后为空，请检查 $SCORES_FILE 的 use/exclude 标注"; exit 3;
fi

echo
echo "== 阶段1: 生成 EEG 蓝图 (Brennan, sentence-only) =="
# 只有当 meta 不存在时才生成；若你想强制重建，先 rm -f "$META"
if [ ! -f "$META" ]; then
  python "$SCRIPT_DIR/create_meta_manifest_brennan_eeg.py" \
    --brennan_root "$EEG_ROOT" \
    --output_dir "$OUT_DIR" \
    --keep_subjects_file "$KEEP_TXT"
fi

# 校验 meta 是否生成成功
if [ ! -f "$META" ]; then
  echo "ERROR: 没有生成 $META，停止。请检查上面的报错输出"; exit 4;
fi

# 简要统计（仅 sentence）
echo "== 蓝图统计（sentence-only） =="
# 统计 sentence
SENT_N=$(grep -E -c '"type"[[:space:]]*:[[:space:]]*"sentence"' "$META" || true)
echo "sentence 段: $SENT_N"

echo
echo "== 阶段2: 生成论文对齐的锚点窗口 splits（sentence-only） =="
SPLIT_DIR="$OUT_DIR/final_splits_sentence"
python "$SCRIPT_DIR/create_local_global_splits.py" \
  --meta_manifest_path "$META" \
  --output_dir "$OUT_DIR" \
  --bids_root_dir "$EEG_ROOT" \
  --split_ratios "0.7,0.1,0.2" \
  --random_seed 42

echo
echo "== 候选池统计: sentence =="
python "$SCRIPT_DIR/report_candidate_pool.py" --dir "$SPLIT_DIR"

echo
echo "全部完成 ✅ 输出根目录：$OUT_DIR"
