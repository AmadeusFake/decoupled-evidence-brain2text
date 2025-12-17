#!/usr/bin/env bash
set -euo pipefail

# === 路径（按需修改） ===
PROJECT_BASE="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
BIDS_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/download"
SCRIPT_DIR="$PROJECT_BASE/data_script"
OUT_DIR="$PROJECT_BASE/data_manifests_local_global"

META="$OUT_DIR/meta_manifest.jsonl"

echo "== 阶段1: 蓝图 =="
if [ ! -f "$META" ]; then
  python "$SCRIPT_DIR/create_meta_manifest.py" \
    --input_dir "$BIDS_DIR" \
    --output_dir "$OUT_DIR" \
    --num_workers 32
else
  echo "已有蓝图：$META"
fi

echo
echo "== 阶段2: 生成论文对齐的锚点窗口 splits =="
for T in sentence word_list; do
  SPLIT_DIR="$OUT_DIR/final_splits_$T"
  python "$SCRIPT_DIR/create_local_global_splits.py" \
    --meta_manifest_path "$META" \
    --output_dir "$OUT_DIR" \
    --bids_root_dir "$BIDS_DIR" \
    --data_type "$T" \
    --split_ratios "0.7,0.1,0.2" \
    --random_seed 42

  echo
  echo "== 候选池统计: $T =="
  python "$SCRIPT_DIR/report_candidate_pool.py" --dir "$SPLIT_DIR"
done

echo
echo "全部完成 ✅"

