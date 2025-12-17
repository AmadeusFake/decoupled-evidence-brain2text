#!/usr/bin/env bash
set -euo pipefail

# === 路径（按需修改） ===
PROJECT_BASE="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
BIDS_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/MOUS_raw"            # MOUS BIDS 根目录
SCRIPT_DIR="$PROJECT_BASE/data_script_MOUS"                                 # 放这几个脚本的目录
OUT_DIR="$PROJECT_BASE/data_mous_local_global"

ALIGN_JSONL="$OUT_DIR/mous_whisper_word_alignments.jsonl"
META="$OUT_DIR/meta_manifest_mous.jsonl"
ALLOWLIST="$OUT_DIR/mous_subject_allowlist.txt"

mkdir -p "$OUT_DIR"

# ---------------- Step 0: 按 BrainMagick 规则生成 96 人 allowlist ----------------
# 规则：A2002–A2125 的 auditory subjects，排除 BrainMagick 标记的 bad 录音
echo "== Step0: 生成 96 auditory subjects allowlist（BrainMagick 对齐） =="

if [[ ! -f "$ALLOWLIST" ]]; then
  # 来自 Schoffelen2019Recording.iter() 中被 skip 掉的编号（>=2000）
  bad_nums=(
    2011 2012 2018 2022 2023 2026 2036
    2043 2044 2045 2048
    2054 2060 2062 2063
    2074 2076
    2081 2082 2084 2087
    2093 2100 2107
    2112 2115 2118 2123
  )

  : > "$ALLOWLIST"
  for num in $(seq 2002 2125); do
    skip=0
    for bad in "${bad_nums[@]}"; do
      if [[ "$num" -eq "$bad" ]]; then
        skip=1
        break
      fi
    done
    [[ "$skip" -eq 1 ]] && continue
    # 注意：这里写成 A####，不要加 sub-，和 meta 里的 subject_id 对齐
    printf "A%04d\n" "$num" >> "$ALLOWLIST"
  done
  echo "[INFO] 写入 allowlist: $ALLOWLIST"
else
  echo "[INFO] 已存在 allowlist: $ALLOWLIST"
fi

echo "[INFO] allowlist 前几行："
head "$ALLOWLIST" || true
echo "[INFO] allowlist 总数 = $(wc -l < "$ALLOWLIST" | tr -d ' ')"

# ---------------- Stage 0: Whisper 对齐 ----------------
echo
echo "== 阶段0: Whisper 词级对齐（MOUS） =="

if [ ! -f "$ALIGN_JSONL" ]; then
  python "$SCRIPT_DIR/mous_whisper_align.py" \
    --mous_root "$BIDS_DIR" \
    --out_dir "$OUT_DIR" \
    --model large-v2 \
    --language nl
else
  echo "已有对齐结果：$ALIGN_JSONL"
fi

# ---------------- Stage 1: 蓝图（meta manifest） ----------------
echo
echo "== 阶段1: 蓝图（MOUS, Whisper 对齐 + 96-subject allowlist） =="

# 使用你已经改好的 create_meta_manifest_mous.py，带上 --subject_allowlist
# 注意：这里不再事后过滤 meta，而是从源头就只保留 96 人
python "$SCRIPT_DIR/create_meta_manifest_mous.py" \
  --mous_root "$BIDS_DIR" \
  --align_jsonl "$ALIGN_JSONL" \
  --out_dir "$OUT_DIR" \
  --subject_allowlist "$ALLOWLIST"

echo "META 写入完成：$META"
echo "meta 中的 subject 数量："
python - <<PY
import json
from collections import Counter
from pathlib import Path

meta = Path("$META")
subs = Counter()
with meta.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sid = r.get("subject_id") or r.get("subject")
        subs[sid] += 1
print("subjects:", len(subs))
for sid, n in sorted(subs.items(), key=lambda x: -x[1])[:10]:
    print(f"  {sid}: {n}")
PY

# ---------------- Stage 2: 3s 锚点窗口 + split ----------------
echo
echo "== 阶段2: 生成论文对齐的锚点窗口 splits（MOUS, 仅 96 subjects） =="

for T in sentence; do
  SPLIT_DIR="$OUT_DIR/final_splits_${T}"

  # 使用你已经改好的 create_local_global_splits.py，带上 --subject_allowlist
  python "$SCRIPT_DIR/create_local_global_splits.py" \
    --meta_manifest_path "$META" \
    --output_dir "$OUT_DIR" \
    --bids_root_dir "$BIDS_DIR" \
    --data_type "$T" \
    --split_ratios "0.7,0.1,0.2" \
    --random_seed 42 \
    --disable_cross_overlap_pruning \
    --subject_allowlist "$ALLOWLIST"

  echo
  echo "== 候选池统计 (MOUS, $T)（已是 96 人版） =="
  python "$SCRIPT_DIR/report_candidate_pool.py" --dir "$SPLIT_DIR"

  echo
  echo "== split 中 subject 数量检查 (MOUS, $T) =="
  python - <<PY
import json
from pathlib import Path
from collections import Counter

base = Path("$SPLIT_DIR")
subs = Counter()
for split in ("train", "valid", "test"):
    p = base / f"{split}.jsonl"
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("subject_id") or r.get("subject")
            subs[sid] += 1
print("subjects in train+valid+test:", len(subs))
for sid, n in sorted(subs.items(), key=lambda x: -x[1])[:10]:
    print(f"  {sid}: {n}")
PY

done

echo
echo "MOUS Stage-1 全部完成 ✅（从 Stage1 起就限制为 96 auditory subjects）"
