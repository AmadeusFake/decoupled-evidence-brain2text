#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT="$PROJECT_ROOT/data_manifests_local_global"
SCRIPT_DIR="$PROJECT_ROOT/data_script_gw"

PY_RESPLIT="$SCRIPT_DIR/resplit_from_existing_manifest.py"
# 输入（原 final_splits）
IN_SENT="$DATA_ROOT/final_splits_sentence_with_sentence_full_audio_attn"
IN_WORD="$DATA_ROOT/final_splits_word_list_with_sentence_full_audio_attn"

# 输出：每折一个子目录
OUT_SENT_ROOT="$DATA_ROOT/resplit_sentence_SLR_kfold"
OUT_WORD_ROOT="$DATA_ROOT/resplit_word_list_SLR_kfold"

# 比例（按“录制数”）：与现用一致
RATIOS="0.7,0.1,0.2"

# 折数：推荐 K=3 能保证 22 个可测被试全部覆盖；也可用 K=5（更细的平均化）
K=5
SEED0=42   # 第 0 折的种子，其余折会用 SEED0+i

MIN_VALID=-1
MIN_TEST=-1

run_family () {
  local NAME="$1" IN_DIR="$2" OUT_ROOT="$3"
  echo "== build k-fold (rotation) for $NAME ==> $OUT_ROOT =="

  for ((i=0;i<${K};i++)); do
    OUT_DIR="${OUT_ROOT}/fold${i}"
    mkdir -p "$OUT_DIR"
    seed=$((SEED0+i))
    echo "[$NAME] fold=${i}/${K} seed=${seed} -> $OUT_DIR"
    python "$PY_RESPLIT" \
      --input_manifest_dir "$IN_DIR" \
      --output_dir        "$OUT_DIR" \
      --split_ratios "$RATIOS" \
      --random_seed $seed \
      --min_valid_recordings $MIN_VALID \
      --min_test_recordings  $MIN_TEST \
      --require_subject_in_train \
      --enforce_covered_test_only \
      --rotation_num_folds $K \
      --rotation_fold_index $i
  done

  # 汇总每折 test 覆盖到的被试数
  python - <<'PY'
import os, json, glob
root = os.environ["ROOT_OUT"]
folds = sorted(glob.glob(os.path.join(root,"fold*")))
def subjset(path):
    s=set()
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d=json.loads(line)
                s.add(str(d["subject_id"]))
    return s
u=set(); stats=[]
for fd in folds:
    S=subjset(os.path.join(fd,"test.jsonl"))
    u|=S; stats.append((os.path.basename(fd), len(S)))
print(f"[{root}] folds={len(folds)}  union_test_subjects={len(u)}")
for n,c in stats: print(f"  {n}: test_subjects={c}")
PY
}

ROOT_OUT="$OUT_SENT_ROOT" run_family "sentence"   "$IN_SENT" "$OUT_SENT_ROOT"
ROOT_OUT="$OUT_WORD_ROOT" run_family "word_list" "$IN_WORD" "$OUT_WORD_ROOT"

echo "✅ k-fold (rotation) splits done:"
echo "- $OUT_SENT_ROOT/fold*/{train,valid,test}.jsonl"
echo "- $OUT_WORD_ROOT/fold*/{train,valid,test}.jsonl"
