#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Root directory containing existing final_splits
DATA_ROOT="${PROJECT_ROOT}/data_manifests_local_global"

# Directory containing GW resplitting utilities
SCRIPT_DIR="${PROJECT_ROOT}/data_script_gw"

###############################################################################
# Input manifests (do NOT rename directories)
###############################################################################

# Sentence-level manifests (with full-audio attention)
IN_SENT="${DATA_ROOT}/final_splits_sentence_with_sentence_full_audio_attn"

# Word-list manifests (with full-audio attention)
IN_WORD="${DATA_ROOT}/final_splits_word_list_with_sentence_full_audio_attn"

###############################################################################
# Output roots (one subdirectory per fold)
###############################################################################

OUT_SENT_ROOT="${DATA_ROOT}/resplit_sentence_SLR_kfold"
OUT_WORD_ROOT="${DATA_ROOT}/resplit_word_list_SLR_kfold"

###############################################################################
# Resplitting configuration
###############################################################################

# Train / valid / test ratios (by recording count)
RATIOS="0.7,0.1,0.2"

# Number of folds for rotation
# K=3 guarantees full coverage for 22 evaluable subjects
# K=5 provides finer averaging
K=5

# Base random seed; fold i uses (SEED0 + i)
SEED0=42

# Minimum recordings per split (-1 disables the constraint)
MIN_VALID=-1
MIN_TEST=-1

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

PY_RESPLIT="${SCRIPT_DIR}/resplit_from_existing_manifest.py"

###############################################################################
# Helper: run k-fold rotation for one manifest family
###############################################################################

run_family () {
  local NAME="$1"
  local IN_DIR="$2"
  local OUT_ROOT="$3"

  echo "== Building k-fold rotation for ${NAME} ==> ${OUT_ROOT} =="

  for ((i=0; i<${K}; i++)); do
    OUT_DIR="${OUT_ROOT}/fold${i}"
    mkdir -p "${OUT_DIR}"

    seed=$((SEED0 + i))
    echo "[${NAME}] fold ${i}/${K} | seed=${seed} -> ${OUT_DIR}"

    python "${PY_RESPLIT}" \
      --input_manifest_dir "${IN_DIR}" \
      --output_dir        "${OUT_DIR}" \
      --split_ratios "${RATIOS}" \
      --random_seed "${seed}" \
      --min_valid_recordings "${MIN_VALID}" \
      --min_test_recordings  "${MIN_TEST}" \
      --require_subject_in_train \
      --enforce_covered_test_only \
      --rotation_num_folds "${K}" \
      --rotation_fold_index "${i}"
  done

  ###########################################################################
  # Fold summary: test-subject coverage
  ###########################################################################

  ROOT_OUT="${OUT_ROOT}" python - <<'PY'
import os, json, glob

root = os.environ["ROOT_OUT"]
folds = sorted(glob.glob(os.path.join(root, "fold*")))

def subjects_in_test(path):
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                s.add(str(d["subject_id"]))
    return s

union = set()
stats = []

for fd in folds:
    test_path = os.path.join(fd, "test.jsonl")
    S = subjects_in_test(test_path)
    union |= S
    stats.append((os.path.basename(fd), len(S)))

print(f"[{root}] folds={len(folds)} | union_test_subjects={len(union)}")
for name, count in stats:
    print(f"  {name}: test_subjects={count}")
PY
}

###############################################################################
# Run resplitting for sentence and word-list families
###############################################################################

run_family "sentence"   "${IN_SENT}" "${OUT_SENT_ROOT}"
run_family "word_list"  "${IN_WORD}" "${OUT_WORD_ROOT}"

###############################################################################
# Final report
###############################################################################

echo
echo "k-fold rotation resplitting completed successfully ✅"
echo "Sentence-level folds:"
echo "- ${OUT_SENT_ROOT}/fold*/{train,valid,test}.jsonl"
echo "Word-list folds:"
echo "- ${OUT_WORD_ROOT}/fold*/{train,valid,test}.jsonl"
