#!/usr/bin/env bash
#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J Eval_WV_plus_GCB_SLR_fold0
#SBATCH -o runs/%x_%j.out
#SBATCH -e runs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail
set -x

# --- Paths / files (privacy-safe placeholders) ---
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
FOLD=0
MANI_DIR="$PROJECT_ROOT/data_manifests_local_global/resplit_sentence_SLR_kfold/fold${FOLD}"
TEST_MANI="$MANI_DIR/test.jsonl"

# Run dir to evaluate (must contain checkpoints + records/)
RUN_DIR_TO_EVAL="$PROJECT_ROOT/<PATH_TO_RUN_DIR>"

# Minimal evaluator that implements WV -> GCB ordering
EVAL_PY="$PROJECT_ROOT/eval/retrieval_window_vote.py"
PY="<PATH_TO_VENV>/bin/python"

# --- NEW: MEG encoder backbone (must match argparse choices) ---
# choices: dense | exp
MEG_ENCODER="dense"

# --- Modules / env ---
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true
source "<PATH_TO_VENV>/bin/activate" || true

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Optional HF cache (keep downloads inside project; change if you prefer)
export HF_HOME="$PROJECT_ROOT/hf_home"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false

# Allow TF32 during eval for speed (optional)
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.9"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
ulimit -n 4096 || true

# --- Sanity checks ---
[[ -f "$TEST_MANI" ]] || { echo "[ERROR] test.jsonl missing: $TEST_MANI"; exit 2; }
[[ -d "$RUN_DIR_TO_EVAL" ]] || { echo "[ERROR] run_dir missing: $RUN_DIR_TO_EVAL"; exit 2; }
[[ -f "$EVAL_PY" ]] || { echo "[ERROR] eval script missing: $EVAL_PY"; exit 2; }
[[ -f "$RUN_DIR_TO_EVAL/records/subject_mapping.json" ]] || { echo "[ERROR] subject_mapping.json missing under $RUN_DIR_TO_EVAL/records"; exit 2; }

# --- Eval knobs ---
AMP=bf16
TOPK_LIST="1,5,10"
SAVE_TOPK=10
USE_CKPT_LOGIT_SCALE=1

SCALE_FLAG=()
[[ "$USE_CKPT_LOGIT_SCALE" -eq 1 ]] && SCALE_FLAG+=(--use_ckpt_logit_scale)

# --- Joint config: Window-Vote + GCB (QCCP off) ---
# Window-Vote: (scan best family: across=sum, mean@topm, bucket_sqrt)
WV_FLAGS=(
  --topk_window 256
  --q_quantile 0.95
  --sent_top_m 3
  --sent_topS 3
  --sent_norm bucket_sqrt
  --gamma 0.7
)

# GCB: post soft-consensus + across=sum
GCB_FLAGS=(
  --gcb_topk 128
  --gcb_q 0.95
  --gcb_top_m 3
  --gcb_norm bucket_sqrt
  --gcb_topS 3
  --gcb_gamma 0.7
)

"$PY" -u "$EVAL_PY" \
  --test_manifest "$TEST_MANI" \
  --run_dir "$RUN_DIR_TO_EVAL" --use_best_ckpt \
  --device cuda \
  --amp "$AMP" \
  --topk "$TOPK_LIST" \
  --save_topk "$SAVE_TOPK" \
  --meg_encoder "$MEG_ENCODER" \
  "${SCALE_FLAG[@]}" \
  --no_qccp \
  "${WV_FLAGS[@]}" \
  "${GCB_FLAGS[@]}"

echo "[DONE] metrics at:  $RUN_DIR_TO_EVAL/results/retrieval_final_min/metrics.json"
echo "       ranks at:    $RUN_DIR_TO_EVAL/results/retrieval_final_min/ranks.txt"
