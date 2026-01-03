#!/usr/bin/env bash
#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J Eval_kfold_SLR
#SBATCH -o runs/%x_%A_%a.out
#SBATCH -e runs/%x_%A_%a.err
#SBATCH --array=0-4%1              # 5 folds, serialized (adjust %N if you want parallel)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -Eeuo pipefail

# =========================
# Configurable paths
# =========================
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
RUN_BASE_ROOT="$PROJECT_ROOT/run_baseline"
MAN_ROOT="$PROJECT_ROOT/data_manifests_local_global/resplit_sentence_SLR_kfold"

PY="<PATH_TO_VENV>/bin/python"
EVAL_PY="$PROJECT_ROOT/eval/retrieval_window_vote.py"

# NEW: MEG encoder backbone (must match argparse choices)
# choices: dense | exp
MEG_ENCODER="dense"

# Fold id from Slurm array
FOLD_ID="${SLURM_ARRAY_TASK_ID:-0}"

# =========================
# Resolve run_dir for this fold
# =========================
RUN_DIR_TO_EVAL=$(
  ls -d "${RUN_BASE_ROOT}/none_baseline_SLR_fold${FOLD_ID}_EBS256_ampoff_tf32off_adam_wd0_e100_"*/ 2>/dev/null \
  | sort -r | head -n1 || true
)
if [[ -z "${RUN_DIR_TO_EVAL}" ]]; then
  echo "[ERROR] cannot locate run_dir for fold=${FOLD_ID} under ${RUN_BASE_ROOT}" >&2
  exit 2
fi
echo "[INFO] FOLD=${FOLD_ID}, RUN_DIR_TO_EVAL=${RUN_DIR_TO_EVAL}"

# Test manifest for this fold
TEST_MAN="${MAN_ROOT}/fold${FOLD_ID}/test.jsonl"
if [[ ! -f "${TEST_MAN}" ]]; then
  echo "[ERROR] TEST_MAN not found: ${TEST_MAN}" >&2
  exit 2
fi
echo "[INFO] Using TEST_MAN=${TEST_MAN}"
echo "[INFO] meg_encoder=${MEG_ENCODER}"

# =========================
# Environment
# =========================
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true
source "<PATH_TO_VENV>/bin/activate" || true

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="<PATH_TO_HF_CACHE>"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
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

# =========================
# Eval knobs
# =========================
AMP=bf16
TOPK_LIST="1,5,10"
SAVE_TOPK=10

COMMON_ARGS=(
  --test_manifest "$TEST_MAN"
  --run_dir "$RUN_DIR_TO_EVAL" --use_best_ckpt
  --device cuda
  --amp "$AMP"
  --topk "$TOPK_LIST"
  --save_topk "$SAVE_TOPK"
  --meg_encoder "$MEG_ENCODER"
  --use_ckpt_logit_scale
  --no_qccp
  --dump_per_query
  --dump_sentence_metrics
  --seed 42
)

# Keep your numeric params unchanged
GCB_ON_ARGS=( --gcb_topk 128 --gcb_q 0.95 --gcb_top_m 3 --gcb_norm bucket_sqrt --gcb_topS 3 --gcb_gamma 0.7 )
VOTE_ON_ARGS=( --topk_window 128 --q_quantile 0.95 --sent_top_m 3 --sent_topS 3 --sent_norm bucket_sqrt --gamma 0.7 )

# Small helper: copy if exists
copy_if() { [[ -f "$1" ]] && cp -f "$1" "$2" || echo "[WARN] missing: $1"; }

RES_DIR="$RUN_DIR_TO_EVAL/results/retrieval_final_min"
mkdir -p "$RES_DIR"

LABELS=("baseline" "vote_only" "gcb_only" "gcb_vote")

for LABEL in "${LABELS[@]}"; do
  EXTRA_ARGS=()
  case "$LABEL" in
    baseline)
      EXTRA_ARGS+=( --no_windowvote --no_gcb )
      ;;
    vote_only)
      EXTRA_ARGS+=( "${VOTE_ON_ARGS[@]}" --no_gcb )
      ;;
    gcb_only)
      EXTRA_ARGS+=( --no_windowvote "${GCB_ON_ARGS[@]}" )
      ;;
    gcb_vote)
      EXTRA_ARGS+=( "${VOTE_ON_ARGS[@]}" "${GCB_ON_ARGS[@]}" )
      ;;
    *)
      echo "[ERROR] Unknown LABEL $LABEL" >&2
      exit 1
      ;;
  esac

  echo "[INFO] Start fold=${FOLD_ID}, label=${LABEL}"
  "$PY" "$EVAL_PY" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
  echo "[INFO] Done fold=${FOLD_ID}, label=${LABEL}"

  # Save a labeled snapshot immediately (avoid being overwritten by the next config)
  STAMP="$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID}_${LABEL}"
  OUT_DIR="$RES_DIR/${STAMP}"
  mkdir -p "$OUT_DIR"

  copy_if "$RES_DIR/metrics.json"                 "$OUT_DIR/metrics_${LABEL}.json"
  copy_if "$RES_DIR/ranks.txt"                    "$OUT_DIR/ranks_${LABEL}.txt"
  copy_if "$RES_DIR/per_query.tsv"                "$OUT_DIR/per_query_${LABEL}.tsv"
  copy_if "$RES_DIR/sentence_metrics.tsv"         "$OUT_DIR/sentence_metrics_${LABEL}.tsv"
  copy_if "$RES_DIR/preds_topk${SAVE_TOPK}.tsv"   "$OUT_DIR/preds_topk${SAVE_TOPK}_${LABEL}.tsv"
  copy_if "$RES_DIR/preds_topk${SAVE_TOPK}.jsonl" "$OUT_DIR/preds_topk${SAVE_TOPK}_${LABEL}.jsonl"

  # Also drop a flat-named copy into RES_DIR for k-fold bootstrap consumption
  copy_if "$OUT_DIR/metrics_${LABEL}.json"         "$RES_DIR/metrics_${LABEL}.json"
  copy_if "$OUT_DIR/sentence_metrics_${LABEL}.tsv" "$RES_DIR/sentence_metrics_${LABEL}.tsv"

  echo "[INFO] Saved labeled outputs for fold=${FOLD_ID}, label=${LABEL}"
done

echo "[INFO] All done for fold=${FOLD_ID}"
