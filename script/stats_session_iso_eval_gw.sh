#!/usr/bin/env bash
#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J Eval_QCCP_OFF_Array
#SBATCH -o runs/%x_%A_%a.out
#SBATCH -e runs/%x_%A_%a.err
#SBATCH --array=0-3%1               # serialized to avoid overwriting shared output files
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>
# If supported by your cluster, you may enable stricter CPU/GPU binding:
# #SBATCH --gres-flags=enforce-binding

set -euo pipefail

# =========================
# Paths (configure these)
# =========================
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
MAN_DIR="$PROJECT_ROOT/data_manifests_local_global/final_splits_sentence_with_sentence_full"
TEST_MAN="$MAN_DIR/test.jsonl"

# The run dir to evaluate (must contain checkpoints + records/)
RUN_DIR_TO_EVAL="$PROJECT_ROOT/<PATH_TO_RUN_DIR>"

PY="<PATH_TO_VENV>/bin/python"
VENV_ACTIVATE="<PATH_TO_VENV>/bin/activate"
EVAL_PY="$PROJECT_ROOT/eval/retrieval_window_vote.py"

# NEW: MEG encoder backbone (must match argparse choices)
# choices: dense | exp
MEG_ENCODER="dense"

# =========================
# Environment
# =========================
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true
source "$VENV_ACTIVATE" || true

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="<PATH_TO_HF_CACHE>"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
unset TQDM_DISABLE
export PYTHONWARNINGS=ignore

# allow TF32 for eval speed (optional)
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
# Sanity checks
# =========================
[[ -f "$TEST_MAN" ]] || { echo "[ERROR] test.jsonl missing: $TEST_MAN"; exit 2; }
[[ -d "$RUN_DIR_TO_EVAL" ]] || { echo "[ERROR] run_dir missing: $RUN_DIR_TO_EVAL"; exit 2; }
[[ -f "$EVAL_PY" ]] || { echo "[ERROR] eval script missing: $EVAL_PY"; exit 2; }
[[ -f "$RUN_DIR_TO_EVAL/records/subject_mapping.json" ]] || { echo "[ERROR] subject_mapping.json missing under $RUN_DIR_TO_EVAL/records"; exit 2; }

echo "[INFO] PROJECT_ROOT   = $PROJECT_ROOT"
echo "[INFO] TEST_MAN       = $TEST_MAN"
echo "[INFO] RUN_DIR_TO_EVAL = $RUN_DIR_TO_EVAL"
echo "[INFO] meg_encoder    = $MEG_ENCODER"

# =========================
# Eval knobs
# =========================
AMP=bf16
TOPK_LIST="1,5,10"
SAVE_TOPK=10
USE_CKPT_LOGIT_SCALE=1

SCALE_FLAG=()
[[ "$USE_CKPT_LOGIT_SCALE" -eq 1 ]] && SCALE_FLAG+=(--use_ckpt_logit_scale)

CASE_IDX="${SLURM_ARRAY_TASK_ID:-0}"
LABELS=("baseline" "vote_only" "gcb_only" "gcb_vote")
LABEL="${LABELS[$CASE_IDX]}"

COMMON_ARGS=(
  --test_manifest "$TEST_MAN"
  --run_dir "$RUN_DIR_TO_EVAL" --use_best_ckpt
  --device cuda
  --amp "$AMP" --topk "$TOPK_LIST" --save_topk "$SAVE_TOPK"
  --meg_encoder "$MEG_ENCODER"
  "${SCALE_FLAG[@]}"
  --no_qccp                 # fixed OFF as requested
  --dump_per_query
  --dump_sentence_metrics
)

# Keep numeric values unchanged; only control on/off
GCB_ON_ARGS=( --gcb_topk 128 --gcb_q 0.95 --gcb_top_m 3 --gcb_norm bucket_sqrt --gcb_topS 3 --gcb_gamma 0.7 )
VOTE_ON_ARGS=( --topk_window 128 --q_quantile 0.95 --sent_top_m 3 --sent_topS 3 --sent_norm bucket_sqrt --gamma 0.7 )

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
    echo "Unknown LABEL $LABEL"; exit 1;;
esac

# =========================
# GPU anti-idle kill: reserve memory + heartbeat
# =========================
HEARTBEAT_PID=""
SMI_MON_PID=""

cleanup() {
  if [[ -n "${SMI_MON_PID}" ]] && kill -0 "${SMI_MON_PID}" 2>/dev/null; then
    kill "${SMI_MON_PID}" || true
    wait "${SMI_MON_PID}" 2>/dev/null || true
  fi
  if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
    kill "${HEARTBEAT_PID}" || true
    wait "${HEARTBEAT_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PY" -u - <<'PY' &
import time, torch
try:
    dev = "cuda"
    torch.cuda.init()
    # Reserve ~0.5GB: 4 * (8192*8192 fp16) ≈ 512 MiB
    hold = [torch.empty((8192,8192), dtype=torch.float16, device=dev) for _ in range(4)]
    a = torch.randn(2048, 2048, device=dev)
    while True:
        _ = a @ a
        torch.cuda.synchronize()
        time.sleep(5.0)
except KeyboardInterrupt:
    pass
PY
HEARTBEAT_PID=$!

# Optional GPU monitoring
nvidia-smi --query-gpu=name,uuid,utilization.gpu,memory.used --format=csv -l 60 &
SMI_MON_PID=$!

# =========================
# Run eval
# =========================
echo "[INFO] Start case=$LABEL"
"$PY" "$EVAL_PY" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
echo "[INFO] Done case=$LABEL"

# =========================
# Archive results immediately (avoid overwrite by next array element)
# =========================
RES_DIR="$RUN_DIR_TO_EVAL/results/retrieval_final_min"
STAMP="$(date +%Y%m%d-%H%M%S)_${SLURM_JOB_ID}_${LABEL}"
OUT_DIR="$RES_DIR/${STAMP}"
mkdir -p "$OUT_DIR"

copy_if() { [[ -f "$1" ]] && cp -f "$1" "$2" || echo "[WARN] missing: $1"; }

copy_if "$RES_DIR/metrics.json"                 "$OUT_DIR/metrics_${LABEL}.json"
copy_if "$RES_DIR/ranks.txt"                    "$OUT_DIR/ranks_${LABEL}.txt"
copy_if "$RES_DIR/per_query.tsv"                "$OUT_DIR/per_query_${LABEL}.tsv"
copy_if "$RES_DIR/sentence_metrics.tsv"         "$OUT_DIR/sentence_metrics_${LABEL}.tsv"
copy_if "$RES_DIR/preds_topk${SAVE_TOPK}.tsv"   "$OUT_DIR/preds_topk${SAVE_TOPK}_${LABEL}.tsv"
copy_if "$RES_DIR/preds_topk${SAVE_TOPK}.jsonl" "$OUT_DIR/preds_topk${SAVE_TOPK}_${LABEL}.jsonl"

# Flat copies for downstream bootstrap scripts
copy_if "$OUT_DIR/metrics_${LABEL}.json"            "$RES_DIR/metrics_${LABEL}.json"
copy_if "$OUT_DIR/per_query_${LABEL}.tsv"           "$RES_DIR/per_query_${LABEL}.tsv"
copy_if "$OUT_DIR/sentence_metrics_${LABEL}.tsv"    "$RES_DIR/sentence_metrics_${LABEL}.tsv"

echo "[INFO] Labeled outputs copied to:"
ls -lh "$OUT_DIR" || true
ls -lh "$RES_DIR"/{sentence_metrics,per_query,metrics}_*.{tsv,json} 2>/dev/null || true

echo "[INFO] All done for case=$LABEL"
