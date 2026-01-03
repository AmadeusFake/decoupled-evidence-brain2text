#!/usr/bin/env bash
#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J Eval_VOTE_GCB_Grid
#SBATCH -o runs/%x_%j.out
#SBATCH -e runs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -Eeuo pipefail

# ============================================================
# 0) User-configurable paths (privacy-safe placeholders)
# ============================================================
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Test manifest MUST match the training setup (dataset + split protocol)
MAN_DIR="$PROJECT_ROOT/<PATH_TO_MANIFEST_DIR>"
TEST_MAN="$MAN_DIR/test.jsonl"

# The trained run directory you want to evaluate (must contain ckpt records / best checkpoint info)
RUN_DIR_TO_EVAL="$PROJECT_ROOT/<PATH_TO_RUN_DIR>"

EVAL_PY="$PROJECT_ROOT/eval/retrieval_window_vote.py"
BOOTSTRAP_PY="$PROJECT_ROOT/eval/paired_bootstrap_ci.py"
PY="<PATH_TO_PYTHON>"  # e.g. "$PROJECT_ROOT/.venv/bin/python"

# Output directory inside the run dir (keeps results self-contained)
RR_DIR="$RUN_DIR_TO_EVAL/results/retrieval_final_min"

# ============================================================
# 1) NEW: MEG encoder backbone (must match Python argparse choices)
# ============================================================
# choices: dense | exp
MEG_ENCODER="dense"

# ============================================================
# 2) Environment (evaluation-optimized; TF32/bf16 allowed)
# ============================================================
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true

# Optional: activate env (placeholder)
# source "<PATH_TO_VENV>/bin/activate" || true

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore

# Optional HF cache (placeholder)
export HF_HOME="<PATH_TO_HF_CACHE>"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"

# Allow TF32 for speed during eval (optional)
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

# ============================================================
# 3) Eval knobs
# ============================================================
AMP=bf16
SAVE_TOPK=10
TOPK_LIST="1,5,10"
BOOTSTRAP_B=10000

# Use checkpoint-provided logit scaling (recommended if your ckpt stores it)
USE_CKPT_LOGIT_SCALE=1
SCALE_FLAG=""
[[ $USE_CKPT_LOGIT_SCALE -eq 1 ]] && SCALE_FLAG="--use_ckpt_logit_scale"

mkdir -p "$RR_DIR"

echo "[INFO] PROJECT_ROOT   = $PROJECT_ROOT"
echo "[INFO] TEST_MAN       = $TEST_MAN"
echo "[INFO] RUN_DIR_TO_EVAL = $RUN_DIR_TO_EVAL"
echo "[INFO] RR_DIR         = $RR_DIR"
echo "[INFO] meg_encoder    = $MEG_ENCODER"

# ============================================================
# 4) Helper: run one config + archive sentence_metrics.tsv
# ============================================================
run_eval () {
  local label="$1"; shift
  local extra_flags=("$@")

  echo "=================================================="
  echo " Running config: ${label}"
  echo " Extra flags   : ${extra_flags[*]}"
  echo "=================================================="

  "$PY" "$EVAL_PY" \
    --test_manifest "$TEST_MAN" \
    --run_dir "$RUN_DIR_TO_EVAL" --use_best_ckpt \
    --device cuda \
    --amp "$AMP" \
    --topk "$TOPK_LIST" \
    --save_topk "$SAVE_TOPK" \
    --meg_encoder "$MEG_ENCODER" \
    $SCALE_FLAG \
    --dump_sentence_metrics \
    --dump_per_query \
    --seed 42 \
    "${extra_flags[@]}"

  local base_tsv="$RR_DIR/sentence_metrics.tsv"
  local labeled_tsv="$RR_DIR/sentence_metrics_${label}.tsv"

  if [[ -f "$base_tsv" ]]; then
    cp "$base_tsv" "$labeled_tsv"
    echo "[INFO] Copied $base_tsv -> $labeled_tsv"
  else
    echo "[WARN] sentence_metrics.tsv not found for label=${label}"
  fi
}

# ============================================================
# 5) 4 configs: Baseline / Vote / GCB / Vote+GCB
#    Always disable QCCP for apples-to-apples comparison
# ============================================================

# 1) baseline: all off
run_eval baseline \
  --no_qccp \
  --no_windowvote \
  --no_gcb

# 2) vote_only: Window-Vote only
run_eval vote_only \
  --no_qccp \
  --no_gcb

# 3) gcb_only: GCB only
run_eval gcb_only \
  --no_qccp \
  --no_windowvote \
  --gcb_topk 128 \
  --gcb_q 0.95 \
  --gcb_top_m 3 \
  --gcb_norm bucket_sqrt \
  --gcb_topS 3 \
  --gcb_gamma 0.7

# 4) gcb_vote: Window-Vote + GCB
run_eval gcb_vote \
  --no_qccp \
  --gcb_topk 128 \
  --gcb_q 0.95 \
  --gcb_top_m 3 \
  --gcb_norm bucket_sqrt \
  --gcb_topS 3 \
  --gcb_gamma 0.7

# ============================================================
# 6) Paired bootstrap CI (sentence-level): deltas + CI + p-values
# ============================================================
echo "=================================================="
echo " Running paired bootstrap (sentence-level)"
echo "=================================================="

"$PY" "$BOOTSTRAP_PY" \
  --runs baseline="$RR_DIR" \
         vote_only="$RR_DIR" \
         gcb_only="$RR_DIR" \
         gcb_vote="$RR_DIR" \
  --baseline baseline \
  --B "$BOOTSTRAP_B" \
  --out_tex "$RR_DIR/bootstrap_vote_gcb_sentence.tex"

echo "[DONE] All configs evaluated."
echo "[INFO] Bootstrap LaTeX table => $RR_DIR/bootstrap_vote_gcb_sentence.tex"
