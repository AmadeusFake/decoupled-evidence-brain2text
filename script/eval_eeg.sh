#!/usr/bin/env bash
#
# ==============================================================================
# SLURM eval script: EEG retrieval baseline vs "GCB-only" (+ paired bootstrap CI)
#
# NEW (important):
#   The evaluation Python now supports:
#     --meg_encoder {dense,exp}
#   This script passes that flag to ensure reproducibility across backbones.
#
# Privacy note:
#   This template intentionally avoids real project IDs, emails, and personal paths.
#   Replace placeholders (e.g., <NAISS_PROJECT_ID>) with your own values.
# ==============================================================================

#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J Eval_EEG_GCB_only
#SBATCH -o <ABS_OR_REL_LOG_DIR>/%x_%j.out
#SBATCH -e <ABS_OR_REL_LOG_DIR>/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail

# ---------------- Paths you MUST configure ----------------
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"

# EEG test manifest (JSONL). Point this to your prepared test split.
EEG_MAN_DIR="$PROJECT_ROOT/EEGdata_manifests_local_global/final_splits_sentence_fully_preprocessed"
TEST_MAN="$EEG_MAN_DIR/test.jsonl"

# Entry points
EVAL_PY="$PROJECT_ROOT/eval/retrieval_window_vote.py"
BOOTSTRAP_PY="$PROJECT_ROOT/eval/paired_bootstrap_ci.py"

# Python interpreter (recommended: a dedicated venv/conda env with pinned deps)
PY="<PATH_TO_PYTHON>"  # e.g., "$PROJECT_ROOT/.venv/bin/python"

# Run directories to evaluate
EEG_RUNS_ROOT="$PROJECT_ROOT/runs_eeg"
EEG_RUN_DIRS=(
  "$EEG_RUNS_ROOT/<RUN_DIR_1>"
  "$EEG_RUNS_ROOT/<RUN_DIR_2>"
)

# ---------------- Environment setup (cluster-dependent) ----------------
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true
# source "<PATH_TO_VENV>/bin/activate"   # optional

cd "$PROJECT_ROOT"

# ---------------- Reproducibility / runtime knobs ----------------
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

export HF_HOME="$PROJECT_ROOT/hf_home"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
unset TQDM_DISABLE
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

# ---------------- Eval configuration ----------------
AMP=bf16
SAVE_TOPK=10
TOPK_LIST="1,5,10"
BOOTSTRAP_B=10000

# NEW: select MEG encoder backbone used inside eval (must match your code/ckpt expectation)
# Options: dense | exp
MEG_ENCODER="dense"

# Use checkpoint-provided logit scaling if supported
USE_CKPT_LOGIT_SCALE=1
SCALE_FLAG=""
[[ $USE_CKPT_LOGIT_SCALE -eq 1 ]] && SCALE_FLAG="--use_ckpt_logit_scale"

# ---------------- Per-run evaluation function ----------------
run_eval_config () {
  local run_dir="$1"

  local RR_DIR="$run_dir/results/retrieval_final_min"
  mkdir -p "$RR_DIR"

  echo "========================================"
  echo " Evaluating run        : $run_dir"
  echo " Results directory     : $RR_DIR"
  echo " meg_encoder backbone  : $MEG_ENCODER"
  echo "========================================"

  _run_eval () {
    local label="$1"; shift
    local extra_flags=("$@")

    echo "------------------------------"
    echo " Config label : ${label}"
    echo " Extra flags  : ${extra_flags[*]}"
    echo "------------------------------"

    "$PY" "$EVAL_PY" \
      --test_manifest "$TEST_MAN" \
      --run_dir "$run_dir" --use_best_ckpt \
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
      echo "[INFO] Archived: $base_tsv -> $labeled_tsv"
    else
      echo "[WARN] Missing expected output: $base_tsv (label=${label})"
    fi
  }

  # 1) Baseline: disable QCCP, disable window voting, disable GCB
  _run_eval baseline \
    --no_qccp \
    --no_windowvote \
    --no_gcb

  # 2) GCB-only: disable QCCP + window voting, enable GCB with chosen hyperparameters
  _run_eval gcb_only \
    --no_qccp \
    --no_windowvote \
    --gcb_topk 128 \
    --gcb_q 0.95 \
    --gcb_top_m 3 \
    --gcb_norm bucket_sqrt \
    --gcb_topS 3 \
    --gcb_gamma 0.7

  # 3) Paired bootstrap CI comparing baseline vs gcb_only
  echo "=============================="
  echo " Running paired bootstrap CI for:"
  echo "   $run_dir"
  echo "=============================="

  "$PY" "$BOOTSTRAP_PY" \
    --runs baseline="$RR_DIR" \
           gcb_only="$RR_DIR" \
    --baseline baseline \
    --B "$BOOTSTRAP_B" \
    --out_tex "$RR_DIR/bootstrap_eeg_baseline_vs_gcb.tex"

  echo "[DONE] Finished run: $run_dir"
  echo "[INFO] Bootstrap LaTeX table: $RR_DIR/bootstrap_eeg_baseline_vs_gcb.tex"
}

# ---------------- Main loop over run directories ----------------
for RUN_DIR in "${EEG_RUN_DIRS[@]}"; do
  if [[ -d "$RUN_DIR" ]]; then
    run_eval_config "$RUN_DIR"
  else
    echo "[WARN] Run directory not found (skipping): $RUN_DIR"
  fi
done

echo "[ALL DONE] Completed baseline vs GCB-only evaluations."
