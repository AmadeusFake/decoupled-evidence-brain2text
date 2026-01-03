#!/usr/bin/env bash
#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J simpleconv_meg_kfold_SLR
#SBATCH -o runs/%x_%A_%a.out
#SBATCH -e runs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --array=0-4         # 5-fold: fold0...fold4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -Eeuo pipefail

echo "======================================================"
echo "Job: ${SLURM_JOB_ID:-NA}  Host: $(hostname)  Fold: ${SLURM_ARRAY_TASK_ID}"
nvidia-smi || true
ulimit -n 4096 || true
echo "Start: $(date)"
echo "======================================================"

# ---------- Paths (configure these) ----------
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

SPLIT_ROOT="$PROJECT_ROOT/data_manifests_local_global/resplit_sentence_SLR_kfold"
FOLD="fold${SLURM_ARRAY_TASK_ID}"
TRAIN_MAN="$SPLIT_ROOT/$FOLD/train.jsonl"
VAL_MAN="$SPLIT_ROOT/$FOLD/valid.jsonl"
TEST_MAN="$SPLIT_ROOT/$FOLD/test.jsonl"

EVAL_SCRIPT="train/eval_retrieval.py"

# ---------- NEW: MEG encoder backbone ----------
# Must match your argparse choices: {dense, exp}
MEG_ENCODER="dense"

# ---------- Environment ----------
module load Python/3.11.3-GCCcore-12.3.0 || true
source "<PATH_TO_VENV>/bin/activate" || true

export TOKENIZERS_PARALLELISM=false
export HF_HOME="<PATH_TO_HF_CACHE>"
unset TRANSFORMERS_CACHE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NVIDIA_TF32_OVERRIDE=0

# ---------- Sanity checks ----------
echo "[CHECK] TRAIN_MAN = $TRAIN_MAN"
echo "[CHECK] VAL_MAN   = $VAL_MAN"
echo "[CHECK] TEST_MAN  = $TEST_MAN"
echo "[CHECK] meg_encoder = $MEG_ENCODER"
[[ -f "$TRAIN_MAN" && -f "$VAL_MAN" && -f "$TEST_MAN" ]] || { echo "[ERROR] manifest missing"; exit 2; }

python - <<'PY'
import torch
print("[DEBUG] torch", torch.__version__, "cuda.is_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[DEBUG] device =", torch.cuda.get_device_name(0))
PY

# ---------- Training args (SimpleConv) ----------
BSZ=256
ACC=1
EBS=$((BSZ*ACC*1))

BACKBONE_DEPTH=5
BACKBONE_KERNEL=3
DILATION_PERIOD=5
GLU_MULT=2
BACKBONE_DROPOUT=0.0

EXP="simpleconv_meg_baseline_SLR_${FOLD}_enc${MEG_ENCODER}_d${BACKBONE_DEPTH}_dp${DILATION_PERIOD}_EBS${EBS}_ampoff_tf32off_adam_wd0_e100"

echo "[INFO] Training EXP=${EXP}"

python -u -m train.train_meg \
  --train_manifest "$TRAIN_MAN" \
  --val_manifest   "$VAL_MAN" \
  --test_manifest  "$TEST_MAN" \
  \
  --meg_encoder "$MEG_ENCODER" \
  \
  --batch_size "$BSZ" \
  --accumulate_grad_batches "$ACC" \
  --gpus 1 \
  --num_workers 8 \
  --amp off \
  --optimizer adam \
  --lr 3e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.10 \
  --max_epochs 100 \
  --gradient_clip_val 1.0 \
  --early_stop_patience 10 \
  --base_bsz_for_lr 256 \
  \
  --in_channels 270 \
  --spatial_channels 270 \
  --fourier_k 32 \
  --d_model 320 \
  --out_channels 1024 \
  \
  --backbone_depth ${BACKBONE_DEPTH} \
  --backbone_kernel ${BACKBONE_KERNEL} \
  --dilation_period ${DILATION_PERIOD} \
  --glu_mult ${GLU_MULT} \
  --backbone_dropout ${BACKBONE_DROPOUT} \
  \
  --subject_layer_pos early \
  --spatial_dropout_p 0.0 \
  --spatial_dropout_radius 0.2 \
  \
  --seed 42 \
  --default_root_dir runs \
  --experiment_name "$EXP" \
  --metrics_every_n_steps 500

status=$?
echo "[INFO] Training finished with status=$status"

if [[ $status -ne 0 ]]; then
  echo "[ERROR] Training failed, skip eval."
  echo "End: $(date)"
  exit $status
fi

# ---------- Locate RUN_DIR for this fold ----------
RUN_DIR=$(ls -td "runs/${EXP}"_* 2>/dev/null | head -n 1 || true)

if [[ -z "$RUN_DIR" ]]; then
  echo "[ERROR] Cannot find run directory for EXP=${EXP}"
  echo "End: $(date)"
  exit 3
fi

echo "[INFO] Using RUN_DIR=${RUN_DIR} for eval"

# ---------- Helper: run eval once ----------
run_eval () {
  local label="$1"; shift
  python -u "$EVAL_SCRIPT" \
    --test_manifest "$TEST_MAN" \
    --train_manifest "$TRAIN_MAN" \
    --run_dir "$RUN_DIR" \
    --use_best_ckpt \
    --device cuda \
    --amp bf16 \
    --topk "1,5,10" \
    --strict_subjects \
    --sim clip \
    --tau 0.0 \
    --save_topk 0 \
    --config_label "$label" \
    --meg_encoder "$MEG_ENCODER" \
    "$@"
}

# ---------------- 4 configs: Window-Vote / GCB grid ----------------
# Keep QCCP disabled in all configs (--no_qccp)

# 1) baseline: everything off
run_eval baseline \
  --no_qccp \
  --no_windowvote \
  --no_gcb

# 2) vote_only: enable Window-Vote only
run_eval vote_only \
  --no_qccp \
  --no_gcb

# 3) gcb_only: enable GCB only
run_eval gcb_only \
  --no_qccp \
  --no_windowvote \
  --gcb_topk 128 \
  --gcb_q 0.95 \
  --gcb_top_m 3 \
  --gcb_norm bucket_sqrt \
  --gcb_topS 3 \
  --gcb_gamma 0.7

# 4) gcb_vote: enable both Window-Vote + GCB
run_eval gcb_vote \
  --no_qccp \
  --gcb_topk 128 \
  --gcb_q 0.95 \
  --gcb_top_m 3 \
  --gcb_norm bucket_sqrt \
  --gcb_topS 3 \
  --gcb_gamma 0.7

echo "End: $(date)"
exit 0
