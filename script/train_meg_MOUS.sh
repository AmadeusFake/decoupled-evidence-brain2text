#!/usr/bin/env bash
#
# ==============================================================================
# SLURM training script (MOUS): SimpleConv baseline
#
# NEW:
#   Pass --meg_encoder {dense,exp} to match the new argparse option in Python.
#
# Privacy note:
#   Replace placeholders (<...>) with your own values (project ID, paths, email).
# ==============================================================================

#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J mous_simpleconv_baseline
#SBATCH -o <LOG_DIR>/runs/%x_%j.out
#SBATCH -e <LOG_DIR>/runs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A40:1
#SBATCH --cpus-per-task=16
#SBATCH -t 24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -Eeuo pipefail

echo "======================================================"
echo "Job: ${SLURM_JOB_ID:-NA}  Host: $(hostname)"
nvidia-smi || true
ulimit -n 4096 || true
echo "Start: $(date)"
echo "======================================================"

# ---------------- Paths you MUST configure ----------------
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ---------------- MOUS manifests (fully_preprocessed) ----------------
TRAIN_MAN="$PROJECT_ROOT/data_mous_local_global/final_splits_sentence_fully_preprocessed/train.jsonl"
VAL_MAN="$PROJECT_ROOT/data_mous_local_global/final_splits_sentence_fully_preprocessed/valid.jsonl"
TEST_MAN="$PROJECT_ROOT/data_mous_local_global/final_splits_sentence_fully_preprocessed/test.jsonl"

# ---------------- Environment (cluster-dependent) ----------------
module load Python/3.11.3-GCCcore-12.3.0 || true
source "<PATH_TO_VENV>/bin/activate" || true

export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
unset TRANSFORMERS_CACHE

# Threads and TF32:
# Keep TF32 disabled for strict FP32 alignment (as your baseline intended)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NVIDIA_TF32_OVERRIDE=0

# ---------------- Sanity checks ----------------
echo "[CHECK] TRAIN_MAN = $TRAIN_MAN"
echo "[CHECK] VAL_MAN   = $VAL_MAN"
echo "[CHECK] TEST_MAN  = $TEST_MAN"
[[ -f "$TRAIN_MAN" ]] || { echo "[ERROR] train manifest not found: $TRAIN_MAN"; exit 2; }
[[ -f "$VAL_MAN"   ]] || { echo "[ERROR] val manifest not found:   $VAL_MAN";   exit 2; }
[[ -f "$TEST_MAN"  ]] || { echo "[ERROR] test manifest not found:  $TEST_MAN";  exit 2; }

python - <<'PY'
import torch
print("[DEBUG] torch", torch.__version__, "cuda.is_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[DEBUG] device =", torch.cuda.get_device_name(0))
PY

# ---------------- Training configuration ----------------
BSZ=256
ACC=1
EBS=$((BSZ*ACC*1))  # single GPU

# SimpleConv backbone hyperparameters
BACKBONE_DEPTH=10
BACKBONE_KERNEL=3
DILATION_PERIOD=5
GLU_MULT=2
BACKBONE_DROPOUT=0.0

# NEW: MEG encoder backbone selection (must match Python argparse choices)
# Options: dense | exp
MEG_ENCODER="dense"

# Experiment name (encode key settings for reproducibility)
EXP="mous_simpleconv_baseline_${MEG_ENCODER}_d${BACKBONE_DEPTH}_dp${DILATION_PERIOD}_EBS${EBS}_ampoff_tf32off_e100_resplit712"

echo "[INFO] EXP=$EXP"
echo "[INFO] SimpleConv: depth=$BACKBONE_DEPTH kernel=$BACKBONE_KERNEL dp=$DILATION_PERIOD glu_mult=$GLU_MULT dropout=$BACKBONE_DROPOUT"
echo "[INFO] meg_encoder=$MEG_ENCODER"

# ---------------- Train (SimpleConv entrypoint) ----------------
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
  --num_workers 6 \
  --amp off \
  --optimizer adam \
  --lr 3e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.10 \
  --max_epochs 100 \
  --gradient_clip_val 1.0 \
  --early_stop_patience 15 \
  --base_bsz_for_lr 256 \
  \
  --in_channels 272 \
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
echo "[DONE] python exit=$status"
echo "End: $(date)"
exit $status
