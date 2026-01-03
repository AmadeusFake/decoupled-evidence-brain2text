#!/usr/bin/env bash
#
# ==============================================================================
# SLURM script: Local paper baseline training
# - Privacy-safe (no personal paths / emails / project IDs)
# - Reproducible: explicit manifests, TF32 off, AMP off
# - NEW: --meg_encoder {dense,exp}
#
# Usage:
#   1) Replace placeholders (<...>) below
#   2) sbatch this_script.sh
# ==============================================================================

#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J local_paper_baseline
#SBATCH -o runs/%x_%j.out
#SBATCH -e runs/%x_%j.err
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

# ---------- Paths (configure these) ----------
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Default: resplit 7:1:2 (sentence-level)
TRAIN_MAN="$PROJECT_ROOT/data_manifests_local_global/final_splits_sentence_fully_preprocessed/train.jsonl"
VAL_MAN="$PROJECT_ROOT/data_manifests_local_global/final_splits_sentence_fully_preprocessed/valid.jsonl"
TEST_MAN="$PROJECT_ROOT/data_manifests_local_global/final_splits_sentence_fully_preprocessed/test.jsonl"

# Alternative manifests (commented out)
# TRAIN_MAN="$PROJECT_ROOT/data_manifests_local_global/final_splits_word_list_with_sentence_full_audio_attn/a025/train.jsonl"
# VAL_MAN="$PROJECT_ROOT/data_manifests_local_global/final_splits_word_list_with_sentence_full_audio_attn/a025/valid.jsonl"
# TEST_MAN="$PROJECT_ROOT/data_manifests_local_global/final_splits_word_list_with_sentence_full_audio_attn/a025/test.jsonl"

# ---------- Environment ----------
module load Python/3.11.3-GCCcore-12.3.0 || true

# Activate your environment (venv/conda). Replace the placeholder.
source "<PATH_TO_VENV>/bin/activate" || true

export TOKENIZERS_PARALLELISM=false
export HF_HOME="<PATH_TO_HF_CACHE>"
unset TRANSFORMERS_CACHE

# Paper-aligned numerics: FP32 + TF32 OFF
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NVIDIA_TF32_OVERRIDE=0

# ---------- Sanity checks ----------
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

# ---------- NEW: MEG encoder backbone ----------
# Must match your Python argparse choices: {dense, exp}
MEG_ENCODER="dense"

# ---------- Training args (paper-aligned; mirrors your JSON) ----------
BSZ=256
ACC=1
EBS=$((BSZ*ACC*1))  # single GPU
EXP="none_baseline_safe_EBS${EBS}_ampoff_tf32off_adam_wd0_e100_resplit812"

python -u -m train.train \
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
  --in_channels 208 \
  --spatial_channels 270 \
  --fourier_k 32 \
  --d_model 320 \
  --out_channels 1024 \
  --backbone_depth 5 \
  --backbone_type cnn \
  --subject_layer_pos late \
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
