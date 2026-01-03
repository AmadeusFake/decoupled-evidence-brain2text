#!/usr/bin/env bash
#
# ==============================================================================
# SLURM script: EEG training ONLY (no eval)
#
# NEW:
#   Pass --meg_encoder {dense,exp} to match the new argparse option in Python.
#
# Privacy note:
#   Replace placeholders (<...>) with your own values (project ID, paths, email).
# ==============================================================================

#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J eeg_megstyle_cnn
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

# ---------------- Paths & environment ----------------
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

DATA_DIR="$PROJECT_ROOT/EEGdata_manifests_local_global"
SPLIT_DIR="$DATA_DIR/train_resplit_from_train"

TRAIN_MAN="$SPLIT_DIR/train.jsonl"
VAL_MAN="$SPLIT_DIR/valid.jsonl"
TEST_MAN="$SPLIT_DIR/test.jsonl"

# Global subject mapping (recommended for consistent reruns)
SUBJECT_MAP="$DATA_DIR/subject_mapping_eeg_union.json"

# Root output directory (train script will create EXP_timestamp_jobID underneath)
RUN_ROOT="$PROJECT_ROOT/runs_eeg"

# Cluster modules / venv (adjust as needed)
module load Python/3.11.3-GCCcore-12.3.0 || true
source <PATH_TO_VENV>/bin/activate || true

export TOKENIZERS_PARALLELISM=false
export HF_HOME="<PATH_TO_HF_CACHE>"
unset TRANSFORMERS_CACHE
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NVIDIA_TF32_OVERRIDE=0  # keep TF32 off for strict FP32 alignment

# ---------------- Sanity checks ----------------
[[ -f "$TRAIN_MAN" ]] || { echo "[ERROR] train manifest not found: $TRAIN_MAN"; exit 2; }
[[ -f "$VAL_MAN"   ]] || { echo "[ERROR] val manifest not found:   $VAL_MAN";   exit 2; }
[[ -f "$TEST_MAN"  ]] || { echo "[ERROR] test manifest not found:  $TEST_MAN";  exit 2; }
[[ -f "$SUBJECT_MAP" ]] || { echo "[ERROR] subject mapping not found: $SUBJECT_MAP"; exit 2; }

mkdir -p "$RUN_ROOT"

# ---------------- Training hyperparameters (MEG-style CNN config) ----------------
BSZ=256
ACC=1
EXP="EEG_MEGStyleCNN_lateSubj270"

# NEW: MEG encoder backbone selection (must match Python argparse choices)
# Options: dense | exp
MEG_ENCODER="dense"

echo "[INFO] Training config:"
echo "  EXP         = $EXP"
echo "  meg_encoder = $MEG_ENCODER"
echo "  BSZ         = $BSZ"
echo "  ACC         = $ACC"
echo "  TRAIN_MAN   = $TRAIN_MAN"
echo "  VAL_MAN     = $VAL_MAN"
echo "  TEST_MAN    = $TEST_MAN"
echo "  SUBJECT_MAP = $SUBJECT_MAP"
echo "  RUN_ROOT    = $RUN_ROOT"
echo

# ---------------- Train ONLY ----------------
python -u train/train_eeg2.py \
  --train_manifest "$TRAIN_MAN" \
  --val_manifest   "$VAL_MAN" \
  --test_manifest  "$TEST_MAN" \
  --subject_namespace_train EEG \
  --subject_namespace_val   EEG \
  --subject_namespace_test  EEG \
  --subject_mapping_path "$SUBJECT_MAP" \
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
  --early_stop_patience 15 \
  --base_bsz_for_lr 256 \
  \
  --in_channels 60 \
  --out_channels 1024 \
  \
  --spatial_channels 270 \
  --fourier_k 32 \
  --d_model 320 \
  --backbone_depth 5 \
  --backbone_type cnn \
  --dropout 0.1 \
  --subject_layer_pos early \
  --use_subjects \
  --spatial_dropout_p 0.0 \
  --spatial_dropout_radius 0.2 \
  \
  --seed 42 \
  --default_root_dir "$RUN_ROOT" \
  --experiment_name "$EXP" \
  --metrics_every_n_steps 500 \
  --normalize
  # Note: no --loss_temp => fixed temperature behavior in code (e.g., scale≈14.3).

echo
echo "======================================================"
echo "[DONE] Training finished."
echo "End: $(date)"
echo "Outputs under: $RUN_ROOT"
echo "======================================================"
