#!/usr/bin/env bash
#SBATCH -A <NAISS_PROJECT_ID>
#SBATCH -J msmCtxReRank_L96_A100fat80G
#SBATCH -o runs/%x_%j.out
#SBATCH -e runs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100fat:1      # 80G GPU
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

set -euo pipefail

# ========= Paths (privacy-safe placeholders) =========
PROJECT_ROOT="<PATH_TO_PROJECT_ROOT>"
MANIFEST_DIR="$PROJECT_ROOT/data_manifests_local_global/final_splits_sentence_with_sentence_full"

TRAIN_MAN="$MANIFEST_DIR/train.jsonl"
VAL_MAN="$MANIFEST_DIR/valid.jsonl"
TEST_MAN="$MANIFEST_DIR/test.jsonl"

# Warm-start checkpoint for local baseline (placeholder)
LOCAL_CKPT="<PATH_TO_LOCAL_BASELINE_CKPT>.ckpt"

cd "$PROJECT_ROOT"

# ========= A100 80GB hyperparams =========
NGPUS=1
SENTS_PER_BATCH=32
WIN_PER_SENT=6
ACC=4
WORKERS=12
PREFETCH=4
AMP_MODE="bf16"

COMPILE_MODE="reduce-overhead"   # if unsupported, set to "none"

LR="3e-4"
WD="0.01"
WARMUP="0.10"
MAX_EPS=24
EARLY_STOP=8
EARLY_STOP_METRIC="mrr"

CTX_MAX_WINDOWS=16
CTX_MEM_LEN=96
MEM_ENC_LAYERS=4
MEM_ENC_HEADS=8
MEM_DROPOUT=0.06
CTX_EXCL_RADIUS=2

WINDOW_TOKEN_AGG="asp"
ASP_HIDDEN=128
CTX_TOKEN_MBATCH=48

EXP_NAME="MSM_CtxReRank_L96h4_ASP_ctx16_A100fat80G_full"

# ========= NEW: MEG encoder backbone =========
# Must match your argparse choices: {dense, exp}
MEG_ENCODER="dense"

# ========= Software environment =========
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true
module load FFmpeg/6.0-GCCcore-12.3.0 || true

# venv (placeholder)
if [[ -d "<PATH_TO_VENV>" ]]; then
  source "<PATH_TO_VENV>/bin/activate"
fi

# ========= Env vars =========
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8"
export MALLOC_ARENA_MAX=2
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CUDA_LAUNCH_BLOCKING=0
export NVIDIA_TF32_OVERRIDE=0

# IMPORTANT: define TF32_MODE (was missing in your original)
# Depending on your code, this may be "1/0" or "true/false".
TF32_MODE="1"

echo "[INFO] TRAIN_MAN   = $TRAIN_MAN"
echo "[INFO] VAL_MAN     = $VAL_MAN"
echo "[INFO] TEST_MAN    = $TEST_MAN"
echo "[INFO] LOCAL_CKPT  = $LOCAL_CKPT"
echo "[INFO] meg_encoder = $MEG_ENCODER"
echo "[INFO] SLURM node  = $(hostname)"
echo "[INFO] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
free -h || true

# ========= Assemble args =========
COMMON_ARGS=(
  --train_manifest "$TRAIN_MAN"
  --val_manifest   "$VAL_MAN"
  --test_manifest  "$TEST_MAN"

  --mode msm_window

  # --- DataModule ---
  --batch_size 256
  --accumulate_grad_batches "$ACC"
  --num_workers "$WORKERS"
  --pin_memory
  --prefetch_factor "$PREFETCH"
  --persistent_workers
  --devices "$NGPUS"
  --amp "$AMP_MODE"
  --tf32 "$TF32_MODE"
  --compile "$COMPILE_MODE"

  # --- Optim ---
  --optimizer adamw
  --lr "$LR"
  --weight_decay "$WD"
  --warmup_ratio "$WARMUP"
  --max_epochs "$MAX_EPS"
  --gradient_clip_val 1.0
  --early_stop_patience "$EARLY_STOP"
  --metrics_every_n_steps 200
  --early_stop_metric "$EARLY_STOP_METRIC"
  --seed 42
  --default_root_dir runs
  --experiment_name "$EXP_NAME"

  # --- Encoder base (keep your channel choices) ---
  --meg_encoder "$MEG_ENCODER"
  --in_channels 208 --spatial_channels 208
  --d_model 320 --out_channels 1024
  --backbone_type cnn --backbone_depth 5
  --subject_layer_pos early

  # --- Memory config ---
  --mem_enc_layers "$MEM_ENC_LAYERS" --mem_enc_heads "$MEM_ENC_HEADS" --mem_dropout_p "$MEM_DROPOUT"
  --context_memory_len "$CTX_MEM_LEN"
  --rpe_max_rel 32
  --window_token_agg "$WINDOW_TOKEN_AGG" --asp_hidden "$ASP_HIDDEN"
  --ctx_token_mbatch "$CTX_TOKEN_MBATCH"

  # --- Context sampling ---
  --ctx_max_windows "$CTX_MAX_WINDOWS"
  --ctx_stride 1
  --sentences_per_batch "$SENTS_PER_BATCH"
  --windows_per_sentence "$WIN_PER_SENT"
  --exclude_self_from_ctx
  --ctx_exclude_radius "$CTX_EXCL_RADIUS"

  # --- Warm-start baseline ---
  --local_ckpt "$LOCAL_CKPT"

  # --- Freeze local branch only ---
  --freeze_local
)

set -x
python -u -m train.train_fused "${COMMON_ARGS[@]}" \
  --out_timesteps -1 \
  --bias_mode diag \
  --bias_rowwise_center \
  --bias_start_epoch 1 \
  --bias_cap 0.5 \
  --rerank_bank_reduce rms \
  --rerank_dropout 0.10 \
  --mem_train_start_epoch 1 \
  --logit_scale_lr_mult 0.05
status=$?
set +x

echo "[DONE] python exit=$status"
exit $status
