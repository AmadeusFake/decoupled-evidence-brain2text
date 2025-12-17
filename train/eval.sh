#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1173
#SBATCH -J CtxSentMax_fold0
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/runs_ctx_sentence_max/CtxSentMax_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/runs_ctx_sentence_max/CtxSentMax_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xinyu4@kth.se

set -euo pipefail

# ------------- 路径配置 -------------
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
MANI_DIR="$PROJECT_ROOT/data_manifests_local_global/resplit_sentence_a025_SLR_kfold/fold0"

# 训练得到的 run 目录（应包含 records/best_checkpoint.txt）
RUN_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/run_baseline/none_baseline_SLR_fold0_EBS256_ampoff_tf32off_adam_wd0_e100_20251015-154812_job5203747"

# ------------- 可调参数（可用 env 覆盖） -------------
CANDIDATE_SPLITS="${CANDIDATE_SPLITS:-test}"   # train|valid|test|train+valid|all
SENT_ROUND_MS="${SENT_ROUND_MS:-10.0}"         # 句子边界对齐四舍五入（毫秒）
AMP="${AMP:-bf16}"                              # off|bf16|fp16|16-mixed
DTYPE_HALF="${DTYPE_HALF:-bf16}"                # off|bf16|fp16
POOL_CHUNK="${POOL_CHUNK:-4096}"                # 候选窗口分块
WIN_BATCH="${WIN_BATCH:-64}"                    # 编码查询句子窗口 batch
MAX_WINDOWS="${MAX_WINDOWS:-0}"                 # 0=使用该句所有窗口
STRICT_SUBJECTS="${STRICT_SUBJECTS:-1}"         # 1=只评测映射到的 subject

# ------------- 环境 -------------
module purge || true
module load Python/3.11.3-GCCcore-12.3.0 || true
source "/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/whisper_venv/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME"

# perf knobs
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS=1
export TORCH_ALLOW_TF32_CUDNN=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.9"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MALLOC_ARENA_MAX=2

mkdir -p "$RUN_DIR/records"
mkdir -p "/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/runs_ctx_sentence_max"

echo "[INFO] RUN_DIR=$RUN_DIR"
nvidia-smi || true
python - <<'PY'
import torch
print("[DEBUG] torch", torch.__version__, "cuda.is_available =", torch.cuda.is_available())
if torch.cuda.is_available(): print("[DEBUG] device =", torch.cuda.get_device_name(0))
PY

cd "$PROJECT_ROOT"

# ------------- 执行 -------------
python -u -m train.retrieval_window_vote \
  --train_manifest "$MANI_DIR/train.jsonl" \
  --val_manifest   "$MANI_DIR/valid.jsonl" \
  --test_manifest  "$MANI_DIR/test.jsonl"  \
  --run_dir "$RUN_DIR" \
  --use_best_ckpt \
  --candidate_splits "$CANDIDATE_SPLITS" \
  --device cuda \
  --amp "$AMP" \
  --dtype_half "$DTYPE_HALF" \
  --pool_chunk "$POOL_CHUNK" \
  --win_batch "$WIN_BATCH" \
  --topk 1,5,10 \
  --max_windows "$MAX_WINDOWS" \
  --sent_round_ms "$SENT_ROUND_MS" \
  $( (( STRICT_SUBJECTS )) && echo --strict_subjects )
