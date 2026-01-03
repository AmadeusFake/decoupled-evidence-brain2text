#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Root directory containing fully-preprocessed manifests (after Stage-3 / 3.5)
DATA_ROOT="${PROJECT_ROOT}/data_manifests_local_global"

# Directory containing GW audio sentence-level scripts
SCRIPT_DIR="${PROJECT_ROOT}/data_script_GW"

###############################################################################
# Scripts (internal, do NOT edit)
###############################################################################

BUILD_SENT_TABLE="${SCRIPT_DIR}/build_sentence_table_from_window_manifest.py"
STAGEA_AUDIO="${SCRIPT_DIR}/stageA_audio_sentence_features_lenrobust.py"
ATTACH_BACK="${SCRIPT_DIR}/attach_sentence_features_to_windows.py"

###############################################################################
# Input window manifests (with sentence-full MEG already attached)
###############################################################################

WIN_SENT_DIR="${DATA_ROOT}/final_splits_sentence_with_sentence_full"
WIN_WORD_DIR="${DATA_ROOT}/final_splits_word_list_with_sentence_full"

###############################################################################
# Output directories (α fixed at 0.25)
###############################################################################

AUDIO_ROOT_SENT="${DATA_ROOT}/precomputed_audio_sentence_features_attn/sentence"
AUDIO_ROOT_WORD="${DATA_ROOT}/precomputed_audio_sentence_features_attn/word_list"

WIN_WITH_AUDIO_SENT="${DATA_ROOT}/final_splits_sentence_with_sentence_full_audio_attn"
WIN_WITH_AUDIO_WORD="${DATA_ROOT}/final_splits_word_list_with_sentence_full_audio_attn"

mkdir -p \
  "${AUDIO_ROOT_SENT}/npy" "${AUDIO_ROOT_SENT}/sent_index" \
  "${AUDIO_ROOT_WORD}/npy" "${AUDIO_ROOT_WORD}/sent_index" \
  "${WIN_WITH_AUDIO_SENT}" "${WIN_WITH_AUDIO_WORD}"

###############################################################################
# Environment configuration
###############################################################################

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# HuggingFace cache (wav2vec2 weights)
export HF_HOME="${PROJECT_ROOT}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

###############################################################################
# Helper functions
###############################################################################

# Step 1: Build a deduplicated sentence table from window manifests
build_sentence_table () {
  local WIN_DIR="$1"
  local OUT_DIR="$2"

  echo "== Step 1: Build deduplicated sentence table from ${WIN_DIR} =="

  python "${BUILD_SENT_TABLE}" \
    --input_window_manifest_dir "${WIN_DIR}" \
    --output_sentence_table_dir "${OUT_DIR}"
}

# Step 2: Stage-A audio sentence tokens (deterministic)
run_stageA_audio () {
  local SENT_TABLE_DIR="$1"
  local FEAT_ROOT="$2"

  echo
  echo "== Step 2: Stage-A audio sentence tokens =="
  echo "   wav2vec2 layers 14–18 | mean | TPP(1,2,4,8) μ|σ | per-token L2 | FP32"

  python "${STAGEA_AUDIO}" \
    --sentence_table_dir "${SENT_TABLE_DIR}" \
    --output_feature_dir "${FEAT_ROOT}/npy" \
    --output_sentence_dir "${FEAT_ROOT}/sent_index" \
    --batch_size 16 \
    --num_workers 8 \
    --prefetch_factor 2 \
    --device cuda \
    --amp off \
    --save_dtype float32 \
    --w2v_model "facebook/wav2vec2-large-xlsr-53" \
    --w2v_layers "14,15,16,17,18" \
    --w2v_agg mean \
    --pooling tpp \
    --tpp_levels "1,2,4,8" \
    --with_std \
    --time_drop_p 0.0 \
    --ensemble 1 \
    --power_norm sqrt \
    --unit_norm \
    --bin_center_alpha 0.25 \
    --resume \
    --verify_existing \
    --recompute_existing
}

# Step 3: Attach sentence-level audio tokens back to window manifests
attach_audio_back () {
  local WIN_IN_DIR="$1"
  local FEAT_ROOT="$2"
  local WIN_OUT_DIR="$3"

  echo
  echo "== Step 3: Attach audio sentence tokens back to windows =="

  python "${ATTACH_BACK}" \
    --input_window_manifest_dir "${WIN_IN_DIR}" \
    --input_sentence_manifest_dir "${FEAT_ROOT}/sent_index" \
    --output_window_manifest_dir "${WIN_OUT_DIR}" \
    --feature_field "audio_sentence_feature_path" \
    --output_feature_field "audio_sentence_feature_path" \
    --emit_mapping_tsv
}

# Step 4: Similarity sanity check (symmetric Chamfer, length-normalized)
check_similarity () {
  local FEAT_ROOT="$1"

  echo
  echo "== Step 4: Similarity check among deduplicated sentence audio tokens =="

  SENT_INDEX_DIR="${FEAT_ROOT}/sent_index" python - <<'PY'
import os, json, numpy as np
from pathlib import Path

def read_jsonl(p):
    out=[]
    if not p.exists(): return out
    with open(p,"r") as f:
        for line in f:
            line=line.strip()
            if line:
                try: out.append(json.loads(line))
                except: pass
    return out

def load_tokens(records):
    T=[]
    for r in records:
        p=r.get("audio_sentence_feature_path","")
        if not p: continue
        try: x=np.load(p)
        except: continue
        if x.ndim!=2: continue
        x=x.astype(np.float32)
        x /= (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
        T.append(x)
    return T

root=Path(os.environ["SENT_INDEX_DIR"])
records=[]
for sp in ("train","valid","test"):
    records += read_jsonl(root/f"{sp}.jsonl")

tokens = load_tokens(records)
N=len(tokens)
if N==0:
    print("[simcheck] no 2D tokens found, skip.")
    raise SystemExit(0)

def chamfer_sym(a,b):
    S=a@b.T
    return 0.5*(S.max(1).sum()+S.max(0).sum())/(np.sqrt(a.shape[0]*b.shape[0])+1e-6)

scores=[]
for i in range(N):
    best=-1e9
    for j in range(N):
        if i==j: continue
        best=max(best, chamfer_sym(tokens[i], tokens[j]))
    scores.append(best)

print(f"[simcheck] N={N}, mean={np.mean(scores):.4f}, "
      f"p95={np.percentile(scores,95):.4f}, p99={np.percentile(scores,99):.4f}")
PY
}

###############################################################################
# Pipeline execution
###############################################################################

# sentence
SENT_TABLE_SENT="${DATA_ROOT}/sent_table_dedup/sentence"
build_sentence_table "${WIN_SENT_DIR}" "${SENT_TABLE_SENT}"
run_stageA_audio "${SENT_TABLE_SENT}" "${AUDIO_ROOT_SENT}"
attach_audio_back "${WIN_SENT_DIR}" "${AUDIO_ROOT_SENT}" "${WIN_WITH_AUDIO_SENT}"
check_similarity "${AUDIO_ROOT_SENT}"

# word_list
SENT_TABLE_WORD="${DATA_ROOT}/sent_table_dedup/word_list"
build_sentence_table "${WIN_WORD_DIR}" "${SENT_TABLE_WORD}"
run_stageA_audio "${SENT_TABLE_WORD}" "${AUDIO_ROOT_WORD}"
attach_audio_back "${WIN_WORD_DIR}" "${AUDIO_ROOT_WORD}" "${WIN_WITH_AUDIO_WORD}"
check_similarity "${AUDIO_ROOT_WORD}"

###############################################################################
# Final report
###############################################################################

echo
echo "Audio sentence-level tokens (TPP μ|σ, per-token L2, FP32, deterministic) completed ✅"
echo "Updated window manifests:"
echo "- ${WIN_WITH_AUDIO_SENT}"
echo "- ${WIN_WITH_AUDIO_WORD}"
