#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths
# =========================
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT="$PROJECT_ROOT/data_manifests_local_global"
SCRIPT_DIR="$PROJECT_ROOT/data_script_gw"

# 脚本
BUILD_SENT_TABLE="$SCRIPT_DIR/build_sentence_table_from_window_manifest.py"
STAGEA="$SCRIPT_DIR/stageA_audio_sentence_features_lenrobust.py"
ATTACH_BACK="$SCRIPT_DIR/attach_sentence_features_to_windows.py"

# 输入窗口 manifests
WIN_SENT_DIR="$DATA_ROOT/final_splits_sentence_with_sentence_full"
WIN_WORD_DIR="$DATA_ROOT/final_splits_word_list_with_sentence_full"

# 输出目录 (固定 α=0.25)
AUDIO_ROOT_SENT="$DATA_ROOT/precomputed_audio_sentence_features_attn/sentence"
AUDIO_ROOT_WORD="$DATA_ROOT/precomputed_audio_sentence_features_attn/word_list"
WIN_WITH_AUDIO_SENT="$DATA_ROOT/final_splits_sentence_with_sentence_full_audio_attn"
WIN_WITH_AUDIO_WORD="$DATA_ROOT/final_splits_word_list_with_sentence_full_audio_attn"

mkdir -p "$AUDIO_ROOT_SENT/npy" "$AUDIO_ROOT_SENT/sent_index" \
         "$AUDIO_ROOT_WORD/npy" "$AUDIO_ROOT_WORD/sent_index" \
         "$WIN_WITH_AUDIO_SENT" "$WIN_WITH_AUDIO_WORD"

# =========================
# Environment
# =========================
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export HF_HOME="$PROJECT_ROOT/hf_home"
export TRANSFORMERS_CACHE="$HF_HOME"

# =========================
# Helpers
# =========================
build_table () {
  local WIN_DIR="$1"
  local OUT_DIR="$2"
  echo "== Step 1: Build dedup sentence table from $WIN_DIR =="
  python "$BUILD_SENT_TABLE" \
    --input_window_manifest_dir "$WIN_DIR" \
    --output_sentence_table_dir "$OUT_DIR"
}

run_stageA_attn () {
  local SENT_TABLE_DIR="$1"
  local FEAT_DIR="$2"
  echo
  echo "== Step 2: StageA-ATTN (14–18 mean → TPP μ|σ → per-token L2 → FP32, deterministic) on $SENT_TABLE_DIR =="
  python "$STAGEA" \
    --sentence_table_dir "$SENT_TABLE_DIR" \
    --output_feature_dir "$FEAT_DIR/npy" \
    --output_sentence_dir "$FEAT_DIR/sent_index" \
    --batch_size 16 \
    --num_workers 8 \
    --prefetch_factor 2 \
    --device cuda \
    --amp off \
    --save_dtype float32 \
    --w2v_model "facebook/wav2vec2-large-xlsr-53" \
    --w2v_layers "14,15,16,17,18" \
    --w2v_agg mean \
    --pooling tpp --tpp_levels "1,2,4,8" --with_std \
    --time_drop_p 0.0 --ensemble 1 \
    --power_norm sqrt --unit_norm \
    --bin_center_alpha 0.25 \
    --resume --verify_existing --recompute_existing
}

attach_back_audio () {
  local WIN_IN_DIR="$1"
  local SENT_INDEX_DIR="$2/sent_index"
  local WIN_OUT_DIR="$3"
  echo
  echo "== Step 3: Attach AUDIO sentence tokens back to windows =="
  python "$ATTACH_BACK" \
    --input_window_manifest_dir "$WIN_IN_DIR" \
    --input_sentence_manifest_dir "$SENT_INDEX_DIR" \
    --output_window_manifest_dir "$WIN_OUT_DIR" \
    --feature_field "audio_sentence_feature_path" \
    --output_feature_field "audio_sentence_feature_path" \
    --emit_mapping_tsv
}

check_similarity () {
  local SENT_INDEX_DIR="$1/sent_index"
  echo
  echo "== Step 4: Similarity check among dedup sentence audio TOKENS (symmetric Chamfer, sqrt length-norm) =="
  SENT_INDEX_DIR="$SENT_INDEX_DIR" python - <<'PY'
import os, json, numpy as np
from pathlib import Path

def read_jsonl(p: Path):
    out=[]
    if not p.exists(): return out
    with open(p, "r") as f:
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
        if x.ndim!=2:   # 我们只检查新版 2D tokens
            continue
        x=x.astype(np.float32)
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        x = x / n
        T.append(x)
    return T

root=Path(os.environ["SENT_INDEX_DIR"])
all_recs=[]
for sp in ("train","valid","test"):
    all_recs += read_jsonl(root/f"{sp}.jsonl")

tokens = load_tokens(all_recs)
N = len(tokens)
if N == 0:
    print("[simcheck] no 2D tokens found, skip.")
    raise SystemExit(0)

def chamfer_sym(a, b):
    # a:[La,D], b:[Lb,D], 已 per-token L2
    S = a @ b.T
    s_q2k = S.max(axis=1).sum()
    s_k2q = S.max(axis=0).sum()
    Lq, Lk = a.shape[0], b.shape[0]
    return 0.5*(s_q2k + s_k2q) / (np.sqrt(Lq*Lk) + 1e-6)

top1 = []
for i in range(N):
    best = -np.inf
    for j in range(N):
        if i==j: continue
        sc = chamfer_sym(tokens[i], tokens[j])
        if sc > best: best = sc
    top1.append(best)

print(f"[simcheck-chamfer] N={N}, mean={np.mean(top1):.4f}, p95={np.percentile(top1,95):.4f}, p99={np.percentile(top1,99):.4f}")
PY
}

# =========================
# Pipeline
# =========================
# sentence
SENT_TABLE_SENT_DIR="$DATA_ROOT/sent_table_dedup/sentence"
build_table "$WIN_SENT_DIR" "$SENT_TABLE_SENT_DIR"
run_stageA_attn "$SENT_TABLE_SENT_DIR" "$AUDIO_ROOT_SENT"
attach_back_audio "$WIN_SENT_DIR" "$AUDIO_ROOT_SENT" "$WIN_WITH_AUDIO_SENT"
check_similarity "$AUDIO_ROOT_SENT"

# word_list
SENT_TABLE_WORD_DIR="$DATA_ROOT/sent_table_dedup/word_list"
build_table "$WIN_WORD_DIR" "$SENT_TABLE_WORD_DIR"
run_stageA_attn "$SENT_TABLE_WORD_DIR" "$AUDIO_ROOT_WORD"
attach_back_audio "$WIN_WORD_DIR" "$AUDIO_ROOT_WORD" "$WIN_WITH_AUDIO_WORD"
check_similarity "$AUDIO_ROOT_WORD"

echo
echo "✅ AUDIO 句子级 TOKENS（TPP μ|σ，per-token L2，FP32，无随机）已完成并回填。"
