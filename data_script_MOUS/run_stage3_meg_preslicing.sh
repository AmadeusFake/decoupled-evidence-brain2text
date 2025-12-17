#!/bin/bash
set -euo pipefail

# ---------- 路径 ----------
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
DATA_ROOT_MOUS="$PROJECT_ROOT/data_mous_local_global"
SCRIPT_PATH="$PROJECT_ROOT/data_script_MOUS/preprocess_meg_to_npy.py"

IN_DIR="$DATA_ROOT_MOUS/final_splits_sentence_precomputed"          # 🔗 来自 Audio Stage2
OUT_MEG="$DATA_ROOT_MOUS/precomputed_meg_windows/sentence"          # 保存 MEG 窗口/坐标/robust 参数
OUT_MAN="$DATA_ROOT_MOUS/final_splits_sentence_fully_preprocessed"  # 新 manifest（含 meg_win_path/coords）

# ---------- 临时目录 & MNE 缓存 ----------
export TMPDIR="$PROJECT_ROOT/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export MNE_CACHE_DIR="$PROJECT_ROOT/mne_cache"
mkdir -p "$TMPDIR" "$MNE_CACHE_DIR"

# 限制 CPU 线程（防止 MNE / numpy 暴走）
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# ---------- 小检查：确认是 96 个 subject ----------
echo "== [MOUS Stage3] sanity check: subject 数量（应该是 96） =="
python - <<'PY'
import json
from pathlib import Path
from collections import Counter

base = Path("/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/data_mous_local_global")
splits_dir = base / "final_splits_sentence_precomputed"

subs = Counter()
for split in ("train", "valid", "test"):
    p = splits_dir / f"{split}.jsonl"
    if not p.exists():
        continue
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("subject_id") or r.get("subject")
            if sid:
                subs[sid] += 1

print(f"[CHECK] subjects in final_splits_sentence_precomputed: {len(subs)}")
print("        (示例前 5 个)")
for k, v in sorted(subs.items(), key=lambda x: -x[1])[:5]:
    print("   ", k, "rows=", v)
PY

# ---------- 跑 Stage3 预切 ----------
echo
echo "== [MOUS] MEG Stage3: sentence =="
python "$SCRIPT_PATH" \
  --input_manifest_dir "$IN_DIR" \
  --output_meg_dir "$OUT_MEG" \
  --output_manifest_dir "$OUT_MAN" \
  --num_workers 4 \
  --target_sfreq 120 \
  --baseline_end_s 0.3 \
  --std_clamp 20 \
  --fit_max_windows_per_recording 200 \
  --resume \
  --verify_existing
  # 如需强制重算可以加:  --recompute_existing

echo
echo "All MOUS MEG pre-slicing done."
echo "Final manifests:"
echo "  $OUT_MAN"
echo "Common head-MEG channel list:"
echo "  $OUT_MEG/common_head_meg_channels.json"
     