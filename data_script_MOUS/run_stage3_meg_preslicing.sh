#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Root directory for MOUS manifests and features
DATA_ROOT_MOUS="${PROJECT_ROOT}/data_mous_local_global"

# MEG preprocessing script (MOUS-adapted)
SCRIPT_PATH="${PROJECT_ROOT}/data_script_MOUS/preprocess_meg_to_npy.py"

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

# Input manifests (after Audio Stage-2)
INPUT_MANIFEST_DIR="${DATA_ROOT_MOUS}/final_splits_sentence_precomputed"

# Output MEG windows and updated manifests
OUTPUT_MEG_DIR="${DATA_ROOT_MOUS}/precomputed_meg_windows/sentence"
OUTPUT_MANIFEST_DIR="${DATA_ROOT_MOUS}/final_splits_sentence_fully_preprocessed"

###############################################################################
# Temporary directories and MNE cache
###############################################################################

# Avoid /tmp overflow on HPC systems
export TMPDIR="${PROJECT_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

# MNE cache for FIF loading / resampling
export MNE_CACHE_DIR="${PROJECT_ROOT}/mne_cache"

mkdir -p "${TMPDIR}" "${MNE_CACHE_DIR}"

###############################################################################
# CPU threading limits
###############################################################################

# Prevent oversubscription in MNE / NumPy
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

###############################################################################
# Sanity check: subject count (should be 96)
###############################################################################

echo "== [MOUS Stage 3] Sanity check: number of subjects (expected: 96) =="

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
print("        (top 5 by sample count)")
for k, v in sorted(subs.items(), key=lambda x: -x[1])[:5]:
    print(f"   {k}: rows={v}")
PY

###############################################################################
# Stage 3: MEG preprocessing and window extraction (sentence-only)
###############################################################################

echo
echo "== [MOUS] MEG Stage 3: sentence-only =="

python "${SCRIPT_PATH}" \
  --input_manifest_dir "${INPUT_MANIFEST_DIR}" \
  --output_meg_dir "${OUTPUT_MEG_DIR}" \
  --output_manifest_dir "${OUTPUT_MANIFEST_DIR}" \
  --num_workers 4 \
  --target_sfreq 120 \
  --baseline_end_s 0.3 \
  --std_clamp 20 \
  --fit_max_windows_per_recording 200 \
  --resume \
  --verify_existing
  # For a full rebuild, add: --recompute_existing

###############################################################################
# Final report
###############################################################################

echo
echo "MOUS MEG preprocessing completed successfully ✅"
echo "Final manifests:"
echo "- ${OUTPUT_MANIFEST_DIR}"
echo "Common head-MEG channel list:"
echo "- ${OUTPUT_MEG_DIR}/common_head_meg_channels.json"
