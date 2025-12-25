#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
# All other parts of the script should remain unchanged.
###############################################################################

# Project root (change this)
PROJECT_BASE="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# Directory containing Python preprocessing scripts (do NOT rename files)
SCRIPT_DIR="${PROJECT_BASE}/data_script_BN"

# Brennan EEG dataset root (must contain audio/, proc/, and metadata files)
EEG_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/Brennan"

# Subject-level comprehension score file
SCORES_FILE="${EEG_ROOT}/comprehension-scores.txt"

# Output directory for generated manifests and splits
OUT_DIR="${PROJECT_BASE}/EEGdata_manifests_local_global"

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

META_MANIFEST="${OUT_DIR}/meta_manifest.jsonl"
KEEP_SUBJECTS_TXT="${OUT_DIR}/keep33.txt"

###############################################################################
# Sanity checks
###############################################################################

echo "== Running sanity checks =="

[[ -d "${EEG_ROOT}/audio" ]] || { echo "ERROR: Missing ${EEG_ROOT}/audio"; exit 2; }
[[ -d "${EEG_ROOT}/proc"  ]] || { echo "ERROR: Missing ${EEG_ROOT}/proc"; exit 2; }
[[ -f "${EEG_ROOT}/AliceChapterOne-EEG.csv" ]] || {
  echo "ERROR: Missing AliceChapterOne-EEG.csv"; exit 2;
}
[[ -f "${SCORES_FILE}" ]] || { echo "ERROR: Missing ${SCORES_FILE}"; exit 2; }

for f in \
  create_meta_manifest_brennan_eeg.py \
  create_local_global_splits.py \
  report_candidate_pool.py
do
  [[ -f "${SCRIPT_DIR}/${f}" ]] || {
    echo "ERROR: Missing script ${SCRIPT_DIR}/${f}"; exit 2;
  }
done

mkdir -p "${OUT_DIR}"

###############################################################################
# Step 0: Generate subject whitelist from comprehension scores
###############################################################################

echo "== Generating subject whitelist (keep only 'use') =="

awk '
  BEGIN { IGNORECASE = 1 }
  /^S[0-9]+/ {
    sid = $1
    line = $0
    if (line ~ /exclude/) next
    if (line ~ /use/) print sid
  }
' "${SCORES_FILE}" | sort -u > "${KEEP_SUBJECTS_TXT}"

echo "Kept subjects:"
cat "${KEEP_SUBJECTS_TXT}"

NUM_SUBJECTS=$(wc -l < "${KEEP_SUBJECTS_TXT}" | tr -d ' ')
echo "Total subjects kept: ${NUM_SUBJECTS}"

if [[ "${NUM_SUBJECTS}" -lt 1 ]]; then
  echo "ERROR: No subjects selected. Check 'use/exclude' labels in ${SCORES_FILE}"
  exit 3
fi

###############################################################################
# Step 1: Create EEG meta-manifest (sentence-level only)
###############################################################################

echo
echo "== Step 1: Creating EEG meta-manifest (sentence-only) =="

# Meta-manifest is created only if it does not already exist.
# Remove the file manually if a full rebuild is required.
if [[ ! -f "${META_MANIFEST}" ]]; then
  python "${SCRIPT_DIR}/create_meta_manifest_brennan_eeg.py" \
    --brennan_root "${EEG_ROOT}" \
    --output_dir "${OUT_DIR}" \
    --keep_subjects_file "${KEEP_SUBJECTS_TXT}"
fi

if [[ ! -f "${META_MANIFEST}" ]]; then
  echo "ERROR: Failed to generate ${META_MANIFEST}"
  exit 4
fi

# Quick statistics
echo "== Meta-manifest statistics (sentence-only) =="
NUM_SENTENCES=$(grep -E -c '"type"[[:space:]]*:[[:space:]]*"sentence"' \
  "${META_MANIFEST}" || true)
echo "Sentence entries: ${NUM_SENTENCES}"

###############################################################################
# Step 2: Generate local/global anchor-window splits
###############################################################################

echo
echo "== Step 2: Creating anchor-window splits (sentence-only) =="

SPLIT_DIR="${OUT_DIR}/final_splits_sentence"

python "${SCRIPT_DIR}/create_local_global_splits.py" \
  --meta_manifest_path "${META_MANIFEST}" \
  --output_dir "${OUT_DIR}" \
  --bids_root_dir "${EEG_ROOT}" \
  --split_ratios "0.7,0.1,0.2" \
  --random_seed 42

###############################################################################
# Step 3: Candidate pool diagnostics
###############################################################################

echo
echo "== Candidate pool report (sentence-level) =="

python "${SCRIPT_DIR}/report_candidate_pool.py" \
  --dir "${SPLIT_DIR}"

echo
echo "Pipeline completed successfully ✅"
echo "Output directory: ${OUT_DIR}"
