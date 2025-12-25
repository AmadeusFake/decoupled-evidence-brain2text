#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_BASE="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# BIDS-style dataset root (raw data)
BIDS_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/download"

# Directory containing preprocessing scripts
SCRIPT_DIR="${PROJECT_BASE}/data_script"

# Output directory for manifests and splits
OUT_DIR="${PROJECT_BASE}/data_manifests_local_global"

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

META_MANIFEST="${OUT_DIR}/meta_manifest.jsonl"

###############################################################################
# Stage 1: Create meta-manifest (blueprint)
###############################################################################

echo "== Stage 1: Creating meta-manifest (blueprint) =="

mkdir -p "${OUT_DIR}"

# Meta-manifest is created only once.
# Remove the file manually if a full rebuild is required.
if [[ ! -f "${META_MANIFEST}" ]]; then
  python "${SCRIPT_DIR}/create_meta_manifest.py" \
    --input_dir "${BIDS_DIR}" \
    --output_dir "${OUT_DIR}" \
    --num_workers 32
else
  echo "Meta-manifest already exists:"
  echo "- ${META_MANIFEST}"
fi

###############################################################################
# Stage 2: Generate anchor-window splits (sentence / word_list)
###############################################################################

echo
echo "== Stage 2: Creating anchor-window splits =="

for DATA_TYPE in sentence word_list; do
  SPLIT_DIR="${OUT_DIR}/final_splits_${DATA_TYPE}"

  echo
  echo "-- Processing data_type=${DATA_TYPE} --"

  python "${SCRIPT_DIR}/create_local_global_splits.py" \
    --meta_manifest_path "${META_MANIFEST}" \
    --output_dir "${OUT_DIR}" \
    --bids_root_dir "${BIDS_DIR}" \
    --data_type "${DATA_TYPE}" \
    --split_ratios "0.7,0.1,0.2" \
    --random_seed 42

  ###########################################################################
  # Candidate pool diagnostics
  ###########################################################################

  echo "Candidate pool statistics (${DATA_TYPE}):"
  python "${SCRIPT_DIR}/report_candidate_pool.py" \
    --dir "${SPLIT_DIR}"
done

###############################################################################
# Final report
###############################################################################

echo
echo "Initial blueprint and splits completed successfully ✅"
echo "Output directory:"
echo "- ${OUT_DIR}"
