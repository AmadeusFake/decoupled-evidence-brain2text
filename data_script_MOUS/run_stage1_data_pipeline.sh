#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable paths
# ---------------------------------------------------------------------------
# Only modify variables in this section when reproducing the pipeline.
###############################################################################

# Project root directory
PROJECT_BASE="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"

# MOUS BIDS root directory (raw data)
BIDS_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/MOUS_raw"

# Directory containing MOUS-specific scripts
SCRIPT_DIR="${PROJECT_BASE}/data_script_MOUS"

# Output directory for MOUS manifests and splits
OUT_DIR="${PROJECT_BASE}/data_mous_local_global"

###############################################################################
# Internal paths (do NOT edit)
###############################################################################

ALIGN_JSONL="${OUT_DIR}/mous_whisper_word_alignments.jsonl"
META_MANIFEST="${OUT_DIR}/meta_manifest_mous.jsonl"
SUBJECT_ALLOWLIST="${OUT_DIR}/mous_subject_allowlist.txt"

mkdir -p "${OUT_DIR}"

###############################################################################
# Step 0: Build 96-subject auditory allowlist (BrainMagick-aligned)
###############################################################################
# Rule:
#   Subjects A2002–A2125 (auditory),
#   excluding recordings marked as bad in BrainMagick
###############################################################################

echo "== Step 0: Build 96-subject auditory allowlist (BrainMagick-aligned) =="

if [[ ! -f "${SUBJECT_ALLOWLIST}" ]]; then

  # Subject numbers skipped in Schoffelen2019Recording.iter()
  # (BrainMagick filtering, >=2000)
  bad_nums=(
    2011 2012 2018 2022 2023 2026 2036
    2043 2044 2045 2048
    2054 2060 2062 2063
    2074 2076
    2081 2082 2084 2087
    2093 2100 2107
    2112 2115 2118 2123
  )

  : > "${SUBJECT_ALLOWLIST}"

  for num in $(seq 2002 2125); do
    skip=0
    for bad in "${bad_nums[@]}"; do
      if [[ "${num}" -eq "${bad}" ]]; then
        skip=1
        break
      fi
    done
    [[ "${skip}" -eq 1 ]] && continue

    # IMPORTANT:
    # Write subject IDs as "A####" (no "sub-"),
    # consistent with subject_id used in meta manifests.
    printf "A%04d\n" "${num}" >> "${SUBJECT_ALLOWLIST}"
  done

  echo "[INFO] Allowlist written to: ${SUBJECT_ALLOWLIST}"
else
  echo "[INFO] Allowlist already exists: ${SUBJECT_ALLOWLIST}"
fi

echo "[INFO] Allowlist preview:"
head "${SUBJECT_ALLOWLIST}" || true
echo "[INFO] Allowlist size = $(wc -l < "${SUBJECT_ALLOWLIST}" | tr -d ' ')"

###############################################################################
# Stage 0: Whisper word-level alignment (MOUS)
###############################################################################

echo
echo "== Stage 0: Whisper word-level alignment (MOUS) =="

if [[ ! -f "${ALIGN_JSONL}" ]]; then
  python "${SCRIPT_DIR}/mous_whisper_align.py" \
    --mous_root "${BIDS_DIR}" \
    --out_dir "${OUT_DIR}" \
    --model large-v2 \
    --language nl
else
  echo "Alignment already exists:"
  echo "- ${ALIGN_JSONL}"
fi

###############################################################################
# Stage 1: Meta-manifest (MOUS + Whisper + 96-subject allowlist)
###############################################################################
# IMPORTANT:
# Subjects are filtered at manifest creation time,
# not post-hoc, ensuring strict 96-subject consistency.
###############################################################################

echo
echo "== Stage 1: Create meta-manifest (MOUS, 96 subjects only) =="

python "${SCRIPT_DIR}/create_meta_manifest_mous.py" \
  --mous_root "${BIDS_DIR}" \
  --align_jsonl "${ALIGN_JSONL}" \
  --out_dir "${OUT_DIR}" \
  --subject_allowlist "${SUBJECT_ALLOWLIST}"

echo "Meta-manifest written:"
echo "- ${META_MANIFEST}"

echo "Subject statistics in meta-manifest:"
python - <<PY
import json
from collections import Counter
from pathlib import Path

meta = Path("${META_MANIFEST}")
subs = Counter()

with meta.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sid = r.get("subject_id") or r.get("subject")
        subs[sid] += 1

print("Total subjects:", len(subs))
for sid, n in sorted(subs.items(), key=lambda x: -x[1])[:10]:
    print(f"  {sid}: {n}")
PY

###############################################################################
# Stage 2: Anchor-window splits (MOUS, sentence-level, 96 subjects)
###############################################################################

echo
echo "== Stage 2: Create anchor-window splits (MOUS, sentence-only) =="

for DATA_TYPE in sentence; do
  SPLIT_DIR="${OUT_DIR}/final_splits_${DATA_TYPE}"

  python "${SCRIPT_DIR}/create_local_global_splits.py" \
    --meta_manifest_path "${META_MANIFEST}" \
    --output_dir "${OUT_DIR}" \
    --bids_root_dir "${BIDS_DIR}" \
    --data_type "${DATA_TYPE}" \
    --split_ratios "0.7,0.1,0.2" \
    --random_seed 42 \
    --disable_cross_overlap_pruning \
    --subject_allowlist "${SUBJECT_ALLOWLIST}"

  echo
  echo "== Candidate pool statistics (MOUS, ${DATA_TYPE}) =="
  python "${SCRIPT_DIR}/report_candidate_pool.py" \
    --dir "${SPLIT_DIR}"

  echo
  echo "== Subject coverage check (MOUS, ${DATA_TYPE}) =="
  python - <<PY
import json
from pathlib import Path
from collections import Counter

base = Path("${SPLIT_DIR}")
subs = Counter()

for split in ("train", "valid", "test"):
    p = base / f"{split}.jsonl"
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("subject_id") or r.get("subject")
            subs[sid] += 1

print("Subjects in train+valid+test:", len(subs))
for sid, n in sorted(subs.items(), key=lambda x: -x[1])[:10]:
    print(f"  {sid}: {n}")
PY
done

###############################################################################
# Final report
###############################################################################

echo
echo "MOUS Stage 0–2 pipeline completed successfully ✅"
echo "From Stage 1 onward, all data are restricted to 96 auditory subjects."
