#!/usr/bin/env bash
set -euo pipefail

# ====== 路径 ======
BIDS_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/MOUS_raw"
OUT_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/data_mous_local_global"

ALLOW="$OUT_DIR/mous_subject_allowlist.txt"
EXCL="$OUT_DIR/mous_subject_excluded.tsv"
COMMON="$OUT_DIR/common_head_meg_channels.txt"

mkdir -p "$OUT_DIR"

echo "== [MOUS RAW PRECHECK] BIDS_ROOT=$BIDS_ROOT"

# 1) 找到所有有 auditory MEG 的 sub-A*
mapfile -t SUBS < <(
  find "$BIDS_ROOT" -type d -path "*/meg/*task-auditory*_meg.ds" \
    | sed -E 's|.*/(sub-[^/]+)/.*|\1|' \
    | sort -u \
    | grep -E '^sub-A'
)

N_SUB_ALL=${#SUBS[@]}
echo "[INFO] auditory subjects (raw, sub-A*) = $N_SUB_ALL"
echo "[INFO] writing allowlist/excluded + common channels to $OUT_DIR"

# 2) excluded 报告表头
echo -e "subject\tn_ds\tn_chan\thead_meg_n\tstatus\treason" > "$EXCL"
> "$ALLOW"

tmpdir="$(mktemp -d)"
first=1

# 3) 每个 subject 做基础检查 + 抽 head-MEG 通道名
for sid in "${SUBS[@]}"; do
  meg_dir="$BIDS_ROOT/$sid/meg"

  # ds 数量
  n_ds=$(find "$meg_dir" -maxdepth 1 -type d -name "${sid}_task-auditory*_meg.ds" 2>/dev/null \
         | wc -l | tr -d ' ')
  # channel tsv 数量
  n_chan=$(find "$meg_dir" -maxdepth 1 -type f -name "${sid}_task-auditory*_channels.tsv" 2>/dev/null \
           | wc -l | tr -d ' ')

  status="OK"
  reason=""

  if [[ "$n_ds" -lt 1 ]]; then
    status="BAD"; reason="missing_ds"
  elif [[ "$n_chan" -lt 1 ]]; then
    status="BAD"; reason="missing_channels_tsv"
  fi

  head_meg_n="NA"

  if [[ "$status" == "OK" ]]; then
    # 找一个 channels.tsv
    cf=$(ls "$meg_dir/${sid}_task-auditory"_*"_channels.tsv" 2>/dev/null | sort | head -n 1 || true)
    if [[ -f "$cf" ]]; then
      # BIDS: type 列是第 2 列，name 列是第 1 列
      # head-MEG 定义：type 以 "meg" 开头且不是 "ref*"（refmag/refgrad）
      head_meg_n=$(
        awk -F'\t' 'NR>1{
          t=$2;
          if(t ~ /^meg/ && t !~ /^ref/) c++
        } END{print c+0}' "$cf"
      )

      # 记录这个 subject 的 head-MEG 通道名集合
      awk -F'\t' 'NR>1{
        t=$2;
        if(t ~ /^meg/ && t !~ /^ref/) print $1
      }' "$cf" | sort -u > "$tmpdir/$sid.ch"

      # 更新交集
      if [[ $first -eq 1 ]]; then
        cp "$tmpdir/$sid.ch" "$tmpdir/intersect.ch"
        first=0
      else
        comm -12 "$tmpdir/intersect.ch" "$tmpdir/$sid.ch" > "$tmpdir/intersect.new"
        mv "$tmpdir/intersect.new" "$tmpdir/intersect.ch"
      fi

      # 通过检查 -> 写入 allowlist
      echo "$sid" >> "$ALLOW"
    else
      status="BAD"; reason="channels_tsv_not_found_glob"
    fi
  fi

  echo -e "${sid}\t${n_ds}\t${n_chan}\t${head_meg_n}\t${status}\t${reason}" >> "$EXCL"
done

# 4) 汇总输出
ALLOW_N=$(wc -l < "$ALLOW" | tr -d ' ')
echo "[RESULT] allowlist subjects = $ALLOW_N  -> $ALLOW"
echo "[RESULT] excluded report     -> $EXCL"

if [[ -f "$tmpdir/intersect.ch" ]]; then
  cp "$tmpdir/intersect.ch" "$COMMON"
  COMMON_N=$(wc -l < "$COMMON" | tr -d ' ')
  echo "[RESULT] common head-MEG channels = $COMMON_N  -> $COMMON"
else
  echo "[WARN] no subjects passed; common channels not generated."
fi

echo "== DONE =="
