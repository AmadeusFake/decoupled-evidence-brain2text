#!/bin/bash
# test_meg_encoder_real.sh
# 用真实数据测试 UltimateMEGEncoder 的多种配置（T=360 对齐论文）

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 修改为你的 fully-preprocessed 清单路径（Stage-3 产物）
MANIFEST="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/data_manifests_local_global/final_splits_sentence_fully_preprocessed/train.jsonl"

# 可选：如果你的项目根目录不是当前目录，设置 PYTHONPATH 以便找到 models/
PROJECT_DIR="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

PYTHON_EXEC=python

CONFIGS=(
    "cnn False early"
    "cnn False late"
    "cnn True early"
    "cnn True late"
    "conformer False early"
    "conformer False late"
    "conformer True early"
    "conformer True late"
)

for cfg in "${CONFIGS[@]}"; do
    read backbone use_context subject_layer_pos <<< "$cfg"
    echo "== 测试: backbone=${backbone}, use_context=${use_context}, subject_layer_pos=${subject_layer_pos} =="

    $PYTHON_EXEC - <<PYCODE

import json
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

# 与项目结构一致：models/meg_encoder.py
from models.meg_encoder import UltimateMEGEncoder

manifest_path = Path("${MANIFEST}")
with open(manifest_path) as f:
    entries = [json.loads(line) for line in f if line.strip()]

# 取一个小 batch
B = 4
batch_entries = entries[:B]

# 准备张量
meg_win_list, sensor_locs_list, subj_idx_list = [], [], []

# subject id → int
subj2idx = {}
for e in batch_entries:
    sid = e["subject_id"]
    if sid not in subj2idx:
        subj2idx[sid] = len(subj2idx)

for e in batch_entries:
    mw = np.load(e["meg_win_path"]).astype(np.float32)         # [C,T] (T=360)
    locs = np.load(e["sensor_coordinates_path"]).astype(np.float32)  # [C,3]
    meg_win_list.append(torch.from_numpy(mw))
    sensor_locs_list.append(torch.from_numpy(locs))
    subj_idx_list.append(subj2idx[e["subject_id"]])

meg_win = torch.stack(meg_win_list, dim=0)       # [B,C,360]
sensor_locs = torch.stack(sensor_locs_list, 0)   # [B,C,3]
subj_idx = torch.tensor(subj_idx_list, dtype=torch.long)

# 由 bash 传入的配置
backbone = "${backbone}"
use_context = True if "${use_context}" == "True" else False
subject_layer_pos = "${subject_layer_pos}"

# 这里设置 out_timesteps=None —— 不池化，保持 T=360
model = UltimateMEGEncoder(
    in_channels=meg_win.shape[1],
    n_subjects=len(subj2idx),
    backbone_type=backbone,
    subject_layer_pos=subject_layer_pos,
    # 兼容旧开关：use_context=True 时走 window→CLS；否则 'none'
    context_mode=("window" if use_context else "none"),
    global_context_type="cls",
    out_timesteps=None  # 关键！输出保持输入时间长度（360）
)

if use_context:
    # 用 sentence_id 分组构造 meg_sent（窗口序列）
    sent_groups = defaultdict(list)
    for e in batch_entries:
        sent_groups[e["sentence_id"]].append(e)

    meg_sent_list, mask_list = [], []
    for e in batch_entries:
        group = sent_groups[e["sentence_id"]]
        wins = [torch.from_numpy(np.load(x["meg_win_path"]).astype(np.float32)) for x in group]  # [C,360]
        meg_sent_list.append(torch.stack(wins, dim=0))                                           # [S,C,360]
        mask_list.append(torch.zeros(len(group), dtype=torch.bool))

    # pad 到最大长度
    max_S = max(m.shape[0] for m in meg_sent_list)
    C, T = meg_sent_list[0].shape[1:]
    padded_sent, mask = [], []
    for m, mk in zip(meg_sent_list, mask_list):
        pad_len = max_S - m.shape[0]
        if pad_len > 0:
            m = torch.cat([m, torch.zeros(pad_len, C, T)], dim=0)
            mk = torch.cat([mk, torch.ones(pad_len, dtype=torch.bool)], dim=0)
        padded_sent.append(m); mask.append(mk)
    meg_sent = torch.stack(padded_sent, dim=0)  # [B,S,C,360]
    meg_sent_mask = torch.stack(mask, dim=0)    # [B,S]

    out = model(meg_win, sensor_locs, subj_idx, meg_sent=meg_sent, meg_sent_mask=meg_sent_mask)
else:
    out = model(meg_win, sensor_locs, subj_idx)

print("输出 shape:", tuple(out.shape))
# 断言 T=360
assert out.shape[0] == B and out.shape[1] == 1024 and out.shape[2] == 360, f"Expect [B,1024,360], got {tuple(out.shape)}"
PYCODE

done

echo "✅ 所有配置（T=360）运行成功"
