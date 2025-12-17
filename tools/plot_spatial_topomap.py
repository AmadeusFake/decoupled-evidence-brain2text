#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_spatial_topomap.py

从训练好的 UltimateMEGEncoder 里提取 SpatialAttention 对每个真实传感器的平均权重，
并用 MNE 画出一个 topomap（头皮图）。

特点：
- 直接复用 runs/<...>/records/config.json 里的 model_cfg / enc_cfg
- 直接复用 runs/<...>/records/best_checkpoint.txt 逻辑
- 传入 test_manifest，就能自动拿第一行的 sensor_coordinates_path 当坐标

用法示例（DenseCNN 那个 run）：

    python -m tools.plot_spatial_topomap \
        --run_dir /mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/runs/none_baseline_safe_EBS256_ampoff_tf32off_adam_wd0_e100_resplit812_20251110-134936_job5297996 \
        --use_best_ckpt \
        --test_manifest /mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project/data_mous_local_global/final/mous_test_manifest.jsonl \
        --output figs/topomap_densecnn.png

ExpCNN 那个 run 只改 run_dir 和 output 即可。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import mne

# ======= 和 retrieval_window_vote 里一致的工具函数 ======= #

def log(msg: str):
    print(msg, flush=True)

def read_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    """
    从 run_dir/records/config.json 里读取 model_cfg / enc_cfg。
    （直接复制自 retrieval_window_vote.py）
    """
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        cfg = json.loads(rec.read_text(encoding="utf-8"))
        return cfg.get("model_cfg", {}) or cfg.get("enc_cfg", {}) or {}
    return {}

def choose_ckpt_path(run_dir: Path, ckpt_path: str, use_best_ckpt: bool) -> Path:
    """
    和 retrieval_window_vote.py 的 choose_ckpt_path 一致，只是把 args 拆开了。
    """
    if use_best_ckpt:
        best_txt = run_dir / "records" / "best_checkpoint.txt"
        assert best_txt.exists(), f"best_checkpoint.txt not found: {best_txt}"
        ckpt = best_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (run_dir / ckpt_path).resolve()
        assert ckpt_path.exists(), f"best ckpt not found: {ckpt_path}"
        log(f"[INFO] Using BEST checkpoint: {ckpt_path}")
        return ckpt_path

    cp = Path(ckpt_path)
    assert cp.exists(), f"--ckpt_path not found: {cp}"
    return cp

from models.meg_encoder2 import UltimateMEGEncoder  # 和 eval 保持一致

def _read_logit_scale_exp(ckpt_path: Path) -> Optional[float]:
    """
    保留和 eval 一样的接口，虽然这里用不到 logit_scale，
    但不影响兼容性。
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)
    for k in ("model.scorer.logit_scale", "scorer.logit_scale", "logit_scale"):
        v = state.get(k, None)
        if v is not None:
            try:
                return float(torch.exp(v).item())
            except Exception:
                try:
                    return float(np.exp(float(v)))
                except Exception:
                    pass
    return None

def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str):
    """
    直接参考 retrieval_window_vote.py 里的实现，构造 UltimateMEGEncoder
    并加载 state_dict。
    """
    model_cfg = load_cfg_from_records(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", {}) or hp.get("enc_cfg", {})
    assert model_cfg, "no model_cfg/enc_cfg found in records or ckpt.hyper_parameters"

    # 检查是否支持 out_timesteps 参数（eval 端不做时间池化）
    if "out_timesteps" in UltimateMEGEncoder.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None

    log(f"[INFO] Model config keys: {list(model_cfg.keys())}")

    model = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    # 去掉 'model.' 前缀
    new_state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        log(f"[WARN] Missing keys: {len(missing)} (e.g., {missing[:8]})")
    if unexpected:
        log(f"[WARN] Unexpected keys: {len(unexpected)} (e.g., {unexpected[:8]})")

    model.eval().to(device)
    meta = {"logit_scale_exp": _read_logit_scale_exp(ckpt_path)}
    return model, meta

# ======= 计算 SpatialAttention 的 channel 权重 ======= #

def compute_sensor_weights_from_spatial(spatial_module, sensor_coords: np.ndarray) -> np.ndarray:
    """
    spatial_module: model.spatial（SpatialAttention 实例）
    sensor_coords:  [C, 3] 或 [C, 2]，来自 sensor_coordinates_path.npy

    返回：
        sensor_w: [C,]，每个真实传感器的平均权重
    """
    spatial_module.eval()
    device = next(spatial_module.parameters()).device

    coords = sensor_coords.astype("float32")
    if coords.ndim == 2 and coords.shape[1] >= 2:
        pass
    else:
        raise ValueError(f"Unexpected sensor_coords shape: {coords.shape}")
    coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)  # [1,C,2/3]

    with torch.no_grad():
        # SpatialAttention.forward 内部逻辑：
        # xy = sensor_locs[..., :2]
        # pos_feat = self.fourier(xy)
        # q = self.query(pos_feat)       # [B,C,S]
        if coords_t.size(-1) > 2:
            xy = coords_t[..., :2]
        else:
            xy = coords_t
        pos_feat = spatial_module.fourier(xy)       # [1,C,pos_dim]
        q = spatial_module.query(pos_feat)          # [1,C,S]

        # eval 模式下不做 dropout，所以这里不用 _make_mask
        attn = torch.softmax(q, dim=1)[0]           # [C,S]
        attn = torch.nan_to_num(attn)

        # 对虚拟通道维度取平均，得到每个真实通道的总体权重
        sensor_w = attn.mean(dim=1).cpu().numpy()   # [C,]

    return sensor_w

# ======= 用 MNE 画 topomap ======= #

def plot_topomap_from_weights(sensor_w: np.ndarray,
                              sensor_coords: np.ndarray,
                              title: str,
                              output_path: Path,
                              sphere_radius: float = 0.5):
    """
    Gwilliams 风格：颜色场可以稍微超出头型一圈，看起来是一整块“热力图”，
    头只是叠在上面做参照。
    """
    assert sensor_w.shape[0] == sensor_coords.shape[0], \
        f"weights {sensor_w.shape[0]} vs coords {sensor_coords.shape[0]} mismatch"

    # ---------- 1. 传感器二维坐标 ----------
    pos = sensor_coords[:, :2].astype(float).copy()

    # 居中
    pos -= pos.mean(axis=0, keepdims=True)

    # 当前半径
    r = np.sqrt((pos ** 2).sum(axis=1))
    max_r = np.max(r)
    if max_r == 0:
        max_r = 1.0

    # 关键：让最外围传感器略微“超出”头型半径一点点
    # 例如 1.05 * sphere_radius，这样插值会延伸到头圈外一点
    target_radius = sphere_radius * 1.05
    scale = target_radius / max_r
    pos *= scale

    # ---------- 2. 权重标准化（对称到 [-1, 1]，边缘柔和） ----------
    w = sensor_w.astype(float)
    w = np.nan_to_num(w)

    # z-score
    w = (w - np.mean(w)) / (np.std(w) + 1e-6)

    # 对称裁剪，稍微狠一点（85%），避免极端点在边缘炸花
    a = np.percentile(np.abs(w), 85)
    if a == 0:
        a = 1.0
    w = np.clip(w, -a, a) / a      # 映射到 [-1, 1]

    # ---------- 3. 画图 ----------
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    sphere = (0., 0., 0., sphere_radius)

    im, _ = mne.viz.plot_topomap(
        w,
        pos,
        axes=ax,
        outlines="head",
        sphere=sphere,
        cmap="RdBu_r",
        contours=0,
        extrapolate="head",     # ★ 允许在头部区域外推
        image_interp="linear",  # 比 cubic 温和，不那么花
        show=False,
    )

    ax.set_title(title, fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.set_ylabel("relative spatial weight", fontsize=8)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["low", "0", "high"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved topomap to: {output_path}")
    # 再顺手保存一个 PDF（给论文 / LaTeX 用）
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")

# ======= CLI 主流程 ======= #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="训练 run 目录（包含 records/config.json 的那个）")
    p.add_argument("--ckpt_path", type=str, default="",
                   help="可选：显式指定 ckpt；若配合 --use_best_ckpt，则忽略本项")
    p.add_argument("--use_best_ckpt", action="store_true",
                   help="若设置，则从 records/best_checkpoint.txt 读取 ckpt")
    p.add_argument("--test_manifest", type=str, required=True,
                   help="任意一个包含 sensor_coordinates_path 字段的 manifest（一般用 test_manifest）")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda 或 cpu")
    p.add_argument("--spatial_attr", type=str, default="spatial",
                   help="模型里 SpatialAttention 的属性名（默认 'spatial'）")
    p.add_argument("--output", type=str, default="figs/spatial_topomap.png",
                   help="输出图片路径（相对 run_dir）")
    p.add_argument("--title", type=str, default="SpatialAttention sensor weights",
                   help="topomap 标题")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt_path = choose_ckpt_path(run_dir, args.ckpt_path, args.use_best_ckpt)

    # 1) 加载模型（UltimateMEGEncoder）
    model, meta = load_model_from_ckpt(ckpt_path, run_dir, device=device)
    log(f"[INFO] Loaded model from {ckpt_path}")
    if meta.get("logit_scale_exp") is not None:
        log(f"[INFO] exp(logit_scale) = {meta['logit_scale_exp']:.6f}")

    # 2) 取 SpatialAttention 模块
    spatial_module = getattr(model, args.spatial_attr, None)
    assert spatial_module is not None, \
        f"Model has no attribute '{args.spatial_attr}'. Available: {dir(model)}"
    log(f"[INFO] Using SpatialAttention module: model.{args.spatial_attr}")

    # 3) 从 test_manifest 里拿一份 sensor_coordinates_path
    manifest_path = Path(args.test_manifest)
    assert manifest_path.exists(), f"test_manifest not found: {manifest_path}"
    rows = read_jsonl(manifest_path)
    assert rows, f"Empty manifest: {manifest_path}"
    coord_path = rows[0].get("sensor_coordinates_path", None)
    assert coord_path, f"sensor_coordinates_path missing in first row of {manifest_path}"
    coord_path = Path(coord_path)
    if not coord_path.is_absolute():
        # 如果 manifest 里存的是相对路径，就相对于它的父目录
        coord_path = (manifest_path.parent / coord_path).resolve()
    assert coord_path.exists(), f"sensor_coordinates_path not found: {coord_path}"
    log(f"[INFO] Using sensor coordinates: {coord_path}")

    sensor_coords = np.load(coord_path)  # [C,3] or [C,2]
    log(f"[INFO] sensor_coords shape = {sensor_coords.shape}")

    # 4) 计算每个 channel 的权重
    sensor_w = compute_sensor_weights_from_spatial(spatial_module, sensor_coords)
    log(f"[INFO] sensor_w shape = {sensor_w.shape}")

    # 5) 画 topomap
    out_path = (run_dir / args.output).resolve()
    plot_topomap_from_weights(sensor_w, sensor_coords, title=args.title, output_path=out_path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
