#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_spatial_topomap.py

Extract average SpatialAttention weights over real sensors from a trained
UltimateMEGEncoder and visualize them as an MNE topomap.

Key features
------------
- Reuses model_cfg / enc_cfg from runs/<run_dir>/records/config.json
- Reuses the standard best-checkpoint resolution logic
- Given a test manifest, automatically loads sensor_coordinates_path
  from the first entry
- --meg_encoder flag to select encoder backbone ("dense" or "exp"),
  following the same import/factory pattern as retrieval_gcb_decode.py.

Example usage
-------------
(DenseCNN run)
    python -m tools.plot_spatial_topomap \
        --run_dir /path/to/run_dir \
        --use_best_ckpt \
        --test_manifest /path/to/test_manifest.jsonl \
        --meg_encoder dense \
        --output figs/topomap_densecnn.png

(ExpDilated run)
    python -m tools.plot_spatial_topomap \
        --run_dir /path/to/run_dir \
        --use_best_ckpt \
        --test_manifest /path/to/test_manifest.jsonl \
        --meg_encoder exp \
        --output figs/topomap_expdilated.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import mne

def get_meg_encoder_class(name: str):
    """
    Factory for selecting MEG encoder backbone.

    Parameters
    ----------
    name : {"dense", "exp"}
        Encoder variant.

    Returns
    -------
    Encoder class (not instance).
    """
    name = name.lower()
    if name == "dense":
        from models.meg_encoder_Dense import UltimateMEGEncoder
        return UltimateMEGEncoder
    elif name == "exp":
        from models.meg_encoder_ExpDilated import UltimateMEGEncoder
        return UltimateMEGEncoder
    else:
        raise ValueError(f"Unknown meg_encoder: {name}")


# ======= Utility functions (aligned with retrieval_window_vote.py) ======= #
def log(msg: str):
    print(msg, flush=True)


def read_jsonl(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    """
    Load model_cfg / enc_cfg from run_dir/records/config.json.

    This logic is copied verbatim from retrieval_window_vote.py to ensure
    consistency between evaluation and visualization.
    """
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        cfg = json.loads(rec.read_text(encoding="utf-8"))
        return cfg.get("model_cfg", {}) or cfg.get("enc_cfg", {}) or {}
    return {}


def choose_ckpt_path(run_dir: Path, ckpt_path: str, use_best_ckpt: bool) -> Path:
    """
    Resolve checkpoint path using the same logic as retrieval_window_vote.py,
    with arguments passed explicitly instead of via argparse namespace.
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


def _read_logit_scale_exp(ckpt_path: Path) -> Optional[float]:
    """
    Keep the same interface as evaluation scripts.

    Although logit_scale is not used here, this preserves compatibility
    with shared utilities and checkpoint formats.
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


def load_model_from_ckpt(
    ckpt_path: Path,
    run_dir: Path,
    device: str,
    meg_encoder: str,
):
    """
    Construct UltimateMEGEncoder (selected by --meg_encoder) and load weights
    from checkpoint.

    Implementation follows retrieval_window_vote.py to ensure that
    the exact same model definition is used for visualization.
    """
    model_cfg = load_cfg_from_records(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", {}) or hp.get("enc_cfg", {})
    assert model_cfg, "no model_cfg/enc_cfg found in records or ckpt.hyper_parameters"

    EncoderCls = get_meg_encoder_class(meg_encoder)

    # Disable temporal pooling for evaluation-style usage
    if "out_timesteps" in EncoderCls.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None

    log(f"[INFO] meg_encoder = {meg_encoder}")
    log(f"[INFO] Model config keys: {list(model_cfg.keys())}")

    model = EncoderCls(**model_cfg)
    state = ckpt.get("state_dict", ckpt)

    # Strip 'model.' prefix if present
    new_state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        log(f"[WARN] Missing keys: {len(missing)} (e.g., {missing[:8]})")
    if unexpected:
        log(f"[WARN] Unexpected keys: {len(unexpected)} (e.g., {unexpected[:8]})")

    model.eval().to(device)
    meta = {"logit_scale_exp": _read_logit_scale_exp(ckpt_path)}
    return model, meta


# ======= Compute SpatialAttention channel weights ======= #
def compute_sensor_weights_from_spatial(spatial_module, sensor_coords: np.ndarray) -> np.ndarray:
    """
    Compute average SpatialAttention weights for each real sensor.

    Parameters
    ----------
    spatial_module:
        The SpatialAttention module (model.spatial).
    sensor_coords:
        Array of shape [C, 3] or [C, 2] loaded from sensor_coordinates_path.npy.

    Returns
    -------
    sensor_w:
        Array of shape [C], average attention weight per sensor.
    """
    spatial_module.eval()
    device = next(spatial_module.parameters()).device

    coords = sensor_coords.astype("float32")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Unexpected sensor_coords shape: {coords.shape}")

    coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)  # [1, C, 2/3]

    with torch.no_grad():
        xy = coords_t[..., :2]
        pos_feat = spatial_module.fourier(xy)       # [1, C, pos_dim]
        q = spatial_module.query(pos_feat)          # [1, C, S]
        attn = torch.softmax(q, dim=1)[0]           # [C, S]
        attn = torch.nan_to_num(attn)
        sensor_w = attn.mean(dim=1).cpu().numpy()   # [C]

    return sensor_w


# ======= Plot topomap with MNE ======= #
def plot_topomap_from_weights(
    sensor_w: np.ndarray,
    sensor_coords: np.ndarray,
    title: str,
    output_path: Path,
    sphere_radius: float = 0.5,
):
    """
    Gwilliams-style topomap visualization.

    The color field is allowed to slightly exceed the head outline so that
    the interpolation forms a smooth, continuous heatmap, with the head
    outline shown as a reference.
    """
    assert sensor_w.shape[0] == sensor_coords.shape[0], \
        f"weights {sensor_w.shape[0]} vs coords {sensor_coords.shape[0]} mismatch"

    # ---------- 1. 2D sensor positions ----------
    pos = sensor_coords[:, :2].astype(float).copy()

    # Center positions
    pos -= pos.mean(axis=0, keepdims=True)

    # Current radius
    r = np.sqrt((pos ** 2).sum(axis=1))
    max_r = np.max(r) if np.max(r) > 0 else 1.0

    # Slightly expand beyond head radius for smoother interpolation
    target_radius = sphere_radius * 1.05
    scale = target_radius / max_r
    pos *= scale

    # ---------- 2. Normalize weights symmetrically to [-1, 1] ----------
    w = np.nan_to_num(sensor_w.astype(float))

    # Z-score normalization
    w = (w - np.mean(w)) / (np.std(w) + 1e-6)

    # Symmetric clipping (85th percentile) to avoid extreme edge artifacts
    a = np.percentile(np.abs(w), 85)
    a = a if a > 0 else 1.0
    w = np.clip(w, -a, a) / a

    # ---------- 3. Plot ----------
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
        extrapolate="head",
        image_interp="linear",
        show=False,
    )

    ax.set_title(title, fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.set_ylabel("relative spatial weight", fontsize=8)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["low", "0", "high"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG + PDF (save before close)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

    plt.close(fig)
    print(f"[INFO] Saved topomap to: {output_path}")
    print(f"[INFO] Saved topomap to: {pdf_path}")


# ======= CLI entry point ======= #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True,
                   help="Training run directory containing records/config.json")
    p.add_argument("--ckpt_path", type=str, default="",
                   help="Optional explicit checkpoint path (ignored if --use_best_ckpt)")
    p.add_argument("--use_best_ckpt", action="store_true",
                   help="Load checkpoint from records/best_checkpoint.txt")
    p.add_argument("--test_manifest", type=str, required=True,
                   help="Manifest containing sensor_coordinates_path (typically test_manifest)")
    p.add_argument("--device", type=str, default="cuda",
                   help="cuda or cpu")
    p.add_argument("--spatial_attr", type=str, default="spatial",
                   help="Attribute name of SpatialAttention module in the model")
    p.add_argument("--output", type=str, default="figs/spatial_topomap.png",
                   help="Output image path (relative to run_dir)")
    p.add_argument("--title", type=str, default="SpatialAttention sensor weights",
                   help="Title shown on the topomap")
    p.add_argument(
        "--meg_encoder",
        default="exp",                    # keep old behavior of this script
        choices=["dense", "exp"],
        help="MEG encoder backbone: dense (UltimateMEGEncoder) or exp (ExpDilated)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    ckpt_path = choose_ckpt_path(run_dir, args.ckpt_path, args.use_best_ckpt)

    # 1) Load model (UltimateMEGEncoder selected by --meg_encoder)
    model, meta = load_model_from_ckpt(
        ckpt_path,
        run_dir,
        device=device,
        meg_encoder=args.meg_encoder,
    )
    log(f"[INFO] Loaded model from {ckpt_path}")
    if meta.get("logit_scale_exp") is not None:
        log(f"[INFO] exp(logit_scale) = {meta['logit_scale_exp']:.6f}")

    # 2) Extract SpatialAttention module
    spatial_module = getattr(model, args.spatial_attr, None)
    assert spatial_module is not None, \
        f"Model has no attribute '{args.spatial_attr}'. Available: {dir(model)}"
    log(f"[INFO] Using SpatialAttention module: model.{args.spatial_attr}")

    # 3) Load sensor coordinates from test manifest
    manifest_path = Path(args.test_manifest)
    assert manifest_path.exists(), f"test_manifest not found: {manifest_path}"
    rows = read_jsonl(manifest_path)
    assert rows, f"Empty manifest: {manifest_path}"
    coord_path = rows[0].get("sensor_coordinates_path", None)
    assert coord_path, "sensor_coordinates_path missing in first manifest entry"
    coord_path = Path(coord_path)
    if not coord_path.is_absolute():
        coord_path = (manifest_path.parent / coord_path).resolve()
    assert coord_path.exists(), f"sensor_coordinates_path not found: {coord_path}"
    log(f"[INFO] Using sensor coordinates: {coord_path}")

    sensor_coords = np.load(coord_path)
    log(f"[INFO] sensor_coords shape = {sensor_coords.shape}")

    # 4) Compute per-channel spatial weights
    sensor_w = compute_sensor_weights_from_spatial(spatial_module, sensor_coords)
    log(f"[INFO] sensor_w shape = {sensor_w.shape}")

    # 5) Plot topomap
    out_path = (run_dir / args.output).resolve()
    plot_topomap_from_weights(
        sensor_w,
        sensor_coords,
        title=args.title,
        output_path=out_path,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
