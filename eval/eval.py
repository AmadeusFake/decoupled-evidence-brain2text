#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script: full-pool retrieval with UNIQUE candidate pool
(+ optional memmap reuse) and context diagnostics.

Key properties:
- Candidate pool is deduplicated by content_id (typically O≈1.4k).
- Evaluation protocol matches training:
  default similarity = dot-product with candidate-side L2 only
  ("CLIP-style": candidate normalized, query not).
- Optional diagnostics:
  * --probe_ctx N
      Print context tensor shapes and fused-vs-local statistics
      for the first N batches.
  * --compare_without_ctx
      Run an additional no-context evaluation in the same script,
      producing two sets of metrics for direct comparison.
"""

from __future__ import annotations
import sys, os, json, argparse, logging, hashlib, warnings
from pathlib import Path as _P
from typing import Dict, Any, Optional, List, Tuple

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

# Ensure project root is importable
_PR = _P(__file__).resolve().parents[1]
if str(_PR) not in sys.path:
    sys.path.insert(0, str(_PR))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from train.meg_utils import MEGDataModule, SubjectRegistry, map_amp_to_precision
from train.train_new import MEGLitModule

# ---------------------------------------------------------------------
# Logging & warnings
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("eval_fullpool_unique")

warnings.filterwarnings(
    "once",
    message="The given NumPy array is not writable.*",
    category=UserWarning
)

# ---------------------------------------------------------------------
# Global constants (protocol-level)
# ---------------------------------------------------------------------

TARGET_T = 360              # target temporal length after interpolation
AUDIO_D  = 1024             # audio feature dimension
F_FEAT   = AUDIO_D * TARGET_T
EPS = 1e-8

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _load_cfg(run_dir: _P) -> Dict[str, Any]:
    """Load training-time config snapshot from run directory."""
    p = run_dir / "records" / "config.json"
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_ckpt(run_dir: _P,
                  user_ckpt: Optional[str],
                  use_best: bool) -> str:
    """Resolve checkpoint path (best / user-specified / last)."""
    if use_best:
        best = run_dir / "records" / "best_checkpoint.txt"
        assert best.exists(), f"best_checkpoint.txt not found: {best}"
        ckpt = best.read_text(encoding="utf-8").strip().splitlines()[0]
        assert _P(ckpt).exists(), f"Best ckpt path invalid: {ckpt}"
        return ckpt

    if user_ckpt:
        assert _P(user_ckpt).exists(), f"Checkpoint not found: {user_ckpt}"
        return user_ckpt

    last = run_dir / "checkpoints" / "last.ckpt"
    assert last.exists(), f"Cannot infer checkpoint (missing {last})"
    return last.as_posix()

def _load_subject_registry_from_run(run_dir: _P,
                                   cfg: Dict[str, Any]) -> SubjectRegistry:
    """Recover subject registry exactly as used during training."""
    snap = run_dir / "records" / "subject_mapping_snapshot.json"
    if snap.exists():
        with open(snap, "r", encoding="utf-8") as f:
            return SubjectRegistry.from_json(json.load(f))

    map_path = cfg.get("subject_mapping_path")
    if map_path and _P(map_path).exists():
        return SubjectRegistry.load(_P(map_path))

    raise FileNotFoundError("Subject mapping not found in run_dir/records.")

def _ensure_DxT(x: np.ndarray, D: int) -> np.ndarray:
    """Ensure audio array is shaped [D, T]."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D audio array, got shape {x.shape}")
    if x.shape[0] == D:
        return x
    if x.shape[1] == D:
        return x.T
    # heuristic fallback
    return x if abs(x.shape[0] - D) <= abs(x.shape[1] - D) else x.T

def _prep_candidate_vector(path: str,
                           sim_is_clip: bool) -> torch.Tensor:
    """
    Load and normalize a single candidate audio feature into
    a flattened [F_FEAT] vector.
    """
    a = np.load(path).astype(np.float32)
    a = _ensure_DxT(a, AUDIO_D)          # -> [D, T_i]
    ta = torch.from_numpy(a)

    if ta.size(1) != TARGET_T:
        ta = F.interpolate(
            ta.unsqueeze(0),
            size=TARGET_T,
            mode="linear",
            align_corners=False
        ).squeeze(0)

    if sim_is_clip:
        ta = ta / (ta.norm(dim=(0, 1), p=2) + EPS)

    return ta.reshape(-1).contiguous()

def _prep_gt_vec_from_batch_audio(batch_audio: torch.Tensor,
                                  idxs: torch.Tensor,
                                  sim_is_clip: bool) -> torch.Tensor:
    """
    Prepare ground-truth audio vectors corresponding to the batch queries.
    """
    aud = batch_audio[idxs]   # [B, D, T] or [B, T, D]

    if aud.dim() == 3 and aud.size(1) != AUDIO_D and aud.size(2) == AUDIO_D:
        aud = aud.transpose(1, 2).contiguous()

    if aud.size(2) != TARGET_T:
        aud = F.interpolate(
            aud, size=TARGET_T,
            mode="linear",
            align_corners=False
        )

    if sim_is_clip:
        aud = aud / (aud.norm(dim=(1, 2), p=2, keepdim=True) + EPS)

    return aud.reshape(aud.size(0), -1).contiguous()

def _content_id_of(row: dict) -> str:
    """
    Stable content identifier used for candidate deduplication.
    """
    if row.get("content_id"):
        return str(row["content_id"])

    a = str(row.get("original_audio_path", ""))
    s0 = float(row.get("local_window_onset_in_audio_s", 0.0))
    s1 = float(row.get("local_window_offset_in_audio_s", 0.0))
    return f"{a}::{s0:.3f}-{s1:.3f}"

def metrics_from_ranks(ranks: torch.Tensor,
                       ks=(1, 5, 10)) -> Dict[str, float]:
    """Standard retrieval metrics from 1-indexed ranks."""
    ranks = ranks.to(torch.float32)
    out = {f"top{k}": (ranks <= k).float().mean().item() for k in ks}
    out["mrr"] = (1.0 / ranks).mean().item()
    out["mean_rank"] = ranks.mean().item()
    return out

# ---------------------------------------------------------------------
# Unique candidate pool construction
# ---------------------------------------------------------------------

def build_unique_audio_pool(ds,
                            *,
                            sim_is_clip: bool,
                            build_memmap: str = "auto",
                            memmap_dir: Optional[str] = None
                           ) -> Tuple[
                               Optional[torch.Tensor],
                               List[str],
                               Optional[Tuple[np.memmap, dict]]
                           ]:
    """
    Build a unique candidate pool over content_id.

    Returns:
      - in-memory tensor bank (or None if memmap is used)
      - list of content_ids
      - optional (memmap_object, metadata)
    """
    rows = getattr(ds, "rows", None)
    assert rows and len(rows) > 0, "Dataset has no rows."

    uniq: Dict[str, str] = {}
    for r in rows:
        cid = _content_id_of(r)
        if cid not in uniq:
            p = str(r.get("audio_feature_path") or "")
            if not p or not _P(p).exists():
                raise FileNotFoundError(f"audio_feature_path missing: {p}")
            uniq[cid] = p

    cids = list(uniq.keys())
    paths = [uniq[c] for c in cids]
    O = len(cids)

    sig = hashlib.sha1("\n".join(cids).encode("utf-8")).hexdigest()[:8]
    logger.info(
        f"[POOL] unique candidates: O={O} | sig={sig} | "
        f"build_memmap={build_memmap} dir={memmap_dir or 'N/A'}"
    )

    use_mm = bool(memmap_dir) and build_memmap in ("auto", "always")
    if use_mm:
        os.makedirs(memmap_dir, exist_ok=True)
        dat_path = _P(memmap_dir) / "bank_f16.dat"
        idx_path = _P(memmap_dir) / "index.json"

        # Try reuse
        if build_memmap == "auto" and idx_path.exists():
            try:
                meta = json.load(open(idx_path, "r", encoding="utf-8"))
                shp = tuple(meta.get("shape", []))
                saved_cids = meta.get("cids", [])
                dat_file = _P(meta.get("dat", dat_path.as_posix()))
                if shp == (O, F_FEAT) and saved_cids == cids and dat_file.exists():
                    logger.info(f"[POOL] reuse memmap @ {idx_path}")
                    mm = np.memmap(
                        dat_file.as_posix(),
                        mode="r",
                        dtype=np.float16,
                        shape=shp
                    )
                    return None, cids, (mm, meta)
            except Exception as e:
                logger.warning(f"[POOL] memmap reuse failed -> rebuild ({e})")

        # Build memmap
        logger.info(f"[POOL] writing memmap -> {dat_path.name}")
        mm_w = np.memmap(
            dat_path.as_posix(),
            mode="w+",
            dtype=np.float16,
            shape=(O, F_FEAT)
        )
        for i, p in enumerate(tqdm(paths, desc="prep unique candidates", total=O)):
            v = _prep_candidate_vector(p, sim_is_clip=sim_is_clip).half()
            mm_w[i, :] = v.cpu().numpy()
        del mm_w

        meta = {
            "cids": cids,
            "paths": paths,
            "shape": [O, F_FEAT],
            "dtype": "float16",
            "dat": dat_path.as_posix(),
            "sig": sig,
        }
        json.dump(meta, open(idx_path, "w", encoding="utf-8"), indent=2)
        logger.info(f"[POOL] memmap saved: {dat_path.name} + {idx_path.name}")

        mm = np.memmap(
            dat_path.as_posix(),
            mode="r",
            dtype=np.float16,
            shape=(O, F_FEAT)
        )
        return None, cids, (mm, meta)

    # In-memory fallback
    vecs = []
    for p in tqdm(paths, desc="prep unique candidates", total=O):
        v = _prep_candidate_vector(p, sim_is_clip=sim_is_clip).half()
        vecs.append(v)
    bank_cpu = torch.stack(vecs, dim=0).contiguous()
    logger.info(f"[POOL] in-memory bank: dtype={bank_cpu.dtype} shape={tuple(bank_cpu.shape)}")
    return bank_cpu, cids, None

# ---------------------------------------------------------------------
# Core evaluation (with optional probes)
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_fullpool_unique(lit: MEGLitModule,
                             dm: MEGDataModule,
                             *,
                             pool_device: str = "auto",
                             pool_chunk: int = 4096,
                             sim_is_clip: bool = True,
                             build_memmap: str = "auto",
                             memmap_dir: Optional[str] = None,
                             ks=(1, 5, 10),
                             save_ranks_path: Optional[str] = None,
                             probe_ctx_batches: int = 0
                             ) -> Dict[str, float]:
    """
    Full-pool retrieval evaluation against a unique candidate set.
    """
    if pool_device == "auto":
        pool_device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(pool_device)

    logger.info(
        f"[EVAL] pool_device={dev} | pool_chunk={pool_chunk} | "
        f"sim={'clip(cand-L2)' if sim_is_clip else 'dot'}"
    )
    logger.info(
        f"[EVAL] context_mode={getattr(dm, 'context_mode', 'none')} | "
        f"group_by_sentence={bool(getattr(dm, 'group_by_sentence', False))}"
    )

    # Dataloader (mirrors training-time context protocol)
    dm.enable_context_batch = dm.context_mode in ("window", "sentence")
    loader = dm.test_dataloader()

    # Candidate pool
    bank_cpu, cids, mm_info = build_unique_audio_pool(
        dm.test_set,
        sim_is_clip=sim_is_clip,
        build_memmap=build_memmap,
        memmap_dir=memmap_dir
    )
    O = len(cids)
    use_mm = mm_info is not None
    if use_mm:
        mm_obj, mm_meta = mm_info
        logger.info(f"[POOL] memmap <- {_P(mm_meta['dat']).name}")

    all_ranks: List[int] = []
    probed = 0

    for step, batch in enumerate(tqdm(loader, desc="eval", total=len(loader))):
        model_dev = next(lit.parameters()).device
        use_ctx = dm.context_mode != "none"

        # Selectively move tensors to model device
        move_keys = {"meg_win", "audio", "sensor_locs", "subject_idx"}
        if use_ctx:
            move_keys |= {
                "meg_sent", "meg_sent_mask", "meg_sent_relpos",
                "sent_unique_full", "sent_unique_mask", "sample2sent",
                "sent_proxy_sensor_locs", "sent_proxy_subj_idx",
                "win_pos_scalar",
            }

        batch_m = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and k in move_keys:
                batch_m[k] = v.to(model_dev, non_blocking=True)
            else:
                batch_m[k] = v

        # ---------------------------------------------------------
        # Context probe (diagnostics only)
        # ---------------------------------------------------------
        if probe_ctx_batches > 0 and probed < probe_ctx_batches:
            def _shape(x):
                return tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__

            ctx_keys = [
                "meg_sent", "meg_sent_mask", "meg_sent_relpos",
                "sent_unique_full", "sent_unique_mask", "sample2sent",
                "sent_proxy_sensor_locs", "sent_proxy_subj_idx",
                "win_pos_scalar",
            ]

            logger.info(f"[PROBE {probed}] Context tensors:")
            for k in ctx_keys:
                if k in batch_m:
                    v = batch_m[k]
                    if isinstance(v, torch.Tensor) and v.numel() == 0:
                        logger.info(f"  - {k:22s}: EMPTY")
                    else:
                        logger.info(f"  - {k:22s}: {_shape(v)}")
                else:
                    logger.info(f"  - {k:22s}: <absent>")

            y_out = lit.forward(batch_m, return_local=True)
            if isinstance(y_out, (tuple, list)) and len(y_out) == 2:
                y_fused, y_local = y_out
            else:
                y_fused, y_local = y_out, None

            if y_local is not None and torch.is_tensor(y_local):
                delta_rms = (y_fused - y_local).pow(2).mean().sqrt().item()
                fuser = getattr(getattr(lit, "model", None), "fuser", None)
                gate_logit = getattr(fuser, "last_gate_logit", None)
                gate_mean = float(gate_logit.mean().item()) if torch.is_tensor(gate_logit) else float("nan")
                logger.info(
                    f"[PROBE {probed}] fuse_delta_rms={delta_rms:.6f} | "
                    f"gate_logit_mean={gate_mean:.6f}"
                )
            else:
                logger.warning(f"[PROBE {probed}] local stream unavailable")

            probed += 1

        # ---------------------------------------------------------
        # Forward pass (queries)
        # ---------------------------------------------------------
        y = lit.forward(batch_m)                  # [B, D, T]
        B = y.size(0)
        q_flat = y.reshape(B, -1).to(dev, dtype=torch.float32)

        # Ground-truth audio scores
        gt_flat = _prep_gt_vec_from_batch_audio(
            batch_m["audio"],
            idxs=torch.arange(B, device=model_dev),
            sim_is_clip=sim_is_clip
        ).to(dev, dtype=torch.float32)

        s_gt = (q_flat * gt_flat).sum(dim=1)      # [B]

        # Count how many candidates beat the GT score
        greater_counts = torch.zeros(B, dtype=torch.long, device=dev)

        if use_mm:
            for start in range(0, O, pool_chunk):
                end = min(start + pool_chunk, O)
                cand_np = mm_obj[start:end, :]
                cand = torch.from_numpy(cand_np).to(dev, dtype=torch.float16).to(torch.float32)
                scores = q_flat @ cand.t()
                greater_counts += (scores > s_gt.unsqueeze(1)).sum(dim=1)
        else:
            for start in range(0, O, pool_chunk):
                end = min(start + pool_chunk, O)
                cand = bank_cpu[start:end].to(dev, dtype=torch.float16).to(torch.float32)
                scores = q_flat @ cand.t()
                greater_counts += (scores > s_gt.unsqueeze(1)).sum(dim=1)

        ranks = (greater_counts + 1).to(torch.long)
        all_ranks.extend(ranks.tolist())

    ranks_t = torch.tensor(all_ranks, dtype=torch.long)
    metrics = metrics_from_ranks(ranks_t, ks=ks)
    metrics["num_queries"] = int(ranks_t.numel())
    metrics["pool_size"] = int(O)
    return metrics

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--use_best_ckpt", action="store_true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--amp", default="off",
                   choices=["off", "bf16", "fp16", "16-mixed", "32"])
    p.add_argument("--topk", default="1,5,10")
    p.add_argument("--pool_chunk", type=int, default=4096)
    p.add_argument("--pool_device", default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--build_memmap", default="auto",
                   choices=["auto", "always", "never"])
    p.add_argument("--memmap_dir", default="")
    p.add_argument("--context_mode", default="",
                   choices=["", "none", "window", "sentence"])
    p.add_argument("--ctx_max_windows", type=int, default=-1)
    p.add_argument("--ctx_stride", type=int, default=-1)
    p.add_argument("--group_by_sentence", default="",
                   choices=["", "true", "false"])
    p.add_argument("--force_sim", default="auto",
                   choices=["auto", "clip", "dot"])
    p.add_argument("--save_json", default="")
    p.add_argument("--save_ranks", default="")
    p.add_argument("--probe_ctx", type=int, default=1,
                   help="Probe first N batches for context diagnostics")
    p.add_argument("--compare_without_ctx", action="store_true")
    return p.parse_args()

def _fmt_summary(tag: str, metrics: Dict[str, Any]) -> str:
    keys = ["top1", "top5", "top10", "mrr", "mean_rank"]
    core = " | ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
    extra = f" | queries={metrics.get('num_queries','?')} O={metrics.get('pool_size','?')}"
    return f"[RESULT {tag}] {core}{extra}"

# ---------------------------------------------------------------------
# DataModule construction (evaluation-time)
# ---------------------------------------------------------------------

def build_dm_for_eval(saved_args: dict,
                      registry: SubjectRegistry,
                      *,
                      override_ctx_mode: Optional[str],
                      ctx_max,
                      ctx_stride,
                      group_by_sentence_opt: Optional[str],
                      batch_size: int,
                      num_workers: int) -> MEGDataModule:
    """
    Reconstruct the DataModule used for evaluation,
    inheriting training-time configuration with optional overrides.
    """
    def pick(key, override, bad):
        if override == bad or override == "":
            return saved_args.get(key)
        return override

    ctx_mode = pick("context_mode", override_ctx_mode, "")
    if ctx_mode is None:
        ctx_mode = "none"

    ctx_max  = pick("ctx_max_windows", ctx_max, -1)
    ctx_strd = pick("ctx_stride", ctx_stride, -1)

    gbs_saved = bool(saved_args.get("group_by_sentence", False))
    if group_by_sentence_opt == "":
        gbs = gbs_saved
    else:
        gbs = (group_by_sentence_opt.lower() == "true")

    test_manifest = saved_args.get("test_manifest")
    ns_test = saved_args.get("subject_namespace_test", "")

    dm = MEGDataModule(
        train_manifest=test_manifest,
        val_manifest=test_manifest,
        test_manifest=test_manifest,
        registry=registry,
        ns_train=ns_test,
        ns_val=ns_test,
        ns_test=ns_test,
        batch_size=int(batch_size or saved_args.get("batch_size", 64)),
        num_workers=int(num_workers or saved_args.get("num_workers", 8)),
        normalize=False,
        context_mode=ctx_mode,
        ctx_max_windows=int(ctx_max) if ctx_max is not None else 0,
        ctx_stride=int(ctx_strd) if ctx_strd is not None else 1,
        group_by_sentence=bool(gbs),
        sentences_per_batch=int(saved_args.get("sentences_per_batch", 8)),
        windows_per_sentence=int(saved_args.get("windows_per_sentence", 4)),
        exclude_self_from_ctx=True,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=(int(num_workers or 0) > 0),
    )
    dm.setup(None)
    return dm

# ---------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = _P(args.run_dir)

    cfg = _load_cfg(run_dir)
    saved_args = cfg.get("args", {})
    model_cfg = cfg.get("model_cfg", {})
    loss_cfg  = cfg.get("loss_cfg", {})

    candidate_l2 = bool(loss_cfg.get("candidate_l2", True))

    registry = _load_subject_registry_from_run(run_dir, cfg)
    logger.info(f"Subject registry loaded (#subjects={registry.num_subjects})")

    # Precision / device setup
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    _ = map_amp_to_precision(args.amp)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and int(args.devices or 1) >= 1 else "cpu"
    )
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # DataModule (with optional context overrides)
    dm_ctx = build_dm_for_eval(
        saved_args, registry,
        override_ctx_mode=args.context_mode,
        ctx_max=args.ctx_max_windows,
        ctx_stride=args.ctx_stride,
        group_by_sentence_opt=args.group_by_sentence,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    lit = MEGLitModule(
        model_cfg=model_cfg,
        lr=1e-5,
        weight_decay=0.0,
        warmup_ratio=0.0,
        max_epochs=1,
        optimizer_name="adam",
        metric_logger=None,
        loss_use_l2=False,
        loss_use_temp=bool(loss_cfg.get("use_temperature", False)),
        metrics_every_n_steps=999999,
        context_mode=dm_ctx.context_mode,
        freeze_ctx_epochs=0,
        freeze_local_epochs=0,
        loss_candidate_l2=candidate_l2,
    ).to(device).eval()

    ckpt_path = _resolve_ckpt(run_dir, args.checkpoint, args.use_best_ckpt)
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    missing, unexpected = lit.load_state_dict(
        ckpt.get("state_dict", ckpt),
        strict=False
    )
    if missing:
        logger.warning(f"Missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    # Similarity protocol
    sim_is_clip = candidate_l2 if args.force_sim == "auto" else (args.force_sim == "clip")

    # ---------------------------------------------------------
    # Evaluation with context
    # ---------------------------------------------------------
    metrics_ctx = evaluate_fullpool_unique(
        lit,
        dm_ctx,
        pool_device=args.pool_device,
        pool_chunk=int(args.pool_chunk),
        sim_is_clip=sim_is_clip,
        build_memmap=args.build_memmap,
        memmap_dir=(args.memmap_dir or None),
        ks=tuple(int(x) for x in args.topk.split(",")),
        probe_ctx_batches=max(0, int(args.probe_ctx or 0)),
    )

    print(_fmt_summary(f"CTX={dm_ctx.context_mode}", metrics_ctx))

    save_json = args.save_json or (
        run_dir / "records" / f"eval_metrics_ctx-{dm_ctx.context_mode}_unique.json"
    ).as_posix()
    try:
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(
                {"metrics": metrics_ctx, "args": vars(args), "mode": f"ctx={dm_ctx.context_mode}"},
                f, indent=2
            )
        logger.info(f"[SAVE] metrics -> {save_json}")
    except Exception as e:
        logger.warning(f"[SAVE] metrics failed: {e}")

    # ---------------------------------------------------------
    # Optional comparison: no-context
    # ---------------------------------------------------------
    if args.compare_without_ctx:
        dm_none = build_dm_for_eval(
            saved_args, registry,
            override_ctx_mode="none",
            ctx_max=args.ctx_max_windows,
            ctx_stride=args.ctx_stride,
            group_by_sentence_opt=args.group_by_sentence,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        metrics_none = evaluate_fullpool_unique(
            lit,
            dm_none,
            pool_device=args.pool_device,
            pool_chunk=int(args.pool_chunk),
            sim_is_clip=sim_is_clip,
            build_memmap=args.build_memmap,
            memmap_dir=(args.memmap_dir or None),
            ks=tuple(int(x) for x in args.topk.split(",")),
            probe_ctx_batches=0,
        )

        print(_fmt_summary("CTX=none", metrics_none))

        save_json2 = (
            run_dir / "records" / "eval_metrics_ctx-none_unique.json"
        ).as_posix()
        try:
            with open(save_json2, "w", encoding="utf-8") as f:
                json.dump(
                    {"metrics": metrics_none, "args": vars(args), "mode": "ctx=none"},
                    f, indent=2
                )
            logger.info(f"[SAVE] metrics(no-ctx) -> {save_json2}")
        except Exception as e:
            logger.warning(f"[SAVE] metrics(no-ctx) failed: {e}")

if __name__ == "__main__":
    main()
