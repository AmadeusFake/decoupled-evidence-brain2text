#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read sentence texts from QC outputs (qc_{split}.tsv + {split}/audio/{sentence_id}.txt),
encode them with E5, and write sentence indices and .npy embeddings.

This avoids key mismatch or alignment issues with original text sources.
"""

import argparse, json, logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("stageT_from_qc_txt_e5")


# -------------------- QC I/O -------------------- #

def read_qc_tsv(p: Path) -> List[Dict]:
    rows = []
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as f:
        header = None
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if i == 0:
                header = line.split("\t")
                continue
            parts = line.split("\t")
            if header and len(parts) >= len(header):
                rows.append(dict(zip(header, parts)))
    return rows


# -------------------- E5 encoding -------------------- #

@torch.inference_mode()
def e5_encode(
    texts: List[str],
    tok,
    model,
    device: str = "cuda",
    bs: int = 128,
    prefix: str = "passage: ",
):
    """
    Encode texts with E5 using mean pooling + L2 normalization.
    """
    embs = []
    for i in tqdm(range(0, len(texts), bs), desc="E5 encode"):
        batch = [prefix + (t or "") for t in texts[i : i + bs]]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)

        # Mean pooling over valid tokens (more robust than CLS-only)
        last_hidden = out.last_hidden_state  # [B, T, D]
        attn = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        summed = (last_hidden * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1)
        mean = summed / counts
        mean = torch.nn.functional.normalize(mean, p=2, dim=1)

        embs.append(mean.detach().float().cpu().numpy())

    return (
        np.concatenate(embs, axis=0)
        if embs
        else np.zeros((0, 1024), dtype=np.float32)
    )


# -------------------- main -------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qc_dir",
        required=True,
        help="Directory containing qc_{split}.tsv and {split}/audio/*.txt",
    )
    ap.add_argument(
        "--out_index_dir",
        required=True,
        help="Output directory for sentence index jsonl files",
    )
    ap.add_argument(
        "--out_feat_dir",
        required=True,
        help="Output directory for sentence embedding .npy files",
    )
    ap.add_argument("--e5_model", default="intfloat/e5-large-v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    qc_root = Path(args.qc_dir)
    out_index = Path(args.out_index_dir); out_index.mkdir(parents=True, exist_ok=True)
    out_feat  = Path(args.out_feat_dir);  out_feat.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.e5_model)
    model = AutoModel.from_pretrained(args.e5_model).to(args.device).eval()
    D = getattr(model.config, "hidden_size", 1024)

    for sp in ("train", "valid", "test"):
        tsv = qc_root / f"qc_{sp}.tsv"
        rows = read_qc_tsv(tsv)
        if not rows:
            log.warning(f"[{sp}] missing QC TSV: {tsv}")
            continue

        sid_list, txt_list, txt_paths = [], [], []
        for r in rows:
            sid = r.get("sentence_id", "")
            txt_rel = r.get("txt_rel", "")
            if not sid or not txt_rel:
                continue
            tp = qc_root / txt_rel
            if not tp.exists():
                continue
            text = open(tp, "r", encoding="utf-8").read().strip()
            sid_list.append(sid)
            txt_list.append(text)
            txt_paths.append(tp)

        if not sid_list:
            log.warning(f"[{sp}] no valid text found in QC TSV")
            continue

        emb = e5_encode(
            txt_list,
            tok,
            model,
            device=args.device,
            bs=args.batch_size,
            prefix="passage: ",
        )
        assert emb.shape[0] == len(sid_list) and emb.shape[1] == D

        # Save per-sentence embeddings
        npy_paths = []
        for sid, vec in zip(sid_list, emb):
            npy_p = out_feat / f"{sid}_E5largeV2_text.npy"
            np.save(npy_p.as_posix(), vec.astype(np.float16))
            npy_paths.append(npy_p.as_posix())

        # Write sentence index
        outp = out_index / f"{sp}.jsonl"
        with open(outp, "w", encoding="utf-8") as f:
            for sid, tp, npyp in zip(sid_list, txt_paths, npy_paths):
                rec = {
                    "sentence_id": sid,
                    "text_used": open(tp, "r", encoding="utf-8").read().strip(),
                    "text_sentence_feature_path": npyp,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        log.info(f"[{sp}] wrote -> {outp.as_posix()} | vecs={len(sid_list)} D={D}")

    log.info("Done.")


if __name__ == "__main__":
    main()
