#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, argparse, numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F

# ==== Sentence-alias utilities ported from your evaluation script ====
_CAND_SENT_KEYS = ["sentence_id","sentence_uid","utt_id","utterance_id","segment_id",
                   "original_sentence_id","sentence_path","sentence_audio_path","transcript_path"]

def _round3(x):
    try: return f"{float(x):.3f}"
    except: return None

def sentence_aliases(row: dict):
    aliases = []
    for k in _CAND_SENT_KEYS:
        v = row.get(k)
        if v not in (None, ""): aliases.append((f"k:{k}", str(v)))
    a = str(row.get("original_audio_path", "") or row.get("sentence_audio_path", "") or row.get("audio_path", ""))
    so = row.get("global_segment_onset_in_audio_s", row.get("original_sentence_onset_in_audio_s", None))
    eo = row.get("global_segment_offset_in_audio_s", row.get("original_sentence_offset_in_audio_s", None))
    if a and so is not None and eo is not None:
        so3, eo3 = _round3(so), _round3(eo)
        if so3 and eo3: aliases.append(("audio+sent", f"{a}::{so3}-{eo3}"))
    if a: aliases.append(("audio", a))
    return aliases

def content_id_of(r: dict) -> str:
    if r.get("content_id"): return r["content_id"]
    a = r["original_audio_path"]
    s0 = float(r["local_window_onset_in_audio_s"])
    s1 = float(r["local_window_offset_in_audio_s"])
    return f"{a}::{s0:.3f}-{s1:.3f}"

def build_sentence_index_with_alias(candidate_rows):
    canon2idx, alias2idx, cand_sent_idx = {}, {}, []
    for r in candidate_rows:
        als = sentence_aliases(r)
        if not als: cand_sent_idx.append(-1); continue
        canon = als[0]
        if canon not in canon2idx: canon2idx[canon] = len(canon2idx)
        sidx = canon2idx[canon]
        for a in als:
            if a not in alias2idx: alias2idx[a] = sidx
        cand_sent_idx.append(sidx)
    return canon2idx, alias2idx, cand_sent_idx

def lookup_sent_idx(row, alias2idx):
    for a in sentence_aliases(row):
        if a in alias2idx: return alias2idx[a]
    return None

# ==== Aggregators ====
def sent_len_norm(n, mode="sqrt"):
    if n<=0 or mode=="none": return 1.0
    if mode=="count": return 1.0/float(n)
    if mode=="sqrt":  return 1.0/float(np.sqrt(n))
    if mode=="log":   return 1.0/float(np.log2(n+1.0))
    return 1.0

def aggregate(vals: torch.Tensor, agg: str, top_m: int):
    m = min(top_m, int(vals.numel()))
    vals = torch.topk(vals, k=m, largest=True, sorted=False).values
    if agg == "max": return torch.max(vals)
    if agg == "logsumexp":
        m0 = torch.max(vals)
        return m0 + torch.log(torch.clamp(torch.exp(vals - m0).sum(), min=1e-8))
    return torch.mean(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--preds_jsonl",  required=True)  # from baseline (no vote/gcb/qccp)
    ap.add_argument("--q", type=float, default=0.8)
    ap.add_argument("--agg", type=str, default="mean", choices=["mean","max","logsumexp"])
    ap.add_argument("--top_m", type=int, default=3)
    ap.add_argument("--norm", type=str, default="sqrt", choices=["none","count","sqrt","log"])
    ap.add_argument("--topS", type=int, default=3)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # Read test rows and construct candidate pool (unique windows + representative rows)
    rows = [json.loads(l) for l in open(args.test_manifest, "r", encoding="utf-8") if l.strip()]
    uniq, rep_rows = {}, []
    for r in rows:
        cid = content_id_of(r)
        if cid in uniq: continue
        uniq[cid] = 1; rep_rows.append(r)
    canon2idx, alias2idx, cand_sent_idx = build_sentence_index_with_alias(rep_rows)

    # content_id -> sent_idx
    cid2s = {}
    for rr, s in zip(rep_rows, cand_sent_idx):
        cid2s[content_id_of(rr)] = s

    # query_index -> GT sentence idx
    qidx2s = []
    for i, r in enumerate(rows):
        sidx = lookup_sent_idx(r, alias2idx)
        qidx2s.append(sidx if sidx is not None else -1)

    # Read baseline topK windows and scores
    # Each record: {"query_index": int, "gt_rank":..., "gt_cid":..., "pred_cids":[...], "pred_scores":[...]}
    ok, tot = 0, 0
    gt_win_rate, margins, betas = [], [], []
    miss_map = 0

    for line in open(args.preds_jsonl, "r", encoding="utf-8"):
        rec = json.loads(line)
        q = int(rec["query_index"])
        gt_s = qidx2s[q]
        if gt_s is None or gt_s < 0:
            miss_map += 1
            continue

        cids = rec["pred_cids"]; scores = torch.tensor(rec["pred_scores"], dtype=torch.float32)
        if scores.numel() == 0: continue

        thr = torch.quantile(scores, q=args.q)
        keep = scores >= thr
        cids = [c for c,k in zip(cids, keep.tolist()) if k]
        vals = scores[keep]

        # Aggregate to sentence level
        sent2vals = defaultdict(list)
        for c, v in zip(cids, vals):
            s = cid2s.get(c, -1)
            if s >= 0: sent2vals[s].append(v)
        if not sent2vals:
            continue

        sent_ids, supports, sizes = [], [], []
        for s, vs in sent2vals.items():
            v = torch.stack(vs)
            sup = aggregate(v, agg=args.agg, top_m=args.top_m)
            sup = sup * sent_len_norm(len(vs), mode=args.norm)
            sent_ids.append(s); supports.append(sup); sizes.append(len(vs))
        sent_ids = torch.tensor(sent_ids, dtype=torch.long)
        supports = torch.stack(supports)

        # Top-S sentences and beta (entropy gating)
        kS = min(args.topS, supports.numel())
        topS_val, topS_idx = torch.topk(supports, k=kS, largest=True, sorted=True)
        p = F.softmax(topS_val, dim=0)
        ent = -(p * (p + 1e-8).log()).sum() / np.log(float(max(1,kS)))
        beta = float(1.0 - ent)  # same normalized form as your main code
        betas.append(beta)

        topS_sent = sent_ids[topS_idx]
        is_hit = (gt_s in topS_sent.tolist())
        ok += int(is_hit); tot += 1

        # Margin: GT sentence support vs best wrong sentence support
        gt_mask = (sent_ids == gt_s)
        sup_gt = supports[gt_mask].max().item() if gt_mask.any() else -1e9
        sup_wrong = supports[~gt_mask].max().item() if (~gt_mask).any() else -1e9
        margins.append(sup_gt - sup_wrong)

        # Record whether the GT window is within the selected sentence's top-1 (optional; omitted here)

    hit_ratio = ok / max(1, tot)
    print(f"[PROBE] Queries probed = {tot},  GT-in-TopS ratio = {hit_ratio:.4f},  "
          f"median margin(GT - best wrong) = {np.median(margins):.4f}, "
          f"beta(mean) = {np.mean(betas):.4f},  miss_map={miss_map}")

    if args.out:
        out_obj = dict(
            queries=tot,
            gt_in_topS=hit_ratio,
            margin_stats=dict(mean=float(np.mean(margins)), median=float(np.median(margins))),
            beta_mean=float(np.mean(betas)),
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out_obj, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
