#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
retrieval_gcb_decode.py — Base vs GCB（无 QCCP/WindowVote），并输出“长句”文本样例
Pipeline: Encode → Similarity(base) → GCB(soft-consensus, across=sum; post)

新增：
- 句子级“解码”可视化：按 anchor_word_idx 逐词取 Top-1 候选所属窗口的词，拼回整句
- 自动挑选长句样例（improved / unchanged / worsened），可控数量和阈值
- 只保留 GCB（post），完全移除 QCCP 和 Window-Vote 的实现与参数

输出（默认在 runs/<run_dir>/results/retrieval_gcb_decode/）：
  - metrics.json                   # Base vs GCB 的全局 Recall@K/MRR
  - ranks.txt                      # 每个 query 的 rank（post）
  - examples_long_sentences.json   # 三类对比样例（长句）
  - 可选：preds_topkK.jsonl / .tsv  # 若 --save_topk > 0

需求字段（test_manifest 的每行）：
  - global_segment_text（整句文本）
  - anchor_word_idx（当前窗口对应的词序号，0..N-1）
  - original_audio_path, local_window_onset/offset_in_audio_s（用于 content_id 唯一化）
  - audio_feature_path, meg_win_path, sensor_coordinates_path, subject_id
"""

import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ------------------------- 常量 -------------------------
TARGET_T = 360
AUDIO_D = 1024
EPS = 1e-8

# ------------------------- 工具 -------------------------
def log(msg: str):
    print(msg, flush=True)

def read_jsonl(p: Path) -> List[dict]:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def content_id_of(r: dict) -> str:
    if r.get("content_id"):
        return r["content_id"]
    a = r["original_audio_path"]
    s0 = float(r["local_window_onset_in_audio_s"])
    s1 = float(r["local_window_offset_in_audio_s"])
    return f"{a}::{s0:.3f}-{s1:.3f}"

def ensure_audio_DxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D audio array, got {x.shape}")
    if x.shape[0] == AUDIO_D: return x
    if x.shape[1] == AUDIO_D: return x.T
    return x if abs(x.shape[0]-AUDIO_D) < abs(x.shape[1]-AUDIO_D) else x.T

def ensure_meg_CxT(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expect 2D MEG array, got {x.shape}")
    return x if x.shape[0] <= x.shape[1] else x.T

def maybe_interp_1DT(x: torch.Tensor, T: int) -> torch.Tensor:
    if x.size(1) == T: return x
    return F.interpolate(x.unsqueeze(0), size=T, mode="linear", align_corners=False).squeeze(0)

def window_centers(rows: List[dict], idxs: List[int]) -> torch.Tensor:
    t = []
    for i in idxs:
        r = rows[i]
        s0 = float(r.get("local_window_onset_in_audio_s", 0.0))
        s1 = float(r.get("local_window_offset_in_audio_s", s0))
        t.append(0.5*(s0+s1))
    return torch.tensor(t, dtype=torch.float32)

# ------------------------- 句子别名/索引 -------------------------
_CAND_SENT_KEYS = ["sentence_id","sentence_uid","utt_id","utterance_id","segment_id",
                   "original_sentence_id","sentence_path","sentence_audio_path","transcript_path"]

def _round3(x):
    try: return f"{float(x):.3f}"
    except Exception: return None

def sentence_aliases(row: dict):
    aliases = []
    for k in _CAND_SENT_KEYS:
        v = row.get(k)
        if v not in (None, ""): aliases.append((f"k:{k}", str(v)))
    a = str(row.get("original_audio_path", "") or row.get("sentence_audio_path", "") or row.get("audio_path", ""))
    so = (row.get("global_segment_onset_in_audio_s", None)
          if row.get("global_segment_onset_in_audio_s", None) is not None
          else row.get("original_sentence_onset_in_audio_s", None))
    eo = (row.get("global_segment_offset_in_audio_s", None)
          if row.get("global_segment_offset_in_audio_s", None) is not None
          else row.get("original_sentence_offset_in_audio_s", None))
    if a and so is not None and eo is not None:
        so3, eo3 = _round3(so), _round3(eo)
        if so3 and eo3: aliases.append(("audio+sent", f"{a}::{so3}-{eo3}"))
    if a: aliases.append(("audio", a))
    return aliases

def build_sentence_index_with_alias(candidate_rows: list):
    canon2idx, alias2idx, cand_sent_idx = {}, {}, []
    for r in candidate_rows:
        aliases = sentence_aliases(r)
        if not aliases: cand_sent_idx.append(-1); continue
        canon = aliases[0]
        if canon not in canon2idx: canon2idx[canon] = len(canon2idx)
        sidx = canon2idx[canon]
        for a in aliases:
            if a not in alias2idx: alias2idx[a] = sidx
        cand_sent_idx.append(sidx)
    return canon2idx, alias2idx, cand_sent_idx

# ------------------------- subject 映射 -------------------------
def _normalize_subject_key(x: Any) -> Optional[str]:
    if x is None: return None
    s = str(x); m = re.search(r"(\d+)", s)
    if not m: return None
    return f"{int(m.group(1)):02d}"

def read_subject_mapping_from_records(run_dir: Path) -> Dict[str, int]:
    p = run_dir / "records" / "subject_mapping.json"
    assert p.exists(), f"[SUBJECT] not found: {p}"
    obj = json.loads(p.read_text(encoding="utf-8"))
    raw = obj["mapping"] if "mapping" in obj and isinstance(obj["mapping"], dict) else obj
    out = {}
    for k, v in raw.items():
        nk = _normalize_subject_key(k)
        if nk is not None: out[nk] = int(v)
    assert out, "[SUBJECT] empty mapping after normalization"
    return out

# ------------------------- 模型加载 -------------------------
def load_cfg_from_records(run_dir: Path) -> Dict[str, Any]:
    rec = run_dir / "records" / "config.json"
    if rec.exists():
        cfg = json.loads(rec.read_text(encoding="utf-8"))
        return cfg.get("model_cfg", {}) or cfg.get("enc_cfg", {}) or {}
    return {}

def choose_ckpt_path(args) -> Path:
    if args.use_best_ckpt:
        best_txt = Path(args.run_dir) / "records" / "best_checkpoint.txt"
        assert best_txt.exists(), f"best_checkpoint.txt not found: {best_txt}"
        ckpt = best_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = (Path(args.run_dir) / ckpt_path).resolve()
        assert ckpt_path.exists(), f"best ckpt not found: {ckpt_path}"
        log(f"[INFO] Using BEST checkpoint: {ckpt_path}")
        return ckpt_path
    ckpt_path = Path(args.ckpt_path)
    assert ckpt_path.exists(), f"--ckpt_path not found: {ckpt_path}"
    return ckpt_path

from models.meg_encoder import UltimateMEGEncoder

def _read_logit_scale_exp(ckpt_path: Path) -> Optional[float]:
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd.get("state_dict", sd)
    for k in ("model.scorer.logit_scale","scorer.logit_scale","logit_scale"):
        v = state.get(k, None)
        if v is not None:
            try: return float(torch.exp(v).item())
            except Exception:
                try: return float(np.exp(float(v)))
                except Exception: pass
    return None

def load_model_from_ckpt(ckpt_path: Path, run_dir: Path, device: str):
    model_cfg = load_cfg_from_records(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not model_cfg:
        hp = ckpt.get("hyper_parameters", {})
        model_cfg = hp.get("model_cfg", {}) or hp.get("enc_cfg", {})
    assert model_cfg, "no model_cfg/enc_cfg found"

    if "out_timesteps" in UltimateMEGEncoder.__init__.__code__.co_varnames:
        model_cfg["out_timesteps"] = None  # 评测端不做时间池化

    model = UltimateMEGEncoder(**model_cfg)
    state = ckpt.get("state_dict", ckpt)
    new_state = { (k[6:] if k.startswith("model.") else k): v for k, v in state.items() }
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:   log(f"[WARN] Missing keys: {len(missing)} (e.g., {missing[:8]})")
    if unexpected: log(f"[WARN] Unexpected keys: {len(unexpected)} (e.g., {unexpected[:8]})")
    model.eval().to(device)

    meta = {"logit_scale_exp": _read_logit_scale_exp(ckpt_path)}
    if meta["logit_scale_exp"] is not None:
        log(f"[INFO] exp(logit_scale) in ckpt: {meta['logit_scale_exp']:.6f}")
    return model, meta

@torch.no_grad()
def encode_meg_batch(model, batch_rows: List[dict], device: str, subj_map: Dict[str,int]) -> torch.Tensor:
    megs, locs, sidx = [], [], []
    miss = 0
    for r in batch_rows:
        mp = r["meg_win_path"]; lp = r["sensor_coordinates_path"]
        assert mp and Path(mp).exists(), f"missing meg_win_path: {mp}"
        assert lp and Path(lp).exists(), f"missing sensor_coordinates_path: {lp}"
        x = np.load(mp, allow_pickle=False).astype(np.float32)
        x = ensure_meg_CxT(x)
        megs.append(torch.from_numpy(x))
        loc = np.load(lp, allow_pickle=False).astype(np.float32)
        locs.append(torch.from_numpy(loc))
        sid = _normalize_subject_key(r.get("subject_id"))
        if sid is not None and sid in subj_map: sidx.append(subj_map[sid])
        else: sidx.append(0); miss += 1
    if miss:
        log(f"[WARN] {miss}/{len(batch_rows)} rows miss subject mapping; use subj_idx=0")

    meg = torch.stack(megs, 0).to(device)
    loc = torch.stack(locs, 0).to(device)
    sid = torch.tensor(sidx, dtype=torch.long, device=device)

    y = model(meg_win=meg, sensor_locs=loc, subj_idx=sid)  # [B,1024,T?]
    if y.dim() != 3 or y.size(1) != AUDIO_D:
        raise RuntimeError(f"encoder must output [B,1024,T], got {tuple(y.shape)}")
    if y.size(2) != TARGET_T:
        y = F.interpolate(y, size=TARGET_T, mode="linear", align_corners=False)
    return y

# ------------------------- 候选池 -------------------------
@torch.no_grad()
def load_audio_pool_unique(test_rows: List[dict], device: str, dtype: torch.dtype):
    uniq: Dict[str, int] = {}; ids: List[str] = []; feats = []; rep_rows = []
    for r in test_rows:
        cid = content_id_of(r)
        if cid in uniq: continue
        uniq[cid] = len(ids); ids.append(cid); rep_rows.append(r)
    for r in tqdm(rep_rows, desc="Load candidates"):
        a = np.load(r["audio_feature_path"], allow_pickle=False).astype(np.float32)
        a = ensure_audio_DxT(a)
        ta = maybe_interp_1DT(torch.from_numpy(a), TARGET_T)
        feats.append(ta)
    A = torch.stack(feats, 0).to(device=device, dtype=dtype)  # [O,1024,360]
    return A, ids, rep_rows

# ------------------------- 相似度（与你的 base 一致） -------------------------
def compute_logits_clip(queries: torch.Tensor, pool: torch.Tensor,
                        scale: Optional[float] = None) -> torch.Tensor:
    q = queries.to(torch.float32)
    A = pool.to(torch.float32)
    inv_norms = 1.0 / (A.norm(dim=(1, 2), p=2) + EPS)  # [O]
    logits = torch.einsum("bct,oct,o->bo", q, A, inv_norms)
    if scale is not None: logits = logits * float(scale)
    return logits.to(torch.float32)

# ------------------------- 句桶辅助 -------------------------
def _precompute_sentence_buckets(cand_sent_idx_o: torch.Tensor) -> Dict[int, torch.Tensor]:
    buckets = {}
    uniq = torch.unique(cand_sent_idx_o)
    for s in uniq.tolist():
        if s < 0: continue
        buckets[int(s)] = torch.nonzero(cand_sent_idx_o == s, as_tuple=False).view(-1)
    return buckets

def _sent_len_norm_factor(n: int, mode: str = "none") -> float:
    if n <= 0 or mode == "none": return 1.0
    if mode in ("bucket_count","count","kept_count"): return 1.0 / float(n)
    if mode in ("bucket_sqrt","sqrt"):               return 1.0 / float(np.sqrt(max(1.0, n)))
    if mode == "log":                                return 1.0 / float(np.log2(n + 1.0))
    return 1.0

# ------------------------- GCB（soft-consensus, across=sum） -------------------------
@torch.no_grad()
def gcb_apply_to_group(
    base_logits_bo: torch.Tensor,
    cand_sent_idx_o: torch.Tensor,
    buckets: Dict[int, torch.Tensor],
    *,
    topk: int = 128,
    q_quantile: float = 0.95,
    top_m: int = 3,               # mean@topm
    sent_norm: str = "bucket_sqrt",
    topS: int = 3,
    gamma: float = 0.7,
) -> torch.Tensor:
    B, O = base_logits_bo.shape
    if B == 0 or O == 0: return base_logits_bo
    K = min(int(topk), O)

    topk_scores, topk_idx = torch.topk(base_logits_bo, k=K, dim=1, largest=True, sorted=False)  # [B,K]
    thr_b1 = torch.quantile(base_logits_bo, q=q_quantile, dim=1, keepdim=True)                   # [B,1]
    keep_mask = topk_scores >= thr_b1                                                            # [B,K]
    sids = cand_sent_idx_o[topk_idx]                                                             # [B,K]
    valid_mask = keep_mask & (sids >= 0)
    if not valid_mask.any(): return base_logits_bo

    sids_all = sids[valid_mask]                                                                  # [M]
    vals_all = (topk_scores - thr_b1).clamp_min_(0.0)[valid_mask].float()                        # [M]
    us, inv = torch.unique(sids_all, return_inverse=True)                                        # [S_sel]
    S_sel = us.numel()
    device = base_logits_bo.device

    # mean@topm per sentence
    sent_support = torch.empty(S_sel, dtype=vals_all.dtype, device=device)
    for k in range(S_sel):
        vk = vals_all[inv == k]
        if vk.numel() == 0:
            sent_support[k] = -1e9
            continue
        m_take = min(max(1, int(top_m)), int(vk.numel()))
        top_vals = torch.topk(vk, k=m_take, largest=True, sorted=False).values
        sent_support[k] = torch.mean(top_vals)

    # 句长归一
    bucket_sizes = [int(buckets.get(int(sid.item()), torch.empty(0, device=device)).numel()) for sid in us]
    norms = torch.tensor([_sent_len_norm_factor(n, sent_norm) for n in bucket_sizes],
                         dtype=sent_support.dtype, device=device)
    sent_support = sent_support * norms

    keepS = min(int(topS), int(sent_support.numel())) if topS > 0 else int(sent_support.numel())
    if keepS <= 0: return base_logits_bo
    topS_val, topS_idx = torch.topk(sent_support, k=keepS, largest=True, sorted=True)
    us_sel = us[topS_idx]

    # across=sum：句内所有窗口统一 +sup（不改句内排序）
    boost_o = torch.zeros(O, dtype=base_logits_bo.dtype, device=device)
    for k in range(keepS):
        sid = int(us_sel[k].item()); sup = gamma * float(topS_val[k].item())
        if sup <= 0: continue
        bucket_idx = buckets.get(sid, None)
        if bucket_idx is None or bucket_idx.numel() == 0: continue
        boost_o[bucket_idx] += sup

    return base_logits_bo + boost_o.unsqueeze(0)

# ------------------------- 简单分词（与 anchor 对齐，尽量别改空格规则） -------------------------
def tokenize_simple(s: str) -> List[str]:
    return s.strip().split()

# 候选 content_id → “该候选窗口对应的词”（用其自身 anchor_word_idx 从该候选所在句子抽取）
def build_cid2word(candidate_rows: List[dict]) -> Dict[str, str]:
    cid2w = {}
    for r in candidate_rows:
        cid = content_id_of(r)
        sent = (r.get("global_segment_text") or "").strip()
        toks = tokenize_simple(sent)
        ai = int(r.get("anchor_word_idx", -1))
        w = toks[ai] if 0 <= ai < len(toks) else "[UNK]"
        cid2w[cid] = w
    return cid2w

# ------------------------- 主流程 -------------------------
def evaluate(args):
    device = args.device
    amp = args.amp.lower()
    autocast_dtype = torch.bfloat16 if amp == "bf16" else (torch.float16 if "16" in amp or "fp16" in amp else None)

    test_rows = read_jsonl(Path(args.test_manifest))
    log(f"[INFO] test rows = {len(test_rows):,}")

    A, pool_ids, candidate_rows = load_audio_pool_unique(test_rows, device=device, dtype=torch.float32)
    O = A.size(0)
    log(f"[INFO] candidate windows O={O}")

    canon2idx, alias2idx, cand_sent_idx = build_sentence_index_with_alias(candidate_rows)
    cand_sent_idx = torch.tensor(cand_sent_idx, dtype=torch.long, device=device)
    buckets = _precompute_sentence_buckets(cand_sent_idx)

    cid_to_index = {cid: i for i, cid in enumerate(pool_ids)}
    gt_index = [cid_to_index[content_id_of(r)] for r in test_rows]

    run_dir = Path(args.run_dir)
    subj_map = read_subject_mapping_from_records(run_dir)
    log(f"[SUBJECT] loaded {len(subj_map)} subjects")

    ckpt_path = choose_ckpt_path(args)
    model, meta = load_model_from_ckpt(ckpt_path, run_dir, device=device)
    scale = meta.get("logit_scale_exp", None) if args.use_ckpt_logit_scale else None

    topk_list = [int(x) for x in args.topk.split(",")]
    recalls = {"base": {k: 0 for k in topk_list}, "post": {k: 0 for k in topk_list}}
    mrr_sum_base = 0.0
    mrr_sum_post = 0.0
    ranks_post: List[int] = []

    out_dir = run_dir / "results" / "retrieval_gcb_decode"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_topk_k = max(0, int(args.save_topk)) if args.save_topk is not None else 0
    preds_topk_file = preds_tsv_file = None
    if save_topk_k > 0:
        preds_topk_path = out_dir / f"preds_topk{save_topk_k}.jsonl"
        preds_tsv_path = out_dir / f"preds_topk{save_topk_k}.tsv"
        preds_topk_file = open(preds_topk_path, "w", encoding="utf-8")
        preds_tsv_file = open(preds_tsv_path, "w", encoding="utf-8")
        preds_tsv_file.write("query_index\trank_base\trank_post\tgt_cid\tpred_cids_base\tpred_cids_post\n")

    # 分句组（同一句子的所有窗口属于同一 group）
    def sent_key_for_group(r: dict) -> Tuple[str, str]:
        als = sentence_aliases(r)
        return als[0] if als else ("unknown", content_id_of(r))

    sent2idx: Dict[Tuple[str,str], List[int]] = {}
    for i, r in enumerate(test_rows):
        k = sent_key_for_group(r)
        sent2idx.setdefault(k, []).append(i)
    groups = list(sent2idx.values())
    log(f"[INFO] Grouping by sentence: groups={len(groups)}, avg windows/sent={len(test_rows)/max(1,len(groups)):.2f}")

    # 候选 content_id → token
    cid2word = build_cid2word(candidate_rows)

    # -------- 编码 + 打分 + GCB --------
    @torch.no_grad()
    def encode_group(q_indices: List[int]):
        rows = [test_rows[i] for i in q_indices]
        if autocast_dtype is None:
            Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        else:
            with torch.amp.autocast("cuda", dtype=autocast_dtype):
                Y = encode_meg_batch(model, rows, device=device, subj_map=subj_map)
        base = compute_logits_clip(Y, A, scale=scale)              # [B,O]
        post = gcb_apply_to_group(base, cand_sent_idx, buckets,    # GCB only
                                  topk=args.gcb_topk, q_quantile=args.gcb_q,
                                  top_m=args.gcb_top_m, sent_norm=args.gcb_norm,
                                  topS=args.gcb_topS, gamma=args.gcb_gamma)
        return base, post, Y

    # -------- 逐组累积指标 + 为长句准备文本解码材料 --------
    long_sent_records = []   # 收集满足 min_windows/min_tokens 的句子样例材料

    for g in tqdm(range(len(groups)), desc="Evaluate (Grouped)"):
        q_idx = groups[g]
        base, post, _ = encode_group(q_idx)
        # 统计 + 保存 topk（可选）
        for j_in_group, global_j in enumerate(q_idx):
            gt = gt_index[global_j]
            s_b = base[j_in_group]; s_p = post[j_in_group]
            rank_b = int((s_b > s_b[gt]).sum().item()) + 1
            rank_p = int((s_p > s_p[gt]).sum().item()) + 1
            mrr_sum_base += 1.0 / rank_b
            mrr_sum_post += 1.0 / rank_p
            ranks_post.append(rank_p)
            for k in topk_list:
                recalls["base"][k] += int(rank_b <= k)
                recalls["post"][k] += int(rank_p <= k)

            if save_topk_k > 0:
                topk_scores_b, topk_idx_b = torch.topk(s_b, k=save_topk_k, largest=True, sorted=True)
                topk_scores_p, topk_idx_p = torch.topk(s_p, k=save_topk_k, largest=True, sorted=True)
                pred_cids_b = [pool_ids[int(t)] for t in topk_idx_b.tolist()]
                pred_cids_p = [pool_ids[int(t)] for t in topk_idx_p.tolist()]
                rec = {
                    "query_index": int(global_j),
                    "gt_rank_base": int(rank_b),
                    "gt_rank_post": int(rank_p),
                    "gt_cid": pool_ids[gt],
                    "pred_cids_base": pred_cids_b,
                    "pred_scores_base": [float(x) for x in topk_scores_b.tolist()],
                    "pred_cids_post": pred_cids_p,
                    "pred_scores_post": [float(x) for x in topk_scores_p.tolist()],
                }
                preds_topk_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                preds_tsv_file.write(f"{global_j}\t{rank_b}\t{rank_p}\t{pool_ids[gt]}\t{','.join(pred_cids_b)}\t{','.join(pred_cids_p)}\n")

        # ---- 为长句生成文本对比材料（按 group 聚合） ----
        rows = [test_rows[i] for i in q_idx]
        sent_text = (rows[0].get("global_segment_text") or "").strip()
        gold_tokens = tokenize_simple(sent_text)
        n_tokens = len(gold_tokens)
        n_windows = len(q_idx)
        if n_windows < args.min_windows or n_tokens < args.min_tokens:
            continue  # 只保留长句

        # 逐 anchor 的 Base/Post Top-1 词
        # 初始化为占位，避免缺失 anchor
        base_tokens = ["[PAD]"] * n_tokens
        post_tokens = ["[PAD]"] * n_tokens

        # 按 anchor_word_idx 回填
        for j_local, global_j in enumerate(q_idx):
            ai = int(test_rows[global_j].get("anchor_word_idx", j_local))
            if not (0 <= ai < n_tokens): 
                continue
            s_b = base[j_local]; s_p = post[j_local]
            top_b = int(torch.argmax(s_b).item())
            top_p = int(torch.argmax(s_p).item())
            cid_b = pool_ids[top_b]; cid_p = pool_ids[top_p]
            wb = cid2word.get(cid_b, "[UNK]")
            wp = cid2word.get(cid_p, "[UNK]")
            base_tokens[ai] = wb
            post_tokens[ai] = wp

        # 词级准确率（只在非 PAD 位置上）
        mask = [t != "[PAD]" for t in base_tokens]
        denom = max(1, sum(mask))
        base_acc = sum(bt == gt for bt, gt, m in zip(base_tokens, gold_tokens, mask) if m) / denom
        post_acc = sum(pt == gt for pt, gt, m in zip(post_tokens, gold_tokens, mask) if m) / denom

        long_sent_records.append({
            "sentence_text": sent_text,
            "subject_id": str(rows[0].get("subject_id","")),
            "n_windows": n_windows,
            "n_tokens": n_tokens,
            "base_text": " ".join(base_tokens),
            "post_text": " ".join(post_tokens),
            "base_acc": base_acc,
            "post_acc": post_acc,
            "delta": post_acc - base_acc,
        })

    # -------- 汇总指标 --------
    num_queries = len(test_rows)
    metrics = {
        "num_queries": num_queries,
        "pool_size": O,
        "recall_at": {
            "base": {str(k): recalls["base"][k] / num_queries for k in topk_list},
            "post": {str(k): recalls["post"][k] / num_queries for k in topk_list},
        },
        "mrr": {"base": mrr_sum_base / num_queries, "post": mrr_sum_post / num_queries},
        "mean_rank_post": float(np.mean(ranks_post)),
        "median_rank_post": float(np.median(ranks_post)),
        "topk_list": topk_list,
        "flags": {
            "use_ckpt_logit_scale": bool(args.use_ckpt_logit_scale),
            "only_gcb": True
        }
    }
    out_json = Path(args.save_json) if args.save_json else (out_dir / "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    out_ranks = Path(args.save_ranks) if args.save_ranks else (out_dir / "ranks.txt")
    with open(out_ranks, "w", encoding="utf-8") as f:
        for r in ranks_post: f.write(str(int(r)) + "\n")

    # -------- 长句样例选择：improved/unchanged/worsened --------
    # 阈值可配：improve >= +delta_thr；worsen <= -delta_thr；unchanged |delta| < eps
    delta_thr = float(args.delta_threshold)
    eps = float(args.unchanged_eps)

    improved = [r for r in long_sent_records if r["delta"] >= delta_thr]
    unchanged = [r for r in long_sent_records if abs(r["delta"]) < eps]
    worsened = [r for r in long_sent_records if r["delta"] <= -delta_thr]

    # 排序并截断
    improved.sort(key=lambda x: (-x["delta"], -x["post_acc"], -x["n_windows"]))
    unchanged.sort(key=lambda x: (-x["post_acc"], -x["n_windows"]))
    worsened.sort(key=lambda x: (x["delta"], -x["n_windows"]))

    improved = improved[:args.n_improved]
    unchanged = unchanged[:args.n_unchanged]
    worsened = worsened[:args.n_worsened]

    examples = {"improved": improved, "unchanged": unchanged, "worsened": worsened}
    with open(out_dir / "examples_long_sentences.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    # -------- 日志 --------
    log("==== Retrieval Results (Base vs GCB) ====")
    ra_b = metrics["recall_at"]["base"]; ra_p = metrics["recall_at"]["post"]
    log(f"[BASE] R@1={ra_b.get('1',0.0):.4f}  R@5={ra_b.get('5',0.0):.4f}  R@10={ra_b.get('10',0.0):.4f}  MRR={metrics['mrr']['base']:.4f}")
    log(f"[POST] R@1={ra_p.get('1',0.0):.4f}  R@5={ra_p.get('5',0.0):.4f}  R@10={ra_p.get('10',0.0):.4f}  MRR={metrics['mrr']['post']:.4f}")
    log(f"[INFO] Metrics saved to: {out_json.as_posix()}")
    log(f"[INFO] Ranks saved to  : {out_ranks.as_posix()}")
    log(f"[INFO] Long-sentence examples saved to: {(out_dir/'examples_long_sentences.json').as_posix()}")
    if save_topk_k > 0:
        preds_topk_file.close()
        preds_tsv_file.close()

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_manifest", required=True, type=str)
    p.add_argument("--run_dir", required=True, type=str)
    p.add_argument("--ckpt_path", type=str, default="")
    p.add_argument("--use_best_ckpt", action="store_true")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="bf16", choices=["off", "bf16", "fp16", "16-mixed"])
    p.add_argument("--topk", type=str, default="1,5,10")

    # GCB（post）
    p.add_argument("--gcb_topk", type=int, default=128)
    p.add_argument("--gcb_q", type=float, default=0.95)
    p.add_argument("--gcb_top_m", type=int, default=3)
    p.add_argument("--gcb_norm", type=str, default="bucket_sqrt", choices=["bucket_sqrt"])
    p.add_argument("--gcb_topS", type=int, default=3)
    p.add_argument("--gcb_gamma", type=float, default=0.7)

    # 评分缩放 + 输出
    p.add_argument("--use_ckpt_logit_scale", action="store_true")
    p.add_argument("--save_json", type=str, default="")
    p.add_argument("--save_ranks", type=str, default="")
    p.add_argument("--save_topk", type=int, default=0)

    # 长句样例选择参数
    p.add_argument("--min_windows", type=int, default=20, help="仅从窗口数>=此值的句子里选样例")
    p.add_argument("--min_tokens", type=int, default=20, help="仅从词数>=此值的句子里选样例")
    p.add_argument("--delta_threshold", type=float, default=0.05, help="acc 提升/下降阈值")
    p.add_argument("--unchanged_eps", type=float, default=1e-6, help="acc 变化视为不变的阈值")
    p.add_argument("--n_improved", type=int, default=6)
    p.add_argument("--n_unchanged", type=int, default=3)
    p.add_argument("--n_worsened", type=int, default=3)
    return p.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = parse_args()
    evaluate(args)
