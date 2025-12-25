# Decoupled Evidence Aggregation Stabilizes Non-invasive Neural Semantic Decoding

Code + manuscript for:

**“Decoupled Evidence Aggregation for Non-invasive Brain-to-Text Retrieval” (Zhang et al., 2025)**

## TL;DR

Non-invasive MEG/EEG brain-to-text systems can look strong on language metrics even when neural evidence is weak or absent, due to priors and evaluation artefacts. We therefore use a **leak-free MEG/EEG → audio retrieval** benchmark as the **primary endpoint**, and introduce **Group Context Boost (GCB)**: a **training-free, test-time** **logit-space** aggregation rule that injects sentence context as an explicit prior without entangling it into encoder training.

---

## What’s in this repo

- **Manuscript draft** (paper figures/tables and methodological details).
- **Dataset-specific preprocessing** for building *content-unique* splits and sentence/window manifests.
- **Neural encoders + audio encoder interface** for retrieval training.
- **Evaluation scripts** for full-pool retrieval, oracle diagnostics, and **GCB / WindowVote** aggregation variants.

> Note: The generative MEG→text pipeline exists **only as a diagnostic tool** to expose failure modes (metric illusion, alignment noise, padding/length leakage). **Primary results are retrieval-based.**

---

## Repository layout

```
data_script_gw/      # Gwilliams MEG: manifest building, content-based splits, window extraction
data_script_MOUS/    # MOUS MEG: BIDS event parsing, Whisper word timestamps, manifest/splits
data_script_BN/      # Brennan EEG: manifest building, content-based splits, window extraction

models/              # Neural encoders, audio embedding interface, GCB / aggregation modules
train/               # Training entrypoints (contrastive retrieval training, checkpoints, logging)
eval/                # Leak-free retrieval evaluation + aggregation (GCB, WindowVote), bootstrap, reports
tools/               # Utilities: visualization, grid search, assistant evaluation scripts
script/.             # Utilities: running entrance: bash/sbatch scripts
.gitignore
README.md
```

---

## Core method (paper-aligned)

### 1) Leak-free MEG/EEG → audio retrieval (primary evaluation)
- Each **fixed-length** neural window queries a **closed candidate pool** of audio segments.
- Splits are **content-unique** (no shared sentence/audio content across train/val/test).
- Chance level is explicit; retrieval metrics are rank-based (R@k, MRR, MedR).

### 2) Decoupled evidence aggregation: Group Context Boost (GCB)
- Keep the encoder and training objective unchanged.
- At **test time**, aggregate window-level logits within each sentence group into a sentence-level bias, then add it back to each window’s logits.
- This injects sentence context as a **controlled, auditable prior in logit space**.

---

## Quickstart (typical workflow)

### Step 0 — Prepare an environment
This repo assumes a standard PyTorch + scientific Python stack, plus speech models used for audio embeddings (e.g., wav2vec2). Exact versions may vary by machine/cluster; treat this as a research codebase under active iteration.

### Step 1 — Build dataset manifests & splits (per dataset)
Each `data_script_*` folder produces a sentence-level **meta manifest** and **content-based** train/val/test splits. See scripts inside each folder (the entrypoints are dataset-specific).

Examples you’ll typically find/do:
- Build a JSONL meta manifest (sentence ID, audio interval, neural interval).
- Construct content keys and split them (e.g., 70/10/20).
- Extract fixed-length word-anchored windows (e.g., 3s windows at 120 Hz).

### Step 2 — Train retrieval models
Training scripts live in `train/`. The objective aligns neural embeddings to frozen audio embeddings under a CLIP-style contrastive loss.

### Step 3 — Evaluate (global pool + oracle diagnostics + GCB)
Evaluation scripts live in `eval/`:
- Global-pool leak-free retrieval (primary)
- Oracle-sentence restricted pool (upper-bound diagnostic)
- Aggregation variants:
  - `gcb_only` (recommended)
  - `vote_only` (negative control)
  - `gcb_vote` (usually weaker than `gcb_only`)

---

## Reproducing the paper’s main comparisons (high level)

The manuscript reports (i) diagnostic generative failure modes and (ii) retrieval-based results across datasets and protocols, plus aggregation ablations.

A typical reproduction checklist:
1. Build **content-unique** splits for each dataset.
2. Train Dense CNN and Exp-dilated CNN retrieval encoders under the same protocol.
3. Run leak-free evaluation on:
   - Gwilliams (global pool; and session-isolated resplitting if available)
   - MOUS (global pool)
   - Brennan EEG (global pool; expected near-chance lower bound)
4. Run GCB on top of the trained checkpoints (no retraining).

---

## Practical notes

- **Generative pipeline ≠ main benchmark.** Treat it as a stress test for evaluation artefacts (alignment noise, padding/mask leakage, metric insensitivity).
- **Fixed-length windows** are critical to eliminate length-based shortcuts in sentence-level decoding protocols.
---
