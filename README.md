# Decoupled Evidence Aggregation Stabilizes Non-invasive Neural Semantic Decoding

This repository contains code and manuscript for:

**“Decoupled Evidence Aggregation Stabilizes Non-invasive Neural Semantic Decoding” (Zhang et al., 2025).**

We revisit non-invasive speech decoding from MEG/EEG and ask:

> How much of the apparent performance in current brain-to-text systems actually comes from the neural signals, and how much from language-model priors and evaluation artefacts?

We show that standard generative MEG→text pipelines can reach strong language metrics even when neural inputs are shuffled or replaced by noise, and propose a **decoupled evidence aggregation** principle that yields more reliable and efficient decoding.

---

## Key ideas

- **Metric and leakage illusions**  
  We re-implement recent MEG-to-text systems and show that:
  - language-based scores (e.g. BERTScore, WER) can be satisfied by prior-driven text that barely uses MEG;
  - padding, attention masks and noisy transcripts can strongly inflate reported performance.

- **Leak-free MEG→audio retrieval**  
  We move to a strictly leak-free **MEG→audio retrieval** paradigm:
  - fixed-length 3 s MEG windows retrieve their matching audio segments from large candidate pools;
  - chance level is explicit, and content-unique pools avoid label overlap and leakage across splits.

- **Narrow, deep encoder under low SNR**  
  We design a **narrow, deep CNN encoder** with strong local nonlinearities that:
  - outperforms the published MEG→audio baseline of Défossez et al. (Nat. Mach. Intell., 2023) under the same protocol;
  - reveals that naively enlarging the temporal receptive field can hurt decoding in low-SNR MEG.

- **Decoupled evidence aggregation (GCB)**  
  Instead of entangling global context inside the encoder, we:
  - keep feature extractors largely fixed;
  - let context act in **logit space** as an explicit prior over candidate sentences.

  We instantiate this as **Group Context Boost (GCB)**:
  - a **training-free**, group-wise logit re-ranking layer;
  - combines window-level retrieval scores with sentence-level consensus;
  - raises zero-shot R@1 from **0.41 to 0.52** (about **27%** relative improvement over the original MEG→audio baseline) with negligible extra FLOPs.

---

## Repository structure (work in progress)

The repository is under active development. A typical layout is:

- `models/` – MEG/EEG encoders, audio encoders, GCB implementation.  
- `eval/` – retrieval, voting and bootstrap evaluation scripts.  
- `configs/` – experiment and model configuration files.  

