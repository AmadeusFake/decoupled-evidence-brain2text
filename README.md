Decoupled Evidence Aggregation Stabilizes Non-invasive Neural Semantic Decoding
===============================================================================

This repository contains the code and scripts used in the semester project report:

  “Decoupled Evidence Aggregation Stabilizes Non-invasive Neural Semantic Decoding”
  (BioRob, EPFL; Sep 2025 – Jan 2026).

The project’s main message is evaluation-first:

  • Sentence-level *generative* decoding can look “good” for the wrong reasons under low SNR
    (language priors, forced-alignment noise, representation mismatch, and protocol side channels
    like length/padding leakage).  
  • Therefore, the primary endpoint in this repo is a leak-aware, closed-pool M/EEG→audio
    retrieval benchmark with strict content-based train/test isolation and explicit chance levels.   
  • On top of window-level retrieval, the main contribution is Group Context Boost (GCB):
    a training-free, test-time logit-space evidence aggregation rule that injects sentence context
    while keeping encoder training and embedding geometry unchanged.  
------------------------------------------------------------------------------
What you can reproduce
------------------------------------------------------------------------------

1) Leak-aware closed-pool retrieval (primary endpoint)
   The report evaluates M/EEG→audio retrieval under closed candidate pools.
   Chance R@1 is exactly 1/N by design, where N is the pool size at test time.

   Datasets and closed-pool sizes used in the report:
     • brennan (EEG, EN):        60 channels, N=388
     • mous (MEG, NL):          272 channels, N=825
     • gwilliams2023 (MEG, EN): 208 channels, N=1,464 (zero-shot) and N=7,011 (session-isolated)

2) Neural encoders + frozen audio encoder
   Two MEG/EEG encoder families are supported:
     • DenseCNN (compact backbone designed for retrieval)
     • Exp-dilated CNN (strong reference backbone from prior MEG decoding work)
   Both map neural windows to 1024-d embeddings matched to frozen wav2vec2 audio embeddings.

3) Test-time aggregation (no retraining)
   • WindowVote: hard/consensus control
   • GCB: soft consensus in logit space (main method)


------------------------------------------------------------------------------
Datasets supported & access notes
------------------------------------------------------------------------------

This project currently supports three studies:

- `brennan`: EEG, 60 sensors, 33 subjects, 6.7 hours, [reference](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207741).
- `mous` : MEG, 272 sensors, 96 subjects, 80.9 hours, [reference](https://www.nature.com/articles/s41597-019-0020-y). **Important:** this study requires registration before being able to download. Please see the original publication for details.
- `gwilliams2023` : MEG, 208 sensors, 27 subjects, 56.2 hours, [reference](https://www.nature.com/articles/s41467-022-34326-1).

Dataset-specific preprocessing pipelines are in:
  data_script_BN/   data_script_MOUS/   data_script_GW/

------------------------------------------------------------------------------
Results snapshot (from the report)
------------------------------------------------------------------------------

Main retrieval improvements from test-time GCB (no retraining):   

  Gwilliams (zero-shot; N=1,464)
    • DenseCNN:       R@1 0.44 → 0.52
    • Exp-dilated CNN: R@1 0.41 → 0.48

  Gwilliams (session isolation; k=5; N=7,011)
    • DenseCNN:  R@1 0.29 → 0.30, MedR 7.5 → 6.0
    • Exp-dilated CNN: R@1 0.18 → 0.19, MedR 30 → 27

  MOUS (zero-shot; N=825)
    • DenseCNN:       R@1 0.21 → 0.28
    • Exp-dilated CNN: R@1 0.16 → 0.21

EEG (Brennan) remains near chance; GCB does not yield reliable gains in the report’s setting.   

------------------------------------------------------------------------------
Key protocol design choices (important for “leak-free” claims)
------------------------------------------------------------------------------

• Fixed-length windows remove duration/padding leakage highlighted by the diagnostics.   
• Content-based isolation: a “content key” (audio path, onset, offset) defines the unit of
  generalization; all occurrences of the same content key are assigned to the same split.   
• Overlap pruning: windows overlapping in time with held-out test windows on the same audio are
  removed from train/val (test has priority), preventing temporal leakage.   
• Closed candidate pools: evaluation ranks against a fixed held-out pool, making chance explicit.   

------------------------------------------------------------------------------
Repository layout
------------------------------------------------------------------------------

script/                 Experiment entrypoints (train/eval/session isolation)
data_script_GW/         Gwilliams MEG: staged pipeline + kfold + utilities
data_script_MOUS/       MOUS MEG: staged pipeline + alignment + utilities
data_script_BN/         Brennan EEG: staged pipeline + windows/features
models/                 network models
train/                  Python training entrypoints (invoked by scripts)
eval/                   Evaluation utilities (retrieval eval, WV/GCB, bootstrap)

------------------------------------------------------------------------------
How to run: two kinds of entrypoints
------------------------------------------------------------------------------

A) data_script_*  (run in order)
--------------------------------
These folders build manifests/splits, preprocess neural signals, and compute audio features.
They are designed to be run *stage-by-stage*.

Gwilliams (MEG) – order:
  bash data_script_GW/run_stage1_data_pipeline.sh
  bash data_script_GW/run_stage2_preprocess_audio.sh
  bash data_script_GW/run_stage3_meg_preslicing.sh
  bash data_script_GW/run_stage4_meg_sentence.sh
  bash data_script_GW/run_stage5_audio_sentence.sh
  bash data_script_GW/make_kfold_splits.sh

MOUS (MEG) – order:
  bash data_script_MOUS/run_stage1_data_pipeline.sh
  bash data_script_MOUS/run_stage2_preprocess_audio.sh
  bash data_script_MOUS/run_stage3_meg_preslicing.sh

Brennan (EEG) – order:
  bash data_script_BN/eeg_stage1_make_splits.sh
  bash data_script_BN/eeg_stage2_audio_features.sh
  bash data_script_BN/eeg_stage3_eeg_windows.sh

Notes
  • Manifests are JSONL. Each split directory typically contains train.jsonl / valid.jsonl / test.jsonl.
  • Candidate pool diagnostics are available via report_candidate_pool.py in each data_script folder.

B) script/  (choose by scenario)
--------------------------------
The script folder contains named experiment runners. You typically:
  1) run the appropriate data pipeline above
  2) pick a training script
  3) pick an evaluation script (or session-isolated protocol)

Available entrypoints (as currently in this repo):

Training
  • train_meg_gw_baseline.sh
  • train_meg_MOUS.sh
  • train_eeg_dense.sh
  • train_eeg_exp.sh
  • train_coupling_model.sh

Evaluation
  • zero_shot_eval.sh
  • eval_eeg.sh

Session isolation (Gwilliams)
  • session_iso_sbatch_train.sh
  • session_iso_sbatch_eval.sh
  • session_iso_eval_gw.sh
  • stats_session_iso_eval_gw.sh

Typical workflows

1) Gwilliams baseline → zero-shot eval
  sbatch script/train_meg_gw_baseline.sh
  sbatch script/zero_shot_eval.sh

2) MOUS baseline → zero-shot eval
  sbatch script/train_meg_MOUS.sh
  sbatch script/zero_shot_eval.sh

3) EEG training + eval
  sbatch script/train_eeg_dense.sh
  sbatch script/eval_eeg.sh
  # or:
  sbatch script/train_eeg_exp.sh
  sbatch script/eval_eeg.sh

4) Gwilliams session isolation protocol
  sbatch script/session_iso_sbatch_train.sh
  sbatch script/session_iso_sbatch_eval.sh
  bash  script/stats_session_iso_eval_gw.sh

------------------------------------------------------------------------------
Important flags & reproducibility knobs
------------------------------------------------------------------------------

--meg_encoder {dense, exp}
  Some entrypoints accept selecting the MEG encoder backbone:
    • dense : DenseCNN family
    • exp   : Exp-dilated CNN family
  Recommendation: keep the encoder choice consistent between training and evaluation.

Aggregation controls (conceptual)
  • baseline: no WindowVote, no GCB
  • vote_only: WindowVote enabled, GCB disabled
  • gcb_only: GCB enabled, WindowVote disabled
  • gcb_vote: both enabled (in a fixed order in the “minimal evaluator”)

Statistical testing
  The report uses paired bootstrap resampling (sentences as units, 10,000 draws) to compare
  conditions, preserving sentence-wise pairing.   

------------------------------------------------------------------------------
Outputs
------------------------------------------------------------------------------

A typical run directory contains:
  checkpoints/     model checkpoints
  records/         metadata (e.g., subject mapping)
  results/         evaluation outputs

Common retrieval artifacts:
  metrics.json
  ranks.txt
  per_query.tsv
  sentence_metrics.tsv

Bootstrap utilities may output:
  bootstrap_*.tex  (LaTeX tables ready to \\input{} into the paper)

------------------------------------------------------------------------------
Minimal reproduction checklist (as in the report)
------------------------------------------------------------------------------

Recommended minimal reproduction path:   

  1. Create the environment and verify dependencies.
  2. Generate (or download) dataset meta-manifests and split files (content keys; no overlap).

  3. Extract frozen audio embeddings with the unified wav2vec2 configuration.
  4. Train neural encoders (DenseCNN / exp-dilated CNN).
  5. Run closed-pool retrieval evaluation and apply test-time aggregation (GCB and controls).
  6. Regenerate the main results table and figures from saved evaluation outputs.

