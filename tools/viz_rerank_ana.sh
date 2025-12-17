PROJECT_ROOT="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/July_Project"
MAN_DIR="$PROJECT_ROOT/data_manifests_local_global/final_splits_sentence_with_sentence_full"
TEST_MAN="$MAN_DIR/test.jsonl"

RUN_DIR_TO_EVAL="$PROJECT_ROOT/runs/simpleconv_meg_baseline_d5_dp5_EBS256_20251127-180227_job5422219"

EVAL_PY="$PROJECT_ROOT/eval/retrieval_window_vote.py"
PY="/mimer/NOBACKUP/groups/naiss2024-5-164/Xinyu/whisper_venv/bin/python"

$PY -m tools.viz_rerank_analysis \
  --eval_py "$EVAL_PY" \
  --test_manifest "$TEST_MAN" \
  --run_dir "$RUN_DIR_TO_EVAL" \
  --use_best_ckpt --use_ckpt_logit_scale \
  --device cuda --amp bf16 \
  --no_qccp --no_windowvote \
  --gcb_topk 128 --gcb_q 0.95 --gcb_top_m 3 \
  --gcb_norm bucket_sqrt --gcb_topS 3 --gcb_gamma 0.7 \
  --feature_topk 40 --bg_limit 100 \
  --pick_query_mode improved --min_windows 10 --n_example_queries 3 \
  --tsne_perplexity 20 --tsne_n_iter 1500 --tsne_lr 200 --tsne_metric cosine \
  --out_dir "$PROJECT_ROOT/viz_full_out" \
  --select_post_top1_only --select_base_wrong_only \
  --skip_global_logits_distribution