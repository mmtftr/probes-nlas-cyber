[ai-generated]

# 16 — Results (2026-06-06)

Regenerated the canonical span-max vulnerability probe and dumped **every**
per-token + per-example logit over the full SVEN dataset (1430 ex, 561,266
tokens, 21,524 positive tokens). Run on the cluster after rebuilding the deleted
`~/scratch/probes`. Debug job 2478207 (1 node, ~13 min wall incl. dep reinstall
+ 32B download + extract + dump).

## Correctness gate — PASSED
`Qwen/Qwen2.5-Coder-32B-Instruct` L25 **`tokens_code_auc = 0.776`**, exact match
to the historical breakdown (`06-honest-metric-sweeps`, 0.776). Recomputed
independently from the dumped `logits_layer25.npz` (not the run's stdout):
identical. `prob == sigmoid(logit)` verified.

## Per-layer held-out AUC

| model | layer | tokens_auc | tokens_code_auc | example_auc |
|---|---|---|---|---|
| Qwen2.5-Coder-32B | 23 | 0.767 | 0.782 | 0.590 |
| Qwen2.5-Coder-32B | 24 | 0.758 | 0.774 | 0.500 |
| **Qwen2.5-Coder-32B** | **25** | 0.761 | **0.776** ✓ | 0.500 |
| Qwen2.5-Coder-32B | 26 | 0.764 | 0.780 | 0.500 |
| Qwen2.5-Coder-32B | 27 | 0.768 | 0.784 | 0.500 |
| Qwen2.5-Coder-7B | 14 | 0.790 | 0.805 | 0.599 |
| **Qwen2.5-Coder-7B** | **16** | 0.802 | **0.813** | 0.573 |
| Qwen2.5-Coder-7B | 18 | 0.791 | 0.808 | 0.594 |
| Qwen2.5-Coder-7B | 20 | 0.785 | 0.804 | 0.511 |
| Qwen2.5-Coder-7B | 22 | 0.779 | 0.794 | 0.505 |

- **7B ≥ 32B** on token-code-AUC (0.813 vs 0.784 peak) — scale buys little for
  this probe, consistent with the small cross-size spread already seen.
- **example_auc ~0.5** is a max-pool **saturation** artifact (max sigmoid over
  all tokens saturates near 1.0 for both classes on these large models), using
  the same pooling as the historical `train_eval`. Token-level is the meaningful,
  verified signal; the per-token logits let you try better example reductions
  (mean, top-k, code-only max) locally.

## Artifacts (per model, per layer NN)
`results/logitdump_<slug>/`:
- `logits_layer{NN}.npz` — flat token table, one row per token over ALL examples:
  `logit, prob (float32); y (int8); example_id (int32); char_start, char_end
  (int32); is_test, is_code (bool)`. **This is "all the logits."**
- `probe_layer{NN}.npz` — `w, b, layer` (the probe, finally persisted).
- `example_scores_layer{NN}.json` — per-example max-pool score + label + cwe + lang.
- `metrics_logitdump.json` — per-layer + best-layer test AUCs.

The heavy `logits_*.npz` (6 MB each) live in this `results/` dir + cluster
scratch; only probes/metrics/example-scores are committed to git. Canonical
wandb/HF persistence is pending the HF token (see below).

## Blocked
All gemma-3 are gated; HF token gone everywhere. Drop it at
`~/scratch/probes/secrets/hf_token` and the gemma roster (1b/4b/12b/27b-it +
12b-pt, hist L25/7/15/19/13) runs via `submit_logitdump.sh`.
