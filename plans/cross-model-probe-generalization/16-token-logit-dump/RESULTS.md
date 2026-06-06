[ai-generated]

# 16 — Results (2026-06-06)

Regenerated the canonical span-max vulnerability probe and dumped **every**
per-token + per-example logit over the full SVEN dataset (1430 ex), for 7 models.
Run on the cluster after rebuilding the deleted `~/scratch/probes`.

## Correctness gate — PASSED for all 6 anchored models
Each model's held-out `tokens_code_auc` at its **historical operating layer**
reproduces the `06-honest-metric-sweeps` breakdown to **±0.000** (recomputed
independently from the dumped `logits_layer{NN}.npz`, not the run stdout;
`prob == sigmoid(logit)` verified):

| model | op. layer | tokens_auc | tokens_code_auc | historical | Δ |
|---|---|---|---|---|---|
| Qwen2.5-Coder-32B | 25 | 0.761 | **0.776** | 0.776 | −0.000 ✓ |
| Qwen2.5-Coder-7B  | 16 | 0.802 | **0.813** | — | (no anchor) |
| gemma-3-1b-it | 25 | 0.735 | **0.744** | 0.744 | +0.000 ✓ |
| gemma-3-4b-it | 7  | 0.760 | **0.775** | 0.775 | −0.000 ✓ |
| gemma-3-12b-it | 15 | 0.758 | **0.763** | 0.763 | −0.000 ✓ |
| gemma-3-27b-it | 19 | 0.745 | **0.759** | 0.759 | −0.000 ✓ |
| gemma-3-12b-pt | 13 | 0.768 | **0.782** | 0.782 | −0.000 ✓ |

## Per-layer band (held-out tokens_code_auc)
- Qwen2.5-Coder-32B: L23 .782 · L24 .774 · **L25 .776** · L26 .780 · L27 .784
- Qwen2.5-Coder-7B:  L14 .805 · **L16 .813** · L18 .808 · L20 .804 · L22 .794
- gemma-3-1b-it:  L23 .635 · L24 .724 · **L25 .744**
- gemma-3-4b-it:  L5 .769 · L6 .767 · **L7 .775** · L8 .758 · L9 .751
- gemma-3-12b-it: **L13 .790** · L14 .763 · **L15 .763** · L16 .799 · L17 .792
- gemma-3-27b-it: L18 .755 · **L19 .759** · L20 .769  (3-layer band; see OOM note)
- gemma-3-12b-pt: L11 .750 · L12 .768 · **L13 .782** · L14 .796 · L15 .771

Notes:
- **7B ≥ 32B** on token-code-AUC (0.813 vs 0.784 peak) — scale buys little here.
- Several models peak a layer or two off the historical pick (e.g. 12b L16=0.799,
  12b-pt L14=0.796) — the historical layer was chosen by (saturated) example-AUC,
  so token-AUC sometimes favours a neighbour. All logits for the band are dumped.
- **example_auc ~0.5** is a max-pool **saturation** artifact (max sigmoid over all
  tokens saturates near 1.0 on large models), same pooling as historical
  `train_eval`. Token-level is the meaningful, verified signal; the per-token
  logits let you try better example reductions (mean / top-k / code-only max).

## Artifacts (per model, per layer NN), `results/logitdump_<slug>/`
- `logits_layer{NN}.npz` — flat token table over ALL examples: `logit, prob
  (f32); y (i8); example_id (i32); char_start, char_end (i32); is_test, is_code
  (bool)`. **This is "all the logits."** (561k tokens Qwen / 690k gemma.)
- `probe_layer{NN}.npz` — `w, b, layer` (the probe, finally persisted).
- `example_scores_layer{NN}.json`, `metrics_logitdump.json`.

Heavy `logits_*.npz` live in this dir + cluster scratch (gitignored); probes /
metrics / example-scores are committed. Canonical wandb/HF persistence pending.

## Operational notes
- gemma-3 is gated → needs `$WORK/secrets/hf_token` (read scope suffices).
- **OOM:** gemma-3-27b is multimodal (loads via ImageTextToText) and OOM'd at a
  5-layer band even alone (>460 GB peak at the CPU vstack). Fixes: `--mem=820000`
  (node has 870 GB) + a 3-layer band. Also: do NOT co-schedule two heavy 5-layer
  extractions on one node — they vstack ~690k×5 token-activation matrices in CPU
  simultaneously. One heavy model per node, or trim layers.
