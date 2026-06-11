[ai-generated]

# 19 — Subtractive-regime results (7/7 models, 2026-06-07)

All numbers are the **honest common eval**: ground-truth = tight-token ∩ is_code,
token-AUC over code-only tokens, on each held-out test set. `pairAcc` =
example-level fraction where a vuln's max code-prob exceeds its safe pair's
(chance = 0.50, n_add = 49 / n_sub = 97 test pairs). Operating layer per model
(exp-16); full per-layer band + 8-config tables in `RESULTS_TABLES.md`.

## Cross-comparison (granularity = token, negatives = X = comments-ignored)

| model | base→sub | **sub→sub** | base→base | sub→base | pairAcc-sub (sub-trained) | pairAcc-add (sub-trained) |
|---|---|---|---|---|---|---|
| Qwen-32B | 0.771 | 0.772 | 0.780 | 0.763 | 0.763 | 0.408 |
| Qwen-7B | 0.779 | 0.782 | 0.789 | 0.778 | 0.784 | 0.531 |
| gemma-12b-it | 0.767 | 0.748 | 0.778 | 0.739 | 0.794 | 0.449 |
| gemma-12b-pt | 0.763 | 0.752 | 0.777 | 0.746 | 0.773 | 0.449 |
| gemma-1b-it | 0.742 | 0.739 | 0.768 | 0.743 | 0.701 | 0.367 |
| gemma-27b-it | 0.736 | 0.761 | 0.744 | 0.749 | 0.722 | 0.388 |
| gemma-4b-it | 0.732 | 0.732 | 0.744 | 0.725 | 0.784 | 0.408 |
| **mean** | **0.756** | **0.755** | **0.769** | **0.749** | **0.760** | **0.429** |

(`base→sub` = trained on SVEN-base, eval on subtractive-test; etc.)

## Findings (consistent across all 7)

1. **Subtractive training is performance-neutral on localizable vulns.**
   mean base→sub = 0.756 vs sub→sub = 0.755 — identical. Per-model Δ is within
   ±0.025 either way (27b *better* subtractive-trained +0.025; 12b-it *worse*
   −0.019). **Dropping the additive third (241/715 pairs) costs nothing** — those
   examples carried no localizable-vuln signal, only label noise.

2. **The base probe's edge is base-test-only and doesn't transfer.**
   base→base (0.769) > sub→base (0.749) everywhere, but that 0.02 advantage
   evaporates on the clean subtractive structure (base→sub ≈ sub→sub). The extra
   "signal" base-training picks up is the whole-line / additive artifact, not
   transferable vulnerability structure.

3. **Additive (missing-check) vulnerabilities are undetectable — the headline.**
   pairAcc-add = 0.43 mean (≈ chance), for every model and both train subsets;
   token-level AUC there is undefined (no positive token by construction).
   Contrast: pairAcc-**sub** = 0.76 mean — the same probes separate the
   *localizable* vulns well. **~⅓ of SVEN is invisible to a token-localized
   linear probe in principle**, regardless of how it's trained. This is the core
   justification for the subtractive split.

4. **token ≥ line on the honest eval.** Per-token (tight) labels match or beat
   whole-line on sub-test in 6/7 models (e.g. 27b 0.761 vs 0.744; Qwen-32B 0.772
   vs 0.766; only gemma-4b prefers line, 0.749 vs 0.732), while removing the
   contradictory-label pathology (identical unchanged lines labeled
   1-in-vuln/0-in-safe). Per-token is the right default.

5. **X ≈ Y on the honest eval.** comments-ignored vs comments-as-negative differ
   by ≤0.01 on token-code-AUC (e.g. Qwen-7B 0.782 vs 0.779). Confirms the
   cheap-recompute call: Y's apparent edge was easy-negative inflation, not real
   discrimination.

## Takeaway for the project
For token-probe work the honest target is **SVEN-subtractive + per-token labels +
is_code-gated positives**; reporting on full SVEN-base inflates apparent
performance with whole-line/additive artifacts. Additive vulnerabilities are an
open, separate (currently unsolved) example-level detection problem — a
token-localized probe cannot reach them. See ADR 0004.

## Provenance
- 7 models, re-extracted activations (label-independent; one extraction/model
  served the whole 8-config grid), probes retrained, loss unchanged.
- Cluster jobs: 2485364 (4 models), 2487357 (12b-pt+7B), 2487569 (27b isolated —
  the original 3-model job 2485395 OOM-killed; 27b too heavy to co-extract).
- Subset: `subtractive_membership.json` (478 vuln+safe pairs). Harness:
  `train_grid.py` (Opus-audited). Per-config metrics + probe npz under `results/`.

## CV addendum — 5-fold × 3-seed (2026-06-07)
Re-ran the grid under grouped 5-fold CV × 3 seeds (15 train/evals per config;
func-pair groups never straddle folds; seed varies fold partition + probe init)
to put error bars on the single-split numbers. Extraction: **vLLM**
(`extract_hidden_states`, EAGLE3) for the 2 Qwen — ~2.2–2.5× faster, cos 0.998
vs HF; **HF** for the 5 gemma-3 (no vLLM EAGLE3 in vllm 0.22.1). Harness
`train_grid_cv.py`, tables `RESULTS_CV.md`, plot `fig_cv.png`.

Findings (mean±std, token granularity, X, honest eval), all 7 models:
- **base→sub ≈ sub→sub for every model** — gaps ≤0.009, all ≪ fold-std
  (0.024–0.039). Subtractive training is statistically **neutral**; exp-19's
  point estimate holds under CV.
- **pairAcc-add 0.24–0.43, all below chance** (Qwen 0.24–0.26; gemma 0.38–0.43).
  Additive vulns undetectable — confirmed with error bars.
- base→base only marginally > sub→base (~0.01–0.02, often within std) — the
  base-training edge is small and base-test-specific.
- Fold-std ≈ 0.024 (Qwen) / 0.03–0.04 (gemma) = the real precision of any
  single-split AUC.
- Cluster jobs: 2488335 (Qwen, vLLM), 2488477 (4 gemma, HF), 2488604 (27b, HF).
