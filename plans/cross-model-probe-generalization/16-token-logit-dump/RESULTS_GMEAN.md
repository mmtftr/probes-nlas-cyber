[ai-generated]

# g-mean / g-mean² readout (2026-06-07)

g-mean = √(TPR·TNR); g-mean² = TPR·TNR. A class-balanced operating-point score
that goes to 0 if either class is ignored — the honest lens for the imbalanced
token-level eval (few vulnerable tokens, many safe code tokens), where plain
accuracy is dominated by the negatives. Reported at the threshold that
**maximises g-mean** (the probe's best class-balanced operating point; the
imbalanced analogue of `threshold_optimized_accuracy`).

- Metric code: `src/eval/metrics.py` (`max_gmean`, `gmean_at_threshold`,
  g-mean fields on `ClfMetrics`).
- Readout script: `gmean.py` → `gmean_results.json` (this dir).
- Substrate: exp-16 per-token logit dumps, **honest label** (tight-diff ∩
  is_code, code-only tokens), held-out test split, operating layer per model.

## Numbers (operating layer per model)

| model | L | AUC_tok | TOK g² | g-mean | TPR | TNR | EX g²(all) | EX g²(sub) |
|---|---|---|---|---|---|---|---|---|
| Qwen-32B | 25 | 0.772 | 0.510 | 0.714 | 0.65 | 0.79 | 0.328 | 0.439 |
| Qwen-7B  | 16 | 0.806 | 0.544 | 0.737 | 0.66 | 0.82 | 0.316 | 0.446 |
| g3-1b-it | 25 | 0.733 | 0.471 | 0.687 | 0.61 | 0.77 | 0.278 | 0.326 |
| g3-4b-it | 7  | 0.769 | 0.493 | 0.702 | 0.65 | 0.76 | 0.270 | 0.356 |
| g3-12b-it| 15 | 0.770 | 0.495 | 0.704 | 0.71 | 0.69 | 0.292 | 0.414 |
| g3-27b-it| 19 | 0.784 | 0.513 | 0.716 | 0.71 | 0.72 | 0.304 | 0.371 |
| g3-12b-pt| 13 | 0.799 | 0.523 | 0.723 | 0.66 | 0.79 | 0.334 | 0.436 |

- **TOK g²** = token-level g-mean² (positive = vulnerable code token, negative =
  safe code token). **EX g²** = example-level (score = max code-token logit per
  function; positive = vuln function, negative = safe function), over all test
  examples (all) and the subtractive-only test subset (sub).
- **AUC_tok** here is the *honest* tight-token ∩ is_code AUC, so it reproduces
  exp-19's `sub→sub` anchors (e.g. Qwen-32B 0.772) — not the older exp-16
  whole-line `tokens_code_auc`. g² is reported on the same honest label.

## Findings
1. **Token-level g-mean ≈ 0.69–0.74 (g² 0.47–0.54).** At its best balanced
   operating point the probe catches ~61–71% of vulnerable code tokens (TPR)
   while passing ~69–82% of safe code tokens (TNR). Consistent with AUC≈0.77
   (0.71²≈0.51): a **moderate** detector — no single threshold gives both high
   TPR and high TNR. Qwen-7B strongest, g3-1b weakest.
2. **EX g²(sub) > EX g²(all) for all 7 models** (e.g. Qwen-7B 0.45 vs 0.32).
   Additive vulns drag down function-level detection — independent
   re-confirmation of [[0004-subtractive-subset-and-training-regime]].
3. **Example g²(all) ≈ 0.32 ≪ within-pair pairAcc 0.76** (exp-19): a single
   global threshold across all functions is much harder than forced-choice
   ranking inside a vuln/safe pair. g-mean² at a global threshold is the
   deployment-realistic number; pairAcc is the easier, idealised one.

## Caveat / provenance
Computed on the **exp-16 probes** (old line/all training regime), under the
honest eval label. exp-19 found token≈line (Δ≤0.01 AUC), so these are
representative of the canonical subtractive/token/X probes. The definitive
canonical number would fold `max_gmean` into exp-19's `train_grid.py` and re-run
on the cluster — deferred (user: local readout suffices, 2026-06-07).
