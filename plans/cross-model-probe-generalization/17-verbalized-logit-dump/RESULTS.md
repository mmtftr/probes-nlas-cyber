[ai-generated]

# 17 — Verbalized logit dump — RESULTS (2026-06-06)

Re-ran each model's OWN verbalized yes/no vulnerability judgment over the full
SVEN dataset (1430 ex) and **persisted every logit** (yes/no token logits,
`yes_lp`/`no_lp`/`margin`, `p_yes`, top-10 first-token logits) per example —
closing the gap exp-05 left (it saved only the scalar `p_yes`). Mirrors exp-16
for the probe side. Score path is bit-identical to exp-05.

## Verbalized example-AUC (P(yes, vulnerable?))

| model | AUC (all 1430) | AUC (test 292) | reproduction gate |
|---|---|---|---|
| gemma-3-1b-it | 0.483 | 0.486 | — (no history) |
| gemma-3-4b-it | 0.546 | 0.533 | — (no history) |
| gemma-3-12b-it | 0.557 | 0.556 | — (no history) |
| gemma-3-27b-it | 0.571 | **0.566** | vs exp-05 0.554 · Δ **+0.012** ✓ |
| Qwen2.5-Coder-7B-Instruct | 0.573 | 0.576 | — (no history) |
| Qwen2.5-Coder-32B-Instruct | 0.613 | **0.623** | vs exp-05 0.632 · Δ **−0.009** ✓ |

Split: seed-42 20% group-clean hold-out (same as exp-16) → 292 test / 1138 train.

## Read

- **Gate passed** for both models exp-05 reported (within ~0.01 of history) →
  verbalized path faithfully reproduced; the dumped logits are trustworthy.
  (exp-05 averaged 5 seeds' test intersections; this is the single seed-42 test
  set — small drift expected.)
- **Verbalized judgment scales with model size** within gemma (0.49 → 0.57) and
  the Coder models verbalize better than same-ish-size gemma (Coder-7B 0.576 >
  gemma-12b 0.556; Coder-32B 0.623 is the strongest verbalizer).
- **All weak in absolute terms** (0.49–0.62) — consistent with exp-05's framing
  that both probe and verbalized reads are weak on the rebuilt dataset.
- gemma-1b sits *below* chance (0.486) — a real model-quality signal (its yes/no
  preference is mildly anti-correlated with vulnerability), not a bug; `p_yes` is
  computed by the identical code path as the gate-passing models.

## Artifacts

Per model under `results/<slug>/`:
- `logits_verbalized.npz` — per-example table: eid, label, is_test, p_yes,
  yes_lp, no_lp, margin, yes_logits_raw[N,5], no_logits_raw[N,5],
  topk_ids[N,10], topk_logits[N,10], + yes_ids/no_ids.
- `example_scores_verbalized.json`, `metrics_verbalized_logits.json`, `run.log`.

Code review (post-run): no blocking issues; p_yes/margin/AUC/eid-alignment/shard-merge
all verified against the npz artifacts.
