[ai-generated]

# Causal steering — proper rescale + specificity control (2026-06-01)

Does adding the memory-family probe DIRECTION to the residual stream causally move the
model's verbalized P("yes, vulnerable")? v2 fixes the two flaws in v1.

## Method (v2)
- **Rescale (proj-std):** steer `α · scale_d · d̂` at the best layer, where
  `scale_d = std over code tokens of (h·d̂)` — so α is in *std-along-the-direction* units.
  (v1 used `scale = median |h|₂`; for Gemma that's ≈6670, i.e. **~2000 std** along the
  probe direction — a colossal, model-breaking push. v2 Gemma scale ≈3.0.)
- **Specificity control:** steer 4 directions, each by its own proj-std — `memory`
  (test), `injection` (a different *real* direction), and `random_0`/`random_1` (random
  unit directions). A causal+specific memory direction must move memory-positives MORE
  than negatives AND more than the random/injection directions.
- α grid ±{1,2,4} std; α=0 self-check (idle hook == no-hook P(yes); max_abs_diff=0.0 on all
  4 → hook correct). Degradation guard (flag if yes+no first-token mass < 0.05).
- Subsets: memory-pos / injection-pos / negatives (n≈40 each), verbalized P(yes).

## Result — FLAT / epiphenomenal on all 4 models

| model | verdict | Δ memory-pos (+α) | Δ(mem−neg) | Δ(mem−random) |
|---|---|---|---|---|
| Qwen2.5-Coder-32B | epiphenomenal / flat | +0.006 | +0.001 | +0.002 |
| Qwen3-32B | epiphenomenal / flat | +0.011 | +0.002 | +0.011 |
| Qwen3.6-27B | epiphenomenal / flat | +0.002 | +0.004 | −0.003 |
| gemma-3-27b | epiphenomenal / flat | +0.001 | +0.001 | +0.012 |

At a fair (in-distribution) magnitude — up to ±4 std along the direction — **steering the
memory direction does NOT move the verbalized judgment** (Δ < 0.012), it moves
memory-positives no more than negatives, and it is **indistinguishable from a random
direction**. No degradation in this range (the model answers yes/no throughout). Example
curves (Qwen3-32B, memory direction, memory-pos): 0.620 / 0.624 / 0.625 / **0.632** /
0.631 / 0.636 / 0.643 across α = −4…+4 — flat.

## Interpretation (revises v1)

**v1's "the memory direction is causal" (P(yes) 0.08→0.36) was an artifact of an
oversized, model-breaking perturbation** (median-norm scale ≈ 2000 std along the
direction). With a properly-scaled intervention, the memory *probe direction* — which
*reads* memory-vuln well (probe AUC 0.66–0.73) — does **not causally drive** the model's
verbalized memory judgment. The probe direction is, at valid magnitudes, a **read-out
correlate, not a control knob** for the stated belief.

This does NOT contradict the belief-audit / prompt-sweep story: the model *can* report
memory-safety when the PROMPT redirects it (exp-14), and the representation *is* there
(family/memory probe). But you recover the judgment by **changing the question**, not by
additively pushing the probe direction in activation space.

(Wide-grid α∈{±64,±16,±4,±1,0} sweep was launched to map effect-vs-degradation and confirm
"flat until the model breaks, no clean causal window"; cluster access ended before it
finished — see job 2453175. The ±4 result above stands as the finding.)
Plot: `data/plots/cross-model/fig10_steering.png`. Data: `results/steer_v2/steer_13_*.json`.
