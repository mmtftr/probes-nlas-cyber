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

## Wide-grid confirmation (α ∈ {±64,±16,±4,±1,0} std, memory direction, 3 models)

Pulled before cluster access ended (memory direction complete for Qwen2.5-Coder, Qwen3-32B,
gemma; data `results/steer_wide/`). The model stays non-degraded throughout (still answers
yes/no even at ±64):

| model | memory-pos: −64 / 0 / +16 / +64 | negative: −64 / 0 / +16 / +64 |
|---|---|---|
| Qwen2.5-Coder | 0.04 / 0.08 / 0.10 / 0.30 | 0.35 / 0.33 / 0.36 / **0.64** |
| Qwen3-32B | 0.50 / 0.63 / 0.67 / 0.36 | 0.62 / 0.69 / 0.73 / 0.53 |
| gemma-3-27b | 0.81 / 0.95 / 0.97 / 0.63 | 0.91 / 0.95 / 0.95 / 0.70 |

- **Flat across ±16 std** — no effect at any in-distribution magnitude.
- **At ±64 std the shift is GLOBAL, not memory-specific**: Qwen2.5-Coder +64 raises
  memory-pos by +0.22 but negatives by **+0.31 (more)**; Qwen3-32B/gemma destabilise
  *downward* at +64 on both subsets. There is **no magnitude at which the memory direction
  specifically lifts memory-positives**. This is the airtight version of the conclusion:
  v1's apparent "effect" was exactly this non-specific extreme-magnitude shift.

**Definitive verdict (full wide grid, all 4 directions incl. random controls; analyze_steer
on the largest +α=64):** "causal but generic" on all 4 models — `Δmem(+64)` is non-trivial
(Qwen2.5 +0.22, Qwen3.6 +0.11; Qwen3-32B/gemma −0.28/−0.32 = destabilise) but `mem−neg < 0`
on ALL four (memory-positives move LESS than negatives) and `mem−random ≤ 0` on 3/4. So the
extreme-α effect is a **generic global shift, never memory-specific**. Combined with the
flat ±4–±16 region: the memory probe direction is **epiphenomenal for the verbalized
judgment at fair magnitude, and only generically (non-specifically) perturbing at
model-stressing magnitude.** Not a control knob for the memory belief.

Plot: `data/plots/cross-model/fig10_steering.png`. Data: `results/steer_v2/` (±4, all 4
directions) + `results/steer_wide/` (±64, all 4 directions for 3 models; Qwen3.6 2/4 —
cluster access ended). Verdict summary: `results/steer_wide/fig10_steering_summary.json`.
