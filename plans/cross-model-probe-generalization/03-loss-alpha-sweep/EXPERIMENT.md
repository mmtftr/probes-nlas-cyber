[ai-generated]

# 03 — Span-max loss sweep: negative-inclusive variant × α

Step 3 of `../PLAN.md`. Exps 01–02 fixed the loss (paper-faithful span-max,
α=10) and swept layers. This sweeps the **loss** itself on the best layers from
the exp-02 variance study, with error bars over 5 group-clean splits.

1. **Aim** — two questions. (a) Does adding a span term to *negative* examples
   help? The baseline span-max span term is one-sided: BCE(1, max in-span p) for
   positives, nothing for negatives. New variant `span_max_loss_neg_incl` adds
   BCE(0, max-over-all-tokens p) for negatives — directly penalising clean code's
   highest false-alarm token, which is exactly the example-level score we eval on.
   (b) How sensitive is probe quality to **α** (the in-span up-weight of the
   per-token BCE term), for both loss variants? Hypothesis: neg_incl lifts
   example-AUC most where token-AUC is good but example-AUC lags (e.g. Gemma L26);
   α has a broad optimum near the default 10.
2. **Inputs** — cached per-layer activation memmaps from exp 02
   (`runs/layersweep_<slug>/acts`, reused — no re-extraction, no model load).
   Grid: loss ∈ {base, neg_incl}, α ∈ {1, 5, 10, 20, 50}, seed ∈ {42–46}.
   Layers: **Gemma-3-27B {9, 19, 26, 61}**, **Qwen2.5-Coder-32B {34, 41, 52, 63}**
   (early / example-peak / token-peak / last for Gemma; plateau-start / peak /
   old-single-split-best / last for Qwen). Probe: AdamW lr=1e-3, 30 epochs, hard
   labels, internal val seed=7 fixed. 2×5×4×5 = 200 cells/model.
3. **Outputs** — on scratch `runs/lossalpha_<slug>/`: one `cells/cell_*.json` per
   cell, aggregated `metrics_loss_alpha.json` (per (layer,loss,α): mean/std ex- &
   tok-AUC over seeds + per-layer best + neg_incl−base delta). Plot locally.
4. **Result format** — `loss_alpha_sweep.png`: grid (row=model, col=layer) of
   example-AUC vs α with base-vs-neg_incl lines + ±1 std bands. Per-layer table:
   best (loss, α), and mean neg_incl−base Δ with whether it clears 1 std.
5. **Interpretation** — neg_incl Δ > +1 std at a layer ⇒ the negative span term is
   a real lift (worth adopting); Δ within ±1 std ⇒ no effect, keep paper-faithful.
   A flat α curve ⇒ probe is α-insensitive (keep 10); a peak ⇒ tune α per the ADR.
   Whether the best (loss, α) differs Gemma-vs-Qwen feeds the same per-model-vs-
   shared question as the layer policy.

**Built-in sanity tie-in:** the (base, α=10, seed=42) cell must reproduce exp-02's
single-split value at that layer (same code path, same canonical split).

## For agents

- New loss: `src/training/train_probe_spanmax.py` — `span_max_loss(..., neg_incl=)`
  + convenience `span_max_loss_neg_incl`; `train_one_layer(..., alpha=, neg_incl=)`.
- Files: `loss_alpha_sweep.py` (GPU-sharded cell grid, resumable, reuses cached
  acts), `aggregate_loss_alpha.py`, `submit_loss_alpha.sh` (one debug job/model,
  no extraction), `plot_loss_alpha.py`.
- Run (login node), sequential (scheduler MaxSubmit=1):
  `MODEL=google/gemma-3-27b-it LAYERS=9,19,26,61 bash .../submit_loss_alpha.sh`
  then `MODEL=Qwen/Qwen2.5-Coder-32B-Instruct LAYERS=34,41,52,63 bash .../submit_loss_alpha.sh`.
- Est. ~8–10 min/model (200 cells / 4 GPUs, ~5–15 s each).

## Decisions (this experiment)

- *neg_incl semantics* `TODO(adhoc-decision)`: negative span term = BCE(0, **max
  over ALL tokens**), the symmetric mirror of the positive term and a direct
  surrogate for the example-level max-pool metric. Alternatives not taken:
  top-k mean of negatives, or a margin loss. Chosen for minimality + metric match.
- *Best-layers, not full sweep:* a full-layer × 10-config sweep is ~100 min > the
  90-min debug cap; best-layers × 5 splits is fast and adds variance bands. The
  layers cover the distinct depth regimes exp-02 found, so we don't miss an α that
  would change which region wins.
- *α grid {1,5,10,20,50}:* log-ish, brackets the default 10 by ±one decade. α only
  scales the per-token BCE term (the span term is unweighted), so the base-vs-
  neg_incl contrast is a clean factorial on the span term.

## Results (2026-05-31, jobs 2441540 Gemma / 2441612 Qwen)

`loss_alpha_sweep.png`; `metrics_loss_alpha_gemma.json`, `metrics_loss_alpha_qwen.json`.
All 200 cells/model done. **Sanity tie-in passes:** (base, α=10, seed=42) at
Gemma L19 = 0.683, identical to exp-02's single-split L19 seed-42 value.

**Finding 1 — neg_incl is a no-op.** The negative-inclusive span term never moves
example-AUC beyond split noise: Δ(neg_incl − base) ranges [−0.017, +0.011] across
all (layer, α) cells of both models, every cell within ±1 std (~0.01–0.02). The
bands overlap everywhere in the plot. → **Do not adopt; keep paper-faithful
span-max.** Likely because once ω→1 the positive span term already shapes the
decision boundary, and clean code's max is held down by the surviving per-token
BCE; an explicit negative max term is redundant.

**Finding 2 — α=1 dominates; the default α=10 was suboptimal.** example-AUC is
**monotone-decreasing in α** at every layer of both models:

| layer | α=1 | α=10 (old default) | α=50 |
|---|---|---|---|
| Gemma L19 | **0.720** | 0.708 | 0.659 |
| Gemma L26 | **0.688** | 0.655 | 0.593 |
| Qwen L34 | **0.731** | 0.707 | 0.703 |
| Qwen L41 | **0.737** | 0.716 | 0.694 |

Dropping α 10→1 buys **+0.012 to +0.033 AUC** (more for Qwen, and most for the
token-peak layer Gemma L26: +0.033). Up-weighting the rare in-span tokens hurts —
it over-fits the probe to the few annotated positions at the expense of the
example-level max-pool we actually score on.

**Best cells:** Gemma L19 @ α=1 (0.720±0.020, base); Qwen L41 @ α=1 (0.737±0.012
base / 0.746±0.012 neg_incl — the neg_incl edge is within noise). Both beat their
exp-02 α=10 numbers (0.708 / 0.716).

**Open question — α=1 is the swept boundary.** The curve is still falling toward
α=1, so the optimum may be **α<1** (down-weight in-span tokens below context) or
even α→0 (drop in-span weighting entirely). Not yet tested; flagged for a cheap
follow-up at α ∈ {0.1, 0.3, 0.5} on the top layers. `TODO(adhoc-decision)`:
ratify α=1 (or lower) as the new default into an ADR before the next sweep.

**Cross-model:** both findings hold for Gemma AND Qwen — unlike the layer policy
(exp 02), the loss knobs generalize: neg_incl off, α low. So α is a shared
default, the layer is per-model.
