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
- Run (login node), sequential (debug-qos MaxSubmit=1):
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

## Results

_(pending run)_
