[ai-generated]

# Span-max loss tuning

The probe loss (Obeso, Arditi et al. 2025, arXiv 2509.03531 §3) and what we've
learned tuning it. Code: `src/training/train_probe_spanmax.py`.

## The loss
`logit_i = w·h_i + b`, `p_i = σ(logit_i)`:
```
L = (1−ω)·Σ_T w_i·BCE(y_i, p_i)  +  ω·Σ_spans BCE(1, max_{i∈span} p_i)
```
- `w_i = α` for in-span tokens else 1 (**α** up-weights the rare positive tokens).
- **ω** annealed linearly 0→1 over steps: start as token classification, end
  optimizing the per-example max-pool (the eval metric).
- Defaults: AdamW lr=1e-3, 30 epochs, batch_examples=8, hard labels
  (`label_window=0`), internal val split seed=7 for epoch selection.
- **α: use ~1, not the old default 10** (exp 03). example-AUC is monotone-
  decreasing in α for both Gemma and Qwen at every layer tested; α 10→1 buys
  +0.012–0.033 AUC. Up-weighting the rare in-span tokens over-fits the few
  annotated positions and hurts the example-level max-pool. The α=1 boundary is
  the swept minimum — α<1 untested, possibly better still.
  - **α is an example-vs-token trade-off, not a free win:** token-AUC is ~flat
    in α (≤0.02, within noise) and if anything *rises* with α — high α aids
    per-token localization but costs the example max-pool. Keep α=1 because
    example-AUC is the headline and its α-effect is larger; token-AUC barely moves.

## Eval metric
- Token-AUC (all held-out tokens) **and** example-AUC (per-example score =
  `max_i σ(logit_i)`, max-pool). Example-AUC is the headline. Always report the
  trivial baselines (random / length / regex) for honest lift.

## neg_incl variant (exp 03)
- Baseline span term is **one-sided**: it pulls positives' in-span max *up* but
  never explicitly pushes clean code's max *down* (negatives are shaped only by
  the per-token BCE, which ω anneals away). `span_max_loss_neg_incl` adds
  `BCE(0, max-over-all-tokens p)` for negatives — a direct surrogate for the
  example-level max-pool. Enable via `train_one_layer(..., neg_incl=True)`.
  **Verdict (exp 03): no-op.** Δ(neg_incl − base) ∈ [−0.017, +0.011] over all
  (layer, α) cells of both models, every one within split noise. Keep it off —
  once ω→1 the positive span term + surviving per-token BCE already hold clean
  code's max down, so the explicit negative term is redundant.

## Layer selection — does NOT generalize across models (exp 02)
- Optimal depth fraction is model-specific: Gemma-3-27B peaks early-mid
  (~frac 0.31) with a final-layer rebound and a mid-late dead zone; Qwen2.5-
  Coder-32B has a broad late-middle plateau (~frac 0.5–0.67). A single fixed
  fraction hurts one of them. **Val-select the layer per model.** The cheap
  `{n/4, n/2, 3n/4, n−1}` grid can miss the peak (it did for Gemma).

## Variance: always resample the split (exp 02)
- A single group-clean split mis-rates a layer by **~0.02–0.05 AUC**; both models'
  single-split "best layers" sat at the lucky high end of their own 5-split
  spread. Report mean ± std over ≥5 seeds, and select layer/α on the band, not a
  point estimate. Probes are cheap on cached activations
  ([[probe-activation-extraction]]) so this is ~10 min/model on 4 GPUs.
