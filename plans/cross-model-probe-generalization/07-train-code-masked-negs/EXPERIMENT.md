[ai-generated]

# 07 — Train-time code-masked negatives

Sweep 4. The other four sweeps (06) only change **scoring**; this one changes
**training**. It tests the hypothesis 06 raises directly: if `tokens_code_auc`
is low, is it because the span-max loss was *trained* to beat trivial negatives?

`src/eval/code_mask.py:16–24` flags this as known-but-unwired: span-max currently
treats every out-of-span token — comments, `def`/`class` signatures, imports,
whitespace — as a negative. That is an easy win that need not transfer to the
hard live-code-negative case. Applying the live-code mask **before training**
forces the probe to separate live-code-positive from live-code-**negative**
directly.

Depends on **06** (uses its acts + its best layer per model; uses the shared
`honest_token_aucs` helper for eval).

## 1. Aim

Does training with the live-code mask on negatives raise `tokens_code_auc`
(the deployed metric) versus the current all-tokens-negative training?

Hypothesis: if 06 shows a large `tokens_auc − tokens_code_auc` gap, masked-negative
training closes part of it (the probe stops spending capacity on trivial
negatives). If 06 shows no gap, this changes little (confirms the easy negatives
weren't the crutch).

## 2. Inputs

- **Acts:** 06's cached per-layer activations + `offsets.npz` (no re-extraction).
- **Models / layers:** each 06 model at its 06 best-`tokens_code` layer (keep it
  to best-layer to bound cost; expand only if a model is surprising).
- **Change under test:** in `train_probe_spanmax`, gate the negative set by
  `code_only_mask(code, lang, offsets)` so out-of-span **non-live-code** tokens are
  neither positives nor negatives (excluded from the loss), not silent negatives.
  - `TODO(adhoc-decision)`: exclude masked tokens from the loss entirely, vs keep
    them as down-weighted negatives. Proposal: **exclude** (cleanest test of the
    hypothesis). Flag for review.
- **Eval:** `tokens_code` (and `tokens` for reference) via the 06 helper.
- **Seeds:** 5 (42–46), matching 03/04 variance convention.

## 3. Outputs

`runs/codemasktrain_<slug>/metrics.json`:
`{model, layer, train_mask ∈ {none, code_only}, tokens_code_auc_mean/std,
tokens_auc_mean/std, ex_auc_mean/std (rides along), per_seed}`.

## 4. Result format

Table `model | layer | train_mask | tokens_code_auc | Δ vs none | tokens_auc`.
Paired (none vs code_only) per model, 5-seed mean ± std.

## 5. Interpretation hints

- `code_only` train ⇒ higher `tokens_code_auc` ⇒ the easy negatives *were* the
  crutch; masked training is the recipe fix, and 06's headline numbers under
  current training understate the achievable honest signal.
- No change (or worse) ⇒ the trivial negatives weren't what the probe leaned on;
  a low 06 `tokens_code_auc` reflects a genuinely weak live-code signal, not a
  training artifact → look to richer probes (04) / different layers / dataset.
- `tokens_auc` drops while `tokens_code_auc` rises ⇒ expected and fine: the probe
  reallocates from trivial negatives to the hard case. Report both; don't read the
  `tokens` drop as regression.

## For agents

- Touches shared training code (`src/training/train_probe_spanmax.py`) — keep the
  mask behind a flag (`--mask-negatives code_only|none`, default `none`) so 02/03/
  04/06 reproduce unchanged. Immutability: thread the mask through, don't mutate
  the existing label arrays in place.
- Cheap relative to 06 (no extraction, best-layer only) — run after 06 lands and
  its best layers are known. If the window has no slack, this can wait; it is the
  rescue path that only matters if 06 shows the gap.
- Same cluster channel / qos / provenance rules as 06.

---

## Results (2026-06-01)

Paired none vs code_only (exclude non-live-code negatives from the loss), 5 seeds,
at each model's 06 val_tokens_code best layer. **delta (code_only − none) ≈ 0 on
every model** (−0.013 … +0.002; slightly negative on average):

| model | none | code_only | delta |
|---|---|---|---|
| gemma-3-1b-it | 0.756 | 0.747 | −0.009 |
| gemma-3-1b-pt | 0.758 | 0.744 | −0.013 |
| gemma-3-4b-it | 0.758 | 0.760 | +0.002 |
| gemma-3-4b-pt | 0.754 | 0.751 | −0.003 |
| gemma-3-12b-it | 0.762 | 0.759 | −0.004 |
| gemma-3-12b-pt | 0.761 | 0.753 | −0.008 |
| gemma-3-27b-it | 0.752 | 0.753 | +0.001 |
| Qwen2.5-Coder-32B | 0.782 | 0.783 | +0.001 |

**Conclusion:** training-time masking of trivial negatives does not help (slightly
hurts, plausibly from less training signal). The probe was never leaning on
comments/signatures — consistent with the mask dropping only ~30% of tokens and the
06 finding that tokens_code ≈ tokens. The hypothesised rescue is unnecessary.
