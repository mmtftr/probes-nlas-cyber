[ai-generated]

# Archived results — OLD (flawed) dataset

These are the result artifacts (metrics JSONs + plots) of experiments 02–05,
run on the **first** `dataset.jsonl` built by `scripts/build_dataset_sven.py`.
That dataset was found (2026-05-31) to not match the intended research design.
**The scripts in each experiment dir are unaffected and correct** — only the
*activations* (and hence these numbers) were computed on the wrong data. See
`decisions/0002-dataset-before-after-contrast.md` for the decision and
`../../REBUILD-PLAN.md` for the re-run plan.

## The problem (three issues)

The old dataset was a **completion / streaming** construction, not the intended
**SVEN before/after contrast**:

1. **Wrong negatives.** Both pos and neg were cut from `func_src_before` (the
   *vulnerable* file). `func_src_after` (the fix) was never used (`SVEN-after`
   rows = 0). positive = vulnerable file truncated *at* the vuln emission point;
   negative = the **same vulnerable file** truncated *earlier* (`SVEN-before-leadup`),
   before the bug appears. The negative is "vulnerable code, not there yet," not
   secure code.
2. **Truncated prefixes, not full snippets.** `code` ends mid-statement at the
   completion point (54% of rows end mid-identifier, e.g. `...values(%d`).
3. **Length confound.** By construction `neg_end < pos_end`, so the positive is
   the longer prefix in **766/767 pairs (100%)** (mean 1383 vs 1287 chars). The
   length baseline (ex-AUC 0.575) partly reflects this — the probe could exploit
   "how much has been emitted" rather than "is there a vuln."

## Intended design (the rebuild)

positive = full `func_src_before`; negative = full `func_src_after` (the fix);
token-labels = the diff'd vulnerable lines on the positive. Same function in each
pair (tight contrast, differs only by the fix), comparable length, and a
token-level probe still covers the streaming case via per-token firing.

## Findings on the OLD data (qualitative — to be re-checked on the rebuild)

Preserved so the rebuild has a baseline to compare against. Treat as
**suggestive, not established** (wrong dataset):

- **02 layer sweep + variance (5 splits).** Optimal depth fraction did NOT
  generalize: Gemma peaked early-mid (~L19, frac 0.31, ex-AUC 0.708±0.018) +
  a final-layer rebound, with a mid-late dead zone; Qwen a broad late plateau
  (~L41, frac 0.65, 0.716±0.010). Single-split "best layers" were split-lucky;
  per-model val-selection recommended.
- **03 loss × α.** `span_max_loss_neg_incl` was a no-op (Δ within ±0.01). α=1
  beat the default α=10 by +0.01–0.03 ex-AUC (monotone in α); token-AUC ~flat
  in α (example-vs-token trade-off).
- **04 richer probes.** MLP head helped both models (+0.02–0.04); layer-concat
  helped only Gemma (distributed signal) and only with an MLP. Best: Gemma
  concat{9,19,26,61}+mlp512 = 0.763±0.009; Qwen L41+mlp512 = 0.758±0.014.
- **05 probe vs verbalized.** Probe > the model's own yes/no judgment by
  Δ≈+0.058 on both (Gemma 0.720 vs 0.661; Qwen 0.737 vs 0.679) — an
  "introspection gap." BUT confounded by the truncation: the model was asked
  yes/no about an incomplete prefix, which likely handicapped the verbalized
  baseline. Not a clean result.

These should be **regenerated** on the before/after dataset before any are
treated as real.
