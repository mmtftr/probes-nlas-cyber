[ai-generated]

> **RE-EXEC (2026-06-09):** the original run reported *pair-accuracy* and was
> retracted. This experiment is now scored on the project-default
> **`tokens_code_auc`** by reusing the saved per-CWE probes (see `RESULTS.md`,
> `recompute_tokenauc.py`). The aim/inputs below are unchanged; the metric and the
> interpretation hints are corrected.

# exp-21 — Per-CWE probes & cross-CWE transfer (on `tokens_code_auc`)

## Aim
Do CWE-specialized probes detect only their own CWE, or do they transfer across
CWEs? Hypothesis: the injection CWEs (089/078/022/079) share the lexical
string-sink feature → a hot injection×injection block (a SQLi probe fires on
cmd-inj, path, XSS); the memory-safety CWEs (125/416/476/190/787) are
idiosyncratic with no shared lexical cue → a cold block, weak even on the
diagonal. I.e. there is no "CWE-specific detector", only a shared injection
family + a memory blind region (exp-20 thesis, tested per-CWE).

## Inputs
- Models: **Qwen2.5-Coder-32B-Instruct L25** (big) + **Gemma-3-1b-it L25** (small)
  — max size & family contrast; 1b detected the most examples in exp-20.
- Operating-layer token activations **re-extracted** (the exp-19 dumps were
  deleted; only offsets survived). KEEP this time (no delete) so the
  acts persist for any follow-up.
- SVEN-**subtractive** subset; honest **tight∩is_code** per-token labels (ADR-0004).
- Per-CWE probe = span-max linear head, **one-vs-paired-safe** (positives = tight
  code tokens of CWE-c vuln; negatives = code tokens of the matched safe pair).

## Imbalance handling (the binding concern)
Train pairs per CWE are wildly uneven (089=159 … 787=15; injection 339 vs memory
139). So each probe is trained at **two sizes**:
- **natural** — all available train pairs (best each probe can do);
- **balanced** — capped to `--balanced-n=15` pairs (= smallest CWE), seeded
  subsample, so the matrix is not "089 has 10× the data".
Every cell carries n_train / n_test; the small memory CWEs have tiny **test** n
(787=2, 190=3, 476=4) → report block-pooled means (which aggregate test pairs)
as the robust signal, with CIs, not individual noisy cells.

## Outputs (scratch `runs/percwe_<slug>/`, pulled here)
- `matrix.json` — CWE×CWE `pairacc`/`auc` (natural+balanced) on held-out test,
  `detrate` at per-CWE F1-max threshold, `block_*` 2×2 injection/memory means,
  n_train/n_test/thr.
- `probes_percwe.npz` (W/b per CWE, both variants), `logits_percwe.npz`
  (per-token logits of every natural probe — persisted, not downloaded).

## Result format
9×9 transfer heatmaps (natural & balanced, pair-acc + AUC) + 2×2 block table with
Wilson CIs; per-CWE self-detection (diagonal) vs transfer (off-diagonal).

## Interpretation hints
- Hot injection block + cold memory block → shared lexical feature, not
  per-CWE detectors. Off-diagonal injection transfer ≈ diagonal → the probe is a
  "string-sink" family detector regardless of which injection CWE trained it.
- If **balanced** training collapses 089's apparent edge → its strength was data
  volume, not a better feature.
- A memory diagonal that stays ≈chance even with a dedicated probe → memory is
  near-chance **in this (subtractive matched-patch) regime**. This is NOT "signal
  absent": exp-10 (full-SVEN, all-clean negatives) resolved memory to 0.73/0.77,
  so the gap is a **regime effect to disentangle**, not a refutation. Make no
  "(un)learnable" claim from this run (every memory cell has test n<10).

## For agents
Self-contained run (`run.sh` drives `train_percwe.py`; reuses
`extract_token_activations.py`, `train_probe_spanmax.py`, `code_mask`,
`train_eval`). Run on a GPU node. Local
machinery check: `validate_local.py` (pooled-probe stand-in reproduces the
injection/memory split).
