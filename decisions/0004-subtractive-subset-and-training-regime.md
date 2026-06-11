[ai-generated]

# 0004 — Subtractive subset + cleaned token-probe training regime

Date: 2026-06-07

## Context
The exp-16 logit explorer surfaced a label-quality bug: diff-derived
`token_labels.evidence` is whole-line and marks the *removed* side of a fix,
without filtering cosmetic hunks — so comments and unchanged tokens get labeled
"vulnerable" (e.g. eid 480 FliDecode: the marked span is a comment; the real
CWE-125 fix is an unmarked *addition*). Measured impact (Qwen-32B L23): 11.2% of
positive training tokens are non-code; whole-line over-marks positives ~3.3×
(4771→1454 tight); **34% of vuln examples (241/715) are additive/cosmetic**
fixes with no localizable code change. Whole-line labeling also creates
contradictory supervision — the same unchanged line is positive in the vuln
example and negative in its safe pair (near-identical activations, opposite
labels).

## Decision
1. **SVEN-subtractive subset.** A vuln example is kept iff its fix
   deletes/replaces ≥1 *live-code* character in `before` (tree-sitter live-code ∩
   difflib delete/replace span — char-level, model-independent). Additive/cosmetic
   fixes are dropped together with their safe pair ("drop-pair"). Result: 956
   examples (478 vuln+safe pairs); 237 additive pairs removed. SVEN-base (full
   1430, old whole-line logic, loss unchanged) is retained untouched for
   continuity.
2. **Cleaned training regime** (applies to the subtractive experiments):
   - Positives = (granularity-positive ∧ is_code). **Non-code tokens are never
     positive** (standing rule).
   - Granularity is an option: **token** (tight changed chars; default) or
     **line** (whole-line).
   - Negatives compared two ways: **X = comments-ignored** (`mask_negatives=
     "code_only"`, non-code dropped) vs **Y = comments-as-negative** (`"none"`).
   - **Loss unchanged** (span-max). The subset guarantees every kept vuln example
     has ≥1 positive token, so no example-level-positivity hack is needed (an
     idea floated then rejected as confusing).
3. **Cross-comparison.** Train probes on each subset; evaluate every probe on
   both subsets' held-out tests under one honest label (tight ∩ is_code,
   code-only). Additive transfer measured at the example level (pair-ranking),
   since additive examples have no positive token.

## Consequences
- exp-19 implements this (`plans/cross-model-probe-generalization/19-subtractive-regime/`):
  build_subtractive.py, train_grid.py (8 configs/model from cached acts), cluster
  drivers. Activations re-extracted once per model (the prior dumps were deleted;
  acts are label-independent so one extraction serves the whole grid).
- Result (7/7 models, exp-19 RESULTS.md): subtractive training is
  **performance-neutral** on localizable vulns (mean base→sub 0.756 ≈ sub→sub
  0.755); **additive vulns are undetectable** by a token-localized probe
  (mean pairAcc-add 0.43 ≈ chance vs pairAcc-sub 0.76) regardless of training;
  **token ≥ line** (6/7) and **X ≈ Y** (≤0.01) on the honest eval. → For
  token-probe work, the honest target is the subtractive subset; reporting on the
  full base inflates apparent performance with whole-line/additive artifacts.
- Future probe experiments should default to subtractive + token + is_code-gated
  positives, and report additive separately as an (currently unsolved)
  example-level detection problem. Generator fix (per-token granularity +
  cosmetic-hunk skip) is still owed upstream in build_dataset_before_after.py;
  exp-19 computes labels in the harness for now.
- Supersedes the implicit whole-line labeling assumed since
  [[0002-dataset-before-after-contrast]] for token-probe *training* (base dump
  unchanged).
