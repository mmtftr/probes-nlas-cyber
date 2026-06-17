[ai-generated]

# exp-33 — Example-level TPR @ 1% FPR, by language and CWE

Closes the standing open thread "TPR @ 1% FPR — TODO (user, 2026-06-12)" and the
blog future-work item NXT4. Reports the *operating-point* number a deployed
monitor lives at, not just AUC, for the two flagship models and the lexical
ceiling, sliced by language and CWE.

## 1. Aim
At a frozen low-FPR operating point (FPR = 0.01), how much does each example-level
scorer actually *detect*? Hypothesis (from the lexical-ceiling story): TPR
concentrates in Python / injection CWEs and is ~0 on C / memory CWEs, and the
char-n-gram baseline tracks the probe — i.e. the operating-point view tells the
same lexical story as AUC, but more starkly.

## 2. Inputs (all reused, no re-extraction)
- **Models:** Qwen2.5-Coder-32B-Instruct, gemma-3-27b-it (the flagship of each family).
- **Scorers (example-level, one score per held-out function):**
  - **probe** — commit-position linear probe (the Fig-7 "strong example probe"),
    re-scored from exp-30 kept hidden states at each model's deployable (layer, C)
    in `30-.../results/introspection_probe.json`. Primed prompt.
  - **char-n-gram** — exp-31's strongest char config (`char_3_5_100k`) on the raw
    function text; model-independent; vectorizer fit train-only.
  - **verbalized** — model's own P(yes) at the same commit position (npz `p_yes`).
- **Data/split:** SVEN `data/dataset.jsonl` (1430), group-clean held-out
  `data/sven_split_meta.json` → 292 test functions (146 vuln / 146 patched),
  141 pair groups. Identical split to exp-29/30/31.
- **Negatives matched to CWE (user-directed):** each patched negative inherits the
  CWE of the positive(s) in its split group. Groups are the 141 split groups, not
  strictly one-pair-each (one collision group holds 6+6 rows, all CWE-089); the
  code enforces a single unique positive CWE per group, so the match is
  well-defined and every kept CWE slice is class-balanced.

## 3. Outputs
- `results/operating_point.json` — per model × scorer × slice:
  `{n_pos, n_neg, auc, tpr@1%fpr (ROC-interpolated), tpr_ci (pair-clustered boot),
  tpr_discrete (smallest threshold with FPR≤0.01), fp_allowed}`.
- Blog figure `fig_operating_point` (added to `docs/blog/make_figs.py`), inserted
  after Fig-7.

## 4. Result format
Two-panel (one per flagship model) grouped bars: TPR @ 1% FPR for {probe,
char-n-gram, verbalized} across slices {overall, Python, C/C++} ∪ {CWE-089, 125,
078, 476, 022, 416, 079} (held-out positives ≥ 10; 787 n=5 / 190 n=4 dropped, n
footnoted). Pair-clustered bootstrap CIs; n annotated per slice.

## 5. Threshold convention (the one implementation fork)
- Metric = **ROC-interpolated TPR at FPR = 0.01**, computed **per slice** on that
  slice's matched positives+negatives (`np.interp(0.01, fpr, tpr)` from
  `sklearn.roc_curve`). Per-slice (not one global threshold) because the user
  asked to match negatives to CWE → each slice gets its own FPR control.
- At these n, FPR = 0.01 allows 0–1 false positives, so the discrete operating
  point is effectively "score above (almost) every CWE-matched patch"; the
  interpolated value is reported as primary and the discrete one alongside.
  `tpr_discrete` and `fp_allowed` are stored so the coarseness is auditable.
- **Gates (must pass before any number is trusted):** probe refit AUC ==
  exp-30 deployable test_auc (≤2e-3); char AUC == exp-31 stored (≤5e-3);
  verbalized npz-p_yes AUC == exp-30 stored (≤5e-3).

## Interpretation hints
- Probe TPR high on Python/injection, ~0 on C/memory, with char-n-gram ≈ probe in
  every slice → confirms the lexical-ceiling reading at the operating point (claim
  #4). This is the expected outcome.
- Probe TPR clearly > char-n-gram in a *memory* CWE slice (CI-separated) → would be
  a genuine above-lexical operating-point signal (would echo exp-28's CWE-787 lead);
  treat as exploratory given the small n.
- Verbalized ≪ probe everywhere → "reading beats asking" carries to the operating
  point too (finding #5/#8). Verbalized ≈ probe → asking is as good as reading here.
