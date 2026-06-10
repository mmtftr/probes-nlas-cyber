[ai-generated]

# exp-23 — Language-stratified rescore of the general probe (LOCAL, CPU-only)

## Aim
Quantify how much of each published GENERAL-probe number survives **within language**,
and decide whether the project's central injection-vs-memory split is a *family* effect
or a *language* (C-vs-Python) effect. A tri-review (2026-06-09) found the exp-10/21
"memory recovery" eval is language-confounded (injection ≈92% Python, memory =100% C/C++,
clean negatives ≈half Python). This experiment is the corrected, language-controlled read,
computed entirely from exp-16's persisted per-token logits — no GPU, no retraining.

## Inputs
- **Logits**: `plans/.../16-token-logit-dump/results/logitdump_<model>/logits_layer{NN}.npz`
  (flat token table: `logit, prob, y, example_id, char_start, char_end, is_test, is_code`).
  7 models at their operating layer: Qwen2.5-Coder-32B L25 / -7B L16,
  gemma-3-1b-it L25 / -4b-it L7 / -12b-it L15 / -27b-it L19 / -12b-pt L13.
- **Dataset**: `data/dataset.jsonl` (1430 rows; fields `code, cwe, lang∈{python,c,cpp},
  label, token_labels`). lang grouping: `c` cell = c+cpp; language indicator c/cpp=1, python=0.
- **Split**: exp-16's split recovered from the npz `is_test` flag (group-clean pairs, seed-42, 20%).
- **Label regime**: PRIMARY = `line/code` (npz `y` ∩ `is_code`) — this is the published
  `tokens_code_auc` headline. SECONDARY = honest `tok/code-X` (difflib tight-diff span ∩ is_code).
  Both stated per number.

## Outputs (this dir)
- `rescore_language.py` — deterministic, seeded (numpy seed 42 for bootstrap).
- `results/<model>.json` — per-model raw numbers (gate, within-language, family×lang, lang-null, pairs).
- `results/_summary.json` — cross-model roll-up.
- `RESULTS.md` — tables + "so what". `STATUS.md`, `LOG.md`.

## Result format
1. Format gate: recomputed vs historical tokens_code_auc, Δ (must be ≤0.001).
2. Within-language: pooled AUC + per-example-mean AUC, n_ex, n_tok, 1000× example-bootstrap 95% CI,
   for python-only and c/cpp-only.
3. Family×language matrix: C-inj / C-mem / Py-inj / Py-mem pooled AUC vs same-language negatives,
   with n test pos examples (cells <10 flagged UNTRUSTED).
4. Language-null table: AUC of the bare language indicator (and reverse) on each published
   eval design's exact token set — the corrected null the project should have compared against.
5. Pooled vs per-example-averaged AUC (inflation check).
6. (secondary) within-pair before/after max-logit pairAcc per family per language.

## Interpretation hints
- **C-inj ≈ C-mem ≈ 0.5–0.6** → the family split is a language split; claim #2 must be reworded,
  claim #3's foundation weakens further.
- **C-inj ≫ C-mem (≥0.75 vs ≤0.6)** → family split is real beyond language (n caveat); claim #2 survives.
- **Published per-CWE numbers inside the language-null band** → those cells carry no language-independent
  evidence of vulnerability signal.
