[ai-generated]

# exp-32 — per-CWE matched-patch consolidation

Re-presents the **matched-patch** evaluation as THE per-CWE result, replacing
the all-clean pooled per-CWE numbers (exp-10/exp-21) that the blog originally
reported. No new extraction or training: this reads already-computed,
dual-reviewed matched-patch numbers and tabulates them for every CWE.

## Why

The all-clean per-CWE eval ranks a CWE's vulnerable tokens against ~39k
unrelated clean tokens from other functions. The dataset is paired (each
vulnerable function has its patched twin), but that pooled metric throws the
pairing away, so language + project + template confounds inflate the score.
The correct per-CWE control uses the pairing: score each vulnerable function
only against its own patched counterpart (matched-patch), holding
language/project/style/template constant. The language null is then 0.5 by
construction for every single-language CWE (all memory CWEs are C/C++).

## Files

- `consolidate.py` — reads exp-25 (probe matched-patch + CIs) and exp-27
  (probe + lexical baselines under matched-patch), runs rigor gates, writes
  `results/percwe_matchedpatch.{json,md}`.
- `make_fig.py` — `results/fig_percwe_matchedpatch_{qwen,both}.png`.

## Rigor gates (in `consolidate.py`, all assert)

- **GATE 0** no-mix: the exp-25 and exp-27 files describe the same model/axis.
- **GATE 1** no-drift: exp-27's copied probe AUC equals exp-25's source.
  (NOT an independent re-score — exp-27 copies the probe number from exp-25.)
- **GATE 1b** shared token axis: each lexical baseline's matched-patch
  `lang_null` and `n_pos_tok` equal the probe's, proving probe and baselines
  were scored on the identical (pos, neg) token set.
- **GATE 2** memory CWEs have matched-patch `lang_null == 0.5` exactly.

## What the negative set actually is

Positives = the CWE's annotated vulnerable tokens. Negatives = the same
before-functions' OTHER live-code tokens PLUS their patched counterparts'
tokens. The counterpart pool is the same-file/same-function patched code
(usually the paired fix; ordinal pairing inside the 7 duplicate file/function
groups). No unrelated function contributes a negative.

## Review

Computation reviewed by two codex agents (`exp32-extract-audit`,
`exp32-method-audit`) and one Opus subagent. Consensus: numbers TRUSTWORTHY,
methodology is the correct leak-free control, consistent re-framing of
exp-25/27 (no contradiction/duplication). Wording constraints applied: no
probe-over-lexical contrast is CI-separated; matched-patch is the replacement
headline, all-clean kept only as confound context.
