[ai-generated]

# 0002 — Dataset: SVEN before/after full-function contrast

Date: 2026-05-31
Status: Accepted (supersedes the completion-truncation dataset for the
cross-model-probe-generalization plan)

## Context

The first `dataset.jsonl` (built by `scripts/build_dataset_sven.py`) was a
**completion/streaming** construction that diverged from the intended design,
discovered while reviewing experiment 05:

- Both positive and negative rows were cut from `func_src_before` (the
  *vulnerable* file); `func_src_after` (the fix) was never used.
- positive = vulnerable file truncated at the vuln **emission point**;
  negative = the **same** vulnerable file truncated **before** the vuln
  (`SVEN-before-leadup`) — i.e. "vulnerable code, not yet emitted," not secure
  code.
- Truncated prefixes (54% end mid-identifier), and a **length confound**: the
  positive is the longer prefix in 766/767 pairs (100%).

The intended target (research-framing §1, §2.1) is the model's belief about
input-stream code, with a **contrast pair** (research-framing Q2) holding the
task fixed and varying only the vulnerability. The standard SVEN realization of
that is the **before (vulnerable) vs after (fixed)** pair, not a same-file
truncation contrast.

## Decision

Rebuild `dataset.jsonl` as the **SVEN before/after full-function contrast**:

- **positive** (label=1) = full `func_src_before`; `token_labels` = the diff'd
  vulnerable lines (the `char_changes`/`line_changes` regions) for the span-max
  span term.
- **negative** (label=0) = full `func_src_after` (the fix); empty `token_labels`.
- Pair grouped by `(_file_name, _func_name)` so before/after never straddle the
  split (existing `pair_group_key`; `sven_split_meta.json` keys are group names,
  so the split is stable under the row rebuild — verify after).

Rationale: tight contrast (same function, differs only by the security fix);
comparable length (length confound largely removed); a **token-level** span-max
probe still covers the streaming case via per-token firing, so this single
dataset serves both the "reading complete code" and "streaming" framings.

The completion/truncation framing remains a *legitimately different* experiment
(streaming-monitor threat model) and its results are archived, not deleted, at
`plans/cross-model-probe-generalization/archive/old-dataset/`.

## Consequences

- **Re-extraction required.** Cached activations under
  `runs/layersweep_<slug>/acts` are for the old (truncated) code → stale. Both
  models (Gemma-3-27B, Qwen2.5-Coder-32B) must be re-extracted on the new code.
- **Re-run 02–05.** Their *scripts* are correct as-is (only the data changed);
  re-run on the new activations. Old quantitative results are archived and
  marked suggestive-only until regenerated.
- A new/adapted builder is needed (none of the three existing builders emit
  full before/after pairs — they all truncate). See `../REBUILD-PLAN.md`.
- Resolves the research-framing §8 open item "canonical contrast structure:
  SVEN pairs vs synthesized" → **SVEN before/after pairs**.
