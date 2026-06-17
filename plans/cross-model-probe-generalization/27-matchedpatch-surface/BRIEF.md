[ai-generated]

# exp-27 brief — surface baselines under matched-patch negatives

You are an autonomous Claude Code session in tmux `probes`, spawned by the
coordinator session (`probes:1`). The user has pre-authorized this experiment
("send some agents to chase it"). Work in THIS repo
(`~/p/probes-nlas-cyber-clean` — the source of truth; the old
`~/p/probes-nlas-cyber` is a read-only backup with local data payloads).

**Read first:** `docs/project-log.md` (ledger + conventions), then
`plans/cross-model-probe-generalization/24-surface-baselines/` (EXPERIMENT.md,
`features.py`, `substrate.py`, `designs.py`, RESULTS.md) and
`plans/cross-model-probe-generalization/25-allclean-language-matched/`
(EXPERIMENT.md, `deconfound.py`, RESULTS.md).

## Aim

exp-25 showed per-CWE memory probes under the **matched-patch** control (score
the vulnerable function's tokens only against its OWN fixed version's tokens)
retain signal for CWE-125 (0.633 qwen32b / 0.657 gemma1b, CIs > 0.5), weakly
for CWE-416, none for CWE-476. But the **char-n-gram surface baseline was never
run under matched-patch negatives**, so we cannot yet say the residue is
non-lexical. Close that gap: run exp-24's surface baselines (char-n-gram LR at
minimum; keyword/unigram if cheap) under exp-25's three negative regimes —
all-clean (repro anchor), C-only, matched-patch — for CWE-125/416/476, on the
identical splits and token sets exp-25 used.

## Constraints & pointers

- Pure CPU/local — no cluster needed. Surface features come from the raw text
  (exp-24's `features.py`); negatives construction is in exp-25's
  `deconfound.py` (reuse its functions rather than reimplementing).
- `data/dataset.jsonl` + `data/sven_split_meta.json` are gitignored here; copy
  them from `~/p/probes-nlas-cyber/data/` first.
- Default metric `tokens_code_auc`; bootstrap-over-examples 95% CIs like exp-25.
- Train surface models on the SAME training recipe exp-24 used for per-CWE
  cells (train on CWE-X positives vs all-clean negatives) AND evaluate under
  each negative regime — mirroring exactly what exp-25 did with the probes, so
  the comparison is cell-for-cell.

## Outputs

`results/` JSONs + a RESULTS.md with the head-to-head table:
per CWE × regime: probe (exp-25 numbers, cite) vs char-n-gram (yours), with CIs.

## Interpretation hints

- Surface ≥ probe under matched-patch → the CWE-125 residue is lexical after
  all; blog claim dies (the post's SUR1/SUR2 get rewritten to a clean negative).
- Surface ≈ 0.5 while probe stays ~0.63–0.66 → first demonstrably non-lexical,
  non-language signal in the project; blog SUR2/FUT1 upgrade.
- In between → report the margin with CIs; no overclaim.

## Process (MUST follow)

1. Write a 5-field EXPERIMENT.md (Aim/Inputs/Outputs/Result format/
   Interpretation hints) before coding.
2. `slack-cc link exp27-mp-surface` and post the brief summary to your thread;
   post progress there, final results via `slack-cc results` ONLY after step 4.
3. Keep a STATUS.md in this dir current (the coordinator polls it).
4. Review gate per CLAUDE.md before any result reaches the user: cj/codex
   adversarial pass + Opus subagent pass (metric/prior-work/methodology/
   conclusion checklist).
5. Update `docs/project-log.md` ledger row for exp-27 when done. Do NOT commit
   unless the user asks.
