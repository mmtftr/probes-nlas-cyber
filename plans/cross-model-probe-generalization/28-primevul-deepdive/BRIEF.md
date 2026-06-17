[ai-generated]

# exp-28 brief — PrimeVul, analyzed properly

You are an autonomous Claude Code session in tmux `probes`, spawned by the
coordinator session (`probes:1`). The user has pre-authorized this experiment
("the primevul was not evaluated as good as we can — analyze it a little
better"). Work in THIS repo (`~/p/probes-nlas-cyber-clean` — source of truth;
old `~/p/probes-nlas-cyber` is a read-only backup with local data payloads).

**Read first:** `docs/project-log.md` (ledger rows exp-22/26 + open threads),
then `plans/cross-model-probe-generalization/22-primevul-paired/` and
`plans/cross-model-probe-generalization/26-primevul-within-family/`
(EXPERIMENT.md / RESULTS.md / LOG.md / `pv_family.py`, `results/pv_within.json`,
`results/cross_shared.json`).

## Aim

exp-26's PrimeVul analysis is preliminary: one model (qwen7b L16), no surface
baselines, single-split SVEN→PV cells without CIs, and PV's paired structure
(vuln function + its fix) unexploited. Strengthen it in priority order:

1. **Surface baselines within PV** (pure CPU): char-n-gram LR on the identical
   PV splits/token sets — per-CWE diagonal + the family-blocks analysis
   (mem→mem vs mem→other). Question: does the per-CWE diagonal (0.61–0.88) and
   the no-family-cluster result survive a surface comparison? (exp-24's
   `features.py` is the reference implementation.)
2. **CIs for the cross-dataset cells**: bootstrap-over-examples CIs for the
   SVEN→PV shared-CWE cells (125 0.668 / 416 0.706 / 476 0.494) — exp-26 saved
   what it needs in `results/` or can rescore from kept per-token scores; check
   LOG.md for what was persisted before assuming re-extraction is needed.
3. **PV matched-pair eval** (PV is paired!): score each PV vulnerable function
   against its own fix (the exp-25 matched-patch construction, exp-19 pairAcc
   as secondary) for the per-CWE probes — with the surface baseline in the
   same regime. This is the PV mirror of exp-25/27.
4. **Stretch (only if 1–3 land and time remains):** a second model. Check
   whether PV activations for another model already exist on scratch (read
   `docs/guides/` + exp-22 LOG for paths; cluster access via `fc`, debug
   partition, per CLAUDE.md). Do NOT start a big extraction without posting to
   your Slack thread first.

## Constraints & pointers

- PV dataset location: exp-22's scripts/LOG say how the PrimeVul-Paired slice
  was built and where the jsonl lives; the old repo may hold a local copy under
  `plans/.../22-primevul-paired/`. Find it before rebuilding.
- Default metric `tokens_code_auc`; per-CWE trust gates like exp-26 (≥10 test
  pos); flag tiny-n cells; bootstrap CIs everywhere.
- Cluster OOM/CPU lessons for PV-sized data: `docs/guides/` (large-dataset OOM
  guide) — train probes on CPU if CUDA OOMs, drop numactl.

## Outputs

`results/` JSONs + RESULTS.md: (a) PV diagonal + family blocks, probe vs
surface side by side with CIs; (b) SVEN→PV cells with CIs; (c) PV matched-pair
table probe vs surface; (d) updated verdict on "no memory-family direction"
and "per-CWE diagonal is real" under surface comparison.

## Interpretation hints

- Surface ≈ probe on the PV diagonal → even single-language per-CWE signal is
  lexical; exp-26's "idiosyncratic but real" downgrades to "lexical".
- Probe > surface on diagonal but blocks unchanged → genuine per-CWE signal,
  still no family direction (current blog story holds, strengthened).
- PV matched-pair probe > surface → converges with exp-27 toward a real
  non-lexical residue; report jointly with exp-27's outcome.

## Process (MUST follow)

1. 5-field EXPERIMENT.md before coding.
2. `slack-cc link exp28-pv-deepdive`; brief to your thread; results via
   `slack-cc results` only after the review gate.
3. Keep STATUS.md in this dir current (the coordinator polls it).
4. Review gate per CLAUDE.md (cj/codex + Opus subagent) before any result
   reaches the user.
5. Update `docs/project-log.md` ledger when done. Do NOT commit unless asked.
