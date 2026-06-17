[ai-generated]

# exp-27 STATUS

**State: DONE** (2026-06-12 ~01:40) — second pass: codex **PASS** (zero
mismatches, no blocking issues; one helper-script nit, fixed) + Opus **PASS**
(~50 numbers verified, zero mismatches, "ready for the user"). Ledger row
added to docs/project-log.md; results posted to Slack. NOT committed (per
brief — awaiting user).

v3 run complete, all gates green. **The review fix materially corrected the
verdict:** with CIs computed for every variant, token-unigram on CWE-125
matched-patch = 0.591 [0.561,0.621] qwen / 0.584 [0.543,0.617] gemma — CI
excludes 0.5 on BOTH axes → "surface collapses to chance" was wrong for
token-identity; verdict moved to the brief's "in between" bucket (window/
n-gram family ∋0.5 in all 6 cells; unigram bar ≈0.59 covers ~½–⅔ of the 125
probe margin; probe tops every surface point in all 4 trusted 125/416 cells,
+0.04–0.07, no CI separation; "demonstrably non-lexical" NOT claimed).
RESULTS.md rewritten; codex + Opus second-pass verification in flight.

Review gate verdicts: codex **PASS-WITH-FIXES**, Opus **PASS-WITH-FIXES** —
no conclusion-reversing findings; both verified gates, metric, prior-work,
faithful exp-25 citation, fast_auc equivalence, pairing construction.
Reconciled fix list (all being applied in v3 rerun + RESULTS rewrite):
1. CIs for ALL surface variants incl. unigram/keyword (was: 4 of 7 CI'd) —
   "every variant CI∋0.5" claim must be artifact-backed. [codex HIGH / opus MED]
2. Soften "demonstrably non-lexical" OR add paired Δ test → v3 adds paired
   probeG−surface Δ-bootstrap (same resamples; probeG is locally available);
   specialized-probe contrast stays unpaired + softened. [codex HIGH]
3. "collapses to chance" → "indistinguishable from chance at this n";
   keep best-surface points visible (gemma-125 conlytr-comb 0.615). [codex MED]
4. Gemma axis: soften "cell-for-cell/byte-identical"; cite exp-16 historical
   ±0.000 anchor + exp-25 gate as shared-extraction anchors. [codex MED]
5. CI seeds: hash() → zlib.crc32 (PYTHONHASHSEED-stable). [codex NIT]
Opus NIT (one false "none above the band" line) → moot after fix 1 rewrite.

v1 run COMPLETE, all gates green on both axes:
- GATE1 (qwen): design-2 bit-repro, max dev 0.00e+00 (63 cells)
- GATE2 (both axes): all 36 token counts == exp-25 exactly
- GATE3 (both axes): lang_null == exp-25 to 1e-9

v1 headline (point estimates final): matched-patch memory surface ≈ chance
(char 125: 0.545/0.596, 416: 0.542/0.506, 476: 0.506/0.434 qwen/gemma; CIs
straddle 0.5) vs exp-25 probe 0.633/0.657 (125, CI>0.5 both). Injection stays
lexical under mp (089 char 0.975 ≥ probe). Rerun in progress adds CIs to the
conly-trained surface variant (strongest per-cell surface — load-bearing for
the ceiling claim); deterministic, reproduces v1 exactly plus extra CI fields.
Log: run.log (v1: run.v1.log).

- [x] Read brief + project-log + exp-24/25 code & results
- [x] Assets staged: dataset.jsonl + sven_split_meta.json (matches exp-25 archived copy),
      qwen32b + gemma1b L25 dumps copied from old repo (local, gitignored)
- [x] EXPERIMENT.md written (5 fields + gates)
- [x] Slack linked (`exp27-mp-surface`), brief posted
- [x] run_exp27.py written — stage A (design-2 verbatim repro + score capture),
      stage B (regime pools + count gates), stage C (regime evals + CIs);
      fast_auc==roc_auc_score parity verified incl. ties
- [x] Gates passed (design-2 bit-repro dev 0.0; exp-25 count match qwen+gemma
      36 cells each; lang_null == exp-25 to 1e-9) — both v1 and v2 runs
- [x] results/exp27_{qwen32b,gemma1b}_axis.json + RESULTS.md drafted
- [x] Review gate: round 1 codex PASS-WITH-FIXES + Opus PASS-WITH-FIXES
      (fixes applied via v3 rerun — materially corrected the verdict);
      round 2 codex PASS + Opus PASS on the rewritten RESULTS.md
- [x] project-log ledger row; slack-cc results post

No blockers.
