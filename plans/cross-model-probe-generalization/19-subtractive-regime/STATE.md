[ai-generated]

# 19 subtractive-regime — live state (overnight 2026-06-06 → 07)

✅ COMPLETE — all 7 models trained + cross-evaluated; RESULTS.md (7/7), ADR 0004,
fig_cross.png done; posted to #probes-results. Files uncommitted (commit on request).

✅ CV PASS COMPLETE (2026-06-07) — 5-fold×3-seed, all 7 models. vLLM extraction
for Qwen; gemma-3=HF, no EAGLE3. RESULTS_CV.md + fig_cv.png;
CV addendum in RESULTS.md; posted to #probes-results. Result: base≈subtractive
within fold-std; additive pairAcc 0.24–0.43 below chance. exp-19 confirmed w/ error bars.

Autonomous run. Update as steps complete.

## Decisions (locked, from session)
- Subtractive subset: char-level membership (tree-sitter live code ∩ tight diff
  deletion), drop-pair. 956 ex (478 vuln+safe), 237 additive pairs dropped.
- Grid: {base, subtractive} × {line, token} × {X=code_only, Y=none}. is_code gates
  positives always. Loss unchanged. SVEN-base keeps old logic (retrained here only
  for cross-comparison).
- Cross-eval on common honest label (tight ∩ is_code, code-only) over
  subtractive/base/additive test sets.
- Re-extraction needed (acts deleted; verified scratch empty of layer npz). One
  extraction per model serves the whole grid.

## Progress
- [x] verify activations gone; dataset md5 matches the source
- [x] build subtractive subset (build_subtractive.py) → 956 ex
- [x] train_grid.py + self-test (line labels reproduce dumped y, 561266/561266 MATCH)
- [x] run.sh
- [x] EXPERIMENT.md briefing
- [x] code review of harness (opus subagent) — ZERO blocking bugs; added example-level pair-rank for additive transfer (token-AUC undefined there)
- [x] run 1 (Qwen-32B, gemma-12b-it/4b/1b) — DONE in ~26 min, all 4 DONE no FAILED, 40/40/40/24 configs
- [x] run 2 (gemma-27b-it, gemma-12b-pt, Qwen-7B)
- [x] analyze_grid.py validated on run-1 (tables + fig_cross.png); needs `uv run --with matplotlib`
- [x] RESULTS.md (4/7) + ADR 0004 written; analyze_grid.py done for 4 models (RESULTS_TABLES.md + fig_cross.png)
- [x] all 7 models analyzed → finalized → 7/7 results

## RECOVERY notes
- First attempt at run 2 OOM-killed: 27b extraction too heavy alongside others.
- 12b-pt + 7B had finished extraction → re-run as a training-only pass.
- 27b partial acts deleted; ran ISOLATED (1 model/run) to avoid OOM.

## Early findings (4/7 models, operating layer, token+X)
- base-trained ≈ subtractive-trained on the clean subtractive-test (Qwen32B 0.771 vs 0.772; 4b 0.732 vs 0.732) → dropping the additive third costs ~nothing.
- base-trained slightly > subtractive on base-test (it trained on the extra examples) — expected.
- ADDITIVE TRANSFER pairAcc-add ≈ 0.37–0.45 (≤ chance) for BOTH regimes → additive/missing-check vulns are ~undetectable by this token probe regardless of training. Strong justification for the subtractive split.
- line vs token, X vs Y: close on the honest eval.
