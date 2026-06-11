[ai-generated]

# 19 subtractive-regime — live state (overnight 2026-06-06 → 07)

✅ COMPLETE — all 7 models trained + cross-evaluated; RESULTS.md (7/7), ADR 0004,
fig_cross.png done; posted to #probes-results. Files uncommitted (commit on request).

✅ CV PASS COMPLETE (2026-06-07) — 5-fold×3-seed, all 7 models. vLLM extraction
enabled on cluster (Qwen; gemma-3=HF, no EAGLE3). RESULTS_CV.md + fig_cv.png;
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
- [x] verify activations gone on scratch; dataset md5 matches cluster
- [x] build subtractive subset (build_subtractive.py) → 956 ex
- [x] train_grid.py + self-test (line labels reproduce dumped y, 561266/561266 MATCH)
- [x] run_subtractive.sh + submit_subtractive.sh
- [x] EXPERIMENT.md briefing
- [x] code review of harness (opus subagent) — ZERO blocking bugs; added example-level pair-rank for additive transfer (token-AUC undefined there)
- [x] rsync src/ + experiment dir to cluster
- [x] job 1 (Qwen-32B, gemma-12b-it/4b/1b) — JID 2485364 DONE in ~26 min, all 4 DONE no FAILED, 40/40/40/24 configs
- [x] submit job 2 (gemma-27b-it, gemma-12b-pt, Qwen-7B) — JID 2485395 RUNNING nid007057; monitor task bg4wwxdpr
- [x] analyze_grid.py validated on job-1 (tables + fig_cross.png); needs `uv run --with matplotlib`
- [~] job 2 (gemma-27b/12b-pt, Qwen-7B) — JID 2485395, ran on cluster; results on scratch but UNREACHABLE
- [x] RESULTS.md (4/7) + ADR 0004 written; analyze_grid.py done for 4 models (RESULTS_TABLES.md + fig_cross.png)
- [x] posted blocker + 4/7 findings + plot to #probes-tmux
- [ ] BLOCKED: pull job-2 metrics → re-run analyze on all 7 → finalize → post 7/7 to #probes-results

## RECOVERY (2026-06-07, cert renewed)
- Original job 2 (2485395) OOM-killed at 4 min: 27b extraction too heavy alongside others.
- 12b-pt + 7B had finished extraction → resubmitted as training-only job 2487357 (monitor b0jr4sde1).
- 27b partial acts deleted; will run ISOLATED next (1 model/job) to avoid OOM.
- Then: pull all 7 → analyze → finalize → #probes-results.

## 🚧 BLOCKER (2026-06-07 ~00:30) [RESOLVED — cert renewed]
HPC SSH cert expired 2026-06-07T00:26 (24h validity, ~/.ssh/HPC-key-cert.pub).
All jobs finished on the cluster; results safe on scratch but unreachable.
FIX: user runs `HPC-key sign` (MFA). Auto-resume monitor task **b7uokksou**
polls ssh every 20 min and re-invokes me on restore → I pull
subtractive_{gemma-27b-it,gemma-12b-pt,Qwen_Qwen2.5-Coder-7B-Instruct}, re-run
analyze_grid (uv run --with matplotlib), finalize RESULTS/ADR, post 7/7 to
#probes-results.
Local files uncommitted (commit on user request).

## Early findings (4/7 models, operating layer, token+X)
- base-trained ≈ subtractive-trained on the clean subtractive-test (Qwen32B 0.771 vs 0.772; 4b 0.732 vs 0.732) → dropping the additive third costs ~nothing.
- base-trained slightly > subtractive on base-test (it trained on the extra examples) — expected.
- ADDITIVE TRANSFER pairAcc-add ≈ 0.37–0.45 (≤ chance) for BOTH regimes → additive/missing-check vulns are ~undetectable by this token probe regardless of training. Strong justification for the subtractive split.
- line vs token, X vs Y: close on the honest eval.

## Cluster
- host `the cluster` (the cluster-ln001), acct compute-account, scratch user.
- $WORK=~/scratch/probes, runs/subtractive_<slug>/. debug partition, 1 job at a time.
