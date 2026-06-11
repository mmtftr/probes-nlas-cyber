[ai-generated]

# Honest evaluation of the probes research — 2026-06-09

Independent tri-review of the cross-model probe-generalization program
(exp-02→22, ADRs 0001–0004), requested by the user ("honest complete feedback").

**Reviewers (all independent, run in parallel):**
1. Opus methodology audit (Agent subagent) — per-claim evidence ratings, red flags.
2. Opus literature sweep (Agent subagent, web-verified) — novelty, scooping, field standards.
3. codex xhigh adversarial review (`cj` job `probes-sci-review`) — claim audit, alternative explanations.

The critical finding was then **verified by hand** against primary artifacts
(`data/dataset.jsonl`, `plans/.../21-per-cwe-cross-cwe/results/qwen32b/*.json`,
exp-06 EXPERIMENT.md). All three reviewers converged independently on the same
two weak points (claims #3 and #6) before reconciliation.

---

## Verdict

The engineering/epistemics infrastructure is genuinely excellent — better than
most published interp work. But the **central scientific claim (#3, "memory
signal exists; capacity-allocation")** currently rests on a language-confounded
eval design, and exp-21's "family-structured transfer" uses the wrong null
hypothesis. The strongest claim the evidence licenses today:

> Linear probes decode a token-local, SVEN-local signal dominated by
> injection/string-sink lexical cues. Memory-safety signal has **not** been
> demonstrated above a language-detector baseline.

The decisive deconfounding experiment is cheap (saved probes + KEPT acts), and
two findings (additive blind spot, honest token-level methodology) are solid
and novel.

---

## 1. The critical problem — claim #3 is confounded by language

Chain of evidence, each link checked against primary artifacts on 2026-06-09:

1. **Family ⊥ language is near-perfectly confounded in SVEN.** From
   `data/dataset.jsonl` (computed directly):
   - injection vulns (022/078/079/089): 378 py / 32 C+cpp → **92% Python**
   - memory vulns (125/190/416/476/787): 305 C+cpp / 0 py → **100% C/C++**
   - clean pool (`cwe==null`, the exp-10/21 negative pool): 378 py / 337 C+cpp
     → **53% Python by rows, ~32% by chars**.

2. **A pure language detector scores inside the headline band.** A probe that
   knows only C-vs-Python (zero vulnerability knowledge) gets token-AUC on the
   memory-diagonal eval design (memory-vuln tokens vs all-clean tokens):
   `AUC = P(neg=py) + 0.5·P(neg=C)` ≈ **0.66** (char-weighted) to **0.77**
   (row-weighted). Reported memory diagonal: **0.64–0.77**
   (CWE-125 0.732, CWE-416 0.766, CWE-476 0.640).

3. **The same language detector reproduces exp-21's "family structure."**
   Cross-family below chance (pure-language predicts mem→inj ≈ 0.20–0.27;
   observed 0.34/0.30), within-family above 0.5. So exp-21's inference —
   "off-diag CIs exclude 0.5 ⇒ shared family direction" — uses the wrong null.
   The correct null is the **language baseline (~0.66 for mem→mem)**, and the
   observed mem→mem off-diagonal (0.570) is *below* it. Same for inj→inj
   off-diag 0.600 vs pure-py-detector ~0.84.

4. **The controlled eval said chance.** The retracted matched-patch rescore
   (language- and function-matched negatives,
   `21-per-cwe-cross-cwe/results/qwen32b/matrix_tokenauc.json`, read directly):
   memory diagonal **0.523 / 0.526 / 0.544** (125/416/476). Its dismissal as a
   "training-regime artifact" has a real argument (difflib labels + no negative
   diversity cripple matched-patch *training*), but the 2×2 of
   {train regime} × {eval pool} was never completed. The decisive cell —
   **exp-10-trained probes, language-matched eval** — was never run. Keeping
   the confounded cell while retracting the controlled one reversed the
   epistemic ordering.

5. **The project knew about this fork and lost it.** exp-06 sweep-6
   (`06-honest-metric-sweeps/EXPERIMENT.md:200-235`): *"Language and CWE tell
   the same story"* + the open question "is C weak because the signal is
   absent, or because SVEN's C pairs are harder?". The MoC paper note
   (`docs/papers/yu2025-moc-secure-code.md`) records that Yu et al. framed the
   identical observation as a **per-language gap**. exp-10/21 answered the
   question with a design that cannot distinguish the two.

**Consequences.** Project-log claim #3, exp-21's headline finding, and the
"Memory regime question — RESOLVED" open-thread line are not currently
supportable. Claim #2 survives only in its within-language form (exp-06
sweep-6: within-Python 0.81 vs within-C 0.59); family-vs-language attribution
is untested — SVEN's ~32 C/C++ injection examples are the crossed cell, never
isolated. Diag > off-diag within memory (0.64–0.77 vs 0.57) does indicate a
CWE-specific component beyond pure language, but that component can be
repo/lexical idiosyncrasy; the controlled eval puts the *vulnerability*-specific
part at ≈ 0.

(Recorded in agent memory: `exp10-21-language-confound`; the earlier
"per-CWE memory signal exists" memory is marked contested.)

---

## 2. Claim-by-claim scorecard (project-log §2, three reviewers reconciled)

| # | Claim | Verdict | Core reason |
|---|---|---|---|
| 1 | Real honest token signal 0.75–0.82 | **Real number, mis-framed** | Reproducible (exp-16 ±0.000), deployable layer selection. But exp-20 shows the AUC is carried by lexical SQL/command-string recognition that also fires on *patched* code (3,125 FPs on fixed functions). It is an injection-sink signal, not "vulnerability belief." |
| 2 | Injection strong / memory weak | **Supported** (within-language form) | Most robust finding: 7+ models, two families, three independent experiment styles (exp-06 per-CWE, exp-20 bimodal detection, exp-22 C-slice). Family-vs-language attribution open. |
| 3 | Memory gap = capacity allocation; signal exists | **Unsupported as evidenced** | §1 above. Also: n = 14–19 test examples per memory cell, single split, no CV. |
| 4 | Pooled probe = lexical string-sink detector | **Strong** | exp-20 FN/FP taxonomy over all 6,119 FP spans; TPs localize sinks, FNs are memory (no lexical sink), FPs on patched SQL. Bimodal, family-agnostic across 7 models. Best mechanistic work in the project. |
| 5 | Verbalized weak; memory blind spot = prompt framing | **Moderate** | Internally consistent (exp-14/15/17), but built on example-AUC where the probe side saturates (max-pool sigmoid ≈ 1.0 on large models, exp-16) — so probe-vs-verbalized example-level comparisons are degraded on both sides. ~54 memory test positives. |
| 6 | Memory direction epiphenomenal | **Overclaimed** | One direction, one layer, additive-only, verbalized first-token readout, n≈40/cell. Honest claim: "additive single-layer steering of this direction does not move verbalized P(yes)." Sandoval et al. (2604.16697) argue the opposite ("inert until final layer") for adjacent directions. Needs ablation/patching + positive control to earn "epiphenomenal." |
| 7 | Additive fixes undetectable; subtractive regime | **Strong — methodological high-water mark** | Only claim with proper replication: grouped 5-fold × 3-seed CV, fold-stds reported, 7/7 models, mechanistic argument (additive fix ⇒ no positive token). Scope it to "this linear span-max regime." |

**Internal tension flagged by all three reviewers:** the project's *default*
regime (ADR-0004 subtractive/matched) is exactly the regime in which memory is
at chance, while claim #3 survives only in the *non-default* all-clean regime.
As written, the central thesis and the deployable default contradict each other.

---

## 3. Literature position (web-verified by the literature reviewer)

### Scooped / crowded

- **Yu et al., MoC, arXiv:2507.09508 (Jul 2025)** — linear probe on SVEN ≫
  prompting (79–82% vs 40–51%), per-CWE difficulty gap (framed as Python-vs-C
  language gap), per-CWE layer selection, probe-gated steering. Owns the
  generic "vulnerability is linearly decodable on SVEN" result.
- **Sandoval et al., "Surgical Repair of Insecure Code Generation",
  arXiv:2604.16697 (Apr 2026)** — 6×6 cross-CWE transfer matrix for security
  directions (diagonal ≫ off-diagonal; unified vectors keep 24–40% of per-CWE
  effect), three-tier CWE-knowledge pattern, rejects epiphenomenality
  ("hierarchical convergence"). Owns the transfer-matrix shape.
- **Wendlinger et al., "Security-by-Design… Concept-Driven Steering",
  arXiv:2603.11212 (Mar 2026)** — input-validation vs memory-handling
  subconcepts cluster separately in residual streams; token-level alignment.
  Qualitative version of the family-geometry claim.
- **Wang et al., "False Sense of Security", arXiv:2509.03888 (2025)** —
  probing-based detectors ride shallow lexical patterns, matched by n-gram
  baselines, collapse OOD. The cautionary mirror for exp-20; sets the baseline
  standard this project hasn't met (token-level lexical baselines).

### Field standards this project doesn't yet meet

- **Monitor bar (AI-control framing):** deception probes (Goldowsky-Dill et
  al., arXiv:2502.03407) report AUROC 0.96–0.999 *and recall at 1% FPR* — and
  still conclude "insufficient as a robust defence." High-stakes probes
  (McKenzie et al., arXiv:2506.10805) set the protocol: recall@fixed-FPR,
  calibration, shift robustness, cost-vs-black-box. Token AUC 0.75–0.82 with
  no FPR operating point does not enter that conversation.
- **Probing methodology:** control tasks / selectivity (Hewitt & Liang
  1909.03368; Belinkov 2102.12452 survey). Length/regex baselines exist in the
  repo but were only ever run at example level (exp-06,
  `src/remotes/the cluster/train_eval.py:137-147`) — never at the headline
  token-level granularity.
- **Causal claims:** steering-validity literature (Tan et al. 2505.22637;
  Arditi et al. 2406.11717; Marks & Tegmark 2310.06824) expects
  ablate-and-measure mediation with positive controls, not a single-magnitude
  additive null.
- **"Belief" framing:** unearned until probe output is linked to model
  behavior; reading↔generation transfer (research-framing §2.1) untouched.

### Still genuinely novel here

1. **Additive-fix blind spot of token-localized supervision** (exp-19,
   ADR-0004) — nobody has published this; clean, defensible, properly powered.
2. **Honest token-level eval methodology** — live-code masking,
   deployable-vs-oracle layer reporting, full logit persistence — vs the
   last-token probes everyone else uses.
3. **Capacity-allocation inversion** ("memory under-allocated, not absent") —
   *if and only if* it survives deconfounding (§1). Currently a hypothesis,
   not a finding.

---

## 4. What's genuinely good

- **exp-19** is venue-grade (grouped CV × seeds, fold-stds, mechanistic
  argument). Every load-bearing claim should look like exp-19.
- **exp-16/17/18 reproduction gates** (re-derive from persisted logits,
  ±0.000) — better provenance than most published work; caught the
  example-AUC saturation artifact.
- **exp-20** — the best and most honest analysis in the project; it disproves
  the project's own headline framing, and was run anyway.
- Split hygiene (pair-grouped, code-verified in
  `src/remotes/the cluster/train_eval.py:37-67`), deployable-vs-oracle layer
  discipline (ADR-0003), self-correcting metric history, documentation/ledger
  culture. Failures here are failures of *claim synthesis under motivated
  reasoning*, not of data handling.

---

## 5. Process findings

- **Review gate failed once at its job:** exp-21 passed Opus + codex pre-exec
  review with the confound intact. Add a checklist item: *"state the strongest
  non-vulnerability explanation, and what the correct null is for this eval
  design"* (confound hunt, not just metric/split check).
- **Provenance rule violated at scale:** exp-16→22 (incl. twice-retracted
  exp-21) are untracked in git; the retraction history exists only in prose.
  Commit the experiment dirs.
- **Number drift:** ledger claim #3 cites Qwen-32B "0.788"; verified exp-16
  reproduction is 0.776 at L25. Stale Tier-1 file
  (`RESULTS-2026-06-01-tier1-tier4.md`) still carries the pre-reversal causal
  conclusion alongside exp-13's opposite verdict, and exp-09's "capacity not
  the lever" ledger row omits Tier-4's layer-artifact caveat.

---

## 6. Recommendations, ordered by value-of-information

1. **Deconfound now — cheap and decisive.**
   - Zero-GPU, local: language-stratified rescore of exp-16's persisted
     per-token logits; isolate the ~32 C/C++-injection examples (the
     family×language crossed cell).
   - One small cluster job: eval saved `probes_allclean.npz` on
     **C-only clean negatives** and **matched-patch negatives** (completes the
     2×2; acts are KEPT on scratch).
   - If memory survives ≳0.65 within-C → claim #3 rescued, becomes the paper's
     spine. If it drops to chance → clean negative result captured *before*
     publication. Either outcome wins — provided it runs before more work
     stacks on claim #3.
2. **Token-level lexical baselines in every headline table:** bag-of-tokens /
   n-gram / regex + language indicator, same splits, same `tokens_code_auc`.
   Non-optional after exp-20 (the probe is known to be substantially lexical).
3. **Fix the ledger before the next experiment:** downgrade claims 1/3/6
   wording; un-retract matched-patch to "unresolved"; propagate 0.776; annotate
   stale Tier-1 file; commit the dirs.
4. **Re-aim at the actual goal (monitor of the model's own belief):**
   reading↔generation transfer; recall@fixed-FPR + calibration; promote
   PrimeVul (5.3× pairs, harder labels) from side-check to primary validation,
   all 7 models with CV. Drop "belief" from claims until probe output is
   linked to model behavior.
5. **Position the paper as diagnostic, not yet-another-probe:** the defensible
   spine is "what a vulnerability probe actually detects — string-sink
   lexicality, an additive-fix blind spot, and an honest eval methodology that
   exposes both" (the 2509.03888 lineage). The "probes detect vulns on SVEN"
   lane is MoC's.

---

## Summary

The project's most-cited finding is its least supported, and its
least-celebrated experiments (19, 20) are its best science. The infrastructure
already built is exactly what's needed to fix this in days, not months.
