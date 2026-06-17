[ai-generated]

# exp-28 — PrimeVul deep-dive: surface baselines, cross-dataset CIs, matched-pair

**Model/layer (probe side):** Qwen2.5-Coder-7B-Instruct L16 — exp-26's cached
per-token logits, NO retraining. **Metric:** `tokens_code_auc` (chance 0.5).
CIs = bootstrap over examples, 95%. Untrusted: <10 test pos. **Repro gate
48/48 PASS** (`results/gate.json`): the local substrate reproduces exp-26
bit-exact (n_neg_test_tokens=287,864; per-CWE n_pos_tok; all 12 diagonal cells;
all 20 SVEN↔PV cells). Surface = exp-24 recipe (char-3-5-gram hashing LR,
±48-char window, per-CWE design-2 training: annotated y==1 live-code train
tokens vs subsampled clean-train code tokens; token-unigram LR secondary), on
the IDENTICAL token axis (extractor offsets), identical eval pools. Note:
"surface" here is the char-n-gram block alone (the cleanest lexical probe),
not exp-24's combined U+H+L primary — all other recipe knobs match exp-24
exactly (independently reproduced to 6 decimals in review).

---

## A. Surface baselines within PV (`results/pv_surface.json`)

### Family blocks — no memory cluster in surface either

| block (token-weighted) | probe (exp-26) | char-ngram surface |
|---|---|---|
| mem → mem off-diag | 0.536 [0.513, 0.564] | 0.531 [0.499, 0.556] |
| mem → other | 0.537 [0.518, 0.554] | 0.497 [0.454, 0.545] |
| other → mem | 0.539 [0.520, 0.559] | 0.534 [0.489, 0.573] |
| other → other off-diag | 0.495 [0.454, 0.541] | 0.452 [0.371, 0.552] |

Surface mem→mem off-diag (0.531) ≈ surface mem→other (0.497), CIs overlapping —
**the no-family-cluster verdict is not an artifact of the probe family: lexical
features don't pool by memory-family on PV either.** exp-26's conclusion holds
under the surface comparison. (Token-unigram blocks all ≈ 0.50–0.53.)

### Diagonal — probe vs surface, per CWE

| CWE | fam | n | probe (exp-26) | char-ngram | unigram |
|---|---|---|---|---|---|
| 119 | mem | 14 | **0.875** [0.643, 0.966] | 0.778 [0.460, 0.993] | 0.647 |
| 125 | mem | 47 | 0.611 [0.481, 0.753] | **0.655** [0.527, 0.790] | 0.559 |
| 190 | mem | 11 | **0.730** [0.617, 0.897] | 0.656 [0.463, 0.941] | 0.613 |
| 415 | mem | 10 | **0.635** [0.485, 0.903] | 0.556 [0.344, 0.908] | 0.605 |
| 416 | mem | 29 | **0.657** [0.470, 0.817] | 0.580 [0.409, 0.737] | 0.609 |
| 476 | mem | 39 | **0.692** [0.520, 0.785] | 0.500 [0.346, 0.676] | 0.504 |
| 787 | mem | 72 | 0.562 [0.492, 0.706] | 0.568 [0.496, 0.674] | 0.493 |
| 20 | other | 14 | 0.514 [0.405, 0.749] | 0.540 [0.326, 0.638] | 0.486 |
| 200 | other | 16 | **0.940** [0.612, 0.983] | 0.919 [0.405, 0.984] | 0.759 |
| 369 | other | 14 | **0.767** [0.627, 0.868] | 0.687 [0.445, 0.808] | 0.558 |
| 617 | other | 12 | **0.804** [0.497, 0.877] | 0.639 [0.460, 0.684] | 0.619 |
| 703 | other | 47 | **0.561** [0.463, 0.644] | 0.462 [0.371, 0.625] | 0.578 |

Probe ≥ char-ngram surface on **9/12** CWEs (mean 0.696 vs 0.628, +0.067;
losses: 125, 787, 20 — the latter two by <0.03); per-cell CIs overlap broadly
(small n), so the per-cell margins are descriptive, the 9/12 + mean-margin
pattern is the aggregate evidence. Two readings:

- **CWE-476 (NULL-deref) is the clearest non-lexical cell:** probe 0.692
  [0.520, 0.785] (CI excludes 0.5) vs surface 0.500 [0.346, 0.676] (point at
  exact chance). The two cells' CIs overlap (0.520 < 0.676), so the
  probe−surface gap is a point-estimate contrast, not itself
  significance-tested; what IS CI-backed is probe>chance + surface
  chance-consistent.
- **CWE-125 is the reverse:** surface 0.655 ≥ probe 0.611 within PV — the
  *within-PV* 125 diagonal is at/below the lexical ceiling (its probe story
  lives in cross-dataset transfer, §B/§C, not the within-PV diagonal).
- CWE-200 (0.94 probe / 0.92 surface) is largely lexical.

## B. Cross-dataset cells, now with CIs (`results/cross_cis.json`)

Point estimates reproduce exp-26's `cross_shared.json` exactly (asserted);
`*` = 95% CI excludes 0.5. SVEN-C cells for 787/190 untrusted (n<10).

| CWE | n_pv / n_svc | SVEN→PV | PV→PV | PV→SVEN-C | SVEN→SVEN-C |
|---|---|---|---|---|---|
| 125 | 47 / 19 | **0.668 [0.571, 0.765]*** | 0.611 [0.486, 0.752] | 0.617 [0.522, 0.699]* | 0.746 [0.681, 0.819]* |
| 416 | 29 / 14 | **0.706 [0.629, 0.860]*** | 0.657 [0.471, 0.813] | 0.857 [0.760, 0.935]* | 0.709 [0.573, 0.813]* |
| 476 | 39 / 16 | 0.494 [0.371, 0.787] | 0.692 [0.532, 0.786]* | 0.609 [0.481, 0.691] | 0.573 [0.490, 0.671] |
| 787 | 72 / 5† | 0.635 [0.592, 0.725]* | 0.562 [0.495, 0.709] | 0.443† | 0.658† |
| 190 | 11 / 4† | 0.755 [0.683, 0.816]* | 0.730 [0.616, 0.904]* | 0.904† | 0.822† |

exp-26's single-split ordering is now CI-backed: **CWE-125 and CWE-416
SVEN→PV transfer excludes chance**; CWE-476 stays chance-consistent (wide CI —
"no detectable transfer", not a significance-tested collapse). New: CWE-787
SVEN→PV 0.635 also excludes chance (modest).

## C. PV matched-pair — vuln vs its OWN fix (`results/pv_matchedpair.json`)

The exp-25 matched-patch construction on PV: positives unchanged (CWE test
vuln code tokens, y_tok labels); negatives = the SAME pairs' fixed versions'
code tokens. CIs resample pairs. pairAcc (P(max-token vuln > fix), exp-19) is
SECONDARY. Surface = the §A all-clean-trained char-ngram models, evaluated in
this regime (mirrors what exp-25 did with probes). `*` = CI excludes 0.5.
(CWE-369/20 CIs retain 974/999 of 1000 reps after degenerate-resample drops.)

Unigram column added post-review after exp-27's split verdict landed (unigram
survives matched-patch on SVEN-125): same pools/seeds/bootstrap, code paths
already audited (`results/pv_matchedpair_unigram.json`).

| CWE | fam | n pairs | PV-probe | SVEN-probe | char-ngram | unigram |
|---|---|---|---|---|---|---|
| 119 | mem | 14 | **0.797 [0.518, 0.910]*** | — | 0.712 [0.356, 0.956] | 0.621 [0.519, 0.672]* |
| 125 | mem | 47 | 0.570 [0.446, 0.712] | **0.660 [0.568, 0.757]*** | 0.623 [0.501, 0.753]* | 0.553 [0.449, 0.640] |
| 190 | mem | 11 | 0.731 [0.549, 0.927]* | 0.736 [0.629, 0.843]* | 0.687 [0.384, 0.975] | 0.602 [0.501, 0.754]* |
| 415 | mem | 10 | 0.648 [0.510, 0.865]* | — | 0.490 [0.267, 0.849] | 0.573 [0.492, 0.672] |
| 416 | mem | 29 | 0.617 [0.449, 0.766] | **0.650 [0.565, 0.794]*** | 0.543 [0.366, 0.710] | 0.577 [0.521, 0.639]* |
| 476 | mem | 39 | 0.612 [0.431, 0.722] | 0.409 [0.289, 0.704] | 0.493 [0.341, 0.677] | 0.487 [0.410, 0.522] |
| 787 | mem | 72 | 0.545 [0.467, 0.694] | **0.645 [0.602, 0.733]*** | 0.537 [0.467, 0.638] | 0.480 [0.452, 0.551] |
| 20 | other | 14 | 0.488 [0.370, 0.704] | — | 0.549 [0.381, 0.618] | 0.492 [0.412, 0.537] |
| 200 | other | 16 | 0.893 [0.536, 0.936]* | — | 0.889 [0.385, 0.942] | 0.729 [0.418, 0.761] |
| 369 | other | 14 | 0.619 [0.429, 0.782] | — | 0.567 [0.281, 0.731] | 0.525 [0.361, 0.586] |
| 617 | other | 12 | 0.743 [0.443, 0.825] | — | 0.640 [0.508, 0.726]* | 0.617 [0.584, 0.641]* |
| 703 | other | 47 | 0.566 [0.459, 0.655] | — | 0.488 [0.380, 0.657] | 0.582 [0.530, 0.608]* |

- **SVEN-trained probes survive matched-patch ON PRIMEVUL** for CWE-125
  (0.660*), 416 (0.650*), 787 (0.645*), 190 (0.736*) — and are
  chance-consistent for 476 (0.409). This matches exp-25's ordering (125/416
  survive, 476 dead) on an independent dataset under the airtight
  same-function control, with CIs — with one upgrade: 416 on PV is as strong
  as 125 (0.650 vs 0.660), firmer than exp-25's "weakly positive" label.
  CWE-787 is newly positive here (untestable in exp-25's trusted set — SVEN
  has only 5 C test positives).
- **Against the two lexical baselines** (per-cell, `*` = CI excludes 0.5):
  - **CWE-787 is the cleanest cell on PV:** SVEN-probe 0.645* while BOTH
    lexical baselines are chance-consistent (char 0.537, unigram 0.480).
  - **CWE-416:** SVEN-probe 0.650* tops both baselines, but unigram also
    clears chance (0.577*).
  - **CWE-125:** SVEN-probe 0.660* tops both, but char-ngram also clears
    (0.623*).
- **Paired probe−lexical Δ-bootstrap** (the decisive contrast exp-27 named;
  same pair-resamples score all models per rep; TIE-CORRECT average-rank AUC —
  a first pass used a tie-blind rank AUC that inflated tie-heavy unigram
  deltas; caught in mini-review, regenerated, and verified against an
  independent sklearn recomputation; `results/pv_matchedpair_delta.json`):
  - **CWE-787: probe exceeds BOTH lexical baselines with Δ-CIs excluding 0**
    — Δ_char +0.108 [+0.036, +0.223], Δ_uni +0.164 [+0.121, +0.222]. **This
    is the project's first CI-separated probe-over-lexical margin.**
  - Δ_uni also excludes 0 for 125 (+0.107 [+0.031, +0.193]) and 190 (+0.133
    [+0.046, +0.181]) — but in those cells Δ_char does NOT (char-ngram is
    the stronger lexical baseline there), so "probe > best lexical" stays
    point-estimate-level for them. Borderline, not excluded: 119 Δ_uni
    [−0.000, +0.247], 416 Δ_uni [−0.004, +0.192].
  - Multiplicity caveat: these are NOMINAL per-cell CIs in an exploratory
    24-test family (4 exclude 0; ≈1.2 expected by chance), not
    familywise-corrected. 787's Δ_uni bound (+0.121) is far from 0 and
    survives any reasonable correction; its Δ_char bound (+0.036) is less
    robust. 787 is the only cell clearing both baselines.
- pairAcc ≈ chance everywhere (0.36–0.69, small n) — at the forced-choice
  example level neither probe nor surface separates vuln from fix
  (consistent with exp-22's near-chance pairAcc on C/C++); the signal is
  token-level. Subtractive-only slices (smaller n) shift no conclusion.

## D. Verdict updates

1. **"No memory-family direction" — STRENGTHENED.** Surface blocks show the
   same flat structure as probe blocks within C/C++; the absence of a family
   cluster is not a probe-family artifact.
2. **"Per-CWE diagonal is real" — survives with a sharper shape.** Probe ≥
   surface on 9/12 diagonals (mean +0.067). CWE-476 is the showcase
   non-lexical within-PV cell (0.692 vs surface 0.500) — but its signal is
   PV-idiosyncratic (no detectable cross-dataset transfer; SVEN-probe
   matched-patch chance-consistent). Conversely CWE-125's within-PV diagonal is lexical-level,
   but its cross-dataset transferable component is what survives every
   control. "Idiosyncratic but real" refines to: *each per-CWE cell is a
   different mix of lexical, dataset-idiosyncratic, and transferable signal;
   no single label fits all.*
3. **Cross-dataset memory transfer is now CI-backed** (125/416 robustly, 787
   newly, 190 small-n) and survives the matched-patch control on the target
   dataset — converging with exp-25 from an independent design. CWE-476 shows
   **no detectable cross-dataset transfer** (SVEN→PV 0.494, wide CI spanning
   0.5; SVEN-probe matched-patch 0.409, also chance-consistent) —
   directionally consistent with exp-25's collapse, but not itself a
   significance-tested negative.
4. **Joint reading with exp-27** (surface-under-matched-patch on SVEN, landed
   2026-06-12 with a split verdict: char/window family ∋0.5 on all trusted
   memory×mp cells, but token-unigram survives mp on SVEN-125 at 0.591/0.584).
   exp-28 is the PV mirror, and the joint picture across both datasets is:
   **which lexical family clears a memory×mp cell is dataset-dependent**
   (SVEN-125: unigram yes/char no; PV-125: char yes/unigram no; PV-416:
   unigram yes/char no; PV-787: neither), **while the specialized probe clears
   chance on 125/416 on both datasets and on 787 on PV** — and tops every
   lexical point estimate in every trusted memory×mp cell. The paired
   Δ-bootstrap (exp-27's named decisive test, run here on the PV side)
   upgrades **PV-787 to the project's first CI-separated probe-over-lexical
   cell** (both Δs exclude 0; Δ_uni robustly). The SVEN-side paired Δ
   (needs the L25 acts, extraction off-repo) remains open.

## Stretch — second model (gemma-3-12b-it L15)

Job (pv_family.py unchanged, cached gemma acts, CPU) was mid-run —
9/12 probes trained at 55 min — when the cluster went unhealthy
(2026-06-12). Resumable (probes checkpointed,
logits cached); fetch/resubmit on recovery and append here.

## Provenance

Local CPU only for §A–C; probe side entirely from exp-26 cached logits
(remote cache → `assets/`, gitignored). Harness: `pv_deepdive.py`
(stages gate/A/B/C, resumable; surface scores cached) +
`mp_unigram_addendum.py` + `mp_paired_delta.py`. Gates in
`results/gate.json`. Review: full dual gate (codex adversarial + Opus
subagent, both with independent recomputation) on A–C; focused codex
mini-review on the addenda — its tie-blind-AUC catch fixed in
`fast_auc_batch` (now average-rank, sklearn-exact incl. ties) and all
bootstrap CIs regenerated (char-ngram CIs unchanged to 3 dp; unigram CIs
shift ≤0.05 with no conclusion change; deltas as reported above). Stretch
job: pv_family.py (exp-26, unchanged) on cached gemma PV acts.
