[ai-generated]

# exp-25 RESULTS — does the memory-CWE signal survive language deconfounding?

**Verdict (claim #3): narrowly rescued, strongly attenuated.** A genuine
memory-safety signal survives the airtight function+language-matched control
for the strongest CWE (CWE-125: token-AUC 0.63/0.66 across two models, CI
excludes chance) — it is **not** merely a C-vs-Python detector. But it is
**much smaller** than the confound-inflated all-clean headline (0.73 → 0.63),
CWE-416 is only weakly positive, and **CWE-476 collapses to chance**. The
original "memory signal exists at ~0.73–0.77" framing was inflated by a
language confound; what remains is a small, CWE-125-carried signal.

Metric throughout = **`tokens_code_auc`** (honest token-level ROC-AUC over
live-code tokens). Models: Qwen2.5-Coder-32B-Instruct L25, gemma-3-1b-it L25.
CIs are 1000× bootstrap **over examples** (not tokens). Trusted = `n_test_pos ≥ 10`.

## Reproduction gate — PASS (bit-exact)
Re-evaluating the held-fixed all-clean-trained probes under the original
all-clean design reproduces exp-21's `transfer_allclean.json` diagonal **to
full float precision** (max |Δ| = 0.000000) for all 9 CWEs, both models. So
the probes and eval harness are identical to prior work; everything below is
trusted. (Independently re-verified by both reviewers.)

## The confound, made concrete
SVEN memory CWEs are 100% C/C++; the all-clean negative pool is 53% Python.
A pure C-vs-Python indicator therefore scores ≈0.765 on that split. The
`lang_null` column (AUC of a C-vs-Python indicator on the *same* token set)
confirms it: under **allclean**, memory-CWE `lang_null` is **0.63–0.66** — i.e.
most of the 0.64–0.77 all-clean diagonal is explainable by language alone.

Removing the confound: under **conly** (C/C++-only-clean negatives) and
**matchedpatch** (the example's own paired patched safe-half), positives and
negatives are 100% C/C++, so `lang_null = 0.500` by construction.

## Decisive table — trusted memory CWEs (allclean-trained probes)
AUC `| lang_null`; CI = bootstrap-over-examples 95%. **matchedpatch is the
airtight control** (same function, same exact language → also controls
positive-side template/style leakage). conly is corroborating but leaves a
residual c-vs-c++ gap (see caveat).

### Qwen2.5-Coder-32B (L25)
| CWE | n | allclean | conly (CI) | **matchedpatch (CI)** |
|---|---|---|---|---|
| CWE-125 | 19 | 0.732 \| 0.631 | 0.675 [0.613,0.745] | **0.633 [0.575,0.701]** ✓>.5 |
| CWE-416 | 14 | 0.766 \| 0.649 | 0.714 [0.595,0.824] | **0.610 [0.514,0.709]** ✓>.5 (marginal) |
| CWE-476 | 16 | 0.640 \| 0.642 | 0.579 [0.517,0.676] | **0.544 [0.483,0.633]** ✗ incl .5 |

### gemma-3-1b (L25)
| CWE | n | allclean | conly (CI) | **matchedpatch (CI)** |
|---|---|---|---|---|
| CWE-125 | 19 | 0.734 \| 0.633 | 0.668 [0.599,0.741] | **0.657 [0.593,0.731]** ✓>.5 |
| CWE-416 | 14 | 0.769 \| 0.650 | 0.697 [0.631,0.756] | **0.603 [0.525,0.677]** ✓>.5 |
| CWE-476 | 16 | 0.619 \| 0.644 | 0.560 [0.506,0.605] | **0.507 [0.454,0.552]** ✗ incl .5 |

**Reading:**
- **CWE-125** survives the airtight matched-patch control in **both** models
  (0.633 / 0.657, CIs exclude 0.5). This is the load-bearing positive result:
  a real, language- and template-independent memory signal — but only ≈0.63–0.66,
  far below the 0.73 all-clean headline.
- **CWE-416** is weakly positive under matched-patch (0.610 / 0.603); qwen's CI
  lower bound (0.514) is marginal, gemma's (0.525) firmer. Suggestive, not strong.
- **CWE-476** collapses to chance under matched-patch in both models — no signal
  above surface confounds.

## C-only-trained probes (retrain with C-only-clean negatives)
Retraining the probes with language-matched negatives gives essentially the
same conly-eval diagonal (qwen: 125=0.677, 416=0.678, 476=0.641; gemma:
125=0.611, 416=0.668, 476=0.542) — the all-clean-trained probe is **not**
relying on the Python negatives to define its direction. Consistent with the
above: the surviving signal is modest and CWE-125/416-carried.

## Pooled memory probe (one direction for 125+416+476+787+190)
Pooling **dilutes**: under conly the pooled probe gives qwen 125=0.672,
416=0.576, 476/190≈0.58; gemma 125=0.671, 416=0.499. Per-CWE specialization
matters — there is no single strong "memory direction"; the recoverable signal
is concentrated in CWE-125.

## Cross-validation — is the conly diagonal a single-split artifact? (No)
Grouped 5-fold × 3-seed CV (pair-level folds, no straddling; job 2510132,
COMPLETED). CV retrains from scratch per fold; mean ± std over the 15 pooled
folds. **CV covers `allclean` and `conly` only** — `matchedpatch` negatives are
the example's *own* paired safe-half (example-intrinsic, no held-out clean pool
to fold), so the decisive control is not CV-able this way; its split-robustness
is already carried by the bootstrap-over-examples CIs above. CV therefore
corroborates the **corroborating** regime, not the decisive one.

Memory CWEs, CV mean ± std (15 folds), vs the single-split conly diagonal:
| CWE | qwen32b conly CV | (single-split) | gemma1b conly CV | (single-split) |
|---|---|---|---|---|
| CWE-125 | 0.662 ± 0.047 | 0.675 | 0.577 ± 0.025 | 0.668 |
| CWE-416 | 0.716 ± 0.057 | 0.714 | 0.676 ± 0.064 | 0.697 |
| CWE-476 | 0.617 ± 0.040 | 0.579 | 0.585 ± 0.056 | 0.560 |

`lang_null = 0.500` exactly across all conly CV folds (by construction).
**Reading:** the conly diagonal is stable across folds — CWE-125/416 sit
above chance in both models (CV mean − 1 std > 0.5 for 125 qwen/gemma and 416
both), so the corroborating result is not a lucky split. CWE-476 is weakly
above chance under conly CV too (0.59–0.62) — but conly is not airtight
(residual c-vs-c++), and 476 still **collapses to chance under the decisive
matched-patch control**, so CV does not rescue it. Injection positive controls
are rock-stable (089 ≈ 0.98–0.99, 078 ≈ 0.96–0.98 conly CV). **CV does not
change the verdict's direction.**

## Positive control — injection CWEs (pyonly = their language-matched regime)
Injection signal is lexical/taint, not language, so it should survive its own
language-matched regime — and it does. pyonly: CWE-089 0.959/0.928,
CWE-022 0.899/0.868, CWE-078 0.879/0.837, CWE-079 0.762/0.811 (qwen/gemma).
Under the stricter matched-patch, 089/078/022 stay strong (0.93/0.83/0.77 qwen)
but **CWE-079 drops to ≈chance** (0.532 qwen) — its all-clean score was
template/language, not function-local. Control behaves as designed.

## Caveats (from adversarial review; reconciled into the framing above)
1. **conly is not airtight.** `lang_null=0.5` under conly only rules out
   C-family-vs-Python; the clean pool is ~85% C even within C-family, so a
   residual c-vs-c++ or style detector could lift conly AUC. **This is why the
   verdict leads with matchedpatch** (same function + exact same language),
   which closes that gap and also controls positive-side vuln-template leakage.
   conly is reported as corroborating evidence, not as the decisive test.
2. **Small n.** Trusted cells are n=19/14/16; CIs are wide. CWE-190 (n=4) and
   CWE-787 (n=5) are **untrusted** and excluded from all conclusions.
3. **CV done (allclean+conly only).** Grouped 5-fold×3-seed CV (job 2510132,
   COMPLETED) confirms the conly diagonal is not a single-split artifact (see
   Cross-validation section). It does **not** cover matchedpatch (example-
   intrinsic negatives can't be folded); the decisive control's split-robustness
   rests on its bootstrap-over-examples CIs. Does not change the verdict.

## Bottom line for the project ledger
Claim #3 ("memory signal exists; under-allocation not absence") is **rescued
only narrowly**: a genuine but small (~0.63) memory signal exists for CWE-125,
language- and template-controlled and replicated across two models; CWE-416 is
weakly positive; CWE-476 shows nothing above surface confounds. The headline
0.73–0.77 all-clean numbers were **language-confound-inflated** and should not
be cited as the strength of the memory signal.

## Artifacts
- `results/repro_gate_{qwen32b,gemma1b}.json` — gate diagonal (bit-matches REPRO_TARGET).
- `results/deconfound_{qwen32b,gemma1b}.json` — full per-CWE × 4-regime × 2-probe-set
  table, lang_null column, allclean-trained diagonal bootstrap CIs, pooled memory.
- `probes_dc.npz` (in `./runs/exp25/<slug>/`) — saved W/b (allclean + conly + pooled).
- `results/cv/<slug>/<regime>_<cwe>_<seed>.json` — 84 CV cells (allclean+conly ×
  7 CWEs × 3 seeds × 2 models; 5 folds each). `results/cv_aggregate.json` —
  pooled mean/std/lang_null per (model, regime, CWE).
