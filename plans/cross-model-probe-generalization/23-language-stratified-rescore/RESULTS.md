[ai-generated]

# exp-23 — Results (2026-06-10)

Language-stratified rescore of exp-16's persisted **general-probe** per-token
logits. LOCAL, CPU-only, no retraining. 7 models at their operating layer.
All numbers are pooled token-level ROC-AUC over held-out **TEST** live-code
tokens. **Label regime: PRIMARY = `line/code`** (npz `y` ∩ is_code = the
published `tokens_code_auc`); the honest `tok/code-X` (difflib tight-span ∩
is_code) regime is reported as a secondary check and tells the same story.

Script: `rescore_language.py` (seed 42, 1000× example-bootstrap). Raw numbers:
`results/<model>.json`, `results/_summary.json`.

---

## Task 1 — Format gate (PASSED, all 7)

Recomputed `tokens_code_auc` at each model's operating layer reproduces the
exp-16 / exp-06 historical anchor to ≤0.0003. No number below is presented on an
unreproduced gate.

| model | L | recomputed | anchor | Δ |
|---|---|---|---|---|
| Qwen2.5-Coder-32B | 25 | 0.7758 | 0.776 | −0.0002 ✓ |
| Qwen2.5-Coder-7B  | 16 | 0.8132 | 0.813 | +0.0002 ✓ |
| gemma-3-1b-it | 25 | 0.7441 | 0.744 | +0.0001 ✓ |
| gemma-3-4b-it | 7  | 0.7748 | 0.775 | −0.0002 ✓ |
| gemma-3-12b-it | 15 | 0.7627 | 0.763 | −0.0003 ✓ |
| gemma-3-27b-it | 19 | 0.7587 | 0.759 | −0.0003 ✓ |
| gemma-3-12b-pt | 13 | 0.7818 | 0.782 | −0.0002 ✓ |

---

## Task 2 — Within-language AUC (primary, line/code)

Positives **and** negatives restricted to one language group (`c` = c+cpp).
`[lo,hi]` = 95% bootstrap CI over examples. `per-ex` = mean of within-example
AUCs (vuln examples with both classes); ≈ pooled ⇒ **pooling does not inflate
the within-language number**.

| model | within-Python pooled [CI] (per-ex) | within-C/C++ pooled [CI] (per-ex) |
|---|---|---|
| Qwen-32B | 0.839 [0.791,0.880] (0.833) | 0.596 [0.545,0.650] (0.584) |
| Qwen-7B  | 0.862 [0.825,0.893] (0.866) | 0.655 [0.610,0.697] (0.578) |
| g3-1b-it | 0.799 [0.759,0.837] (0.829) | 0.563 [0.525,0.602] (0.535) |
| g3-4b-it | 0.819 [0.775,0.862] (0.821) | 0.608 [0.557,0.658] (0.573) |
| g3-12b-it| 0.797 [0.756,0.839] (0.805) | 0.600 [0.562,0.640] (0.551) |
| g3-27b-it| 0.795 [0.759,0.823] (0.793) | 0.602 [0.542,0.659] (0.563) |
| g3-12b-pt| 0.804 [0.769,0.833] (0.819) | 0.629 [0.580,0.674] (0.565) |

- **Within Python ≈ 0.80–0.86; within C/C++ ≈ 0.56–0.66.** Reproduces exp-06
  sweep-6 (py ≈ 0.81, c ≈ 0.59) on the general probe.
- The **0.20 within-language gap is the same size as the published
  "injection-strong / memory-weak" gap** — and it is what the headline pooled
  number (≈0.78) is mostly built on (see Task 4).
- (Honest tok regime, mean across models: within-py 0.779, within-C 0.603 — same.)

---

## Task 3 — Family × language matrix (primary, line/code)

Cell = {vuln examples of (family, lang)} vs {clean examples of same lang}, pooled
test-code-token AUC. `[n]` = # positive (vuln) TEST examples in the cell.
**`*` = <10 positive examples → UNTRUSTED.**

| model | inj·Python [n=83] | inj·C/C++ [n=5]* | mem·C/C++ [n=58] | mem·Python |
|---|---|---|---|---|
| Qwen-32B | 0.839 | 0.710 [0.47,0.81]* | 0.592 | — (none in SVEN) |
| Qwen-7B  | 0.862 | 0.829 [0.52,0.94]* | 0.647 | — |
| g3-1b-it | 0.799 | 0.763 [0.67,0.81]* | 0.557 | — |
| g3-4b-it | 0.819 | 0.715 [0.55,0.80]* | 0.603 | — |
| g3-12b-it| 0.797 | 0.778 [0.71,0.85]* | 0.592 | — |
| g3-27b-it| 0.795 | 0.767 [0.70,0.83]* | 0.602 | — |
| g3-12b-pt| 0.804 | 0.894 [0.83,0.95]* | 0.620 | — |

- **The two trusted cells are pure-language**: inj·Python (n=83, AUC 0.80–0.86)
  and mem·C/C++ (n=58, AUC 0.56–0.65). SVEN has **no** Python memory cell and
  only **5** C/C++-injection test examples, so the matrix is degenerate — the
  family axis and the language axis are nearly collinear in this dataset.
- **inj·C/C++ point estimates (0.71–0.89) run above mem·C/C++ (0.56–0.65)**, and
  for 4/7 models the inj·C bootstrap lower bound exceeds the mem·C estimate —
  *suggesting* a residual family effect inside C. **But n=5 positive examples ⇒
  UNTRUSTED**: the CIs are wide ([0.47,0.81] … [0.52,0.94]) and a 5-example
  bootstrap understates true uncertainty. **The test split cannot settle whether
  C-injection beats C-memory.**

---

## Task 4 — Language-indicator null table (the corrected null) — KEY RESULT

AUC of the **bare language feature** (no probe: token in a C/C++ file = 1,
Python = 0; "pyPos" = reverse) on the *exact token set* of each published eval
design. This is the null the project's confounded evals never subtracted.

### (a) General headline design — all vuln-span pos vs all live-code neg
Identical across models (depends only on class×language composition, not the probe):

| | general probe AUC | lang-null (C=pos) | lang-null (**Py=pos**) |
|---|---|---|---|
| line/code | 0.744–0.813 | 0.323 | **0.677** |
| tok/code-X | mean 0.775 | 0.292 | **0.708** |

→ **A pure "is-this-a-Python-token" detector scores 0.677 (line) / 0.708 (tok)
on the exact headline eval.** The probe scores ≈0.776. In AUC-distance-from-0.5
terms, **language alone accounts for ≈0.177/0.276 ≈ 64% of the headline signal.**

### (b/c) Per-CWE designs — general-probe AUC vs lang-null (mean over 7 models)
Design (c) = exp-10/21 (CWE-pos vs clean-only); design (b) = exp-06 (CWE-pos vs
all-other-tokens). Both give the same null (≈0.665 for the 100%-C memory CWEs).

| CWE | fam | dom-lang | fracC | nEx | (c) probe | (c) **lang-null** | (b) probe | (b) lang-null |
|---|---|---|---|---|---|---|---|---|
| CWE-089 | inj | py | 0.00 | 44 | 0.929 | 0.165 | 0.933 | 0.160 |
| CWE-078 | inj | py | 0.00 | 19 | 0.849 | 0.165 | 0.847 | 0.163 |
| CWE-022 | inj | py | 0.13 | 15 | 0.793 | 0.237 | 0.788 | 0.237 |
| CWE-079 | inj | py | 0.06 | 10 | 0.667 | 0.198 | 0.662 | 0.199 |
| CWE-125 | mem | c | 1.00 | 19 | 0.544 | **0.665** | 0.540 | 0.667 |
| CWE-416 | mem | c | 1.00 | 14 | 0.515 | **0.665** | 0.511 | 0.667 |
| CWE-476 | mem | c | 1.00 | 16 | 0.545 | **0.665** | 0.541 | 0.667 |
| CWE-787 | mem | c | 1.00 | 5* | 0.428 | **0.665** | 0.425 | 0.667 |
| CWE-190 | mem | c | 1.00 | 4* | 0.605 | **0.665** | 0.600 | 0.667 |

- **Injection CWEs: probe ≫ null** (0.67–0.93 vs 0.16–0.24) → genuine,
  language-independent injection signal.
- **Memory CWEs: the lang-null (0.665) EXCEEDS the general probe's own per-CWE
  AUC (0.43–0.61) for every memory CWE.** Under the exp-10/21 design, a bare
  "this is C" feature out-detects the actual general probe on memory.
- **Against the published exp-10/21 *specialized*-probe memory numbers**
  (CWE-125 0.73, CWE-416 0.77, CWE-476 0.64, CWE-787 0.67): the 0.665 null
  brackets them. CWE-476 (0.64) and CWE-787 (0.67) are **at/below** the null;
  CWE-125 (+0.06) and CWE-416 (+0.10) clear it only modestly. **Most of the
  "memory recovery" headline sits inside the language-confound band.**

---

## Task 5 — Pooled vs per-example-averaged

Within-language: pooled ≈ per-example (Δ ≤ 0.02, see Task 2) → pooling across
examples does **not** inflate the within-language numbers. The inflation is
purely *cross-language*: the overall pooled headline (≈0.78) exceeds the harder
within-C number (≈0.60) precisely because the probe also ranks Python-vuln tokens
above C-clean tokens — i.e. the Task-4 language null (0.677) is the cross-language
component baked into the pooled headline.

---

## Task 6 — Within-pair before/after pairAcc (SECONDARY, max code-token logit)

Mean across 7 models:

| cell | pairAcc | n_pairs | |
|---|---|---|---|
| inj·Python | 0.642 | 83 | above chance |
| mem·C/C++ | 0.488 | 58 | **at chance** |
| inj·C/C++ | 0.514 | 5 | UNTRUSTED |

Within a vuln/safe pair the probe localizes injection (0.64) but **not memory
(0.49 = chance)** — consistent with claim #4 (lexical string-sink detector) and
ADR-0004 (memory fixes often non-localizable). Secondary metric; not a headline.

---

## So what

- **The published injection-vs-memory split is, in this dataset, almost entirely a
  Python-vs-C split.** Within language the gap reappears at the same magnitude
  (py ≈ 0.80 vs C ≈ 0.60), the two *trusted* family×language cells are each
  single-language, and SVEN has no Python-memory cell to break the collinearity.
- **The general headline `tokens_code_auc` (≈0.78) is ~64% recoverable from a bare
  language indicator (0.677).** The honest residual signal beyond language is
  ≈0.10 AUC — real (injection CWEs clear their null by 0.4–0.7) but far smaller
  than the headline implies.
- **Claim #3 ("memory signal exists; it's capacity-allocation") is the weakest.**
  Under its own exp-10/21 eval design the language null (0.665) exceeds the general
  probe on every memory CWE and brackets the published *specialized* memory numbers
  (0.64–0.77); CWE-476/787 carry **no** language-independent evidence, CWE-125/416
  only a modest +0.06/+0.10. The "recovery" is mostly the eval rewarding a C-detector.
- **Does the family split survive within C?** *Suggestively yes but UNTRUSTED*:
  inj·C point estimates (0.71–0.89) top mem·C (0.56–0.65), but on only **5**
  C/C++-injection test examples. **The held-out split cannot answer this.** A
  language/function-matched contrast or a larger C-injection sample (PrimeVul has
  many; exp-22) is required before claim #2 can be said to survive deconfounding.
- **Published numbers inside the language-null band** (carry no language-independent
  vuln evidence): all memory CWEs under the general probe; the specialized
  CWE-476 (0.64) and CWE-787 (0.67). Injection CWEs are clear of their null.

### Which interpretation hint fired
Hint 1 (C-inj ≈ C-mem ≈ 0.5–0.6 → split is language) is the **trusted** read for
mem·C (0.60) and is what the language-null table confirms for the memory side.
Hint 2 (C-inj ≫ C-mem → family real) is *suggested* by the n=5 inj·C cell but is
**UNTRUSTED**. Net: **claim #2 must be reworded** (the split is confounded with
language; a residual family effect inside C is plausible but unproven), and
**claim #3's memory-recovery foundation weakens further** — most of it is inside
the corrected language null.

## Caveats / provenance
- exp-16 probes were trained under the old line/all regime; exp-19 showed
  token≈line (Δ≤0.01 AUC) so these are representative of the canonical probe.
- `c` group merges c+cpp. Language-null AUC is probe-independent (a property of
  the eval design's class×language composition) — it is exact, not an estimate.
- Small-n cells (inj·C n=5; CWE-787 n=5, CWE-190 n=4) flagged UNTRUSTED throughout.
- No file outside this dir was modified; no commits; morning review (codex+Opus)
  pending per CLAUDE.md review gate before any project-log edit.

## Post-review erratum (review gate, 2026-06-10 — codex pass-with-fixes)

1. **Per-CWE language-null token sets were approximate, not exact.** The script used
   CWE-positive *span* tokens + clean-row tokens; the exp-10/21 evals also include
   the CWE examples' own negative live-code tokens. Reviewer-recomputed exact nulls
   (Qwen-32B): CWE-125 **0.631**, CWE-416 **0.649**, CWE-476 **0.642**, CWE-787
   **0.654** (vs the ~0.665 reported). Consequences: (a) the statement "general
   probe sits below the memory language-null" **stands** (probe 0.44–0.61 < all
   corrected nulls); (b) the statement "published specialized numbers are bracketed
   by the null" **overclaims** — specialized CWE-125 (0.732) and CWE-416 (0.766)
   clear their corrected nulls by ~+0.10/+0.12 (consistent with exp-25's
   matched-patch survivors); CWE-476 (0.640) ≈ its null (0.642).
2. **Bootstrap CIs resampled examples, not pairs/groups** — treat all CIs in this
   report as descriptive; the C-injection n=5 cell remains untrusted either way.
3. Format-gate deltas: two gemma models are ≤0.00034 (not ≤0.0003); Qwen-7B anchor
   is exp-16.
4. The "~64% of headline margin" figure is *recoverable-by-language-indicator*, not
   a causal attribution. Verdict unchanged in direction; specialized-memory cells
   are *stronger* than the main text states.
