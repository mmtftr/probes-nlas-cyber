[ai-generated]

# exp-24 RESULTS — token-level surface baselines vs the Qwen-32B probe

**TL;DR.** Two halves. **Resolved:** the *general* probe (0.776) does not clear
the per-token surface ceiling (char-ngram 0.803), and every *injection* result —
including the specialized injection probe — is matched or beaten by surface
features; surface even reproduces 3 of the 4 family-transfer blocks. So the
general headline and the injection story need no internal representation.
**Unresolved:** the exp-21 *specialized memory* probes still clear surface
(CWE-125 0.732>0.632, CWE-416 0.766>0.649) and show cross-CWE memory transfer
surface lacks (mem→mem 0.618 vs 0.499). This rides on a mixed-language eval pool
so it is *consistent with* the C-vs-Python confound, but the deconfounding
control here used the **general** probe, not the specialized one — so exp-21's
"≥2 representational directions" is **undercut, not refuted**. Settling it needs
a within-language *specialized* transfer matrix (exp-25).

**Headline metric = pooled `tokens_code_auc`** (honest token-level ROC-AUC over
tree-sitter live-code tokens), the project default. Per-example-mean AUC is
**secondary** and labelled as such. All baselines are trained on the **same
seed-42 group split** and scored through the **same `pool_eval`** harness as the
probe; the probe column is the exp-16 L25 per-token sigmoid (`prob`) on a
**byte-identical token axis** (the dump *is* the probe's extraction).

**Substrate sanity:** L25 `test_tokens_code_auc` reproduced at 0.7758 (≈ headline
0.776) through this harness; neg/pos pools bit-exact vs exp-21
(`n_neg_test_tokens`=38910; per-CWE pos-token counts match).

**Supervision caveat (deliberate).** Baselines get **purely per-token, static**
supervision (positive = `y==1 ∩ is_code` train tokens; negative = other live-code
train tokens; hard labels) — they optimize *exactly* the per-token quantity this
metric scores. The probe (exp-16 span-max trainer) trains an **annealed blend**,
`L = (1−ω)·per-token-BCE + ω·span-max-BCE` with ω linear 0→1, so its per-token
term is soft-labelled + entity-up-weighted and **annealed away** toward an
example-level span-max peak — it is *not* purely per-token, nor purely span-max.
Both columns are scored identically on the per-token metric (eval is
apples-to-apples); only the *training* objective differs. So these baselines are
an **upper lexical bar** — generous to the surface side on purpose: tuned
directly and only on the scored per-token quantity, while the probe's per-token
supervision is diluted toward a different objective. Any probe number that fails
to clear this bar is not evidence of internal representation.

Baselines: (a) token-unigram LR [one-hot token string], (b) char 3–5-gram
HashingVectorizer LR over a ±48-char window, (c) keyword/regex security lexicon
(untrained heuristic + LR), (d) language indicator (C/C++=1), (e) combined a+b+d
LR. Negatives subsampled for *training* only (cap 60k, ≤25×pos); **eval pools
always full**. `liblinear`, C=1.0, seed 42, 1000-boot CIs (500 for per-CWE).

---

## Design 1 — General (all vuln-span pos vs all live-code neg, test split)

`n_ex`=292 (146 vuln / 146 clean), `n_tok`=76489, `n_pos_tok`=4160.

| model | tokens_code_auc | 95% CI | example-mean (sec.) |
|---|---|---|---|
| **probe (L25)** | **0.776** | [.725, .824] | 0.531 |
| char-ngram LR (b) | **0.803** | [.748, .849] | 0.661 |
| combined LR (e) | 0.801 | [.747, .849] | 0.666 |
| token-unigram LR (a) | 0.694 | [.659, .730] | 0.528 |
| keyword LR (c) | 0.625 | [.587, .663] | 0.513 |
| keyword untrained (c) | 0.574 | [.535, .618] | 0.478 |
| language indicator (d) | 0.323 | [.280, .379] | 0.500 |

**Δ = probe − best-surface = −0.027** (CIs overlap heavily).

**So what:** the probe does **not** clear the surface ceiling on the headline
design. A hashed char-n-gram window matches/edges it. The general probe carries
no token-level signal beyond surface lexicon.

---

## Design 2 — exp-10/21 per-CWE all-clean diagonal

Train CWE-X vuln-span vs clean-train; eval CWE-X test vuln vs clean-test
(exp-21's eval pool, bit-identical). Probe column for comparison = **exp-21
specialized per-CWE probe** (`transfer_allclean.json` diagonal); `probeG` = the
exp-16 general probe through this harness. All numbers `tokens_code_auc`.

| CWE | fam | n_te | lang (d) | kwd-un | unigram | char | comb (e) | probeG | **exp21 probe** |
|---|---|---|---|---|---|---|---|---|---|
| 089 | inj | 44 | 0.194 | 0.747 | 0.861 | 0.993 | **0.994** | 0.937 | 0.983 |
| 078 | inj | 19 | 0.190 | 0.414 | 0.821 | 0.956 | **0.966** | 0.862 | 0.923 |
| 022 | inj | 15 | 0.254 | 0.519 | 0.749 | 0.893 | **0.908** | 0.798 | 0.943 |
| 079 | inj | 10 | 0.208 | 0.411 | 0.670 | 0.761 | 0.786 | 0.594 | 0.863 |
| 125 | mem | 19 | 0.631 | 0.495 | 0.631 | 0.610 | 0.632 | 0.606 | 0.732 |
| 416 | mem | 14 | 0.649 | 0.484 | 0.600 | 0.612 | 0.590 | 0.489 | 0.766 |
| 476 | mem | 16 | **0.642** | 0.430 | 0.600 | 0.541 | 0.508 | 0.536 | **0.640** |
| 787 | mem | 5* | 0.654 | 0.565 | 0.574 | 0.711 | 0.801 | 0.457 | 0.670 |
| 190 | mem | 4* | 0.658 | 0.496 | 0.679 | 0.773 | 0.785 | 0.618 | 0.767 |

`*` <10 test vuln examples — **untrusted**.

**So what:**
- **Injection:** the surface combined/char LR **≈ or beats** the specialized
  probe (089 0.994 vs 0.983; 078 0.966 vs 0.923). Injection detection is a
  lexical/string-sink signal — no internal representation required. Language
  indicator is *anti*-predictive (~0.19–0.25: injection is Python, the negative
  direction of a C=1 indicator).
- **Memory (mixed):** a **pure language indicator (zero vulnerability
  knowledge)** scores **0.63–0.66**, accounting for the *floor* of exp-21's
  "memory recovery" (0.64–0.77). **CWE-476: language alone (0.642) equals the
  specialized probe (0.640)** — fully confounded. **But CWE-125 (probe 0.732 vs
  best surface 0.632) and CWE-416 (0.766 vs 0.649) the specialized probe clears
  surface by ~0.10–0.13** — a residual the language axis does *not* explain.
  Caveat: this eval is on the all-clean **mixed-language** pool, so even that
  residual margin is not yet a clean within-language comparison (see designs 3,4).
  Net: the memory "recovery" is **partly** the C-vs-Python confound (fully so for
  476), but **not entirely** — claim #3 is **weakened, not refuted**, and the
  unconfounded part survives into design 4.

---

## Design 3 — within-language deconfound (decisive for the *general* probe)

Restrict every pool to a single language family, removing the C-vs-Python axis.
By construction the language indicator → 0.500 (no variance within a language).
**Caveat:** design-3 `probe` = the exp-16 **general** probe restricted to
language (we have its per-token scores); the exp-21 **specialized** per-CWE
probe's within-language score was not re-extracted, so 3b compares
general-probe + surface within-language (the surface ceiling is the load-bearing
column).

### 3a — general, within language

| pool | probe | unigram | char | combined (e) | n_ex (pos) |
|---|---|---|---|---|---|
| **Python** | 0.839 [.787,.880] | 0.737 | 0.891 | **0.895** [.839,.945] | 166 (83) |
| **C/C++** | 0.596 [.545,.652] | 0.552 | 0.589 | 0.602 [.539,.666] | 126 (63) |

- Python: surface (0.895) **beats** the probe (0.839); CIs overlap on the high side.
- C/C++: probe (0.596) **≈** surface (0.602); **both collapse toward chance**
  once language is removed. The general probe has essentially no within-C signal
  beyond surface.

### 3b — per-CWE diagonal, within the CWE's own language

| CWE | fam | lang | n_te | combined (e) | char | general-probe |
|---|---|---|---|---|---|---|
| 089 | inj | py | 44 | **0.981** | 0.981 | 0.903 |
| 078 | inj | py | 19 | **0.924** | 0.924 | 0.813 |
| 022 | inj | py | 11 | **0.867** | 0.864 | 0.758 |
| 079 | inj | py | 9* | 0.675 | 0.689 | 0.502 |
| 125 | mem | c/cpp | 19 | 0.602 | 0.628 | **0.642** |
| 416 | mem | c/cpp | 14 | 0.529 | 0.554 | 0.524 |
| 476 | mem | c/cpp | 16 | 0.487 | 0.465 | **0.573** |
| 787 | mem | c/cpp | 5* | 0.727 | 0.775 | 0.493 |
| 190 | mem | c/cpp | 4* | 0.790 | 0.774 | 0.659 |

`*` untrusted (<10).

**So what:**
- **Injection (within Python):** surface **dominates** the general probe at every
  trusted cell — the injection signal is lexical, not representational.
- **Memory (within C/C++):** with the language confound removed, the **general**
  probe and surface both sit at **0.49–0.64**; the general probe edges surface
  only on the diagonal (125 +0.04, 476 +0.086, 416 ≈0) — small margins barely
  above chance. **Caveat (important):** this is the **general** exp-16 probe, not
  the exp-21 **specialized** per-CWE memory probes whose "recovery" (0.73/0.77 on
  CWE-125/416) is the actual exp-21 claim. Design 3 shows the *general* probe has
  no within-C signal beyond surface; it does **not** measure the specialized
  probe within-language (would need L25 hidden states). So design 3 cannot, on
  its own, retire exp-21's specialized memory result — see design 4.

---

## Design 4 — 9×9 surface-only transfer matrix (the centerpiece)

Train combined-(e) surface LR on {CWE-X vuln ∪ all-clean}; eval CWE-Y vuln vs
all-clean, full 9×9 (same eval pool as exp-21). Question: does **surface alone**
reproduce exp-21's family-structured transfer (within-family >0.5, cross-family
<0.5)? If yes, exp-21's "≥2 representational directions" inference collapses to
surface/dataset structure.

**9×9 surface-(e) AUC** (rows = train CWE-X, cols = eval CWE-Y; `*` = untrusted
eval col, n_test_pos<10; INJ = 022/078/079/089, MEM = the rest):

```
trainX\Y  022  078  079  089 | 125  416  476  190* 787*
022 i    0.91 0.58 0.76 0.34 |0.46 0.37 0.41  0.22 0.66
078 i    0.78 0.96 0.60 0.56 |0.42 0.36 0.39  0.32 0.57
079 i    0.59 0.68 0.80 0.41 |0.39 0.35 0.36  0.32 0.55
089 i    0.59 0.77 0.68 0.99 |0.42 0.40 0.35  0.36 0.48
---------------------------- + ------------------------
125 m    0.51 0.56 0.59 0.33 |0.64 0.37 0.43  0.61 0.75
416 m    0.36 0.55 0.56 0.35 |0.43 0.61 0.50  0.57 0.73
476 m    0.43 0.57 0.65 0.31 |0.49 0.52 0.48  0.58 0.73
190 m*   0.45 0.61 0.59 0.40 |0.55 0.47 0.43  0.80 0.62
787 m*   0.33 0.53 0.56 0.33 |0.51 0.49 0.47  0.51 0.78
```

lang-indicator reference row (eval Y vs clean, constant across train): inj cols
~0.19–0.25, mem cols ~0.63–0.66.

**Family-block means (trusted cells only, token-weighted), surface vs exp-21
probe** (95% block-bootstrap CI):

| block | surface (e) | exp-21 probe | reproduced? |
|---|---|---|---|
| inj→inj (diag) | 0.646 [.605,.696] | 0.688 [.659,.714] | **yes** (CIs overlap) |
| inj→inj off-diag | 0.542 [.488,.608] | 0.600 [.563,.635] | yes (weakly >0.5) |
| inj→mem (cross) | 0.393 [.334,.462] | 0.411 [.375,.447] | **yes** (anti-transfer) |
| mem→inj (cross) | 0.425 [.362,.493] | 0.341 [.312,.377] | **yes** (anti-transfer) |
| mem→mem (diag) | **0.499** [.451,.558] | **0.618** [.591,.649] | **NO** (CIs disjoint) |
| mem→mem off-diag | 0.457 [.401,.519] | 0.570 [.536,.607] | no (CIs disjoint) |

**So what:** surface features reproduce **3 of the 4 family blocks** — the
within-injection positive transfer (0.65) and **both** cross-family
anti-transfers (<0.5). exp-21 read that sign pattern (within-family transfer,
cross-family anti-transfer) as evidence of ≥2 distinct representational
directions; pure char-n-gram + token + language features produce the same
pattern, so the **structure alone does not require any internal geometry.**

The **one** block surface does **not** reproduce is **mem→mem**: the exp-21
specialized memory probes transfer across memory CWEs (0.618 diag / 0.570
off-diag) while surface-(e) sits at chance (0.499 / 0.457; CIs disjoint).
Surface-(e)'s char-n-grams *anti-transfer* across memory CWEs (off-diagonal cells
~0.4–0.5), so the combined LR has no cross-CWE memory signal, whereas the
specialized probe does.

**Is this genuine memory geometry, or the language axis?** The eval pool is
mixed-language clean, so this block is also where the C-vs-Python confound is
maximal (lang-indicator alone ~0.64 on memory cols), and the probe block is
therefore *consistent with* a language detector. **But this experiment cannot
decide between the two.** The deconfounding control we have (design 3, within
C/C++) was run on the **general** exp-16 probe, **not** these exp-21 specialized
memory probes — an invalid substitution for retiring the specialized result.
Settling it requires a within-language *specialized* transfer matrix (train/eval
each memory probe with C-only negatives), which needs the L25 hidden states (not
in the logit-dump substrate) and is out of scope for this CPU-only run.

**Verdict (design 4):** exp-21's family-block sign pattern is **reproduced by
surface for injection and for both cross-family anti-transfers** — so that
pattern alone does *not* require internal geometry. The inference is therefore
**undercut, not refuted**: the strongest pro-probe evidence — the specialized
**memory** probes' within-family transfer (0.618 vs surface 0.499) — *survives*
this control and remains **unresolved** (genuine memory direction vs C-vs-Python
confound) pending a within-language specialized re-eval.

---

## Overall verdict

This is the field-standard control (Hewitt–Liang; arXiv:2509.03888): a probe is
evidence of internal representation only insofar as it exceeds trivial surface
features under the identical split/eval. The result splits cleanly into a
**resolved** half and an **unresolved** half.

**Resolved — surface explains the general probe and all injection results:**

- **General headline (design 1):** probe 0.776 < char-n-gram surface 0.803
  (Δ=−0.027; CIs overlap). The general probe carries **no** token-level signal
  beyond surface lexicon.
- **Injection (designs 2, 3, 4):** surface ≈/beats even the *specialized*
  injection probe (089 0.994 vs 0.983; within-Python 089 0.981 vs 0.903). A
  lexical string-sink signal, not representation.
- **Family-block sign pattern (design 4):** surface reproduces 3 of 4 blocks
  (within-injection transfer + both cross-family anti-transfers). The transfer
  *structure* alone does not require ≥2 internal directions.

**Unresolved — the specialized memory probe survives this control:**

- On the all-clean diagonal (design 2) the specialized memory probe **clears**
  the surface ceiling for CWE-125 (0.732 vs 0.632) and CWE-416 (0.766 vs 0.649);
  CWE-476 ties language (0.640 vs 0.642).
- In design 4 the specialized memory probes show within-family transfer
  (mem→mem 0.618) that surface does **not** (0.499; CIs disjoint).
- This advantage rides on a **mixed-language** eval pool, so it is *consistent
  with* a C-vs-Python confound (language alone 0.63–0.66) — but this experiment
  does **not** prove that. Design 3's within-C deconfound was run on the
  **general** probe (within-C probe≈surface≈0.5–0.64), **not** these specialized
  memory probes, so it cannot retire the specialized result.

**Bottom line:** the general probe and the injection story are fully explained by
surface features; exp-21's "≥2 representational directions" inference is
**undercut, not refuted**. The specialized memory probes' family transfer is the
one signal surface cannot reproduce and is the project's strongest remaining
candidate for genuine representational content — **resolving it (genuine memory
direction vs language confound) requires a within-language specialized transfer
matrix**, which needs the L25 hidden states and is the clear follow-up (exp-25).

## Caveats / limitations (for the skeptical reviewer)

- **Supervision asymmetry (favors surface):** baselines get per-token labels, the
  probe trains span-max — an upper lexical bar by construction. Compounding this,
  in designs 2/4 the surface LR trains on 719–4993 positive *tokens* vs the
  specialized probe's 26–134 positive *examples* — the surface side also has far
  more training signal. Both asymmetries make any surface≈probe result *favorable
  to surface*, which only strengthens the resolved-half conclusions.
- **Specialized memory probe never deconfounded (the key open gap):** designs 2
  and 4 compare against the exp-21 specialized per-CWE probe, but design 3's
  within-language control was run on the **general** exp-16 probe (the only
  per-token probe scores in the logit-dump substrate). The specialized memory
  probe's within-C performance — the number that would settle "memory direction
  vs C-vs-Python confound" — was not measured and needs L25 hidden states
  (exp-25 follow-up). Until then the specialized memory result stands unrefuted.
- **Bootstrap unit = example-row**, not pair/group (codex review pt a). Point
  estimates are unaffected; group-level CIs would only widen. Per-group repo
  composition (pt b) not enumerated; within-C is_cpp-vs-is_c null (pt d) not run
  (memory CWEs ~85% C → weak power). None are conclusion-changing. See STATUS
  `DESIGN-REVIEW-APPLIED`.
- **Char-n-gram window (±48 char) overlaps the annotated span**, so it can see
  the vulnerable substring directly — appropriate for a *ceiling*, not a fair
  "context-only" detector.
