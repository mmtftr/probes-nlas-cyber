[ai-generated]

# exp-27 RESULTS — surface baselines under matched-patch negatives

**TL;DR — the brief's "in between" outcome, leaning probe.** Under exp-25's
airtight **matched-patch** regime (vulnerable function vs its OWN fixed
version), the **window/n-gram surface family collapses**: char-n-gram,
combined, and both conly-trained refits are indistinguishable from chance in
**all 6** trusted memory cells (2 token axes × CWE-125/416/476; every 95% CI
includes 0.5). But the contrast is **not lexically empty**: a **token-unigram
LR retains a small, CI-solid lexical signal on CWE-125 (0.591 [0.561,0.621]
qwen / 0.584 [0.543,0.617] gemma)**, and keyword/unigram variants sit
marginally above chance on CWE-476. The exp-25 specialized probe exceeds the
best surface point estimate in **all four** trusted 125/416 axis-cells
(125: 0.633/0.657 vs unigram 0.591/0.584; 416: 0.610/0.603 vs best surface
0.542/0.534) and clears chance where no surface variant does (CWE-416) — but
**no probe-vs-surface contrast is CI-separated at n=19/14**, and the paired
probeG−surface Δ-bootstraps do not exclude 0 on the load-bearing cells. Honest
summary: **the CWE-125 matched-patch residue is at least partly lexical
(token-identity bar ≈ 0.59 covers roughly half to two-thirds of the probe's
margin over chance); the probe's excess above that bar (+0.04–0.07) is
directionally consistent across both models but not demonstrated at 95%.**
The strong "demonstrably non-lexical" upgrade is NOT earned; the "residue is
lexical after all, claim dies" outcome is also excluded (surface never
approaches the probe on 125/416). Injection positive controls stay **lexical
even under matched-patch** (char ≥ probe: CWE-089 0.975 vs 0.933/0.886).

Metric throughout = **`tokens_code_auc`** (project default). CIs = 1000×
bootstrap over examples, exp-25 `diag_ci` recipe (matched-patch negatives =
paired safe halves of the resampled positives; seeds `zlib.crc32`-stable).
Trusted = n_test_pos ≥ 10. Surface side = exp-24's feature builders,
**allclean-trained per exp-24 design-2 and held fixed across eval regimes** —
mirroring what exp-25 did with the probes. Probe columns are **cited from
exp-25** (`deconfound_{qwen32b,gemma1b}.json`), not re-run.

## Provenance gates (all hard-passed, every run)

1. **Design-2 bit-repro (qwen axis):** re-running exp-24 design-2 with the
   identical rng stream reproduces all 9 CWE × 7 column all-clean cells with
   **max |Δ| = 0.00e+00** → surface models, harness, and substrate are
   byte-identical to exp-24. (Version parity: sklearn 1.8.0 / scipy 1.17.1 /
   numpy 2.4.6 in both repos.)
2. **Token-count match vs exp-25:** all 36 (CWE × regime) cells per axis have
   `n_pos_tok`/`n_neg_tok` **exactly equal** to exp-25's deconfound JSONs
   (qwen: pos 405/321/313, neg 38910/26211/12699, mp 10413/4126/6186; gemma:
   pos 446/389/362, neg 44581/29847/14734, mp 11762/4840/7245).
3. **lang_null identity:** the language-indicator AUC on every (pos,neg) set
   equals exp-25's `lang_null` to 1e-9 (memory mp/conly ≡ 0.500 by
   construction; allclean 0.63–0.65).

Axis-anchoring note (review point): the qwen axis is bit-anchored end-to-end
(gate 1). The gemma axis has no design-2 reference; its alignment with
exp-25's gemma cells rests on gates 2/3 (36 exact counts + lang_null float
identity) plus the shared extraction history (the exp-16 gemma dump reproduced
the historical token-code-AUC ±0.000, and exp-25's own repro gate matched
exp-21 to full float precision) — counts-and-nulls-pinned, not bit-proven.

## Decisive tables — memory CWEs (probe = exp-25 specialized, cited)

### Qwen2.5-Coder-32B axis

| CWE | regime | probe (exp-25) | char-ngram | combined | **unigram** | conlytr-char | conlytr-comb | probeG |
|---|---|---|---|---|---|---|---|---|
| 125 (n=19) | allclean | **0.732 [.674,.799]** | 0.610 [.521,.693] | 0.632 [.517,.729] | 0.631 [.595,.666] | 0.559 [.427,.673] | 0.500 [.399,.582] | 0.606 [.534,.671] |
| 125 | conly | **0.675 [.613,.745]** | 0.577 [.485,.663] | 0.615 [.500,.713] | 0.599 [.562,.634] | 0.622 [.483,.733] | 0.623 [.513,.718] | 0.642 [.568,.708] |
| 125 | **mp** | **0.633 [.575,.701]** ✓ | 0.545 [.429,.654] | 0.576 [.443,.698] | **0.591 [.561,.621]** ✓ | 0.590 [.446,.714] | 0.584 [.448,.695] | 0.639 [.572,.703] ✓ |
| 416 (n=14) | allclean | **0.766 [.656,.879]** | 0.612 [.530,.685] | 0.590 [.522,.658] | 0.600 [.564,.639] | 0.434 [.345,.528] | 0.408 [.311,.517] | 0.489 [.409,.575] |
| 416 | conly | **0.714 [.595,.824]** | 0.576 [.494,.654] | 0.571 [.498,.655] | 0.568 [.528,.610] | 0.508 [.414,.606] | 0.520 [.413,.637] | 0.524 [.438,.620] |
| 416 | **mp** | **0.610 [.514,.709]** ✓ | 0.542 [.440,.621] | 0.502 [.437,.568] | 0.538 [.496,.576] | 0.465 [.376,.544] | 0.455 [.348,.563] | 0.518 [.451,.594] |
| 476 (n=16) | allclean | **0.640 [.581,.728]** | 0.541 [.440,.661] | 0.508 [.415,.624] | 0.600 [.570,.633] | 0.365 [.261,.483] | 0.352 [.267,.467] | 0.536 [.438,.663] |
| 476 | conly | **0.579 [.517,.676]** | 0.532 [.423,.649] | 0.513 [.415,.637] | 0.560 [.530,.588] | 0.448 [.322,.590] | 0.476 [.377,.600] | 0.573 [.474,.718] |
| 476 | **mp** | **0.544 [.483,.633]** ✗ | 0.506 [.423,.597] | 0.471 [.398,.547] | **0.557 [.528,.584]** ✓ | 0.425 [.318,.546] | 0.456 [.365,.575] | 0.601 [.502,.719] (✓ marginal) |

### gemma-3-1b axis

| CWE | regime | probe (exp-25) | char-ngram | combined | **unigram** | conlytr-char | conlytr-comb | probeG |
|---|---|---|---|---|---|---|---|---|
| 125 (n=19) | allclean | **0.734 [.675,.800]** | 0.679 [.569,.766] | 0.650 [.547,.732] | 0.624 [.576,.666] | 0.571 [.470,.654] | 0.533 [.416,.638] | 0.532 [.470,.599] |
| 125 | conly | **0.668 [.599,.741]** | 0.646 [.546,.739] | 0.627 [.531,.712] | 0.589 [.547,.630] | 0.646 [.543,.725] | 0.657 [.527,.763] | 0.571 [.512,.639] |
| 125 | **mp** | **0.657 [.593,.731]** ✓ | 0.596 [.479,.709] | 0.572 [.448,.683] | **0.584 [.543,.617]** ✓ | 0.607 [.486,.723] | 0.615 [.488,.723] | 0.586 [.531,.645] ✓ |
| 416 (n=14) | allclean | **0.769 [.710,.816]** | 0.611 [.533,.687] | 0.579 [.495,.658] | 0.598 [.563,.641] | 0.534 [.435,.629] | 0.428 [.355,.504] | 0.457 [.411,.501] |
| 416 | conly | **0.697 [.631,.756]** | 0.577 [.490,.657] | 0.558 [.468,.647] | 0.560 [.521,.605] | 0.607 [.505,.698] | 0.546 [.471,.619] | 0.500 [.450,.546] |
| 416 | **mp** | **0.603 [.525,.677]** ✓ | 0.506 [.429,.595] | 0.511 [.441,.583] | 0.523 [.478,.568] | 0.534 [.462,.610] | 0.498 [.432,.558] | 0.507 [.439,.571] |
| 476 (n=16) | allclean | **0.619 [.568,.661]** | 0.465 [.370,.571] | 0.522 [.426,.652] | 0.571 [.548,.598] | 0.376 [.292,.486] | 0.374 [.284,.487] | 0.520 [.440,.638] |
| 476 | conly | **0.560 [.506,.605]** | 0.466 [.379,.575] | 0.528 [.438,.651] | 0.531 [.507,.556] | 0.470 [.370,.592] | 0.502 [.383,.617] | 0.561 [.472,.685] |
| 476 | **mp** | **0.507 [.454,.552]** ✗ | 0.434 [.346,.532] | 0.492 [.416,.601] | 0.518 [.494,.539] | 0.446 [.359,.560] | 0.476 [.376,.575] | 0.535 [.454,.647] |

Keyword-LR (memory × mp): qwen 0.516/0.498/**0.552 [.503,.591]** ✓,
gemma 0.516/0.498/**0.549 [.510,.586]** ✓ (125/416/476). Keyword-untrained
and lang-indicator ≤ chance everywhere on memory × mp.

**Reading (the brief's question, answered per cell family):**

- **The window/n-gram family is indistinguishable from chance** under
  matched-patch in all 6 trusted memory cells — char, combined, conlytr-char,
  conlytr-comb all have CI ∋ 0.5 (the per-cell best, gemma-125 conlytr-comb
  0.615 [0.488,0.723], included). exp-24's all-clean memory surface numbers
  were the language confound, and they drain away monotonically
  (e.g. qwen char 125: 0.610 → 0.577 → 0.545 across allclean → conly → mp).
- **But token-identity survives on CWE-125**: unigram 0.591/0.584 with CIs
  excluding 0.5 on both axes. The patch's lexical footprint (which tokens
  appear, not their window context) is detectable — the matched-patch lexical
  ceiling on 125 is ≈0.59, not 0.5. On CWE-476 unigram (qwen) and keyword-LR
  (both axes) sit marginally above chance (0.55-ish) where the *specialized
  probe shows nothing* — minor, but it means 476's mp contrast also has a
  thin lexical component.
- **The probe stays on top, without CI separation.** On 125 the specialized
  probe (0.633/0.657, CI>0.5 both axes) exceeds the unigram bar by
  +0.042/+0.073 (points); on 416 it is the ONLY signal above chance
  (0.610/0.603, marginal CIs; every surface variant ∋0.5) with +0.068/+0.069
  over the best surface point. All probe-vs-surface CIs overlap at n=19/14.
- **Paired Δ (probeG − surface, same resamples; the locally computable
  contrast):** on 125-mp, probeG−unigram = [−0.024,+0.117] (Δ>0 in 92% of
  boots, qwen) and [−0.052,+0.065] (51%, gemma) — not separated; note gemma's
  probeG (0.586) is itself well below gemma's specialized probe (0.657), so
  the gemma paired test is weak evidence about the specialized probe. The
  specialized probe's per-token scores live on cluster-side activations, so
  its paired Δ is not computable locally — flagged as the natural follow-up.
- Margin accounting (point estimates): of the specialized probe's
  margin-over-chance on 125-mp (0.133/0.157), the token-identity bar covers
  0.091/0.084 — **roughly half to two-thirds of the matched-patch residue is
  reproducible lexically; the remainder (+0.04–0.07) is the probe's
  unexplained excess, consistent in direction across both models.**

## Injection positive controls — matchedpatch

| CWE | qwen: probe vs char | gemma: probe vs char |
|---|---|---|
| 089 (n=44) | 0.933 [.914,.945] vs **0.975 [.957,.987]** | 0.886 [.863,.904] vs **0.975 [.956,.987]** |
| 078 (n=19) | 0.833 [.722,.939] vs **0.872 [.802,.960]** | 0.771 [.664,.868] vs **0.876 [.795,.963]** |
| 022 (n=15) | **0.771 [.685,.828]** vs 0.753 [.605,.875] | 0.718 [.628,.787] vs **0.765 [.601,.889]** |
| 079 (n=10) | 0.532 vs 0.578 (both ≈ chance) | 0.611 vs 0.578 (CIs span .5) |

Surface **matches or beats** the specialized probe on 089/078 under the
airtight control. Injection detection needs no internal representation even at
matched-patch strictness — extending exp-24's resolved half to the airtight
regime; the probes' injection numbers should never be cited as
representational evidence.

## Untrusted cells (n<10) — matchedpatch, listed for completeness only

- CWE-190 (n=4): char 0.805/0.829 **>** probe 0.724/0.713 (qwen/gemma)
- CWE-787 (n=5): char 0.672/0.791 **>** probe 0.509/0.537

## Caveats

1. **Supervision asymmetry favors surface** (exp-24 caveat, unchanged):
   baselines train purely on the scored per-token quantity with ±48-char
   windows over the annotated span — an upper lexical bar. That the *window*
   family still collapses under mp is the informative direction; the unigram
   survival is correspondingly credible lexical signal.
2. **No probe-vs-surface contrast is CI-separated** (n=19/14). The replicated
   direction (probe > best-surface point in 4/4 trusted 125/416 axis-cells,
   probe CI>0.5 on 416 where all surface ∋0.5) is the result; "demonstrably
   non-lexical" is NOT claimed. The decisive missing test is a paired
   specialized-probe−unigram Δ-bootstrap (needs cluster-side activations).
3. **Two axes ≠ two independent replications**: both score the same SVEN test
   examples (different tokenizers/probe models). Cross-axis agreement rules
   out tokenizer artifacts, not example-sampling noise — that's what the
   per-axis example-bootstrap CIs measure.
4. **Multiple comparisons**: unigram-125-mp clears 0.5 on both axes with
   lower bounds well clear (0.561/0.543) — not a marginal pick-the-winner
   cell. The marginal 476 keyword/unigram cells (lower bounds 0.503–0.528)
   could individually be MC flukes; they are reported, not leaned on.
5. **Gemma-axis anchoring** is counts-and-nulls-pinned, not bit-proven (see
   gates note). The qwen axis carries the bit-anchored result.
6. **qwen probeG-476-mp 0.601 [0.502,0.719]** marginally clears chance where
   the specialized probe doesn't, and its paired Δ beats the window family
   (e.g. +[.019,.268] vs combined) but not unigram — single-axis curiosity,
   not a claim.

## Bottom line for the project ledger

exp-24's missing control is closed with a split verdict: **the matched-patch
memory residue is partly lexical** — a token-identity baseline retains
0.591/0.584 (CI>0.5, both axes) on CWE-125, covering ~½–⅔ of the probe's
margin over chance — **and partly unexplained**: the specialized probe tops
every surface variant's point estimate on all four trusted 125/416 axis-cells
(+0.04–0.07), is the only signal above chance on CWE-416, and the entire
window/n-gram surface family sits at chance everywhere on memory × mp. At
n=19/14 none of the probe-vs-surface margins individually clear 95%
separation, so the strong "demonstrably non-lexical" claim is not earned;
equally, the "residue is lexical after all" rewrite is excluded. Injection
remains fully lexical under every regime including matched-patch. Blog
impact: SUR1 (surface ≥ probe on the general/injection story) unchanged and
strengthened to matched-patch; SUR2 gets the nuanced upgrade ("window-family
surface collapses under the airtight control; a token-identity bar of ~0.59
survives on 125; the probe's +0.04–0.07 excess is consistent across two
models but needs more data / the paired specialized-probe test"); FUT1 gains
the concrete follow-up (paired Δ on cluster acts; more 125-like data).

## Artifacts

- `results/exp27_qwen32b_axis.json`, `results/exp27_gemma1b_axis.json` — per
  CWE × regime × baseline cells (auc, lang_null, 1000-boot CIs for every
  surface variant + probeG, paired probeG−surface Δ CIs for memory ×
  {conly,mp}, count-gate records), exp-25 probe columns cited inline.
- `run_exp27.py` (stages A/B/C), `make_results_tables.py` (tables above),
  `run.log` (final v3) / `run.v1.log`, `run.v2.log` (superseded: v1 lacked
  conly-trained CIs; v2 lacked unigram/keyword CIs + paired Δs — point
  estimates identical across all three, bit-anchored by gate 1).
- Local-only inputs (gitignored): `data/dataset.jsonl`,
  `data/sven_split_meta.json`, the two exp-16 L25 dump npz — copied from the
  archived repo; split meta byte-matches exp-25's archived copy.

## Review gate

codex (`cj exp27-review`) PASS-WITH-FIXES + Opus subagent PASS-WITH-FIXES on
the v2 draft; all five reconciled fixes applied (CIs for every surface
variant, paired Δ-bootstrap added, "collapses to chance" wording corrected,
gemma anchoring caveat, crc32-stable CI seeds). The unigram-CI fix
**materially corrected the verdict** (token-identity survives mp on 125 —
caught precisely by the reviewers' CI-less-variant objection); the final
verdict above is written against the v3 artifacts. Verified post-rewrite by a
second pass of both reviewers (see STATUS.md).
