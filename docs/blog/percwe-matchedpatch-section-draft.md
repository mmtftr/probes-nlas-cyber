[ai-generated]

<!--
DRAFT — separate file, for the user to rewrite. Drop-in replacement for the
blog's per-CWE section (currently "### Shared templates within a CWE",
index.qmd lines ~163-173).

WHAT CHANGED
- Per-CWE is now evaluated matched-patch (each vulnerable function vs its own
  fix) for ALL CWEs, both families. The all-clean pooled per-CWE numbers
  (specialist lifts 0.732/0.766/0.640, etc.) are REMOVED from the results; the
  pooled eval stays only as the confound it is. No "we found our mistake then
  did better" narrative; matched-patch is presented as the eval.

FIGURE
- New: fig-percwe-matchedpatch. Source PNGs:
  plans/cross-model-probe-generalization/32-percwe-matchedpatch/results/
    fig_percwe_matchedpatch_qwen.png   (headline, single model)
    fig_percwe_matchedpatch_both.png   (Qwen + Gemma)
  TODO for integration: regenerate light+dark variants via make_figs.py and
  copy into the post's figs/ (see docs/blog/README.md theming).
- This figure can REPLACE the current fig-matchedpatch (which only showed
  memory CWEs under the all-clean -> conly -> matched-patch ladder).

NUMBERS — reviewed (2 codex + 1 Opus), TRUSTWORTHY. Source table:
  plans/cross-model-probe-generalization/32-percwe-matchedpatch/results/percwe_matchedpatch.md
Hard wording constraints from review (do not loosen):
- No probe-vs-lexical contrast is CI-separated under matched-patch. Never write
  "the probe beats the baseline" on any single cell; it is a point-estimate edge.
- CWE-125: probe clears chance AND point-estimate tops both scanners, BUT token
  identity alone also clears chance and covers ~half to two-thirds of the
  probe's margin -> partly lexical.
- CWE-416: probe is the ONLY above-chance signal (scanners at chance), but
  marginal (lower CI ~0.51), not CI-separated.
- CWE-476: chance for everything.
- CWE-190 (n=4) and CWE-787 (n=5): too few held-out pairs, excluded from claims.
- "matched-patch" negatives = the function's own non-vulnerable tokens plus its
  patched counterpart's tokens (same file/function patched pool), NOT only the fix.

STILL ALL-CLEAN ELSEWHERE (flag for the user, not rewritten here): the
initial-results per-CWE asides (SQL 0.92 / use-after-free 0.52), the
injection-vs-memory family figure (fig-family), and the cross-family transfer
blocks (fig-blocks) are still computed against the pooled clean pool. If the
per-CWE eval should be matched-patch "all the time," those need the same pass.
-->

### Per-CWE, against each function's own fix

To read per-CWE signal without the template confound, the negative for each
vulnerable function is its own patched counterpart, not the general clean pool.
The dataset is paired, so this costs nothing and buys a lot: language, project,
coding style, and boilerplate are all held fixed, and the only thing that
separates a positive from its negatives is the change the security fix made. For
every memory CWE, all of which are C or C++, the language null is then 0.5 by
construction. The one class that stays mixed is path traversal (CWE-022), split
across Python and C, which keeps a residual language signal even here.

![Per-CWE token-level AUC (live code) under matched-patch: each vulnerable
function scored only against its own patched counterpart. Linear probe versus a
char-n-gram and a token-unigram text scanner on the identical tokens; whiskers
are 95% bootstrap intervals, the dashed line is chance. CWE-190 and CWE-787
(greyed) have too few held-out pairs to read.](figs/fig_percwe_matchedpatch.png){#fig-percwe-matchedpatch .themed-fig style="--dark-fig:url('figs/fig_percwe_matchedpatch-dark.png')"}

Injection survives the control but it is lexical. On SQL injection the
char-n-gram scanner scores 0.975 against the probe's 0.933; on command injection
0.872 against 0.833. A scanner with no access to activations matches or beats the
probe wherever the family scores at all. XSS runs the other way: against its own
patches it falls to near chance (0.53 on Qwen, 0.61 on the smaller model, only 10
test pairs), so the strong XSS score it posted against generic clean code was the
template, not the bug.

Memory is where a non-lexical signal would have to show, and it is thin.
Out-of-bounds read (CWE-125) is the best case: the probe clears chance (0.633 and
0.657 on the two models, intervals above 0.5) and its point estimate sits above
both text scanners. But bare token identity also clears chance here (0.591,
0.584) and accounts for roughly half to two thirds of the probe's margin over
chance, and at 19 test pairs no probe-versus-scanner gap is separated at 95%.
Use-after-free (CWE-416) is the only cell where the probe is the sole
above-chance signal, with both scanners at chance, though the probe's own margin
is small and not separated (0.610, 0.603; lower interval near 0.51). NULL
dereference (CWE-476) is at chance for everything. The two remaining memory
classes have four and five held-out pairs, too few to read, and are shown greyed.

Per-CWE, then, evaluated against each function's own fix: injection is lexical,
and the only memory residue a text baseline does not fully explain is a small,
not-CI-separated edge on out-of-bounds read and use-after-free, sitting around
0.6. Both model sizes give the same picture.
