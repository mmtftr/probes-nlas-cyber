// [ai-generated]
// Claims-with-evidence edition of the blog draft: every claim is followed by the
// plot that supports it + the data file it was computed from, so each can be
// validated independently. Point codes match docs/blog/outline.typ.
// Figures: make_figs.py (narrative fig_* + claim_*). Compile: typst compile --root <repo> ...

#import "/docs/templates/report.typ": report, callout, finding

#show: report.with(
  title: "Vulnerability probes vs. the lexical ceiling — claims & evidence",
  subtitle: "Every claim with its supporting plot and data source. Codes match the outline.",
  author: "mmtf + agent",
  date: "2026-06-11",
)

#let _mono = ("Menlo", "DejaVu Sans Mono")
#let code(c) = box(
  fill: luma(240), inset: (x: 3pt), outset: (y: 2pt), radius: 2pt,
  text(font: _mono, size: 8pt, weight: "bold", fill: rgb("#2f4b7c"))[#c],
)
#let pt(c, body) = block(above: 0.6em, below: 0.4em,
  grid(columns: (3.6em, 1fr), column-gutter: 7pt, code(c), body))
#let src(body) = block(above: 0.1em, below: 0.9em,
  text(size: 7.5pt, fill: luma(115))[#h(4.3em) data: #body])

#callout(title: "How to read this")[
  Same point codes as the outline. Under each claim: the plot computed from the
  experiment's result files (path given), so the claim can be checked against the
  data directly. §4 explains the language-baseline method in full.
]

= The headline, and probe-side ablations

#pt("RES1")[*Probing yields OK AUC scores* (0.75–0.82), stable across sizes and families.]
#figure(image("figs/fig_a_headline.png"))
#src[`23-…/results/_summary.json` (gate = exp-16 repro ±0.0003) + exp-18 RESULTS table.]

#pt("ABL3")[*Capacity is not the lever:* K∈{1,2,4,8} jointly-trained directions buy ≈nothing overall and never close the memory gap.]
#figure(image("figs/claim_capacity.png"))
#src[`09-…/results/summary_google_gemma-3-{1,4,12,27}b-it.json`, rows agg=max, by_cwe mean over 125/416/476.]

#pt("ABL4")[*The model's own yes/no is weak (0.49–0.62) and is the only read that scales with size;* the probe sits flat above it. (Different metrics: probe = token AUC, verbalized = example AUC — same task.)]
#figure(image("figs/claim_verbalized.png"))
#src[`17-…/results/*/metrics_verbalized_logits.json` (test split) vs exp-23 gate.]

#pt("DAT2")[*Additive (missing-check) vulnerabilities have no token labels by construction* — token AUC is undefined on them; the example-level fallback (rank the vulnerable function above its own fix by max token score) is at chance: 0.43, n=49 pairs, every model, both training regimes. Dropping the additive third is performance-neutral (0.756 vs 0.755).]
#figure(image("figs/claim_additive.png"))
#src[exp-19 RESULTS.md mean row (base→sub 0.756 / sub→sub 0.755 / pairAcc-sub 0.760 n=97 / pairAcc-add 0.429 n=49); metric defined in `train_grid.py` pair_stats (max over live-code tokens, strict >).]

No dedicated plots (by design): #code("RES2") cites MoC's published numbers; #code("RES5") (pt≈it) is visible in FIG-A (gemma 12B vs 12B-pt bars).

= The dataset, and the probe as a classifier

#pt("DAT5")[*The pooled probe is a string-sink detector* — fires on SQL/command/path string literals; misses memory bugs; false-alarms on patched code.]
#figure(image("figs/fig_e_tokenheat.png"))
#src[exp-16 `logits_layer25.npz` (qwen-32B) joined to `data/dataset.jsonl` eid 105 (SVEN-after = patched); example picked from exp-20 `fp_sample.json` safe-alarms.]

#pt("DAT8")[*A bare Python-or-C indicator scores 0.677 on the exact headline token set* (probe 0.776); within-language the probe is Python 0.84 / C 0.60. Method in §4.]
#figure(image("figs/claim_langmethod.png"))
#src[`23-…/results/Qwen2.5-Coder-32B-Instruct.json` + `_summary.json`; null = 1−`general_lang_null_cPos_line`.]

#figure(image("figs/fig_c_withinlang.png", width: 86%))
#src[#code("DAT9") across all 7 models: `23-…/results/_summary.json` `within_py_line` / `within_c_line`.]

#pt("DAT6")[*A char-n-gram classifier beats the probe* on identical splits: 0.803 vs 0.776.]
#figure(image("figs/fig_b_ladder.png", width: 86%))
#src[`24-…/results/design1_general.json` (probe/unigram/char-ngram/keyword with bootstrap CIs) + exp-23 language null.]

#pt("DAT7")[*Surface reproduces 3 of 4 family-transfer blocks* we had read as internal structure.]
#figure(image("figs/fig_f_blocks.png", width: 78%))
#src[probe blocks: exp-21 RESULTS; surface blocks: computed (off-diag means) from `24-…/results/design4_transfer_matrix.json`.]

= Residues — pending the surface comparison

#pt("SUR1")[*CWE-125 survives matched-patch* (0.633/0.657, CIs > 0.5, both models); 416 weak; 476 collapses.]
#figure(image("figs/fig_d_matchedpatch.png", width: 86%))
#src[exp-25 RESULTS tables (allclean / conly / matchedpatch + bootstrap CIs); CV stability: `cv_aggregate.json` (15 folds).]

#pt("SUR2")[*Open caveat:* the char-n-gram baseline was never run under matched-patch negatives — so this shows a non-language, non-template signal but NOT yet a non-lexical one. exp-27 (in progress) closes that gap.]

#pt("SUR3")[*No memory-family direction within one language:* mem→mem off-diag 0.536 ≈ mem→other 0.537 on PrimeVul, while per-CWE diagonals reach 0.61–0.88. *Caveat: preliminary — one model, no surface baselines, single-split cross cells; exp-28 (in progress) strengthens it.*]
#figure(image("figs/claim_family26.png"))
#src[`26-…/results/pv_within.json` `family_blocks(_ci)` + `diagonal` (bootstrap CIs).]

#pt("SUR4")[*The probe direction is epiphenomenal:* ±4σ steering leaves the model's verbalized P(yes) flat on every subset, indistinguishable from random directions.]
#figure(image("figs/claim_steering.png"))
#src[`13-…/results/steer_v2/steer_13_Qwen_Qwen2.5-Coder-32B-Instruct.json` (probe + injection + 2 random directions, degraded-flag false throughout).]

= Supplementary results

#pt("SUP6")[*Cross-dataset transfer is asymmetric:* SVEN→PrimeVul ≈ chance, PrimeVul→SVEN ≈ in-domain.]
#figure(image("/plans/cross-model-probe-generalization/22-primevul-paired/results/fig_cross_transfer.png", width: 72%))
#src[exp-22 `results/fig_cross_transfer.png` (single split, no CIs — flagged).]

= The language baseline — how it works and why it's the right control

*The eval being controlled.* The headline number is one pooled ranking: every
live-code token in the test set gets the probe's score; positives are the
annotated vuln-span tokens, negatives all other live-code tokens. AUC is the
probability a random positive outranks a random negative — it is computed over
*pairs* of tokens.

*The confound.* Positives are much denser in Python (11.0% of Python test tokens
vs 2.7% of C/C++ tokens — panel (a)): SVEN's injection CWEs are ~92% Python,
memory CWEs 100% C/C++, and injection contributes most annotated tokens. So in
the pooled pair set, many (positive, negative) pairs are Python-vs-C — and a
scorer that merely says "Python" wins those pairs with zero vulnerability
information.

*Null #1 — the language indicator (panel b).* Take the *identical* token set and
labels, and replace the probe's score with a constant per language (1 if the
token's example is Python, 0 if C; equal scores split credit, standard AUC tie
handling). That scorer gets *0.677*. This is the exact AUC available from
language identity alone on this eval — so the probe's 0.776 must be judged
against 0.677, not against 0.5. The "~64%" headline is the margin ratio
(0.677−0.5)/(0.776−0.5); AUC margins are not strictly additive, so we treat
this as an attribution heuristic, and rely on the stratified rescore for the
airtight statement:

*Null #2 — the within-language rescore (panel c).* AUC is a pairwise statistic
and the confound lives entirely in *cross-language pairs*, so compute AUC twice,
restricted to same-language pairs only: within-Python 0.839 [0.791, 0.880],
within-C/C++ 0.596 [0.545, 0.650]. No heuristic here — language categorically
cannot contribute when every compared pair shares a language. The pooled 0.776
sitting between (and its margin mostly above) the within-C value is the direct
evidence the headline rode the confound.

*The same logic, per CWE.* Specialized-probe evals (CWE-X positives vs the
all-clean negative pool) inherit the confound: memory-CWE positives are 100% C
while the clean pool is ~53% Python, so their language null is 0.63–0.65 — that,
not 0.5, is "chance" for those cells. This is what re-judged exp-10's memory
numbers (0.64–0.77), and why the final attribution uses the *matched-patch*
control — score the vulnerable function only against its own fixed version, so
language, file, author and style are all held fixed (§3, FIG-D).

*Integrity gate.* Everything above is a rescore of exp-16's *saved* per-token
logits — no re-extraction, no re-training. Before stratifying, the pooled
headline was recomputed from the saved logits and reproduced the historical
number on all 7 models (max Δ 0.00034), so the stratified numbers describe
exactly the probes that produced the headline.

#finding(label: "Reading guide")[Every plot in this document regenerates from
the repo: `make_figs.py` + `make_claim_figs.py` in `docs/blog/` (FIG-E needs the
exp-16 npz via `EXP16_NPZ`/`DATASET_JSONL`). If a claim doesn't match your
understanding, the data path under its plot is the place to look.]
