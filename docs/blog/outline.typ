// [ai-generated]
// Blog-post outline — numbered editing skeleton for Slack review. v3:
// restructured to the bias-breakdown arc (user, 2026-06-11 evening):
// OK AUC → "what does this number mean?" → probe-side ablations → dataset-side
// classifier analysis → residues (caveated) → caveats & future work.
// Every point has a stable code (e.g. DAT6). Reply with code + instruction.
// Figures: make_figs.py — narrative fig_* + claim_* (FIG-E needs EXP16_NPZ/DATASET_JSONL).
// Compile: typst compile --root <repo> docs/blog/outline.typ

#import "/docs/templates/report.typ": report, callout, finding

#show: report.with(
  title: "Vulnerability probes vs. the lexical ceiling",
  subtitle: "Blog-post outline v3 — bias-breakdown arc; point codes are stable handles",
  author: "mmtf + agent",
  date: "2026-06-11",
)

#let _mono = ("Menlo", "DejaVu Sans Mono")
#let code(c) = box(
  fill: luma(240), inset: (x: 3pt), outset: (y: 2pt), radius: 2pt,
  text(font: _mono, size: 8pt, weight: "bold", fill: rgb("#2f4b7c"))[#c],
)
#let pt(c, body) = block(above: 0.5em, below: 0.5em,
  grid(columns: (3.6em, 1fr), column-gutter: 7pt, code(c), body))
#let figbox(path, cap) = figure(image(path), caption: cap)

#callout(title: "How to edit / v3 code mapping")[
  Reference codes over Slack ("cut DAT7", "ABL4: soften"). Mapping from v2:
  RES5→ABL1, RES7+MLP→ABL3, RES8→ABL4, RES9→DAT2, SNK→DAT5, SRF1→DAT6,
  SRF2→DAT7, LNG1/2→DAT8, LNG3→DAT9, SRF3→DAT10. Moved to supplementary:
  RES6 (layers, SUP4), RES10 (cross-dataset, SUP6), SUR4 (steering, SUP7).
]

= Title candidates

#pt("TIT1")[*We trained vulnerability probes on code models. A bag of character n-grams beat them.* (punchy; names the result)]
#pt("TIT2")[*When can a probe read what the model knows? Vulnerability probes and the lexical ceiling.* (sober; names the thesis)]
#pt("TIT3")[One-line thesis either way: a probe only proves it reads the model's belief once it beats the input text itself; for code vulnerability that bar is high, for hallucination it is low.]

= Goal & inspiration

#pt("GOL1")[*The goal:* understand probing better — and how to *honestly evaluate* a probe in a real-life scenario like vulnerability detection. The post is the evaluation playbook we ended up with, told through the case study.]
#pt("INS1")[*Hallucination probes work.* Token-level linear probes flag fabricated entities in real time; AUC 0.90 vs 0.71 for semantic entropy on Llama-3.3-70B (Obeso et al. 2025).]
#pt("INS2")[*The promise:* a monitor that costs a dot product per token — cheap oversight for AI control.]
#pt("INS3")[*Our bet:* the same recipe reads "this code is dangerous" off a code model's hidden states.]

= Setup

#pt("SET1")[*Data:* SVEN before/after pairs (9 CWEs). The fix diff marks which tokens are the vulnerable part → token-level labels.]
#pt("SET2")[*Metric:* token-level ROC-AUC (`tokens_code_auc`; the live-code restriction is itself a story beat — #code("DAT1")).]
#pt("SET3")[*Splits:* group-clean at pair level, 20% held out; a pair never straddles train/test.]
#pt("SET4")[*Probes:* linear token probes trained with a custom loss — supervision starts on the diff tokens and anneals to span-max. Layer sweep per model; validation picks the best-AUC layer.]
#pt("SET5")[*Models:* Gemma-3 1B–27B (it + 12B-pt) and Qwen2.5-Coder 7B/32B.]

= The headline result — and the question it raises

#pt("RES1")[*Probing yields OK AUC scores.* `tokens_code_auc` 0.75–0.82; performance is stable across model sizes and families (1B→32B, Gemma and Qwen).]

#figbox("figs/fig_a_headline.png")[#code("FIG-A") Headline `tokens_code_auc` per model (linear probe, bars; exp-18 MLP head, diamonds). Flat across scale.]

#pt("RES2")[*This replicates the literature:* MoC (Yu et al. 2025) trains linear probes on the same dataset and reports \~79–82% detection vs \~50% for prompting.]
#pt("RES3")[*The question the rest of the post answers:* how and why does the probe yield this AUC — what does the number actually mean? We break the biases down one by one: first the probing side, then the dataset.]

= Part 1 — probe-side ablations: nothing changes the picture

#pt("ABL1")[*Training regime: instruction tuning doesn't help.* Base 12B-pt ≈ instruct 12B-it (0.782 vs 0.763, exp-06) — the signal is already there from pretraining.]
#pt("ABL2")[*Model size: bigger doesn't help.* 1B→32B spans 0.744–0.813 with no trend (#code("FIG-A")) — whatever the probe reads is present at 1B.]
#pt("ABL3")[*Capacity does not help.* K∈{1,2,4,8} jointly-trained linear directions buy +0.016 overall and nothing on memory CWEs (exp-09); an MLP head buys +0.02–0.04 with the same failure structure (exp-04/12/18); family-balanced oversampling trades memory +0.06 for injection −0.03 (exp-11). More probe parameters change the number, never the picture.]

#figbox("figs/claim_capacity.png")[K-direction sweep (exp-09): overall flat, memory flat — capacity is not the missing ingredient.]

#pt("ABL4")[*Maybe verbalizations help? No — worse, and recovering them costs too much.* The model's own yes/no scores 0.49–0.62 example-AUC (exp-05/17). Prompt engineering recovers +0.21–0.33 on memory CWEs (exp-14) — but every variant is a full generation pass per example, against a probe that costs one dot product on states you already have. Cost-adjusted, the probe wins even where prompts catch up.]

#figbox("figs/claim_verbalized.png")[Probe vs verbalized per model (exp-16/17): verbalized is the only read that scales with size, and it never catches the probe.]

= Part 2 — the dataset, and the probe as a classifier

#pt("DAT1")[*First, the easiest bias: comments and trivial tokens are noise.* We parse with tree-sitter and restrict scoring to live-code tokens (`tokens_code_auc`). The honest restriction barely moves the number — and train-time masking of trivial negatives is a no-op (exp-07, ADR-0003) — so the signal isn't "comment detection".]
#pt("DAT2")[*Second bias: a third of SVEN has no token labels by construction.* Additive fixes only add a check — no vulnerable token to mark, token AUC undefined. The example-level fallback (rank the vulnerable function above its own fix by max token score) is at chance (0.43, n=49 pairs, every model). We restrict to the subtractive subset; it costs nothing (0.756 vs 0.755; exp-19, ADR-0004).]
#pt("DAT3")[*Split by CWE: performance is not uniform.* Injection CWEs score high; memory CWEs (125/416/476) are the worst slice (0.52–0.59 under the general probe, exp-06).]
#pt("DAT4")[*Per-CWE probes beat the general probe — and an ensemble of them performs best.* Specialized probes lift their own CWE (memory 0.64–0.77 vs the general's 0.52–0.59, exp-10); a post-hoc ensemble of the specialists is the best overall (≈0.80, beating the single linear everywhere and the MLP on Qwen; exp-09/10). *(Foreshadow: the memory half of this gain is re-judged at #code("DAT8").)*]
#pt("DAT5")[*Now, what IS this classifier? We agentically audit the TP/FP/FN/TNs.* The probe is mostly a string-pattern matcher: it fires on SQL/command/path string literals — catches injection sinks, misses memory bugs, false-alarms on patched code that still contains the SQL string (exp-20).]

#figbox("figs/fig_e_tokenheat.png")[#code("FIG-E") Per-token probe scores (qwen-coder 32B, L25) on a *patched, parameterized* function — already safe, yet the SQL string saturates the probe.]

#pt("DAT6")[*So we train the null the audit suggests: a char-n-gram scanner. It performs BETTER than the probe* — 0.803 vs 0.776 on identical splits and evaluation (exp-24). A naive text scanner would have outperformed our LLM probe.]

#figbox("figs/fig_b_ladder.png")[#code("FIG-B") The baseline ladder on identical test tokens: keywords 0.57–0.63, language indicator 0.677, unigrams 0.694, probe 0.776, char-n-grams 0.803.]

#pt("DAT7")[*Surface features also reproduce the probe's "structure":* 3 of 4 family-transfer blocks we had read as representation geometry come out of the n-gram baseline too (exp-24 vs exp-21).]

#figbox("figs/fig_f_blocks.png")[#code("FIG-F") Family-block transfer means, probe vs surface: the within/cross-family pattern survives with no hidden states at all.]

#pt("DAT8")[*Where is the n-gram getting it? Python vs C is a SVEN imbalance.* Injection is 92% Python, memory 100% C/C++ (exp-23). A bare language indicator scores 0.677 on the exact headline token set — recovering ~64% of the probe's AUC margin over chance with zero vulnerability information.]

#figbox("figs/claim_langmethod.png")[The language-baseline method: (a) positives are 4× denser in Python; (b) a language-only scorer gets 0.677 on the identical eval; (c) within-language AUC removes the confound entirely.]

#pt("DAT9")[*Language-stratified rescore: still better than random, much less impressive.* Within-Python 0.80–0.86; within-C/C++ 0.56–0.66 — on every model (exp-23). The headline lived in Python.]

#figbox("figs/fig_c_withinlang.png")[#code("FIG-C") Within-language rescore across all 7 models.]

#pt("DAT10")[*The rule this buys:* a "the model knows X" probe result needs text-only and dataset-composition nulls on the same splits; without them it cannot be told apart from grep.]

= Residues — what survives so far (with open caveats)

#pt("SUR1")[*The matched-patch control and its point:* per-CWE evals can still ride language/template/style. Matched-patch scores the vulnerable function only against its own fixed version — everything except the fix held constant. Under it, *CWE-125 keeps a small signal* (0.633/0.657 on both models, CIs > 0.5, stable over 15 CV folds); CWE-416 weakly; CWE-476 collapses (exp-25).]
#pt("SUR2")[*Open caveat — the comparison that's missing:* we never ran the char-n-gram baseline under matched-patch negatives, so "survives matched-patch" shows a non-language, non-template signal but NOT yet a non-lexical one. Until surface is run in the same regime, this residue is suggestive, not meaningful. *(Being chased — see #code("FUT1").)*]

#figbox("figs/fig_d_matchedpatch.png")[#code("FIG-D") Per-CWE memory probes under progressively stricter negatives; surface baseline under the same regimes still missing.]

#pt("SUR3")[*PrimeVul (second dataset, all C/C++): no transferable memory-family direction* (mem→mem 0.536 ≈ mem→other 0.537; per-CWE diagonals real but idiosyncratic, exp-26). *Open caveat:* this analysis is preliminary — one model, no surface baselines, single-split cross-dataset cells. *(Being chased — see #code("FUT2").)*]

= Why hallucination probes worked and ours mostly didn't

#pt("THE1")[*The separating variable: how much of the target property is already written in the input text.*]
#pt("THE2")[A fabricated name reads exactly like a real one; "fabricated" exists only in the model's state, so a surface baseline cannot compete.]
#pt("THE3")[Vulnerability largely #emph[is] the text (SQL sinks, `strcpy`, language identity); the lexical ceiling sits high and a trained probe lands on it.]

#finding[The probe didn't read the model's security beliefs; it re-derived a worse version of grep. Whether a real belief exists somewhere we never probed is the open question driving the next phase.]

= Caveats, and what we'll do about them

#pt("LIM1")[*Design gap:* we probed single functions out of repo context and never checked the model holds any belief about them at all.]
#pt("LIM2")[Verbalized AUC 0.49–0.62 on the same task fits "knowledge absent" as well as "knowledge hidden".]
#pt("LIM3")[Two readings of all our negatives — probes lose to surface, or there was no belief to read. The next experiments separate them.]
#pt("FUT1")[*Surface under matched-patch:* run the char-n-gram baseline with matched-patch negatives so the CWE-125 residue gets the comparison that makes it meaningful (in progress).]
#pt("FUT2")[*Proper PrimeVul analysis:* second model, CIs on the cross-dataset cells, surface baselines, and PV's own paired structure exploited (in progress).]
#pt("NXT1")[*Random-direction nulls:* AUC distribution of random unit vectors per layer; the trained probe reported as a percentile of that null (the sleeper-agents check — random directions hit 0.95 there).]
#pt("NXT2")[*QA-contrast probe:* User: ⟨code⟩ "Is this code vulnerable?" / Assistant: "Yes/No, this code is (not) vulnerable" — contrast activations at the answer tokens (sleeper-agents technique).]
#pt("NXT3")[*Generation pivot:* the model writes the code, we probe generation-time states, labels come from analyzing what it wrote. The monitor we actually want watches a model writing.]

= Coda

#pt("COD1")[Tally: real signal ≈0.78 AUC; two-thirds of its margin is language; char-n-grams match the rest; the candidate survivor is one CWE at ≈0.64 — pending its surface comparison.]
#pt("COD2")[Open bar: recall at 1% FPR with a frozen threshold — the operating point a deployed monitor must clear.]

= Figures (status)

#pt("FIG-A")[Headline per model, linear bars + MLP diamonds. At #code("RES1")/#code("ABL2"). `make_figs.py`.]
#pt("FIG-B")[Baseline ladder on identical tokens. At #code("DAT6"). exp-24 design1 + exp-23 null.]
#pt("FIG-C")[Within-language paired bars, 7 models. At #code("DAT9"). exp-23 summary.]
#pt("FIG-D")[Matched-patch forest, 2 models × 3 CWEs × 3 negative regimes. At #code("SUR1"). exp-25.]
#pt("FIG-E")[Token heatmap, patched parameterized SQL example. At #code("DAT5"). exp-16 npz + exp-20.]
#pt("FIG-F")[Probe-vs-surface family-block bars. At #code("DAT7"). exp-21 + exp-24 design4.]
#pt("FIG-G")[Cross-dataset transfer SVEN↔PrimeVul (exp-22). Supplementary (#code("SUP6")).]
Claim-validation versions of every plot: `make_claim_figs.py` / `draft-claims.pdf`.

= Supplementary (pulled out of the main thread)

#pt("SUP1")[Related work, per paper: MoC, Openia, Ribeiro; hallucination/truthfulness lineage (Burns, Azaria-Mitchell, Marks-Tegmark, Orgad, Levinstein-Herrmann, sleeper-agents probes).]
#pt("SUP2")[Probe-vs-verbalized detail + prompt-specialization symmetry (exp-05/14/15/17).]
#pt("SUP3")[Additive-fix blind spot detail: label/granularity ablations, token > line (exp-19, ADR-0004).]
#pt("SUP4")[Training detail: span-max loss α=1 > α=10 (exp-03); layer geometry — linear peaks mid–late, no universal depth fraction; small models' MLP optimum sits early/mid (exp-02/12/18).]
#pt("SUP5")[Full experiment ledger with per-experiment links (`docs/project-log.md`).]
#pt("SUP6")[Cross-dataset transfer is asymmetric: SVEN→PrimeVul ≈ chance, PrimeVul→SVEN ≈ in-domain — the bigger trainer learns the more general direction (exp-22). → #code("FIG-G")]
#pt("SUP7")[Steering the probe direction at ±4σ leaves the model's verbalized judgment unchanged — a correlate, no control knob (exp-13).]
