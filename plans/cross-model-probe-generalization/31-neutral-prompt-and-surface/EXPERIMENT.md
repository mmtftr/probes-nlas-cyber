[ai-generated]

# 31 — Neutral-prompt commit probe + example-level surface baseline

Two controls on exp-30's positive (commit-position probe decodes vuln at
0.66–0.82, ≫ verbalized + code-token), both raised by the user:

1. **De-prime the prompt.** exp-30 used the *vulnerability* question ("Does the
   code above contain a security vulnerability? yes/no"), so the commit-position
   hidden state is conditioned on "answer a vuln question" — the probe might read
   the model's *answer to that question*, not a spontaneous representation. Re-run
   the exact exp-30 pipeline with a **neutral** prompt — "What do you think about
   this code?" — and probe the same `Assistant:`-boundary position. If the signal
   survives un-primed, the representation is intrinsic; if it collapses, exp-30 was
   "the model can answer a vuln question when asked."

2. **Lexicality control (the decisive one).** An example-level **char-n-gram +
   token-unigram surface classifier** on the raw function text, same splits,
   compared to BOTH the primed (exp-30) and neutral probes with paired-Δ CIs.
   probe ≫ surface ⇒ first genuinely non-lexical reading-frame result; probe ≈
   surface ⇒ the commit-position signal folds back into the lexical story
   (exp-20/24/29).

## Five fields

1. **Aim** — (a) does the commit-position vuln decodability survive a neutral
   prompt? (b) is it lexical? Hypotheses neutral: if intrinsic, neutral ≈ primed;
   surface: if the probe reads meaning, probe > char-n-gram.
2. **Inputs** — neutral extraction reuses exp-30 `extract_lasttoken_hidden.py`
   with `--question "What do you think about this code?"` (same code-before-question
   structure, chat template, assistant generation prompt), 6 it-models, float32,
   all layers. Surface: `data/dataset.jsonl` code text, seed-42 group-clean split.
3. **Outputs** — scratch `lasttoken_hidden_neutral_<slug>.npz` (KEEP); neutral
   `introspection_probe.json` (same schema as exp-30, no verbalized gate);
   `results/surface_vs_probe.json` (per model: char-ngram / unigram example-AUC +
   CIs + paired-Δ vs primed-probe and neutral-probe); comparison figure.
4. **Result format** — per model: primed-probe / neutral-probe / char-ngram /
   token-unigram / verbalized, example-level AUC [95% CI]; paired Δ(probe−surface)
   CIs; within-language for the neutral probe.
5. **Interpretation hints**
   - neutral ≈ primed AND probe > surface ⇒ intrinsic, non-lexical commit-position
     vuln representation (the strong result).
   - neutral ≈ primed AND probe ≈ surface ⇒ intrinsic but lexical (the model
     encodes surface vuln cues at the commit point; still a usable monitor, not
     "understanding").
   - neutral ≪ primed ⇒ exp-30 was question-driven (answer-readout after all);
     the spontaneous representation is weak.

## For agents

- **Metric** = example-level AUC (SECONDARY); token-level headline untouched.
- **Neutral prompt has no yes/no answer** → no verbalized gate (trainer
  `--no-verbalized-gate`); the verbalized COMPARISON still uses exp-17's *primed*
  P(yes) as the "ask-the-model" reference (probe needs no question; verbalized
  needs the vuln question). The <think>-guard still applies in extraction.
- **Surface baseline**: char_wb 3–5 n-grams + token unigrams (TF-IDF) → L2-LR,
  C selected on val, example-level, on the FULL function text; same group-clean
  split; 1000-boot CIs; paired Δ vs the probe's per-example test scores (refit the
  deployable layer/C from the exp-30/31 JSON on the kept hidden npz). This is the
  exp-24 control lifted to example level.
- **Nulls/confounds for the neutral probe**: same as exp-30 (label-permutation,
  random-direction, language/length/within-language).
- **Review**: design review (codex+Opus) before submit; result gate before report.
- Reuses exp-30's validated extractor/merge/trainer (parameterized) + kept primed
  hidden states; no re-extraction of the primed run.
