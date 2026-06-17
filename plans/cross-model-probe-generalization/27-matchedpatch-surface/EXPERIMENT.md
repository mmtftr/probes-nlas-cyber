[ai-generated]

# exp-27 — surface baselines under matched-patch negatives

Closes the gap left between exp-24 (surface ceiling, but only under all-clean
negatives) and exp-25 (matched-patch control, but only for probes): the
char-n-gram surface baseline was never scored under matched-patch negatives, so
the CWE-125 residue (probe 0.633/0.657, CI>0.5 both models) is not yet
demonstrably non-lexical.

## 1. Aim

Does the lexical surface ceiling collapse under exp-25's language/function-
matched negative regimes? Hypothesis to adjudicate: the matched-patch CWE-125
probe signal is **non-lexical** (surface ≈ 0.5 under matched-patch while probe
stays 0.63–0.66). Alternative: surface ≥ probe under matched-patch → the
residue is lexical after all and the blog claim dies.

## 2. Inputs

- **Substrate (token axes, reused — NO re-extraction):** exp-16 per-token logit
  dumps `logits_layer25.npz` for Qwen2.5-Coder-32B-Instruct (561,266 tokens) and
  gemma-3-1b-it (690,148 tokens) — per-token `y`, `is_code`, `is_test`,
  `char_start/end`, `prob`, `example_id`. Token axes are byte-identical to each
  model's probe eval (exp-24 verified pool counts bit-exact vs exp-21/25 for
  qwen; gemma axis gated on exact count match vs exp-25, see gates).
- **Dataset/split:** `data/dataset.jsonl` (1430 rows) + `data/sven_split_meta.json`
  (seed-42, 20% group hold-out; byte-identical to exp-25's archived copy — verified).
- **Surface models (exp-24 `features.py`, reused):** char 3–5-gram hashed-window
  LR (headline surface), combined a+b+d LR, token-unigram LR, keyword LR /
  untrained, language indicator; `liblinear`, C=1.0, NEG_CAP 60k, seed 42.
- **Training recipe (= exp-24 design-2, held fixed across regimes):** per CWE,
  positives = `y==1 ∩ is_code` TRAIN tokens of CWE-X vuln examples, negatives =
  all-clean TRAIN live-code tokens (subsampled at fit). Secondary: conly-trained
  variant (C/C++-only clean train negatives), mirroring exp-25's retrain check.
- **Negative regimes at EVAL (= exp-25 `deconfound.py`, positives held identical):**
  `allclean` (repro anchor), `conly`, `pyonly` (context), `matchedpatch`
  (the CWE's OWN paired patched safe-half, `(_file_name,_func_name)` i-th-vuln ↔
  i-th-safe pairing, safe half asserted held-out).
- **Probe comparison columns (cited, not re-run):** exp-25
  `results/deconfound_{qwen32b,gemma1b}.json` allclean-trained diagonal + CIs.

## 3. Outputs (all under this dir)

- `run_exp27.py` — stage A (verbatim exp-24 design-2 re-run = repro gate +
  score capture), stage B (regime pools + count gates), stage C (eval + CIs).
- `results/exp27_qwen32b_axis.json`, `results/exp27_gemma1b_axis.json` —
  per CWE × regime × baseline: auc, lang_null, 95% CI (1000-boot,
  bootstrap-over-examples, paired safe-half resampling for matchedpatch),
  n_pos_tok/n_neg_tok, plus the cited probe column.
- `RESULTS.md` — head-to-head table per CWE × regime: probe (exp-25) vs
  char-n-gram (this exp), with CIs. `STATUS.md` — live status for coordinator.

## 4. Result format

Headline table (per model axis): rows = CWE-125/416/476 (+ injection rows as
positive control), columns = {allclean, conly, matchedpatch}, each cell
`probe (exp-25) vs char-ngram (exp-27)` with 95% CIs. Δ = probe − char-ngram
under matchedpatch is THE number. Metric = `tokens_code_auc` throughout;
example-mean AUC secondary. Untrusted cells (<10 test vuln ex) flagged.

## 5. Interpretation hints

- **Surface ≥ probe under matched-patch** → CWE-125 residue is lexical; blog
  SUR1/SUR2 rewritten to a clean negative (no demonstrated non-lexical signal).
- **Surface ≈ 0.5, probe 0.63–0.66 (CIs separated)** → first demonstrably
  non-lexical, non-language signal in the project; blog SUR2/FUT1 upgrade.
- **In between (CIs overlap)** → report the margin honestly, no overclaim;
  the matched-patch surface number still replaces the missing control.
- Injection rows: expect surface to stay high under matchedpatch only where the
  patch removes the sink string (cf. exp-20: patches often keep the SQL string →
  surface FPs); a surface drop there with probe persistence is context, not the
  claim.

## Gates (hard, in order)

1. **Design-2 repro (qwen axis):** stage A re-runs exp-24 design-2 with the
   identical rng sequence; every all-clean `tokens_code_auc` cell must match
   `24-surface-baselines/results/design2_percwe_diag.json` to ≤1e-9.
2. **Count gates:** per-cell `n_pos_tok`/`n_neg_tok` must equal exp-25's
   `deconfound_*.json` exactly — qwen: pos 405/321/313, neg allclean 38910,
   conly 26211, mp 10413/4126/6186; gemma: pos 446/389/362, neg allclean 44581,
   conly 29847, mp 11762/4840/7245. Mismatch → stop, investigate, no results.
3. **lang_null sanity:** ≡0.5 under conly/matchedpatch (memory CWEs), ≈0.63–0.65
   under allclean — must match exp-25's lang_null column.

## For agents

- Stage A imports exp-24's `run_exp24` module (substrate build at import; needs
  the dump npz at the repo path — copied locally, gitignored). The design-2 loop
  is copied verbatim INCLUDING its `pool_eval` calls so the shared rng stream is
  bit-identical; fitted per-token score vectors are captured per CWE.
- Gemma axis has no exp-24 reference (exp-24 was qwen-only): its gate is #2/#3
  only; rng seed 42 fresh. Substrate loader parameterized by dump path.
- CI recipe mirrors exp-25 `diag_ci`: resample positives w/ replacement;
  matchedpatch negatives = paired safe halves of the resampled positives;
  other regimes resample the clean pool independently. 1000 boots.
- Do NOT write into `24-.../results/`. Everything lands in `27-.../results/`.
