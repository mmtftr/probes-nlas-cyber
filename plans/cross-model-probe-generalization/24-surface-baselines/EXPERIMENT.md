[ai-generated]

# exp-24 — token-level surface baselines vs probe headline numbers

Field-standard control: a probe claim about *internal representations* is only
evidence of representational content to the extent it **beats trivial surface
features under the identical split and eval** (Hewitt & Liang control tasks;
arXiv:2509.03888). The project never ran token-level surface baselines against
its headline `tokens_code_auc`. This experiment establishes the surface-feature
ceiling for every headline eval design.

## 1. Aim

Can token-identity / char-n-gram / regex / language-indicator classifiers,
trained on the SAME seed-42 group split and scored with the SAME
`tokens_code_auc`, match the Qwen-32B probe? Specifically:

- Does a **pure language indicator** (C-vs-Python) reproduce exp-10/21's
  "memory recovery" (memory diag 0.64–0.77)?  ⟹ tests whether claim #3
  (capacity-allocation) needs any internal representation.
- Does a **surface-only combined LR** reproduce exp-21's family-structured 9×9
  transfer (within-family >0.5, cross-family <0.5)?  ⟹ if yes, exp-21's
  "≥2 directions" geometry collapses to surface/dataset structure.
- Where the probe **exceeds** the surface ceiling (esp. within-language) — that
  margin is the project's real representational result; quantify it.

## 2. Inputs

- **Substrate (probe token axis, reused — NO re-extraction):** exp-16 per-token
  logit dump `16-token-logit-dump/results/logitdump_Qwen_Qwen2.5-Coder-32B-Instruct/logits_layer25.npz`
  — per-token `y` (annotated `token_labels` ∩ extractor labelling, == exp-21
  positives), `is_code` (tree-sitter live-code mask), `is_test` (canonical
  seed-42 20% group hold-out, 292 test ex), `char_start`/`char_end` (Qwen fast
  tokenizer offset mapping into `code`, max_length=2048), `prob` (the L25
  general probe's per-token sigmoid — the comparison target), `example_id`.
  Using the dump's offsets guarantees a **byte-identical token axis to the
  probe** (mirrors `src/data/extract_token_activations.py` — the dump *is* that
  extraction). Layer 25 → `test_tokens_code_auc` = 0.7758 (≈ headline 0.776).
- **Dataset:** `data/dataset.jsonl` (1430 rows: code, cwe, lang, label,
  token_labels). `example_id` is the row index; `char_*` index into `code`.
- **Split:** `data/sven_split_meta.json` (seed 42, frac 0.2, 141 heldout pair
  groups) — identical to every probe eval; the dump's `is_test` encodes it.
- **Probe comparison numbers:** exp-16 dump (general, design 1);
  `21-per-cwe-cross-cwe/results/qwen32b/transfer_allclean.json` (per-CWE
  diagonal + 9×9 transfer, designs 2 & 4).

## 3. Outputs (all under this dir)

- `surface_baselines.py` — feature builders + LR training (sparse, CPU).
- `run_exp24.py` — runs designs 1–4, writes `results/*.json`.
- `results/design1_general.json`, `design2_percwe_diag.json`,
  `design3_within_language.json`, `design4_transfer_matrix.json`.
- `RESULTS.md`, `STATUS.md`, `LOG.md`.

## 4. Result format

- Designs 1–3: side-by-side `tokens_code_auc` (headline) + per-example-mean AUC
  (secondary) with bootstrap-over-examples 95% CI and n per cell, probe vs each
  baseline; Δ = probe − best-surface.
- Design 4: 9×9 surface (e)-LR transfer matrix + inj/mem block means (incl./
  off-diagonal), side-by-side with exp-21's probe blocks.
- Every cell with <10 test vuln examples flagged **untrusted**.

## 5. Interpretation hints

- Baseline (e/d) ≈ probe on design 2 (memory diag) → "memory recovery" needs no
  internal representation; claim #3 unsupported (the confound, not the geometry).
- Surface (e) reproduces family-block structure in design 4 → exp-21's "≥2
  directions" inference collapses to surface/dataset structure.
- Probe ≫ baselines **within-language** (design 3) → genuine representational
  content; the margin is the real result.

## Supervision caveat (state in RESULTS)

Baselines get **per-token** supervision (positive = `y==1 ∩ is_code` train
tokens; negative = other live-code train tokens), whereas the probe trains
**span-max** (one max-token signal per example). So these baselines are an
**upper lexical bar** — deliberately generous to the surface side. That is the
point: any probe number not clearing this bar is not evidence of representation.

## For agents — exact knobs

- Token-identity feature = `code[char_start:char_end]` (stripped) — the literal
  substring the Qwen tokenizer assigned; tokenizer-agnostic 1:1 proxy for token
  id, vocab capped to train-observed strings.
- Window feature = `code[max(0,cs-48):ce+48]` per token.
- `clean` example = `label==0 and not cwe` (the exp-06/10/21 negative pool).
- Design 2/4 negatives = clean **examples'** live-code tokens (shared pool),
  positives = `y==1 ∩ is_code` tokens in the CWE's vuln examples — mirrors
  `21-.../transfer_allclean.py` cell construction exactly.
- INJ = {089,078,022,079}; MEM = the rest. min_trust = 10 test vuln examples.
- No GPU. liblinear/saga, sparse. Subsample TRAIN negatives if needed (cap
  documented); **eval pools always full**.
