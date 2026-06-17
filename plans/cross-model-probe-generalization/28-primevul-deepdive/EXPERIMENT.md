[ai-generated]

# exp-28 — PrimeVul deep-dive: surface baselines, CIs, matched-pair

## Aim
exp-26's PrimeVul analysis left three gaps: (1) no surface comparison — is the
per-CWE diagonal (0.61–0.88) and the no-family-cluster result lexical?
(2) the SVEN→PV cells (125 0.668 / 416 0.706 / 476 0.494) are single-split
point estimates; (3) PV's paired structure (vuln + its own fix) was never
exploited — the matched-patch control (exp-25's airtight regime) was never run
on PV. Close all three; stretch: replicate within-PV on a second model.

## Inputs
- **Probe side (NO re-extraction, NO retraining):** exp-26's cached per-token
  logits, qwen7b L16 — `logits_pv.npz` (12 PV per-CWE probes × all 8.0M PV
  tokens), `logits_pv_svenprobes.npz` (5 SVEN-trained probes on PV),
  `logits_sven_{pvprobes,svenprobes}.npz` (SVEN side), `probes_pv.npz` —
  pulled from cluster scratch into `assets/` (gitignored), with
  `pv_offsets.npz` / `sven_offsets.npz` (tokenizer char offsets).
- **Data:** PV = exp-22 `primevul_dataset.jsonl` (9408 rows, 4704 pairs, 100%
  C/C++, `_split` embedded; row 2i = vuln, 2i+1 = its fix); SVEN =
  `data/dataset.jsonl` + `sven_split_meta.json`.
- **Surface:** exp-24 `features.py` recipe — char-3-5-gram HashingVectorizer
  (2^18) on ±48-char windows, LR (liblinear, C=1), NEG_CAP=60k; token-unigram
  LR secondary. Trained per CWE on annotated y==1 live-code train tokens vs
  clean-train code tokens (exp-24 design-2 recipe), on the IDENTICAL token
  axis (pv_offsets.npz) as the probe.
- **Repro gates:** local y_tok/is_code rebuild must reproduce exp-26's
  n_neg_test_tokens=287,864 and per-CWE n_pos_tokens; recomputed point
  estimates must match `pv_within.json` / `cross_shared.json` exactly.

## Outputs
`results/` JSONs + RESULTS.md:
- (a) `pv_surface.json` — surface 12×12 matrix + family blocks + diagonal with
  CIs, probe-vs-surface side by side.
- (b) `cross_cis.json` — SVEN→PV (+ PV→SVEN-C, SVEN→SVEN-C) cells with
  bootstrap-over-examples 95% CIs.
- (c) `pv_matchedpair.json` — per-CWE matched-patch AUC (probe vs surface,
  CIs over pairs) + pairAcc secondary.
- (d) stretch: `pv_within_gemma.json` — gemma-3-12b-it L15 within-PV matrix
  (cluster job on cached acts).

## Result format
Per analysis: tables of `tokens_code_auc` (headline) with 95% CIs and n flags
(<10 test pos untrusted); family blocks token-weighted + unweighted; pairAcc
labelled secondary. Probe and surface columns side by side per cell.

## Interpretation hints
- Surface ≈ probe on the PV diagonal → exp-26's "idiosyncratic but real"
  downgrades to "lexical"; surface blocks also flat → no-family-cluster holds
  trivially (even lexical features don't pool).
- Probe > surface on diagonal, blocks unchanged → genuine per-CWE signal,
  still no family direction (current story holds, strengthened).
- Matched-pair probe > surface (esp. CWE-125) → converges with exp-27 toward
  a real non-lexical residue; report jointly with exp-27.
- SVEN→PV CIs excluding 0.5 for 125/416 → exp-25/26 cross-dataset agreement
  becomes CI-backed; CI spanning 0.5 → downgrade to "suggestive".

## For agents
- Token axis: PV acts row order = jsonl order; eids derived from per-row
  offset counts (contiguity asserted by exp-26 on the same files).
- Eval parity: pos pool = ALL code tokens of CWE-Y test vuln examples labelled
  by annotated y_tok; negatives appended per regime (all-clean = 435 safe
  halves; matched-patch = the CWE's OWN test-pair fixes). exp-25
  `deconfound.py` lines 209–242 is the reference; exp-26 `pv_family.py`
  eval_cell for the all-clean cells.
- Matched-patch bootstrap resamples PAIRS (vuln + its fix move together),
  exp-25 diag_ci style, 1000 reps. Block bootstrap 500 reps, exp-26 style.
- pairAcc = P(max code-token score of vuln > its fix) + 0.5·P(tie), over test
  pairs of the CWE; also the subtractive-only slice (exp-22
  `primevul_membership.json`).
- Surface scores computed only for needed token subsets (train fits + test
  pools) — full 8M-token feature matrix would not fit RAM.
- Cluster only for stretch (gemma L15 pv_family.py rerun, cached acts); cluster
  wiring is out-of-repo.
