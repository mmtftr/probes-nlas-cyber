[ai-generated]

# exp-22 — PrimeVul-Paired as a second training set (vs SVEN)

Status: **DONE (2026-06-08).** 2-model first cut ran; see `RESULTS.md`.
Headline: asymmetric transfer — SVEN probe doesn't generalize to PrimeVul, but
PrimeVul's transfers to SVEN. Next: shared-CWE-stratified eval + all-7-models +
CV error bars.

## 1. Aim
Test whether the vulnerability-belief probe trained on **PrimeVul-Paired**
(larger, cleaner-labeled, different C/C++ distribution) matches the SVEN-trained
probe, and whether probes **transfer across datasets** — i.e. is the probe
learning a transferable vulnerability direction or SVEN idiosyncrasies
(framing §6 "detects dataset bias", §7.3 dataset-bias OOD test).
Hypothesis: PrimeVul→PrimeVul honest token-AUC ≈ SVEN→SVEN (~0.77); cross-dataset
(SVEN→PrimeVul) is positive but lower, the gap = dataset-specific signal.

## 2. Inputs
- **Dataset:** PrimeVul-Paired — HF `colin/PrimeVul`, config `paired` (MIT).
  Rows alternate target=1 (vulnerable `func`) / target=0 (patched `func`);
  consecutive rows = a before/after pair from one fixing commit. C/C++ only.
  Pairs: train/valid/test = 3789/480/435; 9408 rows; C 8526 / C++ 882; 111 CWEs.
  Apply the **same honest regime as ADR 0004**: subtractive subset (fix deletes
  ≥1 live-code char), per-token labels (tight-diff ∩ is_code), X negatives.
  **Subtractive fraction (measured locally): 0.537** (2525 subtractive pairs) —
  lower than SVEN's ~67% *fraction*, but **2525 subtractive pairs ≈ 5.3× SVEN's
  478** in absolute localizable-vuln data. Test: 435 pairs (209 subtractive).
  (Char-level tight diff capped at 6000 chars — all SVEN funcs ≤5833 stay
  char-level/exp-19-consistent; PrimeVul's larger funcs use a line-level fallback.)
- **Models (first cut):** Qwen2.5-Coder-7B-Instruct (best on SVEN) +
  gemma-3-12b-it. (Decision-fork: all 7 exp-19 models vs this 2-model cut.)
- **Activations:** token-level hidden states, operating layer per model
  (exp-16 table), **vLLM** for Qwen / **HF** for gemma (matches exp-19 CV).
  New extraction — PrimeVul code never extracted. KEEP_ACTS=1.
- **Probe:** span-max linear, token granularity, X (code_only) negatives,
  is_code-gated positives — exp-19 canonical regime, loss unchanged.

## 3. Outputs
- wandb run + artifacts: PrimeVul-Paired dataset artifact (or HF SHA),
  per-model probe npz, `metrics_grid.json`.
- Local: `RESULTS.md` + cross-dataset table; `characterize_results.json`
  (already produced); probe npz under `results/`.

## 4. Result format
Cross-dataset table — train ∈ {SVEN, PrimeVul} × eval ∈ {SVEN-test,
PrimeVul-test}, per model, honest token-code-AUC + pairAcc-sub + **g-mean²**:

| model | SVEN→SVEN | SVEN→PV | PV→PV | PV→SVEN | PV subfrac |
|---|---|---|---|---|---|

Plus: PrimeVul subtractive fraction (+ C-vs-C++ lang sensitivity), top-CWE /
language breakdown, and a **CWE-family-stratified** token-AUC slice (see §5).

## 5. Interpretation hints
- **SVEN→PV ≈ PV→PV** → probe found a transferable vuln direction; dataset-bias
  low (good for the §6 failure-mode check).
- **SVEN→PV ≪ PV→PV** → SVEN probe overfit SVEN; direction is dataset-specific.
- **PV→PV ≫ SVEN→SVEN** → PrimeVul is a cleaner/easier (or just larger) signal.
- **PV subfrac ≫ 67%** → PrimeVul is a better localizable-vuln trainer (fewer
  additive blindspots); **≈ 67%** → same additive ceiling as SVEN.

### Confounds to control (rev-primevul)
- **Domain mix, not just dataset bias.** SVEN is injection/Python-heavy; PrimeVul
  is C/C++ memory-safety-heavy (top CWEs 119/125/787/476/416). A raw SVEN→PV drop
  could be *injection-trainer → memory-eval* domain shift rather than overfitting.
  Control by (a) the **C/C++-only SVEN slice** for the headline, and (b) a
  **shared-CWE / family-stratified** token-AUC (CWE-125/476/416/787/190 appear in
  both) — only the stratified gap is clean evidence of dataset bias.
- **Lang sensitivity.** ~36% of PrimeVul files are extension-less and ~1% are
  `.h` (ambiguous C-vs-C++). The subtractive fraction is reported under both
  defaults; if Δ is small it's robust (expected — the live-code gate is similar
  across C/C++).

## For agents
- Cross-eval requires a SVEN probe evaluated on PrimeVul tokens and vice-versa;
  reuse `19-subtractive-regime/train_grid.py` label machinery (`compute_labels`,
  `build_pairs`) — generalise it to read a PrimeVul-shaped jsonl (fields: `func`,
  `target`, `cwe`, `file_name`, pair = consecutive rows).
- Honest eval label = tight-diff(before,after) ∩ is_code, code-only tokens
  (identical to SVEN harness) so SVEN and PrimeVul numbers are directly comparable.
- Cluster: `debug` partition, 4 nodes × ≤22.5 min, resumable per-model files.
  Extraction is the cost; probe training is trivial (batch all configs).
- Language fairness: SVEN is bilingual (≈half Python). For SVEN↔PrimeVul cross-
  eval, restrict SVEN to its **C/C++ slice** so the comparison isn't confounded
  by language. (Decision-fork.)
- Build step: `build_primevul.py` (download paired splits → before/after pairs +
  subtractive membership), mirroring `19-subtractive-regime/build_subtractive.py`.
