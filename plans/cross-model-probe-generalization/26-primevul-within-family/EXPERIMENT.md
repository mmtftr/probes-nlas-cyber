[ai-generated]

# exp-26 — PrimeVul within-C/C++ family structure (language held fixed)

## Aim
exp-21 found family-structured cross-CWE transfer on SVEN (within-family off-diag
> chance, cross-family below chance), used to argue ≥2 coarse "family directions".
But SVEN's families are language-confounded (injection≈Python, memory≈C/C++), so
that geometry could be a language+dataset artifact (see exp-25). PrimeVul-Paired is
**100% C/C++** — a single-language dataset. Question: does per-CWE/family probe
structure replicate **with language held fixed**? If within-family transfer > cross-
family appears within one language on an independent dataset, family geometry is
real (independent of language AND of SVEN). If not, exp-21's geometry was
language/dataset structure and per-CWE probes are CWE-idiosyncratic.

## Inputs (NO extraction needed — both act sets already cached)
- **Model / layer:** Qwen2.5-Coder-7B-Instruct, **L16** (exp-22 operating layer).
  `$RUNS/primevul_Qwen_Qwen2.5-Coder-7B-Instruct/pv_acts/token_activations_layer16.npz` (118 GB),
  `.../sven_acts/token_activations_layer16.npz` (8 GB) — `offsets.npz` byte-identical
  to exp-21's SVEN offsets ⇒ sven_acts = full SVEN, same row set/order.
  (gemma-3-12b-it L15 has the same pair of dirs — optional 2nd model if budget allows.)
- **Datasets:** PV = `22-primevul-paired/primevul_dataset.jsonl` (9408 rows, SVEN-shaped,
  `_split` train/valid/test embedded, `cwe`/`vuln_type`, `token_labels`); SVEN =
  `$DATA/dataset.jsonl` + `$DATA/sven_split_meta.json`.
- **Recipe:** exp-10 all-clean — train CWE-X vuln ∪ ALL clean (label==0, no cwe),
  annotated `token_labels==1` positives, honest code-only token eval, shared clean-test
  negatives. **In PV the all-clean design is language-clean BY CONSTRUCTION** (all PV
  is C/C++) — no Python contrast can leak into the negative pool. State this explicitly.

## PV CWE inventory (test vuln examples, measured locally 2026-06-09)
- **Memory-class (≥10 test pos):** CWE-787 (72), 125 (47), 476 (39), 416 (29), 119 (14), 190 (11), 415 (10).
- **Non-memory-class (≥10 test pos):** CWE-703 (47, improper-check), 200 (16, info-exposure), 20 (14, input-validation), 369 (14, div-by-zero), 617 (12, assert).
- **Injection (089/078/079) essentially ABSENT** (089/079 not present; 078 test=0; 022 test=6).
  ⟹ PIVOT (as anticipated): family-structure test = **memory vs non-memory families
  within C/C++**, not injection-vs-memory. Records the feasible matrix = CWEs with
  ≥10 test positives (untrusted cells <10 flagged). All clean: train 3789 / test 435.

## Steps
1. **Within-PV transfer matrix** (primary): per-CWE probes (all-clean recipe) over the
   ≥10-test-pos CWEs; full matrix + family blocks (mem→mem / mem→other / other→mem /
   other→other), bootstrap CIs over examples. Language held fixed (all C/C++).
2. **Shared-CWE SVEN↔PV transfer** (open thread, project-log §6): memory CWEs
   125/416/476/787/190 appear in BOTH. Train SVEN memory probes on qwen7b-L16 sven_acts
   (exp-10 recipe), eval on PV (vs PV clean); and PV-trained probes eval on SVEN-C
   (C/C++ slice, vs SVEN-C clean). Same model+layer both sides ⇒ dims match.

## Outputs (in `./runs/exp26/<slug>/`, collected into `results/`)
- `pv_within.json` — within-PV matrix + family blocks + diagonal CIs.
- `cross_shared.json` — SVEN↔PV shared-CWE memory transfer table.

## Result format
- Within-PV: matrix (train-CWE × test-CWE) of `tokens_code_auc`, family-block means
  (mem/other × mem/other, on/off-diagonal) with bootstrap CIs, diagonal + CIs, n flags.
- SVEN↔PV: per shared memory CWE, {SVEN→PV, PV→PV, PV→SVEN-C, SVEN→SVEN-C} token-AUC.

## Interpretation hints
- Within-C family structure replicates (mem↔mem off-diag > chance, mem↔other < or ≈
  chance, CIs separating) → family geometry is REAL, independent of language and SVEN.
- No within-family structure on PV (blocks ≈ chance / indistinguishable) → exp-21's
  geometry was language/dataset structure; per-CWE probes are CWE-idiosyncratic.

## HF revision (step 5, checked 2026-06-09)
- exp-22 used `colin/PrimeVul` config `paired` (sha `4fd71583`, lastModified 2024-09-20)
  — still the **latest** revision of that repo; NOT stale. Newer community re-uploads
  exist (`ASSERT-KTH/PrimeVul` 2025-03-02, `Andrefty/PrimeVul-v0.1-hf` 2025-05-26,
  several `*bigvul_primevul` merges) but no newer official release from the original
  authors. exp-22's copy is current; no rebuild needed.

## For agents
- PV row order = primevul_dataset.jsonl order (extractor was given `--pairs <that file>`).
  Use `_split` for train/test (no external split file). cwe = `vuln_type` (clean string)
  not the stringified `cwe` list.
- PV acts are 118 GB ⇒ load on CPU (won't fit 96 GB GPU); device="cpu" for training
  (matches exp-22 train_cross rationale). Probe training must be batched/bounded.
- Runs AFTER exp-25. No extraction needed.
