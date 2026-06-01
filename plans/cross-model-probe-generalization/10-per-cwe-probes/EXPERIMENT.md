[ai-generated]

# 10 — Per-CWE probes vs the general probe

Targets the sweep-6 per-CWE gap (SQLi 0.92, cmd-inj 0.82, path-trav 0.78 vs
OOB-read 0.56, NULL-deref 0.55, UAF 0.52). Train a probe **specialized to one
CWE** (or CWE-family) and compare it, on the same per-CWE test subset, against
the GENERAL probe (trained on all data — the existing ~0.788 / per-CWE breakdown
numbers). Runs on CACHED acts at each model's single best layer — no
re-extraction.

## 1. Aim

Does per-CWE specialization **close the memory-safety gap** (⇒ the signal exists
but the general probe under-allocates capacity to it), or do per-CWE probes
**also fail** on C/memory-safety (⇒ the signal is largely ABSENT in activations,
partly answering the HANDOFF open question)? Data-scarcity-aware: SVEN CWEs are
imbalanced; some memory-safety CWEs have very few test pairs.

## 2. Inputs

- **Cached acts (no extraction)** at each model's best layer:
  `runs/layersweep_<slug>/acts/layer_NN.npy` (+ `offsets.npz`, `y.npy`,
  `example_ids.npy`, `meta.json`).
  - Primary: `Qwen/Qwen2.5-Coder-32B-Instruct` (slug
    `Qwen_Qwen2.5-Coder-32B-Instruct`), **layer 25** (best, 0.788 overall).
  - Secondary: `google/gemma-3-27b-it`, **layer 19**.
- **Data/recipe:** `data/dataset.jsonl` (SVEN before/after, 1430 rows),
  `sven_split_meta.json` (seed-42 20% group hold-out). Span-max probe, honest
  `tokens_code` metric. EXACT group-aware splits + 15% VAL carve from
  `train_all_layers.py` (so per-CWE results are apples-to-apples with 06).
- **Probe heads:** `LinearProbe` (default) or `MLPProbe`, via
  `train_one_layer(probe_factory=…)`.
- **Dataset invariant (verified):** `cwe != null` ⟺ `label == 1` — all 715
  positives carry a CWE; all 715 negatives have `cwe == null`. So "negatives" =
  every `cwe == null` row; there is no per-CWE negative-labelling ambiguity.

## 3. Outputs

Per model, one JSON: `runs/per_cwe_10_<slug>_<head>_<negpool>.json`. Per CWE it
records `n_train_pos`, `n_test_pos`, `n_neg_fit/test` (scarcity), the
head-to-head `general_tokens_code_auc` vs `specialized_tokens_code_auc`, their
Δ, a `trust` flag (False if `n_test_pos < 10`), and the family
(injection / memory). `compare_per_cwe.py` tabulates these across models +
families.

## 4. Result format

Per-CWE head-to-head table:

```
CWE       family     tr_pos te_pos  general   spec      Δ    trust
CWE-089   injection    159     44    0.936   0.94x  +0.00x   ✓
CWE-125   memory        84     19    0.606   0.xxx  ±0.0xx   ✓
CWE-787   memory        42      5    0.5xx   0.xxx  ±0.0xx   LOW-n
…
```

plus a per-FAMILY roll-up (n_test_pos-weighted, trustworthy CWEs only):
`injection: general / specialized / Δ` and `memory: general / specialized / Δ`.

## 5. Interpretation hints

- **Memory family Δ ≈ 0, both AUCs low** ⇒ specialization buys nothing on
  C/memory-safety → the signal is largely **ABSENT** in activations (not a
  capacity-allocation problem). This is the substantive answer to the HANDOFF
  open question.
- **Memory family Δ > 0, spec lifts a low general AUC toward injection levels**
  ⇒ the signal **EXISTS** but the general probe under-allocates capacity to it;
  per-CWE (or per-family) heads recover it.
- **Injection family Δ ≈ 0** (expected): the general probe is already near its
  ceiling on SQLi/cmd-inj, so a specialized probe shouldn't help — confirms the
  baseline isn't being beaten by mere overfitting to a smaller, easier subset.
- **Spec > general on injection but ALSO on memory** by a similar margin ⇒
  suspect the lift is generic small-subset overfit, not memory-specific signal
  recovery — discount accordingly (and lean on the family roll-up Δ, not single
  noisy cells).

## Critique

- **Per-CWE training slashes data → high variance / wide CIs**, worst exactly
  where it matters (the scarce memory-safety CWEs). General-probe fit pool is
  ~970 examples; a specialized CWE-787 probe fits on ~42 positives + the negative
  pool. Expect the specialized AUC to be noisier than the general one on the
  same test subset.
- **Per-CWE TEST sets are tiny** — from the dataset (split-aware, this seed):

  | CWE | family | train_pos (fit) | test_pos | trust? |
  |---|---|---|---|---|
  | CWE-089 SQLi | injection | 159 | 44 | ✓ |
  | CWE-078 cmd-inj | injection | 86 | 19 | ✓ |
  | CWE-125 OOB-read | memory | 84 | 19 | ✓ |
  | CWE-476 NULL-deref | memory | 49 | 16 | ✓ |
  | CWE-022 path-trav | injection | 38 | 15 | ✓ |
  | CWE-416 UAF | memory | 42 | 14 | ✓ |
  | CWE-079 XSS | injection | 39 | 10 | ✓ (borderline) |
  | **CWE-787 OOB-write** | memory | 42 | **5** | ✗ LOW-n |
  | **CWE-190 int-overflow** | injection? | 30 | **4** | ✗ LOW-n |

  (fit_pos = train_pos minus the 15% VAL carve; numbers above are train-side
  positives pre-carve — the runner reports the exact post-carve `n_train_pos`.)
  An AUC on **4–5 positives is not trustworthy** (CI half-width ≳ ±0.15); those
  two CWEs are reported but flagged `LOW-n` and EXCLUDED from the family
  roll-up. **The two worst-AUC memory CWEs in sweep-6 (UAF 0.52, OOB-write) are
  also the scarcest** — so a null result there is partly confounded with
  scarcity, not purely "signal absent." CWE-125 (19 test pos) and CWE-476 (16)
  are the trustworthy memory-safety cells to lean on.
- **Logic of the key inference:** if general ≈ per-CWE on the *trustworthy*
  Python/injection cells AND both stay low on the *trustworthy* memory cells
  (125, 476), that points to **signal absence**, not capacity — the cleanest
  reading. If specialization lifts 125/476 substantially, capacity-allocation is
  in play.
- **MLP head needs more data** → strictly worse under scarcity; default linear.
  Run MLP only as a secondary check on the data-rich injection CWEs.
- **Family granularity mitigates scarcity:** pooling all memory CWEs into one
  "memory" probe (`--cwe memory`) gives ~217 train positives — a more
  trustworthy specialized memory probe than any single memory CWE. If even the
  pooled memory probe stays near chance, the absence conclusion is strong.

## For agents

- **Runner:** `per_cwe_probe.py` — trains the general probe ONCE on the fit
  pool, then per target CWE trains a specialized probe on
  `{CWE-X positives} ∪ {negative pool}` and scores BOTH on the identical subset
  `{CWE-X test positives} ∪ {negative test pool}`. Honest `tokens_code_auc` via
  `src.eval.honest_scoring.honest_token_aucs`. CWE/lang read straight from
  dataset rows (same derivation as `06/breakdown_lang_cwe.py`).
- **Aggregator:** `compare_per_cwe.py <json…>` — per-CWE table + family roll-up.
- **Submit:** `submit_10.sh` — single best layer, **1 GPU**, 30 min walltime,
  idempotent (skips a model whose OUT exists; skips a model whose cached acts
  are missing). Runs AFTER 08 (scheduler = one job at a time).

  ```bash
  # on the login node, AFTER any 08 job has left the queue:
  cd ~/scratch/probes && git -C repo pull
  bash repo/plans/cross-model-probe-generalization/10-per-cwe-probes/submit_10.sh
  # variants (lead's design forks):
  HEAD=mlp  bash …/submit_10.sh        # MLP head (injection CWEs only, really)
  CWE=memory bash …/submit_10.sh       # pooled memory-family probe (more data)
  NEG_POOL=same_lang bash …/submit_10.sh
  ```

  Cost: per model ≈ (1 general fit + ≤9 tiny per-CWE fits) at one layer on
  cached acts → a few minutes each; both default models well under the 30-min
  debug cap, 1 GPU.
- **Local test:** `python test_grouping.py` (no GPU/cluster) — asserts the
  group-aware split + 15% VAL carve are leakage-free globally and after CWE
  filtering. Passes.

### TODO(adhoc-decision) — forks the lead must settle (also marked at the code sites)

1. **Negative pool** for a specialized probe (`--neg-pool`): `all` (default,
   every `cwe==null` neg — keeps the head-to-head EXACTLY comparable to the 06
   general breakdown) vs `same_lang` (negatives whose lang matches the CWE's
   dominant lang — a "specialized" probe that also sees only same-language clean
   code) vs `same_family` (no per-neg family label exists → falls back to
   `same_lang`, flagged in output). Changes what "specialized" means.
2. **Granularity:** per-individual-CWE vs per-CWE-**family** (`--cwe injection`
   / `--cwe memory`). Families give ~3–5× more data per probe → far more
   trustworthy under scarcity. Default ALL runs every individual CWE; family
   roll-up is computed regardless, but a *trained* family probe needs an
   explicit `--cwe memory` run.
3. **Head:** linear (default) vs MLP (`--head mlp`). MLP needs more data → worse
   under scarcity.
4. **Split under scarcity:** the runner reuses the GLOBAL seed-42 group split
   filtered to the CWE (apples-to-apples with 06). A pooled/k-fold scheme for
   the tiny CWEs (787, 190) would give tighter CIs but breaks comparability with
   the general baseline — NOT implemented; flagged for the lead.
5. **CWE-190 family:** listed `injection` (sweep-6's taint/data-flow "detected"
   set) but is a C integer-overflow, arguably a memory-safety precursor.
   One-line change in `FAMILY` if the lead reassigns it.
