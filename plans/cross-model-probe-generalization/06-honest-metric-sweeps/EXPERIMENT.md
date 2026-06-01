[ai-generated]

# 06 — Honest-metric (`tokens_code`) cross-model layer sweep

Extends **02** (same per-layer span-max recipe, same SVEN before/after split)
with two changes that follow from reviewing 02–05:

1. **Headline metric switches to `tokens_code_auc`.** 02 reported `test_tok_auc`
   = `roc_auc_score` over *every* test token (`train_all_layers.py:96–100`).
   That is the `tokens` level, which the protocol docstring flags as inflated:
   ~98% of negatives are trivial (comments, signatures, imports, whitespace), so
   the probe wins by keeping those low. The deployed-relevant number is
   `tokens_code` — live-code-positive (vuln diff lines) vs **live-code-negative**
   (real code that isn't the vuln), comments/sigs/imports/whitespace dropped via
   tree-sitter (`src/eval/code_mask.py`). **We have never computed it in a sweep.**
2. **Roster widens** to the smaller Gemma-3 sizes + base/instruct pairs, to read
   size and post-training (Q5) on the honest metric.

Example-level AUC is recorded but **not** a success axis (user: "not useful to
me, just useful to see"). It rides along; it does not gate anything.

This bundles framing-doc sweeps **1** (honest layer sweep), **2** (base vs
instruct / Q5), **5** (re-run 03/04 on the honest metric), and **6** (per-lang /
per-CWE breakdown). Sweep 3 (proximity-window) is **excluded** — the W-dilation
logic is not trusted yet. Sweep 4 (train-time code-mask) is **07**.

---

## 1. Aim

On the honest `tokens_code` metric: does the span-max vuln signal **survive**
code-masking across the Gemma-3 size ladder + a code anchor, and where (depth)
does it peak? And does **post-training** (pt → it) install/strengthen it (Q5)?

Hypotheses:
- **H1 (collapse risk).** `tokens_code_auc` ≪ `tokens_auc`. If `tokens_code` falls
  to ≈ baseline (~0.5), 02's "probe works" conclusion was scaffold-driven.
- **H2 (depth).** Best-`tokens_code` layer sits at a stable depth *fraction*
  across sizes (lit: mid-late). 02's fraction disagreement (Gemma 0.33 vs Qwen
  0.68) was on the inflated metric and may resolve — or may not.
- **H3 (Q5).** `it` > `pt` on `tokens_code` ⇒ post-training installs the vuln
  direction. Equal ⇒ it's a pretraining feature.

## 2. Inputs

- **Recipe:** unchanged span-max (`src/training/train_probe_spanmax.py`,
  `train_one_layer`, epochs=30), one probe per layer, all layers.
- **Data:** `data/dataset.jsonl` = SVEN **before/after** full-function contrast
  (ADR 0002, already on scratch); split `data/sven_split_meta.json` (seed-42,
  20% group-held-out by `pair_group_key`). Carries `code`, `lang`, `token_labels`.
- **Roster (HF ids; all gated → needs `HF_TOKEN`):**

  | model | axis | note |
  |---|---|---|
  | google/gemma-3-1b-it | size (small) | fast |
  | google/gemma-3-4b-it | size | [mm] text path |
  | google/gemma-3-12b-it | size | [mm] |
  | google/gemma-3-27b-it | size (anchor) | re-extract on before/after |
  | Qwen/Qwen2.5-Coder-32B-Instruct | code-large (anchor) | re-extract |
  | google/gemma-3-1b-pt | post-train base | Q5 pair w/ 1b-it — `TODO(adhoc-decision)` |
  | google/gemma-3-4b-pt | post-train base | Q5 pair w/ 4b-it — `TODO(adhoc-decision)` |
  | google/gemma-3-12b-pt | post-train base | already in models.txt; Q5 pair w/ 12b-it |

  `TODO(adhoc-decision)`: which `-pt` sizes to include for Q5. Proposal: 1b+4b+12b
  (cheap), skip 27b-pt unless the window has slack. User to confirm/trim.
- **Prerequisite code change (gates the whole sweep):** the per-layer scorer must
  emit `tokens_code_auc`. Today `train_all_layers.py` computes a bare AUC over all
  tokens. Factor a shared helper `honest_token_aucs(tok_p, eids, dataset, offsets)`
  that loads `offsets.npz` + per-example `code`/`lang`, builds the mask via
  `src.eval.code_mask.code_only_mask`, and returns `{tokens, tokens_code,
  dropped_fraction}`. Reused by 03/04 re-scoring (sweep 5) and 07. tree-sitter
  (+ python/c/cpp grammars) must be present in the cluster env — verify in smoke;
  if a grammar is missing the mask is a no-op (AUC silently reverts to `tokens`),
  so **assert `dropped_fraction > 0`** on a canary before trusting results.

## 3. Outputs

Per model: `runs/layersweep_<slug>/layers/layer_NN.json` with, per layer:
`{layer, layer_frac, tokens_code_auc, tokens_auc, dropped_fraction,
test_ex_auc (rides along), val_ex_auc, n_test_ex}`. Aggregated into
`metrics_layersweep_<slug>.json` with `best_layer` chosen by **val** then read off
`tokens_code` on test (no test-set layer selection). Baselines (random/length/
regex) computed once on `tokens_code` scope too.

Sweep 5 riders (re-run on these acts, best layers per model):
- `03` loss×α grid → `tokens_code_auc` per cell (does α help the honest metric, or
  only the easy negatives? 02-era finding "high α lifts tok_auc" was inflated).
- `04` richer (MLP256 / layer-concat) → `tokens_code_auc` (does the MLP head's
  0.72→0.79 `tok_auc` lift survive on live code?).

Sweep 6 rider (best layer, per model): `tokens_code_auc` broken down by **language**
(py / c / cpp) and by **CWE**.

## 4. Result format

- **Table** `model | params | pt/it | best_layer | best_layer_frac |
  tokens_code_auc | tokens_auc | gap | dropped_frac | regex_auc`.
- **AUC-vs-depth curves**, two lines per model (`tokens_code` vs `tokens`); the gap
  between them is itself a result.
- **Q5 pairs:** `pt` vs `it` `tokens_code_auc` at matched size + layer.
- **Sweep-5 tables:** 03 (α×layer) and 04 (arch×featureset) on `tokens_code`.
- **Sweep-6:** per-language and per-CWE `tokens_code_auc` at best layer.

## 5. Interpretation hints

- `tokens_code_auc ≈ tokens_auc` ⇒ the signal was real code, not scaffold; 02's
  conclusion holds, just re-baselined.
- **`tokens_code_auc → ~0.5` while `tokens_auc` stays ~0.75 ⇒ H1: the probe was
  reading trivial-negative structure.** This would reframe the whole plan — the
  span-max recipe wouldn't have a usable vuln signal on real code, and 07
  (train-time masking) becomes the priority rescue.
- `tokens_code_auc` peaks at a consistent depth fraction across sizes ⇒ a fixed
  layer rule is safe (settles the §8 layer-policy decision); ragged peaks ⇒ must
  val-select per model.
- `it` ≫ `pt` ⇒ post-training installs the vuln direction (Q5). `pt` already high
  ⇒ pretraining feature, safety-tuning not required.
- Small Gemma (1b/4b) tracks 27b on `tokens_code` ⇒ signal isn't scale-gated,
  cheap to probe. Only 27b works ⇒ scale-dependent.
- Sweep 6: one language/CWE carrying all the signal ⇒ dataset-bias failure mode
  (framing §6) — flag, don't trust the aggregate.

## For agents

- **Supersession:** 06 *is* the 02 layer sweep, made honest + wider. Don't delete
  02; 06 cites it and re-reports its two models under the new metric. The 02
  `metrics_*` JSONs stay as the inflated-metric record for the gap comparison.
- **Extraction is the long pole and everything re-extracts** (`runs/` + HF cache
  were wiped). Order: gated `HF_TOKEN` placed → extract per model (all layers,
  float32 acts — Gemma mid-layer massive-activations overflow f16) → per-layer
  span-max → honest scoring → aggregate. Reuse `02-.../extract_all_layers.py` and
  `submit_layersweep.sh`; the 4-GPU layer-sharding in `train_all_layers.py`
  (`li % n_gpus == gpu_id`) stays.
- **debug-qos = one submitted job.** Sequential login-node `nohup` orchestrator
  (pattern in scratch `run_*_orch.sh`); pack 4 models/node where they fit one
  96 GB GH200 (the small Gemmas easily co-reside; 27b/32b run solo or 2-up).
- **Cluster channel:** the `clariden` tmux session (authenticated SSH to
  clariden-ln001) + `/tmp/ctmux.sh '<cmd>' [timeout]`. Keep it alive — it's the
  cert workaround; if it drops, `ela.cscs.ch` re-sign needed.
- **Smoke before sweep:** one small model (gemma-3-1b-it), assert
  `dropped_fraction > 0` and `tokens_code` n_pos / n_total are sane before
  committing the full roster.
- **Provenance:** log `git rev-parse HEAD` per run (clariden repo at `3209974`+);
  refuse on dirty tree.

---

## Results (2026-06-01)

Run on Clariden, 1 node × 4 GH200 per model, sequential (debug-qos). Acts float32,
all layers; probe = span-max linear; split = SVEN before/after seed-42 group hold-out.

**Headline: `tokens_code` does NOT collapse.** On all 8 models the honest
`tokens_code_auc` ≈ the inflated `tokens_auc` (consistently *slightly higher*).
The mask drops only ~30% of tokens (token-level; ~15% char-level) on full-function
SVEN — most "easy negatives" are live-code-not-vuln, not comments — so masking
barely moves AUC. The signal is genuinely on live code.

**Layer selection (sweep over `val_tokens_code`, not `val_ex_auc`).** val_ex_auc is
near-chance here and selected near-random layers (undershot oracle by up to 0.065).
Switched to **val_tokens_code** on a leakage-free 15% group-aware val split → now
**within ≤0.025 of oracle on every model** (mostly <0.015). Resolves the layer-policy
open item: val_tokens_code selection is reliable; no need to peek at test.

| model | sel layer (frac) | test tokens_code | oracle |
|---|---|---|---|
| gemma-3-1b-it | L25 (1.00) | 0.769 | 0.772 |
| gemma-3-1b-pt | L12 (.48) | 0.750 | 0.775 |
| gemma-3-4b-it | L7 (.21) | 0.767 | 0.779 |
| gemma-3-4b-pt | L33 (1.00) | 0.769 | 0.784 |
| gemma-3-12b-it | L15 (.32) | 0.771 | 0.779 |
| gemma-3-12b-pt | L13 (.28) | 0.767 | 0.778 |
| gemma-3-27b-it | L19 (.31) | 0.770 | 0.770 |
| Qwen2.5-Coder-32B | L25 (.40) | 0.788 | 0.800 |

**Stable across scale (1B–32B) and post-training.** tokens_code ≈ 0.75–0.79
everywhere; best layer clusters mid-network (~0.2–0.4; two models' val peaked at the
last layer, test still fine).

**Q5 — post-training does NOT install the vuln direction.** pt ≈ it at matched size
(12b: 0.771 vs 0.767; 4b: 0.767 vs 0.769; 1b: 0.769 vs 0.750). It's a *pretraining*
feature, not installed by instruct/RLHF.

**Sweep-5 — 03/04 on the honest metric (anchors).** CORRECTION (2026-06-01, after
review): an earlier draft called these "inflated-metric artifacts (+0.004)" — that
was a **bad comparison** (exp-04 MLP vs exp-06 *linear*, different recipe). The
within-experiment truth: `tokens_code` ≈ `tokens` in every cell (metric swap barely
matters), BUT the MLP and α gains are **genuine on tokens_code**:
- exp-04 (richer): linear→MLP is a real jump — Qwen single-layer (fset `25`)
  0.756→**0.791** (+0.035); 3-layer concat (`23,25,27`) 0.721→**0.785** (+0.064).
  gemma-27b exp-04 OOM'd (3-layer concat × mlp512 exceeds node RAM at 4 workers) —
  re-run at NWORKERS=2 to confirm; old inflated run showed the same pattern.
- exp-03 (loss×α): α genuinely helps, optimum ~α=10 — gemma-27b 0.703(α1)→**0.748**
  (α10) = +0.045; Qwen 0.756→**0.770** = +0.014. `neg_incl` ≈ `base`.

**Takeaway:** `tokens_code` doesn't collapse and the signal is real/robust across
family, scale, and post-training — BUT the linear span-max probe is **not at
ceiling**: MLP and α≈10 each buy ~0.04–0.06 over a plain α=1 linear probe. Recovering
that MLP gain *interpretably* is the point of exp-09.

### Sweep 6 — per-language / per-CWE breakdown (best layer, 8 models)

**The ~0.78 aggregate is Python-dominated and hides a sharp split.**

By language (`tokens_code`, mean over 8 models): **python ≈ 0.81**, **c ≈ 0.59
(near-chance)**, cpp ≈ 0.50–0.75 (noisy, only 100 rows). The probe essentially
works on Python and barely on C. The aggregate is carried by the 756 Python rows
(vs 574 C). This is the dataset/per-language failure mode in framing §6 — the
headline "robust ~0.78" must be qualified.

By CWE (`tokens_code`, mean over models, positives-of-CWE vs all negatives):

| CWE | type | tokens_code |
|---|---|---|
| CWE-089 | SQL injection | **0.924** |
| CWE-078 | command injection | 0.824 |
| CWE-022 | path traversal | 0.779 |
| CWE-079 | XSS | 0.641 |
| CWE-125 | out-of-bounds read | 0.559 |
| CWE-476 | NULL deref | 0.549 |
| CWE-416 | use-after-free | **0.521** |

**Clean split: the probe detects injection-class vulns (SQLi/cmd-inj/path-traversal
— string/dataflow patterns, mostly Python web code) and is at/near chance on
memory-safety vulns (OOB-read / NULL-deref / UAF — C/C++ pointer bugs).** Language
and CWE tell the same story: what reads as a "model's vulnerability belief" is
really a belief about *injection-style* vulnerabilities, not memory safety.

**Implication for the framing:** the base property (§1) is detected unevenly —
strong where the vuln is a visible data-flow/taint pattern, weak where it's a
temporal/pointer property with no local syntactic tell. Any AI-control monitor
built on this probe would have a large blind spot on memory-safety bugs. Worth a
follow-up: is C weak because the *signal* is absent in C activations, or because
SVEN's C contrast pairs are harder/labelled differently?
