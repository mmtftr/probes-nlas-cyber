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
