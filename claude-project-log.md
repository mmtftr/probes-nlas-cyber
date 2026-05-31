[ai-generated]

# Claude project log

Append-only. `[human]:` prefix for hand edits.

---

## 2026-05-26 — bootstrap

Carved fresh repo from gemmaforge's last-accepted lineage.

Commits: `1536b2e` carry-over · `4317bdb` workflow scaffold + src/ refactor.

- **Carried over:** spanmax + SVEN training, `src/eval/`, 3 notebooks,
  relevant tests.
- **Dropped:** LoRA, MLP / attention / ensemble / per-CWE / value-head
  variants, adversarial suite, demo / Space / pwnkit. `publish_to_hub.py`
  deleted.
- **Renames:** `gemmaforge.repo_benchmark/v1` → `probes.repo_benchmark/v1`;
  `gemmaforge_top_cwe` → `probe_top_cwe`. Docstring `GemmaForge` stripped
  from `.py`/`.md`/`.toml`. Notebooks left as-is (rewrite TODO).
- **Layout:** `CLAUDE.md` (+ `AGENTS.md` symlink), `plans/`, `decisions/`,
  `docs/{guides,papers}/`. `src/` → `{data,probes,training}/`.
  `data/{datasets,models,probes,plots}/` each with one-line README.
- **TODOs:** Gemma 4→3 in extractors + trainers · notebook rewrite for
  wandb · artifact I/O → `wandb.Artifact` backed by HF · eval JSON cards
  → wandb tables · scan abstraction reintroduce on demand · NLAs scope TBD.

---

## 2026-05-30 — research framing + first papers

Narrowed project scope with the user.

- **Added `docs/research-framing.md`** (living charter): target property =
  model's belief about how vulnerable *input-stream* code is; 3 follow-on
  complications (gen-vs-read direction, own-vs-others OOD bias, intent-to-
  exploit); 5 open framing questions; baselines status; consolidated
  `TODO(adhoc-decision)` list (§6).
- **Decision:** scope first experiments to the **base property**; treat the 3
  complications as follow-on.
- **Papers pulled:** `ribeiro2025-internal-rep-code-correctness.md`
  (arXiv:2512.07404, RepE/LAT contrast-pair, *reads* code) and
  `bui2025-openia-correctness.md` (arXiv:2501.12934, supervised probe on the
  model's *own generations*). Both: signal mid-late layers, final code token,
  model-specific reps. Added `docs/literature-review-1.md` scaffold (user fills
  What-we-learn / What-we-can-adopt).
- **CLAUDE.md:** added *Collaboration model* (agent = collaborator; don't alter
  user ideas; surface ambiguities; `TODO(adhoc-decision)` convention).
- **Baselines:** confirmed random/length/regex + sample-level probe +
  `fit_logreg_on_split` already carried over in `src/eval/`. LAT and
  CodeBERT/CodeT5+ flagged as candidate additions (not implemented).

---

## 2026-05-30 — failure modes & mitigations recorded

- **`docs/research-framing.md`:** added §6 *Failure modes & risks* and §7
  *Mitigation ideas* (user's framing, recorded faithfully); renumbered the
  consolidated decisions list §6 → §8. No prior cross-refs pointed at §6.
- **Failure modes (user):** probe inner misalignment (dataset bias /
  low-quality-not-vulnerable / short→long non-generalization); usefulness
  (over- vs. under-detection); architecture sufficiency vs. scope creep;
  dataset label noise + context difficulty; tunnel vision (white-box ≥ probes);
  provenance non-generalization (in-the-wild vuln probe may not fire on
  assistant-generated code).
- **Mitigations (user):** function→full-file context-length test (cites
  hallucination probes generalizing long→short but not short→long); LLM-rewrite
  secure→low-quality to test quality overfit; off-distribution / other-repo
  eval for dataset-bias overfit; ask-the-LLM-directly + other baselines to avoid
  tunnel vision; classifier-property audit for over/under-detection.
- Added fenced **Agent notes** cross-linking each to existing baselines/splits/
  metrics and the open §8 decisions; no new decisions forced.

---

## 2026-05-31 — Clariden cross-model sweep launched (overnight, GH200)

8 h GH200 window on CSCS Clariden (account `lsaie-ss26`, partition `debug`,
4× GH200/node, 90 node-min/job). Method-generalization sweep per
`plans/cross-model-probe-generalization/`.

- **Orchestrator** `src/remotes/clariden/` (env.sh / run_model.sh / train_eval.py /
  submit.sh / models.txt / overnight.sh). Single-node jobs, 4 models/GPU packed,
  idempotent (DONE markers), deps installed once into shared `.python_deps`
  (transformers 4.57.1 + matched hub/tokenizers; login node is py3.6 so install
  must be in-container).
- **Roster** 23 models (dense ≤33B + small MoE), all confirmed to fit one GH200;
  HF token validated, all gated (Gemma-3 family + Llama-3.1) reachable.
- **Smoke (validated end-to-end):** deepseek-coder-6.7b ex-AUC 0.705 / tok 0.851
  (best layer frac 0.97); Qwen2.5-Coder-7B 0.694 / 0.820 (0.96); Qwen3-8B 0.711 /
  0.845 (**frac 0.50, mid-net**). Probe beats regex (0.53) / length (0.58).
  Early signal: best-layer fraction varies by family → motivates the layer sweep.
- **Teammate fixes folded in:** submit.sh `pmi2`; extractor `dtype`/`torch_dtype`
  fallback; `scripts/build_dataset_sven_canonical.py` (1560 rows, 780/780);
  slug `printf` fix.
- **Unattended:** `overnight.sh` (nohup on login node) waits out the smoke job,
  submits the full roster, and re-submits gaps (≤2 attempts/model). gemma-4 /
  qwen3.6 (2026 archs) may exceed transformers 4.57 → expected MISS, capped.
- Results: `~/scratch/probes/runs/<slug>/metrics.json`; progress in
  `~/scratch/probes/overnight.log`. NOT yet logged to wandb (TODO).
- **QOS gotcha (fixed):** `debug-qos` rejects a 2nd queued job
  (`QOSMaxSubmitJobPerUserLimit`; nominal MaxSubmitPU=2 but races → effective 1).
  `submit.sh` now drip-feeds at `MAXQ=1` (submit only when 0 of our jobs queued)
  → strictly sequential, ~30 min/chunk for big models. Verified: 0 QOS errors
  across 4 consecutive submissions; 18/23 done within ~1 h of the fix.
- Slug fix changed run-dir names → two early-smoke models have stale `_`-suffixed
  dirs (`*_/`) alongside clean ones; harmless, dedupe at collection.

## 2026-05-31 — cross-model sweep: recover 2/5 failures, drop 3, bump to transformers 5.9.0

Diagnosed the 5 overnight `EXTRACT_FAILED` models and drove the sweep to its final
state (20/23). See `decisions/0001-cross-model-roster-drops.md`.

- **Stack bump**: `env.sh` now installs transformers 5.9.0 / tokenizers 0.22.2 /
  hub 1.17.0 into a fresh `.python_deps5` (4.57.1 tree at `.python_deps` kept as
  rollback; `$TF_STACK` overrides). Verified `gemma4`/`qwen3_5`/`mistral3` ∈
  `CONFIG_MAPPING` before pinning, and re-loaded a DONE model (gemma-3-1b) through
  the new loaders end-to-end.
- **Xet fix**: hub 1.17's Xet path calls `download_files(request_headers=)` which
  the image's `hf_xet` rejects → every fresh download died. Set
  `HF_HUB_DISABLE_XET=1` (classic HTTPS). This was the single root cause of all 5
  failures *re-failing* on first 5.9.0 attempt.
- **Extractor** (`src/data/extract_token_activations.py`): `trust_remote_code=True`
  + `AutoProcessor`-tokenizer fallback (now re-raises the original error instead of
  masking) + `AutoModelForImageTextToText` fallback for VLM wrappers (engaged for
  Mistral-Small-3.2).
- **Recovered**: gemma-4-26B-A4B (ex-AUC 0.702 / tok-AUC 0.840, best layer 15/30)
  and Qwen3.6-27B (0.701 / 0.855, best layer 32/64) — both beat baselines on the
  shared split, the two archs that strictly needed >=5.9.0.
- **Dropped (3)**: OpenCoder-8B (slow `INFLMTokenizer`, no offset_mapping);
  Devstral-2507 + Mistral-Small-3.2 (Tekken→151000-vocab vs 131072 embeddings →
  OOB CUDA assert; no clean offset-preserving fix on 5.9.0). User-confirmed both
  drop decisions rather than ship approximate offsets.

---

## 2026-05-31 — training-logic walkthrough notebook

- Added `notebooks/walkthrough/walkthrough_training_logic.{py,ipynb}` — a
  stage-by-stage visual trace of the span-max probe training. Authored as a
  jupytext percent `.py` (reviewable source) + executed `.ipynb` (8 embedded
  plots). `[ai-generated]`.
- **Faithfulness by import, not re-implementation.** Every stage imports the
  real functions: `token_data.{parse_spans,char_spans_to_token_spans,
  token_labels_array}`, `train_probe_spanmax.{LinearProbe,soft_labels_triangular,
  span_max_loss,train_one_layer,_group_by_example,_example_label}`, and the
  `pair_group_key` group split. Two in-notebook assertions enforce it:
  reconstructed per-token labels == on-disk `y`; manual loss decomposition ==
  `span_max_loss` at omega ∈ {0,.25,.5,.75,1}.
- **Stages:** (1) tokens split & labeled, (2) outer group-clean + inner 90/10
  split, (3) probe logits `w·h+b`→sigmoid, (4) loss anatomy (weighted-BCE token
  term + max-pool span term + omega anneal + soft labels), (5) real
  `train_one_layer` loop with loss/AUC/omega curves + before/after, (6) max-pool
  example score + token/example ROC.
- **Data:** uses the staged `notebooks/walkthrough/data/` sample (Gemma-3-1B
  layer-13 acts, 150 examples, + dataset.jsonl/offsets/spans/split_meta).
  Notebook flags AUCs as sample-scale, not the production figure.
- Built with `uvx jupytext` + `jupyter nbconvert --execute` (matplotlib/
  nbconvert added at runtime via `uv run --with`).

---

## 2026-05-31 — Step 2: full per-layer sweep, Gemma-3-27B

- New self-contained experiment `plans/cross-model-probe-generalization/02-layer-sweep-gemma3-27b/`
  (extract_all_layers / train_all_layers / aggregate_layersweep / submit_layersweep.sh).
  Streams all 62 layers to per-layer memmaps, trains the span-max probe per layer
  across 4 GPUs (resumable), aggregates AUC-vs-depth. Reuses the canonical
  loaders/loss/split so logic matches the overnight run.
- First attempt crashed: float16 saturates Gemma-3's massive mid-layer activations
  (>65504) to inf -> NaN training. Fixed to float32 (551GB scratch) + per-layer
  try/except. Job 2438307: 62/62 layers, ~16 min.
- **Validated:** layers 15/31/46/61 reproduce the overnight 4-layer metrics exactly.
- **Finding:** best ex-AUC = layer 27 (frac 0.44, 0.695); token-AUC peaks L26 (0.855).
  The coarse {n/4,n/2,3n/4,n-1} grid missed it — its 3n/4 pick (L46=0.564) is a dead
  zone near baseline. Mid-depth (~0.4) > late for this model. Bears on Q3/layer-policy
  ADR (n=1, needs a 2nd model). See EXPERIMENT.md Results + auc_vs_depth.png.
- Activations left on scratch at runs/layersweep_gemma3-27b/acts (551GB) — delete when
  no longer needed for re-analysis.

- **Walkthrough follow-up:** added a Stage-5 cell showing the trained probe on
  the SVEN-paired *correct* twin of the demo example (same `pair_group_key`,
  label 0). Found generically, not hardcoded. Contrast on layer-13 sample:
  vulnerable eid 38 max p=1.00 (fires in span) vs. fixed eid 435 max p=0.15
  (quiet). Notebook now has 9 embedded plots; re-executed clean.

- **Walkthrough Stage 7 (bad examples):** added a failure-case stage — ranks all
  sample examples by max-pooled score, plots the 2 weakest positives + 2
  strongest negatives, prints their source text. Surfaced a real finding on the
  layer-13 sample: at THR=0.5 the max-pool example score gives 0/75 false
  negatives but 71/75 false positives — concrete instance of the over-detection
  failure mode (research-framing §6) and the operating-point question (Q4).
  Notebook now 10 plots, re-executed clean. (Trained-on-sample caveat noted in
  the cell.)

- **Walkthrough Stage 8 (calibrated operating point):** traced gemmaforge's
  calibration threshold to `data/probe_spanmax_f1.json` in the original repo —
  0.929082, an F1-max threshold on Platt-calibrated (T≈1.794, a≈-0.269) probe
  scores, derived by sweeping 0.01–0.99 on the heldout repo benchmark
  (precision 0.49 / recall 0.35). Added a stage that reproduces the *procedure*
  on the layer-13 sample probe via the repo's own `apply_platt` + a logistic
  Platt fit + F1-sweep. Result (sample, trained-on): T=7.03/a=12.79,
  F1-max=0.290, P=0.57/R=0.93; FP drops 71→52 vs raw 0.5. Notebook stresses the
  threshold is not comparable across probes (different calibrated axes) — only
  the procedure transfers. Now 11 plots, re-executed clean.

---

## 2026-05-31 — paper import: Yu et al. MoC (arXiv:2507.09508)

- Added `docs/papers/yu2025-moc-secure-code.md`. *A Mixture of Linear
  Corrections Generates Secure Code* (Yu, Mangal, Zhuo, Fredrikson, Păsăreanu).
- **Closest paper to our actual setup:** linear vulnerability probe on last-token
  hidden states, trained on **SVEN** (same dataset as `build_dataset_sven.py`),
  with **line-change token spans** as supervision — parallels our span-max
  token labels.
- Headline we care about: probe ≫ prompting (QC-7B 79% vs 49%) — direct support
  for research-framing §7 mitigation #4 (ask-the-LLM-directly baseline). Also:
  their "bug-prone ≈ vulnerable" overlap corroborates failure mode §6
  (probe may track low code quality, not pure vulnerability). Late-layer best;
  Python CWEs easier than C; model-specific transfer.
- Two `TODO(adhoc-decision)` noted in the file: per-CWE probe/layer selection
  vs pooling (Q3), and whether MoC-style *steering* (vs detection) is in scope.

---

## 2026-05-31 — dataset rebuild: SVEN before/after contrast (builder + cap decision)

- New builder `scripts/build_dataset_before_after.py` ([ai-generated]) replaces
  the completion-truncation builders. positive=full `func_src_before`,
  negative=full `func_src_after`. Token labels (positives) = diff'd vulnerable
  lines in `before`, three-tier: (1) `char_changes.deleted` spans, text-verified
  vs `before[cs:ce]` (drops ~25 stale-offset spans), expanded to whole lines;
  (2) fallback `line_changes.deleted`; (3) purely-additive fixes (91 pairs) →
  the `before` line at the common-prefix/suffix divergence point. Every positive
  gets ≥1 span by construction.
- **Why the non-empty guarantee matters:** both trainers derive the example
  label from token spans (`ex_y = y[eids==e].max()>0`), NOT the `label` field.
  An empty-span or token-truncated positive silently flips to negative.
- **Length-cap decision (user):** full functions reach 114k chars (~28k tok);
  extractor truncates at `max_length`, dropping past-cap vulnerable lines. Chose
  **6000-char cap ↔ extractor `max_length` 2048** (was 1024). Drops whole pairs
  over the cap so both models share an identical, un-truncated row set. Bumped
  1024→2048 in submit_layersweep/variance/verbalized, run_model.sh,
  extract_all_layers.py, extract_token_activations.py, verbalized_judge.py.
- **Additive-fix labeling (user):** insertion-point line (vs CWE-regex / whole-func).
- Built local `data/dataset.jsonl`: 715 pairs / 1430 rows, balanced. Dropped 84
  pairs (len>6000) + 4 (before==after). Fresh `sven_split_meta.json` (seed=42,
  20%): 704 groups, 141 held out (old 767-group split retired).
- Validation green: schema 0 failures; SVEN-after==#pos (715); 0 empty-span
  positives; 0 spanned negatives; verbatim vs SVEN. **Length confound gone:**
  pos-longer 27% / neg-longer 70% / equal 3% (was pos-longer 100%).
- **Blocked on extraction:** CSCS SSH cert expired (`Permission denied
  (publickey)` from ela.cscs.ch). Needs user to re-sign before deploy/extract.

### caveat (extraction): residual gemma truncation
- gemma-3-27b: 9/1430 examples hit the 2048-token cap (longest C funcs tokenize
  denser than Qwen, on which the 6000-char cap was calibrated). 1 positive
  (eid 737 `avcodec_align_dimensions2`, span at ~75% of a 5218-char func) loses
  its only vulnerable-line tokens → silently treated as negative. 714/715 gemma
  positives retain labels. Immaterial to AUC (<=1 example, possibly in train).
  Qwen: cap calibrated on it → expect 0 truncations. Documented, not re-extracted.

## 2026-05-31 — before/after rebuild: exp 02–05 results (4-node accelerated)

All four re-run on the SVEN before/after dataset, both models. Ran 4-node debug
jobs (each model's grid split across 2 nodes / 8 GPUs) inside the 1.5 node-hour
cap; launchers in `plans/cross-model-probe-generalization/orchestration/`.

**Headline: the old example-AUCs were substantially confound-inflated.** Length
baseline 0.58 → 0.49 (≈chance); best example-AUC drops ~0.08–0.12 across exps.
Token-level vulnerable-line signal persists (~0.69–0.77). New best layers:
gemma L20, qwen L43 (old were 27/52) — used downstream.

- **02 sweep:** gemma L20 ex0.573/tok0.756; qwen L43 ex0.614/tok0.766 (single-seed).
- **03 loss×α (5-seed, robust):** gemma L20 base α1 0.644±0.005; qwen L49 neg_incl α1
  0.642±0.032. **α>1 no longer helps**; neg_incl ≈ base.
- **04 richer:** gemma best = linear single-L20 (0.644); qwen best = linear concat
  (0.662). **Reverses old finding: MLP hurts both** (old gain was the confound);
  **concat helps only Qwen** (was only Gemma).
- **05 probe vs verbalized:** gemma probe 0.644 > verbalized 0.554 (Δ+0.091, gap
  *widens*); qwen probe 0.601 ≈ verbalized 0.632 (Δ−0.031, within seed noise) —
  **probe's edge over self-report is now model-dependent** (held for both before).

### Infra notes (hard-won)
- 4-node concurrency: each model's grid sharded across 2 nodes via logical
  `--n-gpus 8 --gpu-id` (decoupled from CUDA device). Validated with a 2-node
  canary (concurrent, 4 GPUs each).
- **exp-04 OOM root cause:** `numactl --membind=$gpu` pins each worker to its GPU's
  NUMA node (~115 GB of 460 GB); the multi-layer concat (~59 GB, transiently 2x)
  blows that. Fix = `--interleave=all` (full-node RAM) + 2 workers/node. NOT a
  GPU-memory issue. Fixed in submit_richer.sh too.
- Stale-cell trap: lossalpha/richer/verbalized run dirs held OLD-dataset cells
  (old layers); aggregators glob `cell_*.json` → must wipe before re-running.
- exp-05 `length_baseline` was a hardcoded 0.575 (old set); corrected to 0.49.
- All metrics snapshotted locally at /tmp/probes_snapshot (acts NOT pulled).
