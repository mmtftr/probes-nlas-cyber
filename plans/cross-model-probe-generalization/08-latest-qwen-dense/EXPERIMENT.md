[ai-generated]

# 08 — Latest-Qwen dense models on the honest sweep

Extend exp-06 (honest `tokens_code` layer sweep + val_tokens_code selection +
per-lang/CWE breakdown) to the newest **dense** Qwen models. Same pipeline, two new
models.

## 1. Aim

Does a newer-generation **dense** Qwen detect input-stream vulnerability belief
better on the honest `tokens_code` metric than Qwen2.5-Coder-32B (our best so far,
0.788) — and critically, does it **close the C / memory-safety gap** (06 sweep-6:
C ≈0.59, UAF/NULL-deref/OOB-read ≈0.52–0.56)?

## 2. Inputs

- **Models (dense only — user excluded the MoE coder to avoid an architecture
  confound):**
  - `Qwen/Qwen3.6-27B` — newest dense (2026 arch). `[tx]` needs transformers 5.9.0
    (have it). **Caveat:** Gated-DeltaNet / linear-attention layers — verify
    `output_hidden_states` exposes every layer before trusting (ADR 0001 says it
    worked on the inflated run; re-confirm `meta.json` n_layers is sane).
  - `Qwen/Qwen3-32B` — Qwen3 dense general, size-matched to Qwen2.5-Coder-32B.
- **Data/recipe:** unchanged — SVEN before/after `data/dataset.jsonl`,
  `sven_split_meta.json`, span-max, all layers, val_tokens_code selection.
- **Extraction REQUIRED** (new models, not cached) — ~50–65 GB download each + all
  layers float32. The long pole.

## 3. Outputs

`runs/layersweep_<slug>/metrics_layersweep.json` (+ per-layer JSONs) and
`runs/breakdown_<slug>.json`, same schema as the 8 done.

## 4. Result format

Drop the two new rows into the 06 tables: `model | best layer (frac) | tokens_code |
oracle | gap`, plus per-language and per-CWE breakdown. Compare head-to-head with
Qwen2.5-Coder-32B.

## 5. Interpretation hints

- New dense Qwen `tokens_code` > 0.788 overall ⇒ newer-gen helps the base property.
- **The real test: per-language C and per-CWE memory-safety.** If C jumps from ~0.59
  toward python's ~0.81, the newer model represents memory-safety vulns where the
  older didn't — a substantive capability finding. If C stays ~0.59, the
  memory-safety blind spot is recipe/dataset-level, not model-generation-level.
- Qwen3-32B (general) ≈ Qwen3.6-27B (newer) ⇒ generation matters more than the
  extra Qwen3.6 arch tricks; large gap ⇒ the arch/training of 3.6 matters.

## For agents

- Reuse `06-honest-metric-sweeps/run.sh` (MODEL env) per model; run both
  sequentially (swap the MODELS list), one run at a time.
- After both finish, add their slugs to the breakdown step's MODELS and run it.
- **Smoke `Qwen/Qwen3-32B` first** (cleaner arch) — assert extraction succeeds +
  `dropped_fraction > 0` + sane n_layers — before trusting Qwen3.6-27B's
  linear-attention layers.
- Pull metrics locally, regenerate the 06 plots (add the 2 models), `cursor -r`.
