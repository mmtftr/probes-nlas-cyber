[ai-generated]

# 02 — Fine-grained layer sweep, Gemma-3-27B

Step 2 of `../PLAN.md`. The overnight cross-model run only captured 4 layers per
model (`{n/4, n/2, 3n/4, n−1}`). This sweeps **all 62 layers** of one model to get
the full AUC-vs-depth curve.

1. **Aim** — Is there a single depth fraction near-optimal for vuln-probe
   separability, or must we val-select per model? Hypothesis: token/example AUC
   rises into mid-late layers, plateaus before the last.
2. **Inputs** — `google/gemma-3-27b-it` (62 layers, hidden 5376); `data/dataset.jsonl`
   (1560 rows); the shared SVEN group split (`sven_split_meta.json`, seed=42, 20%
   held out); transformers 5.9.0 stack. Probe hyperparams identical to the overnight
   run: span-max loss (α=10, ω linear 0→1), AdamW lr=1e-3, **30 epochs**, hard labels.
3. **Outputs** — on scratch under `runs/layersweep_gemma3-27b/`: 62 float16 layer
   memmaps (`acts/layer_NN.npy`, ~275 GB total), per-layer `layers/layer_NN.json`,
   aggregated `metrics_layersweep.json`. Activations are NOT committed (path logged).
4. **Result format** — ex-AUC + tok-AUC for all 62 layers, the 3 baselines, best
   layer + fraction. Plot AUC-vs-depth locally from the JSON.
5. **Interpretation** — flat-topped mid-late plateau ⇒ a fixed fraction is safe
   (keeps the `{n/4..n−1}` shortcut); sharp narrow peak ⇒ must val-select per model.
   Resolves the layer-policy `TODO(adhoc-decision)` (research-framing §6 / Q3).

**Built-in sanity check:** layers 15/31/46/61 here must match the overnight 27B
`metrics.json` (`runs/google_gemma-3-27b-it/metrics.json`). A mismatch means this
pipeline diverged from the canonical one.

## For agents

- Files: `extract_all_layers.py` (stream all layers → memmaps, idempotent via
  `DONE_EXTRACT`), `train_all_layers.py` (per-layer span-max probe, resumable via
  `layer_NN.json`, GPU-shardable with `--n-gpus/--gpu-id`), `aggregate_layersweep.py`
  (combine + baselines), `submit_layersweep.sh` (one node, 4 GPUs: extract → 4-GPU
  train fan-out → aggregate).
- All three scripts reuse the canonical functions (`_load_model`, `token_data`,
  `train_one_layer`, `train_eval.load_or_make_split/example_scores`,
  `baselines.all_baselines`) so the logic is identical to the overnight sweep.
- Run on the login node: `cd ~/scratch/probes && bash repo/plans/.../submit_layersweep.sh`.
  Re-submit to resume after a 90-min timeout (extract + finished layers are skipped).
- Memmaps are ~551 GB (float32) on scratch; delete `runs/layersweep_gemma3-27b/acts/`
  when the curve is collected (per-layer JSONs + metrics_layersweep.json are the keepers).

## Results (2026-05-31, job 2438307)

All 62 layers swept. Artifacts: `metrics_layersweep.json`, `auc_vs_depth.png`,
`overnight_gemma3-27b_4layer.json` (the coarse run, for comparison).

**Pipeline validated:** layers 15/31/46/61 reproduce the overnight 4-layer
`metrics.json` *exactly* (ex/tok: 0.657/0.815, 0.617/0.813, 0.564/0.747,
0.677/0.838) → this sweep is bit-identical to the canonical pipeline.

**Finding — the coarse grid missed the peak.**
- True best **example-AUC = layer 27 (frac 0.44), 0.695**; near-tie layer 22.
  token-AUC peaks at **layer 26, 0.855** — a clean mid-depth peak (~0.42).
- The coarse `{15,31,46,61}` picks scored 0.657 / 0.617 / **0.564** / 0.677.
  Its 3n/4 point (L46) lands in a dead zone barely above the length baseline
  (0.58); by ex-AUC it would pick the last layer (L61, 0.677), missing the real
  mid peak by +0.02 AUC and ~19 layers.
- Shape: token-AUC rises smoothly to ~L26 then declines steadily into late
  layers; example-AUC is noisier (~0.66 early-mid, sharp drop after L32 toward
  the length baseline, then a final-layer uptick at L61). **Not** a flat plateau.

**Interpretation (Q3 / layer-policy `TODO(adhoc-decision)`):** for Gemma-3-27B
the vuln signal is strongest at **mid-depth (~0.4)**, not late, and the coarse
4-point grid is actively suboptimal here (its 3n/4 pick is near-useless). This
argues for per-model val-selection or a mid-depth fraction rather than a fixed
late layer — pending a second model to confirm the fraction generalizes. Does
not settle the ADR alone (n=1 model).
