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
- Memmaps are ~275 GB on scratch; delete `runs/layersweep_gemma3-27b/acts/` when the
  curve is collected (the per-layer JSONs + metrics_layersweep.json are the keepers).
