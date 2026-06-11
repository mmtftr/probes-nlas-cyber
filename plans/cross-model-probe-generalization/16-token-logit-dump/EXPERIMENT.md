[ai-generated]

# 16 — Token + example logit dump (materialise every probe prediction)

Closes the gap in memory `persist-token-level-predictions`: prior runs computed
token-level AUCs (~0.76) but never saved the per-token logits, and the probes
themselves were trained on the deleted 4.5 TB of acts. This regenerates both.

1. **Aim** — Materialise EVERY per-token and per-example probe logit over the
   full SVEN dataset (1430 ex), for the canonical span-max vulnerability probe,
   so they can be loaded/compared locally. Not a new hypothesis — a reproduction
   + persistence pass. Correctness gate: reproduce the historical token-code-AUC.

2. **Inputs**
   - *Models* (ungated, tonight): `Qwen/Qwen2.5-Coder-32B-Instruct` (anchor,
     hist L25 tok_code_auc 0.776), `Qwen/Qwen2.5-Coder-7B-Instruct` (no anchor).
     *Gated, on HF-token:* gemma-3 1b/4b/12b/27b-it + 12b-pt (hist L25/7/15/19/13).
   - *Dataset* — `./data/dataset.jsonl` (SVEN before/after, rebuilt 2026-06-01).
   - *Split* — seed-42 20% group-clean hold-out `sven_split_meta.json` (verbatim
     `load_or_make_split`; identical to all prior experiments).
   - *Probe* — span-max linear (`train_one_layer`, epochs=30), trained on TRAIN
     tokens at each layer in the model's band; band brackets the historical best.

3. **Outputs** — `./runs/logitdump_<slug>/`, collected into
   `plans/.../16-token-logit-dump/results/<slug>/`:
   - `logits_layer{NN}.npz` — flat token table: logit, prob, y, example_id,
     char_start, char_end, is_test, is_code (one row per token, all examples).
   - `example_scores_layer{NN}.json` — per-example max-pool score + label + cwe.
   - `probe_layer{NN}.npz` — w, b, layer (the probe, finally persisted).
   - `metrics_logitdump.json` — per-layer + best-layer test AUCs.

4. **Result format** — per model: best layer; held-out `tokens_auc`,
   `tokens_code_auc`, `example_auc`; token count (total / code). Plus the AUC-gate
   check: Qwen2.5-Coder-32B `tokens_code_auc` vs the historical 0.776.

5. **Interpretation hints**
   - 32B `tokens_code_auc` ≈ 0.776 (±~0.01) ⇒ pipeline faithfully reproduced;
     the dumped logits are trustworthy for local comparison.
   - Materially below ⇒ a reproduction drift (layer-index convention, trainer
     seed, transformers version, or split) — investigate before trusting logits.
   - 7B near or above 32B ⇒ scale buys little here (consistent with the small
     cross-size spread already seen); well below ⇒ capacity matters for this probe.

## For agents
- `hidden_states[L+1]` == output of block L (extractor line ~244); the band
  layer index `li` is this same L, matching `train_eval` and the breakdowns.
- Run: `run.sh "Qwen/Qwen2.5-Coder-32B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct"`.
- `run_logitdump.sh` deletes the heavy `token_activations_layer*.npz` after the
  dump unless `KEEP_ACTS=1`; the logits npz + probe npz are the kept artifacts.
- Gemma needs a Hugging Face token. Without it, gemma
  downloads 401 and run_logitdump writes FAILED — Qwen is unaffected.
