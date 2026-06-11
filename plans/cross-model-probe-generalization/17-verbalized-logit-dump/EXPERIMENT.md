[ai-generated]

# 17 — Verbalized logit dump (persist the model's own yes/no judgment logits)

Companion to exp-16. Exp-16 materialised every PROBE logit; exp-05 computed the
VERBALIZED judgment (the model's own P(yes, vulnerable?)) but persisted only the
scalar `p_yes` per example — the raw logits were thrown away. This closes that
gap for the verbalized side, so probe and verbalized predictions are both fully
materialised for local comparison. Reproduction + persistence pass, not a new
hypothesis.

1. **Aim** — Re-run the model's OWN verbalized vulnerability judgment over the
   full SVEN dataset (1430 ex) and persist EVERY logit needed to recompute the
   score (yes/no first-token logits, logsumexp margin, p_yes, top-10 first-token
   logits), per example. Score path is bit-identical to exp-05 (same prompt,
   same yes/no read). Correctness gate: reproduce exp-05's verbalized example-AUC.

2. **Inputs**
   - *Models* — the 6 INSTRUCT models from exp-16 (verbalized needs a chat model;
     gemma-3-12b-**pt** is excluded — base model, no chat template):
     gemma-3 `1b/4b/12b/27b-it`, `Qwen2.5-Coder-7B/32B-Instruct`.
   - *Dataset* — `$WORK/data/dataset.jsonl` (SVEN before/after, rebuilt 2026-06-01,
     1430 rows, balanced).
   - *Split* — seed-42 20% group-clean hold-out `sven_split_meta.json` (VERBATIM
     `load_or_make_split`; same split exp-16 used — so `is_test` lines up).
   - *Framing* — code BEFORE the question, neutral preamble, yes/no first-token
     read; carried verbatim from exp-05 `verbalized_judge.py`. Layer-independent
     (final next-token logits) — no probe, no cached acts, no layer.

3. **Outputs** — on scratch `$WORK/runs/verbalized_logitdump_<slug>/`, pulled to
   `results/<slug>/`:
   - `logits_verbalized.npz` — per-example table: eid, label, is_test, p_yes,
     yes_lp, no_lp, margin, yes_logits_raw[N,|yes|], no_logits_raw[N,|no|],
     topk_ids[N,10], topk_logits[N,10], + yes_ids/no_ids metadata.
   - `example_scores_verbalized.json` — [{eid, p_yes, label, cwe, lang, is_test,
     yes_lp, no_lp, margin}].
   - `metrics_verbalized_logits.json` — verbalized_auc_all (1430) + _test (seed-42),
     yes/no token decodes, question, and the historical-AUC gate.

4. **Result format** — per model: `verbalized_auc_all`, `verbalized_auc_test`,
   yes/no token sets, n_test. Plus the gate for the two exp-05 models:
   gemma-3-27b-it test-AUC vs hist 0.554, Qwen2.5-Coder-32B vs hist 0.632.

5. **Interpretation hints**
   - 27b/32b `verbalized_auc_test` ≈ hist (±~0.05) ⇒ verbalized path faithfully
     reproduced; the dumped logits are trustworthy. (exp-05 averaged over 5 seeds'
     test intersections; this is the single seed-42 test set, so small drift is
     expected — a gross miss signals a reproduction problem.)
   - The 4 NEW models (gemma 1b/4b/12b-it, Qwen-7B) have no verbalized history —
     these AUCs are the first verbalized read for them; report as-is.
   - Sanity in the per-worker debug print: the rendered tail must be the assistant
     turn-start with NO `<think>` token, and yes/no must dominate the first-token
     argmax (else the read is invalid — the script aborts on a `<think>` top-1).

## For agents
- Run (login node): `submit_verbalized_logitdump.sh "google/gemma-3-1b-it
  google/gemma-3-4b-it google/gemma-3-12b-it google/gemma-3-27b-it
  Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Coder-32B-Instruct"`.
  Submits one debug job per model, 4-GPU sharded, serialized (scheduler MaxJobs=1).
- Idempotent: per-model `DONE` marker; resumable per-GPU shards (skip-if-exists),
  so a job that hits the 22.5-min wall just gets resubmitted to continue.
- Gemma is gated → needs `$WORK/secrets/hf_token` (env.sh reads it). Verbalized is
  a light forward (use_cache=False, no hidden-state capture) so no OOM risk even
  on 27b/32b at one model per GPU (unlike exp-16's multi-layer extraction).
- `p_yes` here is bit-identical to exp-05's (same float64 logsumexp/sigmoid
  formula in `verbalized_record`); the persistence is the only widening.
