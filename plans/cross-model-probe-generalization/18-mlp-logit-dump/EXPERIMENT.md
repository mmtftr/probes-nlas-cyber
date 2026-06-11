[ai-generated]

# 18 — MLP-probe logit dump (the third logit variant, 16-GPU resumable)

exp-16 dumped the LINEAR span-max probe's logits; exp-17 the verbalized read.
This dumps the MLP-probe family (exp-12). The MLP scores as `sigmoid(probe(X))`
(NOT the linear `X·w+b`), so it is a genuinely separate dump.

1. **Aim** — Materialise every per-token + per-example MLP-probe logit, for both
   heads (mlp256, mlp512), over the full SVEN set (1430 ex), for the 6 instruct
   models. Reproduction + persistence pass. Gate: reproduce exp-12's MLP
   `tokens_code_auc` for the 2 models it characterised.

2. **Inputs**
   - *Models (6)* — 2 with a known exp-12 MLP best-layer (same layer both heads):
     `gemma-3-27b-it` **L21**, `Qwen2.5-Coder-32B-Instruct` **L37** → extract that
     layer + dump. 4 with NO MLP sweep — `gemma-3 1b/4b/12b-it`,
     `Qwen2.5-Coder-7B-Instruct` → extract ALL layers, MLP-sweep every layer
     (`val_tokens_code_auc`-selected, exp-12's 15%/seed-42 carve), dump at each
     head's best.
   - *Dataset / split* — `./data/dataset.jsonl`, seed-42 group hold-out
     `sven_split_meta.json` (same as exp-12/16). VAL = 15% of TRAIN groups,
     VAL_SEED=42 (exp-12). MLP train: `train_one_layer` + `MLPProbe(H)` via
     `probe_factory`, epochs=30, seed=7 — byte-for-byte exp-12.

3. **Compute shape** — 16 workers (one per GPU, rank passed via --rank/--world-size),
   resumable: every unit is a skip-if-exists file, so a worker interrupted by a
   time limit just gets re-run and continues (`run.sh`). Stages, dependency-gated by
   file presence:
   - extract: (model, shard) example-sharded (eid % 16); each shard forwards its
     ~1/16 rows ONCE, writes `layer{NN}_shard{R}.npz` + offsets + DONE.
   - sweep: (model, layer, head) — concat the 16 shards, train MLP on FIT, write
     `sweep_<head>/layer{NN}.json` (val/test honest tokens_code_auc).
   - dump: (model, head) — best layer, retrain (deterministic → same probe), dump.

4. **Outputs** — `./runs/mlp_logitdump/<slug>/dump_<head>/`, collected into
   `results/<slug>/<head>/`:
   - `logits_mlp.npz` — token table: logit (=probe(X)), prob, y, example_id,
     char_start/end, is_test, is_code, layer, head.
   - `example_scores_mlp.json` — per-example max-pool score + label + cwe + lang.
   - `metrics_mlp.json` — best layer, test tokens_auc / tokens_code_auc /
     example_auc, + the exp-12 gate (fixed models).

5. **Result format** — per (model, head): best layer; held-out tokens_code_auc,
   example_auc. Gate for the 2 fixed models: gemma-27b-it L21 vs 0.822/0.824,
   Coder-32B L37 vs 0.817/0.816 (mlp256/mlp512).

6. **Interpretation hints**
   - Fixed-model `tokens_code_auc` ≈ exp-12 (±~0.02) ⇒ MLP path reproduced;
     logits trustworthy. (MLP train is seeded → expect tight reproduction.)
   - The 4 sweep models have no MLP history — their best layer + tc are the first
     MLP read for them; exp-12 found the MLP best layer is deeper than the linear
     one and varies widely (depth frac 0.22–0.70), hence the full all-layer sweep.

## For agents
- Preflight (validate flow + GPU pinning on the smallest model):
  `bash run.sh google/gemma-3-1b-it`
  (world=4; still writes all 16 shards). Then run the full roster via `run.sh`.
- EXTRACT_SHARDS=16 is FIXED so a 4-GPU preflight and the 16-GPU run share shards.
- GPU pinning: one worker per local GPU; rank = global worker index in [0, world);
  world = total worker count (all passed as CLI args).
- Gemma is gated → needs a Hugging Face token. MLP train is
  deterministic (seed=7) so the gate reproduces exp-12's tc.
- `TODO(adhoc-decision)`: the dump trains the MLP on FIT (val carved), matching
  exp-12's selected-layer number, NOT on full TRAIN — so the gate is exact. The
  `is_test` column lets downstream restrict to held-out for fair cross-variant
  comparison with exp-16/17.
