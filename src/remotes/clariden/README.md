[ai-generated]

# Clariden cross-model probe sweep

Method-generalization sweep: re-fit the **same** token-level span-max recipe on
each model's own activations (probes are model-specific — no cross-model
transfer here) and compare probe *properties* across models. See
`docs/research-framing.md` §1/§5 and Q5.

## Cluster facts (clariden, account `lsaie-ss26`)

- Node = **4× GH200 (96 GB each)**, 460 GB RAM, 288 CPU. Container `--environment=alps3`.
- Budget = **90 node-minutes/job**: 1 node × 90 min (what we use) or 4 nodes × 22.5 min.
- Partition `debug` (1:30 limit, 1384 nodes → no queue). Unlimited jobs.
- All paths under `~/scratch` (= `/iopsstor/scratch/cscs/course_00136`, permanent).

## Layout

```
~/scratch/probes/
  repo/                 # this repo (git clone, pull to update)
  data/                 # dataset.jsonl (+ token_labels), sven_split_meta.json  [shared]
  hf_cache/             # HF_HOME (model downloads)
  .python_deps/         # uv-installed HF deps (added to PYTHONPATH in env.sh)
  secrets/hf_token      # <-- USER drops HF token here (one line). Gated models need it.
  runs/<model_slug>/    # token_activations/, probe.npz, metrics.json, DONE|FAILED, run.log
  logs/                 # slurm job logs
```

## Files

- `models.txt`   — roster (one HF id/line; axis tag; VERIFY-ID / GATED flags).
- `env.sh`       — container-side env + idempotent dep install (sourced in the job).
- `run_model.sh` — per-model: extract → `train_eval.py`. Idempotent, GPU-pinned.
- `train_eval.py`— span-max per layer + baselines + metrics.json (+ makes the split).
- `submit.sh`    — chunks roster into 4-per-node-job, sbatch to `debug`.

## Run order

1. **Token** — user writes `~/scratch/probes/secrets/hf_token` and accepts model
   licenses on HF (Gemma/Llama/Mistral/StarCoder are gated).
2. **Dataset prep** (login node, network ok) — build SVEN with the rich token
   labels the span-aware extractor needs:
   ```
   cd ~/scratch/probes/repo
   python scripts/build_dataset_sven.py --out ~/scratch/probes/data/dataset.jsonl
   python scripts/derive_rich_labels.py  ~/scratch/probes/data/dataset.jsonl   # token_labels spans
   python scripts/validate_dataset.py    ~/scratch/probes/data/dataset.jsonl
   ```
   (exact flags may differ — confirm against each script's argparse.)
3. **Smoke** — one model per tokenizer family before the full sweep:
   ```
   echo -e "google/gemma-3-1b-it\nQwen/Qwen2.5-Coder-7B-Instruct" > /tmp/smoke.txt
   ./src/remotes/clariden/submit.sh /tmp/smoke.txt 00:30:00
   ```
   Check `runs/*/metrics.json` exists and `runs/*/FAILED` does not.
4. **Full sweep** — `./src/remotes/clariden/submit.sh`  (re-run to fill gaps; DONE is skipped).
5. **Collect** — `jq -s '.' ~/scratch/probes/runs/*/metrics.json` → compare
   best_layer_frac / AUC / baseline-lift across models.

## Known risks (smoke catches these)

- **Container lacks `transformers`** → `env.sh` installs into `.python_deps`.
- **Tokenizer offset_mapping** — the span extractor needs `return_offset_mapping`;
  some fast tokenizers (certain BPE families) don't provide it → needs a fallback.
- **Gated models** 401 without a valid HF token + accepted license.
- **VERIFY-ID rows** (gemma-4, qwen3.6) — confirm the real HF repo id; drop if absent.
- **MoE / >32B** won't fit one GPU — keep to the roster.
