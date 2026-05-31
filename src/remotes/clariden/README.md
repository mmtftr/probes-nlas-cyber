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

- **Container lacks `transformers`** → `env.sh` installs into `.python_deps5`
  (transformers 5.9.0 stack; the older 4.57.1 tree at `.python_deps` is kept as a
  rollback). 5.9.0 is the first stack that registers the 2026 archs gemma4 +
  qwen3_5; override the whole pin set via `$TF_STACK`.
- **Xet downloader** — huggingface_hub 1.17 calls `download_files(request_headers=)`
  which the image's older `hf_xet` rejects (`unexpected keyword argument`), killing
  every fresh download. `env.sh` sets `HF_HUB_DISABLE_XET=1` → classic HTTPS.
- **Tokenizer offset_mapping** — the span extractor needs `return_offset_mapping`;
  slow/custom tokenizers (e.g. OpenCoder's `INFLMTokenizer`) don't provide it and
  have no fast variant → can't be probed with this recipe. **OpenCoder dropped.**
- **Tekken vocab mismatch** — Mistral Tekken models (Devstral-2507,
  Mistral-Small-3.2) convert to a 151000-token vocab vs the model's 131072
  embeddings → out-of-range ids → CUDA assert. No clean fix on 5.9.0
  (no `MistralCommonTokenizer`, mistral-common gives no offsets). **Both dropped.**
- **Gated models** 401 without a valid HF token + accepted license.
- **gemma-4 / qwen3.6** — CONFIRMED working on 5.9.0 (no longer "verify-id"; both DONE).
- **MoE / >32B** won't fit one GPU — keep to the roster.

## Final sweep status (2026-05-31)

**20/23 DONE.** Dropped: OpenCoder-8B (no offset_mapping), Devstral-2507 and
Mistral-Small-3.2 (Tekken vocab mismatch). gemma-4 and Qwen3.6 recovered via the
5.9.0 bump (both beat baselines: ex-AUC ~0.70, tok-AUC ~0.84/0.85).
