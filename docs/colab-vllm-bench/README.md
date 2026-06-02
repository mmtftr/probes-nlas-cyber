[ai-generated]

# Colab vLLM cold-start + extraction benchmark harness

Reconstructed 2026-06-02 (the originals lived on a Colab `/content` that was
recycled). Companion to `../vllm-hidden-states-extraction.md`.

| File | Purpose |
|---|---|
| `cold_start.sh` | Parallelized cold setup: bootstraps `uv`/`hf_transfer`, then overlaps model download ∥ vllm+cu130-torch install ∥ optional compile-cache pull. Prints per-phase wall-clock. `bash cold_start.sh <MODEL> [CACHE_HF_DATASET]` |
| `bench_vllm.py` | vLLM `extract_hidden_states` benchmark over 512 synthetic prompts. Run as a subprocess. Emits a `BENCH_JSON {...}` line with `engine_init`/`extract` timings, tok/s, and a verification of written SafeTensors. `python bench_vllm.py <MODEL> [N]` |
| `push_cache.sh` | Push `/root/.cache/vllm` (torch.compile cache) to a private HF dataset for cross-runtime cache pulls. `bash push_cache.sh <hf_dataset_repo>` |

## Run (driven from a laptop via colab-cli)

```bash
HF=$(grep ^HF_TOKEN= .env | cut -d= -f2)
SID=$(colab-cli session new --variant GPU --accelerator G4 --wait --json | jq -r .id)
colab-cli files put docs/colab-vllm-bench/cold_start.sh /content/cold_start.sh
colab-cli files put docs/colab-vllm-bench/bench_vllm.py /content/bench_vllm.py
colab-cli exec "$SID" --code "open('/content/.hf','w').write('$HF')"

# Long jobs: launch DETACHED on the runtime and poll the output file via the
# file API — exec over the ws can idle out on multi-minute jobs.
colab-cli exec "$SID" --code "import os,subprocess; os.environ['HF_TOKEN']=open('/content/.hf').read().strip(); subprocess.Popen('bash /content/cold_start.sh Qwen/Qwen3-14B > /content/coldlog/setup.out 2>&1; python /content/bench_vllm.py Qwen/Qwen3-14B 512 > /content/coldlog/bench.out 2>&1', shell=True, env=os.environ)"
colab-cli files get /content/coldlog/bench.out -   # poll until BENCH_JSON appears
```

Blackwell (sm120) note: `bench_vllm.py` sets `VLLM_USE_FLASHINFER_SAMPLER=0` —
vllm 0.22 bundles flashinfer and its sampler's `check_cuda_arch()` rejects sm120.
