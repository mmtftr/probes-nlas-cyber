[ai-generated]

# vLLM Hidden-State Extraction on Colab Blackwell

Session summary: investigation and validation of vLLM's `extract_hidden_states` API as a faster alternative to the repo's current transformers-batch-1 extraction pipeline.

---

## What this is

vLLM ≥0.18.0 (PR #33736) exposes a `extract_hidden_states` speculative method that hooks a `KVConnector` to write per-request hidden states to disk (SafeTensors) without modifying the model forward pass. Used with `ExampleHiddenStatesConnector`.

```python
from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector

llm = LLM(
    model="Qwen/Qwen3-8B",
    enable_chunked_prefill=False,
    enable_prefix_caching=False,  # MUST be off — see gotchas
    attention_backend="FLASH_ATTN",
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": [10, 19, 28, 36]},
            # aux ids: vllm aux[i] == transformers hidden_states[i] (not i+1)
        },
    },
    kv_transfer_config=KVTransferConfig(
        kv_connector="ExampleHiddenStatesConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"shared_storage_path": "/dev/shm/hs_extract"},
    ),
)
outputs = llm.generate(prompts, SamplingParams(max_tokens=1))
for o in outputs:
    path = o.kv_transfer_params["hidden_states_path"]
    obj = example_hidden_states_connector.load_hidden_states(path)
    # obj["token_ids"]: (n_tokens,)
    # obj["hidden_states"]: (n_tokens, n_aux_layers, hidden_dim)
```

Output shape: `[num_tokens, num_aux_layers, hidden_dim]` per request, one SafeTensors file per request in `/dev/shm`.

---

## Environment: Colab G4 (RTX PRO 6000 Blackwell, sm_120)

The G4 is a Blackwell GPU (sm_120, 96 GB VRAM, 87 GB `/dev/shm`). The base Colab image ships CUDA 12.8 + torch cu128, which does not work with vllm 0.22 (CUDA 13 wheels). Required stack:

- **torch 2.11.0+cu130** (vllm 0.22 wheels need `libcudart.so.13`)
- **vllm 0.22.0**
- **hf_transfer 0.1.9** (parallel multi-chunk HF downloads)

### Minimal setup (FLASH_ATTN — fastest path)

```bash
# install vllm + force torch to cu130 (base image's cu128 "satisfies" vllm, so force-reinstall)
uv pip install --system --torch-backend=cu130 "vllm==0.22.0" hf_transfer
uv pip install --system --torch-backend=cu130 --force-reinstall "torch==2.11.0"
uv pip uninstall --system torchvision torchaudio  # cu128 leftovers crash `import transformers`
```

**Run via subprocess** (not inside IPython kernel) to avoid vLLM's `stdout.fileno()` crash. The `VLLM_LOGGING_LEVEL=debug` hack the original code used is no longer needed.

---

## Gotchas discovered

| Issue | Root cause | Fix |
|---|---|---|
| `ImportError: libcudart.so.13` | torch is cu128, vllm 0.22 needs CUDA 13 | `--force-reinstall torch==2.11.0 --torch-backend=cu130` |
| `transformers` import crash | torchvision/torchaudio are still cu128 | uninstall them (not needed for text models) |
| Hidden states incomplete (fewer tokens than input) | Prefix caching on — SVEN before/after pairs share long prefixes; cached tokens emit no hidden states | `enable_prefix_caching=False` |
| `VLLM_ATTENTION_BACKEND` silently ignored | Removed in vllm 0.22 | `LLM(attention_backend="FLASH_ATTN")` (constructor arg) |
| `enforce_eager=True` hangs | Disables cudagraphs → Triton JIT recompiles per sequence length → thrash on 1430 varied lengths | Don't use it; keep torch.compile on |
| Layer index off-by-one | vllm `eagle_aux_hidden_state_layer_ids=[i]` returns `hidden_states[i]` = output of layer `i-1`. Repo stores "layer L" as `hs[L+1]` | To get repo-layer L, pass aux id `L+1` |
| Numerical match (cos ~0.93) looked wrong | Comparator had the off-by-one; at the correct index cos=0.9999 | At matched index, hidden states are numerically faithful (bf16 + FA2 vs sdpa) |
| `RuntimeError: FlashInfer requires GPUs with sm75 or higher` during engine init | vllm 0.22 now bundles `flashinfer-python`; the **sampler** JIT-compiles a kernel and its `check_cuda_arch()` mis-parses sm120 (Blackwell) and aborts. Independent of `attention_backend`. | `VLLM_USE_FLASHINFER_SAMPLER=0` (forces the native torch sampler), or `uv pip uninstall flashinfer-python`. The doc's original FLASH_ATTN path simply had no flashinfer installed. |

---

## Benchmark: full SVEN dataset (1430 examples, 561k tokens, Qwen3-8B, 4 quarter-point layers)

**Warm numbers** (torch.compile cache present on disk):

| Arm | Compute | ex/s | tok/s | vs transformers |
|---|---|---|---|---|
| **transformers batch-1** (current method) | 51.3s | 27.9 | 10.9k | 1.0× |
| **vllm FLASH_ATTN** | 23.5s | 61.1 | 24.0k | **2.18×** |
| vllm FLASHINFER | 27.9s | 51.2 | 20.1k | 1.84× (slower) |

**flashinfer is slower than FLASH_ATTN** for this workload: extraction is prefill-only (`max_tokens=1`), so flashinfer's decode/paging wins don't apply. Not worth the JIT overhead.

Inputs for all arms: pre-tokenized with HF tokenizer, truncated to 2048 tokens, passed as `prompt_token_ids` → identical token counts between transformers and vllm.

---

## Setup timing (Colab G4, cold runtime)

| Phase | Time | Notes |
|---|---|---|
| uv install (vllm + cu130 torch, 190 packages) | **~31s** | Parallel wheel download; not the bottleneck |
| Model download (Qwen3-8B, 16 GB, hf_transfer) | **~49s** | ~333 MB/s, 15 files in parallel |
| vllm engine init — cold | **~53s** | torch.compile ~14s + cudagraph capture + first-`generate()` penalty |
| vllm engine init — warm (cache on disk) | **~18s** | Cache survives process/kernel restarts; wiped only on runtime unassign |

The **torch.compile cache** (`/root/.cache/vllm`, 99 MB uncompressed / 14 MB gzip) is the only avoidable cost on a cold runtime.

### Cold-start optimizations applied

1. **Parallel setup**: model download ∥ vllm install ∥ compile-cache pull run concurrently via background `&` jobs.
2. **Compile cache persistence** via dufs file server (`https://185.21.216.164:61432`):
   - Pull before first run: `bash /content/pull_cache.sh` (~1s transfer)
   - Push after first compile: `bash /content/push_cache.sh`
   - Validated round-trip: wipe local cache → pull from server → vllm loads compiled graphs directly (`torch.compile took 2.34s` vs ~14s cold)
3. **No flashinfer toolchain** in the FLASH_ATTN path — removes ~1 min of unnecessary setup.
4. **hf_transfer** enabled (`HF_HUB_ENABLE_HF_TRANSFER=1`) — already saturating Colab's link.
5. **Keep runtime assigned** between jobs — warm load is ~18s; every unassign re-pays the full ~53s cold recompile.

Estimated optimized cold start: **~70s** (uv install overlapped with model download + cache pull, then warm load ~20s + compute ~23s).

---

## Parallelized cold-start benchmark across model sizes (2026-06-02)

Re-ran the full cold-start path on a freshly-assigned G4 with the install,
download, and (optional) cache-pull jobs overlapped via background `&` jobs
(`docs/colab-vllm-bench/cold_start.sh`). Extraction harness:
`docs/colab-vllm-bench/bench_vllm.py` (512 synthetic code prompts, lengths
cycling 128–2048 tok = 405k tok total, 4 evenly-spaced aux layers, run as a
subprocess). Extraction was verified each run: 512 SafeTensors written, shapes
`[n_tok, 4, hidden]`.

### Setup (parallel install ∥ download), wall-clock

| Model | Weights | bootstrap | download (∥) | install (∥) | **setup wall** | bound by |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 3.4 GB | ~14s cold | 9.6s | **41.9s** | **~56s** | install |
| Qwen3-14B | 28 GB | ~14s cold¹ | **81.6s** | 42s cold¹ | **~96s** cold | download |

¹ Measured warm (bootstrap 1.1s, install 6.3s) because the 14B run reused the
1.7B runtime; the cold figures are the 1.7B-measured bootstrap/install, which
are model-independent. Either way install hides **fully** under the 28 GB
download.

**Takeaway:** install is a model-independent ~42s floor; download scales with
weights at ~345 MB/s on G4. Crossover ≈ 14 GB — below it the cold start is
install-bound (~56s), above it download-bound. The parallelization makes the
smaller of {install, download} free.

### Extraction (engine init + compute), 512 prompts / 405k tokens

| Model | layers | hidden | engine_init (cold) | extract | tok/s | ex/s |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 28 | 2048 | 32.6s | 27.6s | 14.7k | 18.6 |
| Qwen3-14B | 40 | 5120 | 43.0s | 31.3s | 12.9k | 16.3 |

**End-to-end cold (assign → extracted 512):** 1.7B ≈ **116s**, 14B ≈ **170s**.

Notes:
- vLLM logs `torch.compile ... does not support` for these Qwen3 dense models →
  it runs **piecewise/cudagraph only, no full-graph compile**. So the
  compile-cache pull (the avoidable cold cost for Qwen3-8B above) buys little
  here; `engine_init` is dominated by weight load + cudagraph capture + the
  profile run.
- tok/s here is not comparable to the 24k SVEN number above: different length
  mix and a cold process. These are standalone cold-start figures.
- Needed `VLLM_USE_FLASHINFER_SAMPLER=0` on Blackwell (see gotchas).

These runs were driven through a hardened `colab-cli` (auto kernel
reconnect/restart on a dropped ws + ws ping/pong heartbeat + `session status`);
the long, detached extraction jobs no longer hang the client when the kernel ws
idles out — poll the output file via the file API instead.

---

## Scripts on Colab runtime

> Runtime `/content` is wiped on unassign. Reusable copies of the cold-start +
> extraction harness are checked in at `docs/colab-vllm-bench/` (`cold_start.sh`,
> `bench_vllm.py`, `push_cache.sh`) — re-upload with `colab-cli files put`.

| File | Purpose |
|---|---|
| `/content/setup_fa.sh` | Minimal FLASH_ATTN setup (no flashinfer) |
| `/content/setup_full.sh` | Full setup including flashinfer toolchain |
| `/content/bench_transformers.py` | Transformers batch-1 extraction benchmark |
| `/content/bench_vllm.py` | vLLM extraction benchmark (`HS_ATTN_BACKEND`, `HS_EAGER`, etc.) |
| `/content/bench_compare.py` | Numerical equivalence check (cos, MAE) between arms |
| `/content/pull_cache.sh` | Pull compile cache from dufs server before first run |
| `/content/push_cache.sh` | Push compile cache to dufs server after first compile |

---

## flashinfer on Blackwell (recipe, in case needed later)

vllm pulls nvcc 13.3 which breaks flashinfer two ways: (a) trips cccl compat guard (cccl + torch both target CUDA 13.0, not 13.3) and (b) emits PTX 9.3 that the 13.0 `ptxas` rejects. Fix:

```bash
# pin toolchain to 13.0.88
uv pip install --system nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88
# fix flashinfer's linker path expectations
CU=/usr/local/lib/python3.12/dist-packages/nvidia/cu13
ln -sfn $CU/lib $CU/lib64
ln -sf libcudart.so.13 $CU/lib/libcudart.so
mkdir -p $CU/lib/stubs
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so $CU/lib/stubs/libcuda.so
# runtime env for any flashinfer run
export CUDA_HOME=$CU PATH=$CU/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0+PTX
# and use: LLM(attention_backend="FLASHINFER")
```

First run JIT-compiles flashinfer kernels (a few minutes, then cached). Not recommended for extraction — FLASH_ATTN is faster.

---

## the cluster (GPU, container container) — validated 2026-06-07

Container: aarch64 + Hopper sm_90, py3.12, NGC torch 2.10.0a0/CUDA 13.1. vLLM is
**not** preinstalled. Install into an isolated deps dir (uv resolves a consistent
stack — vllm 0.22.1 + torch 2.11.0+cu130; cu13 torch runs fine under the 13.1
driver):

```bash
uv pip install --target $WORK/.python_deps_vllm --python "$(command -v python)" vllm hf_transfer
# extraction subprocess only (don't shadow the container torch used for training):
PYTHONPATH="$WORK/.python_deps_vllm:$PYTHONPATH" python src/data/extract_token_activations.py --backend vllm ...
```

Wired into `src/data/extract_token_activations.py` as `--backend vllm` (default;
`extract_vllm()`), reusing the HF tokenizer for input_ids/offsets/labels.

**Gotchas hit on the cluster (beyond the Colab table):**
- **multiprocessing spawn:** vLLM v1 spawns its engine-core process and re-imports
  the entry module → all top-level work must be under `if __name__ == "__main__":`
  (the extractor's `main()` already is; standalone scripts must guard too).
- **EAGLE3 per-arch support:** `extract_hidden_states` is EAGLE3-based and only
  works for models that implement the interface. In vllm 0.22.1: **qwen2/qwen3,
  llama, deepseek, gemma-4 YES; gemma-3 NO** (`Model does not support EAGLE3
  interface but aux_hidden_state_outputs was requested`). Use HF for gemma-3.
- **Numerical:** vLLM (FLASH_ATTN bf16) vs HF (eager bf16) mean cos ≈ 0.998 on
  Qwen2.5-Coder-7B L16 (short seqs 0.9995+; a few low-norm tokens in long seqs
  drag the min — benign, not an alignment bug since short rows stay tight).
