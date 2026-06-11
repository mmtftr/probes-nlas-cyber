[ai-generated]

# Activations on Hugging Face (`mmtf/probes-activations`)

Per-model, per-layer hidden states published as a HF dataset so a single layer
can be pulled on demand (e.g. on Colab) without re-running the base model.
Companion to [[probe-activation-extraction]] (how the raw acts are produced) and
[[the cluster-cluster]] (where they were produced).

## What's published

- **Dataset:** <https://huggingface.co/datasets/mmtf/probes-activations> (public).
- **Scope:** the **top-10 layers per model** by `val_tokens_code_auc` (from the
  layer sweep), stored **bfloat16**, token-level, over the full SVEN set
  (1,430 examples).
- **460 GB, 154 files.** Per-model directory = model id with `/`→`_`
  (**single** underscore), e.g. `google_gemma-3-27b-it`,
  `Qwen_Qwen2.5-Coder-32B-Instruct`.

| model | dir | layers | hidden | tokens | best L (AUC) | 10-layer size |
|---|---|--:|--:|--:|--:|--:|
| gemma-3-1b-it  | `google_gemma-3-1b-it`  | 26 | 1152 | 690,148 | 25 (0.737) | 15.9 GB |
| gemma-3-1b-pt  | `google_gemma-3-1b-pt`  | 26 | 1152 | 690,148 | 12 (0.755) | 15.9 GB |
| gemma-3-4b-it  | `google_gemma-3-4b-it`  | 34 | 2560 | 690,148 | 7 (0.753)  | 35.3 GB |
| gemma-3-4b-pt  | `google_gemma-3-4b-pt`  | 34 | 2560 | 690,148 | 33 (0.768) | 35.3 GB |
| gemma-3-12b-it | `google_gemma-3-12b-it` | 48 | 3840 | 690,148 | 15 (0.755) | 53.0 GB |
| gemma-3-12b-pt | `google_gemma-3-12b-pt` | 48 | 3840 | 690,148 | 13 (0.761) | 53.0 GB |
| gemma-3-27b-it | `google_gemma-3-27b-it` | 62 | 5376 | 690,148 | 19 (0.766) | 74.2 GB |
| Qwen2.5-Coder-32B-Instruct | `Qwen_Qwen2.5-Coder-32B-Instruct` | 64 | 5120 | 561,266 | 25 (0.785) | 57.5 GB |
| Qwen3-32B    | `Qwen_Qwen3-32B`    | 64 | 5120 | 561,266 | 27 (0.782) | 57.5 GB |
| Qwen3.6-27B  | `Qwen_Qwen3.6-27B`  | 64 | 5120 | 612,249 | 30 (0.772) | 62.7 GB |

Top-10 layer indices per model (sorted; argmax = "best L" above):

```
gemma-3-1b-it   3 4 6 7 9 10 13 14 23 25      gemma-3-12b-it  9 10 11 12 13 14 15 19 24 28
gemma-3-1b-pt   4 5 7 9 11 12 20 22 24 25      gemma-3-12b-pt  10 11 13 14 15 16 18 22 26 47
gemma-3-4b-it   3 4 7 8 9 10 12 23 30 33       gemma-3-27b-it  16 17 18 19 20 21 22 23 25 26
gemma-3-4b-pt   3 4 6 7 8 9 10 13 14 33        Qwen2.5-Coder-32B  25 36 37 39 40 41 42 43 44 45
Qwen3-32B       17 23 24 25 26 27 28 29 30 42  Qwen3.6-27B        14 16 17 18 19 30 31 32 46 49
```

Note `gemma-3-1b-it` and `gemma-3-4b-pt` argmax onto the **final** layer; for
1b-it it's a near-tie with mid layers (0.737 vs 0.730 @ L6) — prefer a mid layer
unless you specifically want last-layer behaviour.

## Repo layout

Per model `<dir>/`:

| file | dtype | shape | meaning |
|---|---|---|---|
| `layer_NN.safetensors` | bf16 | (T_tokens, hidden) | hidden state at layer NN; tensor key `activations` |
| `y.npy` | int8 | (T_tokens,) | per-token positive-span (vulnerable) label |
| `example_ids.npy` | int32 | (T_tokens,) | example index each token belongs to |
| `offsets.npz` | int32 | per-row (T_row, 2) | char-span offsets per example (extractor parity) |
| `meta.json` | — | — | model, n_layers, hidden, n_tokens, n_rows, max_length, pos_tokens |
| `top_layers.json` | — | — | the 10 layers + their sweep scores |

Shared at the repo root: `data/dataset.jsonl`, `data/sven_split_meta.json`,
`README.md` (auto-generated dataset card).

## Loading

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import numpy as np

REPO, m = "mmtf/probes-activations", "google_gemma-3-27b-it"   # dir = model id, "/" → "_"
acts = load_file(hf_hub_download(REPO, f"{m}/layer_19.safetensors", repo_type="dataset"))["activations"]
y    = np.load(hf_hub_download(REPO, f"{m}/y.npy",          repo_type="dataset"))   # int8 token label
eids = np.load(hf_hub_download(REPO, f"{m}/example_ids.npy", repo_type="dataset"))  # int32 example id
```

- Token rows align 1:1 across `layer_NN.safetensors`, `y.npy`, `example_ids.npy`.
- Split is defined **by example** in `sven_split_meta.json`; map tokens→examples
  via `example_ids`. Pairs are group-clean (see [[probe-activation-extraction]]
  for the split rationale) — never split a vuln/fix pair across train/test.
- For span-max training ([[span-max-loss-tuning]]), `y` is the per-token label
  over code-mask positions; `offsets.npz` recovers char spans if needed.

## Why bfloat16 (not fp16, not fp32)

- fp16 **overflows** on Gemma-3 "massive activations" (>65504 → inf → NaN in
  training/AUC). bf16 keeps fp32's exponent range, so values like ~3.6e5 survive
  (verified: absmax 362,496 round-trips cleanly).
- The **source** acts on scratch are fp32 (the working format, per
  [[probe-activation-extraction]]); bf16 is the **distribution** format — half
  the bytes, no precision issue for linear probes.
- Converted on the GPU login node (no GPU needed): chunked `np.load(mmap)` →
  `torch.bfloat16` → safetensors, ~3–18 s/layer. Shared Lustre throws the odd
  multi-minute cold-read stall — expect a few per full run, not a failure.

## Generation reference (1× GPU, batch=1, full sequence, all layers)

What it cost to produce the source fp32 acts — use to estimate re-extraction.

| model | fwd time | throughput | fp32 all-layers |
|---|--:|--:|--:|
| gemma-3-1b-pt  | 75 s  | 9.2k tok/s | 78 GB |
| gemma-3-1b-it  | 86 s  | 8.0k tok/s | 78 GB |
| gemma-3-4b-it  | 155 s | 4.5k tok/s | 224 GB |
| gemma-3-4b-pt  | 161 s | 4.3k tok/s | 224 GB |
| gemma-3-12b-pt | 288 s | 2.4k tok/s | 474 GB |
| gemma-3-12b-it | 294 s | 2.3k tok/s | 474 GB |
| Qwen2.5-Coder-32B | 393 s | 1.4k tok/s | 686 GB |
| Qwen3-32B      | 418 s | 1.3k tok/s | 686 GB |
| gemma-3-27b-it | 481 s | 1.4k tok/s | 857 GB |
| Qwen3.6-27B    | 868 s | 0.7k tok/s | 748 GB |

- These exclude one-time HF weight download (~1–4 min) and weight load.
- **batch=1 leaves the GPU underfed** (a 1B at 8k tok/s is ~1–2 orders below
  what a GPU can do). Length-bucketed batching would speed extraction several×;
  the big models are also partly bound by writing all-layer fp32 to scratch.
- Acts are **split-independent** — extract once, reuse across splits / losses /
  α without re-extracting (what makes the sweeps cheap).

## Download vs regenerate (Colab)

- Pulling a precomputed layer skips **both** the model-weight download and the
  forward pass → roughly **10–25× faster** than regenerating, for a single layer.
- Crossover: regen cost is *fixed* in #layers (one forward emits all layers);
  download scales with #layers. For 27b, break-even ≈ **27 of 62** layers. Pull
  fewer than ~half → download wins; full re-sweep → regenerate.
- A single bf16 layer (1.5–7 GB) fits in Colab RAM, so just `hf_hub_download` +
  `load_file`. True HF streaming only pays off if the data won't fit on disk/RAM.

## Re-uploading / adding models

- Scripts on scratch: `probes/hf_convert.py` (fp32→bf16 stage, idempotent,
  top-K by `val_tokens_code_auc`) and `probes/hf_upload.py` (folder upload + card
  generation). Env: `probes/.hf_convert_venv` (CPU torch + safetensors + hub).
- **Set `HF_HUB_DISABLE_XET=1`.** Xet throttled to ~3 MB/s (single-core) on these
  pre-chunked, incompressible bf16 tensors; classic LFS multipart with
  `num_workers=16` ran ~10 files in parallel at ~40 MB/s each. Raw egress from
  the login node measured ~39 MB/s+.
- Call `api.upload_large_folder(...)` (the **bound** `HfApi` method) so the token
  is used — the module-level function defaults to no token → 401 on create_repo.
