[ai-generated]

# 0001 — Drop 3 models from the cross-model sweep; bump to transformers 5.9.0

Date: 2026-05-31

## Context

The overnight the cluster sweep finished 18/23, with 5 `EXTRACT_FAILED` models. Root
causes (under the original transformers 4.57.1 stack):

- `gemma-4-26B-A4B-it`, `Qwen3.6-27B` — 2026 archs (`gemma4`, `qwen3_5`) not
  registered in 4.57.1.
- `OpenCoder-8B-Instruct` — custom-code repo; needed `trust_remote_code`.
- `Devstral-Small-2507`, `Mistral-Small-3.2-24B` — Tekken tokenizer mis-routing.

Investigation under a bumped **transformers 5.9.0** stack (tokenizers 0.22.2,
huggingface_hub 1.17.0; installed into `.python_deps5`, the 4.57.1 tree kept as
rollback; `TF_STACK` overrides the pin set) resolved the arch and download issues
but exposed two hard tokenizer limits. A shared incidental blocker — hub 1.17's
Xet downloader calling `download_files(request_headers=)`, which the image's older
`hf_xet` rejects — was fixed with `HF_HUB_DISABLE_XET=1` (classic HTTPS).

## Decision

**Final roster = 20/23 DONE.** Recovered `gemma-4` and `Qwen3.6` via the 5.9.0
bump (both beat baselines: ex-AUC ~0.70, tok-AUC ~0.84/0.85, full layer sweep,
shared SVEN split). Dropped 3 models:

- **OpenCoder-8B** — only ships a slow custom `INFLMTokenizer` (SentencePiece, no
  fast variant / no `tokenizer.json`) that returns no `offset_mapping`. The
  span-aware extractor requires char→token offsets.
- **Devstral-2507**, **Mistral-Small-3.2-24B** — transformers 5.9.0 converts their
  Tekken `tekken.json` to a 151000-token vocab while the models have only 131072
  embeddings; real code maps to ids ~149k → embedding OOB → CUDA device-side
  assert. The official tokenizer is `mistral-common`, which 5.9.0 does not expose
  as `MistralCommonTokenizer` and which yields no char offsets (same wall as
  OpenCoder).

Rejected the alternative (mistral-common + a validated offset reconstruction):
approximate offsets risk silently mislabeling vuln spans, violating the repo's
integrity rule. The code axis stays well-covered (Qwen2.5-Coder-7B/32B,
deepseek-coder, Qwen3-Coder-30B, starcoder2).

## Consequences

- Sweep comparisons use 20 models. The `code-modern` (Devstral) and
  `family-modern` (Mistral-Small-3.2) axes are unrepresented; the `code` and
  general-`family` axes remain covered.
- Probing any Tekken-tokenizer or offset-less-tokenizer model needs either an
  upstream `MistralCommonTokenizer` (future transformers) or a validated offset
  fallback before it can join the roster.
- `env.sh` now defaults to the 5.9.0 stack; reproducing the 4.57.1 results means
  setting `PYTHON_DEPS_DIR=.python_deps` + the old `TF_STACK`.
