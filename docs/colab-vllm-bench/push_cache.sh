#!/usr/bin/env bash
# [ai-generated]
# Push the vLLM torch.compile cache to a private HF dataset so a later cold
# runtime can pull it in parallel with install/download.
# Usage: bash push_cache.sh <hf_dataset_repo>   e.g. user/vllm-compile-cache-qwen3-1.7b
set -euo pipefail
REPO="${1:?need hf dataset repo}"
: "${HF_TOKEN:?}"
export PATH="$HOME/.local/bin:$PATH"
hf upload "$REPO" /root/.cache/vllm . --repo-type dataset --token "$HF_TOKEN" --private \
  >/content/coldlog/push.log 2>&1
echo "pushed /root/.cache/vllm -> $REPO"
