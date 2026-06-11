#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
OUT="${OUT:-./results/mlp_logitdump}"
ONLY_MODEL="${ONLY_MODEL:-}"      # optional: restrict to one model id

ONLY_ARG=(); [ -n "$ONLY_MODEL" ] && ONLY_ARG=(--only-model "$ONLY_MODEL")
mkdir -p "$OUT"

# Resumable MLP-logit-dump pipeline (extract -> sweep -> dump). Single process
# (rank 0 / world-size 1); the original sharded it across GPUs/nodes.
python plans/cross-model-probe-generalization/18-mlp-logit-dump/mlp_pipeline.py \
    --rank 0 --world-size 1 \
    --dataset "$DATA" --split "$SPLIT" \
    --out "$OUT" --time-budget 1140 "${ONLY_ARG[@]}"
