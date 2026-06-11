#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODELS="${MODELS:-Qwen/Qwen2.5-Coder-32B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct google/gemma-3-1b-it google/gemma-3-4b-it google/gemma-3-12b-it google/gemma-3-12b-pt google/gemma-3-27b-it}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
OUT="${OUT:-./results}"

HERE="plans/cross-model-probe-generalization/17-verbalized-logit-dump"
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

for MODEL in $MODELS; do
    SLUG="$(slug "$MODEL")"
    RUN="$OUT/verbalized_logitdump_$SLUG"
    mkdir -p "$RUN"

    # phase 1: verbalized yes/no forward + logit dump (LOADS MODEL)
    python "$HERE/verbalized_logit_dump.py" \
        --model "$MODEL" --pairs "$DATA" --out "$RUN" \
        --max-length 2048 --topk 10 --n-gpus 1 --gpu-id 0

    # phase 2: merge shards -> logits + example_scores + metrics (+ AUC gate)
    python "$HERE/aggregate_verbalized_logits.py" \
        --model "$MODEL" --shards "$RUN" --dataset "$DATA" \
        --split "$SPLIT" --out "$RUN"
done
