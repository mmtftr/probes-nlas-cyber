#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

# model:repo_layer pairs.
PAIRS="${PAIRS:-Qwen/Qwen2.5-Coder-32B-Instruct:25 google/gemma-3-1b-it:25}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
OUT="${OUT:-./results}"

EXTRACT="src/data/extract_token_activations.py"
TRAIN="plans/cross-model-probe-generalization/21-per-cwe-cross-cwe/train_percwe.py"
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

for pair in $PAIRS; do
    M="${pair%%:*}"; L="${pair##*:}"
    SLUG="$(slug "$M")"
    RUN="$OUT/percwe_$SLUG"; ACTS="$RUN/token_activations"
    mkdir -p "$RUN"

    # extract the operating layer (keep acts)
    python "$EXTRACT" --model "$M" \
        --pairs "$DATA" --out "$ACTS" --layers "$L" --max-length 2048 --backend hf

    # per-CWE probes + cross-CWE detection matrix
    python "$TRAIN" --model "$M" --acts "$ACTS" \
        --dataset "$DATA" --split "$SPLIT" \
        --layer "$L" --out "$RUN"
done
