#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
BEST="${BEST:-25}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
ACTS="${ACTS:-./acts}"            # reuse cached acts
OUT="${OUT:-./results}"
SAMPLER="${SAMPLER:-none family_balanced}"

P="plans/cross-model-probe-generalization/11-family-balanced-head"
mkdir -p "$OUT"
SLUG="$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')"

# family-balanced vs unbalanced general probe, per-family honest tokens_code AUC
for S in $SAMPLER; do
    python "$P/family_balanced_probe.py" \
        --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" --best-layer "$BEST" \
        --sampler "$S" --head linear \
        --out "$OUT/family_balanced_${SLUG}_linear_${S}.json" --model "$MODEL" --epochs 30
done
