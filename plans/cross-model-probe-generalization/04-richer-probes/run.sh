#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-27b-it}"
DATA="${DATA:-./data/dataset.jsonl}"
OUT="${OUT:-./results}"
ACTS="${ACTS:-./acts}"            # reuse exp-02 activations
FEATURESETS="${FEATURESETS:-9,19,26,61;19;17,19,22}"
ARCHS="${ARCHS:-linear,mlp256,mlp512}"
SEEDS="${SEEDS:-42,43,44,45,46}"
ALPHA="${ALPHA:-1.0}"

EXP="plans/cross-model-probe-generalization/04-richer-probes"
CELLS="$OUT/cells"
mkdir -p "$CELLS"

# phase 1: cell grid (feature-sets x archs) on cached acts
python "$EXP/richer_probe_sweep.py" \
    --acts-dir "$ACTS" --dataset "$DATA" --out "$CELLS" \
    --feature-sets "$FEATURESETS" --archs "$ARCHS" --seeds "$SEEDS" \
    --epochs 30 --alpha "$ALPHA" --n-gpus 1 --gpu-id 0

# phase 2: aggregate
python "$EXP/aggregate_richer.py" --cell-dir "$CELLS" \
    --out "$OUT/metrics_richer.json" --model "$MODEL"
