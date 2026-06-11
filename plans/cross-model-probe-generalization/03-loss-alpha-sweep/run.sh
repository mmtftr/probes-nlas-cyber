#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-27b-it}"
DATA="${DATA:-./data/dataset.jsonl}"
OUT="${OUT:-./results}"
ACTS="${ACTS:-./acts}"            # reuse exp-02 activations
LAYERS="${LAYERS:-10,19,26,61}"
ALPHAS="${ALPHAS:-1,5,10,20,50}"
LOSSES="${LOSSES:-base,neg_incl}"
SEEDS="${SEEDS:-42,43,44,45,46}"

EXP="plans/cross-model-probe-generalization/03-loss-alpha-sweep"
CELLS="$OUT/cells"
mkdir -p "$CELLS"

# phase 1: cell grid (loss x alpha) on cached acts
python "$EXP/loss_alpha_sweep.py" \
    --acts-dir "$ACTS" --dataset "$DATA" --out "$CELLS" \
    --layers "$LAYERS" --alphas "$ALPHAS" --losses "$LOSSES" --seeds "$SEEDS" \
    --epochs 30 --n-gpus 1 --gpu-id 0

# phase 2: aggregate
python "$EXP/aggregate_loss_alpha.py" --cell-dir "$CELLS" \
    --out "$OUT/metrics_loss_alpha.json" --model "$MODEL"
