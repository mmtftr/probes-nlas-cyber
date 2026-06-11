#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-27b-it}"
HEAD="${HEAD:-mlp256}"            # mlp256 | mlp512
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
ACTS="${ACTS:-./acts}"            # reuse cached all-layer acts (exp-06)
OUT="${OUT:-./results}"

EXP12="plans/cross-model-probe-generalization/12-mlp-layer-sweep"
LAYERDIR="$OUT/layers_$HEAD"
FINAL="$OUT/metrics_$HEAD.json"
mkdir -p "$OUT"

# phase 2: train per-layer MLP probes (no extraction; reuse cached acts)
python "$EXP12/train_all_layers_mlp.py" \
    --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" --out "$LAYERDIR" \
    --head "$HEAD" --epochs 30 --n-gpus 1 --gpu-id 0

# phase 3: aggregate (val-selected best + oracle)
python "$EXP12/aggregate_mlp_sweep.py" --acts-dir "$ACTS" --dataset "$DATA" \
    --split "$SPLIT" --layer-dir "$LAYERDIR" --out "$FINAL"
