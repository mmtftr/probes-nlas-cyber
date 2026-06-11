#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-27b-it}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
OUT="${OUT:-./results}"
SEEDS="${SEEDS:-42,43,44,45,46}"

EXP="plans/cross-model-probe-generalization/02-layer-sweep-gemma3-27b"
ACTS="$OUT/acts"
LAYERDIR="$OUT/layers"
LAYERDIR_VAR="$OUT/layers_var"
mkdir -p "$OUT"

# --- single-split per-layer sweep ---
# phase 1: extract all layers
python "$EXP/extract_all_layers.py" \
    --model "$MODEL" --pairs "$DATA" --out "$ACTS" --max-length 2048

# phase 2: train one span-max probe per layer
python "$EXP/train_all_layers.py" \
    --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" --out "$LAYERDIR" \
    --epochs 30 --n-gpus 1 --gpu-id 0

# phase 3: aggregate
python "$EXP/aggregate_layersweep.py" --acts-dir "$ACTS" --dataset "$DATA" \
    --split "$SPLIT" --layer-dir "$LAYERDIR" --out "$OUT/metrics_layersweep.json"

# --- repeated-split variance (reuses the same acts) ---
python "$EXP/splits_variance.py" \
    --acts-dir "$ACTS" --dataset "$DATA" --out "$LAYERDIR_VAR" \
    --seeds "$SEEDS" --epochs 30 --n-gpus 1 --gpu-id 0

python "$EXP/aggregate_variance.py" --acts-dir "$ACTS" --dataset "$DATA" \
    --layer-dir "$LAYERDIR_VAR" --out "$OUT/metrics_variance.json" --seeds "$SEEDS"
