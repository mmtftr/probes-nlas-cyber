#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

# Honest per-layer sweep (tokens_code_auc) over the model roster, then the
# per-lang/per-CWE breakdown at each model's val-selected best layer.
MODELS="${MODELS:-google/gemma-3-1b-it google/gemma-3-1b-pt google/gemma-3-4b-it google/gemma-3-4b-pt google/gemma-3-12b-it google/gemma-3-12b-pt google/gemma-3-27b-it Qwen/Qwen2.5-Coder-32B-Instruct}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
OUT="${OUT:-./results}"

PLANS="plans/cross-model-probe-generalization"
EXP02="$PLANS/02-layer-sweep-gemma3-27b"   # extractor (unchanged)
EXP06="$PLANS/06-honest-metric-sweeps"     # honest train + aggregate
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

for MODEL in $MODELS; do
    SLUG="$(slug "$MODEL")"
    RUN="$OUT/layersweep_$SLUG"
    ACTS="$RUN/acts"; LAYERDIR="$RUN/layers"; FINAL="$RUN/metrics_layersweep.json"
    mkdir -p "$RUN"

    # phase 1: extract all layers (reuses exp-02 extractor)
    python "$EXP02/extract_all_layers.py" \
        --model "$MODEL" --pairs "$DATA" --out "$ACTS" --max-length 2048

    # phase 2: train per-layer probes (honest metric)
    python "$EXP06/train_all_layers.py" \
        --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" --out "$LAYERDIR" \
        --epochs 30 --n-gpus 1 --gpu-id 0

    # phase 3: aggregate (val-selected best + oracle)
    python "$EXP06/aggregate_layersweep.py" --acts-dir "$ACTS" --dataset "$DATA" \
        --split "$SPLIT" --layer-dir "$LAYERDIR" --out "$FINAL"

    # per-lang/per-CWE tokens_code breakdown at the best layer
    BEST="$(grep -m1 '"best_layer":' "$FINAL" | grep -oE '[0-9]+' | head -1)"
    python "$EXP06/breakdown_lang_cwe.py" \
        --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" \
        --layer "$BEST" --out "$RUN/breakdown_$SLUG.json" --model "$MODEL" --epochs 30
done
