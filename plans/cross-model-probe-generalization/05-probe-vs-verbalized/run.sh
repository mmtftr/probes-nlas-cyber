#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-google/gemma-3-27b-it}"
DATA="${DATA:-./data/dataset.jsonl}"
OUT="${OUT:-./results}"
ACTS="${ACTS:-./acts}"            # reuse exp-02 activations (probe side)
LAYER="${LAYER:-19}"
ALPHA="${ALPHA:-1.0}"
SEEDS="${SEEDS:-42,43,44,45,46}"
MAXLEN="${MAXLEN:-2048}"

EXP="plans/cross-model-probe-generalization/05-probe-vs-verbalized"
mkdir -p "$OUT"

# phase 1: verbalized P(yes) (LOADS MODEL)
python "$EXP/verbalized_judge.py" \
    --model "$MODEL" --pairs "$DATA" --out "$OUT" \
    --max-length "$MAXLEN" --n-gpus 1 --gpu-id 0

# phase 2: probe vs. verbalized comparison (cached acts, no model)
python "$EXP/compare_probe_vs_verbalized.py" \
    --acts-dir "$ACTS" --dataset "$DATA" --scores-glob "$OUT" \
    --layer "$LAYER" --alpha "$ALPHA" --seeds "$SEEDS" --epochs 30 \
    --out "$OUT/metrics_verbalized.json"
