#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

# model:best_layer pairs to steer.
PAIRS="${PAIRS:-Qwen/Qwen2.5-Coder-32B-Instruct:25 google/gemma-3-27b-it:19 Qwen/Qwen3-32B:27 Qwen/Qwen3.6-27B:30}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
ACTS_ROOT="${ACTS_ROOT:-./acts}"   # per-model: $ACTS_ROOT/<slug>/acts
OUT="${OUT:-./results}"
ALPHAS="${ALPHAS:--4 -2 -1 0 1 2 4}"
NPER="${NPER:-40}"
MAXLEN="${MAXLEN:-2048}"

EXP="plans/cross-model-probe-generalization/13-causal-steering"
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

# causal steering: train pooled family directions, hook layer L, alpha-sweep P(yes)
for pair in $PAIRS; do
    M="${pair%%:*}"; BEST="${pair##*:}"
    SLUG="$(slug "$M")"
    ACTS="$ACTS_ROOT/$SLUG/acts"
    python "$EXP/steer_judge.py" \
        --model "$M" --best-layer "$BEST" --acts-dir "$ACTS" \
        --dataset "$DATA" --split "$SPLIT" --out "$OUT/steer_13_${SLUG}.json" \
        --n-per-subset "$NPER" --max-length "$MAXLEN" \
        --alphas $ALPHAS
done

# aggregation diagnostic (analysis-only) for the two anchor models
python "$EXP/aggregation_diag.py" \
    --acts-dir "$ACTS_ROOT/Qwen_Qwen2.5-Coder-32B-Instruct/acts" \
    --dataset "$DATA" --layer 25 --model Qwen2.5-Coder-32B-Instruct \
    --out "$OUT/aggdiag_qwen25coder32b.json"
python "$EXP/aggregation_diag.py" \
    --acts-dir "$ACTS_ROOT/google_gemma-3-27b-it/acts" \
    --dataset "$DATA" --layer 19 --model gemma-3-27b-it \
    --out "$OUT/aggdiag_gemma27b.json"
