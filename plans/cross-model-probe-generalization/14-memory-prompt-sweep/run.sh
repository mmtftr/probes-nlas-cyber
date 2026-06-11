#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

# model:best_layer pairs (best_layer recorded for provenance; not load-bearing here).
PAIRS="${PAIRS:-Qwen/Qwen2.5-Coder-32B-Instruct:25 google/gemma-3-27b-it:19 Qwen/Qwen3-32B:27 Qwen/Qwen3.6-27B:30}"
DATA="${DATA:-./data/dataset.jsonl}"
OUT="${OUT:-./results}"
MAXLEN="${MAXLEN:-2048}"

EXP="plans/cross-model-probe-generalization/14-memory-prompt-sweep"
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

for pair in $PAIRS; do
    M="${pair%%:*}"; BEST="${pair##*:}"
    SLUG="$(slug "$M")"
    RUN="$OUT/promptsweep_$SLUG"
    mkdir -p "$RUN"
    printf '%s\n' "$BEST" > "$RUN/best_layer.txt"

    # verbalized P(yes) for all prompt variants (LOADS MODEL)
    python "$EXP/prompt_variants_judge.py" \
        --model "$M" --pairs "$DATA" --out "$RUN" \
        --max-length "$MAXLEN" --n-gpus 1 --gpu-id 0

    # analyze: memory + injection example-AUC per variant over the 5-seed splits (CPU)
    python "$EXP/analyze_prompt_sweep.py" \
        --run-dir "$RUN" --dataset "$DATA" --model "$M" \
        --out "$RUN/promptsweep_${SLUG}_aucs.json"
done
