#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

# model:best_layer pairs.
PAIRS="${PAIRS:-Qwen/Qwen2.5-Coder-32B-Instruct:25 google/gemma-3-27b-it:19 Qwen/Qwen3-32B:27 Qwen/Qwen3.6-27B:30}"
DATA="${DATA:-./data/dataset.jsonl}"
ACTS_ROOT="${ACTS_ROOT:-./acts}"   # per-model cached acts: $ACTS_ROOT/<slug>/acts
OUT="${OUT:-./results}"
MAXLEN="${MAXLEN:-2048}"
SEEDS="${SEEDS:-42,43,44,45,46}"
COMBINE="${COMBINE:-max}"

EXP14="plans/cross-model-probe-generalization/14-memory-prompt-sweep"
EXP15="plans/cross-model-probe-generalization/15-ensemble-comparison"
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

for pair in $PAIRS; do
    M="${pair%%:*}"; BEST="${pair##*:}"
    SLUG="$(slug "$M")"
    PSWEEP="$OUT/promptsweep_$SLUG"   # verbalized shards (shared with exp-14)
    RUN="$OUT/ensemble15_$SLUG"
    ACTS="$ACTS_ROOT/$SLUG/acts"
    mkdir -p "$PSWEEP" "$RUN"
    printf '%s\n' "$BEST" > "$RUN/best_layer.txt"

    # (i) verbalized: all prompt variants (LOADS MODEL)
    python "$EXP14/prompt_variants_judge.py" \
        --model "$M" --pairs "$DATA" --out "$PSWEEP" \
        --max-length "$MAXLEN" --n-gpus 1 --gpu-id 0

    # (ii) probe side: member scores on cached acts at the best layer
    PROBE_OUT="$RUN/probe_member_scores.json"
    python "$EXP15/probe_members_scorer.py" \
        --acts-dir "$ACTS" --dataset "$DATA" --best-layer "$BEST" \
        --seeds "$SEEDS" --model "$M" --out "$PROBE_OUT"

    # (iii) matrix: merge verbalized shards + probe scores (CPU)
    python "$EXP15/build_matrix.py" \
        --probe-scores "$PROBE_OUT" --promptsweep-dir "$PSWEEP" \
        --dataset "$DATA" --seeds "$SEEDS" --combine "$COMBINE" \
        --model "$M" --out "$RUN/ensemble15_${SLUG}_matrix.json"
done
