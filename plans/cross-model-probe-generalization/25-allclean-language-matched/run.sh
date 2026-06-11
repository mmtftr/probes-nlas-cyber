#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

# model:repo_layer pairs; reuses the kept full-SVEN L25 token activations.
PAIRS="${PAIRS:-Qwen/Qwen2.5-Coder-32B-Instruct:25 google/gemma-3-1b-it:25}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
ACTS_ROOT="${ACTS_ROOT:-./acts}"   # per-model acts: $ACTS_ROOT/<slug>/token_activations
OUT="${OUT:-./results}"
MODE="${MODE:-main}"               # main (decisive eval) | cv (grouped 5-fold x 3-seed)

DECONFOUND="plans/cross-model-probe-generalization/25-allclean-language-matched/deconfound.py"
mkdir -p "$OUT"

PYFLAGS=(); [ "$MODE" = "cv" ] && PYFLAGS=(--cv-only --do-cv)

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

for pair in $PAIRS; do
    M="${pair%%:*}"; L="${pair##*:}"
    SLUG="$(slug "$M")"
    ACTS="$ACTS_ROOT/$SLUG/token_activations"
    RUN="$OUT/$SLUG"
    mkdir -p "$RUN"

    python "$DECONFOUND" --model "$M" --acts "$ACTS" \
        --dataset "$DATA" --split "$SPLIT" \
        --layer "$L" --out "$RUN" "${PYFLAGS[@]}"
done
