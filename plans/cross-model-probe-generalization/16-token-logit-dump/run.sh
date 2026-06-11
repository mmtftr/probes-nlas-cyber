#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

# Per-model layer bands bracket each model's historical best layer; dump_logits
# re-selects the operating point within the band.
MODELS="${MODELS:-Qwen/Qwen2.5-Coder-32B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct google/gemma-3-1b-it google/gemma-3-4b-it google/gemma-3-12b-it google/gemma-3-12b-pt google/gemma-3-27b-it}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
OUT="${OUT:-./results}"

EXTRACT="src/data/extract_token_activations.py"
DUMP="plans/cross-model-probe-generalization/16-token-logit-dump/dump_logits.py"
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

layers_for(){
    case "$1" in
        Qwen/Qwen2.5-Coder-32B-Instruct) echo "23,24,25,26,27" ;;
        Qwen/Qwen2.5-Coder-7B-Instruct)  echo "14,16,18,20,22" ;;
        google/gemma-3-1b-it)            echo "23,24,25" ;;
        google/gemma-3-4b-it)            echo "5,6,7,8,9" ;;
        google/gemma-3-12b-it)           echo "13,14,15,16,17" ;;
        google/gemma-3-12b-pt)           echo "11,12,13,14,15" ;;
        google/gemma-3-27b-it)           echo "18,19,20" ;;
        *)                               echo "" ;;
    esac
}

for MODEL in $MODELS; do
    SLUG="$(slug "$MODEL")"
    RUN="$OUT/logitdump_$SLUG"; ACTS="$RUN/token_activations"
    mkdir -p "$RUN"
    LAYERS="$(layers_for "$MODEL")"
    LAYER_ARG=(); [ -n "$LAYERS" ] && LAYER_ARG=(--layers "$LAYERS")

    # 1) extract per-token hidden states at the probe layer band
    python "$EXTRACT" \
        --model "$MODEL" --pairs "$DATA" \
        --out "$ACTS" --max-length 2048 "${LAYER_ARG[@]}"

    # 2) train probe + dump every per-token/per-example logit + verify AUC
    python "$DUMP" \
        --model "$MODEL" --acts "$ACTS" --dataset "$DATA" \
        --split "$SPLIT" --out "$RUN"
done
