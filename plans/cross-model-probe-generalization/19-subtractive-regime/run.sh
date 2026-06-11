#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODELS="${MODELS:-Qwen/Qwen2.5-Coder-32B-Instruct Qwen/Qwen2.5-Coder-7B-Instruct google/gemma-3-1b-it google/gemma-3-4b-it google/gemma-3-12b-it google/gemma-3-12b-pt google/gemma-3-27b-it}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
OUT="${OUT:-./results}"

EXTRACT="src/data/extract_token_activations.py"
HERE="plans/cross-model-probe-generalization/19-subtractive-regime"
MEMBERSHIP="$HERE/subtractive_membership.json"
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

# layer band (for the 8-config grid) and single operating layer (for CV).
band_for(){
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
oplayer_for(){
    case "$1" in
        Qwen/Qwen2.5-Coder-32B-Instruct) echo "25" ;;
        Qwen/Qwen2.5-Coder-7B-Instruct)  echo "16" ;;
        google/gemma-3-1b-it)            echo "25" ;;
        google/gemma-3-4b-it)            echo "7"  ;;
        google/gemma-3-12b-it)           echo "15" ;;
        google/gemma-3-12b-pt)           echo "13" ;;
        google/gemma-3-27b-it)           echo "19" ;;
        *)                               echo "" ;;
    esac
}
# vLLM EAGLE3 aux-hidden for qwen, HF otherwise.
backend_for(){ case "$1" in Qwen/*) echo vllm ;; *) echo hf ;; esac; }

for MODEL in $MODELS; do
    SLUG="$(slug "$MODEL")"

    # --- subtractive grid (extract band) ---
    GRID_OUT="$OUT/subtractive_$SLUG"; GRID_ACTS="$GRID_OUT/token_activations"
    mkdir -p "$GRID_OUT"
    BAND="$(band_for "$MODEL")"
    BAND_ARG=(); [ -n "$BAND" ] && BAND_ARG=(--layers "$BAND")
    python "$EXTRACT" \
        --model "$MODEL" --pairs "$DATA" \
        --out "$GRID_ACTS" --max-length 2048 "${BAND_ARG[@]}"
    python "$HERE/train_grid.py" \
        --model "$MODEL" --acts "$GRID_ACTS" --dataset "$DATA" \
        --split "$SPLIT" --membership "$MEMBERSHIP" --out "$GRID_OUT"

    # --- CV variance (extract single operating layer, keep acts) ---
    CV_OUT="$OUT/cv_$SLUG"; CV_ACTS="$CV_OUT/token_activations"
    mkdir -p "$CV_OUT"
    OPLAYER="$(oplayer_for "$MODEL")"
    OP_ARG=(); [ -n "$OPLAYER" ] && OP_ARG=(--layers "$OPLAYER")
    BACKEND="$(backend_for "$MODEL")"
    python "$EXTRACT" \
        --model "$MODEL" --pairs "$DATA" --backend "$BACKEND" \
        --out "$CV_ACTS" --max-length 2048 "${OP_ARG[@]}"
    python "$HERE/train_grid_cv.py" \
        --model "$MODEL" --acts "$CV_ACTS" --dataset "$DATA" \
        --membership "$MEMBERSHIP" --out "$CV_OUT" --folds 5 --seeds 1,2,3
done
