#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

# model:operating_layer pairs (from exp-16).
PAIRS="${PAIRS:-Qwen/Qwen2.5-Coder-7B-Instruct:16 google/gemma-3-12b-it:15}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
OUT="${OUT:-./results}"

EXTRACT="src/data/extract_token_activations.py"
HERE="plans/cross-model-probe-generalization/22-primevul-paired"
PV_DATASET="$HERE/primevul_dataset.jsonl"
SVEN_MEMBERSHIP="$HERE/subtractive_membership.json"
PV_MEMBERSHIP="$HERE/primevul_membership.json"
mkdir -p "$OUT"

slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }

extract(){  # <pairs_jsonl> <out_acts> <model> <layer>
    python "$EXTRACT" --backend hf \
        --model "$3" --pairs "$1" --out "$2" --max-length 2048 --layers "$4"
}

for pair in $PAIRS; do
    M="${pair%%:*}"; L="${pair##*:}"
    SLUG="$(slug "$M")"
    RUN="$OUT/primevul_$SLUG"
    SVEN_ACTS="$RUN/sven_acts"; PV_ACTS="$RUN/pv_acts"
    mkdir -p "$RUN"

    # extract operating-layer token acts for SVEN + PrimeVul (keep acts)
    extract "$DATA"        "$SVEN_ACTS" "$M" "$L"
    extract "$PV_DATASET"  "$PV_ACTS"   "$M" "$L"

    # cross-dataset training (SVEN <-> PrimeVul)
    python "$HERE/train_cross.py" --model "$M" \
        --sven-acts "$SVEN_ACTS" --pv-acts "$PV_ACTS" \
        --sven-dataset "$DATA" \
        --sven-membership "$SVEN_MEMBERSHIP" \
        --sven-split "$SPLIT" \
        --pv-dataset "$PV_DATASET" \
        --pv-membership "$PV_MEMBERSHIP" \
        --layer-sven "$L" --layer-pv "$L" --out "$RUN"
done
