#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
BEST="${BEST:-25}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
ACTS="${ACTS:-./acts}"            # reuse cached acts
OUT="${OUT:-./results}"
KS="${KS:-1 2 4 8}"
AGGS="${AGGS:-max logsumexp softmax_gate}"
TAU="${TAU:-1.0}"
GATE_MODE="${GATE_MODE:-per_token}"
HEADS="${HEADS:-linear mlp256 mlp512}"

P="plans/cross-model-probe-generalization/09-ensemble-linear-probes"
CELLS="$OUT/cells"
mkdir -p "$CELLS"

# baseline reference heads (linear floor + MLP ceiling) at the best layer
for H in $HEADS; do
    python "$P/train_head_baseline.py" \
        --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" --best-layer "$BEST" \
        --head "$H" --out "$CELLS/head_${H}.json" --model "$MODEL" --epochs 30
done

# K-ensemble grid {K x agg}
for K in $KS; do
    for AGG in $AGGS; do
        python "$P/train_ensemble.py" \
            --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" \
            --best-layer "$BEST" --K "$K" --agg "$AGG" --tau "$TAU" --gate-mode "$GATE_MODE" \
            --out "$CELLS/K${K}_${AGG}.json" --model "$MODEL" --epochs 30
    done
done

# cosine-divergence-loss sweep (K=8 x {logsumexp,max} x lambda grid)
CELLS_DIV="$OUT/cells_div"
mkdir -p "$CELLS_DIV"
for AGG in logsumexp max; do
    for LAM in 0 0.001 0.01 0.1 0.3; do
        python "$P/train_ensemble.py" \
            --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" --best-layer "$BEST" \
            --K 8 --agg "$AGG" --div-lambda "$LAM" \
            --out "$CELLS_DIV/K8_${AGG}_lam${LAM}.json" --model "$MODEL" --epochs 30
    done
done

# aggregate {K x agg} vs K=1 baseline
python "$P/aggregate_ensemble.py" --cells-dir "$CELLS" \
    --out "$OUT/summary.json" --markdown "$OUT/summary.md"
