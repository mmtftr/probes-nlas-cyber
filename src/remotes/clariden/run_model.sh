#!/bin/bash
# [ai-generated]
# Per-model driver: extract token activations -> train span-max probe per layer
# -> eval vs baselines. Idempotent (skips if DONE), GPU-pinned. Run inside the
# alps3 container after sourcing env.sh.
#   run_model.sh <hf_model_id> <gpu_id>
set -uo pipefail

MODEL_ID="$1"; GPU_ID="${2:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

SLUG="$(printf '%s' "$MODEL_ID" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')"
OUT="$RUNS/$SLUG"
ACTS="$OUT/token_activations"
LOG="$OUT/run.log"
mkdir -p "$OUT"

if [ -f "$OUT/DONE" ]; then echo "[run_model] $SLUG already DONE, skipping" >&2; exit 0; fi
rm -f "$OUT/FAILED"
echo "[run_model] $(date) model=$MODEL_ID gpu=$GPU_ID slug=$SLUG" | tee "$LOG" >&2

PYBIN="$(command -v python || command -v python3)"

# 1) Extract per-token hidden states (auto-selects {n/4,n/2,3n/4,n-1} layers).
"$PYBIN" "$REPO/src/data/extract_token_activations.py" \
    --model "$MODEL_ID" --pairs "$DATA/dataset.jsonl" \
    --out "$ACTS" --max-length 2048 >>"$LOG" 2>&1 || { echo "EXTRACT_FAILED" >"$OUT/FAILED"; exit 1; }

# 2) Train span-max probe per layer + eval vs baselines on the held-out SVEN split.
"$PYBIN" "$REPO/src/remotes/clariden/train_eval.py" \
    --model "$MODEL_ID" --acts "$ACTS" --dataset "$DATA/dataset.jsonl" \
    --split "$DATA/sven_split_meta.json" --out "$OUT" >>"$LOG" 2>&1 \
    || { echo "TRAINEVAL_FAILED" >"$OUT/FAILED"; exit 1; }

touch "$OUT/DONE"
echo "[run_model] $(date) DONE $SLUG" | tee -a "$LOG" >&2
