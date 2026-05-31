#!/bin/bash
# [ai-generated] Sequential exp-02 layer-sweep orchestrator for both models.
# debug-qos allows ONE submitted job -> run models sequentially; resume on the
# 1:30 walltime cap (extract/train are idempotent); retry submit past transient
# QOSMaxSubmitJobPerUserLimit. Launch with nohup so it survives ssh drops.
set -uo pipefail
WORK=$HOME/scratch/probes
EXP=$WORK/repo/plans/cross-model-probe-generalization/02-layer-sweep-gemma3-27b
MODELS=("google/gemma-3-27b-it" "Qwen/Qwen2.5-Coder-32B-Instruct")

ts() { date '+%Y-%m-%d %H:%M:%S'; }

for M in "${MODELS[@]}"; do
  SLUG=$(printf '%s' "$M" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')
  RUN="$WORK/runs/layersweep_$SLUG"
  FINAL="$RUN/metrics_layersweep.json"
  echo "[$(ts)] ===== $M (slug=$SLUG) ====="
  for attempt in 1 2 3 4 5 6; do
    if [ -f "$FINAL" ]; then echo "[$(ts)] [$SLUG] FINAL present -> done"; break; fi
    echo "[$(ts)] [$SLUG] attempt $attempt: submitting layersweep"
    JID=""
    for s in $(seq 1 8); do
      JID=$(MODEL="$M" bash "$EXP/submit_layersweep.sh" 2>/dev/null | tail -1)
      [[ "$JID" =~ ^[0-9]+$ ]] && break
      echo "[$(ts)] [$SLUG]   submit retry $s (got '${JID:-empty}'); sleep 30"; sleep 30
    done
    if ! [[ "$JID" =~ ^[0-9]+$ ]]; then echo "[$(ts)] [$SLUG] submit FAILED, abort model"; break; fi
    echo "[$(ts)] [$SLUG] job $JID submitted; polling squeue"
    while squeue -j "$JID" -h 2>/dev/null | grep -q .; do sleep 30; done
    echo "[$(ts)] [$SLUG] job $JID left queue"
    sleep 5
  done
  if [ -f "$FINAL" ]; then echo "[$(ts)] [$SLUG] COMPLETE -> $FINAL"; else echo "[$(ts)] [$SLUG] INCOMPLETE after retries"; fi
done
echo "[$(ts)] orchestrator finished"
