#!/bin/bash
# [ai-generated] Sequential exp-06 honest-metric layer-sweep orchestrator over the
# full roster. debug-qos allows ONE submitted job -> run models sequentially;
# resume on the walltime cap (extract/train idempotent); retry submit past
# transient QOSMaxSubmitJobPerUserLimit. Launch with nohup so it survives ssh
# drops:  cd ~/scratch/probes && nohup bash repo/plans/.../06-.../run_honest_sweep_orch.sh \
#           > logs/honest_orch.log 2>&1 &
#
# Roster order: smallest/fastest first (fail fast), anchors last. Gemma -pt give
# the base<->instruct (Q5) contrast. A model whose FINAL never appears after the
# retries is logged and SKIPPED (e.g. gated-access denied) — the rest proceed.
set -uo pipefail
WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
SUBMIT="$REPO/plans/cross-model-probe-generalization/06-honest-metric-sweeps/submit_layersweep.sh"
WALLTIME=${WALLTIME:-01:30:00}

MODELS=(
  "google/gemma-3-1b-it"
  "google/gemma-3-1b-pt"
  "google/gemma-3-4b-it"
  "google/gemma-3-4b-pt"
  "google/gemma-3-12b-it"
  "google/gemma-3-12b-pt"
  "google/gemma-3-27b-it"
  "Qwen/Qwen2.5-Coder-32B-Instruct"
)

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
      JID=$(MODEL="$M" WALLTIME="$WALLTIME" bash "$SUBMIT" "$WALLTIME" 2>/dev/null | tail -1)
      [[ "$JID" =~ ^[0-9]+$ ]] && break
      echo "[$(ts)] [$SLUG]   submit retry $s (got '${JID:-empty}'); sleep 30"; sleep 30
    done
    if ! [[ "$JID" =~ ^[0-9]+$ ]]; then echo "[$(ts)] [$SLUG] submit FAILED, abort model"; break; fi
    echo "[$(ts)] [$SLUG] job $JID submitted; polling squeue"
    while squeue -j "$JID" -h 2>/dev/null | grep -q .; do sleep 30; done
    echo "[$(ts)] [$SLUG] job $JID left queue"
    sleep 5
  done
  if [ -f "$FINAL" ]; then echo "[$(ts)] [$SLUG] COMPLETE -> $FINAL"; else echo "[$(ts)] [$SLUG] INCOMPLETE after retries (skipping)"; fi
done
echo "[$(ts)] honest-sweep orchestrator finished"
touch "$WORK/runs/HONEST_ORCH_DONE"
