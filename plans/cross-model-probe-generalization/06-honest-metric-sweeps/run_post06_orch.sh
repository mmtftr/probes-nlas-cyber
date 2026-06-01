#!/bin/bash
# [ai-generated] Sequential post-06 orchestrator: exp-07 (all 8 models) + exp-03/04
# (anchors) at each model's 06 best layer. debug-qos = one job; run sequentially,
# retry past QOS limit, resume on walltime cap. Launch with nohup.
set -uo pipefail
WORK=${WORK:-$HOME/scratch/probes}; REPO=${REPO:-$WORK/repo}
SUBMIT="$REPO/plans/cross-model-probe-generalization/06-honest-metric-sweeps/submit_post06.sh"
MODELS=(
  "google/gemma-3-1b-it" "google/gemma-3-1b-pt"
  "google/gemma-3-4b-it" "google/gemma-3-4b-pt"
  "google/gemma-3-12b-it" "google/gemma-3-12b-pt"
  "google/gemma-3-27b-it" "Qwen/Qwen2.5-Coder-32B-Instruct"
)
ts() { date '+%Y-%m-%d %H:%M:%S'; }
for M in "${MODELS[@]}"; do
  SLUG=$(printf '%s' "$M" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')
  DONEFILE="$WORK/runs/codemask_$SLUG/metrics.json"  # 07 output = per-model completion marker
  echo "[$(ts)] ===== $M ====="
  for attempt in 1 2 3 4 5; do
    [ -f "$DONEFILE" ] && { echo "[$(ts)] [$SLUG] codemask metrics present -> done"; break; }
    JID=""
    for s in $(seq 1 8); do
      JID=$(MODEL="$M" bash "$SUBMIT" 2>/dev/null | tail -1)
      [[ "$JID" =~ ^[0-9]+$ ]] && break
      echo "[$(ts)] [$SLUG] submit retry $s; sleep 30"; sleep 30
    done
    [[ "$JID" =~ ^[0-9]+$ ]] || { echo "[$(ts)] [$SLUG] submit FAILED"; break; }
    echo "[$(ts)] [$SLUG] job $JID; polling"
    while squeue -j "$JID" -h 2>/dev/null | grep -q .; do sleep 30; done
    sleep 5
  done
  [ -f "$DONEFILE" ] && echo "[$(ts)] [$SLUG] COMPLETE" || echo "[$(ts)] [$SLUG] INCOMPLETE (skipping)"
done
echo "[$(ts)] post-06 orchestrator finished"; touch "$WORK/runs/POST06_DONE"
