#!/bin/bash
# [ai-generated] Unattended overnight orchestrator for the cross-model probe sweep.
# Waits for the smoke job, launches the full roster, and re-submits gaps
# (timeouts / transient download fails), capped per-model so a stuck model
# can't loop forever. Idempotent via run_model.sh DONE markers.
#   nohup bash overnight.sh <smoke_jobid> >> ~/scratch/probes/overnight.log 2>&1 &
set -uo pipefail
WORK="$HOME/scratch/probes"; REPO="$WORK/repo"
MODELS="$REPO/src/remotes/clariden/models.txt"
LOG="$WORK/overnight.log"
SMOKE_JID="${1:-}"
MAX_ATTEMPTS=2
cd "$REPO"
slug(){ printf '%s' "$1" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'; }
log(){ echo "[overnight $(date +%H:%M:%S)] $*" >> "$LOG"; }

log "start; smoke job=$SMOKE_JID"
if [ -n "$SMOKE_JID" ]; then
  while squeue -j "$SMOKE_JID" -h 2>/dev/null | grep -q "$SMOKE_JID"; do sleep 30; done
  log "smoke job cleared"
fi

mapfile -t ALL < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$MODELS" | awk '{print $1}')
log "roster: ${#ALL[@]} models"

for iter in $(seq 1 12); do
  # wait for our jobs to drain before assessing gaps
  while squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -q "probe-sweep"; do sleep 60; done
  todo=()
  for m in "${ALL[@]}"; do
    s="$(slug "$m")"; d="$WORK/runs/$s"
    if [ -f "$d/DONE" ]; then continue; fi
    a=$(cat "$d/attempts" 2>/dev/null || echo 0)
    if [ "$a" -ge "$MAX_ATTEMPTS" ]; then log "give up $m (attempts=$a)"; continue; fi
    mkdir -p "$d"; echo $((a+1)) > "$d/attempts"
    todo+=("$m")
  done
  if [ "${#todo[@]}" -eq 0 ]; then log "iter $iter: no work left -> COMPLETE"; break; fi
  log "iter $iter: submitting ${#todo[@]}: ${todo[*]}"
  printf '%s\n' "${todo[@]}" > "$WORK/.todo_models.txt"
  WORK="$WORK" REPO="$REPO" "$REPO/src/remotes/clariden/submit.sh" "$WORK/.todo_models.txt" 01:30:00 >> "$LOG" 2>&1
  sleep 150
done

{
  echo "=== FINAL $(date) ==="
  ok=0
  for m in "${ALL[@]}"; do
    s="$(slug "$m")"
    if [ -f "$WORK/runs/$s/DONE" ]; then
      echo "DONE  $m"; ok=$((ok+1))
    else
      echo "MISS  $m  $(head -1 "$WORK/runs/$s/FAILED" 2>/dev/null)"
    fi
  done
  echo "completed $ok/${#ALL[@]}"
} >> "$LOG"
log "exiting"
