#!/bin/bash
# [ai-generated] Chain the 4-node experiment tail: wait for the already-running
# exp-03 job, then run exp-04 and exp-05 as 4-node jobs (each model split across
# 2 nodes / 8 GPUs). Resumable: each stage re-submits until BOTH models' metrics
# exist (grids skip finished cells). debug = 1 running job, so this serializes.
#   args: <running_exp03_jobid>
set -uo pipefail
WORK=$HOME/scratch/probes
RUN03=$1
ts(){ date '+%H:%M:%S'; }
GSLUG=google_gemma-3-27b-it; QSLUG=Qwen_Qwen2.5-Coder-32B-Instruct

metric_for(){ case "$1" in 3) echo "lossalpha/metrics_loss_alpha.json";; 4) echo "richer/metrics_richer.json";; 5) echo "verbalized/metrics_verbalized.json";; esac; }
wait_job(){ while squeue -j "$1" -h 2>/dev/null | grep -q .; do sleep 30; done; }

# 0) wait for the in-flight exp-03 job
echo "[$(ts)] waiting on running exp-03 job $RUN03"
wait_job "$RUN03"
echo "[$(ts)] exp-03 job ended"

for stage in 3 4 5; do
  rel=$(metric_for $stage); pfx=${rel%%/*}; m=${rel#*/}
  wt=00:22:00; [ $stage = 4 ] && wt=00:15:00; [ $stage = 5 ] && wt=00:15:00
  for attempt in 1 2 3; do
    if [ -f "$WORK/runs/${pfx}_$GSLUG/$m" ] && [ -f "$WORK/runs/${pfx}_$QSLUG/$m" ]; then
      echo "[$(ts)] exp-0$stage both metrics present -> done"; break
    fi
    echo "[$(ts)] exp-0$stage attempt $attempt: submit 4-node ($wt)"
    JID=""
    for s in 1 2 3 4 5; do JID=$(bash "$WORK/run_4node.sh" "$stage" "$wt" 2>/dev/null | tail -1); [[ "$JID" =~ ^[0-9]+$ ]] && break; echo "  submit retry $s"; sleep 30; done
    [[ "$JID" =~ ^[0-9]+$ ]] || { echo "[$(ts)] exp-0$stage submit FAILED"; break; }
    echo "[$(ts)] exp-0$stage job $JID running"; wait_job "$JID"; echo "[$(ts)] exp-0$stage job $JID ended"; sleep 5
  done
done
echo "[$(ts)] tail orchestrator finished"
