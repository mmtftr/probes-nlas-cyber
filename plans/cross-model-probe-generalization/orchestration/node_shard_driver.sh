#!/bin/bash
# [ai-generated] One node's slice of a 4-node, 8-GPU-per-model run. A model's
# grid is split across 2 nodes: this node runs its 4 local GPUs as logical
# shards [BASE..BASE+3] of NGPUS total (sweeps shard by `i % n_gpus == gpu_id`,
# decoupled from the physical CUDA device). MODE=grid runs the sharded sweep;
# MODE=agg runs the single aggregate after all shards finish.
#   args: <MODEL> <EXP 3|4|5> <NGPUS> <SHARD_BASE> <grid|agg>
set -uo pipefail
MODEL="$1"; EXP="$2"; NGPUS="$3"; BASE="$4"; MODE="$5"; NWORKERS="${6:-4}"
source "$REPO/src/remotes/clariden/env.sh"; set +e
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')
DATA="$WORK/data/dataset.jsonl"
ACTS="$WORK/runs/layersweep_$SLUG/acts"
PLANS="$REPO/plans/cross-model-probe-generalization"
SEEDS=42,43,44,45,46
[ -f "$ACTS/DONE_EXTRACT" ] || { echo "[$SLUG] NO acts at $ACTS"; exit 1; }
case "$SLUG" in
  google_gemma-3-27b-it)            LAYERS="2,4,11,20";   FSETS="2,4,11,20;20;18,20,22"; LAYER=20 ;;
  Qwen_Qwen2.5-Coder-32B-Instruct)  LAYERS="20,43,45,49"; FSETS="20,43,45,49;43;41,43,45"; LAYER=43 ;;
  *) echo "[$SLUG] unknown model"; exit 1 ;;
esac

# Run the given sweep command on this node's 4 local GPUs as shards BASE..BASE+3.
# Uses "$@" so args with embedded ';' (FSETS) stay one token.
# NUMA policy: default pins each worker to its GPU's NUMA node (fast, local), but
# that caps host RAM at ~1/4 of the node (~115GB). Memory-heavy cells (exp-04's
# multi-layer concat ~59GB, transiently ~118GB) need the full node RAM -> set
# NUMA=interleave to spread allocations across all NUMA nodes (~460GB).
grid() {
  for loc in $(seq 0 $((NWORKERS-1))); do
    local gid=$((BASE+loc))
    local nc="numactl --membind=$loc"
    [ "${NUMA:-membind}" = interleave ] && nc="numactl --interleave=all"
    CUDA_VISIBLE_DEVICES=$loc $nc "$@" --n-gpus "$NGPUS" --gpu-id "$gid" &
  done
  wait
}

case "$EXP" in
  3) E="$PLANS/03-loss-alpha-sweep"; R="$WORK/runs/lossalpha_$SLUG"; mkdir -p "$R/cells"
     if [ "$MODE" = grid ]; then
       echo "[$SLUG][03] grid shards $BASE..$((BASE+3))/$NGPUS layers=$LAYERS"
       grid python "$E/loss_alpha_sweep.py" --acts-dir "$ACTS" --dataset "$DATA" --out "$R/cells" \
            --layers "$LAYERS" --alphas 1,5,10,20,50 --losses base,neg_incl --seeds "$SEEDS" --epochs 30
     else
       python "$E/aggregate_loss_alpha.py" --cell-dir "$R/cells" --out "$R/metrics_loss_alpha.json" --model "$MODEL" \
         && echo "[$SLUG][03] agg OK -> $R/metrics_loss_alpha.json"
     fi ;;
  4) E="$PLANS/04-richer-probes"; R="$WORK/runs/richer_$SLUG"; mkdir -p "$R/cells"
     if [ "$MODE" = grid ]; then
       echo "[$SLUG][04] grid shards $BASE..$((BASE+NWORKERS-1))/$NGPUS fsets=$FSETS (NUMA interleave)"
       NUMA=interleave grid python "$E/richer_probe_sweep.py" --acts-dir "$ACTS" --dataset "$DATA" --out "$R/cells" \
            --feature-sets "$FSETS" --archs linear,mlp256,mlp512 --seeds "$SEEDS" --epochs 30 --alpha 1.0
     else
       python "$E/aggregate_richer.py" --cell-dir "$R/cells" --out "$R/metrics_richer.json" --model "$MODEL" \
         && echo "[$SLUG][04] agg OK -> $R/metrics_richer.json"
     fi ;;
  5) E="$PLANS/05-probe-vs-verbalized"; R="$WORK/runs/verbalized_$SLUG"; mkdir -p "$R"
     if [ "$MODE" = grid ]; then
       echo "[$SLUG][05] verbalized shards $BASE..$((BASE+3))/$NGPUS (loads model)"
       grid python "$E/verbalized_judge.py" --model "$MODEL" --pairs "$DATA" --out "$R" --max-length 2048
     else
       python "$E/compare_probe_vs_verbalized.py" --acts-dir "$ACTS" --dataset "$DATA" --scores-glob "$R" \
            --layer "$LAYER" --alpha 1.0 --seeds "$SEEDS" --epochs 30 --out "$R/metrics_verbalized.json" \
         && echo "[$SLUG][05] agg OK -> $R/metrics_verbalized.json"
     fi ;;
  *) echo "unknown EXP=$EXP"; exit 1 ;;
esac
