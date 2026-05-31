#!/bin/bash
# [ai-generated] Run exp 03/04/05 inner pipelines for ONE model on THIS node's
# 4 GPUs. No sbatch — invoked via `srun -N1 --environment=alps3` inside a 2-node
# allocation (one model per node, concurrent). Reuses cached exp-02 acts.
# Inner python invocations copied verbatim from the per-exp submit_*.sh scripts;
# only the sbatch/srun wrapper is dropped. Best-layer params are the NEW exp-02
# derivations (do NOT reuse the archived old layers).
#   args: <MODEL_ID> <EXP...>   e.g.  google/gemma-3-27b-it 3        |  Qwen/... 4 5
set -uo pipefail
MODEL="$1"; shift
EXPS="$*"
source "$REPO/src/remotes/clariden/env.sh"
set +e  # env.sh uses -e; we want to manage failures per-exp below
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')
DATA="$WORK/data/dataset.jsonl"
ACTS="$WORK/runs/layersweep_$SLUG/acts"
PLANS="$REPO/plans/cross-model-probe-generalization"
SEEDS=42,43,44,45,46
[ -f "$ACTS/DONE_EXTRACT" ] || { echo "[$SLUG] NO cached acts at $ACTS"; exit 1; }

# NEW per-model best-layer params (re-derived from the before/after exp-02 sweep).
case "$SLUG" in
  google_gemma-3-27b-it)
    LAYERS="2,4,11,20";   FSETS="2,4,11,20;20;18,20,22"; LAYER=20 ;;
  Qwen_Qwen2.5-Coder-32B-Instruct)
    LAYERS="20,43,45,49"; FSETS="20,43,45,49;43;41,43,45"; LAYER=43 ;;
  *) echo "[$SLUG] unknown model — no layer params"; exit 1 ;;
esac

run_3() {  # exp-03 loss x alpha (cached acts, linear probes)
  local E="$PLANS/03-loss-alpha-sweep" R="$WORK/runs/lossalpha_$SLUG"; mkdir -p "$R/cells"
  echo "[$SLUG][03] grid layers=$LAYERS"
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g numactl --membind=$g python "$E/loss_alpha_sweep.py" \
      --acts-dir "$ACTS" --dataset "$DATA" --out "$R/cells" \
      --layers "$LAYERS" --alphas 1,5,10,20,50 --losses base,neg_incl --seeds "$SEEDS" \
      --epochs 30 --n-gpus 4 --gpu-id $g &
  done; wait
  python "$E/aggregate_loss_alpha.py" --cell-dir "$R/cells" --out "$R/metrics_loss_alpha.json" --model "$MODEL" \
    && echo "[$SLUG][03] OK -> $R/metrics_loss_alpha.json"
}
run_4() {  # exp-04 richer probes (cached acts; linear + mlp heads, layer concat)
  local E="$PLANS/04-richer-probes" R="$WORK/runs/richer_$SLUG"; mkdir -p "$R/cells"
  echo "[$SLUG][04] grid fsets=$FSETS"
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g numactl --membind=$g python "$E/richer_probe_sweep.py" \
      --acts-dir "$ACTS" --dataset "$DATA" --out "$R/cells" \
      --feature-sets "$FSETS" --archs linear,mlp256,mlp512 --seeds "$SEEDS" \
      --epochs 30 --alpha 1.0 --n-gpus 4 --gpu-id $g &
  done; wait
  python "$E/aggregate_richer.py" --cell-dir "$R/cells" --out "$R/metrics_richer.json" --model "$MODEL" \
    && echo "[$SLUG][04] OK -> $R/metrics_richer.json"
}
run_5() {  # exp-05 probe vs verbalized (LOADS MODEL for the P(yes) forward)
  local E="$PLANS/05-probe-vs-verbalized" R="$WORK/runs/verbalized_$SLUG"; mkdir -p "$R"
  echo "[$SLUG][05] verbalized judge (loads model) layer=$LAYER"
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g numactl --membind=$g python "$E/verbalized_judge.py" \
      --model "$MODEL" --pairs "$DATA" --out "$R" --max-length 2048 --n-gpus 4 --gpu-id $g &
  done; wait
  python "$E/compare_probe_vs_verbalized.py" --acts-dir "$ACTS" --dataset "$DATA" \
    --scores-glob "$R" --layer $LAYER --alpha 1.0 --seeds "$SEEDS" --epochs 30 --out "$R/metrics_verbalized.json" \
    && echo "[$SLUG][05] OK -> $R/metrics_verbalized.json"
}

rc=0
for e in $EXPS; do
  echo "===== [$SLUG] exp-0$e start $(date '+%H:%M:%S') ====="
  run_$e || { echo "[$SLUG][0$e] FAILED"; rc=1; break; }
done
echo "===== [$SLUG] driver finished (rc=$rc) for exps: $EXPS ====="
exit $rc
