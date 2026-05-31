#!/bin/bash
# [ai-generated]
# Probe vs. the model's OWN verbalized vulnerability judgment, one debug job/model.
#
# Phase 1 LOADS THE MODEL (unlike exp 03/04, which only touch cached acts) — it
# is the forward-pass half: for each SVEN row, ask "is this code vulnerable?"
# (code BEFORE the question, neutral preamble) and read P(yes). 4-GPU fanout, each
# worker writes its own verbalized_scores.gpu{id}.json shard, resumable.
# Phase 2 reuses the cached per-layer acts (runs/layersweep_<slug>/acts) to train
# the span-max probe and compares its example-AUC to the verbalized AUC on the
# SAME held-out examples — no model load.
#
#   MODEL=google/gemma-3-27b-it LAYER=19 bash .../submit_verbalized.sh
#   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct LAYER=41 bash .../submit_verbalized.sh
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
WALLTIME=${WALLTIME:-01:30:00}
MODEL=${MODEL:-google/gemma-3-27b-it}
LAYER=${LAYER:-19}
ALPHA=${ALPHA:-1.0}
SEEDS=${SEEDS:-42,43,44,45,46}
MAXLEN=${MAXLEN:-2048}
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')

EXP="$REPO/plans/cross-model-probe-generalization/05-probe-vs-verbalized"
ACTS="$WORK/runs/layersweep_$SLUG/acts"          # reuse exp-02 activations (probe side)
RUN="$WORK/runs/verbalized_$SLUG"
SCORES="$RUN"                                     # shards land here
FINAL="$RUN/metrics_verbalized.json"
mkdir -p "$RUN" "$WORK/logs"
[ -f "$ACTS/DONE_EXTRACT" ] || { echo "[verbalized] ERROR: no cached acts at $ACTS (run exp-02 extract first)" >&2; exit 1; }
SB="$RUN/verbalized.sbatch"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=verbalized-$SLUG
#SBATCH --output=$WORK/logs/%x-%j.log
#SBATCH --error=$WORK/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=$WALLTIME
#SBATCH --no-requeue
set -euo pipefail
export WORK=$WORK REPO=$REPO
srun -lu --mpi=pmi2 --environment=alps3 --cpus-per-task \$SLURM_CPUS_PER_TASK bash -c '
    source \$REPO/src/remotes/clariden/env.sh
    DATA=\$WORK/data/dataset.jsonl
    EXP=$EXP; ACTS=$ACTS; RUN=$RUN; SCORES=$SCORES; FINAL=$FINAL

    echo "[verbalized] phase 1: verbalized P(yes) across 4 GPUs (model=$MODEL, LOADS MODEL)"
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=\$g numactl --membind=\$g python \$EXP/verbalized_judge.py \
            --model $MODEL --pairs \$DATA --out \$RUN \
            --max-length $MAXLEN --n-gpus 4 --gpu-id \$g &
    done
    wait

    echo "[verbalized] phase 2: probe vs. verbalized comparison (cached acts, no model)"
    python \$EXP/compare_probe_vs_verbalized.py \
        --acts-dir \$ACTS --dataset \$DATA --scores-glob \$SCORES \
        --layer $LAYER --alpha $ALPHA --seeds $SEEDS --epochs 30 --out \$FINAL
'
EOF

JID=$(sbatch --parsable "$SB")
echo "[verbalized] submitted job $JID  (model=$MODEL  layer=$LAYER  final=$FINAL)" >&2
echo "$JID"
