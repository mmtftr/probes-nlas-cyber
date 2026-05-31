#!/bin/bash
# [ai-generated]
# Loss-variant x alpha sweep on the best layers of ONE model, as one debug job.
# Reuses the cached per-layer activation memmaps from exp 02
# (runs/layersweep_<slug>/acts) -- NO extraction, NO model load, just linear
# probes on cached activations. GPU-sharded over the cell grid; then aggregate.
# Resumable: re-submit to continue (finished cell_*.json are skipped).
#
#   MODEL=google/gemma-3-27b-it LAYERS=10,19,26,61 ./submit_loss_alpha.sh
#   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct LAYERS=34,41,48,52 ./submit_loss_alpha.sh
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
WALLTIME=${WALLTIME:-01:30:00}
MODEL=${MODEL:-google/gemma-3-27b-it}
LAYERS=${LAYERS:-10,19,26,61}
ALPHAS=${ALPHAS:-1,5,10,20,50}
LOSSES=${LOSSES:-base,neg_incl}
SEEDS=${SEEDS:-42,43,44,45,46}
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')

EXP="$REPO/plans/cross-model-probe-generalization/03-loss-alpha-sweep"
ACTS="$WORK/runs/layersweep_$SLUG/acts"          # reuse exp-02 activations
RUN="$WORK/runs/lossalpha_$SLUG"
CELLS="$RUN/cells"; FINAL="$RUN/metrics_loss_alpha.json"
mkdir -p "$RUN" "$WORK/logs"
[ -f "$ACTS/DONE_EXTRACT" ] || { echo "[loss-sweep] ERROR: no cached acts at $ACTS (run exp-02 extract first)" >&2; exit 1; }
SB="$RUN/lossalpha.sbatch"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=lossalpha-$SLUG
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
    EXP=$EXP; ACTS=$ACTS; CELLS=$CELLS; FINAL=$FINAL

    echo "[loss-sweep] phase 1: cell grid across 4 GPUs (layers=$LAYERS alphas=$ALPHAS losses=$LOSSES seeds=$SEEDS)"
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=\$g numactl --membind=\$g python \$EXP/loss_alpha_sweep.py \
            --acts-dir \$ACTS --dataset \$DATA --out \$CELLS \
            --layers $LAYERS --alphas $ALPHAS --losses $LOSSES --seeds $SEEDS \
            --epochs 30 --n-gpus 4 --gpu-id \$g &
    done
    wait

    echo "[loss-sweep] phase 2: aggregate"
    python \$EXP/aggregate_loss_alpha.py --cell-dir \$CELLS --out \$FINAL --model $MODEL
'
EOF

JID=$(sbatch --parsable "$SB")
echo "[loss-sweep] submitted job $JID  (model=$MODEL  layers=$LAYERS  final=$FINAL)" >&2
echo "$JID"
