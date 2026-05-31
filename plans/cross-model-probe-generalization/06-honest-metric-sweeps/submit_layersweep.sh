#!/bin/bash
# [ai-generated]
# Submit the HONEST per-layer sweep (exp-06) for one model as ONE single-node
# debug job:
#   phase 1: extract ALL layers to float32 memmaps (reuses exp-02's extractor,
#            unchanged — it already saves offsets.npz needed for the code mask)
#   phase 2: train one span-max probe per layer (4 GPUs), recording the honest
#            tokens_code_auc alongside the inflated tokens_auc (exp-06 scripts)
#   phase 3: aggregate -> metrics_layersweep.json (val-selected best + oracle)
# Idempotent / resumable: extract skips if DONE_EXTRACT exists; training skips
# any layer_{NN}.json already written. Re-submit to continue after a timeout.
#   MODEL=google/gemma-3-1b-it ./submit_layersweep.sh   (login node)
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
WALLTIME=${1:-01:30:00}
MODEL=${MODEL:-google/gemma-3-27b-it}
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')

PLANS="$REPO/plans/cross-model-probe-generalization"
EXP02="$PLANS/02-layer-sweep-gemma3-27b"   # extractor lives here (unchanged)
EXP06="$PLANS/06-honest-metric-sweeps"     # honest train + aggregate
RUN="$WORK/runs/layersweep_$SLUG"
ACTS="$RUN/acts"; LAYERDIR="$RUN/layers"; FINAL="$RUN/metrics_layersweep.json"
mkdir -p "$RUN" "$WORK/logs"
SB="$RUN/layersweep.sbatch"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=layersweep06-$SLUG
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
    SPLIT=\$WORK/data/sven_split_meta.json
    EXP02=$EXP02; EXP06=$EXP06; ACTS=$ACTS; LAYERDIR=$LAYERDIR; FINAL=$FINAL

    echo "[layersweep06] phase 1: extract all layers"
    CUDA_VISIBLE_DEVICES=0 numactl --membind=0 python \$EXP02/extract_all_layers.py \
        --model $MODEL --pairs \$DATA --out \$ACTS --max-length 2048

    echo "[layersweep06] phase 2: train per-layer probes (honest metric) across 4 GPUs"
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=\$g numactl --membind=\$g python \$EXP06/train_all_layers.py \
            --acts-dir \$ACTS --dataset \$DATA --split \$SPLIT --out \$LAYERDIR \
            --epochs 30 --n-gpus 4 --gpu-id \$g &
    done
    wait

    echo "[layersweep06] phase 3: aggregate"
    python \$EXP06/aggregate_layersweep.py --acts-dir \$ACTS --dataset \$DATA \
        --split \$SPLIT --layer-dir \$LAYERDIR --out \$FINAL
'
EOF

JID=$(sbatch --parsable "$SB")
echo "[layersweep06] submitted job $JID  (acts=$ACTS  final=$FINAL)" >&2
echo "$JID"
