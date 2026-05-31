#!/bin/bash
# [ai-generated]
# Submit the full per-layer sweep for one model as ONE single-node debug job:
#   phase 1: extract ALL layers to float32 memmaps on scratch (1 GPU)
#   phase 2: train one span-max probe per layer, fanned across 4 GPUs
#   phase 3: aggregate per-layer JSONs + baselines -> metrics_layersweep.json
# Idempotent / resumable: extract skips if DONE_EXTRACT exists; training skips
# any layer_{NN}.json already written. Re-submit to continue after a timeout.
#   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct ./submit_layersweep.sh   (login node)
#   ./submit_layersweep.sh            (defaults to google/gemma-3-27b-it)
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
WALLTIME=${1:-01:30:00}
MODEL=${MODEL:-google/gemma-3-27b-it}
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')

EXP="$REPO/plans/cross-model-probe-generalization/02-layer-sweep-gemma3-27b"
RUN="$WORK/runs/layersweep_$SLUG"
ACTS="$RUN/acts"; LAYERDIR="$RUN/layers"; FINAL="$RUN/metrics_layersweep.json"
mkdir -p "$RUN" "$WORK/logs"
SB="$RUN/layersweep.sbatch"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=layersweep-$SLUG
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
    EXP=$EXP; ACTS=$ACTS; LAYERDIR=$LAYERDIR; FINAL=$FINAL

    echo "[layersweep] phase 1: extract all layers"
    CUDA_VISIBLE_DEVICES=0 numactl --membind=0 python \$EXP/extract_all_layers.py \
        --model $MODEL --pairs \$DATA --out \$ACTS --max-length 2048

    echo "[layersweep] phase 2: train per-layer probes across 4 GPUs"
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=\$g numactl --membind=\$g python \$EXP/train_all_layers.py \
            --acts-dir \$ACTS --dataset \$DATA --split \$SPLIT --out \$LAYERDIR \
            --epochs 30 --n-gpus 4 --gpu-id \$g &
    done
    wait

    echo "[layersweep] phase 3: aggregate"
    python \$EXP/aggregate_layersweep.py --acts-dir \$ACTS --dataset \$DATA \
        --split \$SPLIT --layer-dir \$LAYERDIR --out \$FINAL
'
EOF

JID=$(sbatch --parsable "$SB")
echo "[layersweep] submitted job $JID  (acts=$ACTS  final=$FINAL)" >&2
echo "$JID"
