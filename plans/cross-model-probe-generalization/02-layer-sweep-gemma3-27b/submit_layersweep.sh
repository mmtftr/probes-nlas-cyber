#!/bin/bash
# [ai-generated]
# Submit the full per-layer sweep for Gemma-3-27B as ONE single-node debug job:
#   phase 1: extract all 62 layers to float16 memmaps on scratch (1 GPU)
#   phase 2: train one span-max probe per layer, 62 layers fanned across 4 GPUs
#   phase 3: aggregate per-layer JSONs + baselines -> metrics_layersweep.json
# Idempotent / resumable: extract skips if DONE_EXTRACT exists; training skips
# any layer_{NN}.json already written. Re-submit to continue after a timeout.
#   ./submit_layersweep.sh            (run on the clariden login node)
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
WALLTIME=${1:-01:30:00}
MODEL=google/gemma-3-27b-it

EXP="$REPO/plans/cross-model-probe-generalization/02-layer-sweep-gemma3-27b"
RUN="$WORK/runs/layersweep_gemma3-27b"
ACTS="$RUN/acts"; LAYERDIR="$RUN/layers"; FINAL="$RUN/metrics_layersweep.json"
mkdir -p "$RUN" "$WORK/logs"
SB="$RUN/layersweep.sbatch"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=layersweep-g3-27b
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
        --model $MODEL --pairs \$DATA --out \$ACTS --max-length 1024

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
