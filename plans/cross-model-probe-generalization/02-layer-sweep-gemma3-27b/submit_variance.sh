#!/bin/bash
# [ai-generated]
# Repeated-split variance for the per-layer AUC curve, one model, ONE debug job:
#   phase 1: extract ALL layers to float32 memmaps (1 GPU) -- IDEMPOTENT, skips
#            if acts/DONE_EXTRACT exists (so a cached sweep is reused for free;
#            a cleaned one is re-extracted).
#   phase 2: per layer, retrain the span-max probe under K group-clean splits
#            (--seeds), fanned across 4 GPUs.
#   phase 3: aggregate per-layer JSONs -> mean+/-std + per-seed baselines.
# Reuses runs/layersweep_$SLUG/acts; writes variance outputs to a SEPARATE
# layers_var/ + metrics_variance.json so the single-split sweep is untouched.
# Resumable: re-submit to continue (extract + finished layer_var files skipped).
#   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct ./submit_variance.sh   (login node)
#   ./submit_variance.sh            (defaults to google/gemma-3-27b-it)
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
WALLTIME=${WALLTIME:-01:30:00}
MODEL=${MODEL:-google/gemma-3-27b-it}
SEEDS=${SEEDS:-42,43,44,45,46}
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')

EXP="$REPO/plans/cross-model-probe-generalization/02-layer-sweep-gemma3-27b"
RUN="$WORK/runs/layersweep_$SLUG"
ACTS="$RUN/acts"; LAYERDIR="$RUN/layers_var"; FINAL="$RUN/metrics_variance.json"
mkdir -p "$RUN" "$WORK/logs"
SB="$RUN/variance.sbatch"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=variance-$SLUG
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
    EXP=$EXP; ACTS=$ACTS; LAYERDIR=$LAYERDIR; FINAL=$FINAL; SEEDS=$SEEDS

    echo "[variance] phase 1: extract all layers (idempotent)"
    CUDA_VISIBLE_DEVICES=0 numactl --membind=0 python \$EXP/extract_all_layers.py \
        --model $MODEL --pairs \$DATA --out \$ACTS --max-length 2048

    echo "[variance] phase 2: K-split retrain per layer across 4 GPUs (seeds=\$SEEDS)"
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=\$g numactl --membind=\$g python \$EXP/splits_variance.py \
            --acts-dir \$ACTS --dataset \$DATA --out \$LAYERDIR \
            --seeds \$SEEDS --epochs 30 --n-gpus 4 --gpu-id \$g &
    done
    wait

    echo "[variance] phase 3: aggregate mean+/-std"
    python \$EXP/aggregate_variance.py --acts-dir \$ACTS --dataset \$DATA \
        --layer-dir \$LAYERDIR --out \$FINAL --seeds \$SEEDS
'
EOF

JID=$(sbatch --parsable "$SB")
echo "[variance] submitted job $JID  (model=$MODEL  acts=$ACTS  final=$FINAL)" >&2
echo "$JID"
