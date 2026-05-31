#!/bin/bash
# [ai-generated]
# Richer-probe sweep on ONE model, as one debug job. Tests whether an MLP head
# and/or layer-concat features beat the linear@single-layer probe on held-out
# example-AUC. Reuses the cached per-layer activation memmaps from exp 02
# (runs/layersweep_<slug>/acts) -- NO extraction, NO model load. α=1, neg_incl
# off, max-pool, 5 group-clean splits. GPU-sharded over the cell grid; then
# aggregate. Resumable: re-submit to continue (finished cell_*.json are skipped).
#
#   MODEL=google/gemma-3-27b-it FEATURESETS="9,19,26,61;19;17,19,22" ./submit_richer.sh
#   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct FEATURESETS="34,41,52,63;41;40,41,43" ./submit_richer.sh
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
WALLTIME=${WALLTIME:-01:30:00}
MODEL=${MODEL:-google/gemma-3-27b-it}
FEATURESETS=${FEATURESETS:-"9,19,26,61;19;17,19,22"}
ARCHS=${ARCHS:-linear,mlp256,mlp512}
SEEDS=${SEEDS:-42,43,44,45,46}
ALPHA=${ALPHA:-1.0}
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')

EXP="$REPO/plans/cross-model-probe-generalization/04-richer-probes"
ACTS="$WORK/runs/layersweep_$SLUG/acts"          # reuse exp-02 activations
RUN="$WORK/runs/richer_$SLUG"
CELLS="$RUN/cells"; FINAL="$RUN/metrics_richer.json"
mkdir -p "$RUN" "$WORK/logs"
[ -f "$ACTS/DONE_EXTRACT" ] || { echo "[richer] ERROR: no cached acts at $ACTS (run exp-02 extract first)" >&2; exit 1; }
SB="$RUN/richer.sbatch"

# NOTE: FEATURESETS contains ';' -- it is quoted everywhere and the srun body is
# a SINGLE-quoted heredoc so the compute-node shell never splits on ';'. The
# value is expanded into the body by THIS shell (login node) at heredoc-write
# time, then passed to python as one quoted arg.
cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=richer-$SLUG
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

    echo "[richer] phase 1: cell grid across 4 GPUs (feature_sets=$FEATURESETS archs=$ARCHS alpha=$ALPHA seeds=$SEEDS)"
    # --interleave=all (NOT --membind): a multi-layer concat feature-set is a big
    # host array (~59 GB on the before/after set, transiently ~2x during
    # np.concatenate). --membind pins a worker to its GPU's NUMA node (~115 GB of
    # the 460 GB), so the concat OOM-kills. Interleave spreads it across all NUMA
    # nodes. If 4 concurrent concats still OOM, use the OOM-safe 2-worker 4-node
    # driver in ../orchestration/ (run_4node.sh 4 <wt> 2).
    for g in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=\$g numactl --interleave=all python \$EXP/richer_probe_sweep.py \
            --acts-dir \$ACTS --dataset \$DATA --out \$CELLS \
            --feature-sets "$FEATURESETS" --archs "$ARCHS" --seeds $SEEDS \
            --epochs 30 --alpha $ALPHA --n-gpus 4 --gpu-id \$g &
    done
    wait

    echo "[richer] phase 2: aggregate"
    python \$EXP/aggregate_richer.py --cell-dir \$CELLS --out \$FINAL --model $MODEL
'
EOF

JID=$(sbatch --parsable "$SB")
echo "[richer] submitted job $JID  (model=$MODEL  feature_sets=$FEATURESETS  final=$FINAL)" >&2
echo "$JID"
