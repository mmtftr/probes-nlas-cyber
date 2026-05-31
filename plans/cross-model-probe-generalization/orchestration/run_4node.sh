#!/bin/bash
# [ai-generated] Submit ONE 4-node debug job for a single experiment, splitting
# each model's grid across 2 nodes (8 GPUs/model): gemma on nodes 0,1 and qwen
# on nodes 2,3. Grids run concurrently, then a single aggregate per model.
# BUDGET: 4 nodes x walltime <= 1.5 node-h  =>  walltime <= 00:22:30.
#   args: <EXP 3|4|5> <WALLTIME HH:MM:SS>   e.g.  run_4node.sh 3 00:22:00
set -euo pipefail
WORK=${WORK:-$HOME/scratch/probes}; REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
EXP="${1:?exp 3|4|5}"; WALLTIME="${2:?HH:MM:SS}"
NWORKERS="${3:-4}"          # GPUs used per node (lower => less peak RAM; exp-04 needs 2)
NGPUS=$((2*NWORKERS))       # 2 nodes per model
BASE2=$NWORKERS             # second node's shard base
NAME="4n_e$EXP"
SB="$WORK/runs/${NAME}.sbatch"; mkdir -p "$WORK/logs" "$WORK/runs"
SR="--ntasks=1 --exclusive --gpus-per-node=4 --cpus-per-task=288 -lu --mpi=pmi2 --environment=alps3"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=$NAME
#SBATCH --output=$WORK/logs/%x-%j.log
#SBATCH --error=$WORK/logs/%x-%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=$WALLTIME
#SBATCH --no-requeue
set -uo pipefail
export WORK=$WORK REPO=$REPO
G=google/gemma-3-27b-it; Q=Qwen/Qwen2.5-Coder-32B-Instruct
D=$WORK/node_shard_driver.sh
nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))
echo "[4node] exp=$EXP nodes=\${nodes[*]} start=\$(date)"
# --- grids: each model split across 2 nodes (shards 0-3 and 4-7 of 8) ---
srun --nodes=1 --nodelist=\${nodes[0]} $SR bash \$D \$G $EXP $NGPUS 0 grid $NWORKERS &
srun --nodes=1 --nodelist=\${nodes[1]} $SR bash \$D \$G $EXP $NGPUS $BASE2 grid $NWORKERS &
srun --nodes=1 --nodelist=\${nodes[2]} $SR bash \$D \$Q $EXP $NGPUS 0 grid $NWORKERS &
srun --nodes=1 --nodelist=\${nodes[3]} $SR bash \$D \$Q $EXP $NGPUS $BASE2 grid $NWORKERS &
wait
echo "[4node] grids returned; aggregating \$(date)"
# --- one aggregate per model (cheap) ---
srun --nodes=1 --nodelist=\${nodes[0]} $SR bash \$D \$G $EXP $NGPUS 0 agg $NWORKERS &
srun --nodes=1 --nodelist=\${nodes[2]} $SR bash \$D \$Q $EXP $NGPUS 0 agg $NWORKERS &
wait
echo "[4node] done \$(date)"
EOF

JID=$(sbatch --parsable "$SB")
NH=$(python3 -c "h,m,s='$WALLTIME'.split(':');print(round(4*(int(h)+int(m)/60+int(s)/3600),3))")
echo "[4node] submitted $JID name=$NAME nodes=4 walltime=$WALLTIME node-h=$NH exp=$EXP" >&2
echo "$JID"
