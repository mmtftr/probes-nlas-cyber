#!/bin/bash
# [ai-generated] Submit ONE 2-node debug job that runs the given experiments for
# BOTH models concurrently — gemma on node 0, qwen on node 1 — via two
# backgrounded, node-pinned srun steps in a single allocation. This is the
# legitimate way around debug-qos MaxJobsPerUser=1 (it's one job using 2 nodes).
# BUDGET: debug allows <= 1.5 node-hours/job, so with 2 nodes walltime <= 00:45:00.
#   args: <EXPS> <WALLTIME>   e.g.  run_2node.sh "3" 00:45:00  |  run_2node.sh "4 5" 00:35:00
set -euo pipefail
WORK=${WORK:-$HOME/scratch/probes}; REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
EXPS="${1:?exps e.g. \"3\" or \"4 5\"}"; WALLTIME="${2:?HH:MM:SS}"
NAME="2n_$(echo "$EXPS" | tr -d ' ')"
SB="$WORK/runs/${NAME}.sbatch"; mkdir -p "$WORK/logs" "$WORK/runs"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=$NAME
#SBATCH --output=$WORK/logs/%x-%j.log
#SBATCH --error=$WORK/logs/%x-%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=$WALLTIME
#SBATCH --no-requeue
set -uo pipefail
export WORK=$WORK REPO=$REPO
nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))
echo "[2node] nodes=\${nodes[*]}  exps='$EXPS'  start=\$(date)"
srun --nodes=1 --ntasks=1 --nodelist=\${nodes[0]} --exclusive --gpus-per-node=4 \\
     --cpus-per-task=288 -lu --mpi=pmi2 --environment=alps3 \\
     bash $WORK/node_exp_driver.sh google/gemma-3-27b-it $EXPS &
PID0=\$!
srun --nodes=1 --ntasks=1 --nodelist=\${nodes[1]} --exclusive --gpus-per-node=4 \\
     --cpus-per-task=288 -lu --mpi=pmi2 --environment=alps3 \\
     bash $WORK/node_exp_driver.sh Qwen/Qwen2.5-Coder-32B-Instruct $EXPS &
PID1=\$!
wait \$PID0; R0=\$?
wait \$PID1; R1=\$?
echo "[2node] gemma rc=\$R0  qwen rc=\$R1  end=\$(date)"
EOF

JID=$(sbatch --parsable "$SB")
echo "[2node] submitted $JID name=$NAME nodes=2 walltime=$WALLTIME (node-h=$(python3 -c "h,m,s='$WALLTIME'.split(':');print(round(2*(int(h)+int(m)/60+int(s)/3600),3))")) exps='$EXPS'" >&2
echo "$JID"
