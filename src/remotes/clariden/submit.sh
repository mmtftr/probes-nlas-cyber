#!/bin/bash
# [ai-generated]
# Chunk the roster into groups of 4 (one model per GPU on a node) and submit one
# single-node debug job per chunk. Node-minute budget = 90 => 1 node x 90 min.
# Idempotent: run_model.sh skips models already marked DONE, so re-running
# submit.sh after failures only re-runs the gaps. Unlimited jobs allowed.
#   ./submit.sh [models.txt] [walltime]   (run on the clariden login node)
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
REPO=${REPO:-$WORK/repo}
HERE="$REPO/src/remotes/clariden"
MODELS_FILE="${1:-$HERE/models.txt}"
WALLTIME="${2:-01:30:00}"
ACCOUNT="${SBATCH_ACCOUNT:-lsaie-ss26}"
SBATCH_DIR="$WORK/runs/sbatch"
mkdir -p "$SBATCH_DIR" "$WORK/logs"

mapfile -t MODELS < <(grep -vE '^\s*#|^\s*$' "$MODELS_FILE" | awk '{print $1}')
echo "[submit] ${#MODELS[@]} models from $MODELS_FILE" >&2

i=0; chunk=0
while [ $i -lt ${#MODELS[@]} ]; do
    GROUP=("${MODELS[@]:$i:4}")
    chunk=$((chunk + 1)); i=$((i + 4))
    SB="$SBATCH_DIR/sweep-chunk${chunk}.sbatch"
    # Build the per-GPU launch lines.
    LAUNCH=""
    g=0
    for m in "${GROUP[@]}"; do
        LAUNCH+="    numactl --membind=$g bash \$REPO/src/remotes/clariden/run_model.sh $m $g &\n"
        g=$((g + 1))
    done
    cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=probe-sweep-c${chunk}
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
srun -lu --environment=alps3 --cpus-per-task \$SLURM_CPUS_PER_TASK bash -c '
    source \$REPO/src/remotes/clariden/env.sh
$(printf "%b" "$LAUNCH")    wait
'
EOF
    JID=$(sbatch --parsable "$SB")
    echo "[submit] chunk $chunk -> job $JID  models: ${GROUP[*]}" >&2
done
