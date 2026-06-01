#!/bin/bash
# [ai-generated] Sweep 6: per-lang/per-CWE tokens_code breakdown for ALL 8 models
# at their 06 best layers, in ONE single-node debug job (cheap: one layer/model,
# cached acts). Writes runs/breakdown_<slug>.json per model + touches BREAKDOWN_DONE.
set -euo pipefail
WORK=${WORK:-$HOME/scratch/probes}; REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}; WALLTIME=${1:-00:30:00}
P="$REPO/plans/cross-model-probe-generalization/06-honest-metric-sweeps"
MODELS="google/gemma-3-1b-it google/gemma-3-1b-pt google/gemma-3-4b-it google/gemma-3-4b-pt google/gemma-3-12b-it google/gemma-3-12b-pt google/gemma-3-27b-it Qwen/Qwen2.5-Coder-32B-Instruct"
SB="$WORK/runs/breakdown.sbatch"; mkdir -p "$WORK/logs"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=breakdown06
#SBATCH --output=$WORK/logs/%x-%j.log
#SBATCH --error=$WORK/logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=$WALLTIME
#SBATCH --no-requeue
set -uo pipefail
export WORK=$WORK REPO=$REPO
srun -lu --mpi=pmi2 --environment=alps3 --cpus-per-task \$SLURM_CPUS_PER_TASK bash -c '
  source \$REPO/src/remotes/clariden/env.sh; set +e
  DATA=\$WORK/data/dataset.jsonl; SPLIT=\$WORK/data/sven_split_meta.json
  rm -f \$WORK/runs/BREAKDOWN_DONE
  for M in $MODELS; do
    SLUG=\$(printf "%s" "\$M" | tr "/" "_" | tr -c "A-Za-z0-9._-" "_")
    MET=\$WORK/runs/layersweep_\$SLUG/metrics_layersweep.json
    BEST=\$(grep -m1 "\"best_layer\":" \$MET | grep -oE "[0-9]+" | head -1)
    echo "[breakdown] \$SLUG best=\$BEST"
    CUDA_VISIBLE_DEVICES=0 numactl --membind=0 python \$REPO/plans/cross-model-probe-generalization/06-honest-metric-sweeps/breakdown_lang_cwe.py \
      --acts-dir \$WORK/runs/layersweep_\$SLUG/acts --dataset \$DATA --split \$SPLIT \
      --layer \$BEST --out \$WORK/runs/breakdown_\$SLUG.json --model "\$M" --epochs 30
  done
  touch \$WORK/runs/BREAKDOWN_DONE
'
EOF
JID=$(sbatch --parsable "$SB")
echo "[breakdown] submitted $JID" >&2; echo "$JID"
