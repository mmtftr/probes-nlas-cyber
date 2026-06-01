#!/bin/bash
# [ai-generated]
# Post-06 sweeps for ONE model as a single-node 4-GPU debug job, at the model's
# 06 val_tokens_code-selected best layer (read from metrics_layersweep.json):
#   - exp-07 codemask_train (paired none vs code_only negatives, 5 seeds) — ALL models
#   - exp-03 loss x alpha (honest tokens_code) + exp-04 richer probes — ANCHORS only
#     (gemma-3-27b-it, Qwen2.5-Coder-32B-Instruct), where 03/04 were designed.
# Reuses cached acts (no extraction). Idempotent: skips outputs that exist.
#   MODEL=google/gemma-3-27b-it ./submit_post06.sh
set -euo pipefail
WORK=${WORK:-$HOME/scratch/probes}; REPO=${REPO:-$WORK/repo}
ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}; WALLTIME=${1:-00:40:00}
MODEL=${MODEL:-google/gemma-3-27b-it}
SLUG=$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')
P="$REPO/plans/cross-model-probe-generalization"
ACTS="$WORK/runs/layersweep_$SLUG/acts"
METRICS="$WORK/runs/layersweep_$SLUG/metrics_layersweep.json"
[ -f "$METRICS" ] || { echo "no 06 metrics for $SLUG"; exit 1; }
BEST=$(grep -m1 '"best_layer":' "$METRICS" | grep -oE '[0-9]+' | head -1)
[ -n "$BEST" ] || { echo "could not read best_layer for $SLUG"; exit 1; }
ANCHOR=0; case "$SLUG" in google_gemma-3-27b-it|Qwen_Qwen2.5-Coder-32B-Instruct) ANCHOR=1;; esac
SEEDS=42,43,44,45,46
SB="$WORK/runs/post06_$SLUG.sbatch"; mkdir -p "$WORK/logs"

cat > "$SB" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=debug
#SBATCH --job-name=post06-$SLUG
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
  ACTS=$ACTS; BEST=$BEST; SLUG=$SLUG; P=$P; SEEDS=$SEEDS

  echo "[post06][\$SLUG] exp-07 codemask_train at layer \$BEST"
  R7=\$WORK/runs/codemask_\$SLUG; mkdir -p \$R7
  CUDA_VISIBLE_DEVICES=0 numactl --membind=0 python \$P/07-train-code-masked-negs/codemask_train.py \
    --acts-dir \$ACTS --dataset \$DATA --split \$SPLIT --layer \$BEST --out \$R7 \
    --model $MODEL --epochs 30 && echo "[post06][\$SLUG][07] OK -> \$R7/metrics.json"

  if [ $ANCHOR = 1 ]; then
    echo "[post06][\$SLUG] exp-03 loss x alpha (honest) at layer \$BEST"
    R3=\$WORK/runs/lossalpha_\$SLUG; mkdir -p \$R3/cells
    for g in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=\$g numactl --membind=\$g python \$P/03-loss-alpha-sweep/loss_alpha_sweep.py \
        --acts-dir \$ACTS --dataset \$DATA --out \$R3/cells --layers \$BEST \
        --alphas 1,5,10,20,50 --losses base,neg_incl --seeds \$SEEDS --epochs 30 --n-gpus 4 --gpu-id \$g &
    done; wait
    python \$P/03-loss-alpha-sweep/aggregate_loss_alpha.py --cell-dir \$R3/cells \
      --out \$R3/metrics_loss_alpha.json --model $MODEL && echo "[post06][\$SLUG][03] OK"

    echo "[post06][\$SLUG] exp-04 richer probes (honest) around layer \$BEST"
    R4=\$WORK/runs/richer_\$SLUG; mkdir -p \$R4/cells
    FSETS="\$BEST;\$((BEST-2)),\$BEST,\$((BEST+2))"
    for g in 0 1 2 3; do
      NUMA=interleave CUDA_VISIBLE_DEVICES=\$g numactl --interleave=all python \$P/04-richer-probes/richer_probe_sweep.py \
        --acts-dir \$ACTS --dataset \$DATA --out \$R4/cells --feature-sets "\$FSETS" \
        --archs linear,mlp256,mlp512 --seeds \$SEEDS --epochs 30 --alpha 1.0 --n-gpus 4 --gpu-id \$g &
    done; wait
    python \$P/04-richer-probes/aggregate_richer.py --cell-dir \$R4/cells \
      --out \$R4/metrics_richer.json --model $MODEL && echo "[post06][\$SLUG][04] OK"
  fi
  echo "[post06][\$SLUG] done"
'
EOF

JID=$(sbatch --parsable "$SB")
echo "[post06] submitted $JID for $SLUG (best layer $BEST, anchor=$ANCHOR)" >&2
echo "$JID"
