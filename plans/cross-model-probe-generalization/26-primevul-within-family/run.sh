#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
LAYER="${LAYER:-16}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
# reuses the kept exp-22 PrimeVul/SVEN operating-layer acts.
PV_ACTS="${PV_ACTS:-./acts/primevul/pv_acts}"
SVEN_ACTS="${SVEN_ACTS:-./acts/primevul/sven_acts}"
PV_DATASET="${PV_DATASET:-plans/cross-model-probe-generalization/22-primevul-paired/primevul_dataset.jsonl}"
OUT="${OUT:-./results}"

PVFAMILY="plans/cross-model-probe-generalization/26-primevul-within-family/pv_family.py"
mkdir -p "$OUT"

# CPU training (PrimeVul acts won't fit a single GPU). mode=both ->
# within-PV transfer matrix + SVEN<->PV shared-CWE.
export CUDA_VISIBLE_DEVICES=""
export BOOT_JOBS="${BOOT_JOBS:-64}"   # threads for the parallel rank-AUC bootstrap

python "$PVFAMILY" --model "$MODEL" --layer "$LAYER" \
    --pv-acts "$PV_ACTS" --pv-dataset "$PV_DATASET" \
    --sven-acts "$SVEN_ACTS" --sven-dataset "$DATA" \
    --sven-split "$SPLIT" \
    --out "$OUT" --mode both
