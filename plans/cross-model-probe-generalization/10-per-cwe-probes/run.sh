#!/usr/bin/env bash
# [ai-generated] Generic reproduction command (de-clustered).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct}"
BEST="${BEST:-25}"
DATA="${DATA:-./data/dataset.jsonl}"
SPLIT="${SPLIT:-./data/sven_split_meta.json}"
ACTS="${ACTS:-./acts}"            # reuse cached acts
OUT="${OUT:-./results}"
HEAD="${HEAD:-linear}"            # linear | mlp
NEG_POOL="${NEG_POOL:-all}"       # all | same_lang | same_family
CWE="${CWE:-ALL}"                 # ALL | <CWE-id> | injection | memory
SPEC_HEADS="${SPEC_HEADS:-linear mlp256}"
SPEC_SETS="${SPEC_SETS:-family both}"
REF_MLP="${REF_MLP:-mlp512}"

P="plans/cross-model-probe-generalization/10-per-cwe-probes"
mkdir -p "$OUT"
SLUG="$(printf '%s' "$MODEL" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')"

# per-CWE specialized probes vs the general probe
python "$P/per_cwe_probe.py" \
    --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" \
    --best-layer "$BEST" --cwe "$CWE" --head "$HEAD" --neg-pool "$NEG_POOL" \
    --out "$OUT/per_cwe_${SLUG}_${HEAD}_${NEG_POOL}_${CWE}.json" --model "$MODEL" --epochs 30

# post-hoc ensemble of specialist probes vs general-linear floor / general-MLP ceiling
for SH in $SPEC_HEADS; do
    for SS in $SPEC_SETS; do
        python "$P/posthoc_ensemble.py" \
            --acts-dir "$ACTS" --dataset "$DATA" --split "$SPLIT" --best-layer "$BEST" \
            --spec-head "$SH" --spec-set "$SS" --ref-mlp "$REF_MLP" \
            --out "$OUT/posthoc_${SLUG}_${SH}_${SS}.json" --model "$MODEL" --epochs 30
    done
done
