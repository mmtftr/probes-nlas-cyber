[ai-generated]

# exp-12: MLP layer sweep — the TRUE MLP ceiling (Tier-4 #8)

## Aim

Find the honest MLP ceiling. exp-09 ran the MLP head ONLY at the LINEAR-selected
best layer; the MLP's own optimal layer may differ. Sweep the MLP head over ALL
layers per model, val-select by `tokens_code_auc`, and report the MLP's true
best-layer TEST `tokens_code` — the honest MLP ceiling vs the linear ceiling.

Hypothesis: if the MLP's val-selected layer ≠ the linear-best layer, the
exp-09 MLP numbers under-state the MLP's true ceiling.

## Inputs

- **Models (4):** `Qwen/Qwen2.5-Coder-32B-Instruct`, `google/gemma-3-27b-it`,
  `Qwen/Qwen3-32B`, `Qwen/Qwen3.6-27B`. n_layers ≈ 62–64 each. Sweep visits
  EVERY layer (no `:layer` suffix).
- **Activations (cached, NO re-extraction):** `runs/layersweep_<slug>/acts/` —
  all-layer float32 memmaps `layer_NN.npy` + `meta.json` (n_layers, hidden,
  model) + `offsets.npz` + `y.npy` + `example_ids.npy`. Produced by exp-06's
  extractor; reused read-only.
- **Dataset / split:** `$WORK/data/dataset.jsonl`,
  `$WORK/data/sven_split_meta.json` (persisted seeded 20% group hold-out =
  test). VAL = a further group-aware 15% of TRAIN, `VAL_SEED=42` — identical
  carve to exp-06/exp-09.
- **Head:** `MLPProbe(in_dim, hidden=H)` from `src.training.train_probe_spanmax`,
  built via `_factory_for(head)` and passed as `probe_factory` to
  `train_one_layer(..., epochs=30)`. `HEAD=mlp256` default; `mlp512` second pass.

## Outputs

Per model (and per HEAD):

- `runs/mlp_sweep_<slug>/layers_<HEAD>/layer_{NN}.json` — resumable, one per
  layer. Fields: `head`, `val_tokens_code_auc` (selection), `tokens_code_auc` /
  `tokens_auc` (TEST), `test_tok_auc`, `dropped_fraction`, `val_ex_auc` /
  `test_ex_auc`, counts.
- `runs/mlp_sweep_<slug>/metrics_<HEAD>.json` — aggregate carrying `head`,
  `best_layer` (val-selected), `best_tokens_code_auc` (the TRUE MLP ceiling),
  `best_tokens_auc`, `oracle_tokens_code_*` (test upper bound, val-vs-oracle gap
  only), `baseline_auc`, full per-layer `layers` list.

## Result format

One row per model:

| model | linear best-L / test tc (exp-06) | MLP-at-linear-L (exp-09) | MLP best-L / test tc (this sweep) | MLP L ≠ linear L? |
|---|---|---|---|---|
| Qwen2.5-Coder-32B-Instruct | L25 / 0.788 | 0.788 | _val-L_ / _tc_ | _?_ |
| gemma-3-27b-it | L19 / 0.770 | 0.814 | _val-L_ / _tc_ | _?_ |
| Qwen3-32B | _(from exp-06 once run)_ | 0.789 | _val-L_ / _tc_ | _?_ |
| Qwen3.6-27B | _(from exp-06 once run)_ | 0.795 | _val-L_ / _tc_ | _?_ |

- Linear ceilings confirmed from exp-06 metrics: Qwen2.5 = L25/0.788,
  gemma-3-27b = L19/0.770. Qwen3-32B / Qwen3.6-27B linear sweeps not yet in
  `06-honest-metric-sweeps/` — fill from their exp-06 `metrics_layersweep` when
  available.
- exp-09 "MLP at linear-best-layer" reference: Qwen2.5 0.788, Qwen3-32B 0.789,
  Qwen3.6 0.795, gemma 0.814.
- Also report the MLP val-vs-oracle gap (`best_tokens_code_auc` vs
  `oracle_tokens_code_auc`) per model.

## Interpretation hints

- **MLP best-L ≠ linear-L AND MLP test tc > exp-09 number** → the exp-09 MLP
  was layer-handicapped; the MLP's true ceiling is higher than reported. The
  honest MLP-vs-linear gap widens.
- **MLP best-L = linear-L (or tc ≈ exp-09)** → exp-09 already found the MLP's
  best operating point; the linear layer is also the MLP's layer. No hidden
  headroom; the exp-09 conclusion stands.
- **MLP test tc ≈ linear ceiling even at the MLP's own best layer** → the gain
  from the nonlinear head is small; the property is close to linearly decodable
  and a single direction nearly saturates it.
- **Large val-vs-oracle gap for the MLP** → MLP layer selection is unstable /
  the val signal is noisy for this head; treat the val-selected ceiling as a
  conservative lower bound.

## For agents

MLPProbe import path (confirmed by reading the source):
`from src.training.train_probe_spanmax import MLPProbe, train_one_layer`.
MLP returns `w=None, b=None`; the trained torch module is always in `r["probe"]`,
so scoring uses the module forward `torch.sigmoid(probe(X))` — NOT the closed-form
`X@w+b` that exp-06 uses for the linear head.

Per-model CLI (run inside `submit_12.sh` via srun; `<slug>` =
`MODEL | tr '/' '_'`, `<HEAD>` = mlp256 default):

```bash
# phase 2 — 4-GPU shard
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python 12-mlp-layer-sweep/train_all_layers_mlp.py \
    --acts-dir   $WORK/runs/layersweep_<slug>/acts \
    --dataset    $WORK/data/dataset.jsonl \
    --split      $WORK/data/sven_split_meta.json \
    --out        $WORK/runs/mlp_sweep_<slug>/layers_mlp256 \
    --head mlp256 --epochs 30 --n-gpus 4 --gpu-id $g &
done; wait

# phase 3 — aggregate
python 12-mlp-layer-sweep/aggregate_mlp_sweep.py \
  --acts-dir  $WORK/runs/layersweep_<slug>/acts \
  --dataset   $WORK/data/dataset.jsonl \
  --split     $WORK/data/sven_split_meta.json \
  --layer-dir $WORK/runs/mlp_sweep_<slug>/layers_mlp256 \
  --out       $WORK/runs/mlp_sweep_<slug>/metrics_mlp256.json
```

Submit one job per model: `MODEL=<m> HEAD=mlp256 ./submit_12.sh`. SBATCH header
mirrors `06/submit_layersweep.sh` verbatim; the orchestrator overrides
partition/qos/time at submit time. `mlp512` is a second pass (separate
`layers_mlp512/` + `metrics_mlp512.json`).

`TODO(adhoc-decision)`: per-model MLP-sweep orchestrator (analog of
`06/run_honest_sweep_orch.sh`) is NOT written — the spec said the orchestrator
submits one job per model and overrides partition/qos/time. Add a roster loop
over the 4 models × {mlp256, mlp512} if/when sequential submission is wanted.
