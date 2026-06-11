[ai-generated]

# 13 — Causal steering: does the memory-family direction DRIVE the stated belief?

Tier-1 causal test built on exp-05 (belief audit) and exp-10 (memory-family
probe). Belief audit asks if the model VERBALIZES memory-vuln. This asks the
CAUSAL question: ADD the memory-family probe direction to the residual stream
and watch the model's own verbalized P("yes, vulnerable"). If P(yes) rises with
+alpha, the direction CAUSALLY drives the stated belief — a MoC-style linear
correction (Marks & Tegmark 2023; ITI, Li et al. 2023; activation steering,
Turner et al. 2023) — not just a correlate the probe reads off.

## 1. Aim

Steering the residual stream along the unit memory-safety probe direction
`w_hat` at the best layer should RAISE the model's verbalized P(yes) on +alpha
(most on memory-family positives) IF the direction is causal. Flat response =>
the direction is an epiphenomenal correlate.

## 2. Inputs

- **Direction:** pooled MEMORY-family linear probe (exp-10 FAMILY map, memory
  positives + all `cwe==null` negatives, VAL groups excluded) trained at the
  model's best layer on **cached acts** `layersweep_<slug>/acts/layer_NN.npy`
  (+ `y.npy`, `example_ids.npy`). Head weight `w` -> unit `w_hat`.
- **Models + best layers:** `Qwen/Qwen2.5-Coder-32B-Instruct:25`,
  `google/gemma-3-27b-it:19`, `Qwen/Qwen3-32B:27`, `Qwen/Qwen3.6-27B:30`.
  Loaded via `_load_model` (CausalLM -> ImageTextToText VLM text-decoder
  fallback for Qwen3.6).
- **Data/split:** `data/dataset.jsonl` (SVEN before/after), seed-42 20% group
  hold-out `sven_split_meta.json` (`load_or_make_split`). Subsets drawn from the
  leakage-free TEST split only.
- **Scale:** `scale` = median over tokens of the L2 norm of the layer-L hidden
  state, from cached acts at layer L (== `hidden_states[L+1]`, the tensor the
  probe trained on). Makes alpha interpretable across models.
- **Hyperparams:** alpha grid `{-1,-0.5,0,+0.5,+1}` x scale; `--n-per-subset 40`;
  `--epochs 30` (direction fit); `--max-length 2048`.

## 3. Outputs

- `runs/steer_13_<slug>.json`:
  `{model, layer, scale, scale_def, raw_w_norm, intervention, alpha_grid,
    by_subset: {memory_pos, injection_pos, negative} -> [mean P(yes) per alpha],
    n_per_subset, n_fit_pos, n_fit_neg, baseline_pyes_alpha0_selfcheck, question,
    direction_path}`.
- `runs/steer_13_<slug>.json.dir.pt`: `{w_hat, raw_norm, scale, layer, model,
  family}` (the saved steering direction; reusable by a Tier-2 follow-up).

## 4. Result format

Per model, mean P(yes) vs alpha, one curve per subset:

```
alpha        -1.0   -0.5    0.0   +0.5   +1.0
memory_pos   0.xx   0.xx   0.xx   0.xx   0.xx
injection_pos 0.xx  0.xx   0.xx   0.xx   0.xx
negative     0.xx   0.xx   0.xx   0.xx   0.xx
```

plus `baseline_pyes_alpha0_selfcheck.max_abs_diff` (must be < 1e-4, ok=true).

## 5. Interpretation hints

- **Monotone RISE on +alpha, strongest on `memory_pos`** => the memory-family
  direction CAUSALLY drives the stated belief; adding it makes the model say
  "yes, vulnerable" more. Supports a MoC-style linear correction over the
  opaque MLP.
- **Rise on ALL subsets equally (incl. `negative`)** => the direction encodes a
  generic "vulnerable" / yes-bias axis, not memory-specific causation. Still
  causal on belief, but less specific than hoped.
- **FLAT P(yes) across alpha** => the direction is an epiphenomenal correlate
  the probe reads but that does not feed the verbalized belief — steering it
  changes nothing the model says.
- **`negative` rises with +alpha while it should stay low** => steering induces
  false "yes" (over-correction); informs the alpha magnitude a real correction
  could safely use.
- **Sign sanity:** -alpha should push P(yes) DOWN if the axis is signed as
  expected; a symmetric flip is corroborating evidence of causation.

## Caveats / gates

- **alpha=0 self-check GATES validity.** The idle hook (delta=None) must
  reproduce the no-hook P(yes) to < 1e-4 per example; steer_judge.py ABORTS
  otherwise. A failure means the hook is on the wrong tensor (e.g. layer INPUT
  vs OUTPUT, or the wrong module), so every steered number would be untrustworthy.
- **Hook target.** The cached acts store `hidden_states[L+1]` = the OUTPUT of
  `model...layers[L]` (extract_all_layers.py line 123). The hook adds to that
  layer's `output[0]` — the exact tensor the probe trained on. Adding to the
  INPUT or a different layer would mismatch the direction.
- **GEMMA activation-scale caveat.** Gemma-3 has massive mid-layer activations
  (>65504; extract_all_layers.py stores float32 to avoid f16 saturation). The
  activation-norm `scale` is what keeps the fixed alpha grid meaningful for
  Gemma — without it alpha=+1.0 would be a rounding error on its residual
  stream. Expect Gemma's logged `scale` to be MUCH larger than the Qwen models'.

## Open knobs (TODO(adhoc-decision), lead owns)

- **Positions:** all-positions (default) vs code-only positions.
- **Operator:** additive along `w_hat` (default) vs projection-removal then add.
- **Scale unit:** activation-RMS / median token-norm (default) vs probe-margin
  std vs fixed constant.
- **alpha grid:** the briefed `{-1,-0.5,0,+0.5,+1}` (0 is MANDATORY — it gates
  the self-check; the runner refuses a grid without it).

## For agents

- Run: `bash run.sh <model> <best_layer>`.
  Needs `WORK`/`REPO` env, `env.sh`, cached
  acts under `runs/layersweep_<slug>/acts`, `data/dataset.jsonl`,
  `data/sven_split_meta.json`. skip-if-exists on `runs/steer_13_<slug>.json`.
- Direct: `python steer_judge.py --model M --best-layer L --acts-dir ACTS
  --dataset DATA --split SPLIT --out OUT [--n-per-subset 40] [--alphas ...]`.
- The direction fit reuses `train_one_layer` (linear head); `w_hat` is the
  unit-normalized linear weight. The forward-pass scoring reuses
  verbalized_judge.py's `render_chat` (Qwen3 thinking guard) + `p_yes_from_logits`.
- Self-check logic: forward with hook present & `delta=None`, remove the hook,
  forward again, assert per-example abs P(yes) diff < 1e-4 on one eid per subset;
  ABORT (SystemExit) on failure before any steered measurement.
