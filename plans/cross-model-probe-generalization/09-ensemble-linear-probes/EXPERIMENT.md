[ai-generated]

# 09 — Interpretable ensemble of linear probes

Replace the single linear span-max head with an ensemble of **K linear
directions** aggregated to one per-token scalar logit, so different directions
can specialise (e.g. taint/injection vs memory-safety) while each stays
inspectable. Runs on **cached acts at one best layer** — no re-extraction.

## 1. Aim

Beat a single linear probe on honest `tokens_code` — **especially on C /
memory-safety CWEs** (sweep-6: C ≈0.59, UAF ≈0.52, NULL-deref ≈0.55, OOB-read
≈0.56) — by letting K small directions specialise, instead of forcing one
direction to cover both the injection and memory-safety vuln families.

## 2. Inputs

- **Acts:** cached float32 all-layer memmaps from exp-06,
  `runs/layersweep_<slug>/acts/` (`layer_NN.npy`, `offsets.npz`, `y.npy`,
  `example_ids.npy`, `meta.json`). Only the ONE best-layer `.npy` is loaded.
- **Primary model:** `Qwen/Qwen2.5-Coder-32B-Instruct` (slug
  `Qwen_Qwen2.5-Coder-32B-Instruct`), best layer **25**, single-probe test
  tokens_code = **0.788**.
- **Secondary:** `gemma-3-27b-it`, best layer **19** (run by re-invoking the
  submit script; `TODO(adhoc-decision)` — lead decides whether to run it).
- **Data/recipe:** SVEN before/after `data/dataset.jsonl`,
  `sven_split_meta.json`. Span-max loss, same group-aware test hold-out + 15%
  group-aware VAL carve (VAL_SEED=42) as exp-06, so cells are apples-to-apples
  with 0.788.
- **Head:** `Linear(d, K)` → K per-token logits → aggregate to one scalar via
  one of `{max, logsumexp(τ), softmax_gate}`. Sweep **K ∈ {1, 2, 4, 8}**
  (K=1 == single-linear baseline). Plugged into `train_one_layer` via
  `probe_factory`. 30 epochs, defaults otherwise.

## 3. Outputs

`runs/ensemble_<slug>/`:
- `cells/K{K}_{agg}.json` — one per (K × agg) cell: val + test `tokens_code`,
  overall + per-lang + per-CWE breakdown, example AUC.
- `cells/K{K}_{agg}.dirs.pt` — the K weight directions + biases + gate params,
  saved for later inter-direction cosine-sim and per-CWE firing analysis.
- `summary_<slug>.json` + `summary_<slug>.md` — the {K × agg}-vs-K=1 table.
- `ENSEMBLE09_DONE` marker.

## 4. Result format

The {K × agg} table (overall + per-lang + per-CWE `tokens_code`), each cell
showing its AUC and Δ vs the K=1 baseline. Headline numbers to report back:
- overall `tokens_code` best cell vs 0.788;
- **C** and per-CWE memory-safety (UAF/NULL/OOB) best cell vs the ~0.52–0.59
  sweep-6 anchors;
- which (K, agg) wins, and whether the winning cell was the one VAL-selected
  (val_tokens_code) — honest selection, not test-peeking.

## 5. Interpretation hints

- **Best cell C ≫ 0.59 (toward python's ~0.81) ⇒ the win we want:** a second
  specialised direction recovers memory-safety signal the single direction
  couldn't carry. Inspect `.dirs.pt` — expect low cosine between the
  injection-firing and C-firing directions.
- **Overall up but C flat ⇒ ensemble just sharpened the injection signal**
  (more capacity on the already-easy family); does not address the blind spot.
- **K=1 wins / Δ≈0 across the board ⇒ a single linear direction is at the
  data ceiling here** (consistent with sweep-6's "signal is injection-class
  only"); the C gap is representational/data-level, not a head-capacity
  problem — pushes the question to exp-10 (per-CWE) / exp-08 (newer models).
- **softmax_gate ≫ max/logsumexp ⇒ token-conditioned routing matters** (the
  gate learns *which* direction to trust per token); if max ≈ gate, the extra
  gate params don't pay off and the simpler max ensemble is preferred.

## Critique

Honest reasons K>1 may **not** beat K=1, and how this design hedges them:

- **Span-max already max-pools over TOKENS.** The loss takes
  `max_i p_i` per example, so the probe only needs ONE token to fire per vuln.
  Adding `max` over K DIRECTIONS is a second max on top — for a single vuln
  family this is largely redundant and mostly adds overfitting surface, not new
  expressivity. The honest test is therefore the **per-CWE/per-lang** cells,
  not overall: K>1 should pay off only where two *distinct* families coexist
  and one direction can't serve both.
- **logsumexp ≈ smooth max.** At τ=1 it is a soft maximum; as τ→∞ it is exactly
  `max`. So `logsumexp` is unlikely to differ from `max` except in gradient
  smoothness early in training — it is a regularisation knob, not a new
  hypothesis. Included mainly as a controlled interpolation.
- **softmax_gate adds parameters that may not pay off on scarce C data.** The
  per-token gate is another `Linear(d, K)` — on the few hundred C rows in SVEN's
  train split this can overfit. `gate_mode="global"` (one shared K-vector) is
  provided as the lower-variance fallback, and VAL-based selection guards
  against picking an overfit cell.
- **The ensemble can collapse to one effective direction.** Nothing forces the
  K directions to differ; with a max/logsumexp agg and a single dominant family,
  training can drive K−1 directions to be ignored (redundant copies). We detect
  this post-hoc via the saved directions (inter-direction cosine ≈ 1 ⇒
  collapse). If it collapses, that is itself the finding: capacity wasn't the
  bottleneck.
- **No diversity pressure.** A stronger design would add an explicit
  orthogonality/decorrelation penalty or per-CWE supervision to *force*
  specialisation. We deliberately keep the head minimal here (the lead's framing
  is "small K, each direction inspectable"); if K>1 shows promise but collapses,
  the natural follow-up is a diversity-regularised variant — and exp-10
  (per-CWE probes) is the supervised-specialisation counterpart that sidesteps
  the collapse risk entirely.

**Bottom line (agent's take, non-binding):** overall `tokens_code` is unlikely
to move much past 0.788 — span-max + a single linear direction is already
near the injection-class ceiling. The one place K>1 could genuinely win is the
**C / memory-safety** cells, *if* the memory-safety signal is linearly present
in these activations but on a direction the single probe never selected (because
the injection family dominates the loss). If C stays ~0.59 across all K and aggs,
that is strong evidence the memory-safety signal is absent/weak at this layer —
a clean negative that hands the question to exp-08/10.

## For agents

- Run pattern mirrors `06`'s breakdown step but cheaper: **1 GPU, single
  layer, all {K × agg} cells in one short run**. `run.sh` is
  idempotent (each cell skips if its JSON exists; aggregate always re-runs).
- **Run AFTER exp-08** — run one experiment at a time; don't
  launch while an 08 layersweep is still running.
- Cached acts MUST exist (`runs/layersweep_<slug>/acts/`); the job asserts the
  dir is present and exits if not.
- Score path: ensemble heads are non-linear, so `train_one_layer` returns
  `w=None`; the runner scores test/val tokens by RUNNING the trained module
  (not `X @ w`). This is the one substantive deviation from the linear sweep.
- Local interface test (no GPU): `test_ensemble_probe.py` — asserts forward
  shape `(n_tokens,)`, differentiability of every agg, and K=1 collapse.
- `TODO(adhoc-decision)` markers in the code/this brief:
  1. **τ default = 1.0** for logsumexp (knob, not hard-coded). Possible sweep
     {1, 4, 10}.
  2. **gate granularity** per_token (default) vs global.
  3. **K sweep {1,2,4,8}** and the **agg set** {max, logsumexp, softmax_gate}.
  4. **secondary model** gemma-3-27b-it (L19) — run only if the lead wants it.
