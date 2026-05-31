[ai-generated]

# 04 — Richer probes: MLP head & layer-concat features

Step 4 of `../PLAN.md`. Exps 01–03 fixed the probe family (linear head on a
single layer) and tuned its loss (paper-faithful span-max, α=1, neg_incl off).
This asks whether a **richer probe** — a non-linear MLP head and/or
**concatenating several layers'** activations per token — beats that linear@
single-layer probe on held-out example-AUC. It feeds the §8 "probe family"
decision: stay linear, or move toward an Openia-style MLP / layer-concat probe.

1. **Aim** — does a richer probe beat the current linear@single-layer probe on
   held-out example-AUC? Two richness axes, crossed: **arch** ∈ {linear, mlp256,
   mlp512} (MLPProbe hidden=256/512), and **feature-set** ∈ {single best layer,
   small neighbour-concat, wide multi-layer concat} (layers concatenated per
   token). Hypothesis: if the linear single-layer probe is already near the
   ceiling of what these activations carry, neither axis clears split noise; a
   real win from MLP or concat would justify the extra complexity.
2. **Inputs** — cached per-layer activation memmaps from exp 02
   (`runs/layersweep_<slug>/acts`, reused — no re-extraction, no model load).
   Grid: feature_set × arch × seed.
   - **Gemma-3-27B** feature-sets `{9,19,26,61; 19; 17,19,22}` (wide concat /
     single best L19 / neighbours of L19).
   - **Qwen2.5-Coder-32B** feature-sets `{34,41,52,63; 41; 40,41,43}` (wide /
     single best L41 / neighbours of L41).
   - archs `{linear, mlp256, mlp512}`, seeds `{42–46}`. Best single layers from
     exp-02 variance (Gemma L19, Qwen L41). α=1, neg_incl off, internal val
     seed=7 fixed — all inherited from exp-03. 3×3×5 = 45 cells/model.
   Probe: AdamW lr=1e-3, 30 epochs, hard labels, MAX-pool example score.
3. **Outputs** — on scratch `runs/richer_<slug>/`: one `cells/cell_*.json` per
   cell, aggregated `metrics_richer.json` (per (feature_set,arch): mean/std ex- &
   tok-AUC over seeds; overall `best`; `linear_single_best` = the single-layer
   linear config). Plot locally.
4. **Result format** — `richer_probe_sweep.png`: one subplot/model, a
   point±1-std for each (feature_set × arch) config, with a dashed line at the
   linear single-best mean and a dotted line at the length baseline (0.575).
   Table: per-config ex-AUC mean±std and Δ vs linear_single_best.
5. **Interpretation** — a config whose ex-AUC mean exceeds linear_single_best by
   **> +1 std** is a real lift ⇒ the richer probe is worth pursuing; everything
   within ±1 std ⇒ the linear single-layer probe is already at the activation
   ceiling, **stay linear** (the simpler, more interpretable head). MLP-only wins
   ⇒ non-linearity matters; concat-only wins ⇒ cross-layer features matter;
   both ⇒ pursue the full Openia-style probe. Whether the winning richness
   differs Gemma-vs-Qwen feeds the same per-model-vs-shared question as the
   layer policy.

**Built-in sanity tie-in:** the (single-best-layer, linear, seed=42) cell must
reproduce exp-03's (base, α=1, seed=42) value at that layer (Gemma L19, Qwen
L41) — same code path, same canonical split, same α/loss.

## For agents

- Interface (landed in parallel in `src/training/train_probe_spanmax.py`):
  `MLPProbe(in_dim, hidden=256, dropout=0.0)`; `train_one_layer(..., alpha=1.0,
  neg_incl=False, probe_factory=None)` returns `r["probe"]` (trained module on
  CPU) plus `r["w"]`/`r["b"]` (None for non-linear). EVAL uses the module
  forward, not `Xte@w+b`.
- Files: `richer_probe_sweep.py` (GPU-sharded cell grid, resumable, reuses
  cached acts; bounded-memory layer-concat with a per-(feature_set,seed)
  feature cache), `aggregate_richer.py`, `submit_richer.sh` (one debug job/model,
  no extraction), `plot_richer.py`.
- Run (login node), sequential (debug-qos MaxSubmit=1):
  `MODEL=google/gemma-3-27b-it FEATURESETS="9,19,26,61;19;17,19,22" bash .../submit_richer.sh`
  then `MODEL=Qwen/Qwen2.5-Coder-32B-Instruct FEATURESETS="34,41,52,63;41;40,41,43" bash .../submit_richer.sh`.
- FEATURESETS contains `;` — keep it quoted; the srun body is single-quoted so
  the compute-node shell never word-splits it.
- Est. ~3–6 min/model (45 cells / 4 GPUs; concat cells cost more RAM/compute).

## Decisions (this experiment)

- *Feature-set choice* `TODO(adhoc-decision)`: single best layer (exp-03
  baseline), a ±2 neighbour-concat around it, and a wide 4-layer concat spanning
  exp-02's depth regimes. Covers "does any cross-layer mix help" without a full
  layer-pair grid. Alternatives not taken: all-pairs concat, learned layer
  attention.
- *Arch grid {linear, mlp256, mlp512}:* one linear baseline + two MLP widths.
  Same training recipe as exp-03 (α=1, neg_incl off, 30 epochs) so the only
  varied factor is the head + features — a clean contrast against the linear
  single-layer probe.
- *Eval via module forward:* required because `w`/`b` are None for non-linear
  heads; logits come from `r["probe"](Xte)`. Linear cells go through the same
  path, so the sanity tie-in still holds bit-for-bit.

## Results

_(pending run)_
