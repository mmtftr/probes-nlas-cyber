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
- Run (login node), sequential (scheduler MaxSubmit=1):
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

## Results (2026-05-31, jobs 2441849 Gemma / 2441932 Qwen)

`richer_probe_sweep.png`; `metrics_richer_gemma.json`, `metrics_richer_qwen.json`.
All 45 cells/model. Sanity tie-in holds: (single-best-layer, linear) reproduces
exp-03's (base, α=1) at that layer — Gemma L19 = 0.720, Qwen L41 = 0.737.

| config | Gemma Δ vs linear@L19 (0.720) | Qwen Δ vs linear@L41 (0.737) |
|---|---|---|
| single best layer + **mlp512** | +0.020 (0.740±0.006) | **+0.020 (0.758±0.014)** |
| single + mlp256 | +0.015 | +0.013 |
| wide-concat + **mlp512** | **+0.043 (0.763±0.009)** | +0.013 (0.750±0.020) |
| wide-concat + linear | **−0.026** | +0.003 |
| ±2 neighbour-concat + linear | **−0.041** | −0.002 |

**Finding 1 — a non-linear (MLP) head helps both models** (+0.013 to +0.043), with
tight cross-split variance (Gemma best ±0.009). So the linear single-layer probe
is **not** at the activation ceiling — there is extractable non-linear structure.
mlp512 ≥ mlp256 throughout.

**Finding 2 — layer-concat helps only models whose signal is spread across depth,
and only with an MLP.** For Gemma, concatenating {9,19,26,61} lifts the MLP to the
best result (+0.043) but *hurts* a linear head (−0.026 to −0.041 — too many
correlated dims to fit linearly). For Qwen, concat does nothing (single-layer
mlp512 0.758 ≥ concat mlp512 0.750; linear concat ≈ 0). This mirrors exp-02
exactly: Gemma's vuln signal is **bimodal/distributed** across depth (early-mid +
last), so gathering layers adds complementary information; Qwen's is a **tight
single-depth plateau**, so concat only adds noise/params.

**Probe-family verdict (§8 open decision):** **adopt an MLP head** (mlp512) — it's a
consistent, low-variance win on both models. **Layer-concat is per-model**: worth
it where exp-02 showed a distributed signal (Gemma), not where it's concentrated
(Qwen), and only paired with a non-linear head. So the recipe generalizes as
"MLP head, single best layer by default; add cross-layer concat when the depth
profile is distributed."

**Caveat (flagged, not acted on):** the MLP has ~10–100× the params of the linear
head, so part of the +Δ could be fitting SVEN's distribution rather than a better
vulnerability representation. The ±0.009 cross-split variance argues it's real
signal, but the clean confirmation is an **OOD / confound check** (eval the MLP
probe on a different-distribution vuln set; rewrite-secure-to-ugly test) — the
anti-overfit experiments from the research-framing §6/§7 list. Recommend running
one before committing the MLP head into an ADR.
