[ai-generated]

# Cross-model probe generalization

## Goal

How well does our token-level span-max vulnerability probe **recipe** generalize
across open-weight models, and how do its properties vary? Probes are
model-specific (different hidden spaces — no weight transfer), so "generalize"
means: re-fit the same recipe per model on the same SVEN split, then compare
probe *properties*. Grounds framing-doc Q3/Q5 and complication §2.1 with data
from one ~8 h multi-GPU window.

## Steps

1. **Core sweep — method generalization.** Per model: extract token activations
   (auto layers {n/4,n/2,3n/4,n−1}) → span-max probe per layer → eval on the
   held-out group-clean SVEN split. Compare across models: best-layer *fraction
   of depth*, example/token AUC, lift over regex/length baselines.
   *Lit:* model-specific-but-method-transfers reps (Ribeiro 2512.07404; Bui/Openia
   2501.12934); both peak mid-late layers.
2. **Fine-grained layer sweep** on one large model (Gemma-3-27B or Qwen2.5-Coder-32B):
   extract *every* layer, full AUC-vs-depth curve + per-layer calibration.
   Settles the "sweep-then-select vs fixed layer" open decision.
   *Lit:* Openia optimal layers ≈16–28; Ribeiro val-layer within ~7pp of oracle.
3. **Reading-vs-generation pilot** (complication §2.1): does a probe fit on
   *read* code transfer to *generation-time* states? Fit on one, eval on the other.
   *Lit:* Openia probes generation; Ribeiro probes reading.
4. **Data-efficiency + calibration** (near-free riders on §1's activations):
   AUC vs #training pairs; Platt vs temperature per model.
   *Lit:* Openia ~60% data → near-optimal; AI-control monitors run at fixed threshold.

## Success criteria

A table over ≥8 models with best-layer-fraction, AUC, and baseline-lift; a
decision on layer policy from step 2; a first read↔gen transfer number from
step 3. Every run = one set of `runs/<model>/metrics.json` (later → wandb).

## For agents

Run the sweep one model per GPU (4 GPUs → 4 models/job); smoke-test before the
full sweep. Step 1 is the running experiment; steps 2–4 are briefed below and
await user sign-off before launch.

### Step 1 — Core sweep (briefing)
- **Aim:** the span-max probe recipe yields a usable vuln probe (AUC ≫ regex)
  across model families/sizes/post-training, with best layer at a stable depth fraction.
- **Inputs:** model roster (dense ≤32B + small MoE);
  `bstee615/sven` rebuilt with token_labels; seed-42 20%-group-held-out split.
- **Outputs:** `runs/<slug>/{metrics.json,probe.npz}`.
- **Result format:** table `model | params | best_layer | best_layer_frac |
  test_ex_auc | test_tok_auc | regex_auc | length_auc`.
- **Interpretation:** AUC≈regex ⇒ probe adds nothing for that model; high AUC but
  best_layer_frac varies wildly ⇒ no universal layer rule (kills fixed-layer
  shortcut); base≪instruct ⇒ post-training installs the vuln direction (Q5).

### Step 2 — Fine-grained layer sweep (briefing)
- **Aim:** is there a single depth fraction that's near-optimal for all models, or
  must we val-select per model?
- **Inputs:** 1–2 large models; extractor patched to capture all layers (shard the
  dataset → ≤4 jobs to stay under 90 node-min; concat).
- **Outputs:** per-layer AUC + ECE arrays; AUC-vs-depth plot.
- **Result format:** curve + argmax layer vs the {n/4,n/2,3n/4,n−1} picks.
- **Interpretation:** flat-topped mid-late plateau ⇒ a fixed fraction is safe;
  sharp narrow peak ⇒ must val-select. `TODO(adhoc-decision)` resolves here.

### Step 3 — Reading-vs-generation pilot (briefing)
- **Aim:** probe fit on read-code states ↔ generation-time states transfer?
- **Inputs:** same SVEN snippets in (a) tool-output/read framing vs (b) assistant
  teacher-forced continuation; capture states for each.
  `TODO(adhoc-decision)`: exact gen-time framing + position set. Needs a new
  extraction path — scope only if smoke leaves margin.
- **Outputs:** 2×2 transfer matrix (fit∈{read,gen} × eval∈{read,gen}) of AUC.
- **Result format:** the 2×2 AUC matrix per model.
- **Interpretation:** strong off-diagonal drop ⇒ direction is framing-specific
  (complication §2.1 is real, a read-probe won't catch model-written vulns).

### Step 4 — Data-efficiency + calibration (briefing)
- **Aim:** how much labeled data the probe needs; how calibrated its scores are.
- **Inputs:** §1 cached activations; subsample train pairs at 10–100%.
- **Outputs:** AUC-vs-N curve; ECE pre/post Platt & temperature.
- **Result format:** efficiency curve + calibration table per model.
- **Interpretation:** plateau ≪100% ⇒ cheap to add models; large ECE ⇒ scores
  need calibration before a fixed-threshold AI-control monitor (Q4).
