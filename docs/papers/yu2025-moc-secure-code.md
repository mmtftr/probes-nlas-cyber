[ai-generated]

# A Mixture of Linear Corrections Generates Secure Code

- **Citation** — Weichen Yu, Ravi Mangal, Terry Yue Zhuo, Matt Fredrikson,
  Corina S. Păsăreanu. *A Mixture of Linear Corrections Generates Secure Code.*
  arXiv:2507.09508 (Jul 2025). CMU / Colorado State / Monash.
  - HTML: https://arxiv.org/html/2507.09508v1 · Code: https://github.com/viviable/MoC
- **Related in our notes** — `bui2025-openia-correctness.md` (supervised probe on
  own generations), `ribeiro2025-internal-rep-code-correctness.md` (unsupervised
  LAT). This paper is the **closest to our actual setup**: linear vulnerability
  probe on last-token hidden states, **trained on SVEN**, the same dataset our
  `build_dataset_sven.py` produces. Where Openia/Ribeiro probe *correctness*,
  this probes *vulnerability* — our exact target property.

## TL;DR

LLMs can't detect vulnerabilities by prompting (~50% acc, near chance) but their
hidden states linearly encode vulnerability — a last-token linear probe hits
~79–82%. The paper turns that probe into **MoC (Mixture of Corrections)**: an
inference-time steering method that, *only when the probe flags risk*, adds a
vulnerability-correcting vector to the residual stream (with a decay over
generated tokens) to push the model toward secure code without retraining.
For us the **detection half is the interesting half** — same probe family,
same data — and the steering half is a downstream use of the same direction.

## Method (the bits we'd borrow / compare against)

- **Probe.** Linear `c(s) = W·s + b` on hidden states at the **last token
  position**, per transformer block, BCE loss `CE(c(s), v)`. Select block `L*`
  with the smallest probe loss **per CWE type**. (Variants: linear, linear+PCA
  d'≈50–100, small MLP.) This is the same family as our `train_probe.py`;
  per-CWE layer selection is a sharper version of our layer sweep `[8,17,26,34]`.
- **Data.** **SVEN** — 9 CWE types, paired vulnerable/secure code, ~50–150 train
  samples per type (imbalanced). Labels include **"line changes"** = the changed
  lines between the pair, used as **vulnerable token spans** `m..m+n` for the
  dynamic correction's per-token supervision. Direct parallel to our token-span
  labels + span-max loss.
- **Correction vectors (4 ways to get the steering direction from the probe):**
  group-mean difference `mean(s⁺)−mean(s⁻)`; decision-boundary normal
  `W₁−W₀`; PCA-reduced normal; and a dynamic NN `N(s)` trained with MSE+CE+KL.
- **Inference.** Conditional (apply only if probe detects risk — unconditional
  steering wrecks functionality), with exponential decay `Δs := α(t)·Δs` over
  tokens generated, and a linear **mixture** when several CWEs fire at once.
- **Models.** Qwen2.5-Coder 3B/7B/14B/32B, CodeLlama 7B/13B/34B, plus Claude as
  a closed baseline.

## Results (numbers worth comparing against)

- **Detection — probe ≫ prompting.** Qwen2.5-Coder-7B: prompting **49%** vs
  linear probe **79%** (PCA 76%, MLP 77%); 14B: prompting 40% vs probe **82%**.
  Direct-prompt accuracy 23–51% across models — *worse than random* on some CWEs.
- **Best block is late** (last attention block, their Fig. 3) — consistent with
  Openia/Ribeiro mid-to-late finding.
- **Per-language gap:** Python CWEs (022/078/079/089) detect well; C CWEs
  (125/416/476/787) are harder. Relevant to our heldout-CWE/lang splits.
- **Secure generation (Security Ratio):** QC-7B 76.8%→85.7% (+8.9, dynamic NN)
  with HumanEval pass@1 also up 79.9%→82.0%; gains shrink on already-secure 14B.
  Beats SVEN prior work (QC-7B 65.0% SR).

## Notes / relevance to us

- **Strongest external evidence for the project's core bet:** vulnerability *is*
  linearly decodable from hidden states on SVEN, and the probe **crushes the
  ask-the-LLM-directly baseline** (79% vs 49%). That baseline is exactly the
  anti-tunnel-vision check in research-framing §7 (mitigation #4) — cite this
  result when we run it; expect probe ≫ prompt, but verify on Gemma 3.
- **Generation-time states.** Their detection is on the model's *own generation*
  hidden states (then steered) — speaks to complication §2.1 (generation vs
  reading direction). They never test reading-direction transfer; we can.
- **"Bug-prone ≈ vulnerable" overlap.** They note steering away from vulnerability
  also fixes logical errors — i.e. the direction partly tracks *low code quality*,
  not pure vulnerability. This is precisely our failure mode §6 "probe detects
  low-quality code, not necessarily vulnerable." Direct corroboration that the
  worry is real; their LLM-rewrite-style control (research-framing §7 #2) matters.
- **Per-CWE probe/layer selection** is a design point we haven't committed to —
  relates to the layer-policy `TODO(adhoc-decision)` (Q3). They fit a probe (and
  pick a block) *per CWE*; we currently pool. Worth an experiment.
- **Caveats they state:** known CWEs only (no novel vulns); steering cost scales
  with tokens; corrections transfer 3B↔7B but **fail to larger models** and can
  harm functionality on transfer — reinforces the model-specific-probe expectation
  (Q5). Small/imbalanced per-CWE training sets.
- **AI-control angle:** they frame the probe as a lightweight inference-time
  **monitor** and explicitly flag adversarial misuse (steering *toward*
  vulnerability to weaken future models) + urge human-in-the-loop — aligns with
  our downstream-task framing §3.
- **TODO(adhoc-decision):** does MoC-style steering (not just detection) enter
  scope, or is it out-of-scope downstream? Flagged, not decided — the repo's bet
  is detection/monitoring; steering is a different deliverable.
