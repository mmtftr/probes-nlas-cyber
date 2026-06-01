[ai-generated]

# Tier-1 (belief audit + steering) + Tier-4 (family-balanced, MLP ceiling) — 2026-06-01

One session, 4 experiments, all 4 big models (`Qwen2.5-Coder-32B` L25, `Qwen3-32B` L27,
`Qwen3.6-27B` L30, `gemma-3-27b-it` L19). Honest pipeline (leakage-free group split,
`val_tokens_code` selection, tree-sitter code mask). Run on the cluster as chained
4-node debug jobs (one model/node; see `orchestration/`). Result JSONs in each
experiment's `results/`.

---

## ★ Tier-1 belief audit — memory-safety is REPRESENTED but not VERBALIZED (headline)

For each model, MEMORY-family (CWE-416/476/125/787) vs INJECTION (CWE-089/078/079/022/190):
- **probe** sides = `tokens_code` AUC (token-level, our metric): general probe vs pooled
  family probe (exp-10 recipe).
- **verbalized** = the model's own P("yes, vulnerable") — **necessarily EXAMPLE-level**
  (one yes/no per function; there is no token-level verbalized score). 5 seeds.

| model | MEM general / family (tokens_code) | MEM verbalized (ex) | INJ general / family | INJ verbalized |
|---|---|---|---|---|
| Qwen2.5-Coder | 0.519 / **0.662** | **0.388** | 0.880 / 0.891 | 0.750 |
| Qwen3-32B | 0.585 / **0.685** | **0.486** | 0.878 / 0.893 | 0.731 |
| Qwen3.6-27B | 0.534 / **0.727** | **0.545** | 0.869 / 0.891 | 0.789 |
| gemma-3-27b | 0.561 / **0.665** | **0.392** | 0.840 / 0.875 | 0.680 |

**Finding.** Memory-safety vulnerability is linearly decodable from activations (family
probe 0.66–0.73 `tokens_code`, recovering it from the weak general probe 0.52–0.59) — but
the model's **own verbalized judgment is at/below chance (0.39–0.55)**, consistent across
all 4 models. Injection is represented AND verbalized (all ≈0.68–0.89). ⇒ the model
*represents* memory-vuln without *reporting* it: a white-box activation monitor catches
what asking the model misses (charter §1 belief-vs-representation, §3 AI-control). Newer
Qwen verbalizes memory slightly better (0.49–0.55 vs 0.39) — echoes exp-08's "newer Qwen
partially closes the C gap" — but still far below its activations.

Plot: `data/plots/cross-model/fig8_belief_audit.png` (regen `results/belief/make_belief_plot.py`).
Qwen3 verbalized read verified valid: `enable_thinking=False` pre-fills a *closed empty*
`<think></think>` block so the first generated token is the yes/no answer (abort-guard +
`logits[0,-1,:]`). Caveat: example-level, ~54 memory test positives → wide CI; the
qualitative gap (family ≫ verbalized) is robust across seeds, individual memory CWEs are not.

## Tier-1 causal steering — preliminary/mixed (the direction is causal, not clean)

Add `α·scale·ŵ_mem` to the residual stream at the best layer (scale = median |h|₂), sweep
α, measure verbalized P(yes) on memory-pos / injection-pos / negatives. α=0 self-check
(idle hook == no-hook) **passed on all 4** → hook target correct.

- **The memory direction IS causally linked to the stated judgment** — +α raises P(yes)
  (clearest Qwen2.5-Coder: memory-pos 0.08→0.36). Not epiphenomenal.
- **But the effect is global, not memory-specific** (negatives rise comparably), and
  fragile: Qwen3-32B breaks at α=+1; **gemma's scale=8367 (massive mid-layer activations)
  destroys the forward pass at |α|≥0.5**.
- ⇒ preliminary causal evidence; the intervention needs refinement before claiming
  memory-specific belief-steering. **Follow-up:** scale by the projection-std onto ŵ (not
  median |h|); add a random-direction specificity control (does memory rise *more* than
  negatives / a random direction?). Results: `13-causal-steering/results/steer_13_*.json`.

---

## Tier-4 #8 — MLP layer-sweep: the true MLP ceiling (exp-09's null was a layer artifact)

Swept ALL layers with mlp256 + mlp512, `val_tokens_code`-selected. The MLP's own best
layer is **different and deeper** than the linear-selected one, and the MLP beats the
single linear probe there:

| model | linear (L / tc) | exp-09 MLP@linL | MLP own-best (tc) | mlp512 (tc) |
|---|---|---|---|---|
| Qwen2.5-Coder | L25 / 0.788 | 0.788 | **0.817** (L37) | 0.816 (L37) |
| Qwen3-32B | L27 / 0.806 | 0.789 | **0.815** (L45) | 0.824 (L41) |
| Qwen3.6-27B | L30 / 0.787 | 0.795 | 0.793 (L14)* | 0.795 (L13)* |
| gemma-3-27b | L19 / 0.770 | 0.814 | **0.822** (L21) | 0.824 (L21) |

⇒ **the MLP true ceiling is ≈0.82 across all 4**, +0.02–0.05 over the single linear probe —
exp-09's "no MLP gain on honest `tokens_code`" was a **layer-selection artifact** (MLP was
only run at the linear-selected layer). The interpretable post-hoc specialist ensemble
(exp-09, ~0.81–0.82 on Qwen) still **matches** this opaque MLP ceiling.
*Qwen3.6 MLP val-selection picks an early layer (L13/14); its oracle is L47–49 ≈0.81–0.82 —
a real val-selection instability to flag. Results: `12-mlp-layer-sweep/results/`.

## Tier-4 #7 — family-balanced head: confirms the lead's prediction (rule-out)

Single general probe with memory-family oversampling. `sampler=none` reproduces every
baseline exactly (sanity check passed). `family_balanced`:

| model | overall | memory | injection |
|---|---|---|---|
| Qwen2.5-Coder | 0.788→0.781 | 0.519→**0.578** | 0.880→0.848 |
| Qwen3-32B | 0.806→0.792 | 0.585→0.588 | 0.878→0.859 |
| Qwen3.6-27B | 0.787→0.798 | 0.534→**0.601** | 0.869→0.860 |
| gemma-3-27b | 0.770→**0.719** | 0.561→0.572 | 0.840→**0.771** |

⇒ at most a **minor** memory gain (+0.06–0.07 on two models, ~0 on the others), far below
the dedicated family probe (+0.10–0.19), and it **costs injection/overall** (gemma drops
hard). One linear direction can't hold both families by reweighting — consistent with the
"≈one useful linear axis" finding. Results: `11-family-balanced-head/results/`.

---

## Synthesis

The architecture/capacity axis is now fully closed: more head capacity helps only via an
opaque nonlinear MLP at its own layer (≈0.82, exp-09 null was a layer artifact); reweighting
a single linear probe (#7) does not; forcing direction diversity (cosine sweep) does not.
The **charter-central** result is Tier-1: memory-safety is **represented but not verbalized**
— the probe reads a property the model does not report — with preliminary causal evidence
that the representation can move the belief. This is the AI-control-relevant headline (a
monitor reading activations beats asking the model on memory-safety).

**Owed follow-ups:** refine steering (projection-std scale + specificity control);
external-validity (OOD non-SVEN C-vuln set) + code-quality confound (Tier-2) before
publishing the memory headline.
