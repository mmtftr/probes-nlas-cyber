[ai-generated]

# Ensemble comparison — specialized PROBE ≈ specialized PROMPT (2026-06-01)

Symmetric matrix: probe-side vs verbalized-side specialization, EXAMPLE-level AUC,
4 big models, 5 seeds, combine = MAX. The verbalized analogue of a specialized probe
is a specialized PROMPT. Members:

| member | probe | verbalized |
|---|---|---|
| general | pooled-ALL-positives probe | generic "is this vulnerable?" |
| **memory probe** | pooled-MEMORY-category probe | memory prompt |
| injection probe | pooled-INJECTION-category probe | injection prompt |
| ind-ensemble | MAX over per-individual-CWE probes (family-aware per cell) | MAX over per-CWE prompts |
| cat-ensemble | MAX(memory probe, injection probe) | MAX(memory prompt, injection prompt) |

(Naming: what earlier docs called the "family probe" is the **memory probe** — a single
probe pooled over the memory CWE category. "category probe" = memory/injection probe.)

## MEMORY cell — probe / verbalized example-AUC

| member | Qwen2.5-Coder | Qwen3-32B | Qwen3.6-27B | gemma-3-27b |
|---|---|---|---|---|
| general | 0.32 / 0.39 | 0.27 / 0.49 | 0.23 / 0.55 | 0.47 / 0.39 |
| **memory** | 0.74 / 0.69 | 0.71 / 0.74 | 0.76 / 0.77 | 0.73 / 0.64 |
| injection | 0.27 / 0.24 | 0.24 / 0.26 | 0.22 / 0.28 | 0.29 / 0.25 |
| **ind-ensemble** | 0.76 / 0.70 | 0.74 / 0.72 | 0.75 / 0.76 | 0.74 / 0.68 |
| cat-ensemble | 0.29 / 0.56 | 0.27 / 0.60 | 0.24 / 0.61 | 0.47 / 0.39 |

## OVERALL cell (all-pos ∪ neg) — probe / verbalized

| member | Qwen2.5-Coder | Qwen3-32B | Qwen3.6-27B | gemma-3-27b |
|---|---|---|---|---|
| general | 0.58 / 0.60 | 0.55 / 0.63 | 0.56 / 0.69 | 0.62 / 0.56 |
| ind-ensemble | 0.63 / 0.70 | 0.59 / 0.72 | 0.58 / 0.74 | 0.64 / 0.61 |
| cat-ensemble | 0.59 / 0.68 | 0.58 / 0.70 | 0.56 / 0.72 | 0.59 / 0.59 |

## Findings

1. **Specialization is symmetric** — a *memory probe* and a *memory prompt* recover
   memory-safety comparably (~0.69–0.77 both) where the *general* member fails on both
   sides (0.23–0.55). The probe↔prompt analogy holds member-by-member.
2. **ind-ensemble (family-aware MAX over per-CWE members) is the best per-family
   detector on BOTH sides.** Asking/probing per-CWE and taking the max recovers each
   family — memory (0.74–0.76 probe, 0.68–0.77 verbalized) and injection.
3. **cat-ensemble is contaminated on the memory cell** (esp. probe, 0.24–0.47): the
   non-family-aware MAX(memory,injection) lets the injection member's false-positives on
   negatives drag the memory ranking down. The family-aware ind-ensemble avoids this.
   (Clean max-combine lesson: combining a mis-pointed member hurts via its FPs.)
4. **At the example level, verbalized ind-ensemble (0.70–0.74) ≥ probe ind-ensemble
   (0.58–0.64) overall.** Caveat: probes' native strength is TOKEN-level `tokens_code`
   (where the general probe is anti-pointed on memory → its example max-pool is below
   chance, per the aggregation diagnostic). The example level is the only granularity
   comparable to verbalized (one judgment per function), so this is the fair head-to-head
   for "probe vs prompt", not a claim that probes are weak in general.
5. **gemma is the outlier** — its general probe is less anti-pointed on memory (0.47) and
   its verbalized side is weaker/noisier (verbalized ind-ensemble overall 0.61 vs Qwen
   ~0.72).

**Connects to the belief audit + prompt sweep:** memory-safety is represented; whether a
PROBE or the model's VERBALIZED judgment reads it depends on SPECIALIZATION (a memory
probe / a memory prompt), not on the model's default behaviour (general probe / generic
prompt), which is injection-shaped. Data-scarcity caveat: per-CWE members for CWE-787
(n≈5) / CWE-190 (n≈4) are noisy and can drag the MAX ind-ensemble; flagged in the matrix
JSONs (`ind_ensemble_members[*].low_n`).

Plot: `data/plots/cross-model/fig9_ensemble_matrix.png` (regen `make_matrix_plot.py`).
Matrices: `results/ensemble15_<slug>_matrix.json`.
