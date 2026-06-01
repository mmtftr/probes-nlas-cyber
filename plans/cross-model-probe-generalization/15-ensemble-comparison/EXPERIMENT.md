[ai-generated]

# 15 — Ensemble comparison (probe vs verbalized specialization, symmetric)

> Does VERBALIZED specialization (a specialized PROMPT) track PROBE specialization
> (a specialized probe), at the example level? The verbalized analogue of a
> specialized probe is a specialized prompt.

1. **Aim** — Build a SYMMETRIC matrix contrasting probe-side specialization against
   verbalized-side specialization. For both sides the members are: `general`,
   `memory`, `injection`, `ind-ensemble` (MAX over per-individual-CWE members),
   `cat-ensemble` (MAX of the memory + injection members). The probe member is a
   specialized linear probe; the verbalized member is a specialized prompt. Tests:
   (a) does prompt-specialization track probe-specialization (do rows move the same
   way on both sides)? (b) does `ind-ensemble` beat the single category member?
   (c) does `cat-ensemble` recover BOTH families in the overall cell?

2. **Inputs**
   - *Models* (one per node) + best layers: `Qwen/Qwen2.5-Coder-32B-Instruct` L25,
     `google/gemma-3-27b-it` L19, `Qwen/Qwen3-32B` L27, `Qwen/Qwen3.6-27B` L30.
   - *Dataset* — SVEN before/after `$WORK/data/dataset.jsonl` (rebuilt 2026-06-01).
     `cwe` → family via the FAMILY map loaded VERBATIM from `compare_belief_audit.py`
     (memory = CWE-416/476/125/787; injection = CWE-089/078/022/079/190).
   - *Probe side* — `probe_members_scorer.py` at the best layer on CACHED acts
     (`$WORK/runs/layersweep_<slug>/acts`); trains `general`, `memory`, `injection`,
     and 9 per-CWE probes (span-max linear, reusing `train_one_layer`), dumps
     per-seed per-example MAX-pooled sigmoid scores.
   - *Verbalized side* — `14-.../prompt_variants_judge.py` P(yes) at the first
     assistant-token, for the prompts: `V0_generic`(general), `V1_memory`(memory),
     `V_injection`(injection), and `V_cwe416..V_cwe190` (the per-CWE members).
     Shards SHARED with exp-14 in `$WORK/runs/promptsweep_<slug>/`; resumable.
   - *Splits* — 5 group-clean seeds (42–46), `make_split_for_seed` loaded VERBATIM
     from `compare_belief_audit.py` (identical shuffle to the probe scorer).

3. **Outputs** — on scratch:
   - `runs/promptsweep_<slug>/variant_<id>.gpu{0..3}.json` (verbalized, resumable).
   - `runs/ensemble15_<slug>/probe_member_scores.json` (per-seed per-example probe
     member scores + labels/cwe map + per-member n_train_pos / n_test_pos).
   - `runs/ensemble15_<slug>/ensemble15_<slug>_matrix.json` — the symmetric matrix
     (means±std, n_pos per cell, low-n flags) + a readable table printed to stderr.

4. **Result format** — for each `side ∈ {probe, verbalized}`, a `member × cell`
   table of example-AUC (mean±std over seeds), cells `{memory, injection, overall}`:

   | member | memory cell | injection cell | overall cell |
   |---|---|---|---|
   | general | … | … | … |
   | memory | high | ~chance/low | mid |
   | injection | ~chance/low | high | mid |
   | ind-ensemble | MAX(memory CWEs) | MAX(injection CWEs) | MAX(all 9 CWEs) |
   | cat-ensemble | MAX(mem,inj) | MAX(mem,inj) | MAX(mem,inj) → both |

   Plus per-CWE `low_n` flags (probe `n_test_pos < MIN_TRUST_POS`) that explain a
   weak `ind-ensemble` cell.

5. **Interpretation**
   - **Probe and verbalized rows move TOGETHER** (memory↑ only in the memory cell,
     injection↑ only in the injection cell, on BOTH sides) ⇒ verbalized
     specialization tracks probe specialization: prompting a model for a specific
     family is the verbalized analogue of training a family-specialized probe.
   - **`ind-ensemble` ≥ the single category member** on its family cell ⇒ pooling
     individual-CWE specialists adds signal beyond the pooled-category member; if
     it is BELOW, suspect a low-n noisy CWE member dragging the MAX (check the
     `low_n` flags).
   - **`cat-ensemble` recovers BOTH families overall** (overall cell ≈ max of the
     two category cells, well above either single category member's overall) ⇒ the
     union of two specialists is a competent general detector — on whichever side
     it holds. Asymmetry between sides is the headline: if probe `cat-ensemble`
     recovers overall but verbalized does NOT, the model REPRESENTS both families
     but only REPORTS one even when prompted (an introspection gap that prompt-
     specialization cannot close).

## For agents

- Files:
  - `probe_members_scorer.py` (GPU, cached acts): per-seed per-example probe
    member scores. Loads `compare_belief_audit.py` by file path for FAMILY /
    MIN_TRUST_POS / `make_split_for_seed`; reuses `train_one_layer` + the pooled
    fit recipe from `per_cwe_probe.py` / `compare_belief_audit.py`.
  - `build_matrix.py` (CPU-only): merges verbalized shards + probe scores,
    assembles the symmetric matrix (`combine=MAX`, param).
  - `run_ensemble_node.sh` (per-NODE, one model): (i) verbalized judge for ALL
    variants on 4 GPUs (resumable, fills new prompts + Qwen3.6 gpu0), (ii) probe
    member scorer on 1 GPU, (iii) build_matrix. skip-if-exists on the matrix json.
  - `test_build_matrix.py`: synthetic smoke test of the combine + per-cell AUC
    logic (no model/acts). `uv run python …/test_build_matrix.py`.
- Per-node CLI (the human submits these, one model per node):
  - `bash run_ensemble_node.sh Qwen/Qwen2.5-Coder-32B-Instruct 25`
  - `bash run_ensemble_node.sh google/gemma-3-27b-it 19`
  - `bash run_ensemble_node.sh Qwen/Qwen3-32B 27`
  - `bash run_ensemble_node.sh Qwen/Qwen3.6-27B 30`
- Member → prompt / probe mapping:

  | member | probe | verbalized |
  |---|---|---|
  | general | pooled-ALL-positives probe | `V0_generic` |
  | memory | pooled-MEMORY-category probe | `V1_memory` |
  | injection | pooled-INJECTION-category probe | `V_injection` |
  | ind-ensemble | MAX over per-CWE probes (family-relevant per cell) | MAX over per-CWE prompts (`V_cwe*`) |
  | cat-ensemble | MAX(memory probe, injection probe) | MAX(`V1_memory`, `V_injection`) |

- `ind-ensemble` is FAMILY-AWARE per cell: memory cell = MAX over the 4 memory CWE
  members; injection cell = MAX over the 5 injection CWE members; overall = MAX
  over all 9. Holds for both sides.
- Preflight (Qwen3): read the per-VARIANT debug print in the node log — verify the
  rendered tail is the assistant turn-start with NO `<think>` token and yes/no
  dominate the first-token argmax, for EVERY variant.
- Sanity: the `general` member here (probe `V0_generic` analogue) should ~match the
  belief audit's `verbalized_auc` / `family_auc` columns for the same model (same
  prompt, same splits, same family eval set).

## Data-scarcity caveat

Per-CWE TEST positives are tiny for some CWEs (e.g. CWE-787 ≈ 5, CWE-190 ≈ 4).
Their individual probes/prompts are noisy and can DRAG DOWN the MAX `ind-ensemble`.
`build_matrix.py` reports, per `ind-ensemble` cell, the per-CWE member
`n_test_pos` (median over seeds) with a `low_n` flag (`< MIN_TRUST_POS`, default
10) — flagged, NOT hidden. A weak `ind-ensemble` family/overall cell should be read
against these flags before concluding the ensemble fails.

## Decisions (this experiment)

- *Combine rule = MAX over members* (lead-confirmed) `TODO(adhoc-decision)`: held
  as a `--combine {max,mean}` parameter (default `max`); `mean` offered only as a
  robustness alternative. Single site: `build_matrix.py --combine`.
- *Per-seed probe fit, NO 15% VAL carve* `TODO(adhoc-decision)`: `probe_members_
  scorer.py` fits every probe member on the seed's FULL train pool (mirroring
  `compare_belief_audit.py`, NOT `posthoc_ensemble.py` which carves a 15% VAL for a
  learned combiner). Rationale: the matrix must line up with the belief-audit
  EXAMPLE-level reference, which does not carve. Probe scores are stored per seed;
  the matrix averages example-AUC over seeds.
- *alpha = 1.0* `TODO(adhoc-decision)`: matches `compare_belief_audit`'s CLI default
  (the example-level reference) rather than `train_one_layer`'s 10.0 default used
  by exp-10. Single site: `probe_members_scorer.py --alpha`.
- *Prompt wording* `TODO(adhoc-decision)`: `V_injection` + the 9 `V_cwe*` prompts
  are held identical in FORM to `V0`/`V1` (code-before-question, neutral preamble,
  one-word yes/no demand); only the named vulnerability changes. Single source of
  truth: `PROMPT_VARIANTS` in `14-.../prompt_variants_judge.py`. The lead confirms
  exact phrasing.
- *Eval per cell* mirrors `compare_belief_audit` + `analyze_prompt_sweep`: memory
  cell = (memory pos ∪ all neg), injection cell = (injection pos ∪ all neg),
  overall = (all pos ∪ all neg), restricted to eids BOTH sides scored (identical
  eval set per cell so the probe-vs-verbalized contrast is apples-to-apples).

## Results

_(pending run)_
