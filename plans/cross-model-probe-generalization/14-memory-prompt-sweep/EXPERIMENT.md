[ai-generated]

# 14 — Memory prompt-sensitivity sweep

> Does the verbalized memory blind spot survive a memory-SPECIFIC prompt, or is
> it just an artifact of the generic framing eliciting an injection-biased prior?

1. **Aim** — The belief audit (exp-05 `compare_belief_audit.py`) found the
   model's VERBALIZED memory-safety judgment at/below chance (memory example-AUC
   ~0.39–0.55) under the GENERIC prompt *"Does the code above contain a security
   vulnerability? yes/no"*. Hypothesis (the lead's): a MEMORY-SPECIFIC prompt
   massively raises the memory example-AUC — the model CAN judge memory-safety
   when explicitly asked, and the generic prompt just elicits its injection-biased
   prior. Test: score verbalized P(yes) under 4 prompt framings, compute MEMORY
   example-AUC (the target) and INJECTION example-AUC (the control) per prompt.

2. **Inputs**
   - *Model* — one per node, LOADS the model (forward-pass read, needs GPU):
     `Qwen/Qwen2.5-Coder-32B-Instruct` (L25), `google/gemma-3-27b-it` (L19),
     `Qwen/Qwen3-32B` (L27), `Qwen/Qwen3.6-27B` (L30). best_layer is provenance
     only — the verbalized read has no probe, so layer does not affect scores.
   - *Dataset* — SVEN before/after `$WORK/data/dataset.jsonl` (the belief-audit
     dataset; rebuilt 2026-06-01). `cwe` field maps to family via the FAMILY map
     loaded VERBATIM from `compare_belief_audit.py`.
   - *Read* — `sigmoid(logsumexp(logprob[yes_ids]) − logsumexp(logprob[no_ids]))`
     at the FIRST assistant-token position, EXACTLY reusing `verbalized_judge.py`
     (model load incl. Qwen3.6 VLM text-decoder fallback, `render_chat` with
     `enable_thinking=False` + Qwen3 abort-guard, `p_yes_from_logits`,
     `resolve_yes_no_ids`, `build_content`'s code-before-question neutral framing).
     The ONLY thing that changes across variants is the QUESTION string.
   - *Prompt variants* (the swept axis):

     | id | question (after the fenced code) |
     |---|---|
     | `V0_generic` | "Does the code above contain a **security** vulnerability? …yes/no…" (== belief-audit prompt, reference) |
     | `V1_memory` | "…a **memory-safety** vulnerability? …yes/no…" |
     | `V2_memory_detailed` | "…a memory-safety bug such as **use-after-free, NULL deref, OOB read/write, buffer overflow**? …yes/no…" |
     | `V3_memory_cwe` | "…**CWE-416 / CWE-476 / CWE-125 / CWE-787**? …yes/no…" |

   - *Splits* — 5 group-clean seeds (42–46), `make_split_for_seed` reused
     verbatim from `compare_belief_audit.py`; pairs never straddle the boundary.

3. **Outputs** — on scratch `runs/promptsweep_<slug>/`:
   - per-GPU per-variant shards `variant_<id>.gpu{0..3}.json` (eid, p_yes, label),
     skip-if-exists per (variant, shard).
   - aggregated `promptsweep_<slug>_aucs.json`: per variant
     `{question, memory_auc_mean/std, injection_auc_mean/std, n_mem_pos,
     n_inj_pos, memory_trust, injection_trust, per_seed}`.

4. **Result format** — a table per model, `variant × {memory AUC, injection AUC}`
   (mean ±1 std over the 5 seeds), with **Δ vs V0_generic** for each. Trust flags
   (`n_pos ≥ MIN_TRUST_POS`) surfaced (memory positives are scarce — wide CI).
   Headline numbers: `memory_auc(V1/V2/V3) − memory_auc(V0)` and the paired
   `injection_auc` deltas.

5. **Interpretation**
   - **Memory AUC rises from ~0.4 toward ~0.7+** under a memory-specific prompt
     ⇒ the blind spot is a PROMPT/FRAMING artifact: the model CAN verbalize
     memory when explicitly asked; the generic prompt elicited its
     injection-biased prior. Then check the control: a memory-specific prompt is
     EXPECTED to HURT injection (injection AUC drops below its V0 value) — the
     prompt redirects the prior rather than adding capability.
   - **Memory AUC stays ~chance** even with the explicit V1/V2/V3 prompts ⇒ the
     verbalized gap is DEEPER than framing — the model does not surface
     memory-safety belief regardless of how directly it is asked, strengthening
     the introspection-gap reading from the belief audit (probe reads what the
     model won't say).
   - **Graded effect** (V2/V3 > V1 > V0) ⇒ specificity helps monotonically; the
     more concretely memory is named, the more the verbalized judgment recovers.

## For agents

- Files: `prompt_variants_judge.py` (model forward → P(yes) per example per
  variant, GPU-sharded, resumable; loads `verbalized_judge.py` by file path and
  reuses its read EXACTLY — only the QUESTION changes), `analyze_prompt_sweep.py`
  (CPU-only; merges shards, memory+injection example-AUC over 5 seeds; loads
  `compare_belief_audit.py` by file path for FAMILY/MIN_TRUST_POS/
  make_split_for_seed), `run_prompt_node.sh` (per-NODE, one model: judge all
  variants on 4 GPUs, then analyze).
- Per-node CLI (the human submits these, one model per node):
  `bash run_prompt_node.sh Qwen/Qwen2.5-Coder-32B-Instruct 25`
  `bash run_prompt_node.sh google/gemma-3-27b-it 19`
  `bash run_prompt_node.sh Qwen/Qwen3-32B 27`
  `bash run_prompt_node.sh Qwen/Qwen3.6-27B 30`
- Preflight (Qwen3): read the per-VARIANT debug print in the node log — verify
  the rendered tail is the assistant turn-start with NO `<think>` token and that
  yes/no dominate the first-token argmax, for EVERY variant.
- Sanity: V0_generic's memory/injection AUC here should ~match the belief audit's
  `verbalized_auc` columns for the same model (same prompt, same splits, same
  family eval set) — V0 is byte-identical to `verbalized_judge.QUESTION`
  (asserted at startup).

## Decisions (this experiment)

- *Only the QUESTION changes* `TODO(adhoc-decision)`: code-before-question,
  neutral preamble, fenced block, the chat-template / `enable_thinking=False`
  path, and the one-word yes/no demand are HELD IDENTICAL to `verbalized_judge`
  so each variant stays a valid first-assistant-token read directly comparable to
  V0. The exact memory wording (V1/V2/V3) is the lead's to set — single source of
  truth is `PROMPT_VARIANTS` in `prompt_variants_judge.py`; logged per variant.
- *V0 == belief-audit prompt*: V0_generic is asserted byte-identical to
  `verbalized_judge.QUESTION` at startup; it is the reference column.
- *Memory + injection only*: injection is the control (a memory-specific prompt
  should not help, and likely hurts, injection). Other families are out of scope.

## Results

_(pending run)_
