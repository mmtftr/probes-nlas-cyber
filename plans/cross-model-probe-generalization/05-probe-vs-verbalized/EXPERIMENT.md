[ai-generated]

# 05 — Probe vs. the model's own verbalized judgment

> ⚠️ **Archived dataset.** Results below were produced on the flawed
> completion-truncation `dataset.jsonl` — see `../archive/old-dataset/README.md`
> and `decisions/0002-dataset-before-after-contrast.md`. Result artifacts moved
> to `../archive/old-dataset/05-verbalized/`. The truncation directly confounded
> this experiment (the model was asked yes/no about an incomplete prefix).
> Scripts are correct as-is; **re-run on the SVEN before/after dataset**
> (`../REBUILD-PLAN.md`) — the prefix confound largely disappears there.

Step 5 of `../PLAN.md`. The anti-tunnel-vision check from research-framing §7.4
("just ask the LLM") and the §6 "white-box ≥ probes?" failure mode: before
claiming a trained span-max probe is worth anything, show it beats the cheapest
possible white-box read — the model's OWN yes/no answer to "is this vulnerable?".

1. **Aim** — does a trained span-max probe beat the model's verbalized
   vulnerability judgment on the same held-out SVEN examples? For each example we
   ask the model "does this code contain a vulnerability?" and read P(yes) over
   its yes/no answer tokens; we compare that verbalized example-AUC to the probe's
   example-AUC. Hypothesis: **probe > verbalized** — the model internally
   represents vulnerability better than it states when asked (the introspection
   gap), the headline AI-control-relevant result.
2. **Inputs** — target property = the model's belief about the **input-stream**
   code (§1). Two halves:
   - *verbalized:* a NEW model forward (LOADS the model — like extraction, needs
     `--environment=container` + GPU). Input-stream framing, **code BEFORE the
     question**, minimal neutral preamble, so the code is encoded the same
     neutral way the probe's cached activations were (a question before the code
     would condition the encoding). Score = `sigmoid(logsumexp(logprob[yes_ids])
     − logsumexp(logprob[no_ids]))` at the first assistant-token position.
   - *probe:* reuses the cached per-layer acts `runs/layersweep_<slug>/acts` (NO
     re-extraction). Best single layer: **Gemma-3-27B L19, Qwen2.5-Coder-32B
     L41** (exp 02), linear span-max, **α=1** (exp 03), 30 epochs, internal val
     seed=7. 5 group-clean splits (seeds 42–46).
3. **Outputs** — on scratch `runs/verbalized_<slug>/`: per-GPU shards
   `verbalized_scores.gpu{0..3}.json` (eid, p_yes, label), aggregated
   `metrics_verbalized.json` (model_layer, n_examples_scored, per-seed +
   mean/std probe_auc / verbalized_auc / delta, spearman, disagreement lists +
   seed42 arrays). Plot locally → `probe_vs_verbalized.png`.
4. **Result format** — `probe_vs_verbalized.png`: per model a grouped bar
   probe-AUC vs verbalized-AUC (mean ±1 std) with the dotted length-baseline
   0.575 line, plus a scatter of probe score vs P(yes) colored by label on the
   seed-42 test set. The headline number is **Δ = probe_auc − verbalized_auc**
   with its std; plus counts of "probe catches, model verbally denies" examples.
5. **Interpretation** —
   - Δ > +1 std ⇒ the probe reads an internal belief the model does not state →
     **introspection gap, validates probing** over "just ask the LLM".
   - Δ ≈ 0 (within ±1 std) ⇒ the probe adds little over asking; reconsider its
     cost.
   - verbalized ≫ probe ⇒ asking is enough; rethink the whole probe program.
   The **disagreement analysis** is the qualitative payoff: examples that are
   truly vulnerable, the probe ranks in its top tercile, yet the model verbally
   says "no" (p_yes < 0.5) — concrete cases where the internal signal contradicts
   the stated answer.

## For agents

- Files: `verbalized_judge.py` (model forward → P(yes) per example, GPU-sharded,
  resumable shards), `compare_probe_vs_verbalized.py` (cached-acts probe vs.
  merged verbalized scores, no model), `submit_verbalized.sh` (one debug
  job/model: phase 1 = 4-GPU model fanout, phase 2 = compare),
  `plot_verbalized.py` (local matplotlib).
- `make_split_for_seed` copied verbatim from exp-02; `pair_group_key` /
  `example_scores` come from `src/remotes/the cluster/train_eval.py`; model loading
  from `src/data/extract_token_activations.py` (`_load_model`/`_load_tokenizer`,
  bfloat16 on cuda). Same SBATCH/srun header as exp-03.
- Run (login node), sequential (scheduler MaxSubmit=1):
  `MODEL=google/gemma-3-27b-it LAYER=19 bash .../submit_verbalized.sh`
  then `MODEL=Qwen/Qwen2.5-Coder-32B-Instruct LAYER=41 bash .../submit_verbalized.sh`.
- Phase 1 is a per-example forward over ~all SVEN rows (batch 1, no cache) —
  the slow part; phase 2 is fast (5 probes on cached acts).
- Sanity: phase-2 probe_auc at this layer should ~match exp-03's base/α=1 cell
  (Gemma L19 ≈ 0.720, Qwen L41 ≈ 0.737) — same code path, same splits.

## Decisions (this experiment)

- *Code-before-question, neutral preamble* `TODO(adhoc-decision)`: the eliciting
  question comes AFTER the code so the model encodes the code neutrally (matching
  how the probe's activations were extracted — raw code, no question). A question
  first would condition the encoding and make the comparison unfair to the probe.
  Preamble is minimal and non-priming ("Here is a code snippet:").
- *Score = P(yes) over yes/no first-token logits* (continuous, for AUC), robust
  over casing / leading-space variants via `resolve_yes_no_ids`. We do NOT
  generate; we read the next-token distribution at the generation position.
- *Best single layer + α=1*: carried from exp 02 (layer) and exp 03 (α). The
  probe side is deliberately the tuned best-case so the comparison is the
  strongest probe vs. the cheapest verbalized read.
- *Verbalized AUC over the intersection* of scored eids and each split's test
  eids — guards against a shard gap silently dropping examples from one side.

## Results

_(pending run)_
