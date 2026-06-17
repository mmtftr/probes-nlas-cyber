[ai-generated]

# 30 — Last-token introspection probe (the `Assistant:` turn-boundary)

The genuine NXT2b. Every prior probe read code-token positions; exp-29 showed the
deployed probe at the final *code* token is chance. This probes the hidden state
at the **`Assistant:` turn-boundary** — the last prompt token in the verbalized QA
format, the exact position whose next-token logits give verbalized P(yes). It is
the position where the model commits to a response and where introspective signal
is hypothesized to concentrate (sleeper-agents). Needs a NEW extraction: exp-17
ran this forward but saved only the yes/no vocab logits, never the hidden state.

## The five fields

1. **Aim** — Is there a *linear, introspective* vulnerability direction at the
   commit point that the code-token probes missed? Train a linear probe on the
   last-prompt-token hidden state (verbalized QA format) and ask whether its
   example-level AUC beats (a) verbalized P(yes) read at the *same* position
   (exp-17) and (b) the code-token probe reads (exp-29 max-pool / last-code-token).
   Hypothesis under test: signal the per-token probe missed could live here.

2. **Inputs**
   - *Forward* — exp-17's verbalized prompt VERBATIM (`build_content`: code, then
     "Does the code above contain a security vulnerability? … yes/no", chat
     template, `add_generation_prompt=True`, Qwen3 thinking-guard). Position =
     last input token (index −1), the one exp-17 read for P(yes).
   - *Capture* — `output_hidden_states=True`; save `hidden_states[i][:, -1, :]`
     for ALL layers i (repo-layer L = `hidden_states[L+1]`). One vector/example/
     layer. Persist on scratch (KEEP).
   - *Models* — the 6 verbalized it-models (user-confirmed): Qwen 7B/32B,
     gemma 1B/4B/12B/27B-it. Linear probe only (user-confirmed).
   - *Labels + split* — true fn label `row['label']`; seed-42 group-clean split
     (`load_or_make_split`), inner 15% val for layer + C selection. Same 292 test.

3. **Outputs** — `30-last-token-introspection/`
   - scratch: per-GPU shards → merged `lasttoken_hidden_<slug>.npz` (H[n,
     n_layers+1, hidden] **float32**, eid, label, is_test, p_yes, meta) via
     `merge_shards.py` with hard coverage/label/finite gates — KEEP.
   - `results/introspection_probe.json` — per model: deployable (val-selected
     layer+C) + oracle test AUC; **label-permutation null** (p-value, p95) and
     **random-direction null** (p-value, p95); **confounds** (language-indicator
     AUC, code-length AUC, within-Python / within-C AUC); verbalized (HARD-gated
     vs exp-17) + exp-29 reads; 1000-boot CIs + paired Δ vs verbalized.
   - figure: introspection probe vs verbalized vs code-token probe, example level.

4. **Result format** — per model: example-level AUC [95% CI] for the
   last-token-introspection probe (val-selected layer) vs verbalized vs exp-29
   max-pool/last-code-token; the probe's percentile in the label-permutation null;
   deployable-vs-oracle layer gap.

5. **Interpretation hints**
   - probe ≫ verbalized AND clears the permutation null ⇒ a real introspective
     linear direction the code-token probes missed (the one positive result that
     would reopen "knowledge hidden, not absent").
   - probe ≈ verbalized ⇒ the commit point holds no more than the model's own
     yes/no; introspection is not linearly decodable beyond what asking reveals.
   - probe ≈ chance / inside permutation null ⇒ nothing introspective here either;
     with the code-token (exp-20/24/29) and this read both null, the reading-frame
     negative is robust → motivates the generation pivot (NXT3).
   - probe high on test but inside permutation null ⇒ layer-selection overfit
     (d≫n artifact), NOT signal — the null is the guard.

## For agents

- **d ≫ n caveat (critical).** hidden_dim (2048–5120) ≫ n_train (1138), so an
  unregularized linear probe separates train trivially and layer-sweep selection
  inflates test AUC. Mitigations baked in: (1) strong L2 LR with C selected on
  val; (2) layer selected on val, reported on held-out test; (3)
  **label-permutation null** — retrain the identical pipeline (incl. layer
  selection) on shuffled labels K times, report the real probe as a percentile of
  that null. A test-AUC claim that doesn't clear the null is overfit, not signal.
- **Metric** = example-level AUC, SECONDARY per project-log §3; the token-level
  headline (`tokens_code_auc` 0.75–0.82) is untouched. Compare like-with-like vs
  verbalized (also example-level, same position).
- **Gate** — recomputed verbalized P(yes) test AUC from this forward must match
  exp-17 `verbalized_auc_test` (±~0.005); the assistant-boundary debug print
  (rendered tail has no `<think>`, yes/no dominate first-token argmax) carried
  from exp-17.
- **Compute** — adapt exp-17 `verbalized_logit_dump.py` (+`output_hidden_states`,
  +last-token capture). GPU-sharded by row. Train probes on-cluster (tiny); fetch
  small JSON + probe npz. Cluster wiring stays out-of-repo (de-clustered
  `run.sh` only in-repo). Layer convention: vLLM/HF `hidden_states[L+1]` = repo L.
- **Default extras (confirm in brief):** linear probe headline; neutral exp-17
  prompt only (no memory-prompt variants — that's exp-14's axis); MLP optional
  secondary.
- **Review** — design review (codex + Opus) BEFORE submit; result review gate
  BEFORE reporting; then ledger row + finding update.

## Design-review fixes applied (2026-06-14, codex + Opus → GO-WITH-FIXES)

- **float32 storage** (not fp16): late-layer "massive activations" exceed fp16's
  65504 → inf → silent layer drop. + finite assert on save and load.
- **merge_shards.py**: per-GPU shards → merged npz; HARD-fails unless every eid
  present once, all test eids present, npz label == dataset label, H finite;
  stamps is_test.
- **Null cost/validity**: label-permutation null reruns the *full* layer-selection
  on shuffled train/val labels (test intact) at fixed C=C*, N_PERM=1000,
  loky-parallel over memmapped stacked arrays (real process parallelism — threads
  were GIL-bound at ~2.4×). Reports proper p=(1+#≥)/(N+1). Plus a random-direction
  (untrained) null at the deployable layer.
- **Confounds** (codex HIGH; given exp-23 language is 64% of the token-level
  margin): example-level language-indicator AUC + code-length AUC + within-Python
  / within-C test AUC. NB at the *example* level language is ~0.5 by construction
  (each vuln is paired with its own-language fix), so a probe beating chance is not
  trivially language — but within-language AUC localizes where its signal lives.
- **Verbalized gate is HARD** (SystemExit on mismatch vs exp-17), not a warning.
- **C_GRID extended up** to 1.0 (smoke showed C=0.1 was the old grid edge; now
  interior); `C_at_grid_edge` flagged in output.
- **eager/all-layer exception**: reuses exp-17's `_load_model` (attn_impl=eager)
  for bit-fidelity to exp-17's verbalized read (the gate); `output_hidden_states`
  all layers. Smoke-validated: Qwen-32B extracts in ~7 min on one GH200, H finite.
- **Trainer is torch-free** (inlines split helpers) so it runs in the CPU env.
