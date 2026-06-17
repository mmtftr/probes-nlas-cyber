[ai-generated]

# 29 — Last-code-token readout vs verbalized (example-level, no new compute)

Adds the "last-token probe" the user asked for to the blog's verbalized
comparison, and fixes that figure's metric mismatch (it currently plots a
TOKEN-AUC probe against an EXAMPLE-AUC verbalized read — see the
`TODO(metric-fairness)` markers in `index.qmd`). Pure CPU rescore of exp-16's
saved per-token logits + exp-17's saved verbalized scores. No GPU, no retraining,
no extraction.

## The five fields

1. **Aim** — Read the *existing* span-max vulnerability probe at each test
   function's **final live-code token**, turn that single score into an
   example-level prediction, and compare it like-with-like against (a) the
   model's verbalized P(yes), (b) the max-pooled and (c) mean-pooled token
   probe. Hypothesis: the final code token carries little example-level signal
   beyond what pooling already gives — the probe is a *distributed* lexical
   detector (exp-20/24), not a commit-point reader.

   > **Agent's concern (fenced, no decision weight).** The last *code* token in
   > exp-16's raw-code extraction format is NOT the `Assistant:` turn-boundary
   > that the blog's NXT2b ("last-token introspection") ultimately targets — that
   > position only exists in the verbalized QA format (exp-17) and its hidden
   > state was never saved, so the true NXT2b probe needs a (small) new
   > extraction. This experiment is the cheap, saved-logits version the user
   > chose. A near-chance result is the *expected* outcome and is itself the
   > finding (signal is not concentrated at the final code position); it does
   > NOT bear on the token-level headline.

2. **Inputs**
   - *Probe logits* — exp-16 `logits_layer{NN}.npz` at each model's canonical
     operating layer (the blog headline layer): Qwen-32B L25, Qwen-7B L16,
     gemma-1b L25, gemma-4b L7, gemma-12b-it L15, gemma-12b-pt L13, gemma-27b L19.
     Schema: `logit, prob, y, example_id, char_start, char_end, is_test, is_code`.
     (2 anchors local; 5 pulled from the remote cache.)
   - *Verbalized* — exp-17 `example_scores_verbalized.json` (`eid, p_yes, label`),
     6 it-models (gemma-12b-pt has no verbalized run — probe-only there).
   - *Labels + split* — `data/dataset.jsonl` (TRUE function vuln label) +
     `data/sven_split_meta.json` (seed-42, 20% group-clean held-out; verbatim
     `load_or_make_split`). Test pool = the same 292 functions the verbalized
     read used.

3. **Outputs** — `29-last-token-readout/results/`
   - `last_token_readout.json` — per model × pool ∈ {last_code_token, max_pool,
     mean_pool, verbalized} → example-AUC (+ 1000-resample bootstrap CI, n) on
     {full test 292, subtractive-test subset}; paired probe−verbalized Δ-CIs.
   - `fig_last_token.png` (+ `-dark`) — grouped bars per model, all example-level.

4. **Result format** — per model: example-level ROC-AUC [95% CI] for
   last-code-token / max-pool / mean-pool / verbalized, on the full test pool
   (primary, matches the verbalized figure) and the subtractive subset
   (probe-fair robustness). Plus the headline sentence for the blog: "at the
   example level, last-code-token probe = X vs verbalized = Y."

5. **Interpretation hints**
   - last-code-token ≈ 0.5 ⇒ no commit-point signal in the code probe; expected;
     motivates the real assistant-boundary probe (NXT2b) as a follow-up.
   - last-code-token ≈ max-pool ⇒ the final token already carries the pooled
     signal (unlikely; max-pool saturates, see below).
   - last-code-token > verbalized ⇒ even a degenerate probe read beats asking
     the model (introspection gap persists at the example level).
   - last-code-token < verbalized ⇒ asking the model is at least as good as this
     probe read — supports softening the blog's "probe is the better reader."

## For agents

- **Metric discipline (review gate).** Everything here is **example-level AUC**,
  a SECONDARY metric per `docs/project-log.md` §3. It must be labelled secondary
  and must NOT be used to make any "signal absent / probe works" claim about the
  token-level headline (`tokens_code_auc` 0.75–0.82 stands untouched). The only
  claim this supports is the narrow example-level, single-position comparison.
- **AUC is monotone-invariant**, so logit vs prob give identical AUC for
  last-token and max-pool — but `prob` saturates to 1.0 in float32 (ties → AUC
  0.5), so ALL reads (last/max/mean) use the raw `logit`. Mean-pool = mean of
  logit (not prob), secondary.
- **Last code token** = for each example, the token with `is_code==True` having
  the largest sequence index (tokens are concatenated in order within each
  `example_id`); equivalently largest `char_start` among its code tokens. Score =
  that token's `logit`.
- **Label = TRUE function vuln label** (`row["label"]`) for ALL four reads, so the
  probe shares the verbalized read's exact axis. This deliberately differs from
  exp-16's `example_scores` label (`max token y > 0`), which silently relabels
  *additive*-vulnerable functions to 0. Consequence: on the full pool the token
  probe is penalized on additive-vulnerable pairs (no token to fire on — the
  known additive blind spot, ADR-0004). The **subtractive-test subset** is the
  probe-fair cut where token-label and true-label agree; report both.
- **Verbalized re-AUC**: recompute from `example_scores_verbalized.json`
  restricted to test eids (don't just copy `verbalized_auc_test`) so pool + label
  definition are bit-identical across all reads. Gate: must reproduce exp-17's
  `verbalized_auc_test` (±~0.01).
- **CIs**: 1000-resample example bootstrap; paired bootstrap for the
  probe−verbalized Δ per model (same resample indices).
- **Repro gate (as implemented)**: recompute `tokens_code_auc` from the npz
  (AUC over test live-code tokens) and HARD-FAIL vs exp-16's stored
  `test_tokens_code_auc` — non-degenerate, proves logit/is_code/is_test/eid
  alignment. (An earlier draft proposed an all-token max-pool gate; dropped — it
  saturates to AUC 0.5 and proves nothing.) For the 5 reduced-path models the
  gate compares the cluster-computed `tokens_code_auc` (numpy rank-AUC, tol 1e-4).
- **Review**: pre-execution design review (codex + Opus) BEFORE running; result
  review gate (codex + Opus) BEFORE reporting. Then update the ledger row.
