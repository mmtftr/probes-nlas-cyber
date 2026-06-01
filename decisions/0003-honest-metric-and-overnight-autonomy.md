[ai-generated]

# 0003 — Honest token metric (`tokens_code`) + overnight autonomous-run decisions

Date: 2026-06-01
Status: Accepted (made autonomously by the agent overnight under explicit user
mandate; user asleep — to be reviewed on return). Supersedes the inflated
`tokens` metric as the headline for the cross-model sweep.

## Context

Reviewing 02–05 on the rebuilt SVEN before/after dataset (ADR 0002) surfaced two
things:

1. The layer sweep (`02-.../train_all_layers.py`) reports `test_tok_auc` =
   `roc_auc_score` over **every** test token. Per `src/eval/token_protocol.py`
   that is the `tokens` level, which is **inflated**: ~98% of negatives are
   trivial (comments, signatures, imports, whitespace), so the probe wins by
   keeping those low. The deployment-relevant metric — `tokens_code`, restricted
   to live-code tokens (tree-sitter, `src/eval/code_mask.py`) so negatives are
   real-code-but-not-vuln — was **never computed in any sweep**.
2. Example-level AUC collapsed to ~0.57–0.61 on the before/after data, but the
   user does not weight example-AUC ("not useful to me, just useful to see").

The user set an overnight goal: "complete the sweeps we agreed on" (framing-doc
sweeps 1, 2, 4, 5, 6) and authorized autonomous decision-making provided each
decision is documented. Exit condition: sweeps complete OR cluster access lost.

## Decision

**Headline metric = `tokens_code_auc`.** Every sweep reports it; `tokens_auc`
stays only as the inflation reference (the gap is itself a result). Example-AUC
rides along and gates nothing.

**Agreed sweep set (this run):**
- Exp **06** (`plans/.../06-honest-metric-sweeps/`): sweeps **1** (honest
  cross-model layer sweep), **2** (base↔instruct / Q5), **5** (re-run 03/04 on
  the honest metric), **6** (per-language / per-CWE breakdown at best layer).
- Exp **07** (`plans/.../07-train-code-masked-negs/`): sweep **4** (train-time
  live-code mask on negatives).
- Sweep **3** (proximity-window) **excluded** — user judged the W-dilation logic
  not yet trustworthy.

**Roster (8 models; gated → `secrets/hf_token`):**
`gemma-3-{1b,4b,12b,27b}-it`, `Qwen2.5-Coder-32B-Instruct`, and
`gemma-3-{1b,4b,12b}-pt` for the Q5 base↔instruct contrast. 27b-pt skipped to
bound cost (27b-it is the size anchor). Any gated-access failure → documented,
proceed with the accessible subset (resolves the §06 `TODO(adhoc-decision)` on
which `-pt` sizes).

**Layer selection (no test leakage):** best layer chosen by **max `val_ex_auc`**
(already recorded per layer); report `tokens_code_auc` *there* (the deployable
number) **plus** an explicitly-labelled oracle (argmax-test `tokens_code`) for the
val-vs-oracle gap. This is itself the data for the §8 layer-policy question.

**07 mask treatment:** masked (non-live-code, non-positive) tokens are
**excluded** from the span-max loss entirely — not down-weighted (cleanest test
of "were trivial negatives the crutch?"). Behind an additive flag
`--mask-negatives {none,code_only}`, default `none`, so 02/03/04 reproduce.

**Cluster env fix:** `tree-sitter` + `tree-sitter-{python,c,cpp}` were **absent**
from the cluster deps (`env.sh` DEPS) — `code_only_mask` would have silently
no-op'd and `tokens_code` would equal `tokens`. Added to `env.sh`; the `.deps_ok`
sentinel is busted once so they install. A smoke gate asserts
`dropped_fraction > 0` before any full run is trusted.

## Consequences

- 02's `metrics_*` JSONs are kept as the inflated-metric record for the gap
  comparison; 06 supersedes 02 as the live layer sweep. 02 scripts are NOT
  modified (06 has its own copies).
- All shared changes are additive (new flags/fields); existing experiments
  reproduce with defaults.
- Code lands on branch `honest-tokens-code-sweep`; the cluster checks it out.
  `runs/` and the HF cache were wiped by the user → every model re-pulls + all
  activations regenerate.
- Resolves the research-framing §8 headline-metric open item → **`tokens_code`**
  (not raw AUC / not example-AUC). The layer-policy item is what 06 measures;
  leave it open until 06's val-vs-oracle gap is in.
- Overnight execution is orchestrated by the agent: local code via an opus
  subagent, cluster driving via the authenticated `the cluster` tmux channel
  (`/tmp/ctmux.sh`); reconnect-on-drop is tolerated, loss-of-cert is the stop.

## Update (2026-06-01, user awake) — layer selection switched to `val_tokens_code`

The first 06 run (val_ex_auc selection) revealed val ex-AUC is near-chance
(0.50–0.59) on this data, so it selected near-random early layers (4b-it→L0,
12b-it→L2) that **badly undershoot** the oracle tokens_code layer (e.g. 12b-it:
val-sel 0.734 @ L2 vs oracle 0.799 @ L16). Per user ("pick according to
tokens_code auc"), selection now uses **`val_tokens_code_auc`** on a *leakage-free,
group-aware 15% val split carved from train* (seed 42, disjoint from the test
hold-out). 06 was re-run on the cached activations (retrain only; no
re-extraction). This is the deployable layer-selection rule going forward, and
the layers it picks feed sweep-5 (03/04) and exp-07.

First-run findings that already hold (metric-level, selection-independent):
- **`tokens_code` does not collapse** — ≈ `tokens` (slightly higher) on all 8.
- **Oracle tokens_code ~0.78–0.80** across 1B–32B, base and instruct.
- **Q5: pt ≈ it** at oracle → vuln direction is a *pretraining* feature, not
  installed by post-training.

## Open (for user review)
- `-pt` roster used = 1b+4b+12b (27b-pt skipped). Confirm/trim.
- The mask drops only ~30% of tokens on full-function SVEN (vs ~98% in the old
  truncation framing) — so `tokens_code` is a modest filter here; revisit if the
  threat model shifts back to streaming/truncated code.
