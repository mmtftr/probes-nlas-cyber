[ai-generated]

# 19 — Subtractive-regime probe training

Driven autonomously overnight (2026-06-06) under user delegation; decisions
recorded in the session log + ADR. Motivated by a label-quality bug found via
the exp-16 logit explorer: whole-line, diff-derived evidence marks comments and
unchanged tokens as "vulnerable" (e.g. eid 480 FliDecode: the marked span is a
comment; the real CWE-125 fix is an unmarked addition).

## Aim
Does training the span-max probe on a *cleaned* label regime change what it
learns? Two fixes: (1) restrict to **subtractive** vulnerabilities (the fix
deletes/replaces real code, so the vuln is token-localizable; additive/cosmetic
fixes dropped); (2) **per-token** (tight changed chars) vs whole-line labels, and
**comments-as-negative (Y)** vs **comments-ignored (X)** — non-code tokens never
positive either way.

## Inputs
- Models: Qwen2.5-Coder-7B/32B-Instruct; gemma-3 {1b,4b,12b,27b}-it + 12b-pt (7).
- Activations: re-extracted per model at its historical layer band (full base
  SVEN, max_length 2048), then deleted in-job. (exp-16 deleted the raw acts; they
  don't depend on labels so one extraction serves the whole grid.)
- Dataset: `data/dataset.jsonl` (1430 ex, md5 5a27465…) + `sven_split_meta.json`
  (seed-42 group-clean). Subtractive subset: `subtractive_membership.json`
  (956 ex = 478 vuln+safe pairs; 237 additive pairs dropped; char-level
  tree-sitter membership, model-independent).
- Probe/loss: canonical span-max linear, `train_one_layer`, **loss unchanged**.
  X = `mask_negatives="code_only"`, Y = `"none"`. Positives = granularity ∩ is_code.

## Grid (per model × band layer)
2×2×2 = 8 probes: train_subset {base, subtractive} × granularity {line, token}
× negatives {X, Y}. SVEN-base retrained here only for cross-comparison; the
canonical base result stays exp-16.

## Outputs
- `./runs/subtractive_<slug>/metrics_grid.json` — per config:
  AUCs on subtractive-test, base-test, additive-test (common honest eval) + probe
  npz per config. Collected into `results/` here.

## Result format
Per model, at the operating layer: a table of the 8 configs ×
{subtractive-test, base-test, additive-test} token-code-AUC, plus the
train→eval cross-subset matrix (base-trained vs subtractive-trained, evaluated
on each subset). Headline: does subtractive+token+X beat the exp-16 baseline on
the honest (tight, code-only) eval, and how do base-trained probes transfer to
the subtractive test (and vice-versa)?

## Interpretation hints
- **Common eval** = tight-token ∩ is_code positives, code-only tokens, so Y's
  easy-negative inflation can't bias the comparison.
- If **subtractive-trained ≈ base-trained on subtractive-test** → the additive
  third was just noise; dropping it costs nothing and cleans the signal.
- If **base-trained ≫ subtractive-trained on base-test** but **≈ on
  subtractive-test** → base-training's extra "signal" was the additive/whole-line
  artifact, not transferable vuln structure.
- **token vs line**: if token ≥ line on the honest eval, the tight labels are
  better-aligned; if token ≪ line, whole-line context was actually helping (the
  probe needs the surrounding line, not just the changed chars).
- **X vs Y on the common eval**: should be close; large gaps mean the negative
  set materially shapes the decision boundary, not just the metric.
