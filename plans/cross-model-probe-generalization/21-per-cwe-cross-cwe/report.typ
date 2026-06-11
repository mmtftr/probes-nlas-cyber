// [ai-generated] typeset render of exp-21 RESULTS.md (CORRECTED — exp-10 recipe).
// Supersedes the retracted pair-accuracy report AND the matched-patch rescore.
// Compile from repo root:
//   typst compile --root /Users/mmtf/p/probes-nlas-cyber \
//     plans/cross-model-probe-generalization/21-per-cwe-cross-cwe/report.typ
#import "/docs/templates/report.typ": report, callout, finding, statrow, accent, accent2, muted

#show: report.with(
  title: "Cross-CWE transfer on tokens_code_auc",
  subtitle: "Per-CWE probes trained vs all-clean (exp-10 recipe), extended to the full transfer matrix — Qwen-32B & Gemma-1b, SVEN",
  author: "exp-21 (corrected)",
  date: "2026-06-09",
)

#callout(title: "Corrects two earlier versions", bar: accent2)[
  Supersedes (1) the retracted *pair-accuracy* report and (2) a rescore of exp-21's
  *matched-patch-trained* probes that showed memory near-chance — a *training-regime
  artifact* (subtractive difflib labels give additive memory fixes an empty positive
  span; matched-patch negatives give ~no diversity). Trained the *exp-10 way* (vs
  ALL-clean, annotated `token_labels` positives, full SVEN), memory recovers. Two
  independent review passes (Opus + codex/Azure) cleared it.
]

#finding(label: "Two results")[
  *(1)* Self-detection reproduces exp-10 *bit-exact* (Qwen Δ=±0.000): injection
  0.86–0.98, *memory 0.64–0.77* — memory IS learnable when trained on its own data.
  *(2)* Cross-CWE transfer is *family-structured*: within-family off-diagonal blocks
  are above chance (inj 0.60, mem 0.57), cross-family blocks below chance (inj→mem
  0.41, mem→inj 0.34) ⟹ ≥2 coarse family directions, not one universal nor purely
  per-CWE.
]

#statrow((
  ([Δ ±0.000], "Qwen diagonal vs exp-10"),
  ([0.64–0.77], "memory self-detect"),
  ([0.60 / 0.57], "inj / mem within-family transfer"),
  ([0.41 / 0.34], "cross-family (< chance)"),
))

Recipe = exp-10: positives = annotated `token_labels` spans; train CWE-X probe on
*{CWE-X vuln} ∪ {ALL `cwe==null` clean}*, full SVEN, all tokens; 15% group-aware
val carve (seed-42) excluded; eval `tokens_code_auc` (live-code only) on *{CWE-Y
test pos} ∪ {all clean test}*, shared clean-test negatives. Reuses exp-21's KEPT
full-SVEN activations (no re-extraction).

= The transfer matrix

Row = TRAIN-probe CWE; column = TEST CWE. Green = self-detection; hatched = test
n\<10 (noise). Block-diagonal: injection (top-left) hot, memory (bottom-right) warm
on its own block, cross-family cold.

#figure(image("figs/allclean_transfer_heatmaps.png", width: 100%))

= Self-detection reproduces exp-10 · transfer blocks

#grid(columns: (1fr, 1fr), column-gutter: 10pt,
  image("figs/allclean_diagonal_repro.png", width: 100%),
  image("figs/allclean_blocks.png", width: 100%),
)

#table(
  columns: (auto, auto, auto),
  align: (left, center, center),
  table.header([block (trusted n≥10, 95% boot CI)], [Qwen-32B], [Gemma-1b]),
  [inj → inj *off-diagonal*], [*0.600* [.56,.64]], [*0.579* [.54,.62]],
  [mem → mem *off-diagonal*], [*0.570* [.54,.61]], [*0.619* [.59,.66]],
  [inj → mem], [0.411 [.38,.45]], [0.382 [.35,.42]],
  [mem → inj], [0.341 [.31,.38]], [0.300 [.27,.33]],
)

Within-family off-diagonal CIs exclude 0.5 from *above* (transfer is real);
cross-family CIs exclude 0.5 from *below* (the two directions are anti-correlated on
average). Memory within-family transfer is real but *modest* (one cell at chance) —
weaker than injection's.

= So what

There are *at least two coarse family-level directions* — a taint/string-sink
injection direction and a memory-safety direction — each transferring within its
family, mutually anti-correlated across families, both individually learnable. This
*confirms finding #3* (the memory signal exists as its own direction; the single
general probe under-allocates to it, per exp-09/11) and *sharpens finding #4*
(exp-20's pooled "string-sink detector" is the injection direction). Deployable
implication: a per-family / ≥2-head probe, not one linear direction.

#v(6pt)
#text(size: 8pt, fill: muted)[
  *Caveats.* CWE-190/787 test n\<10 → excluded from trusted blocks (hatched); the
  190 family label is moot (dropped). Linear span-max head, one operating layer
  (L25). Gemma-1b has no exp-10 per-CWE table → its diagonal is independent
  corroboration, not a reproduction. Metric `tokens_code_auc` throughout. Probes
  saved (`probes_allclean.npz`). Source: `results/*/transfer_allclean.json`,
  `transfer_allclean.py`, `make_figures_allclean.py`.
]
