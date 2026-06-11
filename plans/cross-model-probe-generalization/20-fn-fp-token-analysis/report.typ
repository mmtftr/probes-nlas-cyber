// [ai-generated] — typeset render of RESULTS.md. Compile from repo root:
//   typst compile --root /Users/mmtf/p/probes-nlas-cyber \
//     plans/cross-model-probe-generalization/20-fn-fp-token-analysis/report.typ
#import "/docs/templates/report.typ": report, callout, finding, statrow, accent, accent2, muted

#show: report.with(
  title: "What the vulnerability probe actually detects",
  subtitle: "Token-level false-negative / false-positive error analysis on SVEN-subtractive — Qwen vs Gemma",
  author: "exp-20",
  date: "2026-06-07",
)

#finding(label: "Thesis")[
  The pooled span-max vulnerability probe behaves as a *lexical string-sink
  detector, not a vulnerability detector* — and does so *identically across Qwen
  and Gemma*. It fires on SQL / OS-command / path string literals. That one
  behavior explains all four quadrants: *TP* on the changed token of injection
  vulns, *FN* on memory-safety vulns (no string to latch), *FP-spread* across the
  rest of the tainted query, and *FP-safe-alarm* on patched code that still
  contains the SQL string.
]

#statrow((
  ([55 / 97], "caught by all 7 probes"),
  ([16 / 97], "caught by none"),
  ([0 vs 4], "qwen-only vs gemma-only"),
  ([3 125], "FPs on patched code"),
))

Held-out *subtractive* test split: 97 vulnerable examples (+ paired safe), 7
probes (Qwen2.5-Coder 7B/32B; Gemma-3 1b/4b/12b-it/12b-pt/27b-it), each at its
operating layer, honest tight∩`is_code` labels (ADR-0004), per-model F1-max
threshold fit on train code tokens. A vuln example counts as *detected* iff ≥1 of
its true vulnerable code tokens clears the probe's threshold.

= Detection is bimodal and family-agnostic

#table(
  columns: (auto,) + 8 * (1fr,),
  align: center + horizon,
  table.header([Probes detecting], [0], [1], [2], [3], [4], [5], [6], [7]),
  [Examples (of 97)], [16], [10], [6], [3], [2], [3], [2], [*55*],
)

A vuln is either trivially visible to every probe or invisible to all — the
middle is nearly empty. The split is by *CWE class*:

#grid(columns: (1fr, 1fr), column-gutter: 12pt,
  callout(title: "Consistently detected — ≥6/7 : 57", bar: accent)[
    All *injection / string-sink*: \
    CWE-089 SQLi ×43 · CWE-078 cmd-inj ×10 · CWE-022 path ×3 · CWE-079 XSS ×1.
  ],
  callout(title: "Consistently missed — ≤1/7 : 26", bar: accent2, fill: rgb("#f7ece9"))[
    22/26 *memory-safety*: \
    CWE-416 UAF ×8 · CWE-125 OOB-read ×7 · CWE-476 null-deref ×3 · CWE-787 ×2 ·
    CWE-190 ×2 (+ CWE-022 ×2, CWE-079 ×2).
  ],
)

*Qwen vs Gemma differ only in calibration, not capability.* `qwen-only = 0` and
`gemma-only = 4` (scattered across CWE-416/079/190/089 — no coherent skill).
Gemma probes fire 2–4× more (lower precision, slightly higher detection); Qwen
probes are conservative.

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: (left, center, right, right, center, center, center),
  table.header([Probe], [Layer], [FP], [TP], [Precision], [Recall], [Det. ex.]),
  [qwen-32b], [L25], [1325], [430], [0.24], [0.30], [62/97],
  [qwen-7b], [L16], [778], [324], [0.29], [0.22], [57/97],
  [gemma-1b-it], [L25], [3480], [474], [0.12], [0.30], [70/97],
  [gemma-4b-it], [L7], [2811], [534], [0.16], [0.34], [66/97],
  [gemma-12b-it], [L15], [1703], [369], [0.18], [0.23], [65/97],
  [gemma-27b-it], [L19], [2882], [461], [0.14], [0.29], [68/97],
  [gemma-12b-pt], [L13], [1190], [427], [0.26], [0.27], [63/97],
)

= False negatives — 6 categories (35 hardest-missed, n#sub[detect] ≤ 3)

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, center),
  table.header([Category], [What the fix changed], [Count]),
  [Missing-check insertion], [fix *adds* a guard/bounds/null/error return — the vuln is an *absence*, no token to localize], [7],
  [Use-after-free / stale-pointer / ordering], [inserted/moved free·null·destroy·flag-set; no syntactic free marker (all CWE-416)], [6],
  [Integer-overflow / unvalidated size·count], [size/count arithmetic overflows or is unchecked; bare operators carry no cue], [5],
  [Wrong pointer/cast/offset or comparison], [subtle logic edit, correct-looking syntax, no token cue], [7],
  [Unbounded / wrong copy-or-read length], [input-derived length to strcpy / memcpy / read exceeds buffer], [3],
  [Python injection on a *non-sink* token], [fix touches a taint source, sanitizer, or template-render sink — not a SQL/command string], [7],
)

#callout(title: "Meta-theme")[
  Categories 1–5 (*28 examples*) are *all C/C++ memory-safety* and share one root
  cause: there is *no localizable lexical sink token* — the vulnerability is an
  absence, an ordering, or a subtle numeric/pointer change. Category 6 (7 Python)
  is injection the probe still misses because the fix is not on the
  SQL/command-string token it keys on (a removed `.replace(' ','-')` sanitizer, a
  `markdown()` render sink, a `request.args.get` source).
]

= False positives — 5 categories

Compiled from a 129-span curated sample (sub-agents) plus a reproducible lexical
breakdown over *all 6 119* FP spans.

#table(
  columns: (1fr, auto, auto),
  align: (left, center, center),
  table.header([Category (a constituent of a query/command string)], [Sample (n=129)], [Population (n=6119)]),
  [SQL keyword / clause (`SELECT WHERE SET FROM LIKE INSERT`)], [42], [929],
  [SQL identifier / column / value / placeholder (`%s`, table·col, tainted var)], [33], [3111],
  [Format / punctuation / quote operator (`%` `.format` parens quotes dots)], [31], [1527],
  [DB / cursor / ORM API call (`cursor` `execute` `order_by`)], [9], [455],
  [OS-command / path / shell string], [15], [97],
)

Nothing keys on the *injection itself* (taint reaching the sink unsanitized).
Stratified by where the FP lands:

- *safe-alarm — 3 125 FPs fire on patched code with no vulnerability at all.* Of
  the cross-model ones (≥2 models, 350), *336/350 are injection-CWE pairs* (225
  CWE-089, 57 CWE-078, 34 CWE-022, 20 CWE-079); only \~14 are memory-safety. The
  probe cannot tell a parameterized query from a vulnerable one — the SQL string
  alone trips it.
- *spread — 1 473 FPs* land within \~40 chars of the true span; SQL-keyword
  density peaks here (27%) → the probe paints the whole clause, not just the
  changed token. 38 of the 48 most-shared (≥4-model) FPs are spread.
- *misplaced — 1 521 FPs* land far from the bug, mostly on generic identifiers
  (59%) and punctuation (29%) — low-information firings inflated by the permissive
  F1-max threshold.

= Takeaway

The probe's headline token-AUC (\~0.74–0.78) is carried almost entirely by
*lexical SQL/command-string recognition*, which generalizes trivially across
model families because that cue is model-independent. It is *blind to the
memory-safety half of SVEN* and *cannot distinguish vulnerable from patched
injection code*. Any "vulnerability probe" claim should be reported *split by CWE
class*; the honest hard problem is localizing memory-safety vulns that have no
lexical sink.

#v(6pt)
#text(size: 8pt, fill: muted)[
  *Caveats.* Operating point = per-model F1-max on train code tokens; at this
  point precision is low (0.12–0.29), so misplaced/punctuation FPs are partly
  threshold noise — the bimodal detection and safe-alarm findings are
  threshold-robust. Probe trained on whole-line base labels; FN/FP measured under
  the honest tight∩`is_code` label on the held-out test split. 1 of 97 (eid 45,
  CWE-125) is a structural auto-miss (entire diff on non-code tokens). Reviewed
  adversarially — no conclusion-changing bug. Source: `analysis.json`,
  `categorization.json`, `fp_buckets.json`, `RESULTS.md`. Deferred: per-CWE-trained
  probe FN/FP (needs a per-CWE logit dump not held locally — see repo `todo.md`).
]
