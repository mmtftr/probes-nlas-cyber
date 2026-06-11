[ai-generated]

# exp-20 RESULTS — Token-level FN/FP of the pooled vulnerability probe

Held-out **subtractive** test split: 97 vuln examples (+ paired safe), 7 probes
(Qwen-Coder 7B/32B, Gemma-3 1b/4b/12b-it/12b-pt/27b-it), each at its operating
layer, honest tight∩is_code labels, per-model F1-max threshold on train code
tokens. Source: `analysis.json`, `categorization.json`, `fp_buckets.json`.

## Headline

**The pooled span-max probe behaves as a lexical *string-sink detector*, not a
vulnerability detector — and it does so identically across Qwen and Gemma.** It
fires on SQL / OS-command / path string literals and their constituents. That one
behavior explains every quadrant:

- **TP** — it localizes the changed token of *injection* vulns (the sink string).
- **FN** — it misses *memory-safety* vulns entirely (no string sink to latch).
- **FP (spread)** — it spills across the rest of the tainted query string.
- **FP (safe-alarm)** — it fires on the **patched** code too, because the fixed
  version still contains the SQL/command string.

## Cross-model detection is bimodal and family-agnostic

Per vuln example, count how many of 7 probes fire on ≥1 true vulnerable code token:

| #models detecting | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| # examples (of 97) | 16 | 10 | 6 | 3 | 2 | 3 | 2 | **55** |

55/97 caught by **all 7**; 16/97 by **none**. The middle is nearly empty — a
vuln is either trivially visible to every probe or invisible to all.

- **Consistently detected (≥6/7): 57** — CWE-089 SQLi ×43, CWE-078 cmd-inj ×10,
  CWE-022 path ×3, CWE-079 XSS ×1. **All injection / string-sink.**
- **Consistently missed (≤1/7): 26** — CWE-416 UAF ×8, CWE-125 OOB-read ×7,
  CWE-476 null-deref ×3, CWE-022 ×2, CWE-787 ×2, CWE-190 ×2, CWE-079 ×2.
  **All memory-safety** (no localizable string sink).

**What differs between Qwen and Gemma: essentially nothing in *capability*.**
`qwen-only = 0`, `gemma-only = 4` (eids 500/1102/1152/1355, one each across
CWE-416/079/190/089 — scattered, not a coherent skill). The real family
difference is **calibration / trigger-happiness**, not which vulns:

| probe | layer | FP | TP | precision | recall | det-examples |
|---|---|---|---|---|---|---|
| qwen-32b | L25 | 1325 | 430 | 0.24 | 0.30 | 62/97 |
| qwen-7b | L16 | 778 | 324 | 0.29 | 0.22 | 57/97 |
| gemma-1b-it | L25 | 3480 | 474 | 0.12 | 0.30 | 70/97 |
| gemma-4b-it | L7 | 2811 | 534 | 0.16 | 0.34 | 66/97 |
| gemma-12b-it | L15 | 1703 | 369 | 0.18 | 0.23 | 65/97 |
| gemma-27b-it | L19 | 2882 | 461 | 0.14 | 0.29 | 68/97 |
| gemma-12b-pt | L13 | 1190 | 427 | 0.26 | 0.27 | 63/97 |

Gemma probes fire **2–4× more** (lower precision, slightly higher detection);
Qwen probes are more conservative (fewer FPs, higher precision). Same target,
different threshold behavior.

## False negatives — 6 categories (35 hardest-missed, n_detect ≤ 3)

Compiled from 3 sub-agents (`categorization.json`).

| # | Category | Count | What the fix changed |
|---|---|---|---|
| 1 | Missing-check insertion | 7 | fix ADDS a guard/bounds/null/error return — vuln is an *absence*, no token to localize |
| 2 | Use-after-free / stale-pointer / ordering | 6 | inserted/moved free·null·destroy·flag-set; no syntactic free marker (all CWE-416) |
| 3 | Integer-overflow / unvalidated size·count | 5 | size/count arithmetic overflows or is unchecked; bare operators carry no cue |
| 4 | Wrong pointer/cast/offset or comparison | 7 | subtle logic edit, correct-looking syntax, no token cue |
| 5 | Unbounded / wrong copy-or-read length | 3 | input-derived length to strcpy/memcpy/read exceeds buffer |
| 6 | Python injection on a non-sink token | 7 | fix touches a taint source, sanitizer, or template-render sink — not a SQL/command string |

**Meta-theme:** categories 1–5 (28 examples) are **all C/C++ memory-safety** and
share one root cause — *there is no localizable lexical sink token*; the
vulnerability is an absence, an ordering, or a subtle numeric/pointer change.
Category 6 (7 Python) is injection that the probe nonetheless misses because the
fix isn't on the SQL/command-string token it keys on (e.g. removed
`.replace(' ','-')` sanitizer, a `markdown()` render sink, a `request.args.get`
source).

## False positives — 5 categories

Compiled from 3 sub-agents over a 129-span curated sample, plus a reproducible
lexical breakdown over **all 6119** FP spans (`fp_buckets.json`).

| Category | sample (n=129) | population (n=6119) |
|---|---|---|
| SQL keyword/clause token (SELECT/WHERE/SET/FROM/LIKE/INSERT) | 42 | 929 (15%) |
| SQL identifier / column / value / placeholder (%s, table/col, tainted var) | 33 | 3111 (51%) |
| Format / punctuation / quote operator (%, .format, parens, quotes, dots) | 31 | 1527 (25%) |
| DB/cursor/ORM API call (cursor, execute, order_by) | 9 | 455 (7%) |
| OS-command / path / shell string | 15 | 97 (1.6%) |

Every category is a **constituent of a query/command string** — nothing keys on
the *injection itself* (taint reaching the sink unsanitized). Stratified by where
the FP lands (`analysis.json` → `fp_stats`):

- **safe-alarm — 3125 FPs fire on PATCHED code with no vulnerability at all.**
  Of the cross-model ones (≥2 models, 350), **336/350 are injection-CWE pairs**
  (225 CWE-089, 57 CWE-078, 34 CWE-022, 20 CWE-079); only ~14 are memory-safety.
  The probe cannot tell a parameterized/escaped query from a vulnerable one —
  the SQL string alone trips it.
- **spread — 1473 FPs** land within ~40 chars of the true span; SQL-keyword
  density peaks here (27%) → the probe paints the whole query clause, not just
  the changed token. 38 of the 48 most-shared (≥4-model) FPs are spread.
- **misplaced — 1521 FPs** land far from the bug, mostly on generic identifiers
  (59%) and punctuation (29%) — low-information firings inflated by the
  permissive F1-max threshold.

## Caveats
- Operating point = per-model F1-max on train code tokens (`TODO(adhoc-decision)`).
  At this point precision is low (0.12–0.29), so the *misplaced* / punctuation FPs
  are partly threshold noise; the **detection** (example-level) and **safe-alarm**
  findings are threshold-robust (bimodal histogram; safe-alarms clear threshold by
  large margins).
- The probe was trained on whole-line base labels; FN/FP measured here under the
  honest tight∩is_code label (label-definition shift, per exp-16 caveat). Detection
  is evaluated only on the **held-out** subtractive test split.
- FP sample counts are stratum-capped (50/45/35); population counts (`fp_buckets`)
  are the unbiased view. Deferred: per-CWE-trained probe FN/FP (needs a per-CWE
  logit dump we don't have locally — see repo `todo.md`).
- "Detected iff ≥1 in-span token clears threshold" is a *generous* recall bit:
  precision is low (FP ≫ TP), so a true-span token clearing threshold is weak
  evidence of clean localization. The injection-vs-memory contrast is robust to
  this; the word "detected" is not "cleanly localized."
- 1 of the 97 test-vuln examples (eid 45, CWE-125) has its entire tight diff on
  non-code tokens → zero positive labels → a *structural* auto-miss, correctly
  in `consistently_missed`. It reinforces the memory-safety blindspot rather than
  distorting it (verified by exp-20 review, agent a00891352cf6db2cd).

## So what
The probe's headline token-AUC (~0.74–0.78) is carried almost entirely by
**lexical SQL/command-string recognition**, generalizing trivially across model
families because that cue is model-independent. It is blind to the memory-safety
half of SVEN and cannot distinguish vulnerable from patched injection code. Any
"vulnerability probe" claim should be reported **split by CWE class**, and the
honest hard problem is localizing memory-safety vulns that have no lexical sink.
