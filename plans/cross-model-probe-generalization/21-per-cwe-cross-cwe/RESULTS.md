[ai-generated]

> **This is the corrected exp-21.** It supersedes two earlier wrong versions (git
> history): (1) the **retracted pair-accuracy** report, and (2) a **rescore of
> exp-21's saved probes** (`recompute_tokenauc.py`, `matrix_tokenauc.json`) that
> reported memory near-chance — that was a **training-regime artifact**: those
> probes were trained *one-vs-matched-patch* with *difflib subtractive* labels,
> which (a) give additive memory fixes an empty positive span and (b) give the
> probe ~no negative diversity. Trained the **exp-10 way** (below), memory recovers.
> Passed the review gate (Opus + codex/Azure pre-exec; bit-exact exp-10 diagonal is
> the built-in correctness gate).

# exp-21 RESULTS — Cross-CWE transfer matrix on `tokens_code_auc` (exp-10 recipe)

Per-CWE span-max linear probes trained the **exp-10 way** and evaluated as a
CWE×CWE **transfer** matrix. Recipe (reproduces exp-10 exactly): positives =
annotated `token_labels` spans (`parse_spans`, label 1 — populated for additive
fixes too); train CWE-X probe on **{CWE-X vuln} ∪ {ALL `cwe==null` clean}**, full
SVEN, all tokens; 15% group-aware val carve (seed-42) excluded; eval =
`tokens_code_auc` (live-code only) on **{CWE-Y test pos} ∪ {all clean test}**,
shared clean-test negatives. Qwen2.5-Coder-32B L25 + Gemma-3-1b-it L25, reusing
exp-21's KEPT full-SVEN activations (no re-extraction). Source:
`results/*/transfer_allclean.json`, `transfer_allclean.py`, `figs/allclean_*`.

## 1. Self-detection (diagonal) reproduces exp-10 — memory IS learnable on its own data

Qwen-32B diagonal reproduces exp-10's specialized `tokens_code_auc` **bit-exact
(Δ = ±0.000 on all 9 CWEs)**:

| CWE | this (Qwen-32B) [95% CI] | exp-10 | Gemma-1b |
|---|---|---|---|
| CWE-089 SQLi | 0.983 [0.98, 0.99] | 0.983 | 0.971 |
| CWE-022 path | 0.943 [0.91, 0.97] | 0.943 | 0.916 |
| CWE-078 cmd-inj | 0.923 [0.84, 0.98] | 0.923 | 0.916 |
| CWE-079 XSS | 0.863 [0.80, 0.94] | 0.863 | 0.909 |
| **CWE-416 UAF** | **0.766** [0.65, 0.85] | 0.766 | 0.769 |
| **CWE-125 OOB-read** | **0.732** [0.67, 0.80] | 0.732 | 0.734 |
| **CWE-476 NULL-deref** | **0.640** [0.58, 0.73] | 0.640 | 0.619 |
| CWE-787 OOB-write * | 0.670 [0.61, 0.76] | 0.670 | 0.724 |
| CWE-190 int-ovf * | 0.767 [0.70, 0.88] | 0.767 | 0.857 |

`*` test n<10 (untrusted). Memory CWEs reach **0.64–0.77** when trained on their
own data — confirming exp-10 / project finding #3. Gemma independently corroborates
(memory 0.62–0.77); it is *not* in exp-10's per-CWE table, so its row is its own
result, not a reproduction.

## 2. Cross-CWE transfer is family-structured (the new finding)

Block-mean `tokens_code_auc` (trusted CWEs, n_test≥10: inj = 089/078/022/079,
mem = 125/416/476; weighted by test pos-tokens, 95% bootstrap CI):

| block | Qwen-32B | Gemma-1b |
|---|---|---|
| **inj → inj** (incl. self) | 0.688 [0.659, 0.714] | 0.670 [0.641, 0.700] |
| inj → inj *off-diagonal* | **0.600 [0.563, 0.635]** | **0.579 [0.540, 0.616]** |
| **mem → mem** (incl. self) | 0.618 [0.591, 0.649] | 0.650 [0.620, 0.682] |
| mem → mem *off-diagonal* | **0.570 [0.536, 0.607]** | **0.619 [0.585, 0.655]** |
| inj → mem | **0.411 [0.375, 0.447]** | **0.382 [0.348, 0.416]** |
| mem → inj | **0.341 [0.312, 0.377]** | **0.300 [0.266, 0.332]** |

- **Within-family transfer is real** — both off-diagonal CIs exclude 0.5 *from
  above*. An injection probe partially detects other injection CWEs (e.g. Qwen
  022↔079 0.70–0.83, 089→078 0.63); a memory probe partially detects other memory
  CWEs (476↔416 0.62–0.64, 476→125 0.60).
- **Cross-family transfer is below chance on average** — inj→mem and mem→inj block
  CIs exclude 0.5 *from below* (most inj↔mem cells <0.5; injection and memory
  directions are anti-correlated on average, though a few individual cells sit just
  above 0.5).

## So what

Transfer is **family-structured**: the data argue against both a single universal
"vulnerability" direction and purely CWE-specific detectors — they support **at
least two coarse family-level directions** (a taint/string-sink injection direction
and a memory-safety direction), each transferring within its family while
cross-family transfer is below chance on average. Both are individually learnable
(diagonal). The within-family memory transfer is real but **modest** (off-diag 0.57,
one cell at chance) — weaker/patchier than injection's. This **confirms finding #3** (the memory signal exists as its own
direction — the general pooled probe under-allocates to it, per exp-09/11) and
**sharpens finding #4** (exp-20's pooled "string-sink detector" is the *injection*
direction; a separate memory direction exists but the single general probe doesn't
capture it). The deployable implication: a per-family (≥2-head) probe, not a single
linear direction.

## Caveats
- CWE-190 / CWE-787 have test n<10 → excluded from trusted blocks (hatched in
  figures); the 190 family label (injection vs memory) is moot here as it is dropped.
- Linear span-max head, one operating layer per model (exp-16 best L25).
- Gemma-1b has no exp-10 per-CWE table → its diagonal is independent corroboration.
- Backend = HF (consistent with exp-16/19). Probes saved (`probes_allclean.npz`).
