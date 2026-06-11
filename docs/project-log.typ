// [ai-generated] typeset summary of docs/project-log.md. Compile from repo root:
//   typst compile --root /Users/mmtf/p/probes-nlas-cyber docs/project-log.typ docs/project-log.pdf
#import "/docs/templates/report.typ": report, callout, finding, statrow, accent, accent2, muted

#show: report.with(
  title: "Probes-NLAs — project state",
  subtitle: "High-level log: goal · what we know · conventions · experiment ledger · open threads (summary of docs/project-log.md)",
  author: "project-log",
  date: "2026-06-07",
)

#finding(label: "Where the project stands")[
  There is a *real, honest token-level vulnerability signal* in code-LLM hidden
  states (`tokens_code_auc` ≈ 0.75–0.82). It splits sharply by vuln class:
  *injection is strong* (SQLi ≈ 0.92–0.96), *memory-safety is weak for the general
  probe* (≈ 0.43–0.59) — but that gap is *capacity-allocation, not signal-absence*:
  per-CWE specialization recovers memory (CWE-125 0.73, CWE-416 0.77). The pooled
  probe behaves as a *lexical string-sink detector*.
]

#statrow((
  ([0.75–0.82], "tokens_code_auc (honest)"),
  ([0.92–0.96], "injection (CWE-089)"),
  ([0.57→0.73], "memory CWE-125: gen→specialized"),
  ([20], "experiments run"),
))

= Goal & scope

*Target property:* the model's own internal belief about how vulnerable the code
in its input is — a linear read off hidden states, output as a calibrated
probability. Fit on *SVEN* before/after function pairs (vulnerable vs its fix),
token-level supervision with char-span localization.

*Scope:* Gemma-3 (1B–27B, it+pt) and Qwen2.5-Coder / Qwen3. Linear *span-max*
probes (small MLP variants tried), mid–late layers.

= What we currently know

+ *A real honest token signal exists* — `tokens_code_auc` ≈ 0.75–0.82; not a
  trivial-negative artifact (live-code-only barely drops it; train-time masking is
  a no-op). [exp-06/07/16, ADR-0003]
+ *Strong class split (the central finding):* injection (Python: SQLi/cmd-inj/path/
  XSS) strong (0.82–0.96); memory-safety (C: OOB/UAF/NULL/int-ovf) weak for the
  *general* probe (0.43–0.59). [exp-06]
+ *The memory gap is capacity-allocation, not absence.* Per-CWE specialization
  recovers it — CWE-125 0.57→*0.73*, CWE-416 0.44→*0.77*, CWE-787 0.46→0.67,
  CWE-476 0.49→0.64 (Δ +0.15–0.33) [exp-10]. Family-balancing is a partial fix
  (memory +0.06) [exp-11]; ensembles of K directions do *not* unlock it (capacity
  not the lever) [exp-09]; MLP heads are scale-flat [exp-04/12/18].
+ *Pooled probe = lexical string-sink detector.* Fires on SQL/command/path string
  literals → catches injection sinks, misses memory (no sink), false-alarms on
  patched code still holding the SQL string. Bimodal, family-agnostic. [exp-20]
+ *Verbalized side weak.* Probe > the model's own yes/no for Gemma (+0.09
  introspection gap), ~tied for Qwen [exp-05/17]. The memory verbalized blind spot
  is a *prompt-framing artifact* — memory-specific prompts recover +0.21–0.33
  [exp-14]; prompt-spec mirrors probe-spec [exp-15].
+ *Causality:* the memory probe direction is *epiphenomenal* — steering it at ±4σ
  doesn't move verbalized P(yes); a correlate, not a control knob [exp-13].
+ *Additive fixes (~⅓ of SVEN) are undetectable* by a token-localized probe
  (pairAcc ≈ chance) → token-probe work uses the subtractive subset [exp-19,
  ADR-0004].

= Standing conventions

#callout(title: "Default metric")[
  *`tokens_code_auc`* — honest token-level ROC-AUC over live-code tokens only
  (tree-sitter mask). The headline for every probe eval. Example-AUC, pair-ranking,
  and detection-rate are *secondary* and must be labelled so — never base a
  "signal absent or works" claim on a non-default metric.
]

- *Dataset:* SVEN before/after (1430 rows); for token probes, the *subtractive*
  subset (956 ex / 478 pairs, localizable-fix-only) [ADR-0004].
- *Split:* group-clean at pair level, seed-42, 20% held-out (pairs never straddle);
  inner 15% val for layer/epoch selection.
- *Layer:* per-model by max `val_tokens_code_auc`. Repo-layer L = `hidden_states[L+1]`.
- *Negative pool (specialized probes):* all `cwe==null` clean rows (exp-06/10 recipe).
- *Cluster:* the cluster `debug` only (1.5 node-h/job, MaxJobs=1); `fc` for unattended.
  Default extractor vLLM (HF fallback `--backend hf`). KEEP operating-layer acts.
- *Review gate:* every result passes a cj/codex + Opus-subagent review before it
  reaches the user.

= Decisions (ADRs)

- *0001* — roster: transformers 5.9.0 recovers Gemma-4/Qwen3.6; dropped 3 models
  (tokenizer offset mismatch).
- *0002* — dataset = SVEN before/after full-function contrast (vary only the vuln).
- *0003* — honest `tokens_code_auc` replaces inflated `tokens_auc`; established the
  injection-strong / memory-weak split.
- *0004* — subtractive subset + cleaned regime (tight∩is_code positives); additive
  vulns undetectable by token probes.

= Experiment ledger

All AUCs are `tokens_code_auc` unless noted. ✅ done · ⏸ partial · ⛔ retracted.

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, center),
  table.header([Exp], [Aim → headline finding], [St.]),
  [02 layer-sweep], [best layer → peaks mid-late, no universal fraction (Gemma L19 0.77, Qwen L25 0.79); select per-model.], [✅],
  [03 loss-α], [span-max α → *α=1 beats α=10* (+0.01–0.03); neg_incl no-op.], [✅],
  [04 richer-probes], [MLP / concat → MLP head +0.02–0.04; concat helps Gemma only.], [✅],
  [05 probe-vs-verbalized], [probe > verbalized for Gemma (+0.09 introspection gap), ~tied Qwen.], [✅],
  [06 honest-sweeps], [8-model honest sweep → signal real; *injection strong, memory 0.52–0.59*; Python≫C; pt≈it.], [✅],
  [07 code-masked-negs], [mask trivial negs → no benefit (Δ≈0).], [✅],
  [08 latest-qwen], [Qwen3-32B → 0.806; memory gap shrinks, not closed.], [✅],
  [09 ensemble-linear], [K∈{1..8} dirs → +0.016 overall, *memory flat* → capacity not the lever.], [✅],
  [*10 per-cwe-probes*], [specialized vs general → *memory signal EXISTS: CWE-125 0.57→0.73, CWE-416 0.44→0.77* (Δ +0.15–0.33).], [✅],
  [11 family-balanced], [oversample memory → memory +0.06, injection −0.03; partial fix.], [✅],
  [12 mlp-layer-sweep], [MLP ceiling → 0.79–0.82; best layers vary (frac 0.22–0.71).], [✅],
  [13 causal-steering], [steer memory dir → *epiphenomenal* (±4σ moves P(yes) under 0.012).], [✅],
  [14 memory-prompt], [memory-specific prompts → recover ex-AUC +0.21–0.33; blind spot is framing.], [✅],
  [15 ensemble-comparison], [probe-vs-verbalized matrix → symmetric; prompt-spec tracks probe-spec.], [✅],
  [16 token-logit-dump], [persist per-token logits → reproduces history ±0.000; 7 models saved.], [✅],
  [17 verbalized-dump], [persist verbalized logits → reproduces exp-05 ±0.01; verbalized weak.], [✅],
  [18 mlp-logit-dump], [persist MLP logits → reproduces exp-12 ~bit-exact; mlp512≈mlp256.], [✅],
  [19 subtractive-regime], [clean labels → subtractive perf-neutral; *additive undetectable* (pairAcc 0.43); token>line. CV-confirmed.], [✅],
  [20 fn-fp-analysis], [token FN/FP → *lexical string-sink detector*; injection caught, memory missed, FP on patched SQL.], [✅],
  [21 per-cwe-cross-cwe], [cross-CWE transfer → *RETRACTED: reported pair-acc not tokens_code_auc, missed exp-10, wrong "memory unlearnable"*. Transfer matrix on token-AUC still owed.], [⛔],
  [22 primevul-paired], [cross-dataset SVEN↔PrimeVul (2-model, C/C++) → *asymmetric*: SVEN→PV 0.51–0.55 ≪ PV→PV 0.59–0.64, but PV→SVEN ≈ SVEN→SVEN — PrimeVul learns the more general direction; SVEN carries dataset bias. (was exp-20.)], [✅],
)

= Open threads

- *Cross-CWE transfer matrix on `tokens_code_auc`* (train CWE-X → test CWE-Y) — the
  one new piece exp-21 aimed at; redo from saved per-CWE logits.
- *PrimeVul shared-CWE-stratified transfer* (exp-22 follow-up) — exp-22 cross-dataset
  transfer *done*; shared-CWE/family-stratified eval would separate dataset bias from
  injection→memory domain shift. Extend past 2-model first cut; add error bars.
- *Per-CWE FN/FP categorization* (exp-20 fn-fp style) for injection CWEs.
- *Unlock memory in a single deployable probe* — per-family / multi-task head.
- *vLLM as default extractor on the cluster* — wire `.python_deps_vllm` + validate
  `extract_vllm` end-to-end.
- Research-framing open Qs: generation-transfer, own-code OOD, intent-to-exploit.

#v(6pt)
#text(size: 8pt, fill: muted)[
  Summary of `docs/project-log.md` (the living high-level log; verbose narrative in
  `claude-project-log.md`). Per-exp numbers extracted from each experiment's
  RESULTS.md — flag any that look off. exp-21 is retracted; see exp-10 for the
  correct per-CWE token-AUC result.
]
