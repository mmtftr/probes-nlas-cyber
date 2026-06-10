[ai-generated]

# Project log — high-level state

**Read this first, every session.** It is the unified high-level picture so the
whole project is held in mind at once: the research goal, what we currently know,
the standing conventions, the per-experiment ledger, and the open threads. It
exists to stop re-running prior work or contradicting an existing finding.

- Verbose chronological narrative: `claude-project-log.md` (append-only).
- This file: the curated state. **Keep the ledger row updated when an experiment
  lands or is retracted** (same change as the result).
- Decisions: `decisions/NNNN-*.md` (ADRs). Framing: `docs/research-framing.md`.

---

## 1. Research goal & scope

**Target property:** the model's *own internal belief* about how vulnerable the
code in its input is — a linear "vulnerability" read off hidden states, output as
a calibrated probability. Fit against SVEN before/after function pairs (vulnerable
vs its fix), token-level supervision with char-span localization.

**Scope:** Gemma-3 (1B–27B, it + pt) and Qwen2.5-Coder / Qwen3 families. Linear
**span-max** probes (small MLP variants tried) on mid–late layers.

**Open questions (from research-framing.md):** generation-vs-reading transfer;
own-code vs others'-code OOD; intent-to-exploit signal; layer-selection policy;
linear-vs-MLP probe family.

---

## 2. What we currently know (consolidated)

1. **There is a real, honest token-level vulnerability signal.** `tokens_code_auc`
   ≈ **0.75–0.82** across models (exp-06, validated/persisted exp-16). It is *not*
   an artifact of trivial negatives — restricting to live-code tokens barely drops
   it, and train-time masking of trivial negatives is a no-op (exp-07, ADR-0003).
2. **The apparent vuln-class split is mostly a LANGUAGE split (exp-23, 2026-06-10).**
   SVEN confounds family with language (injection vulns 92% Python, memory vulns
   100% C/C++; clean pool mixed). Within-language the general probe scores
   **python 0.80–0.86 vs C/C++ 0.56–0.66**; a bare language indicator scores
   **0.677** on the exact headline token set (probe 0.776) — **~64% of the
   headline's AUC-margin-over-chance is language**. The C-injection crossed cell
   (n=5 test ex) is too small to attribute the residual split to family on SVEN.
   Injection CWEs clear their language-null decisively (0.67–0.93); memory CWEs
   do not (general-probe memory AUCs sit *below* the corrected per-CWE nulls of
   0.63–0.65; exact-token-set nulls per exp-23 post-review erratum).
3. **Memory: no family direction; small per-CWE signals survive controls (exp-25/26,
   2026-06-10; retires the exp-10/21 "capacity-allocation" framing).**
   - The exp-10/21 all-clean eval design was **language-confounded** (memory=100% C
     vs a 53%-Python clean pool; corrected per-CWE nulls 0.63–0.65). Against the
     corrected nulls, specialized CWE-476 (0.640) is null-level, while CWE-125
     (0.732) / CWE-416 (0.766) retain ~+0.10 above null — the part that the
     matched-patch control (next bullet) then adjudicates.
   - Under the airtight **matched-patch** control (same function, language held
     fixed), a genuine attenuated signal survives for **CWE-125 (0.633/0.657 on
     both models, CI>0.5, stable across 15 CV folds)**, weakly for CWE-416;
     **CWE-476 collapses to chance**.
   - **exp-26 (PrimeVul-Paired, single-language C/C++):** per-CWE diagonal signal
     is real but **idiosyncratic** (0.61–0.88); memory CWEs do **NOT** form a
     transferable family cluster (mem→mem off-diag 0.536 ≈ mem→other 0.537).
   - Historical sub-results (exp-09 ensembles flat, exp-11 resampling partial,
     exp-04/12/18 MLP modest) stand but are reinterpreted under the confound.
4. **Pooled probe behaves as a lexical string-sink detector (exp-20) — and does NOT
   exceed the lexical surface ceiling (exp-24).** It fires on SQL/command/path
   string literals → catches injection sink tokens, misses memory (no string sink),
   false-alarms on patched code that still contains the SQL string. Bimodal across
   7 models (55/97 caught by all, 16/97 by none). **exp-24:** a char-n-gram surface
   baseline on identical splits scores **0.803 vs the probe's 0.776** on the
   headline and matches/beats the probe on ALL injection cells; surface-only also
   reproduces 3 of 4 of exp-21's family-transfer blocks. The one surface-resistant
   cell on SVEN (specialized mem→mem block transfer 0.618 vs surface 0.499) failed
   to replicate within-language on PrimeVul (exp-26).
5. **Verbalized side (model's own yes/no) is weak.** Probe > verbalized for Gemma
   (introspection gap, +0.09), ~tied for Qwen (exp-05/17). The memory verbalized
   blind spot is a **prompt-framing artifact** — memory-specific prompts recover
   +0.21–0.33 example-AUC (exp-14), and prompt-specialization mirrors
   probe-specialization (exp-15).
6. **Causality:** the memory probe direction is **epiphenomenal** — steering along
   it at fair magnitude (±4σ) doesn't move verbalized P(yes); it's a correlate,
   not a control knob (exp-13).
7. **Additive fixes (~1/3 of SVEN) are undetectable by a token-localized probe**
   (pairAcc ≈ chance) — honest token-probe work uses the **subtractive subset**
   with tight∩is_code labels (exp-19, ADR-0004).

---

## 3. Standing conventions (MUST know)

- **Default metric: `tokens_code_auc`** — honest token-level ROC-AUC over live-code
  tokens only (tree-sitter `code_only_mask`). This is the headline metric for every
  probe eval. Example-AUC / pair-ranking / detection-rate are **secondary** and must
  be labelled as such — never headline a "signal absent/works" claim on them.
- **Dataset:** SVEN before/after pairs (1430 rows). For token-probe work, the
  **subtractive subset** (956 ex / 478 pairs, localizable-fix-only) per ADR-0004.
- **Split:** group-clean at pair level, seed-42, 20% held-out; pairs never straddle.
  Inner 15% val (seed-42, group-aware) for layer/epoch selection.
- **Layer:** per-model, selected by max `val_tokens_code_auc`; report deployable +
  oracle. Repo-layer L = transformers `hidden_states[L+1]` = output of block L.
- **Negative pool (per-CWE / specialized probes):** all `cwe==null` clean rows
  (the exp-06/10 recipe) — keeps head-to-head comparable.
- **Cluster:** the cluster `debug` only (1.5 node-h/job, MaxJobs=1); use `fc`
  (job-API) for unattended. **Default extractor = vLLM** (HF fallback `--backend
  hf`; note vLLM is installed at `.python_deps_vllm` but off the default PYTHONPATH).
  Activations float32 on scratch; KEEP operating-layer acts.
- **Review gate (CLAUDE.md):** every result passes a cj/codex + Opus-subagent review
  (metric / prior-work / methodology / conclusion) **before** reaching the user.

---

## 4. Decisions (ADRs)

- **0001** — Roster: transformers 5.9.0 recovers Gemma-4/Qwen3.6; dropped
  OpenCoder-8B / Devstral / Mistral-Small (tokenizer offset mismatch). Roster 20/23.
- **0002** — Dataset = SVEN before/after full-function contrast (vary only the vuln,
  hold task fixed); token labels mark diff'd vulnerable lines.
- **0003** — Honest metric `tokens_code_auc` (live-code only) replaces inflated
  `tokens_auc`. **Established the injection-strong / memory-weak split.**
- **0004** — Subtractive subset + cleaned regime (tight∩is_code positives, token
  granularity). Additive vulns undetectable by token probes; default to subtractive.

---

## 5. Experiment ledger

Status: ✅ done · ⏸ partial/awaiting · ⛔ retracted. All AUCs are `tokens_code_auc`
unless noted.

| Exp | Aim → headline finding | Status |
|---|---|---|
| 02 layer-sweep | Best layer/depth → peaks mid-late, no universal fraction (Gemma L19 0.77, Qwen L25 0.79); select per-model by val. | ✅ |
| 03 loss-α-sweep | Span-max α → **α=1 beats α=10** (ex-AUC +0.01–0.03); neg_incl no-op. New default α=1. | ✅ |
| 04 richer-probes | MLP / layer-concat → MLP head +0.02–0.04; concat helps Gemma only. | ✅ |
| 05 probe-vs-verbalized | Probe vs model yes/no → probe > verbalized for Gemma (+0.09 introspection gap), ~tied Qwen. | ✅ |
| 06 honest-metric-sweeps | Honest sweep, 8 models → signal real ~0.75–0.79; **injection strong, memory ~0.52–0.59; Python≫C**. pt≈it. | ✅ |
| 07 code-masked-negs | Train-time mask trivial negs → no benefit (Δ≈0); probe never leaned on them. | ✅ |
| 08 latest-qwen-dense | Newer Qwen (3-32B/3.6) → 0.806 overall; memory gap shrinks slightly, not closed. | ✅ |
| 09 ensemble-linear | K∈{1,2,4,8} directions → +0.016 overall, **memory flat** → capacity not the lever. | ✅ |
| **10 per-cwe-probes** | Specialized vs general per CWE → **memory signal EXISTS: CWE-125 0.57→0.73, CWE-416 0.44→0.77** (Δ +0.15–0.33). Under-allocation, not absence. | ✅ |
| 11 family-balanced-head | Oversample memory family → memory +0.06, injection −0.03; partial fix. | ✅ |
| 12 mlp-layer-sweep | MLP ceiling per layer → 0.79–0.82; best layers vary (frac 0.22–0.71). | ✅ |
| 13 causal-steering | Steer memory direction → **epiphenomenal** (±4σ moves P(yes) <0.012); v1 effect was model-breaking magnitude. | ✅ |
| 14 memory-prompt-sweep | Memory-specific prompts → recover memory ex-AUC +0.21–0.33 (V0 0.39–0.55 → V3 0.71–0.75); blind spot is framing. | ✅ |
| 15 ensemble-comparison | Probe-vs-verbalized specialization matrix → symmetric; prompt-spec tracks probe-spec member-by-member. | ✅ |
| 16 token-logit-dump | Persist every per-token logit → reproduces history ±0.000 (Qwen-32B 0.776); 7 models' logits saved. | ✅ |
| 17 verbalized-logit-dump | Persist verbalized logits → reproduces exp-05 ±0.01; verbalized weak (0.49–0.62). | ✅ |
| 18 mlp-logit-dump | Persist MLP logits (256/512) → reproduces exp-12 ~bit-exact; mlp512≈mlp256. | ✅ |
| 19 subtractive-regime | Clean labels/subset → subtractive performance-neutral (0.755≈0.756); **additive undetectable (pairAcc 0.43)**; token>line. CV-confirmed. | ✅ |
| 20 fn-fp-token-analysis | Token-level FN/FP of pooled probe → **lexical string-sink detector**; injection caught, memory missed, FP on patched SQL. Bimodal, family-agnostic. | ✅ |
| 21 per-cwe-cross-cwe | **Cross-CWE transfer matrix on `tokens_code_auc`**, exp-10 recipe (train CWE-X vuln vs ALL-clean, full SVEN, annotated token_labels positives), reusing KEPT acts. **Diagonal reproduces exp-10 bit-exact** (Qwen Δ±0.000): injection 0.86–0.98, **memory 0.64–0.77** (125 0.73, 416 0.77, 476 0.64) — memory IS learnable on its own data. **NEW: transfer is family-structured** — within-family off-diag >chance (inj 0.60 [.56,.64], mem 0.57 [.54,.61]), cross-family below chance (inj→mem 0.41, mem→inj 0.34; CIs exclude 0.5). ⟹ ≥2 coarse family directions (taint/string-sink + memory-safety), not one universal nor purely per-CWE. Confirms #3, sharpens #4. 32B≈1B. *(Superseded: ⛔ pair-acc version; and a matched-patch rescore that showed memory near-chance — a training-regime artifact, git history.)* | ✅ |
| 22 primevul-paired | Cross-dataset SVEN↔PrimeVul transfer (2-model, C/C++ slice) → **asymmetric**: SVEN→PV 0.51–0.55 ≪ PV→PV 0.59–0.64, but PV→SVEN 0.64–0.65 ≈ SVEN→SVEN — PrimeVul (5.3× pairs) learns the more general direction; SVEN probe carries dataset bias. (was exp-20; single split, no error bars.) | ✅ |
| **23 language-stratified-rescore** | Within-language rescore of exp-16 saved logits (7 models, format gates ≤0.00034) → **~64% of the headline AUC margin recoverable by a bare language indicator** (0.677 vs probe 0.776); within-py 0.80–0.86 vs within-C 0.56–0.66; corrected per-CWE memory nulls (0.63–0.65) ≥ general probe on every memory CWE; specialized 125/416 clear nulls by ~+0.10 (476 null-level); C-inj cell n=5 untrusted. Review-gated (pass-with-fixes; erratum in RESULTS.md). | ✅ |
| **24 surface-baselines** | Token-level surface ceiling, identical splits/eval → **char-ngram 0.803 > probe 0.776**; ALL injection cells ≈/≤ surface; surface reproduces 3/4 exp-21 family-transfer blocks. Specialized memory probes beat surface on all-clean diag (125: 0.732>0.632, 416: 0.766>0.649) and mem→mem block (0.618 vs 0.499, disjoint CIs). Verdict: claim #3 "undercut, not refuted" → settled by exp-25/26. | ✅ |
| **25 allclean-language-matched** | The decisive 2×2 (all-clean-trained per-CWE probes × {all-clean repro ±0.000, C-only, matched-patch} negatives + C-only retrain + 5f×3s CV, qwen32b+gemma-1b L25) → all-clean memory diag inflated ~0.05–0.09 by language; **CWE-125 survives matched-patch (0.633/0.657 both models, CI>0.5, CV-stable ×15 folds)**, 416 weakly positive, **476 collapses**. Claim #3 narrowly rescued, attenuated; lead with matched-patch. | ✅ |
| **26 primevul-within-family** | Within-C/C++ family structure (PV-Paired, qwen7b L16, 12 CWEs ≥10 test pos; PV has NO injection CWEs — scope-limited to memory-vs-other) → **no transferable memory family cluster** (mem→mem off-diag 0.536 [0.513,0.564] ≈ mem→other 0.537 [0.518,0.554]); per-CWE diagonal real but idiosyncratic (119 0.875, 190 0.730, 476 0.692, 416 0.657, 125 0.611). Secondary SVEN→PV (single-split): 125 0.668, 416 0.706, 476 0.494 — consistent w/ exp-25. | ✅ |

---

## 6. Open threads

- **Memory regime question — RE-RESOLVED (exp-23/24/25/26, 2026-06-10).** The
  exp-21 "RESOLVED" verdict was itself wrong: the all-clean regime it preferred is
  language-confounded (the corrected null is the language baseline ≈0.665, not 0.5),
  and the matched-patch near-chance result was the controlled measurement. Final
  state: **no memory family direction** (exp-26 within-C/C++: mem→mem off-diag ≈
  mem→other); **small genuine per-CWE signals survive matched-patch** (CWE-125
  robustly, CWE-416 weakly, CWE-476 not). The "per-family ≥2-head probe" follow-on
  is retired with the family framing.
- **Injection-vs-family attribution within C** — still open: SVEN has only n=5
  C-injection test examples and PrimeVul has no injection CWEs. Needs a dataset
  with real C/C++ injection vulns (or a Python memory-vuln source) to cross
  language × family.
- **exp-26 second model + CIs for SVEN→PV cells** — within-PV matrix is qwen7b-only;
  SVEN→PV shared-CWE cells are single-split point estimates.
- **Probe-above-surface margin** — after exp-24, the honest claim for the pooled
  probe is ≤ the lexical ceiling; the open question is whether ANY probe target
  (per-CWE matched-patch cells like CWE-125) supports a deployable monitor at a
  fixed-FPR operating point (recall@1%FPR, calibration — the AI-control bar).
- **Per-CWE FN/FP categorization** (exp-20 fn-fp style, per-CWE) for injection CWEs.
- **vLLM as the default extractor on the cluster** — wire `.python_deps_vllm` onto the
  path + validate the uncommitted `extract_vllm` end-to-end.
- Research-framing open questions (generation transfer, own-code OOD, intent).
