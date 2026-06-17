[ai-generated]

# Project log — high-level state

**Read this first, every session.** It is the unified high-level picture so the
whole project is held in mind at once: the research goal, what we currently know,
the standing conventions, the per-experiment ledger, and the open threads. It
exists to stop re-running prior work or contradicting an existing finding.

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
   - **exp-27 (2026-06-12): that residue is partly lexical.** A token-unigram LR
     scores 0.591/0.584 on 125×matched-patch (CI>0.5, both token axes) — ~½–⅔
     of the probe's margin-over-chance; the window/n-gram surface family
     (char/combined/conly-trained) is ∋0.5 in all 6 trusted memory×mp cells.
     The probe tops every surface point on 125/416 (+0.04–0.07; no contrast
     CI-separated at n=19/14) and is the only >chance signal on 416×mp.
     "Demonstrably non-lexical" not earned; "purely lexical" excluded.
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
5. **Verbalized side (model's own yes/no) is weak — and on held-out data is ≥ the
   probe at the example level (exp-29, 2026-06-13 supersedes the exp-05 "Gemma
   introspection gap").** exp-05/17 reported probe > verbalized for Gemma (+0.09),
   ~tied Qwen, BUT that probe AUC was example-level max-pool scored **in-sample on
   the full 1430** (fresh 5-seed probes, per-model layer). exp-29 re-scores the
   single *deployed* exp-16 probe on the **held-out 292**: at the example level
   **verbalized ≥ every probe read on 5/6 models** (the gemma "+0.09" REVERSES —
   gemma-27b max-pool 0.535 < verbalized 0.566), and the probe **read at the final
   live-code token is chance on all 7** (0.51, CI∋0.5). Qwen agrees with exp-05
   (verbalized ahead). Example-level = SECONDARY; the token-level headline
   `tokens_code_auc` 0.75–0.82 is untouched. NB this is the last *code* token, not
   the `Assistant:`-boundary introspection position (blog NXT2b), which still needs
   a small extraction. The memory verbalized blind spot is a **prompt-framing
   artifact** — memory-specific prompts recover +0.21–0.33 example-AUC (exp-14),
   and prompt-specialization mirrors probe-specialization (exp-15).
6. **Causality:** the memory probe direction is **epiphenomenal** — steering along
   it at fair magnitude (±4σ) doesn't move verbalized P(yes); it's a correlate,
   not a control knob (exp-13).
7. **Additive fixes (~1/3 of SVEN) are undetectable by a token-localized probe**
   (pairAcc ≈ chance) — honest token-probe work uses the **subtractive subset**
   with tight∩is_code labels (exp-19, ADR-0004).
8. **The `Assistant:`-commit position decodes vuln above the model's own answer,
   but it's LEXICAL — exp-30's apparent positive folds into the lexical ceiling
   under controls (exp-30 + exp-31, 2026-06-14).** A linear probe on the
   commit-position hidden state (last prompt token) decodes the SVEN vuln label at
   **0.66–0.82 example-AUC** (exp-30), ≫ verbalized P(yes) (0.49–0.62; Δ +0.18–0.23,
   CI-sep) and ≫ the exp-29 code-token read (0.51–0.57); clears permutation +
   random nulls; lang/len ≈ chance; mostly Python (within-C 0.53–0.62). **BUT the
   two exp-31 controls dissolve the "positive":** (a) a char-n-gram surface
   classifier on the raw code text scores **0.778** example-AUC — only Qwen-32B-primed
   clears it CI-separated (11/12 probe cells do not; gemma-1b probe 0.66 ≪ char); (b) the lone
   above-ceiling cell (Qwen-32B primed, Δ +0.045 over char, pair-clustered CI>0) is
   **priming-dependent** — under a NEUTRAL prompt ("What do you think about this
   code?") it drops to the ceiling (Δ −0.013, ns). So commit-position decodability
   is **lexical**, extending claim #4 to a new position + prompt; there is NO robust
   intrinsic above-lexical signal. What survives: probe > verbalized (reading
   activations beats asking the model) and the probe clears nulls (decodes real, but
   lexical, structure). Neutral probe (0.65–0.76) ≈ intrinsic (only ~0.02–0.07 below
   primed) so the representation isn't purely question-driven — it's just lexical.
   Example-level = SECONDARY; token-level headline untouched. Dual-reviewed ×2
   (exp-30 + exp-31). The exp-30→31 arc vindicates the surface-baseline + de-prime
   methodology: it caught an apparent breakthrough.

---

## 3. Standing conventions (MUST know)

- **Default metric: `tokens_code_auc`** — honest token-level ROC-AUC over live-code
  tokens only (tree-sitter `code_only_mask`). This is the headline metric for every
  probe eval. Example-AUC / pair-ranking / detection-rate are **secondary** and must
  be labelled as such — never headline a "signal absent/works" claim on them.
- **Metric glossary (naming collision — fixed convention 2026-06-11).** Two result
  keys with near-identical names mean different things; in future writeups use the
  bold names below, never the raw keys, and state the pooling.
  - **pooled token AUC** = `tokens_code_auc`: one ranking over all live-code test
    tokens (the default, above).
  - **example AUC (max-pool | mean-pool)**: one score per *example*, label = example
    contains ≥1 vuln token, AUC across examples. exp-16/17/18's
    `example_scores_*`/verbalized AUCs are **max-pool**; exp-24's
    `example_mean_auc` key is **mean-pool**. Example-level, secondary.
  - **within-example macro token AUC** (rename of exp-23's `per_example_mean_auc`
    key, future name `within_example_macro_auc`): token AUC computed *inside* each
    example that has both token classes, macro-averaged across those examples.
    Token-level but immune to cross-example confounds (each example is its own
    ranking) — the per-example cousin of the within-language stratification.
- **Dataset:** SVEN before/after pairs (1430 rows). For token-probe work, the
  **subtractive subset** (956 ex / 478 pairs, localizable-fix-only) per ADR-0004.
- **Split:** group-clean at pair level, seed-42, 20% held-out; pairs never straddle.
  Inner 15% val (seed-42, group-aware) for layer/epoch selection.
- **Layer:** per-model, selected by max `val_tokens_code_auc`; report deployable +
  oracle. Repo-layer L = transformers `hidden_states[L+1]` = output of block L.
- **Negative pool (per-CWE / specialized probes):** all `cwe==null` clean rows
  (the exp-06/10 recipe) — keeps head-to-head comparable.
- **Compute:** GPU node; jobs kept resumable. **Default extractor = vLLM** (HF
  fallback `--backend hf`). Activations stored float32; KEEP operating-layer acts.
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
| 09 ensemble-linear | K∈{1,2,4,8} directions → +0.016 overall, **memory flat** → capacity not the lever. Blog addon (2026-06-14): +Qwen-7B ensemble on cv-acts; vs MLP@best-layer the linear K-ensemble sits below the MLP at every K on BOTH overall and memory (Δ −0.01..−0.05 / −0.03..−0.17) → stacking linear dirs ≠ nonlinear capacity. | ✅ |
| **10 per-cwe-probes** | Specialized vs general per CWE → **memory signal EXISTS: CWE-125 0.57→0.73, CWE-416 0.44→0.77** (Δ +0.15–0.33). Under-allocation, not absence. | ✅ |
| 11 family-balanced-head | Oversample memory family → memory +0.06, injection −0.03; partial fix. | ✅ |
| 12 mlp-layer-sweep | MLP ceiling per layer → 0.79–0.82; best layers vary (frac 0.22–0.71). | ✅ |
| 13 causal-steering | Steer memory direction → **epiphenomenal** (±4σ moves P(yes) <0.012); v1 effect was model-breaking magnitude. | ✅ |
| 14 memory-prompt-sweep | Memory-specific prompts → recover memory ex-AUC +0.21–0.33 (V0 0.39–0.55 → V3 0.71–0.75); blind spot is framing. | ✅ |
| 15 ensemble-comparison | Probe-vs-verbalized specialization matrix → symmetric; prompt-spec tracks probe-spec member-by-member. | ✅ |
| 16 token-logit-dump | Persist every per-token logit → reproduces history ±0.000 (Qwen-32B 0.776); 7 models' logits saved. | ✅ |
| 17 verbalized-logit-dump | Persist verbalized logits → reproduces exp-05 ±0.01; verbalized weak (0.49–0.62). | ✅ |
| 18 mlp-logit-dump | Persist MLP logits (256/512) → reproduces exp-12 ~bit-exact; mlp512≈mlp256. Blog addon (2026-06-14): per-CWE memory token-AUC recomputed at each MLP's own best layer from the dumps (`per_cwe_bestlayer.py` → `mlp_memory_bestlayer.json`); overall reproduced bit-for-bit; memory mean 0.57–0.63. | ✅ |
| 19 subtractive-regime | Clean labels/subset → subtractive performance-neutral (0.755≈0.756); **additive undetectable (pairAcc 0.43)**; token>line. CV-confirmed. | ✅ |
| 20 fn-fp-token-analysis | Token-level FN/FP of pooled probe → **lexical string-sink detector**; injection caught, memory missed, FP on patched SQL. Bimodal, family-agnostic. | ✅ |
| 21 per-cwe-cross-cwe | **Cross-CWE transfer matrix on `tokens_code_auc`**, exp-10 recipe (train CWE-X vuln vs ALL-clean, full SVEN, annotated token_labels positives), reusing KEPT acts. **Diagonal reproduces exp-10 bit-exact** (Qwen Δ±0.000): injection 0.86–0.98, **memory 0.64–0.77** (125 0.73, 416 0.77, 476 0.64) — memory IS learnable on its own data. **NEW: transfer is family-structured** — within-family off-diag >chance (inj 0.60 [.56,.64], mem 0.57 [.54,.61]), cross-family below chance (inj→mem 0.41, mem→inj 0.34; CIs exclude 0.5). ⟹ ≥2 coarse family directions (taint/string-sink + memory-safety), not one universal nor purely per-CWE. Confirms #3, sharpens #4. 32B≈1B. *(Superseded: ⛔ pair-acc version; and a matched-patch rescore that showed memory near-chance — a training-regime artifact, git history.)* | ✅ |
| 22 primevul-paired | Cross-dataset SVEN↔PrimeVul transfer (2-model, C/C++ slice) → **asymmetric**: SVEN→PV 0.51–0.55 ≪ PV→PV 0.59–0.64, but PV→SVEN 0.64–0.65 ≈ SVEN→SVEN — PrimeVul (5.3× pairs) learns the more general direction; SVEN probe carries dataset bias. (was exp-20; single split, no error bars.) | ✅ |
| **23 language-stratified-rescore** | Within-language rescore of exp-16 saved logits (7 models, format gates ≤0.00034) → **~64% of the headline AUC margin recoverable by a bare language indicator** (0.677 vs probe 0.776); within-py 0.80–0.86 vs within-C 0.56–0.66; corrected per-CWE memory nulls (0.63–0.65) ≥ general probe on every memory CWE; specialized 125/416 clear nulls by ~+0.10 (476 null-level); C-inj cell n=5 untrusted. Review-gated (pass-with-fixes; erratum in RESULTS.md). | ✅ |
| **24 surface-baselines** | Token-level surface ceiling, identical splits/eval → **char-ngram 0.803 > probe 0.776**; ALL injection cells ≈/≤ surface; surface reproduces 3/4 exp-21 family-transfer blocks. Specialized memory probes beat surface on all-clean diag (125: 0.732>0.632, 416: 0.766>0.649) and mem→mem block (0.618 vs 0.499, disjoint CIs). Verdict: claim #3 "undercut, not refuted" → settled by exp-25/26. | ✅ |
| **25 allclean-language-matched** | The decisive 2×2 (all-clean-trained per-CWE probes × {all-clean repro ±0.000, C-only, matched-patch} negatives + C-only retrain + 5f×3s CV, qwen32b+gemma-1b L25) → all-clean memory diag inflated ~0.05–0.09 by language; **CWE-125 survives matched-patch (0.633/0.657 both models, CI>0.5, CV-stable ×15 folds)**, 416 weakly positive, **476 collapses**. Claim #3 narrowly rescued, attenuated; lead with matched-patch. | ✅ |
| **26 primevul-within-family** | Within-C/C++ family structure (PV-Paired, qwen7b L16, 12 CWEs ≥10 test pos; PV has NO injection CWEs — scope-limited to memory-vs-other) → **no transferable memory family cluster** (mem→mem off-diag 0.536 [0.513,0.564] ≈ mem→other 0.537 [0.518,0.554]); per-CWE diagonal real but idiosyncratic (119 0.875, 190 0.730, 476 0.692, 416 0.657, 125 0.611). Secondary SVEN→PV (single-split): 125 0.668, 416 0.706, 476 0.494 — consistent w/ exp-25. | ✅ |
| **27 matchedpatch-surface** | exp-24's surface baselines under exp-25's negative regimes (allclean gate / conly / matched-patch), 2 token axes (qwen32b bit-anchored: design-2 repro dev 0.0; gemma1b counts+lang_null-pinned), CIs for EVERY variant + paired probeG−surface Δ → **split verdict**: window/n-gram family (char/combined/conly-trained) ∋0.5 in all 6 trusted memory×mp cells, **but token-unigram survives mp on CWE-125 (0.591/0.584, CI>0.5 both axes) — ~½–⅔ of the probe's margin-over-chance is lexical**; specialized probe tops every surface point in all 4 trusted 125/416 cells (+0.04–0.07, no contrast CI-separated) and is the **only** >chance signal on 416×mp. Injection lexical even under mp (089 char 0.975 ≥ probe 0.933/0.886). Review-gated ×2 rounds (round-1 "CI every variant" fix materially corrected the verdict). | ✅ |
| **28 primevul-deepdive** | PV analyzed properly (qwen7b L16 from exp-26 CACHED logits, no re-extraction; surface = exp-24 char-ngram + unigram on the identical token axis; repro gate 48/48 bit-exact; dual-reviewed) → (1) **surface shows no memory cluster either** (char mem→mem off-diag 0.531 ≈ mem→other 0.497; probe 0.536 ≈ 0.537) — exp-26's no-cluster verdict is not a probe-family artifact; (2) probe ≥ char-surface on 9/12 PV diagonals (mean +0.067): **CWE-476 clearest non-lexical diagonal** (probe 0.692* vs char 0.500, unigram 0.504) but PV-idiosyncratic (no detectable cross-dataset transfer); CWE-125's within-PV diagonal is lexical-level (char 0.655 > probe 0.611); (3) **SVEN→PV transfer now CI-backed**: 125 0.668 [.57,.77], 416 0.706 [.63,.86], 787 0.635 [.59,.73]; 476 chance-consistent (wide CI); (4) **PV matched-pair (vuln vs OWN fix): SVEN-trained probes survive** — 125 0.660*, 416 0.650* (firmer than exp-25's "weak"), 787 0.645* (newly testable; SVEN n=5), 190 0.736*; 476 chance. Joint w/ exp-27: which lexical family clears a memory×mp cell is dataset-dependent (PV-125 char yes/unigram no — mirror of SVEN; PV-416 unigram yes; PV-787 neither); probe tops every lexical point in every trusted memory×mp cell. **PAIRED Δ-bootstrap (exp-27's decisive test, PV side, tie-correct AUC): CWE-787×mp probe exceeds BOTH lexical baselines with nominal Δ-CIs excluding 0 (Δ_char +0.108 [+0.04,+0.22], Δ_uni +0.164 [+0.12,+0.22]) — the project's FIRST CI-separated probe-over-lexical margin** (exploratory 24-test family, 4 exclude 0; 787's Δ_uni survives any correction and is the only cell clearing both); Δ_uni also separates for 125/190 (Δ_char doesn't). pairAcc ≈ chance (secondary). Stretch (gemma-12b-it L15 within-PV replicate): job mid-run at a cluster outage — pending. | ✅ |
| **29 last-token-readout** | Example-level (SECONDARY) like-with-like for the blog's verbalized comparison: read the *existing* exp-16 span-max probe at each test fn's FINAL live-code token vs verbalized P(yes) + max/mean-pool, 7 models, true-fn label, n=292 (+ ADR-0004 subtractive n=194), 1000-boot CIs + paired Δ. CPU rescore of saved logits (5/7 via cluster-side numpy reduction `reduce_logits.py`; the cluster's file-transfer API can't move the 7 MB npz). Gate: recomputed `tokens_code_auc` == exp-16 stored, bit-exact 7/7. → **last-code-token probe = CHANCE on all 7** (0.51, CI∋0.5) — signal is distributed, not at the final position. **Verbalized ≥ every probe read on 5/6 models**; only CI-separated full-test contrast = Qwen-32B verbalized > last-tok (Δ −0.106 [−.18,−.03]); subtractive separates on 3 (Qwen-32B/7B, gemma-12b-it). max-pool-vs-verbalized a wash everywhere. **REVERSES exp-05's gemma "+0.09 introspection gap"** (that was in-sample full-1430; held-out gemma-27b max-pool 0.535 < verb 0.566) — see finding #5. Fixes the fig-verbalized token-vs-example mismatch. NB last *code* token ≠ `Assistant:` boundary (NXT2b still needs extraction). Dual-reviewed (design + result, GO-w/-fixes both). | ✅ |
| **30 last-token-introspection** | The genuine NXT2b: linear L2-LR probe on the hidden state at the verbalized-QA **`Assistant:`-commit position** (last prompt token), 6 it-models, ALL layers val-selected (+C), label-perm null (N=1000, reselect layer on shuffled tr/val at deployable C) + random-dir null (N=2000) + lang/len/within-lang confounds. NEW GPU extraction (exp-17 forward verbatim + `output_hidden_states`, **float32**, finite-gated merge), trained on cluster 288-core loky. Verbalized re-AUC hard-gated == exp-17. → **commit-position hidden state DECODES the SVEN vuln label at 0.66–0.82 example-AUC** (Qwen-32B L60 0.823, 7B L24 0.809, gemma 1b/4b/12b/27b 0.662/0.739/0.770/0.770) — **≫ verbalized (0.49–0.62; paired Δ +0.18–0.23, ALL CI-separated)** and **≫ exp-29 code-token max-pool (0.51–0.57)**. Clears perm-null (p=0.001, probe > null max on all 6) + beats all 2000 random dirs; lang=0.500/len=0.491 (example-level, pairs language-matched). **Mostly Python** (within-Py 0.75–0.92 vs within-C 0.53–0.62). Answer-readout **excluded for gemma-1b** (peaks L4, probe⊥P(yes) Spearman −0.08) but **open for big models** (peak late; per-layer curve TODO). **Lexicality UNRESOLVED** (example-level surface baseline at this position not yet run) → claim = *decodability above verbalized + code-token*, NOT belief/non-lexical. Example-level (SECONDARY); token-level code-token headline untouched. Dual-reviewed: both REPRODUCED the numbers; blocking fixes were interpretive (reframing, applied). First strong reading-frame positive — **but TEMPERED by exp-31 (surface + neutral controls): folds into the lexical ceiling**; see finding #8. | ✅ |
| **31 neutral-prompt-and-surface** | The two controls on exp-30 (user-directed). (a) **Neutral prompt** ("What do you think about this code?" — de-primes the vuln question): re-extract + re-probe the same commit position. (b) **Example-level char-n-gram + token-unigram surface baseline** on raw code text (exp-24 lifted to example level; vectorizer train-only, **pair-clustered bootstrap** over the 141 groups, strongest of 3 char configs = ceiling), paired-Δ vs primed + neutral probes; probe-refit gated to reproduce exp-30 (≤2e-3). → **char-n-gram = 0.778; only Qwen-32B primed clears it CI-separated (11/12 probe cells do not; point-estimate char tops 4/6 primed + all 6 neutral; gemma-1b probe 0.66 ≪ char)** (Δ +0.045 [+.009,+.082]) — exploratory (1/12 cells) AND **priming-dependent: neutral Qwen-32B drops to the ceiling (Δ −0.013, ns)**. Neutral probe 0.65–0.76 (≈primed −0.02–0.07; clears nulls, beats verbalized Δ +0.14–0.18) → intrinsic but lexical. **Verdict: commit-position decodability is LEXICAL — exp-30 folds into claim #4** (extends it to commit position + neutral prompt). Survives: probe > verbalized; probe clears nulls. Dual-reviewed (methodology GO-w/-fixes applied: train-only vec, pair-clustered boot, stronger char, refit gate). Vindicates the surface+de-prime methodology. | ✅ |
| **33 operating-point-tpr** | Example-level (SECONDARY) TPR @ 1% FPR by language + CWE for the 2 flagship models (Qwen-32B, gemma-27b) + the lexical ceiling — the blog NXT4 / standing "TPR@1%FPR" thread. CPU rescore of KEPT exp-30 commit-position hidden states (deployable L/C) + exp-31 char-ngram + npz verbalized P(yes); all 3 hard-gated to reproduce exp-30/31 AUC. Negatives matched to CWE via split group (1:1-balanced). **Headline = GLOBAL threshold** (one frozen 1%-FPR threshold per scorer on the full 146-neg pool, TPR by slice — deployable + comparable across slices; the per-slice ROC-interp number is 1-FP-dominated at n_neg≤83, kept JSON-only as a labelled companion, NOT headlined). → **operating-point view reproduces the lexical ceiling (#4) in starker form: detection is Python/SQL-injection ONLY** (Qwen probe SQLi 0.98 / char 0.61; gemma 0.75 / 0.61); **memory CWEs (125/476/416) + C/C++ ≈ 0 for probe, char, AND verbalized alike**. Probe ≈ char overall (Δ not CI-sep); the **only** CI-separated global probe>char cell is Qwen×SQLi (+0.50 [+.18,+.98]) and it's **cleaner ranking, NOT above-lexical** (within-089 AUC tied 0.983 vs 0.976; cf exp-20 char fires on patched SQL strings); char even **beats** probe on cmd-injection (both models); gemma 0/10 CI-sep. Verbalized ≤ probe/char everywhere bar ties. Confirms #4/#8, adds no finding. drop_intermediate=False ROC fix (an earlier interp artifact had inflated python-char .02→.37) + metric self-test + global paired-Δ bootstrap. **Dual-reviewed ×2 rounds** (2 codex + 2 Opus; the ROC fix changed the conclusion mid-review → re-reviewed; all GO-w/-fixes applied). Deliverables: `33-operating-point-tpr/` (compute_tpr.py, EXPERIMENT.md, results/operating_point.json); blog fig `fig_operating_point` + prose after fig-7. | ✅ |
| **32 percwe-matchedpatch** | Consolidation/re-framing (no new compute): promotes **matched-patch as THE per-CWE result for ALL 9 CWEs both families**, replacing the all-clean pooled per-CWE numbers (exp-10/21) the blog reported — the pooled token-AUC discards SVEN's pairing, so language/template inflate it. Reads exp-25 (probe+CI) + exp-27 (probe+lexical) matched-patch cells; 4 assert-gates (no-mix, no-drift, **shared-token-axis** via lang_null+n_pos match, memory lang_null==0.5 exact). → **injection survives but is lexical** (089 char 0.975 ≥ probe 0.933; 078 char 0.872 ≥ 0.833; **XSS collapses to ~chance vs own patches** 0.53/0.61); **memory thin**: 125 probe 0.633/0.657 clears chance + tops both scanners but unigram covers ~½–⅔ (partly lexical), 416 probe 0.610/0.603 the ONLY >chance signal (scanners at chance) but marginal, 476 chance for all; **no probe-vs-lexical contrast CI-separated** (n=14–19). 190/787 n=4/5 excluded. CWE-022 keeps token-level lang residual (0.37, only mixed Py/C class). Reviewed 2 codex + 1 Opus → TRUSTWORTHY, consistent re-framing of exp-25/27 (no new finding). Deliverables: `32-percwe-matchedpatch/` (consolidate.py, make_fig.py, table+fig), draft section `docs/blog/percwe-matchedpatch-section-draft.md`. NOTE: initial-results per-CWE asides, fig-family, fig-blocks still all-clean (not yet redone). | ✅ |

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
- **exp-26 second model + CIs for SVEN→PV cells** — CIs DONE (exp-28: 125/416/787
  exclude chance; 476 chance-consistent). Second model (gemma-12b-it L15 within-PV
  on cached acts) submitted as exp-28 stretch — job was mid-run when the cluster went
  unhealthy (2026-06-12); fetch + append on recovery
  (resumable: probes checkpointed, logits cached; resubmit once if walled).
- **Probe-above-surface margin — PV side ANSWERED for one cell (exp-28,
  2026-06-12); SVEN side open.** The pooled probe is ≤ the lexical ceiling
  (exp-24). The paired probe−lexical Δ-bootstrap on PV matched-patch gives
  **CWE-787: probe 0.645 exceeds char-ngram AND unigram with nominal Δ-CIs
  excluding 0 — the first CI-separated probe-over-lexical margin in the
  project** (tie-correct AUC; Δ_uni +0.164 [+0.121,+0.222] survives any
  multiplicity correction). Δ_uni also separates for 125/190 but Δ vs the
  stronger lexical (char) does not.
  Remaining: the SVEN-side paired Δ (needs cluster L25 acts to score exp-25
  probes per token); the deployable-monitor question (recall@1%FPR,
  calibration) after that.
- **TPR @ 1% FPR — DONE for the 2 flagship models (exp-33, 2026-06-16); extend +
  calibrate next.** Frozen 1%-FPR operating point, probe + char-ngram + verbalized,
  by language + CWE (Qwen-32B, gemma-27b). Headline = GLOBAL threshold (deployable).
  Reproduces the lexical ceiling in starker form: detection Python/SQL-injection only,
  memory + C/C++ ≈ 0 for ALL three scorers; the lone CI-sep probe>char cell (Qwen
  ×SQLi) is cleaner ranking (within-089 AUC tied), not above-lexical. Blog
  `fig_operating_point` after fig-7. Remaining: extend to every probe/baseline +
  calibration (blog future-work NXT4).
- **Per-CWE FN/FP categorization** (exp-20 fn-fp style, per-CWE) for injection CWEs.
- **vLLM as the default extractor** — validate the `extract_vllm` path end-to-end.
- **Blog post in drafting** (2026-06-11; figs extended 2026-06-13) — outline at
  `docs/blog/outline.typ` (code-addressable points, Slack-edited); v2 =
  results-overview-first, exps 02–22 findings folded in, figures FIG-A–H generated
  (`docs/blog/make_figs.py`; FIG-E needs the exp-16 npz via env vars). FIG-H
  (added 2026-06-13; recolored + numbers corrected 2026-06-16) = per-model
  injection-vs-memory family split of the headline AUC (**inj 0.85–0.91**, mem
  at/near chance 0.49–0.57). Colors: softened green (inj) / vermillion (mem),
  pooled tick = FIG-A's ACCENT blue (same number across figs). **Family bar is now
  the TRUE pooled family-vs-rest AUC** (rescore_language.py `family_pooled` field +
  standalone `family_pooled_recompute.py`, numpy-only, cluster-recomputed for all 7
  on 2026-06-16, every model's tokens_code_auc gate reproduced |Δ|≤0.0003) —
  replaces the earlier positive-token-weighted mean of per-CWE `b_probe_auc`, which
  under-reported injection by ~0.008 (sibling-family positives contaminating each
  per-CWE negative pool); memory unchanged. 2 local models matched the cluster
  numbers bit-for-bit vs sklearn. Companion
  `docs/blog/draft-claims.typ`
  = claims-&-evidence edition (one fig6-style plot per claim from result JSONs,
  `make_claim_figs.py`; §4 = language-baseline methodology). RES4 ⚠ resolved:
  MLP "scaling" was exp-09's fixed-layer policy; swept (exp-12/18) it's flat.
  **Example-level honesty figures added (2026-06-14):** three new figs in
  `make_figs.py` re-plotting already-landed exp-29/30/31 numbers —
  `fig_example_chance` (Initial-results upfront beat: token-level AUC 0.74–0.81 vs
  example-level max-pool/last-token 0.51–0.57, "near-random at ranking functions"),
  `fig_commit_honest` (the figure-5 replacement: commit-position probe beats
  verbalized under primed+neutral prompts but sits at the char-n-gram ceiling
  0.778), and `fig_example_verbalized` (exp-29 like-with-like: code-token probe
  max-pool + final-code-token vs verbalized, all example-level, all ≈chance —
  supersedes the old `fig_verbalized_examplelevel`). All use the gemma-blue/
  qwen-brick family hue (verbalized = base + tint hatch). Ready-to-paste prose:
  `posts/.../draft-example-level-story.qmd` (supersedes the two-figure version in
  `draft-commit-position-sections.qmd`). No new finding — presentation of #5/#8.
  **exp-20 string-matcher figure added (2026-06-16):** `fig_stringmatcher` in
  `make_figs.py` (light+dark) for the blog's "Investigating the probe's
  classifications" paragraph (DAT5) — bimodal cross-probe agreement stacked by
  family (16 caught by none / 55 by all 7) from `20-*/fn_corpus.json`, reusing fig_h
  injection-green/memory-vermillion hues. No new finding — presentation of #4. NB the
  exact 0-detect set is **14/16 memory** (+2 CWE-022 path-traversal), not "purely
  memory" as the draft prose says. **An FP-lexical-breakdown second panel was built
  then DROPPED 2026-06-16 (user):** the "51% SQL identifiers / 25% punctuation / 15%
  SQL keywords" split is a method artifact — `fp_buckets.py` is a catch-all
  keyword-regex (the 51% "identifier" bucket is the regex else-branch, holding generic
  vars/literals incl. C), the hand taxonomy was built on a 127/130-**Python** curated
  sample, and `fp_corpus.json` (the 6,119-span population) is gitignored/absent so
  51/25/15 can't be re-verified. The robust FP evidence is the **safe-alarm** result
  (3,125 FPs on patched code; ≈96% of the 350 cross-model FPs on injection pairs — the
  @fig-flips story), not the lexical breakdown. Figure is now single-panel; prose +
  caption edits flagged via a `TODO(edit)` comment in `index.qmd` (not edited directly,
  per user).
- **exp-27 DONE (2026-06-12, ledger row above); exp-28 (PrimeVul deep-dive) IN
  FLIGHT** (chaser sessions in tmux `probes`, briefs at
  `plans/cross-model-probe-generalization/2{7,8}-*/BRIEF.md`). exp-28: PV
  surface baselines + CIs + matched-pair.
- **Last-token introspection (blog NXT2b) — RESOLVED, lexical (exp-29 + exp-30 +
  exp-31, 2026-06-13/14).** exp-29: deployed probe at the final live-code token =
  chance. exp-30: commit-position probe decodes 0.66–0.82 (looked like a positive).
  exp-31 controls settle it: char-n-gram (0.778) ≥ probe for ~all models, and the
  one above-ceiling cell (Qwen-32B primed) is priming-dependent (neutral → ceiling).
  **Verdict: lexical** (finding #8). Survives: probe > verbalized; clears nulls.
  - **Still open (smaller):** per-layer AUC curve (where the signal emerges; only
    deployable+oracle layers persisted) and a yes/no-direction comparison for the
    big models — these refine *mechanism* but won't change the lexical verdict.
    Within-C/C++-specific probing (signal is mostly Python). Hidden states KEPT
    (scratch `lasttoken_hidden_[neutral_]<slug>.npz` + local `30-.../hidden/`).
  - **The genuinely different next frame is the generation pivot (NXT3):** probe
    the model *writing* code, labels from its output — the reading frame (code
    tokens, commit position, primed/neutral) is now thoroughly lexical-ceilinged.
- **Next-phase probes (user direction, 2026-06-11, not yet briefed/run):**
  (1) random-direction AUC nulls per layer; (2) QA-contrast probe on assistant
  answer tokens (sleeper-agents recipe); (3) generation pivot — probe the model
  *writing* (possibly vulnerable) code, labels from analyzing its output.
  Motivation: current reading-frame design never established the model holds a
  vulnerability belief about out-of-context functions (verbalized 0.49–0.62
  consistent with absent, not hidden).
- Research-framing open questions (generation transfer, own-code OOD, intent).
