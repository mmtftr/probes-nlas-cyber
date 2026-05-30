[ai-generated]

# Research framing

Living charter for this project's goals and approach. Plans (`plans/`) cite it;
decisions (`decisions/`) lock individual choices once made. Last narrowed:
2026-05-30.

> **Authorship:** the *User's framing* blocks below are the user's ideas,
> recorded verbatim-in-spirit. The agent must not alter them without explicit
> approval. Blocks marked **Agent note** are suggestions/nudges only — they
> carry no decision weight until the user adopts them. Implementation forks the
> agent would otherwise pick silently are marked `TODO(adhoc-decision)`.

---

## 1. The property we want to detect

**User's framing.** The property is **the model's own belief about how
vulnerable the code it *sees in its input stream* is** — code appearing in the
assistant section or in a tool-output section. Not "is this code vulnerable"
ground truth; the *model's internal judgment* of it.

**Committed first target (decision 2026-05-30):** scope the first experiments to
this **base property — vulnerability belief over input-stream code**. The three
complications in §2 are real but are *follow-on* directions, not the first cut.

**Agent note.** This is exactly the shape of the code-correctness work we have
notes on: a code property the model represents linearly in its hidden states.
- Ribeiro et al. 2512.07404 — unsupervised contrast-pair RepE/LAT, *reads* code.
- Bui et al. (Openia) 2501.12934 — supervised probe, reads the model's *own
  generations*. (`docs/papers/`.)
Both find the signal in middle-to-late layers at the final code token. Our
existing `train_probe*.py` (supervised, span-max) is in the Openia family.

---

## 2. Complications before this generalizes to a "model-specific vulnerability detector"

**User's framing** (recorded; follow-on, not first cut):

1. **Generation vs. reading direction.** The model may encode vulnerability in a
   *different direction* when it is *generating* code on its own, vs. when it
   *reads* code in its input. A probe fit on reading may not transfer to
   generation.

2. **Own-code vs. others'-code OOD bias.** For out-of-distribution code, the
   model might be biased such that it *will not* recognize its *own* vulnerable
   code but *will* recognize the vulnerable code of others.

3. **Intent to exploit (not author).** The model could intend to generate code
   that is vulnerable *only* when combined with other vulnerable components — so
   the model is not writing the vulnerable code itself, but *exploiting* it. This
   is an intent-to-exploit signal, distinct from authoring a vulnerability.

**Agent note.** The literature partly speaks to each:
- (1) ↔ Openia probes generation-time states; Ribeiro probes reading. We can
  test transfer directly by fitting on one and evaluating on the other.
- (2) ↔ Ribeiro claims correctness reps are *source-agnostic* (fitted-on-self,
  scores any author). That's a *claim to test* for vulnerability, not assume.
- (3) has **no clean analogue** in the correctness papers — it's an intent
  property, not a code-property. Likely the hardest to dataset honestly and the
  most AI-control-relevant. `TODO(adhoc-decision)`: whether (3) is in scope for
  this repo at all or spins out separately. Flagged, not decided.

---

## 3. Downstream tasks (why this matters)

**User's framing.** We precisely care about many downstream tasks, including
**AI-control scenarios** — this is not a generic "vulnerable code" classifier.

**Agent note (candidate concretizations — for user to accept/reject):**
- Monitor: flag when a model under control is *reading* attacker-supplied
  vulnerable code (prompt-injection / poisoned tool output) — base property §1.
- Monitor: flag when a model *writes* vulnerable code — needs direction (1).
- Detect intent-to-exploit in multi-component settings — needs (3).
These are suggestions; the AI-control task list is the user's to define.

---

## 4. Open framing questions

The user's questions, preserved. Each has an **Agent note** with nudges grounded
in our papers/baselines. None of these answers are decided.

**Q1 — What property do I expect my probes to detect?**
See §1. *Agent note:* keep the target operational: "score in [0,1] = model's
belief that input-stream code is vulnerable," fit against a chosen vulnerability
label. The label choice (CWE-positive? SVEN pair? human?) is itself a decision.

**Q2 — How best can I model the wanted property in my eval datasets, and how can
I eliminate the biases that'd stem from modeling it?**
*Agent note:* the contrast-pair construction (same task, vuln vs. fixed) is the
bias-control workhorse in both papers — it holds task/topic fixed so the probe
can't cheat on surface features. Our repo already worries about this: see
`src/eval/baselines.py` (length/regex baselines exist precisely to catch
length-artifact and pattern-count leakage) and the leakage-aware splits in
`src/eval/splits.py` (group-by-repo, heldout-CWE/lang). `TODO(adhoc-decision)`:
which contrast structure is canonical (SVEN vuln/fix pairs vs. synthesized
pairs). Flagged.

**Q3 — What baselines should I evaluate, how complex should I make my probes, and
how should I select which layer to probe?**
*Agent note:* baselines status in §5. Probe complexity — both papers find a
*linear/small* probe at one mid-late layer suffices (Openia: 2-layer MLP;
Ribeiro: single linear PC). Our `fit_logreg_on_split` is the linear baseline.
Layer selection — both pick by held-out validation accuracy; selected layers sit
middle-to-late (Openia ≈16–28; Ribeiro within ~7pp of oracle). `TODO(adhoc-decision)`:
sweep-all-layers-then-select vs. fix a layer a priori. Flagged.

**Q4 — Should the probe be treated and evaluated as a classifier? If so, what
metrics?**
*Agent note:* yes for the base property — it's binary (vuln belief). Our eval
already uses ROC-AUC vs. baselines (`src/eval/metrics.py`). But "belief" is
calibrated-probability-flavored, so report **calibration** (we have
`src/probes/calibration.py`: Platt/temperature) alongside AUC, and the
threshold-dependent operating point (precision/recall at the deployed threshold)
since AI-control monitors run at a fixed threshold. `TODO(adhoc-decision)`:
headline metric (AUC vs. recall@fixed-FPR). Flagged.

**Q5 — What are my expectations on how different models / post-training
differences affect probe performance?**
*Agent note:* both papers show reps are **model-specific** (one model's probe
can't read another's states) but the *method* transfers across models. Expect:
per-model probes; post-training (instruct vs. base, RLHF) plausibly *strengthens*
a "vulnerability belief" direction if safety training made the model track it.
This is a hypothesis to test, not an assumption. We are on **Gemma 3** (README);
the correctness papers used 1.3–13B models incl. Gemma-3-1B as a generator.

---

## 5. Baselines (status)

Already carried over and live in `src/eval/`:
- `RandomBaseline` (AUC≈0.5 floor), `LengthBaseline` (length-artifact detector),
  `RegexBaseline` (Semgrep-style pattern count) — `baselines.py`.
- `ProbeBaseline` / `BroadcastProbeBaseline` — wrap a shipped sample-level probe
  as a baseline for the token-level probe to beat — `baselines.py`.
- `fit_logreg_on_split` — linear logistic probe on activations — `probe_io.py`.

Candidate *additional* baselines from the literature (not yet added):
- **LAT / RepE contrast-pair** (Ribeiro) — unsupervised PCA-on-difference-vectors.
- **CodeBERT / CodeT5+ post-hoc classifier** (Openia's black-box baselines).
`TODO(adhoc-decision)`: which of these to implement, and whether LAT enters as a
*baseline*, a *probe family*, or both. Flagged.

---

## 6. Failure modes & risks

**User's framing** (recorded; these are the things that could make the result
wrong or useless, grouped by kind):

**Probe inner misalignment** — the probe fires on the wrong thing:
- Probe detects **dataset bias** rather than vulnerability.
- Probe detects **low-quality code**, not necessarily *vulnerable* code.
- A **short-context** probe does not generalize to **long-context** code.

**Probe usefulness** — even a "working" probe can fail to help:
- Too general / **overdetects** → not useful.
- **Underdetects** → lulls human reviewers into complacency.

**Probe architecture**:
- Architecturally simple probes might not be sufficient to capture code-
  vulnerability understanding.
- Exploring other architectures is valuable, but risks **scope creep**.

**Dataset problems**:
- Dataset code may contain **unmarked vulnerabilities** (label noise).
- Giving the model the **proper context** is difficult.

**Tunnel vision on probes**:
- **White-box techniques** other than probes may do as well or better.

**Probe non-generalization across code provenance**:
- A probe that detects vulnerable code *in the wild* might **not fire on
  assistant-generated code**.

**Agent note (mapping to existing framing — fenced, no decision weight):**
- Provenance non-generalization ↔ §2.1 (generation vs. reading) and §2.2
  (own-code vs. others'-code OOD bias) — same risk, viewed as a failure mode.
- "Detects dataset bias / low-quality / length" ↔ exactly what the
  length/regex/random baselines and leakage-aware splits in §5 / Q2 exist to
  catch; these failure modes are the *reason* those baselines must clear bar.
- "Simple probe insufficient vs. scope creep" ↔ the open probe-family decision
  in §8 (stay linear vs. Openia-style MLP/layer-concat).
- "Underdetection breeds complacency" is the operating-point argument in Q4:
  report recall at the deployed threshold, not just AUC.

---

## 7. Mitigation ideas

**User's framing** (recorded; experiments/checks that probe each risk in §6):

1. **Context-length generalization test.** Run short-context probes on the
   *longer source document* the snippet came from (function → full-file). If
   generalization holds, good; if not, pivot to longer-context training.
   *User's cited prior:* hallucination probes generalize **long → short** but
   **not short → long**. (Tests "short-context doesn't generalize" in §6.)

2. **Code-quality overfit test.** Use an LLM to **rewrite secure code into
   low-quality (but still secure) code**; if the probe fires, it's tracking
   quality, not vulnerability. (Tests "detects low-quality code" in §6.)

3. **Dataset-bias overfit test.** Evaluate on a **different-distribution
   dataset**, and/or pull a few examples from **other repositories**, to see if
   the probe overfits dataset bias. (Tests "detects dataset bias" in §6.)

4. **Anti-tunnel-vision baselines.** Add baselines beyond probes — including
   **just asking the LLM directly** — to check whether the probe is performing
   as well as it can. (Tests "white-box ≥ probes" in §6; extends §5 baselines.)

5. **Classifier-property audit.** Compute the probe's classifier properties on
   the current code to quantify **over- vs. under-detection**. (Tests the
   "usefulness" failure modes in §6; feeds the metric choice in Q4.)

**Agent note (where these touch the repo — fenced):**
- (2) and (3) are contrast-control checks the existing baselines (`baselines.py`)
  and leakage-aware splits (`splits.py`) are built to support.
- (4) "ask the LLM directly" is a new baseline family — relates to the §8 open
  item on additional baselines; the prompted-LLM judge is a black-box analogue
  to Openia's CodeBERT/CodeT5+ baselines.
- (5) maps onto `src/eval/metrics.py` (ROC) + `src/probes/calibration.py`;
  report the confusion matrix / precision-recall at threshold, not only AUC.

---

## 8. Decisions still owed by the user (consolidated `TODO(adhoc-decision)`)

- Is complication §2.3 (intent-to-exploit) in scope for this repo? (§2)
- Canonical contrast structure for datasets: SVEN pairs vs. synthesized. (§2/Q2)
- Layer policy: sweep-then-select vs. fixed a-priori layer. (Q3)
- Headline metric: AUC vs. recall@fixed-FPR. (Q4)
- Probe family: stay linear vs. adopt Openia-style small MLP / layer-concat. (Q3)
- Additional baselines to implement: LAT, CodeBERT/CodeT5+. (§5)

When any of these is settled, record it as a `decisions/NNNN-<slug>.md` ADR and
delete its line here.
