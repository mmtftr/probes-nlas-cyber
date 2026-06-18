# probes-nlas-cyber

**Blog post → [Measuring Probe Performance in Cybersecurity](https://mmtf.dev/posts/vulnerability-probes-lexical-ceiling/)** (mmtf.dev) — the full writeup of what this repo found.

Mech-interp research: can a linear probe on a code model's activations detect the
*tokens* that cause a vulnerability? Probes are trained on Gemma-3 (1B–27B) and
Qwen2.5-Coder (7B, 32B) activations and stress-tested against strong lexical
baselines and a stack of deconfounding controls.

The *North Star* is cheap inference-time monitoring for model sabotage; the
*proxy task* here is detecting vulnerabilities in existing code.

> Done as part of Bluedot Impact's Technical AI Safety Project course.

## What it found

Token-level probes reach 0.75–0.82 AUC, but under stricter controls and stronger
baselines the signal is largely **lexical**, not an understanding of
vulnerability:

- A character n-gram baseline **beats** the probe (0.803 vs 0.776 token AUC).
- A language-only baseline (Python→1, C→0) recovers ~64% of the probe's
  margin over chance, carrying zero vulnerability information.
- Probes catch injection-class bugs (SQL, command; 0.85–0.91) but sit near
  chance on memory-safety bugs.
- Naive example-level readouts (max-over-tokens, final-token) are near chance;
  specific prompts plus probing the assistant turn recover some of it.
- Inspecting predictions, the probe paints whole SQL/command sink strings in
  both the vulnerable and the patched version — it matches code patterns, not
  the edit that fixes the bug.

**Takeaway:** linear probing should not be used to detect vulnerabilities in
isolated pieces of code — lexical baselines are cheaper and stronger. Whether
LLMs encode "I'm writing vulnerable code" remains open; the proper test is on
model-generated code, not a fixed dataset.

## Method (short)

- **Data** — SVEN paired vulnerable/patched functions, turned into token-level
  labels (tokens the fix removed = positive); PrimeVul (C/C++) for cross-dataset
  checks.
- **Probe** — linear token probe on the residual stream, span-max annealed loss
  (hallucination-probes), best layer chosen on validation AUC. Also MLP heads and
  K-ensembles of linear probes.
- **Metric** — ROC-AUC over live-code tokens only (comments/trivia filtered out).
- **Baselines** — character n-gram, token-unigram, language-only.
- **Controls** — per-CWE, per-language, subtractive-vs-additive pairs,
  matched-patch, cross-dataset transfer, causal steering.

Full framing is in `docs/research-framing.md`; the experiment ledger and
consolidated current understanding live in `docs/project-log.md` (read it first).

## Conventions

| Aspect | Choice |
|---|---|
| Models | **Gemma 3** (1B–27B) + **Qwen2.5-Coder** (7B, 32B) |
| Approach | linear / MLP / ensemble **probes** on cached activations |
| Hidden states | **vLLM** `extract_hidden_states` (HF fallback) |
| Experiment tracking | **Weights & Biases** runs + artifacts |
| Artifact store | **Hugging Face Hub** (datasets + probe models) |
| Scope | research-only — no product / demo surfaces here |

## Layout

```
src/
  data/     extract_activations.py / extract_token_activations.py  # hidden states → npz (+ offsets, spans)
  eval/     splits, metrics, protocols, live-code (AST) mask, lexical baselines
  probes/   calibration.py          # post-hoc Platt / temperature fitting
  remotes/  train_eval.py           # shared SVEN split / grouping / scoring helper
  training/ train_probe.py          # baseline last-token linear probe
            train_probe_spanmax.py  # span-max annealed loss (primary)
scripts/    dataset builders, probe training/eval CLIs, calibration
plans/cross-model-probe-generalization/   # experiments 02–33 (see PLAN.md, CLAUDE.md)
decisions/  # ADRs for cross-experiment choices
docs/       # project-log.md, research-framing.md, guides/, papers/, blog/
data/       # local scratch (datasets/ models/ probes/ plots/) — payloads .gitignored
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
wandb login
huggingface-cli login
```
