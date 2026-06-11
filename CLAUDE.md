[ai-generated]

# Agent guide

This repo is a mech-interp research project. Probes + NLAs on Gemma 3.
Read `README.md` for what's in `src/` and `scripts/`.
Read `docs/research-framing.md` before designing any experiment — it holds the
target property, scope, and open questions the work is narrowing toward.

**Read `docs/project-log.md` FIRST, every session.** It is the unified
high-level log: the experiment ledger (every exp with its aim + headline finding
+ metric + status), the consolidated current understanding, the standing
conventions, and the open threads. It exists so you hold the whole project in
mind and never re-run or contradict prior work. Keep it updated: when an
experiment lands (or is retracted), add/edit its ledger row in the same change.

## Hidden-state extraction — default to vLLM (do not re-litigate)

For per-token hidden-state extraction, **use vLLM's `extract_hidden_states` API,
not transformers batch-1.** It is ~2.2–2.5× faster (FLASH_ATTN, prefill-only).
This is settled — do not argue for HF `output_hidden_states`; HF is the
**fallback only** when vLLM genuinely can't be installed on the target.

- Reference: `docs/vllm-hidden-states-extraction.md` (API + gotchas); working
  harness `docs/colab-vllm-bench/bench_vllm.py`.
- **Layer convention:** repo-layer L = transformers `hidden_states[L+1]` =
  output of block L. In vLLM pass `eagle_aux_hidden_state_layer_ids = L+1` to
  get repo-layer L (vLLM aux id `i` → `hidden_states[i]`).
- **Must-haves:** `enable_prefix_caching=False` (SVEN before/after pairs share
  long prefixes → cached tokens emit no hidden states), `attention_backend=
  "FLASH_ATTN"`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `SamplingParams(max_tokens=1)`,
  and pass pre-tokenized `prompt_token_ids` (truncated to match HF token counts).
- **Install matched to your GPU:** vLLM isn't always preinstalled on a given
  host; install it matched to that host's torch/CUDA build via `uv pip install`.
- **Persist single/few-layer acts — never re-extract them.** One layer is small
  (~hidden_dim × n_tokens × 4 B ≈ 11 GB for a 32B model, far less for smaller
  ones). ALWAYS keep extracted single/operating-layer activations on disk
  (`KEEP_ACTS=1`, no delete) so any follow-up (more seeds, relabel, new probe
  family) reuses them. Only the full multi-layer band (hundreds of GB–TBs) gets
  deleted after use.

## Probe training — batch, don't loop

Probes are linear (or tiny-MLP) heads on **cached** activations; training is
trivial FLOPs. Train all configs/folds/seeds **together, vectorized** — one
`X @ W` matmul (hidden → K) for all K probes at once, segment-max pooling
vectorized, **full-batch** GD (no 8-example mini-batches for a linear head).
Do NOT run K separate `train_one_layer` calls in a Python loop: the
epoch×batch×example triple-loop is overhead-bound and ~100× too slow (e.g. 120
CV probes took ~1 h that should be minutes). Parallelism is across probe heads
in one GPU pass, not across processes.

## Collaboration model

The agent is a **collaborator**, not an autonomous driver, in this repo.

- **Don't change the user's ideas without explicit approval.** Record their
  framing faithfully (see the *User's framing* blocks in
  `docs/research-framing.md`). Suggestions and nudges are welcome but must be
  clearly fenced as the agent's, carrying no decision weight until adopted.
- **Surface ambiguities; get an explicit decision.** Anything the user left
  unspecified that changes what you produce → ask, don't guess.
- **Mark silent decision-forks.** When implementation forces a choice the user
  hasn't made, leave a `TODO(adhoc-decision)` marker at the site (code comment
  or doc line) so it can be reviewed. Consolidate open ones in
  `docs/research-framing.md` §6 and clear them into ADRs once settled.

## Writing style

Human-facing docs are concise. The user has the context — no hand-holding,
no restating, no follow-up suggestions. Bullets over paragraphs. Skip prose
that carries no new information.

Agent-facing sections can be verbose (disambiguation, exact commands,
assumptions worth stating). Mark them with `## For agents` inside an
otherwise-concise doc. Plans illustrate this: `Goal` and `Steps` are tight
for the human; optional `## For agents` holds the detailed playbook.

## Marking AI-generated content

Disclose AI authorship inline. Conventions:

- **New file written entirely by an agent**: literal `[ai-generated]` on the
  first line (markdown) or `# [ai-generated]` (Python). Stays visible — it's
  a provenance disclosure, not a hidden tag.
- **Section inside a mixed-authorship file**: prefix the section heading
  with `[ai-generated]`, e.g. `## [ai-generated] Method`.
- **Carried-over code from prior projects** keeps its original provenance
  (no tag added retroactively).

Human authorship is the default — mark only AI work.

## Plans group experiments

Work lives in `plans/<plan-slug>/`. Each plan has a `PLAN.md` with:

- **Goal** — the question the plan answers
- **Steps** — ordered, each citing literature (arXiv, paper, blog) that
  motivates or methodologically grounds the step

Experiments inside a plan live in `plans/<plan-slug>/<NN>-<exp-slug>/`.
Each experiment dir is **self-contained**: its briefing (`EXPERIMENT.md`),
its scripts, and any local outputs all live there. Don't add experiment
scripts to the global `scripts/` — that's for shared CLIs only.

## Experiment workflow

Before running an experiment, brief the user with these five fields. Keep
each tight — the user needs to understand the experiment, not skim a wall.

1. **Aim** — what the experiment tests; the hypothesis in one sentence.
2. **Inputs** — dataset, model, probe, hyperparams. Reference wandb
   artifacts or HF revision SHAs by name, not by local path.
3. **Outputs** — what artifacts the run produces and where they land
   (wandb run name, artifact names).
4. **Result format** — the specific numbers, plots, or tables to be
   reported back.
5. **Interpretation hints** — what each plausible outcome would mean. If
   you can't write these, the experiment isn't well-scoped yet.

**Do not run until the user signals understanding.**

## Review gate — MANDATORY before any result reaches the user

A result/analysis is **not** presentable until it has passed an independent
review. This is a hard gate, not a nicety: it exists because an un-reviewed run
once shipped a wrong-metric, prior-work-duplicating conclusion to the user. Do
**both** passes, in parallel, before you write up or post anything:

1. **Independent adversarial pass — `cj` (codex) or `aj`.** Hand it the script +
   the headline claim and tell it to try to break the conclusion. Fresh model,
   no shared context → catches what you rationalized.
2. **Opus subagent one-pass** (Agent tool) over the same artifacts.

Both reviewers must explicitly check, and you must clear, this checklist:

- [ ] **Metric is the default.** Headline is `tokens_code_auc` (honest token-level
      AUC) unless the user asked otherwise. Any other metric (example-AUC,
      pair-ranking, detection-rate) is *secondary* and labelled as such — **never**
      base a "signal absent / unlearnable / works" claim on a non-default metric.
- [ ] **Prior work checked (reuse).** Search `plans/` and `docs/project-log.md`:
      did this experiment already run? Does it contradict an existing finding? If
      it duplicates or conflicts, stop and reconcile before presenting.
- [ ] **Methodology sound.** Correct split (no leak), right negative pool, honest
      labels, no off-by-one, n large enough for the claim (flag tiny-n cells).
- [ ] **Conclusion matches the metric and the numbers** — no overclaim beyond what
      the chosen metric supports.

Apply **blocking** fixes (anything that would change the conclusion). Skip
nitpicks. Only after both passes clear the checklist do you write the result up
or send it to the user. Then update the `docs/project-log.md` ledger row.

## Tracking

- Every experiment = one wandb run.
- Inputs and outputs go in as `wandb.Artifact`s.
- If an artifact would exceed ~100 MB, store it on HF Hub and log only the
  HF revision SHA + path as wandb run config. Never silently truncate.
- Log `git rev-parse HEAD` per run. Refuse to start on a dirty tree unless
  the user explicitly overrides — provenance breaks otherwise.

## Decisions

Anything that affects future experiments (model swap, split redefinition,
loss change, probe family change) gets an ADR in
`decisions/NNNN-<slug>.md` with sections: Context, Decision, Consequences.
Date-stamp the file.

## Guides

Recurring mech-interp lessons accumulate in `docs/guides/<topic>.md`.
Read the existing guides before designing a new experiment. Keep entries
short, factual, and citation-backed.

## Papers

Markdown summaries / notes of relevant papers live in `docs/papers/`.
One file per paper. Treat these as agent-readable reference material —
grep them when designing experiments to cite literature.

## Archiving

Anything superseded or shelved moves to an `archive/` subdir rather than
being deleted:

- `decisions/archive/`     — ADRs explicitly replaced by a newer one
- `plans/archive/`         — plans that are done or abandoned
- `plans/<slug>/archive/`  — individual experiments within an active plan

Archiving preserves history without cluttering the active workspace.
