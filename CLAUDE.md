[ai-generated]

# Agent guide

This repo is a mech-interp research project. Probes + NLAs on Gemma 3.
Read `README.md` for what's in `src/` and `scripts/`.
Read `docs/research-framing.md` before designing any experiment — it holds the
target property, scope, and open questions the work is narrowing toward.

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
  (no tag added retroactively). The carry-over event itself is logged in
  `claude-project-log.md`.
- The running session log is `claude-project-log.md`. Append, don't rewrite.

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

After the run completes, spawn a second subagent to review. Apply
**blocking** fixes — anything that would change the conclusion (wrong
split, leaky baseline, miscounted positives, off-by-one in metrics).
Skip nitpicks (style, wording, harmless redundancy).

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
