# Plans

A plan is a goal-directed group of experiments. Layout:

```
plans/<plan-slug>/
  PLAN.md               # goal + ordered, literature-grounded steps
  01-<exp-slug>/        # first experiment
  02-<exp-slug>/
  ...
```

`PLAN.md` answers:

- **Goal** — one paragraph. What question does this plan answer?
- **Steps** — ordered list. Each step cites the paper / arXiv id / blog
  that motivates it (method or hypothesis).
- **Success criteria** — how we'll know the plan is done.

Each experiment dir has its own `EXPERIMENT.md` with the five-field
briefing (Aim / Inputs / Outputs / Result format / Interpretation hints —
see `CLAUDE.md`).
