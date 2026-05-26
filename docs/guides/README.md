[ai-generated]

# Guides

Recurring mech-interp lessons learned in this repo. Agents read these
before designing a new experiment.

One file per topic: `docs/guides/<topic>.md`. Keep entries short, factual,
and citation-backed. If a guide grows past ~1 page, split it.

Suggested topics as they come up:

- Layer selection for probe training
- Span-max loss tuning (ω schedule, α weight)
- Calibration: Platt vs. temperature, when each fits
- Leakage-aware split design (group-by-repo, heldout-CWE, heldout-lang)
- Code-mask AST quirks per language
- NLA design patterns (TBD)
