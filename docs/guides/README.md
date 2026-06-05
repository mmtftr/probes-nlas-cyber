[ai-generated]

# Guides

Recurring mech-interp lessons learned in this repo. Agents read these
before designing a new experiment.

One file per topic: `docs/guides/<topic>.md`. Keep entries short, factual,
and citation-backed. If a guide grows past ~1 page, split it.

## Existing guides

- [the cluster-cluster](the cluster-cluster.md) — HPC access: container, scheduler
  (1-job limit), sequential orchestration, dependency isolation.
- [probe-activation-extraction](probe-activation-extraction.md) — float32 acts,
  streaming memmaps, tokenizer-offset requirement, group-clean splits.
- [activations-hf-dataset](activations-hf-dataset.md) — `mmtf/probes-activations`
  HF dataset: per-model/per-layer bf16 files, layout, loading, generation
  reference, download-vs-regenerate, Xet-disable upload lesson.
- [span-max-loss-tuning](span-max-loss-tuning.md) — the loss, α/ω, neg_incl
  variant, per-model layer selection, repeated-split variance.

## Suggested topics as they come up:

- Layer selection for probe training
- Span-max loss tuning (ω schedule, α weight)
- Calibration: Platt vs. temperature, when each fits
- Leakage-aware split design (group-by-repo, heldout-CWE, heldout-lang)
- Code-mask AST quirks per language
- NLA design patterns (TBD)
