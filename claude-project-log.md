[ai-generated]

# Claude project log

Append-only. `[human]:` prefix for hand edits.

---

## 2026-05-26 — bootstrap

Carved fresh repo from gemmaforge's last-accepted lineage.

Commits: `1536b2e` carry-over · `4317bdb` workflow scaffold + src/ refactor.

- **Carried over:** spanmax + SVEN training, `src/eval/`, 3 notebooks,
  relevant tests.
- **Dropped:** LoRA, MLP / attention / ensemble / per-CWE / value-head
  variants, adversarial suite, demo / Space / pwnkit. `publish_to_hub.py`
  deleted.
- **Renames:** `gemmaforge.repo_benchmark/v1` → `probes.repo_benchmark/v1`;
  `gemmaforge_top_cwe` → `probe_top_cwe`. Docstring `GemmaForge` stripped
  from `.py`/`.md`/`.toml`. Notebooks left as-is (rewrite TODO).
- **Layout:** `CLAUDE.md` (+ `AGENTS.md` symlink), `plans/`, `decisions/`,
  `docs/{guides,papers}/`. `src/` → `{data,probes,training}/`.
  `data/{datasets,models,probes,plots}/` each with one-line README.
- **TODOs:** Gemma 4→3 in extractors + trainers · notebook rewrite for
  wandb · artifact I/O → `wandb.Artifact` backed by HF · eval JSON cards
  → wandb tables · scan abstraction reintroduce on demand · NLAs scope TBD.

---

## 2026-05-30 — research framing + first papers

Narrowed project scope with the user.

- **Added `docs/research-framing.md`** (living charter): target property =
  model's belief about how vulnerable *input-stream* code is; 3 follow-on
  complications (gen-vs-read direction, own-vs-others OOD bias, intent-to-
  exploit); 5 open framing questions; baselines status; consolidated
  `TODO(adhoc-decision)` list (§6).
- **Decision:** scope first experiments to the **base property**; treat the 3
  complications as follow-on.
- **Papers pulled:** `ribeiro2025-internal-rep-code-correctness.md`
  (arXiv:2512.07404, RepE/LAT contrast-pair, *reads* code) and
  `bui2025-openia-correctness.md` (arXiv:2501.12934, supervised probe on the
  model's *own generations*). Both: signal mid-late layers, final code token,
  model-specific reps. Added `docs/literature-review-1.md` scaffold (user fills
  What-we-learn / What-we-can-adopt).
- **CLAUDE.md:** added *Collaboration model* (agent = collaborator; don't alter
  user ideas; surface ambiguities; `TODO(adhoc-decision)` convention).
- **Baselines:** confirmed random/length/regex + sample-level probe +
  `fit_logreg_on_split` already carried over in `src/eval/`. LAT and
  CodeBERT/CodeT5+ flagged as candidate additions (not implemented).

---

## 2026-05-30 — failure modes & mitigations recorded

- **`docs/research-framing.md`:** added §6 *Failure modes & risks* and §7
  *Mitigation ideas* (user's framing, recorded faithfully); renumbered the
  consolidated decisions list §6 → §8. No prior cross-refs pointed at §6.
- **Failure modes (user):** probe inner misalignment (dataset bias /
  low-quality-not-vulnerable / short→long non-generalization); usefulness
  (over- vs. under-detection); architecture sufficiency vs. scope creep;
  dataset label noise + context difficulty; tunnel vision (white-box ≥ probes);
  provenance non-generalization (in-the-wild vuln probe may not fire on
  assistant-generated code).
- **Mitigations (user):** function→full-file context-length test (cites
  hallucination probes generalizing long→short but not short→long); LLM-rewrite
  secure→low-quality to test quality overfit; off-distribution / other-repo
  eval for dataset-bias overfit; ask-the-LLM-directly + other baselines to avoid
  tunnel vision; classifier-property audit for over/under-detection.
- Added fenced **Agent notes** cross-linking each to existing baselines/splits/
  metrics and the open §8 decisions; no new decisions forced.
