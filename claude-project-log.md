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
