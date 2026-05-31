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

---

## 2026-05-31 — Clariden cross-model sweep launched (overnight, GH200)

8 h GH200 window on CSCS Clariden (account `lsaie-ss26`, partition `debug`,
4× GH200/node, 90 node-min/job). Method-generalization sweep per
`plans/cross-model-probe-generalization/`.

- **Orchestrator** `src/remotes/clariden/` (env.sh / run_model.sh / train_eval.py /
  submit.sh / models.txt / overnight.sh). Single-node jobs, 4 models/GPU packed,
  idempotent (DONE markers), deps installed once into shared `.python_deps`
  (transformers 4.57.1 + matched hub/tokenizers; login node is py3.6 so install
  must be in-container).
- **Roster** 23 models (dense ≤33B + small MoE), all confirmed to fit one GH200;
  HF token validated, all gated (Gemma-3 family + Llama-3.1) reachable.
- **Smoke (validated end-to-end):** deepseek-coder-6.7b ex-AUC 0.705 / tok 0.851
  (best layer frac 0.97); Qwen2.5-Coder-7B 0.694 / 0.820 (0.96); Qwen3-8B 0.711 /
  0.845 (**frac 0.50, mid-net**). Probe beats regex (0.53) / length (0.58).
  Early signal: best-layer fraction varies by family → motivates the layer sweep.
- **Teammate fixes folded in:** submit.sh `pmi2`; extractor `dtype`/`torch_dtype`
  fallback; `scripts/build_dataset_sven_canonical.py` (1560 rows, 780/780);
  slug `printf` fix.
- **Unattended:** `overnight.sh` (nohup on login node) waits out the smoke job,
  submits the full roster, and re-submits gaps (≤2 attempts/model). gemma-4 /
  qwen3.6 (2026 archs) may exceed transformers 4.57 → expected MISS, capped.
- Results: `~/scratch/probes/runs/<slug>/metrics.json`; progress in
  `~/scratch/probes/overnight.log`. NOT yet logged to wandb (TODO).
- **QOS gotcha (fixed):** `debug-qos` rejects a 2nd queued job
  (`QOSMaxSubmitJobPerUserLimit`; nominal MaxSubmitPU=2 but races → effective 1).
  `submit.sh` now drip-feeds at `MAXQ=1` (submit only when 0 of our jobs queued)
  → strictly sequential, ~30 min/chunk for big models. Verified: 0 QOS errors
  across 4 consecutive submissions; 18/23 done within ~1 h of the fix.
- Slug fix changed run-dir names → two early-smoke models have stale `_`-suffixed
  dirs (`*_/`) alongside clean ones; harmless, dedupe at collection.

## 2026-05-31 — cross-model sweep: recover 2/5 failures, drop 3, bump to transformers 5.9.0

Diagnosed the 5 overnight `EXTRACT_FAILED` models and drove the sweep to its final
state (20/23). See `decisions/0001-cross-model-roster-drops.md`.

- **Stack bump**: `env.sh` now installs transformers 5.9.0 / tokenizers 0.22.2 /
  hub 1.17.0 into a fresh `.python_deps5` (4.57.1 tree at `.python_deps` kept as
  rollback; `$TF_STACK` overrides). Verified `gemma4`/`qwen3_5`/`mistral3` ∈
  `CONFIG_MAPPING` before pinning, and re-loaded a DONE model (gemma-3-1b) through
  the new loaders end-to-end.
- **Xet fix**: hub 1.17's Xet path calls `download_files(request_headers=)` which
  the image's `hf_xet` rejects → every fresh download died. Set
  `HF_HUB_DISABLE_XET=1` (classic HTTPS). This was the single root cause of all 5
  failures *re-failing* on first 5.9.0 attempt.
- **Extractor** (`src/data/extract_token_activations.py`): `trust_remote_code=True`
  + `AutoProcessor`-tokenizer fallback (now re-raises the original error instead of
  masking) + `AutoModelForImageTextToText` fallback for VLM wrappers (engaged for
  Mistral-Small-3.2).
- **Recovered**: gemma-4-26B-A4B (ex-AUC 0.702 / tok-AUC 0.840, best layer 15/30)
  and Qwen3.6-27B (0.701 / 0.855, best layer 32/64) — both beat baselines on the
  shared split, the two archs that strictly needed >=5.9.0.
- **Dropped (3)**: OpenCoder-8B (slow `INFLMTokenizer`, no offset_mapping);
  Devstral-2507 + Mistral-Small-3.2 (Tekken→151000-vocab vs 131072 embeddings →
  OOB CUDA assert; no clean offset-preserving fix on 5.9.0). User-confirmed both
  drop decisions rather than ship approximate offsets.
