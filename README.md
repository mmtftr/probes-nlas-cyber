# probes-nlas-cyber

Mech-interp research on linear probes and natural-language activations (NLAs)
for cyber-vulnerability classification.

## Conventions

| Aspect | Choice |
|---|---|
| Base model | **Gemma 3** |
| Approach | **Probes + NLAs** (natural-language activations) |
| Experiment tracking | **Weights & Biases** runs + artifacts |
| Artifact store | **Hugging Face Hub** (datasets + probe models) |
| Provenance | **wandb artifact lineage** for every input/output |
| Scope | research-only — no product / demo surfaces here |

## Layout

```
src/
  data/
    extract_activations.py       # example-level hidden states → .npz
    extract_token_activations.py # token-level hidden states + offsets + spans
  probes/
    calibration.py               # post-hoc Platt / temperature fitting
  training/
    train_probe.py               # baseline linear probe (last-token)
    train_probe_spanmax.py       # span-max loss (Obeso/Arditi 2025) — primary
  eval/                          # split definitions, metrics, protocols, AST mask
scripts/
  build_dataset_sven.py        # SVEN dataset builder (primary)
  build_dataset_v2.py          # earlier dataset variant (kept for reference)
  derive_rich_labels.py        # rich label derivation
  validate_dataset.py          # dataset validators
  validate_rich_labels.py
  build_repo_benchmark.py      # heldout-repo benchmark assembly
  extract_token_probs.py       # producer for token-level probs npz
  retrain_spanmax_sven_split.py# canonical retrain command
  calibrate_probe.py
  eval_probe.py
  eval_splits.py
  run_token_eval.py
  eval_repo_leads.py
  apply_calibration_eval.py
notebooks/
  training/colab_train_gemma4_probe.ipynb   # will be ported to Gemma 3
  remote/colab_train_probe_31b.ipynb
  remote/kaggle_train_probe_sven_weak.ipynb
tests/
configs/                       # wandb sweep / run configs (empty for now)
plans/                         # goal-directed experiment groups (see CLAUDE.md)
decisions/                     # ADRs for cross-experiment choices
docs/guides/                   # accumulated mech-interp lessons
data/                          # local scratch (datasets/ models/ probes/ plots/) — payloads .gitignored
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
wandb login
huggingface-cli login
```

## Known TODOs

- The 3 carry-over notebooks still reference the old project (paths, pip
  URLs, model IDs). They're staged here as a starting point — full rewrite
  needed for Gemma 3 + wandb logging.
- `train_probe*.py` and `extract_*_activations.py` still hardcode `gemma-4`
  model IDs. Switch to `google/gemma-3-*` and verify activation shapes.
- All artifact I/O still writes to local `data/`. Rewrite to push/pull
  `wandb.Artifact` objects backed by HF Hub. (The old `publish_to_hub.py`
  was deleted in the carry-over — the new flow is wandb-linked, not a
  standalone publisher.)
- The eval framework writes JSON cards next to artifacts. Log everything as
  wandb tables + summary metrics instead.
- A scan abstraction was intentionally left out of the carry-over (it was
  entangled with downstream demo glue). Reintroduce a clean one if needed.
- NLAs: scope TBD.
