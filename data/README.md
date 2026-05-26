[ai-generated]

# `data/` — local scratch / cache

This directory is **not** the source of truth — wandb + HF Hub are.
Files here are local cache for artifacts pulled from wandb/HF or staged
before upload. Only the subdir scaffold (`.gitkeep`) is tracked; payloads
(`.npz`, `.jsonl`, `.png`, `.parquet`, etc.) are gitignored.

## Layout

| Subdir       | What lives here |
|---|---|
| `datasets/`  | JSONL / parquet datasets pulled from HF Hub (vuln corpora, SVEN, etc.) |
| `models/`    | Local snapshots of base models (Gemma 3 checkpoints) and saved hidden-state activations |
| `probes/`    | Trained probe `.npz` bundles + their JSON cards, pre-upload to HF |
| `plots/`     | Figures rendered by report scripts / notebooks, pre-upload to wandb |

Add new subdirs freely as needs come up (e.g., `nlas/`, `eval/`,
`activations/`). Update this table when you do.
