[ai-generated]

# 18 — MLP-probe logit dump — RESULTS (2026-06-06)

Third logit-dump variant: exp-16 = linear span-max probe, exp-17 = verbalized
read, this = the MLP-probe family (exp-12). Materialised every per-token +
per-example MLP logit for 6 instruct models × both heads (mlp256, mlp512). MLP
scores as `sigmoid(probe(X))`. Ran as ONE resumable multi-GPU run.

## MLP `tokens_code_auc` (held-out) + best layer per head

| model | n_layers | mlp256 best-L / test_tc | mlp512 best-L / test_tc | gate (vs exp-12) |
|---|---|---|---|---|
| gemma-3-1b-it | 26 | L13 / 0.801 | L16 / 0.791 | — (new) |
| gemma-3-4b-it | 34 | L8 / 0.799 | L8 / 0.806 | — (new) |
| gemma-3-12b-it | 48 | L13 / 0.806 | L12 / 0.809 | — (new) |
| gemma-3-27b-it | 62 | L21 / **0.822** | L21 / **0.824** | 0.822 / 0.824 · Δ≤0.0002 ✓ |
| Qwen2.5-Coder-7B-Instruct | 28 | L5 / 0.823 | L9 / 0.828 | — (new) |
| Qwen2.5-Coder-32B-Instruct | 64 | L37 / **0.817** | L37 / **0.816** | 0.817 / 0.816 · Δ≤0.0005 ✓ |

Example-level AUC (max-pool, rides along) is much weaker: 0.56–0.66 across the board.

## Read

- **Gate reproduced exp-12 to ≤0.0005** (effectively bit-exact) on all 4
  historical cells → re-extraction + the seeded MLP train (seed=7) are faithful;
  all 12 logit sets are trustworthy. Internal check: for every sweep model×head
  the dump's test_tc equals the sweep cell at the selected layer to 6 decimals.
- **MLP token-code AUC is remarkably flat with scale** — ~0.79–0.83 from 1b to
  32b. The token-level vulnerability signal is decodable by a small 2-layer MLP
  even at 1b. (Contrast exp-17's verbalized read, which scaled 0.49→0.62.)
- **Small models' MLP best layer is early/mid, not deep** — Coder-7B L5/L9,
  gemma-4b L8, gemma-1b/12b L12–16; vs the big models' deeper picks (27b L21,
  32b L37). The first MLP best-layer characterisation for these 4 models.
- **mlp512 ≈ mlp256** (within ~0.01 everywhere) — the extra hidden capacity buys
  almost nothing, consistent with exp-12's finding on the big models.

## Artifacts

Per model under `results/<slug>/{mlp256,mlp512}/`:
- `logits_mlp.npz` (~6–7 MB) — token table: logit (=probe(X)), prob, y,
  example_id, char_start/end, is_test, is_code, layer, head.
- `example_scores_mlp.json`, `metrics_mlp.json`.
Sweep models also: `results/<slug>/sweep_curves.json` — per-layer val/test
tokens_code_auc for both heads (the full layer profile the best-layer came from).

Compute: one multi-GPU run, resumable per-unit (96 extract
shards + 272 sweep cells + 12 dumps). The ~1 TB of all-layer activations were
deleted after the dump (regenerable — re-extraction reproduces exp-12 exactly).
Post-run review: no blocking issues.
