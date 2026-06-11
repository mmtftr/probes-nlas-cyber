[ai-generated]

# exp-16 logit explorer

Browse SVEN dataset samples with exp-16 span-max linear-probe logits. Per-token
calibrated probe scores are overlaid on the source code; ground-truth vulnerable
spans are underlined. Threshold defaults to a **locally-fit Platt** decision
threshold (not the gemmaforge production number).

## Run

```bash
cd plans/cross-model-probe-generalization/16-token-logit-dump/explorer
uv run python calibrate.py     # one-time: fits Platt per probe -> calibration.json
uv run python server.py        # serves 0.0.0.0:8011
# view on tailnet: http://<this-host-tailscale-ip>:8011/
```

`calibrate.py` only needs re-running if the `logits_layer*.npz` change.

## Controls

- **probe type** — only `Linear (span-max)` was dumped in exp-16 (MLP/others not yet).
- **model variant** — 7 variants (Qwen2.5-Coder 7B/32B, gemma-3 1b/4b/12b-it/12b-pt/27b-it).
- **layer** — the per-layer trained probe (★ = best layer by example-AUC; shows token-code-AUC).
- **calibration** — `token-level (local fit)` (default) · `example-level (local fit)` · `production (gemmaforge ref)`.
- **apply platt** on/off, plus editable **T**, **a**, and a **threshold** slider. `reset to fitted` restores the fitted values for the current probe/level.
- Left: filters (split/label/pred-correctness/lang/cwe) + sort, a live confusion matrix over the filtered set, and the example list. Click a sample to load its code.
- Code panel: token background opacity ∝ calibrated prob; `pos` outline = above threshold; amber underline = ground-truth `evidence` span. Hover a token for logit / σ / calibrated prob.

## For agents

Data (all read-only):
- `../results/logitdump_<MODEL>/metrics_logitdump.json` — per-layer AUCs, best layer.
- `../results/logitdump_<MODEL>/example_scores_layer<NN>.json` — per-example `{eid, score, logit_max, label, cwe, lang, is_test}`. `eid` = line index into `data/dataset.jsonl`.
- `../results/logitdump_<MODEL>/logits_layer<NN>.npz` — per-token `{logit, prob, y, example_id, char_start, char_end, is_test, is_code}` (gitignored heavy file; present locally).
- repo `data/dataset.jsonl` — `code`, `label`, `cwe`, `lang`, `token_labels.evidence` (char spans), `_func_name`, `_file_name`.

Calibration (`calibrate.py`):
- Lin et al. 2007 Platt fit `p = sigmoid((logit - a)/T)` on the held-out (`is_test`) split, at both **token** and **example** level. Threshold = F1-max on the calibrated held-out scores. Writes `calibration.json` keyed `model -> layer -> {token|example}`; `production` key holds the gemmaforge reference `T=1.794, a=-0.269, thr=0.929`.
- **Token-level fits are well-behaved** (brier drops, A<0). **Example-level fits are largely degenerate** — span-max example-AUC is weak (~0.59), so F1-max sits at the predict-all-positive point (F1≈0.668) and several layers fail the monotonicity guard (AUC≤0.5 → `null`). The UI defaults to token-level and auto-falls-back from a `null` example fit with a ⚠ flag.
- Caveat surfaced in-UI: the gemmaforge `production` numbers were fit on a *different* probe; they are reference-only, not a calibration of these probes.

Endpoints: `/api/config`, `/api/models`, `/api/calibration`, `/api/examples?model&layer`, `/api/example?model&layer&eid`. No new deps (stdlib `http.server` + numpy). npz are LRU-cached (4) in the server; recolouring on slider moves is fully client-side from raw logits.
