[ai-generated]

# REBUILD PLAN — switch to SVEN before/after dataset, re-run 02–05

Decision: `decisions/0002-dataset-before-after-contrast.md`. Old (flawed)
results: `archive/old-dataset/` (+ its README explains the problem).

**One-line goal:** replace the completion-truncation `dataset.jsonl` with the
SVEN **before/after full-function contrast**, re-extract activations for both
models, and re-run experiments 02–05 on it.

## State at handoff (2026-05-31)

- **Nothing running** on Clariden; queue empty. Old activations at
  `runs/layersweep_{google_gemma-3-27b-it,Qwen_Qwen2.5-Coder-32B-Instruct}/acts`
  are **stale** (old truncated code) — must be re-extracted.
- **Scripts are correct as-is** (extract / train_one_layer / sweeps / submit /
  plot). Only the dataset + activations change. exp-05's transformers-5.9.0
  `apply_chat_template` fix is already committed.
- Result artifacts for 02–05 moved to `archive/old-dataset/`; each
  `EXPERIMENT.md` has a banner pointing there.
- Two models in scope: `google/gemma-3-27b-it`, `Qwen/Qwen2.5-Coder-32B-Instruct`.

## Steps

1. **New builder** → full before/after pairs. None of the 3 existing builders do
   this (all truncate). Adapt `scripts/build_dataset_sven_canonical.py` (it
   already emits the canonical schema + token_labels) so that:
   - positive (label=1) = full `func_src_before`; `token_labels` = the vulnerable
     lines = the diff regions (`char_changes`/`line_changes`) mapped onto
     `func_src_before` (reuse `derive_rich_labels.py` span logic).
   - negative (label=0) = full `func_src_after`; empty `token_labels`, cwe=None.
   - `source` = `SVEN-before` / `SVEN-after`; keep `_file_name`,`_func_name` for
     `pair_group_key`. Drop or repurpose `is_completion_vulnerable` (no longer
     meaningful). Keep the schema `validate_dataset.py` checks.
2. **Validate locally**: `scripts/validate_dataset.py` + `validate_rich_labels.py`.
   Sanity-check: `SVEN-after` rows > 0 and == #positives; code is NOT truncated
   (ends at function boundaries, not mid-identifier); positive vs negative char
   length roughly balanced (the 100%-pos-longer confound is gone); positives have
   non-empty token_labels, negatives empty.
3. **Deploy**: commit + push; on Clariden `git pull`; rebuild
   `~/scratch/probes/data/dataset.jsonl`. The split is by group name
   (`pair_group_key`), so `sven_split_meta.json` should still be valid — verify
   its `heldout_groups` still match (regenerate if row/group set changed).
4. **Re-extract activations** (the long pole, needs model + `--environment=alps3`,
   ~7 min/model). Wipe the stale acts first:
   `rm -rf runs/layersweep_<slug>/acts` then run extraction. Easiest: the
   `02-.../submit_layersweep.sh` phase-1 extracts; or run `extract_all_layers.py`
   directly. Do both models (sequential, debug-qos=1).
5. **Re-run experiments on the fresh acts** (scripts unchanged), sequential via a
   login-node nohup orchestrator (pattern below):
   - **02** layer sweep (`submit_layersweep.sh`) + variance
     (`submit_variance.sh`, 5 seeds) → new per-model best layers + AUC-vs-depth.
   - **03** `submit_loss_alpha.sh` (loss×α) — best layers from NEW 02.
   - **04** `submit_richer.sh` — FEATURESETS from NEW 02 best layers (do NOT
     reuse the old 9,19,26,61 / 34,41,52,63 — re-derive).
   - **05** `submit_verbalized.sh` — `LAYER=` from NEW 02 best layer per model.
     NOTE: now the code is full functions, so the verbalized "is this code
     vulnerable?" framing is no longer asking about a truncated prefix — the
     exp-05 caveat largely goes away; re-evaluate the introspection-gap cleanly.
6. **Write fresh Results** in each `EXPERIMENT.md` (replace the archived banner),
   plot (open with `cursor`), and **compare against `archive/old-dataset/`** to
   see which qualitative findings survive. Commit per experiment.

## Carried-forward gotchas (hard-won)

- **debug-qos allows ONE submitted job** (`QOSMaxSubmitJobPerUserLimit`). Run
  sequentially; use a login-node `nohup` orchestrator that waits on `squeue -j`
  then submits the next (retry past the QOS error). Pattern is in prior
  `run_*_orch.sh` on scratch.
- **CSCS SSH cert expires (~24h).** `Permission denied (publickey)` from
  `ela.cscs.ch` = ask the user to re-sign the key; my background pollers fail
  silently when it lapses.
- **transformers 5.9.0**: `apply_chat_template(...)` returns a `BatchEncoding`,
  not a tensor (exp-05 handles it via `return_dict=True` + `model(**enc)`).
- **float32 activations** (Gemma mid-layer "massive activations" overflow f16).
- Background-agent code that can't run locally (model-dependent paths) → verify
  on the cluster with a tiny canary before trusting; don't trust fake-stub tests.
- Artifacts: open plots with `cursor <path>`, HTML with `open <path>`.
- One node = 4 GH200; `srun -lu --mpi=pmi2 --environment=alps3`; numactl membind
  per GPU; `source $REPO/src/remotes/clariden/env.sh`.

## Done when

Experiments 02–05 have fresh Results on the before/after dataset, each
`EXPERIMENT.md` updated, archived-vs-new findings compared, committed.
