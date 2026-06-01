[ai-generated]

# HANDOFF — honest tokens_code line of work (2026-06-01)

Master context for picking this up in a fresh session. Read this, then the three
new experiment briefs (08/09/10) and ADR 0003. Companion: `PLAN.md` (older),
`decisions/0003-honest-metric-and-overnight-autonomy.md` (all decisions made).

## Where we are

The cross-model sweep was redone on the **honest `tokens_code` metric** (live-code
tokens only; comments/sigs/imports/whitespace dropped via tree-sitter). All agreed
sweeps (framing-doc 1,2,4,5,6) are **complete** for 8 models. Results + writeups are
**merged to `main`** (2026-06-01) — the whole `honest-tokens-code-sweep` line of work
is now the trunk. Walkthrough notebook payloads (`notebooks/walkthrough/data/*.npz|
jsonl`) are gitignored, regenerate locally; the `.ipynb`/`.py` + small JSON scaffold
are tracked. New work (08/09/10) starts from a clean `main`.

### Headline findings (solid; see 06/07 EXPERIMENT.md for tables)
1. **`tokens_code` does NOT collapse** — ≈ `tokens` (slightly higher) on all 8.
   The mask only drops ~30% of tokens on full-function SVEN (NOT the ~98% the old
   docstring claimed — that was wrong, now fixed), so most "easy negatives" were
   live code all along.
2. **~0.75–0.79** (val_tokens_code-selected) across 1B–32B, base + instruct.
3. **Layer selection must be `val_tokens_code`**, NOT `val_ex_auc` (ex-AUC is
   near-chance here → picks near-random layers). With val_tokens_code (leakage-free
   15% group-aware val split), selection is **near-oracle (gap ≤0.025)**.
4. **Q5: pt ≈ it** → vuln direction is a *pretraining* feature, not installed by
   post-training.
5. **Exp-07 (train-time negative masking): Δ≈0** → probe never leaned on trivial
   negatives.
6. **Sweep-5 (03/04 honest): MLP and α gains are GENUINE on tokens_code** (NOT
   artifacts — corrected 2026-06-01 after review). `tokens_code` ≈ `tokens` in every
   cell (metric swap barely matters), but within-experiment: linear→MLP = +0.035
   (single layer) to **+0.064** (3-layer concat) on Qwen tokens_code; α optimum ~10
   = +0.045 (gemma-27b) / +0.014 (Qwen) over α=1. The linear span-max probe is NOT
   at ceiling — motivates exp-09.
7. **★ Sweep-6 reframes everything:** the ~0.78 aggregate is **Python /
   injection-class only**. By language: python ≈0.81, **C ≈0.59 (near-chance)**.
   By CWE: SQLi 0.92, cmd-inj 0.82, path-traversal 0.78 vs OOB-read 0.56,
   NULL-deref 0.55, **UAF 0.52**. The probe detects injection-style
   (data-flow/taint) vulns, NOT memory-safety. This drives experiments 09 & 10.

### Per-model best layers (val_tokens_code-selected; for 07/09/10 reuse)
| model | best layer | test tokens_code |
|---|---|---|
| gemma-3-1b-it | 25 | 0.769 |
| gemma-3-1b-pt | 12 | 0.750 |
| gemma-3-4b-it | 7 | 0.767 |
| gemma-3-4b-pt | 33 | 0.769 |
| gemma-3-12b-it | 15 | 0.771 |
| gemma-3-12b-pt | 13 | 0.767 |
| gemma-3-27b-it | 19 | 0.770 |
| Qwen2.5-Coder-32B-Instruct | 25 | 0.788 |

Best signal = Qwen2.5-Coder-32B (0.788) → natural primary testbed for 09/10.

## New work (this handoff plans it)
- **08 — latest-Qwen dense:** add `Qwen3.6-27B` + `Qwen3-32B` (dense only; user
  excluded the MoE coder to avoid a confound). 06 honest sweep + per-lang/CWE
  breakdown. Q: does a newer dense Qwen do better, esp. on C/memory-safety?
- **09 — interpretable ensemble of linear probes:** `Linear(d,K)` + {max | logsumexp
  | softmax-gate} → scalar. Beat a single linear probe (esp. on C/memory-safety),
  with small K so each direction is inspectable. (Critique in its EXPERIMENT.md.)
- **10 — per-CWE probes:** train per-CWE (linear/mlp) vs the general probe; targets
  the sweep-6 per-CWE gap. Data-scarcity-aware. (Critique in its EXPERIMENT.md.)

09 & 10 run on **cached acts at the best layers above** — cheap, no re-extraction.

## Cluster infra (how to run)

- **Access:** direct `ssh -o BatchMode=yes clariden 'cmd'` works (account
  `course_00136`, project `lsaie-ss26`). The old `clariden` tmux + `/tmp/ctmux.sh`
  channel is GONE (machine rebooted) — use direct ssh. If ssh fails on `publickey`,
  the cert expired → ask the user to run **`cscs-key sign`** locally (re-signs from a
  stored token for a few hours), then retry. `scp -o BatchMode=yes clariden:… .` works.
- **Scratch:** `~/scratch/probes` (= `/iopsstor/scratch/cscs/course_00136/probes`).
  - `repo/` = git checkout (keep on the working branch; `git -C repo pull`).
  - `data/dataset.jsonl` = SVEN before/after (1430 rows); `data/sven_split_meta.json`
    = seed-42 20% group hold-out. `.old-20260531` = archived truncation dataset.
  - `runs/layersweep_<slug>/acts/` = **cached float32 all-layer activations +
    offsets.npz + y.npy + example_ids.npy + meta.json** for all 8 models (~2 TB
    total; `DONE_EXTRACT` marker). REUSE THESE for 09/10. New models (08) extract.
  - `runs/<other>/` = lossalpha_, richer_, codemask_, breakdown_ outputs.
  - `secrets/hf_token` (gated Gemma/Qwen access); `env.sh` loads it.
- **Run pattern (proven):** per-model single-node job = 1 node × 4 GH200. See
  `06-honest-metric-sweeps/submit_layersweep.sh` (extract→per-layer train→aggregate)
  and `submit_post06.sh` / `submit_breakdown.sh`. **debug-qos = ONE submitted job at
  a time** → drive multiple models with a login-node `nohup` orchestrator that polls
  `squeue` and submits the next (see `run_honest_sweep_orch.sh`). Poll progress from
  the laptop with a `run_in_background` bash loop doing `ssh … 'test -f DONE; ...'`
  (NOT foreground sleep). Examples used this session: `/tmp/poll_*.sh`.
- **env:** `source repo/src/remotes/clariden/env.sh` inside the srun container
  (`--environment=alps3`). transformers 5.9.0 stack in `.python_deps5`. Login node
  has NO python — only inside the srun container. tree-sitter + grammars are in
  `env.sh` DEPS (required for the code mask); if you change DEPS, `rm
  .python_deps5/.deps_ok` to force reinstall.

## Gotchas (hard-won)
- **float32 acts** (Gemma mid-layer massive activations overflow f16).
- **tokens_code needs `offsets.npz` + dataset `code`/`lang`** → `src/eval/
  honest_scoring.py` (`honest_token_aucs`, `build_code_mask`, `load_offsets_npz`,
  `load_dataset_rows`). Mask logic in `src/eval/code_mask.py`. ALWAYS assert
  `dropped_fraction > 0` on a canary (else tree-sitter missing → silent no-op).
- **Memory:** the 3-layer-concat × mlp512 on 27B/32B **OOMs at 4 workers/node** even
  with NUMA interleave. Use `NWORKERS=2` (2 GPUs) for heavy richer/MLP cells on big
  models. (gemma-3-27b exp-04 OOM'd this session — one cell still missing.)
- **Layer selection = val_tokens_code** (see finding 3). Don't reintroduce val_ex_auc.
- **Qwen3.6-27B** has Gated-DeltaNet/linear-attention layers — verify
  `output_hidden_states` exposes every layer before trusting its sweep (ADR 0001 says
  it worked on the inflated run; re-confirm n_layers).
- **Per-layer probe weights are NOT saved** by the sweep — 09/10/breakdown retrain
  the (single) best layer on cached acts (cheap).

## Key code map
- `src/eval/honest_scoring.py` — honest token AUC + masks (USE for all eval).
- `src/eval/code_mask.py` — tree-sitter live-code mask.
- `src/training/train_probe_spanmax.py` — `train_one_layer(..., seed, mask_negatives
  ∈ {none,code_only}, code_mask=None)`. Span-max loss, internal val split.
- `06-honest-metric-sweeps/` — train_all_layers (val_tokens_code), aggregate,
  submit_layersweep, run_honest_sweep_orch, submit_post06, submit_breakdown,
  breakdown_lang_cwe, all metrics + plots scripts.
- `07-train-code-masked-negs/codemask_train.py` — paired none/code_only runner.
- Plots: `data/plots/honest-sweep/*.png` (gitignored). Open with
  `cursor -r <files>` (reuse window; no dir arg).

## Open questions / follow-ups
- Is C/memory-safety weak because the **signal is absent** in activations, or because
  SVEN's C pairs are harder/labelled differently? (10 partly tests this.)
- ~~Merge `honest-tokens-code-sweep` → main before new work?~~ DONE — pushed to main 2026-06-01.
- Sweep-3 (proximity-window) still parked (logic untrusted).
