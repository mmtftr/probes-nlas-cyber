[ai-generated]

# exp-25 — all-clean-trained per-CWE probes under language/function-matched negatives

## Aim
Does the exp-10/21 memory recovery (diagonal `tokens_code_auc` 0.73/0.77/0.64 for
CWE-125/416/476) survive when the eval negative pool no longer leaks the
C-vs-Python language confound? This decides project claim #3 ("memory signal
exists; under-allocation not absence"). In SVEN, memory CWEs are **100% C/C++**
and the all-clean negative pool is **53% Python** — a pure language indicator
scores ~0.765 on that exact split (verified analytically: 0.53 + 0.47·0.5). So the
all-clean diagonal cannot, on its own, separate "memory-safety signal" from
"language detector".

## Inputs
- **Models / layer:** Qwen2.5-Coder-32B-Instruct (L25), gemma-3-1b-it (L25).
- **Activations (KEPT, reused — NO re-extraction):**
  `$RUNS/percwe_Qwen_Qwen2.5-Coder-32B-Instruct/token_activations/token_activations_layer25.npz` (11.5 GB)
  `$RUNS/percwe_google_gemma-3-1b-it/token_activations/token_activations_layer25.npz` (3.2 GB)
  each with `offsets.npz`. `$RUNS=/data/probes/runs`.
- **Dataset / split:** `$DATA/dataset.jsonl` (1430 rows), `$DATA/sven_split_meta.json`
  (group-clean pair-level, seed-42, 20% held-out). `$DATA=/.../scratch/probes/data`.
- **Reproduction target:** `results/REPRO_TARGET_{qwen32b,gemma1b}.json` (= exp-21
  `transfer_allclean.json`, the all-clean diagonal that headlines claim #3).
- **Probe recipe (parity with exp-10/21):** annotated `token_labels==1` positives,
  span-max `train_one_layer`, train CWE-X vuln ∪ {clean pool}, all tokens (no is_code
  gate at train), 15% group-aware VAL carve (seed-42). Eval = honest "own" recipe:
  positives = ALL code tokens of CWE-Y vuln test examples labelled by `y_tok`, + the
  regime's clean-negative pool; live-code tokens only.

## Lang inventory (SVEN, verified locally)
- Memory CWEs 125/190/416/476/787: **100% C/C++**. Injection 089: 100% Python; 078/022/079: mostly Python.
- Clean pool (label==0, cwe==null): 715 rows = 378 Python (53%) + 337 C/C++ (47%).

## Outputs (land in `$WORK/exp25/<slug>/`, downloaded to `results/`)
- `repro_gate.json` — all-clean diagonal; MUST match REPRO_TARGET ±0.001 (else nothing downstream trusted).
- `deconfound.json` — per-CWE diagonal under 4 negative regimes × 2 probe sets,
  each with a **language-null** column; bootstrap CIs (over examples) for the
  all-clean-trained probes; pooled-memory-probe diagonal.
- `cv/<regime>_<cwe>_<seed>.json` — grouped 5-fold × 3-seed CV cells (resumable).
- `probes_dc.npz` — saved W/b (allclean + conly + pooled).

### Negative regimes (positives held identical; only the appended clean-neg pool changes)
- `allclean` — original mixed pool → reproduction gate.
- `conly` — C/C++-only clean test tokens (language-matched for **memory**).
- `pyonly` — Python-only clean test tokens (language-matched for **injection**).
- `matchedpatch` — the CWE's OWN paired safe-half (patched) code tokens (function + language matched).

Probe sets: **allclean-trained** (held fixed from the gate) and **conly-trained**
(retrained with C-only-clean negatives) + one pooled memory-family probe (125+416+476+787+190).

## Result format
Per model: a table with rows = CWE (family-tagged, n_test_pos, trust flag), columns =
{allclean, conly, pyonly, matchedpatch} each as `auc | lang_null`; CV mean±std for the
trusted CWEs in allclean & conly regimes. Headline metric `tokens_code_auc`.

## Interpretation hints
- Memory diag **drops to ≈ language-null / chance** under conly or matchedpatch → claim #3
  FALLS; honest conclusion = "no demonstrated memory-safety signal above surface
  (language/lexical) confounds" — a clean publishable negative.
- Memory diag **≥0.65 under conly with CI excluding the lang-null** → claim #3 rescued
  and STRENGTHENED → becomes the paper's spine.
- Injection rows are the positive control: expected to stay strong under pyonly (their
  language-matched regime) since the injection signal is lexical/taint, not language.

## For agents — asset paths discovered (2026-06-09)
- All assets present and verified via `fc ls`. probes_allclean.npz + transfer_allclean.json
  already on scratch for qwen32b (the gate target); gemma-1b target downloaded too.
- SLUG = model with "/"→"_". Run dir `$WORK/exp25/<SLUG>/`. GPU0=qwen32b, GPU1=gemma1b.
- env: `source $REPO/src/remotes/the cluster/env.sh` sets DATA/RUNS/PYTHONPATH. backend = hf (CPU/GPU torch only; no vLLM needed — eval on cached acts).
