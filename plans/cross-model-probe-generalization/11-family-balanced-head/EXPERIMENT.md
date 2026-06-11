[ai-generated]

# 11 — Family-balanced general probe (Tier-4 #7)

The standard general probe under-allocates capacity to the memory family: on
honest `tokens_code` it scores ~0.88 on injection but ~chance on memory (exp-10
confirmed the per-CWE memory gap; exp-09 supplies the linear floor). The fit set
is injection-dominated, so a single linear direction that maximizes overall
token AUC effectively ignores memory. This runner tests the **cheapest** fix:
rebalance the FIT set so memory-family positives are not drowned out, train ONE
probe, and re-measure per-family AUC head-to-head against the unbalanced general
probe on the IDENTICAL test pools. Runs on CACHED acts at each model's best
layer — no re-extraction.

## 1. Aim

Can a SINGLE general probe trained with **family-balanced sampling** hold BOTH
vuln families at once — lifting memory off chance without sacrificing injection
— vs the standard general probe that under-allocates to memory? This is a
**RULE-OUT**: the lead expects at most a minor improvement or even a slight
decrease. Correctness and an apples-to-apples comparison matter more than
winning.

## 2. Inputs

- **Cached acts (no extraction)** at each model's best layer:
  `runs/layersweep_<slug>/acts/layer_NN.npy` (+ `offsets.npz`, `y.npy`,
  `example_ids.npy`). The 4 model:layer pairs (val-`tokens_code`-selected,
  IDENTICAL to exp-09's baselines):
  - `Qwen/Qwen2.5-Coder-32B-Instruct` : layer **25**
  - `google/gemma-3-27b-it` : layer **19**
  - `Qwen/Qwen3-32B` : layer **27**
  - `Qwen/Qwen3.6-27B` : layer **30**
- **Data/recipe:** `data/dataset.jsonl` (SVEN before/after), `sven_split_meta.json`
  (seed-42 20% group hold-out). Span-max probe, linear head, honest `tokens_code`
  metric. EXACT group-aware test split + 15% VAL carve (VAL_SEED=42) from exp-09 —
  the ONLY divergence is the fit-set sampler.
- **CWE→family map:** VERBATIM from `10-per-cwe-probes/per_cwe_probe.py` (injection:
  CWE-089/078/022/079/190; memory: CWE-125/476/416/787) so this is exactly
  comparable to exp-10. `cwe != null ⟺ label == 1`; negatives = `cwe == null`.
- **Sampler:**
  - `none` — plain general fit; MUST reproduce the exp-09 linear baseline
    EXACTLY (Qwen2.5-Coder 0.788, Qwen3-32B 0.806, Qwen3.6-27B 0.787,
    gemma-3-27b 0.770). This is the harness sanity check.
  - `family_balanced` — oversample memory-family POSITIVE examples by integer
    factor `k = min(round(n_inj/n_mem), 8)`, duplicating each memory-positive
    eid's token-block `(k-1)` extra times under FRESH synthetic eids (FIT only;
    never VAL/TEST).

## 3. Outputs

Per model × sampler, one JSON:
`runs/family_balanced_11_<slug>_linear_<sampler>.json`. Each carries
`overall.tokens_code_auc`, `by_lang`, `by_cwe` (IDENTICAL computation to exp-09 —
apples-to-apples), and the NEW `by_family.{memory,injection}` block (pooling
mirrors exp-10: family pos pool ∪ ALL cwe==null test negatives), plus `sampler`,
`oversample_k`, and `sampler_info` (family fit-example counts, k, token counts
before/after).

## 4. Result format

Per-family head-to-head, `none` vs `family_balanced`, per model:

```
model                       sampler          overall  memory   injection  k
Qwen2.5-Coder-32B  L25       none              0.788   0.5xx    0.88x      1
Qwen2.5-Coder-32B  L25       family_balanced   0.7xx   0.x      0.8x       N
gemma-3-27b-it     L19       none              0.770   ...      ...        1
gemma-3-27b-it     L19       family_balanced   0.7xx   ...      ...        N
...
```

The headline is the per-family **memory** and **injection** `tokens_code_auc`
delta (`family_balanced` − `none`) per model, and whether memory moves off
chance without injection collapsing.

## 5. Interpretation hints

- **memory ↑ meaningfully (e.g. >0.6) AND injection ≈ unchanged** → the memory
  signal exists in activations and the unbalanced general probe was merely
  under-allocating capacity; cheap rebalancing recovers it. (Would partly
  CONTRADICT the lead's prior; escalate to the multi-task head below to push
  further.)
- **memory roughly flat (≈ chance) and/or injection drops** → balancing the fit
  set does NOT recover memory (consistent with the lead's prior). Either the
  signal is largely absent at a single linear layer (matching exp-10's per-CWE
  memory failures) or a single linear direction cannot serve both families at
  once. RULE-OUT confirmed; next step is the multi-task head.
- **`none` does NOT reproduce the exp-09 floor** → the harness diverged from
  exp-09 somewhere; STOP and fix before trusting the `family_balanced` column.
- **Data-scarcity caveat:** only ~54 memory test positives (CI half-width ~±0.1
  on AUC). The per-family **memory aggregate** is the trustworthy number — NOT
  individual memory CWEs in `by_cwe` (several have <10 test positives, flagged by
  `trust`). Read `by_family.memory.trust` before drawing conclusions.

## For agents

### Validation status

- `family_balanced_probe.py` passes `ast.parse` (syntax OK).
- Synthetic-array smoke test of `family_balanced_resample` (no model load):
  verifies (a) inputs not mutated, (b) `k = min(round(n_inj/n_mem), 8)` computed
  right, (c) duplicated copies get fresh synthetic eids that never collide with
  reals or with each other, (d) token-count grows by exactly
  `(k-1) × n_memory_tokens`.

### Sampler invariants (read before changing)

- `none` is byte-for-byte the exp-09 fit (`Xfit, yfit, efit` passed straight
  through). If it stops matching the 09 floor, the bug is here, not in the
  balancing.
- Synthetic eids = `orig_eid + C·copy_idx`, `C = max_observed_eid + 1`,
  `copy_idx ∈ [1, k-1]`. They are guaranteed > every real eid and unique across
  copy indices, so `_group_by_example` / `honest_token_aucs` treat each copy as
  a distinct example. Duplicates ONLY enter FIT — VAL/TEST are untouched, so no
  leakage and per-family/by_cwe test pools are unaffected.
- Family counts are over distinct EXAMPLES (eids), not tokens.

### TODO(adhoc-decision) markers left

1. **CWE-190 → injection** (in `FAMILY`, carried verbatim from exp-10). A
   judgement call (integer-overflow is C, often a memory-safety precursor).
   Change in BOTH exp-10 and exp-11 if the lead re-decides.
2. **Sampler strategy fork** (`--sampler` help): oversample-positives vs full
   class-balance vs token-weight. Defaulted to **memory-positive oversampling**
   (least destructive to injection — it adds memory mass without removing
   injection mass). `train_one_layer` has NO per-example sample-weight arg, so
   token-weighting would need a trainer change; resampling is the drop-in path.
3. **Multi-task per-family-head variant** (planned escalation, NOT implemented):
   if family-balanced sampling fails to lift memory, the next step is a 2-logit
   head (one logit per family) with a per-family loss. That needs a custom probe
   factory + custom loss — NOT a drop-in into `train_one_layer` — so it is
   deliberately deferred. TODO(adhoc-decision): the lead confirms whether to
   build it after seeing the balanced-sampling result.
