[ai-generated]

# exp-22 — PrimeVul-Paired vs SVEN, cross-dataset transfer (2026-06-08)

> Renumbered from exp-20 (2026-06-09): collided with `20-fn-fp-token-analysis`.

2-model first cut: **Qwen2.5-Coder-7B** (L16) and **gemma-3-12b-it** (L15).
Canonical honest regime (token granularity, X=code-only negatives, is_code-gated
positives, subtractive subset). A probe trained on dataset A is applied to
dataset B's activations — cross-dataset transfer. All numbers on the **C/C++
slice** of SVEN (PrimeVul is C/C++-only; SVEN's Python half is dropped for a
fair comparison).

## Headline — token-code-AUC [g-mean²], operating layer, held-out test

| model | SVEN→SVEN | SVEN→PV | PV→PV | PV→SVEN |
|---|---|---|---|---|
| Qwen-7B | 0.636 [0.36] | **0.548** [0.28] | 0.637 [0.35] | 0.649 [0.39] |
| gemma-12b-it | 0.641 [0.37] | **0.509** [0.26] | 0.589 [0.32] | 0.642 [0.38] |

(`A→B` = trained on A, evaluated on B's held-out test. PV eval set is large and
well-estimated: ~5000 positive code tokens, 419 test pairs. SVEN-C/C++ test is
small: 365–414 positive tokens, 63 pairs — noisier.)

## Findings

1. **Asymmetric transfer — the headline.** The SVEN-trained probe does **not**
   generalize to PrimeVul (SVEN→PV 0.51–0.55 ≪ PV→PV 0.59–0.64; the gap is
   robust — large PV eval set). But the PrimeVul-trained probe transfers to SVEN
   as well as SVEN's own probe (PV→SVEN 0.64–0.65 ≈ SVEN→SVEN 0.64). PrimeVul
   (2525 subtractive pairs ≈ 5.3× SVEN's 478) learns a **more general
   vulnerability direction**; the SVEN probe carries dataset-specific bias. This
   is direct evidence for the framing §6 "detects dataset bias" failure mode —
   and validates PrimeVul as the better/more-transferable trainer.

2. **SVEN's C/C++ slice is weak.** SVEN→SVEN on C/C++ is only ~0.64, vs the
   bilingual SVEN sub→sub ~0.78 (exp-19). SVEN's vulnerability signal is
   concentrated in its **Python/injection** half; on C/C++ memory-safety vulns
   it is much weaker. (Matches the project-log note that SVEN is injection-heavy
   and weak on memory CWEs.)

3. **g-mean² tracks AUC, uniformly low (0.26–0.39).** g-mean ≈ 0.51–0.62 — these
   are mediocre operating points on C/C++; no threshold gives both high TPR and
   TNR. Consistent with the weak C/C++ AUCs.

## pairAcc (example-level, vuln vs safe within test pairs) — subtractive

| model | SVEN→SVEN | SVEN→PV | PV→PV | PV→SVEN |
|---|---|---|---|---|
| Qwen-7B | 0.54 | 0.43 | 0.39 | 0.43 |
| gemma-12b-it | 0.39 | 0.39 | 0.37 | 0.54 |

**All near chance (0.37–0.54).** At the example level (forced choice: does the
vuln function's max code-token score beat its safe pair's), neither dataset's
probe reliably separates vuln from safe on **C/C++** — vs exp-19's bilingual
SVEN pairAcc-sub ≈ 0.76. The SVEN-C/C++ subtractive test is tiny (28 pairs), so
these are noisy; the asymmetric-transfer signal lives in the token-AUC (large PV
eval), not pairAcc. (pairAcc ranks on **logit**; the first run's prob-ranking
saturated to ties and collapsed gemma sven_cpp to 0.00 — fixed.)

## Caveats
- **Single split, no error bars.** exp-19's CV showed fold-std ≈ 0.024 (Qwen) /
  0.03–0.04 (gemma); the SVEN→PV gap (~0.09) exceeds that, but PV→SVEN≈SVEN→SVEN
  is within noise on the small SVEN-C/C++ test.
- **Domain-mix confound (rev-primevul).** SVEN is injection-heavy, PrimeVul is
  memory-safety-heavy. The SVEN→PV drop is partly genuine dataset bias and partly
  injection-trainer→memory-eval domain shift. A shared-CWE/family-stratified
  eval (CWE-125/476/416/787/190 appear in both) would separate the two — not yet
  run.
- **PrimeVul subtractive fraction 0.537** (2525/4704 pairs); SVEN ~0.67. Lower
  fraction but ~5.3× more absolute localizable-vuln pairs.

## Provenance
- Cluster: debug partition, 1 node × 2 GPU, HF extraction (operating layer only),
  CPU probe training. Jobs 2491400 (OOM, NUMA-membind), 2491484 (OOM, train-step
  GPU contention), 2491529 (success), 2491559 (pairAcc fix).
- Two OOM fixes: dropped `numactl --membind` (PrimeVul ~6.6× SVEN exceeds one
  ~120 GB NUMA node); moved probe training to CPU (GPU busy with the other
  model's extraction). difflib char-diff length-guarded (PrimeVul funcs ≤480 KB).
- Harness: `train_cross.py` (synthetic-tested), `build_primevul.py`,
  `run_primevul.sh`/`submit_primevul.sh`/`overnight_primevul.sh`. Acts cached on
  scratch (`KEEP_ACTS=1`). Metrics: `results/metrics_cross_*.json`.
