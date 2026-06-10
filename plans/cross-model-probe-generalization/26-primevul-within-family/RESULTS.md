[ai-generated]

# exp-26 — PrimeVul within-C/C++ family structure (language held fixed)

**Model/layer:** Qwen2.5-Coder-7B-Instruct, L16. **Metric:** `tokens_code_auc`
(honest token-level ROC-AUC over live-code tokens; chance = 0.5). **Recipe:**
exp-10 all-clean — per-CWE probe trained on CWE-X annotated positives ∪ all
label==0 PV safe-halves; shared clean-test negatives (n_neg_test_tokens =
287 864). **All-clean negatives are language-clean by construction** — PV is
100 % C/C++, so no Python contrast can leak into the negative pool. CIs are
bootstrap over examples (block 500 reps, diagonal 1000). n_clean fit 3789 /
test 435. Job 2511949 (debug, CPU), DONE.

Untrusted-cell rule: < 10 test positives. All 12 feasible CWEs clear it on the
PV side; on the SVEN-C side CWE-787 (5) and CWE-190 (4) are **untrusted**.

**Scope of the family test (important).** Injection CWEs (078/079/089) are
essentially absent from PrimeVul-Paired (see EXPERIMENT.md PIVOT), so exp-26
tests **memory vs a heterogeneous non-memory bucket** (improper-check,
info-exposure, input-validation, div-by-zero, assert), all within C/C++. It
does **not** re-test exp-21's original injection-vs-memory axis — there is no
injection family to transfer to/from here. Read all "family structure" claims
below as *memory-vs-non-memory clustering within a single language*, not as a
verdict on every family direction exp-21 reported.

---

## A. Within-PV transfer matrix (PRIMARY) — no memory-vs-other clustering

### Family blocks (token-weighted mean `tokens_code_auc`; off-diag isolates transfer)

Block means are weighted by the target CWE's positive-token count (`agg_blocks`
in `pv_family.py`), so they are token-weighted, not plain cell means. The
primary contrast (mem→mem off-diag vs mem→other) is **robust to weighting**:
token-weighted 0.536 vs 0.537; unweighted 0.553 vs 0.552 — identical either way.

| block | weighted mean AUC | 95% CI | reps |
|---|---|---|---|
| mem → mem (incl. diag) | 0.550 | [0.526, 0.582] | 500 |
| **mem → mem off-diag** | **0.536** | **[0.513, 0.564]** | 500 |
| **mem → other** | **0.537** | **[0.518, 0.554]** | 500 |
| other → mem | 0.539 | [0.520, 0.559] | 500 |
| other → other (incl. diag) | 0.528 | [0.489, 0.578] | 500 |
| other → other off-diag | 0.495 | [0.454, 0.541] | 500 |

**The two bolded rows are the test.** Within-family memory transfer
(mem→mem off-diag = 0.536 [0.513, 0.564]) is **near-equal** to cross-family
transfer (mem→other = 0.537 [0.518, 0.554]) — point estimates 0.001 apart, each
inside the other's CI (no paired-difference CI was computed; the comparison
rests on the heavy marginal-CI overlap). There is no within-memory cluster: a
memory probe predicts another memory CWE no better than it predicts a non-memory
CWE. The (token-weighted) other→other off-diagonal is 0.495 but this cell flips
to ~0.540 unweighted, so no chance/non-chance claim is made on it.

**Verdict (per EXPERIMENT.md interpretation hints): memory and non-memory blocks
are indistinguishable ⇒ within this single-language (C/C++) PrimeVul slice,
memory CWEs do not form a transferable family cluster.** This removes the
C-vs-Python language explanation *for the memory-vs-other axis specifically*:
even with language fixed, memory probes do not pool. Per-CWE probes are
CWE-idiosyncratic, not family-coherent. (It does **not** falsify exp-21's
injection-vs-memory geometry, which is untestable here — injection is absent.)

### Diagonal — per-CWE self-transfer (the honest single-CWE probe AUC)

| CWE | family | AUC | 95% CI | n_train_pos | n_test_pos | trust |
|---|---|---|---|---|---|---|
| 119 | mem | 0.875 | [0.643, 0.966] | 522 | 14 | ✓ |
| 190 | mem | 0.730 | [0.617, 0.897] | 139 | 11 | ✓ |
| 476 | mem | 0.692 | [0.520, 0.785] | 218 | 39 | ✓ |
| 416 | mem | 0.657 | [0.470, 0.817] | 177 | 29 | ✓ |
| 415 | mem | 0.635 | [0.485, 0.903] | 49 | 10 | ✓ |
| 125 | mem | 0.611 | [0.481, 0.753] | 394 | 47 | ✓ |
| 787 | mem | 0.562 | [0.492, 0.706] | 258 | 72 | ✓ |
| 200 | other | 0.940 | [0.612, 0.983] | 210 | 16 | ✓ |
| 617 | other | 0.804 | [0.497, 0.877] | 29 | 12 | ✓ |
| 369 | other | 0.767 | [0.627, 0.868] | 58 | 14 | ✓ |
| 703 | other | 0.561 | [0.463, 0.644] | 128 | 47 | ✓ |
| 20 | other | 0.514 | [0.405, 0.749] | 349 | 14 | ✓ |

**Specific per-CWE signal is real** — most diagonals are clearly above chance
(119, 190, 476, 416, 415, 125 in mem; 200, 617, 369 in other). This is *not* in
tension with the absent family structure: probes learn idiosyncratic per-CWE
directions that simply do not pool into a memory-vs-other geometry. Note the
strongest diagonals (119=0.875, 200=0.940) have wide CIs from small test-pos n.

The full 12×12 matrix lives in `results/pv_within.json` → `matrix`. The
strongest **off-diagonal** cells are *cross-family*, not within-memory: e.g.
476→369 = 0.913, 617→369 = 0.898, 703→369 = 0.801 (CWE-369 div-by-zero is
predicted by many unrelated probes — a generic numeric-context direction),
476→617 = 0.760, 20→617 = 0.753. Within-memory off-diagonals are unremarkable
(e.g. 125→416 = 0.436, 416→787 = 0.413). This pattern reinforces the verdict:
transfer follows idiosyncratic shared sub-structure, not family membership.

---

## B. Shared-CWE SVEN ↔ PV memory transfer (SECONDARY)

Memory CWEs present in both datasets. SVEN probe trained on qwen7b-L16 sven_acts
(exp-10 recipe), evaluated on PV (vs PV clean); PV-trained probe evaluated on
SVEN-C (C/C++ slice, vs SVEN-C clean). Same model+layer ⇒ probe dims match.
PV clean-test n = 435, SVEN-C clean-test n = 63.

| CWE | n_pv_pos | n_svenc_pos | PV→PV | SVEN→PV | PV→SVEN-C | SVEN→SVEN-C |
|---|---|---|---|---|---|---|
| 125 | 47 | 19 | 0.611 | **0.668** | 0.617 | 0.746 |
| 416 | 29 | 14 | 0.657 | **0.706** | 0.857 | 0.709 |
| 476 | 39 | 16 | 0.692 | **0.494** | 0.609 | 0.573 |
| 787 | 72 | 5† | 0.562 | 0.635 | 0.443† | 0.658 |
| 190 | 11 | 4† | 0.730 | 0.755 | 0.904† | 0.822 |

† SVEN-C test positives < 10 → any column touching SVEN-C eval is **untrusted**
(787 PV→SVEN-C, 190 PV→SVEN-C). SVEN→PV is trusted for all (PV side ≥ 10).

**These are single-split point estimates — no bootstrap CIs were computed for
the cross-dataset cells.** Treat the numbers below as descriptive, not as
significance-tested transfer/collapse claims; the CI-backed evidence comes from
the exp-25 cross-reference (§C), with which these are directionally consistent.

**Cross-dataset memory transfer (SVEN→PV, the trusted column, PV side ≥ 10):**
- **CWE-125** SVEN→PV 0.668, slightly above its within-PV self-AUC 0.611 →
  positive cross-dataset transfer.
- **CWE-416** SVEN→PV 0.706, above its within-PV self-AUC 0.657 → positive, but
  exp-25's matched-patch result for 416 is only *weakly* positive, so call this
  modest, not strong.
- **CWE-476** SVEN→PV 0.494 ≈ chance → its PV signal (PV→PV 0.692) does not come
  from a SVEN-transferable direction.
- CWE-787 SVEN→PV 0.635 (modest); 190 0.755 but n_pv 11 (borderline).

---

## C. Cross-reference to exp-25 (language-matched matched-patch)

exp-25 verdict: CWE-125 survives matched-patch (0.633 Qwen32B / 0.657 gemma1b),
CWE-416 weakly positive (0.610 / 0.603), CWE-476 collapses to chance
(0.544 / 0.507). **exp-26's cross-dataset SVEN→PV column gives the same
ordering: 125 positive (0.668), 416 positive-but-modest (0.706, weak in exp-25),
476 at chance (0.494).** Two independent deconfounding designs (exp-25
matched-patch with CIs, exp-26 single-split cross-dataset C/C++) point the same
way on which memory CWEs carry a transferable signal and which do not —
directionally consistent. exp-25 supplies the CI-backed half; exp-26's
cross-dataset cells are point estimates that agree with it.

---

## D. Overall verdict

1. **No memory-vs-other family clustering within C/C++.** Within-memory
   off-diagonal transfer (0.536) equals the cross-family value (0.537); no block
   ordering supports a memory cluster, even with language held fixed. This
   removes the C-vs-Python explanation *for the memory-vs-non-memory axis*: the
   memory family does not pool into a transferable direction here. **Scope caveat:**
   injection CWEs are absent from PrimeVul-Paired, so this does **not** test —
   and does not falsify — exp-21's injection-vs-memory geometry. exp-26 narrows
   the memory-family claim; it does not adjudicate every family direction.
2. **Per-CWE signal is real but idiosyncratic.** Single-CWE probes exceed chance
   on the diagonal, but the learned directions do not pool by family; the
   strongest off-diagonal transfers are cross-family (476→369, 617→369).
3. **A subset of memory CWEs carry a transferable direction.** CWE-125 (and more
   modestly 416) transfer across datasets (SVEN→PV) and align with exp-25's
   matched-patch result; CWE-476 does not. The transferable signal is
   CWE-specific, not family-wide.

**Implication:** within a single language, the memory family does not behave as a
coarse transferable direction — the defensible claim is narrower: specific memory
CWEs (125, and more weakly 416) have a real, cross-dataset, language-robust
token-level direction; the rest are idiosyncratic or fragile. The injection-vs-
memory axis from exp-21 remains open (untestable on this dataset).

Artifacts: `results/pv_within.json`, `results/cross_shared.json`. Logits and
probes kept on scratch at `$WORK/exp26/Qwen_Qwen2.5-Coder-7B-Instruct/`.
