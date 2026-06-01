# [ai-generated]
"""Local synthetic smoke test for build_matrix's combine + cell logic — pure
numpy/sklearn, NO model / cached acts / cluster.

Asserts:
  1. combine_scores("max")  == rowwise max over members, per eid.
  2. combine_scores("mean") == rowwise mean, per eid.
  3. combine_scores drops eids no member scores, ignores None/NaN contributions,
     and a member missing an eid does not pull the max down.
  4. _both_scored returns the key intersection (the shared eval example set).
  5. End-to-end AUC sanity on a hand-built example set: for a 'memory' cell with
     a member that scores memory positives high and negatives low, example-AUC
     over (memory-pos ∪ all-neg) is ~1.0; for the 'overall' cell with all
     positives ∪ all negatives the same member (only good on memory) is mediocre.
     This exercises the exact per-cell pos/neg construction build_matrix uses.

Run:  uv run python plans/.../15-ensemble-comparison/test_build_matrix.py
(exits non-zero on any failure; no pytest dependency.)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_matrix import combine_scores, _both_scored, _auc  # noqa: E402


def _check(name: str, cond: bool):
    if not cond:
        raise AssertionError(name)
    print(f"  ok: {name}")


def main() -> None:
    # --- 1/2/3: combine_scores semantics ---
    m1 = {0: 0.1, 1: 0.9, 2: 0.4}
    m2 = {0: 0.5, 1: 0.2, 3: 0.7}          # m2 is missing eid 2, scores eid 3
    m3 = {0: None, 1: float("nan")}        # None/NaN must be ignored
    eids = {0, 1, 2, 3}

    cmax = combine_scores([m1, m2, m3], eids, "max")
    _check("max eid0 = max(0.1,0.5) ignoring None", np.isclose(cmax[0], 0.5))
    _check("max eid1 = max(0.9,0.2) ignoring NaN", np.isclose(cmax[1], 0.9))
    _check("max eid2 = 0.4 (m2 missing it)", np.isclose(cmax[2], 0.4))
    _check("max eid3 = 0.7 (only m2 scores it)", np.isclose(cmax[3], 0.7))

    cmean = combine_scores([m1, m2], eids, "mean")
    _check("mean eid0 = mean(0.1,0.5)", np.isclose(cmean[0], 0.3))
    _check("mean eid2 = 0.4 (only m1)", np.isclose(cmean[2], 0.4))

    # eid with no scoring member is omitted entirely.
    none_map = combine_scores([{0: None}], {0}, "max")
    _check("eid with no valid member is dropped", 0 not in none_map)

    # --- 4: _both_scored ---
    inter = _both_scored({0: 1.0, 1: 1.0, 2: 1.0}, {1: 1.0, 2: 1.0, 5: 1.0})
    _check("_both_scored = key intersection", inter == {1, 2})

    # --- 5: per-cell pos/neg AUC sanity (mirror build_matrix's cell construction) ---
    # eids: 0,1 = memory pos; 2,3 = injection pos; 4,5,6,7 = negatives.
    cwe = {0: "CWE-416", 1: "CWE-125", 2: "CWE-089", 3: "CWE-078",
           4: None, 5: None, 6: None, 7: None}
    family = {"CWE-416": "memory", "CWE-125": "memory",
              "CWE-089": "injection", "CWE-078": "injection"}
    te_eids = set(cwe)

    # A 'memory member' that fires on memory pos, but is BLIND to injection: it
    # ranks injection positives (2,3) BELOW several negatives, so its overall AUC
    # (all-pos ∪ all-neg) is dragged below its memory-cell AUC.
    mem_member = {0: 0.95, 1: 0.90, 2: 0.10, 3: 0.05,
                  4: 0.40, 5: 0.30, 6: 0.20, 7: 0.15}

    def cell_pos_neg(cell, scored):
        neg = {e for e in te_eids if cwe[e] is None and e in scored}
        if cell == "overall":
            pos = {e for e in te_eids if cwe[e] is not None and e in scored}
        else:
            pos = {e for e in te_eids if family.get(cwe[e]) == cell and e in scored}
        return pos, neg

    scored = set(mem_member)
    # memory cell: (memory pos ∪ all neg). The memory member separates perfectly.
    pos, neg = cell_pos_neg("memory", scored)
    _check("memory cell pos = {0,1}", pos == {0, 1})
    _check("memory cell neg = {4,5,6,7}", neg == {4, 5, 6, 7})
    eval_eids = sorted(pos | neg)
    y = np.array([1 if e in pos else 0 for e in eval_eids])
    s = np.array([mem_member[e] for e in eval_eids])
    auc_mem = _auc(y, s)
    _check(f"memory cell AUC ~1.0 (={auc_mem:.2f})", auc_mem > 0.99)

    # overall cell: (all pos ∪ all neg). The memory-only member ranks injection
    # positives (low score) BELOW negatives sometimes -> mediocre overall AUC.
    pos, neg = cell_pos_neg("overall", scored)
    _check("overall cell pos = {0,1,2,3}", pos == {0, 1, 2, 3})
    eval_eids = sorted(pos | neg)
    y = np.array([1 if e in pos else 0 for e in eval_eids])
    s = np.array([mem_member[e] for e in eval_eids])
    auc_ov = _auc(y, s)
    _check(f"overall AUC < memory AUC (memory-only member; ={auc_ov:.2f})",
           auc_ov < auc_mem)

    # cat-ensemble: MAX(memory member, injection member) recovers BOTH families
    # -> overall AUC ~1.0. (This is the load-bearing 'does cat-ensemble recover
    # both families overall' check.)
    # injection member: fires on injection pos, low on memory pos AND negatives.
    # MAX(mem_member, inj_member) then ranks ALL four positives above ALL negs:
    #   eid0 max(0.95,0.10)=0.95  eid1 0.90  eid2 max(0.10,0.92)=0.92  eid3 0.88
    #   negs eid4 max(0.40,0.05)=0.40 ... all < 0.88 -> overall AUC ~1.0.
    inj_member = {0: 0.10, 1: 0.08, 2: 0.92, 3: 0.88,
                  4: 0.05, 5: 0.07, 6: 0.06, 7: 0.03}
    cat = combine_scores([mem_member, inj_member], te_eids, "max")
    pos, neg = cell_pos_neg("overall", set(cat))
    eval_eids = sorted(pos | neg)
    y = np.array([1 if e in pos else 0 for e in eval_eids])
    s = np.array([cat[e] for e in eval_eids])
    auc_cat = _auc(y, s)
    _check(f"cat-ensemble overall AUC ~1.0 (={auc_cat:.2f}) > memory-only overall",
           auc_cat > auc_ov and auc_cat > 0.99)

    # cross-check our _auc against sklearn directly.
    _check("_auc matches sklearn roc_auc_score",
           np.isclose(auc_cat, roc_auc_score(y, s)))

    print("ALL OK")


if __name__ == "__main__":
    main()
