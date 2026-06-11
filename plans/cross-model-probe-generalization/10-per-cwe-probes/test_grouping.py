# [ai-generated]
"""Local unit test for exp-10 CWE-grouping / split logic. No GPU, no cluster,
no cached acts — synthetic dataset rows + the real split helpers from
src/remotes/train_eval.py.

Run:  python plans/cross-model-probe-generalization/10-per-cwe-probes/test_grouping.py

Asserts:
  1. The seed-42 group hold-out + the 15% VAL carve are GROUP-CLEAN: no group
     (pair) straddles fit / val / test — globally AND after filtering to a
     single CWE subset. This is the property that makes the per-CWE head-to-head
     leakage-free.
  2. Filtering to {CWE-X positives ∪ negatives} never moves a token's eid across
     the fit/test boundary (subset ⊆ parent split membership).
  3. cwe != null ⟺ label == 1 invariant the runner relies on holds on the
     synthetic data construction (and is the documented dataset invariant).
"""
from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def _load_train_eval():
    p = REPO / "src" / "remotes" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VAL_FRAC, VAL_SEED = 0.15, 42


def _make_synthetic_rows(n_groups=40):
    """Each group = one vuln/fix pair: a positive (carries a CWE) + a negative
    (cwe=null). Distinct _func_name per group so pair_group_key groups them."""
    cwes = ["CWE-089", "CWE-078", "CWE-125", "CWE-476", "CWE-787"]
    rows = []
    for g in range(n_groups):
        cwe = cwes[g % len(cwes)]
        lang = "python" if cwe in ("CWE-089", "CWE-078") else "c"
        fname = f"file_{g}.py"
        func = f"func_{g}"
        # positive
        rows.append({"code": f"vuln code {g}", "lang": lang, "cwe": cwe,
                     "label": 1, "_file_name": fname, "_func_name": func})
        # negative (same pair group)
        rows.append({"code": f"fixed code {g}", "lang": lang, "cwe": None,
                     "label": 0, "_file_name": fname, "_func_name": func})
    return rows


def main() -> None:
    te = _load_train_eval()
    rows = _make_synthetic_rows()

    # Invariant 3: cwe != null  <=>  label == 1.
    for r in rows:
        assert (r["cwe"] is not None) == (r["label"] == 1), r
    print("[ok] invariant: cwe!=null <=> label==1")

    with tempfile.TemporaryDirectory() as td:
        ds = Path(td) / "dataset.jsonl"
        ds.write_text("\n".join(json.dumps(r) for r in rows))
        split = Path(td) / "split.json"  # not pre-existing -> helper creates it

        all_rows, train_eids, test_eids = te.load_or_make_split(ds, split)
        assert train_eids and test_eids
        assert train_eids.isdisjoint(test_eids)

        # 15% VAL carve, exactly as per_cwe_probe.py does it.
        g_of = {e: te.pair_group_key(all_rows[e]) for e in train_eids}
        groups = sorted(set(g_of.values()))
        rng = np.random.default_rng(VAL_SEED)
        rng.shuffle(groups)
        n_val = max(1, int(round(VAL_FRAC * len(groups))))
        val_groups = set(groups[:n_val])
        val_eids = {e for e, gg in g_of.items() if gg in val_groups}
        fit_eids = train_eids - val_eids

        # Invariant 1 (global): each pair group lands entirely in one of
        # fit / val / test — never split.
        group_to_bucket = {}
        for e in range(len(all_rows)):
            g = te.pair_group_key(all_rows[e])
            bucket = ("test" if e in test_eids else
                      "val" if e in val_eids else
                      "fit" if e in fit_eids else "?")
            assert bucket != "?", e
            if g in group_to_bucket:
                assert group_to_bucket[g] == bucket, (
                    f"group {g} straddles {group_to_bucket[g]} and {bucket}")
            else:
                group_to_bucket[g] = bucket
        print(f"[ok] group-clean: {len(group_to_bucket)} pairs, none straddle "
              f"fit/val/test (fit={len(fit_eids)} val={len(val_eids)} "
              f"test={len(test_eids)})")

        # Invariant 1 + 2 (per-CWE subset): filtering to {CWE-X pos ∪ all neg}
        # preserves each eid's bucket — the subset never moves a token across
        # the fit/test boundary.
        def cwe_of(e):
            return all_rows[e].get("cwe")

        neg_fit = {e for e in fit_eids if not cwe_of(e)}
        neg_test = {e for e in test_eids if not cwe_of(e)}
        assert neg_fit and neg_test, "negative pool must be non-empty"
        n_checked = 0
        for cwe in {cwe_of(e) for e in range(len(all_rows)) if cwe_of(e)}:
            pos_fit = {e for e in fit_eids if cwe_of(e) == cwe}
            pos_test = {e for e in test_eids if cwe_of(e) == cwe}
            spec_fit = pos_fit | neg_fit
            eval_set = pos_test | neg_test
            # Subset membership must agree with the parent split (the core
            # leakage property — a CWE filter never moves an eid's bucket).
            assert spec_fit <= fit_eids, cwe
            assert eval_set <= test_eids, cwe
            # No eid is in both the specialized fit and the eval set.
            assert spec_fit.isdisjoint(eval_set), cwe
            # Scarcity is realistic: a CWE may have 0 test positives. The runner
            # guards this (writes an "error" record); the test mirrors that by
            # only checking subsets where the AUC would be defined.
            if pos_test:
                assert pos_fit, f"{cwe}: test positives but no fit positives"
                n_checked += 1
        assert n_checked > 0, "no CWE had test positives — bad synthetic split"
        print(f"[ok] per-CWE subset preserves split + train/test disjoint "
              f"({n_checked} CWEs had defined eval; scarce CWEs skipped as the "
              f"runner does)")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
