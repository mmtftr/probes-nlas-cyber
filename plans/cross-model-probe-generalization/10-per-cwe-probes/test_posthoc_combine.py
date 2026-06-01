# [ai-generated]
"""Local unit test for posthoc_ensemble's combiner math — pure numpy/torch, NO
GPU / cluster / cached acts.

Asserts:
  1. combine(S, "max")  == rowwise max  -> shape (n_tokens,).
  2. combine(S, "mean") == rowwise mean -> shape (n_tokens,).
  3. combine(S, "learned") returns shape (n_tokens,) and is fit on VAL not TEST:
     the learned combiner trained on a val matrix where specialist column 0 is
     perfectly correlated with the val label recovers that label ordering on a
     test matrix with the SAME structure (separates test classes), proving the
     fit used val labels — and `combine(..., "learned")` never touches test
     labels (its signature has none).
  4. The torch fallback combiner exposes predict_proba(S)[:,1] with the right
     shape and is also fit on val only.
  5. combine raises on a bad combiner name and on the wrong S rank.

Run:  uv run python plans/.../10-per-cwe-probes/test_posthoc_combine.py
(exits non-zero on any failure; no pytest dependency.)
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc_ensemble import (  # noqa: E402
    combine, _fit_learned_combiner, _TorchLogisticCombiner,
)


def _check(name: str, cond: bool):
    if not cond:
        raise AssertionError(name)
    print(f"  ok: {name}")


def main() -> None:
    rng = np.random.default_rng(0)
    n_val, n_test, k = 60, 40, 3

    # --- max / mean shape + value ---
    S = rng.random((n_test, k)).astype(np.float32)
    cmax = combine(S, "max")
    cmean = combine(S, "mean")
    _check("max shape (n,)", cmax.shape == (n_test,))
    _check("mean shape (n,)", cmean.shape == (n_test,))
    _check("max == rowwise max", np.allclose(cmax, S.max(axis=1)))
    _check("mean == rowwise mean", np.allclose(cmean, S.mean(axis=1)))

    # --- learned: build a VAL matrix where column 0 carries the label signal. ---
    y_val = (rng.random(n_val) > 0.5).astype(np.int64)
    S_val = rng.random((n_val, k)).astype(np.float32)
    S_val[:, 0] = 0.15 + 0.7 * y_val + 0.05 * rng.standard_normal(n_val)  # informative

    # TEST matrix with the SAME structure (col 0 informative for a fresh label).
    y_test = (rng.random(n_test) > 0.5).astype(np.int64)
    S_test = rng.random((n_test, k)).astype(np.float32)
    S_test[:, 0] = 0.15 + 0.7 * y_test + 0.05 * rng.standard_normal(n_test)

    clf, backend = _fit_learned_combiner(S_val, y_val, device="cpu")
    _check("learned combiner fit (val 2-class)", clf is not None)
    print(f"  info: learned backend = {backend}")

    learned_test = combine(S_test, "learned", clf=clf)
    _check("learned shape (n,)", learned_test.shape == (n_test,))
    # Fit-on-val recovers the test label ordering -> AUC well above chance. This is
    # the load-bearing leakage check: the combiner only ever saw VAL labels, yet it
    # separates TEST classes because both share the col-0 signal it learned.
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test, learned_test)
    _check(f"learned separates test (auc={auc:.2f} > 0.8, fit on val only)", auc > 0.8)

    # `combine(..., 'learned')` signature carries NO test labels -> cannot leak.
    import inspect
    params = set(inspect.signature(combine).parameters)
    _check("combine() takes no label argument", params == {"S", "how", "clf"})

    # --- torch fallback combiner: shape + val-only fit ---
    tc = _TorchLogisticCombiner(k, device="cpu", epochs=200).fit(S_val, y_val)
    proba = tc.predict_proba(S_test)
    _check("torch fallback predict_proba shape (n,2)", proba.shape == (n_test, 2))
    _check("torch fallback proba rows sum to 1", np.allclose(proba.sum(axis=1), 1.0, atol=1e-5))
    p1 = combine(S_test, "learned", clf=tc)
    _check("torch fallback combine shape (n,)", p1.shape == (n_test,))

    # --- error paths ---
    for bad in ("median", "vote", ""):
        try:
            combine(S, bad)
            raise AssertionError(f"combine should reject how={bad!r}")
        except ValueError:
            pass
    _check("combine rejects unknown combiner names", True)
    try:
        combine(S[:, 0], "max")  # 1-D -> rank error
        raise AssertionError("combine should reject non-2-D S")
    except ValueError:
        pass
    _check("combine rejects non-2-D S", True)
    try:
        combine(S, "learned", clf=None)  # learned needs a clf
        raise AssertionError("combine('learned') should require clf")
    except ValueError:
        pass
    _check("combine('learned') requires a clf", True)

    print("ALL OK")


if __name__ == "__main__":
    main()
