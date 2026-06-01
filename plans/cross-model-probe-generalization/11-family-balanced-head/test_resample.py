# [ai-generated]
"""Synthetic smoke test for family_balanced_resample — no model load.

Verifies the resampling invariants the SPEC requires:
  (a) inputs are never mutated in place,
  (b) k = min(round(n_inj / n_mem), MAX_OVERSAMPLE_K) computed right,
  (c) duplicated copies get FRESH synthetic eids that collide with neither real
      eids nor each other across copy indices,
  (d) token count grows by exactly (k - 1) * n_memory_tokens, and the duplicated
      tokens/labels are exact copies of the memory-positive blocks.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "fam11", HERE / "family_balanced_probe.py")
fam11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fam11)


def _make_rows():
    """Map eid -> {'cwe': ...}. 6 injection-positive eids, 2 memory-positive
    eids, 2 negatives. n_inj/n_mem = 3 -> expect k=3."""
    rows = {}
    for e in range(6):                       # eids 0..5 injection (CWE-089)
        rows[e] = {"cwe": "CWE-089"}
    for e in (6, 7):                         # eids 6,7 memory (CWE-125)
        rows[e] = {"cwe": "CWE-125"}
    for e in (8, 9):                         # eids 8,9 negatives
        rows[e] = {"cwe": None}
    return rows


def _make_flat_arrays(rows):
    """Build flat per-token X/y/eid arrays: 2 tokens per example, positives have
    a positive token, negatives all-zero. d=4 features."""
    X_blocks, y_blocks, e_blocks = [], [], []
    rng = np.random.default_rng(0)
    for e in sorted(rows):
        Xi = rng.standard_normal((2, 4)).astype(np.float32)
        is_pos = rows[e]["cwe"] is not None
        yi = np.array([1, 0] if is_pos else [0, 0], dtype=np.int64)
        ei = np.array([e, e], dtype=np.int64)
        X_blocks.append(Xi); y_blocks.append(yi); e_blocks.append(ei)
    return (np.concatenate(X_blocks), np.concatenate(y_blocks),
            np.concatenate(e_blocks))


def main() -> None:
    rows = _make_rows()
    X, y, e = _make_flat_arrays(rows)
    X0, y0, e0 = X.copy(), y.copy(), e.copy()  # snapshots for mutation check

    Xb, yb, eb, info = fam11.family_balanced_resample(X, y, e, rows)

    # (b) k right: n_inj=6, n_mem=2 -> round(3.0)=3, min(3,8)=3.
    assert info["n_inj_fit_examples"] == 6, info
    assert info["n_mem_fit_examples"] == 2, info
    assert info["oversample_k"] == 3, info["oversample_k"]

    # (a) inputs untouched.
    assert np.array_equal(X, X0) and np.array_equal(y, y0) and np.array_equal(e, e0), \
        "inputs were mutated"

    # (d) token-count growth = (k-1) * n_memory_tokens. 2 mem eids * 2 tok = 4.
    n_mem_tok = 4
    assert len(eb) == len(e) + (info["oversample_k"] - 1) * n_mem_tok, \
        (len(eb), len(e))
    assert info["n_fit_tokens_before"] == len(e)
    assert info["n_fit_tokens_after"] == len(eb)

    # Original eids preserved as the first block.
    assert np.array_equal(eb[:len(e)], e)

    # (c) synthetic eids fresh: > all real eids, unique per copy, decode back to
    # the original memory eids modulo C.
    C = info["synthetic_eid_offset_C"]
    assert C == max(rows) + 1, (C, max(rows))
    real_eids = set(int(x) for x in np.unique(e))
    synth = eb[len(e):]
    assert set(int(x) for x in synth).isdisjoint(real_eids), "synthetic collides with real"
    # Each synthetic eid maps to an original memory eid: synth = orig + C*copy_idx.
    mem_eids = {6, 7}
    for s in synth:
        copy_idx, orig = divmod(int(s), C)
        assert copy_idx in (1, 2), copy_idx          # k-1 = 2 extra copies
        assert orig in mem_eids, (s, orig)
    # Unique across copy indices: total synthetic eids == (k-1)*n_mem distinct.
    assert len(set(int(x) for x in synth)) == (info["oversample_k"] - 1) * len(mem_eids)

    # Duplicated feature/label blocks are exact copies of the memory tokens.
    mem_tok_mask = np.isin(e, list(mem_eids))
    Xmem, ymem = X[mem_tok_mask], y[mem_tok_mask]
    for copy_idx in range(1, info["oversample_k"]):
        lo = len(e) + (copy_idx - 1) * n_mem_tok
        hi = lo + n_mem_tok
        assert np.array_equal(Xb[lo:hi], Xmem), copy_idx
        assert np.array_equal(yb[lo:hi], ymem), copy_idx

    # k=1 / no-mem edge cases: returns a copy, no duplication.
    rows_no_mem = {e_: {"cwe": "CWE-089"} for e_ in range(4)}
    rows_no_mem.update({4: {"cwe": None}})
    Xn, yn, en = _make_flat_arrays(rows_no_mem)
    Xr, yr, er, info2 = fam11.family_balanced_resample(Xn, yn, en, rows_no_mem)
    assert info2["oversample_k"] == 1, info2
    assert len(er) == len(en)
    assert er is not en and np.array_equal(er, en)   # fresh copy, equal content

    print("smoke test PASSED: k=3, no-mutation, fresh non-colliding synthetic "
          "eids, exact token growth, k=1 edge case ok")


if __name__ == "__main__":
    main()
