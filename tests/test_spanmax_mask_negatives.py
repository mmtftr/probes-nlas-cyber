# [ai-generated]
"""Focused tests for the additive `mask_negatives` knob on train_one_layer.

Synthetic, tiny, CPU only.
  - mask_negatives="none" reproduces the prior behavior EXACTLY (regression
    guard): identical (w, b) for the same seed/data.
  - mask_negatives="code_only" with a mask that drops some negatives lowers the
    negative token count fed into the loss accordingly, and never mutates the
    caller's input arrays.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.training import train_probe_spanmax as sm
from src.training.train_probe_spanmax import train_one_layer


def _toy_data(seed: int = 0):
    """3 examples, 4 tokens each (12 tokens), hidden_dim=5. Mixed labels so the
    internal stratified split has both classes."""
    rng = np.random.default_rng(seed)
    n_ex, t_per, hid = 3, 4, 5
    X = rng.standard_normal((n_ex * t_per, hid)).astype(np.float32)
    example_ids = np.repeat(np.arange(n_ex), t_per).astype(np.int64)
    y = np.zeros(n_ex * t_per, dtype=np.int8)
    # Example 0 + 2 positive (one positive token each); example 1 all-negative.
    y[0] = 1
    y[t_per * 2] = 1
    return X, y, example_ids


def test_mask_negatives_none_is_regression_identical():
    X, y, eids = _toy_data()
    r_default = train_one_layer(X, y, eids, epochs=3, device="cpu", verbose=False)
    r_none = train_one_layer(
        X, y, eids, epochs=3, device="cpu", verbose=False, mask_negatives="none"
    )
    assert np.allclose(r_default["w"], r_none["w"]), "none must reproduce default"
    assert r_default["b"] == pytest.approx(r_none["b"])
    assert r_default["tok_auc"] == pytest.approx(r_none["tok_auc"], nan_ok=True) \
        if not np.isnan(r_default["tok_auc"]) else np.isnan(r_none["tok_auc"])


def test_code_only_drops_negative_tokens_from_loss(monkeypatch):
    X, y, eids = _toy_data()
    # Mask: drop half the tokens. A dropped token survives only if it's positive.
    code_mask = np.ones(len(y), dtype=bool)
    # Mark several NEGATIVE tokens as non-live-code (to be excluded).
    drop_idx = [1, 2, 3, 5, 6, 7]  # all negative in _toy_data
    assert all(y[i] == 0 for i in drop_idx)
    code_mask[drop_idx] = False

    # Spy on how many tokens reach the grouping (== the loss token set).
    seen = {}
    orig_group = sm._group_by_example

    def spy_group(Xg, yg, eg):
        seen["n_tokens"] = len(yg)
        seen["n_neg"] = int((yg == 0).sum())
        return orig_group(Xg, yg, eg)

    monkeypatch.setattr(sm, "_group_by_example", spy_group)

    X_in = X.copy()
    y_in = y.copy()
    eids_in = eids.copy()
    cm_in = code_mask.copy()

    train_one_layer(
        X_in, y_in, eids_in, epochs=2, device="cpu", verbose=False,
        mask_negatives="code_only", code_mask=cm_in,
    )

    # Expected kept = code_mask | positive. drop_idx were negatives -> excluded.
    expected_kept = int((code_mask | (y != 0)).sum())
    assert seen["n_tokens"] == expected_kept
    assert seen["n_tokens"] < len(y), "some negatives must have been dropped"
    # The dropped tokens were all negatives.
    assert seen["n_neg"] == expected_kept - int((y != 0).sum())

    # Immutability: caller arrays untouched.
    assert np.array_equal(X_in, X)
    assert np.array_equal(y_in, y)
    assert np.array_equal(eids_in, eids)
    assert np.array_equal(cm_in, code_mask)


def test_code_only_requires_mask():
    X, y, eids = _toy_data()
    with pytest.raises(ValueError, match="requires code_mask"):
        train_one_layer(X, y, eids, epochs=1, device="cpu", verbose=False,
                        mask_negatives="code_only")


def test_invalid_mode_rejected():
    X, y, eids = _toy_data()
    with pytest.raises(ValueError, match="must be 'none' or 'code_only'"):
        train_one_layer(X, y, eids, epochs=1, device="cpu", verbose=False,
                        mask_negatives="garbage")
