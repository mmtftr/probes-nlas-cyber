# [ai-generated]
"""Unit tests for src/eval/honest_scoring.py — synthetic only, no model.

Builds a tiny Python snippet with an obvious comment + import + a function body,
hand-builds per-char offsets, and checks:
  - dropped_fraction > 0 (the tree-sitter mask is REAL, not the no-op fallback).
    The test fails loudly if tree-sitter is missing so we never silently exercise
    keep-all.
  - comment / import tokens are excluded from the live-code subset.
  - tokens_code_auc differs from tokens_auc on a crafted case where comment
    tokens are easy negatives (low prob) and a body token is a hard positive
    (high prob) — masking changes the AUC.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.honest_scoring import (
    build_code_mask,
    honest_token_aucs,
    load_dataset_rows,
    load_offsets_npz,
)


def _require_tree_sitter():
    """Fail loudly if tree-sitter (Python grammar) is unavailable.

    The whole point of these tests is to exercise the REAL mask. If the grammar
    is missing, `code_only_mask` falls back to keep-all and the assertions below
    would silently pass against a no-op — so we hard-fail instead.
    """
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401
    except ImportError:  # pragma: no cover - environment guard
        pytest.fail(
            "tree-sitter / tree_sitter_python not installed; honest_scoring "
            "tests require the real mask. Install: uv pip install tree-sitter "
            "tree-sitter-python tree-sitter-c tree-sitter-cpp"
        )


# A snippet whose first three lines are trivially-droppable (comment, import,
# import-from) and whose body lines are live code.
CODE = (
    "# top comment\n"        # line 0: comment           -> dropped
    "import os\n"            # line 1: import            -> dropped
    "from a import b\n"      # line 2: import_from       -> dropped
    "def f(x):\n"            # line 3: signature         -> dropped (def head)
    "    y = x + 1\n"        # line 4: body              -> KEPT
    "    return y\n"         # line 5: body              -> KEPT
)


def _per_char_offsets(code: str) -> np.ndarray:
    return np.array([(i, i + 1) for i in range(len(code))], dtype=np.int32)


def test_dropped_fraction_real_and_excludes_comment_import():
    _require_tree_sitter()
    offsets = _per_char_offsets(CODE)
    eid = 7
    offsets_by_eid = {eid: offsets}
    rows_by_eid = {eid: {"code": CODE, "lang": "python"}}

    token_eids = np.full(len(CODE), eid, dtype=np.int32)
    mask = build_code_mask(token_eids, offsets_by_eid, rows_by_eid)

    dropped_fraction = 1.0 - mask.mean()
    assert dropped_fraction > 0.0, "mask must be real (tree-sitter active)"

    kept = "".join(CODE[i] for i, keep in enumerate(mask) if keep)
    # Comment / import / signature chars are gone.
    assert "comment" not in kept
    assert "import" not in kept
    assert "def" not in kept and "f(x)" not in kept
    # Body survives.
    assert "y=x+1" in kept.replace(" ", "")
    assert "returny" in kept.replace(" ", "")


def test_tokens_code_auc_differs_from_tokens_auc():
    """Craft easy-negative comment tokens vs a hard-positive body token.

    tok_y: only one body char ('y' in the body) is positive.
    tok_p: comment/import chars get very low prob (easy negatives that inflate
           the all-token AUC); the positive body char gets a HIGH prob; the
           remaining body chars (live-code negatives) get a MEDIUM-HIGH prob so
           that, restricted to live code, the positive no longer dominates and
           the AUC drops below the all-token AUC.
    """
    _require_tree_sitter()
    offsets = _per_char_offsets(CODE)
    eid = 3
    offsets_by_eid = {eid: offsets}
    rows_by_eid = {eid: {"code": CODE, "lang": "python"}}
    token_eids = np.full(len(CODE), eid, dtype=np.int32)

    mask = build_code_mask(token_eids, offsets_by_eid, rows_by_eid)
    assert (1.0 - mask.mean()) > 0.0

    # Pick a positive body token: the 'y' that is assigned in line 4.
    pos_idx = CODE.index("    y = x + 1") + 4  # the 'y' char
    assert CODE[pos_idx] == "y"
    assert mask[pos_idx], "chosen positive must be a live-code token"

    n = len(CODE)
    tok_y = np.zeros(n, dtype=np.int8)
    tok_y[pos_idx] = 1

    # Build probs:
    #  - dropped (non-code) tokens: very low prob -> easy negatives.
    #  - live-code negatives: medium-high prob -> hard negatives.
    #  - the positive: highest prob.
    tok_p = np.where(mask, 0.6, 0.01).astype(float)
    tok_p[pos_idx] = 0.95

    out = honest_token_aucs(tok_p, tok_y, token_eids, offsets_by_eid, rows_by_eid)

    assert out["dropped_fraction"] > 0.0
    assert out["n_total_code"] < n
    assert out["n_pos_code"] == 1
    # All-token AUC: positive (0.95) beats every easy negative (0.01) AND every
    # live-code negative (0.6) -> perfect separation -> 1.0.
    assert out["tokens_auc"] == pytest.approx(1.0)
    # Code-only AUC: positive (0.95) vs live-code negatives (0.6) -> still 1.0
    # here because 0.95 > 0.6, so to truly DIFFER we tie one live-code negative
    # ABOVE the positive. Do that and recompute.
    # Find a live-code negative position and push it above the positive.
    code_neg_positions = np.nonzero(mask & (tok_y == 0))[0]
    assert code_neg_positions.size > 0
    tok_p2 = tok_p.copy()
    tok_p2[code_neg_positions[0]] = 0.99  # a live-code negative now outranks pos
    out2 = honest_token_aucs(tok_p2, tok_y, token_eids, offsets_by_eid, rows_by_eid)

    # All tokens: the easy negatives (0.01) still drag the average separation up
    # vs the single inversion among live code.
    assert out2["tokens_code_auc"] < out2["tokens_auc"], (
        f"masking must change the AUC: code={out2['tokens_code_auc']} "
        f"all={out2['tokens_auc']}"
    )


def test_loaders_roundtrip(tmp_path):
    """load_offsets_npz / load_dataset_rows parse the on-disk acts layout."""
    # offsets.npz with offsets_row_NNNN keys.
    off_path = tmp_path / "offsets.npz"
    o0 = np.array([(0, 1), (1, 2)], dtype=np.int32)
    o1 = np.array([(0, 3)], dtype=np.int32)
    np.savez(off_path, offsets_row_0000=o0, offsets_row_0001=o1)
    offs = load_offsets_npz(off_path)
    assert set(offs.keys()) == {0, 1}
    assert offs[0].shape == (2, 2)
    assert offs[1].shape == (1, 2)

    # dataset.jsonl
    ds_path = tmp_path / "dataset.jsonl"
    ds_path.write_text(
        '{"code": "x = 1\\n", "lang": "python"}\n'
        '{"code": "int y;\\n", "lang": "c"}\n'
    )
    rows = load_dataset_rows(ds_path)
    assert rows[0]["lang"] == "python"
    assert rows[1]["lang"] == "c"
    assert rows[0]["code"] == "x = 1\n"


def test_missing_metadata_keeps_all():
    """An eid absent from the offsets/rows maps falls back to keep-all."""
    token_eids = np.array([5, 5, 9, 9], dtype=np.int32)
    # Only eid 5 has metadata; eid 9 missing -> kept.
    offsets_by_eid = {5: np.array([(0, 4), (4, 8)], dtype=np.int32)}
    rows_by_eid = {5: {"code": "x = 1\n", "lang": "python"}}
    mask = build_code_mask(token_eids, offsets_by_eid, rows_by_eid)
    assert mask[2] and mask[3], "missing-eid tokens must be kept"


def test_single_class_returns_nan():
    eid = 1
    offsets = _per_char_offsets("x = 1\n")
    offsets_by_eid = {eid: offsets}
    rows_by_eid = {eid: {"code": "x = 1\n", "lang": "python"}}
    token_eids = np.full(len("x = 1\n"), eid, dtype=np.int32)
    tok_y = np.zeros(len("x = 1\n"), dtype=np.int8)  # all negative
    tok_p = np.linspace(0.1, 0.9, len("x = 1\n"))
    out = honest_token_aucs(tok_p, tok_y, token_eids, offsets_by_eid, rows_by_eid)
    assert np.isnan(out["tokens_auc"])
    assert np.isnan(out["tokens_code_auc"])
