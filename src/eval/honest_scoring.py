# [ai-generated]
"""Honest token-level AUC: the `tokens` (inflated) vs `tokens_code` (live-code-only)
contrast, computed from flat per-token arrays + the means to build the mask.

This mirrors `src/eval/token_protocol.py`'s `tokens` vs `tokens_code` levels but
operates on the *flat, concatenated* per-token arrays the per-layer sweeps
produce (one entry per token, tagged with its example id / eid). The motivating
problem is the same as in `code_mask.py`: ~98% of tokens are trivially negative
(comments, signatures, imports, whitespace), so a bare token AUC over ALL tokens
inflates. Restricting the AUC to live-code tokens removes that confound.

`honest_token_aucs` is the one entry point the sweeps call; `load_offsets_npz`
and `load_dataset_rows` are loaders for the on-disk acts layout (see
`02-.../extract_all_layers.py`):
  - `offsets.npz` : keys `offsets_row_{i:04d}` -> (T_row, 2) int char offsets.
  - `dataset.jsonl`: one JSON row per line, row index == eid, each row has
    `code` (str) and `lang` (str).

Robustness: if tree-sitter (or a grammar) is unavailable, `code_only_mask`
returns an all-True mask, so `dropped_fraction == 0` and `tokens_code_auc ==
tokens_auc`. We never crash on that — callers that *require* a real mask assert
`dropped_fraction > 0` themselves.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .code_mask import code_only_mask


def load_offsets_npz(path: str | Path) -> dict[int, np.ndarray]:
    """Load `offsets.npz` into {eid -> (T_row, 2) int array}.

    Keys are `offsets_row_{i:04d}`; the integer suffix is the eid (dataset row
    index). Raises FileNotFoundError if the path is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"offsets npz not found: {path}")
    npz = np.load(path)
    prefix = "offsets_row_"
    out: dict[int, np.ndarray] = {}
    for k in npz.files:
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix):]
        try:
            eid = int(suffix)
        except ValueError:
            # Unexpected key shape — skip rather than crash the whole load.
            continue
        out[eid] = np.asarray(npz[k])
    if not out:
        raise ValueError(
            f"no `{prefix}NNNN` keys in {path}; got {list(npz.files)[:5]}..."
        )
    return out


def load_dataset_rows(path: str | Path) -> dict[int, dict]:
    """Load `dataset.jsonl` into {line_index -> {"code", "lang"}}.

    Line index == eid. Only `code` and `lang` are retained (the mask needs
    nothing else). Missing fields default to "" so the mask falls back to
    keep-all for that row rather than crashing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset jsonl not found: {path}")
    rows: dict[int, dict] = {}
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows[i] = {
                "code": obj.get("code", "") or "",
                "lang": obj.get("lang", "") or "",
            }
    if not rows:
        raise ValueError(f"no rows parsed from {path}")
    return rows


def build_code_mask(
    token_eids: np.ndarray,
    offsets_by_eid: Mapping[int, np.ndarray],
    dataset_rows_by_eid: Mapping[int, dict],
) -> np.ndarray:
    """Assemble a global boolean live-code mask aligned to `token_eids`.

    `token_eids[i]` is the eid that the i-th flat token belongs to. The tokens
    for one eid are assumed to appear as a contiguous run in the SAME order as
    that eid's `offsets_row` array (this is how the extractor concatenates).

    For each eid we compute `code_only_mask(code, lang, offsets_row)` once and
    scatter it back to the token positions of that eid. If the per-eid token
    count doesn't match its offsets length, or the eid is missing from either
    map, we keep-all for that eid (conservative — never silently misalign).

    Immutability: builds and returns a fresh array; inputs are not mutated.
    """
    token_eids = np.asarray(token_eids)
    n = token_eids.shape[0]
    mask = np.ones(n, dtype=bool)
    if n == 0:
        return mask

    for eid in np.unique(token_eids):
        eid_int = int(eid)
        positions = np.nonzero(token_eids == eid)[0]
        offs = offsets_by_eid.get(eid_int)
        row = dataset_rows_by_eid.get(eid_int)
        if offs is None or row is None:
            # Missing metadata — keep-all for this eid.
            continue
        offs = np.asarray(offs)
        if offs.shape[0] != positions.shape[0]:
            # Length mismatch — keep-all to avoid scattering a misaligned mask.
            continue
        tok_offsets = [(int(s), int(e)) for s, e in offs]
        eid_mask = code_only_mask(row["code"], row["lang"], tok_offsets)
        # `code_only_mask` returns one bool per offset, in offset order, which
        # matches the position order for this eid.
        mask[positions] = eid_mask.astype(bool)
    return mask


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    """roc_auc_score, or NaN if the label subset is single-class / empty."""
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def honest_token_aucs(
    tok_p: np.ndarray,
    tok_y: np.ndarray,
    token_eids: np.ndarray,
    offsets_by_eid: Mapping[int, np.ndarray],
    dataset_rows_by_eid: Mapping[int, dict],
) -> dict:
    """Return the honest token-AUC contrast for a flat set of tokens.

    Args:
        tok_p, tok_y, token_eids: 1-D arrays, one entry per token, aligned and
            in the concatenated order. `tok_y` is the per-token 0/1 label.
        offsets_by_eid: eid -> (T_row, 2) char offsets (from `load_offsets_npz`).
        dataset_rows_by_eid: eid -> {"code", "lang"} (from `load_dataset_rows`).

    Returns dict with:
        tokens_auc        AUC over ALL tokens (inflated reference). NaN if
                          single-class.
        tokens_code_auc   AUC over the live-code-masked subset only. NaN if that
                          subset is single-class / empty.
        dropped_fraction  1 - code_mask.mean() (0.0 when tree-sitter unavailable).
        n_pos_code        number of positive tokens surviving the mask.
        n_total_code      number of tokens surviving the mask.

    Does NOT mutate inputs. NaN AUCs are returned (not raised) for single-class
    subsets, matching the sweeps' existing roc_auc guards.
    """
    tok_p = np.asarray(tok_p, dtype=float)
    tok_y = np.asarray(tok_y)
    token_eids = np.asarray(token_eids)
    if not (tok_p.shape == tok_y.shape == token_eids.shape):
        raise ValueError(
            f"tok_p/tok_y/token_eids must be aligned 1-D arrays; got shapes "
            f"{tok_p.shape}, {tok_y.shape}, {token_eids.shape}"
        )

    code_mask = build_code_mask(token_eids, offsets_by_eid, dataset_rows_by_eid)

    tokens_auc = _safe_auc(tok_y, tok_p)
    y_code = tok_y[code_mask]
    p_code = tok_p[code_mask]
    tokens_code_auc = _safe_auc(y_code, p_code)
    dropped_fraction = float(1.0 - code_mask.mean()) if code_mask.size else 0.0

    return {
        "tokens_auc": tokens_auc,
        "tokens_code_auc": tokens_code_auc,
        "dropped_fraction": dropped_fraction,
        "n_pos_code": int((y_code == 1).sum()),
        "n_total_code": int(y_code.size),
    }
