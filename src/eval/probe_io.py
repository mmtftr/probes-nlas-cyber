"""Probe loading + scoring helpers.

A `Probe` here is the tiny `(w, b, layer)` triple saved by
`src/train_probe.py` as a .npz. We load it once and apply it as
   prob = sigmoid(X @ w + b)
to a stack of activations. No torch required at this layer.

For the *layered* probe (`data/probe_multilayer.npz`, multiple layers
combined) we read the keys flexibly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Probe:
    w: np.ndarray  # (dim,)
    b: float
    layer: int
    source: str = ""  # path or label

    def score(self, X: np.ndarray) -> np.ndarray:
        """X has shape (N, dim). Returns vulnerability probabilities (N,)."""
        logits = X @ self.w + self.b
        return _sigmoid(logits)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically-stable sigmoid via piecewise formulation.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def load_probe(path: str | Path) -> Probe:
    """Load a probe .npz with keys (w, b, layer)."""
    path = Path(path)
    npz = np.load(path)
    w = npz["w"].astype(np.float32)
    b = float(npz["b"])
    layer = int(npz["layer"])
    return Probe(w=w, b=b, layer=layer, source=str(path))


def load_activations(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) from an activations_layer*.npz file."""
    path = Path(path)
    npz = np.load(path)
    return npz["X"].astype(np.float32), npz["y"].astype(np.int8)


def load_pairs(path: str | Path) -> list[dict]:
    """Load a pairs .jsonl into a list of dicts."""
    path = Path(path)
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fit_logreg_on_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Probe:
    """Fit a linear probe on a training subset and return it as a Probe.

    Required when evaluating a *split* — we re-fit on train_idx, score on
    test_idx. For the OOD splits (heldout_cwe etc) this is the only honest
    way to test generalisation; using a globally-trained probe would leak
    the held-out CWE into its fit.
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=max_iter, C=C)
    clf.fit(X[train_idx], y[train_idx])
    return Probe(
        w=clf.coef_[0].astype(np.float32),
        b=float(clf.intercept_[0]),
        layer=-1,
        source="fit_on_split",
    )
