"""Eval orchestration: take (X, y, rows, split, probe) and produce a report.

There are two probe modes:
  * `refit=True`  - fit a fresh LogisticRegression on train_idx and score test_idx.
                    This is the right thing for **OOD** generalisation claims:
                    "does the technique generalise" needs the held-out class to
                    be unseen at fit time too.
  * `refit=False` - use a pre-trained Probe; only score test_idx.
                    This is the right thing for **deployment** claims:
                    "given the probe we shipped, how does it do on slice X".

For a fair OOD evaluation across splits, refit=True is the default.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .baselines import Baseline
from .metrics import (
    ClfMetrics,
    bootstrap_auc_ci,
    calibration_metrics,
    compute_clf_metrics,
)
from .probe_io import Probe, fit_logreg_on_split
from .splits import Split, make_splits


@dataclass
class SplitReport:
    split_name: str
    n_train: int
    n_test: int
    n_test_pos: int
    note: str
    auc: float
    auc_ci_lo: float
    auc_ci_hi: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    optimal_threshold: float
    threshold_optimized_accuracy: float
    recall_at_fpr: dict[str, float]
    brier: float
    ece: float
    baseline_aucs: dict[str, float] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FullReport:
    activations_path: str
    pairs_path: str
    layer: int
    n_examples: int
    n_pos: int
    refit: bool
    splits: list[SplitReport]

    def to_dict(self) -> dict:
        return {
            "activations_path": self.activations_path,
            "pairs_path": self.pairs_path,
            "layer": self.layer,
            "n_examples": self.n_examples,
            "n_pos": self.n_pos,
            "refit": self.refit,
            "splits": [s.to_dict() for s in self.splits],
        }


def evaluate_split(
    X: np.ndarray,
    y: np.ndarray,
    rows: list[dict],
    split: Split,
    *,
    probe: Optional[Probe] = None,
    refit: bool = True,
    threshold: float = 0.5,
    bootstrap_n: int = 1000,
    baselines: Optional[list[Baseline]] = None,
) -> SplitReport:
    """Run one split and return a populated SplitReport."""
    tr, te = split.train_idx, split.test_idx
    y_te = y[te]

    # Probe path.
    if refit:
        if len(np.unique(y[tr])) < 2:
            return _empty_report(split, len(tr), len(te), int(y_te.sum()), note="train set single-class")
        probe = fit_logreg_on_split(X, y, tr)
    if probe is None:
        raise ValueError("probe required when refit=False")
    probs = probe.score(X[te])
    preds = (probs >= threshold).astype(int)

    clf = compute_clf_metrics(preds, y_te, probs)
    _, lo, hi = bootstrap_auc_ci(y_te, probs, n_iter=bootstrap_n)
    cal = calibration_metrics(y_te, probs)

    # Baselines on the same test slice; rows[te] only. Baselines that
    # require activations (the shipped sample-level probe) get X[te].
    baseline_aucs: dict[str, float] = {}
    if baselines:
        from sklearn.metrics import roc_auc_score
        rows_te = [rows[i] for i in te.tolist()]
        for bl in baselines:
            try:
                s = bl.score(rows_te, X=X[te] if getattr(bl, "needs_activations", False) else None)
                if len(np.unique(y_te)) >= 2:
                    baseline_aucs[bl.name] = float(roc_auc_score(y_te, s))
                else:
                    baseline_aucs[bl.name] = float("nan")
            except Exception as e:  # noqa: BLE001
                baseline_aucs[bl.name] = float("nan")
                baseline_aucs[f"{bl.name}__error"] = str(e)

    return SplitReport(
        split_name=split.name,
        n_train=int(len(tr)),
        n_test=int(len(te)),
        n_test_pos=int(y_te.sum()),
        note=split.note,
        auc=clf.auc,
        auc_ci_lo=lo,
        auc_ci_hi=hi,
        accuracy=clf.accuracy,
        precision=clf.precision,
        recall=clf.recall,
        f1=clf.f1,
        optimal_threshold=clf.optimal_threshold,
        threshold_optimized_accuracy=clf.threshold_optimized_accuracy,
        recall_at_fpr=clf.recall_at_fpr,
        brier=cal["brier"],
        ece=cal["ece"],
        baseline_aucs=baseline_aucs,
    )


def _empty_report(split: Split, n_tr: int, n_te: int, n_pos: int, note: str) -> SplitReport:
    return SplitReport(
        split_name=split.name,
        n_train=n_tr,
        n_test=n_te,
        n_test_pos=n_pos,
        note=note,
        auc=float("nan"),
        auc_ci_lo=float("nan"),
        auc_ci_hi=float("nan"),
        accuracy=float("nan"),
        precision=float("nan"),
        recall=float("nan"),
        f1=float("nan"),
        optimal_threshold=float("nan"),
        threshold_optimized_accuracy=float("nan"),
        recall_at_fpr={},
        brier=float("nan"),
        ece=float("nan"),
    )


def full_report(
    X: np.ndarray,
    y: np.ndarray,
    rows: list[dict],
    *,
    activations_path: str,
    pairs_path: str,
    layer: int,
    probe: Optional[Probe] = None,
    refit: bool = True,
    baselines: Optional[list[Baseline]] = None,
    include: tuple[str, ...] = ("random", "group_repo", "heldout_cwe", "heldout_lang", "heldout_source"),
    seed: int = 7,
    bootstrap_n: int = 1000,
    threshold: float = 0.5,
) -> FullReport:
    """Run the full split bundle and return a `FullReport`."""
    splits = make_splits(rows, y, include=include, seed=seed)
    reports = [
        evaluate_split(
            X, y, rows, sp,
            probe=probe, refit=refit, baselines=baselines,
            bootstrap_n=bootstrap_n, threshold=threshold,
        )
        for sp in splits
    ]
    return FullReport(
        activations_path=activations_path,
        pairs_path=pairs_path,
        layer=layer,
        n_examples=len(y),
        n_pos=int(y.sum()),
        refit=refit,
        splits=reports,
    )
