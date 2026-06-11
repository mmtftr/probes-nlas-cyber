"""Classification metrics for probe evaluation.

Mirrors the metric set used in the hallucination-probes paper
(`utils/metrics.py::compute_clf_metrics`) and adds:
  * bootstrap CI on AUC — small dataset (~1200 ex), point estimates lie.
  * calibration metrics (Brier, ECE) — the probe's score feeds a UI threshold,
    so it has to be honest about its confidence, not just rank-correct.
  * recall at several FPR targets, not just 0.1 — for the streaming UI we
    care about FPR=0.05 (one false alarm per ~20 tokens of safe code).

All functions take numpy arrays; no torch dependency. Probe forward passes
live in `probe_io.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)


FPR_TARGETS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)


@dataclass
class ClfMetrics:
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    optimal_threshold: float
    threshold_optimized_accuracy: float
    recall_at_fpr: dict[str, float] = field(default_factory=dict)
    n_pos: int = 0
    n_neg: int = 0
    n_total: int = 0
    # g-mean = sqrt(TPR * TNR); g-mean^2 = TPR * TNR. Reported at the threshold
    # that MAXIMISES g-mean (the probe's best class-balanced operating point),
    # which is the imbalanced-data analogue of `threshold_optimized_accuracy`.
    # Fields default to NaN so older callers/serialised cards stay valid.
    gmean: float = float("nan")
    gmean_squared: float = float("nan")
    gmean_threshold: float = float("nan")
    tpr_at_gmean: float = float("nan")
    tnr_at_gmean: float = float("nan")

    def to_dict(self) -> dict:
        d = asdict(self)
        # JSON-friendly: round floats for printing; keep raw for downstream.
        return d


def gmean_at_threshold(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> dict[str, float]:
    """g-mean (= sqrt(TPR*TNR)) and g-mean^2 (= TPR*TNR) at a fixed threshold.

    g-mean rewards joint performance on BOTH classes and collapses to 0 the
    moment one class is ignored — the right lens for the heavily imbalanced
    token-level eval (few vulnerable tokens, many safe code tokens), where
    plain accuracy is dominated by the negative class.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    g2 = tpr * tnr
    return {
        "threshold": float(threshold),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "gmean": float(np.sqrt(g2)) if g2 == g2 and g2 >= 0 else float("nan"),
        "gmean_squared": float(g2),
    }


def max_gmean(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Sweep ROC thresholds and return the one maximising g-mean.

    Returns {gmean, gmean_squared, threshold, tpr, tnr}. With TNR = 1 - FPR the
    sweep is exact over `roc_curve` breakpoints (every distinct score). NaN if
    only one class is present.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    nan = float("nan")
    if len(np.unique(labels)) < 2:
        return {"gmean": nan, "gmean_squared": nan, "threshold": nan,
                "tpr": nan, "tnr": nan}
    fpr, tpr, thr = roc_curve(labels, probs)
    tnr = 1.0 - fpr
    g = np.sqrt(np.clip(tpr * tnr, 0.0, None))
    k = int(np.argmax(g))
    return {
        "gmean": float(g[k]),
        "gmean_squared": float(tpr[k] * tnr[k]),
        # roc_curve's first threshold is +inf (predict-none point); clamp it.
        "threshold": float(min(thr[k], probs.max())),
        "tpr": float(tpr[k]),
        "tnr": float(tnr[k]),
    }


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def compute_clf_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    fpr_targets: Iterable[float] = FPR_TARGETS,
) -> ClfMetrics:
    """Classification metrics at the supplied threshold + threshold sweep.

    `preds` are the thresholded 0/1 predictions used for accuracy/P/R/F1.
    `probs` (continuous scores) drive AUC, optimal-threshold search, and
    recall@FPR.
    """
    preds = np.asarray(preds).astype(int)
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)

    auc = _safe_auc(labels, probs)
    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))

    optimal_thr = 0.5
    best_acc = float("nan")
    recall_at_fpr: dict[str, float] = {}

    if len(np.unique(labels)) == 2:
        fpr, tpr, thresholds = roc_curve(labels, probs)
        # Optimal threshold = the one that maximises accuracy on this split.
        # Candidate thresholds: 100 percentiles of unique scores.
        uniq = np.unique(probs)
        if len(uniq) > 100:
            cand = np.percentile(uniq, np.linspace(0, 100, 100))
        else:
            cand = uniq
        best_acc = 0.0
        for t in cand:
            y_pred = (probs >= t).astype(int)
            a = accuracy_score(labels, y_pred)
            if a > best_acc:
                best_acc = a
                optimal_thr = float(t)
        # Recall at each target FPR.
        for target in fpr_targets:
            idx = np.where(fpr <= target)[0]
            recall_at_fpr[f"recall_at_fpr_{target:.2f}"] = float(tpr[idx[-1]]) if len(idx) else 0.0

    # g-mean at its maximising threshold (NaN-filled when single-class).
    mg = max_gmean(labels, probs)

    return ClfMetrics(
        auc=auc,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        optimal_threshold=float(optimal_thr),
        threshold_optimized_accuracy=float(best_acc),
        recall_at_fpr=recall_at_fpr,
        n_pos=int((labels == 1).sum()),
        n_neg=int((labels == 0).sum()),
        n_total=int(len(labels)),
        gmean=mg["gmean"],
        gmean_squared=mg["gmean_squared"],
        gmean_threshold=mg["threshold"],
        tpr_at_gmean=mg["tpr"],
        tnr_at_gmean=mg["tnr"],
    )


def bootstrap_auc_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on AUC.

    Returns (auc, lo, hi). With ~150-700 test examples typical here, the AUC
    CI half-width is often ±0.03, which is the same order as the gaps
    between splits — reporting just the point estimate is misleading.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(np.unique(labels)) < 2 or len(labels) < 4:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(labels)
    aucs = np.empty(n_iter, dtype=np.float64)
    fills = 0
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        lb = labels[idx]
        if len(np.unique(lb)) < 2:
            continue
        aucs[fills] = roc_auc_score(lb, probs[idx])
        fills += 1
    if fills == 0:
        return float("nan"), float("nan"), float("nan")
    aucs = aucs[:fills]
    point = _safe_auc(labels, probs)
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return point, lo, hi


def calibration_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    """Brier score + expected calibration error.

    A probe with AUC=0.95 can still be miscalibrated — it ranks well but the
    score 0.7 doesn't mean "70% chance of vulnerability". Calibration matters
    when the UI uses a fixed threshold to highlight tokens.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(np.unique(labels)) < 2:
        return {"brier": float("nan"), "ece": float("nan")}
    brier = float(brier_score_loss(labels, probs))
    # ECE: bin probabilities, take |mean(prob) - mean(label)| per bin, weighted by bin size.
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return {"brier": brier, "ece": float(ece)}


def reliability_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability curve points for plotting.

    Returns (bin_centers, bin_acc, bin_count). Pass to matplotlib to draw
    the diagonal-vs-actual curve. Empty bins are returned as NaN.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    acc = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (probs >= lo) & (probs < hi)
        count[i] = int(mask.sum())
        if mask.any():
            acc[i] = labels[mask].mean()
    return centers, acc, count


def span_max_metrics(
    token_probs_per_example: list[np.ndarray],
    spans_per_example: list[list[tuple[int, int, int]]],
) -> ClfMetrics:
    """Span-max aggregation, mirroring paper Section 4.

    For each example: collect (start, end, label) spans (positive=1 for
    vulnerable-line / sink / source / evidence; negative=0 for sanitiser).
    Score the span as max(probe_prob[start:end+1]). Build a span-level
    (label, prob) list and compute standard binary metrics on top.

    Token-level path; only useful once token activations are extracted.
    """
    span_probs: list[float] = []
    span_labels: list[int] = []
    for probs, spans in zip(token_probs_per_example, spans_per_example):
        for start, end, lbl in spans:
            if end < start:
                continue
            end_c = min(end, len(probs) - 1)
            if end_c < start:
                continue
            span_probs.append(float(np.max(probs[start : end_c + 1])))
            span_labels.append(int(lbl))
    probs_arr = np.asarray(span_probs, dtype=float)
    labels_arr = np.asarray(span_labels, dtype=int)
    preds_arr = (probs_arr >= 0.5).astype(int)
    return compute_clf_metrics(preds_arr, labels_arr, probs_arr)
