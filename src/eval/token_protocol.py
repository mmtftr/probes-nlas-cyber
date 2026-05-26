"""Token-level eval orchestration for the new probe path.

Mirrors `protocol.py` but operates on per-token probe scores and the
char-range spans from `data/dataset.jsonl`. Reports metrics at four
aggregation levels (the "scope" axis — what gets scored):

  tokens                       every token in every example. Negative
                               class = complement of positive spans
                               (includes comments / signatures / imports /
                               whitespace). Inflates AUC because most
                               negatives are trivial.
  tokens_code                  same scoring as `tokens`, but restricted
                               to live-code tokens via tree-sitter
                               (drops comments, imports, def / class /
                               namespace signatures, decorators, C/C++
                               preprocessor). Negatives are then
                               live-code-but-not-vuln tokens — the
                               actual hard case a deployed probe has to
                               beat. Empty (`n_total: 0`) when no
                               `code_only_masks` supplied. See
                               `src/eval/code_mask.py`.
  tokens_annotated_negative    only tokens with an explicit annotation
                               (positive *or* sanitizer-negative). The
                               complement is NOT inferred as negative.
                               Single-class on the current SVEN dataset
                               (sanitizer spans nearly empty) → AUC NaN.
                               Useful once more sanitizer annotations
                               land.
  example_max                  one (row.label, max(probs)) pair per
                               example. Pure example-level decision —
                               does not consult the annotated spans at
                               score time. Matches the streaming-UI
                               rule "file is risky iff any token's prob
                               is high" and the `ex_AUC` reported by
                               `src/train_probe_spanmax.py`.

Orthogonal preprocessing axis: `proximity_window` (W, default 0). When
W > 0, the positive token labels used by `tokens` and `tokens_code` are
dilated by ±W tokens around each annotated positive span, giving the
probe credit for firing in the halo around the diff anchor. `W` is
stamped into each scope-level metrics dict for traceability. Does not
affect `tokens_annotated_negative` (annotation-only scoping) or
`example_max` (row-level label, no per-token mask).

Token-level scores have to come from somewhere — see
`scripts/run_token_eval.py` for the expected on-disk format. The eval
side here is tokenizer-agnostic; the caller supplies token offsets.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, Sequence

import numpy as np

from .baselines import Baseline, BroadcastProbeBaseline
from .metrics import (
    ClfMetrics,
    bootstrap_auc_ci,
    calibration_metrics,
    compute_clf_metrics,
)
from .splits import Split, make_splits
from .token_data import TokenSpan, parse_spans, token_labels_array


@dataclass
class TokenSplitReport:
    split_name: str
    n_train_examples: int
    n_test_examples: int
    note: str
    # Per-aggregation-level metric dicts. Each contains: auc, ci_lo, ci_hi,
    # accuracy, precision, recall, f1, recall_at_fpr_*, brier, ece, n_total,
    # n_pos, proximity_window (the W used to dilate positive labels, 0 if not
    # applicable). See the module docstring for the per-level scoring rules.
    tokens_metrics: dict[str, float] = field(default_factory=dict)
    tokens_code_metrics: dict[str, float] = field(default_factory=dict)
    tokens_annotated_negative_metrics: dict[str, float] = field(default_factory=dict)
    example_max_metrics: dict[str, float] = field(default_factory=dict)
    baseline_aucs: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FullTokenReport:
    dataset_path: str
    n_examples: int
    n_pos_examples: int
    splits: list[TokenSplitReport]

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "n_examples": self.n_examples,
            "n_pos_examples": self.n_pos_examples,
            "splits": [s.to_dict() for s in self.splits],
        }


def _level_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    bootstrap_n: int,
) -> dict[str, float]:
    """Pack ClfMetrics + bootstrap CI + calibration into a flat dict."""
    if len(labels) == 0:
        return {"n_total": 0, "n_pos": 0}
    preds = (probs >= 0.5).astype(int)
    clf = compute_clf_metrics(preds, labels, probs)
    _, lo, hi = bootstrap_auc_ci(labels, probs, n_iter=bootstrap_n)
    cal = calibration_metrics(labels, probs)
    return {
        "auc": clf.auc,
        "auc_ci_lo": lo,
        "auc_ci_hi": hi,
        "accuracy": clf.accuracy,
        "precision": clf.precision,
        "recall": clf.recall,
        "f1": clf.f1,
        "recall_at_fpr_0.05": clf.recall_at_fpr.get("recall_at_fpr_0.05", float("nan")),
        "recall_at_fpr_0.10": clf.recall_at_fpr.get("recall_at_fpr_0.10", float("nan")),
        "brier": cal["brier"],
        "ece": cal["ece"],
        "n_total": clf.n_total,
        "n_pos": clf.n_pos,
    }


def _collect_for_indices(
    indices: Iterable[int],
    rows: list[dict],
    token_probs: list[np.ndarray],
    token_spans: list[list[tuple[int, int, int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, float]]]:
    """Stack per-example token scores and labels.

    Returns:
      all_probs           (T,)  every token, every example concatenated
      all_labels          (T,)  exact-span positives (no dilation)
      in_annotated_span   (T,)  bool — token sits inside any annotated span
      example_max_pairs   list of (row.label, max(probs)) per example

    `example_max` is one pair per example: (row-level label, max over
    all tokens in that example). This matches both the streaming-UI
    operational rule ("file is risky iff any token's prob is high") and
    `src/train_probe_spanmax.py`'s `ex_AUC` reporting.

    Proximity dilation (W > 0) is intentionally applied later in
    `evaluate_token_split`, not here — this collector returns the raw
    exact-span labels that all downstream levels start from.
    """
    all_probs_chunks: list[np.ndarray] = []
    all_labels_chunks: list[np.ndarray] = []
    in_span_chunks: list[np.ndarray] = []
    example_max_pairs: list[tuple[int, float]] = []
    for i in indices:
        probs = token_probs[i]
        spans = token_spans[i]
        labels, mask = token_labels_array(len(probs), spans)
        all_probs_chunks.append(probs)
        all_labels_chunks.append(labels)
        in_span_chunks.append(mask)
        if len(probs) > 0:
            ex_label = int(rows[i].get("label", int(any(lbl == 1 for _, _, lbl in spans))))
            example_max_pairs.append((ex_label, float(np.max(probs))))
    if not all_probs_chunks:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=int),
            np.empty(0, dtype=bool),
            example_max_pairs,
        )
    return (
        np.concatenate(all_probs_chunks),
        np.concatenate(all_labels_chunks),
        np.concatenate(in_span_chunks),
        example_max_pairs,
    )


def _dilate_positive_labels(
    indices: Iterable[int],
    token_probs: list[np.ndarray],
    token_spans: list[list[tuple[int, int, int]]],
    proximity_window: int,
) -> np.ndarray:
    """Build the concatenated dilated-positive label vector for `indices`.

    For each example, marks every token within ±`proximity_window` of an
    annotated positive span as positive. Iteration order must match
    `_collect_for_indices`.
    """
    chunks: list[np.ndarray] = []
    for i in indices:
        n_tok = len(token_probs[i])
        local = np.zeros(n_tok, dtype=np.int8)
        if n_tok > 0:
            for s, e, lbl in token_spans[i]:
                if lbl != 1:
                    continue
                s2 = max(0, s - proximity_window)
                e2 = min(n_tok - 1, e + proximity_window)
                if e2 >= s2:
                    local[s2 : e2 + 1] = 1
        chunks.append(local)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int8)


def evaluate_token_split(
    rows: list[dict],
    token_probs: list[np.ndarray],
    token_spans: list[list[tuple[int, int, int]]],
    split: Split,
    *,
    bootstrap_n: int = 1000,
    baselines: Optional[list[Baseline]] = None,
    sample_X: Optional[np.ndarray] = None,
    proximity_window: int = 0,
    code_only_masks: Optional[list[np.ndarray]] = None,
) -> TokenSplitReport:
    """Evaluate one split at token level.

    `token_probs[i]` is the per-token probability array for row `i`.
    `token_spans[i]` is the row's char-spans already mapped to token
    indices (see `token_data.char_spans_to_token_spans`).

    The probe scores are evaluated as-is — no refit at token level (a
    fixed token-probe is the deployment artifact; refitting per split
    requires per-token activations, which slots in cleanly above this
    layer once #17 lands).

    Baselines are scored at the *example* level and broadcast to every
    token in each test row. That's the right comparison for the streaming
    UI: the old sample-level probe outputs a single decision per file,
    and we want to know what AUC that single decision achieves when
    treated as a token-level signal.
    """
    te = split.test_idx
    te_list = te.tolist()
    all_p, all_y_exact, annot_mask, ex_max_pairs = _collect_for_indices(
        te_list, rows, token_probs, token_spans,
    )

    # `tokens` and `tokens_code` use the dilated positive labels when
    # proximity_window > 0; everything else uses the exact-span labels.
    # `tokens_annotated_negative` and `example_max` are unaffected by W.
    if proximity_window > 0:
        scope_y = _dilate_positive_labels(
            te_list, token_probs, token_spans, proximity_window,
        )
    else:
        scope_y = all_y_exact

    tokens_m = _level_metrics(scope_y, all_p, bootstrap_n)
    tokens_m["proximity_window"] = int(proximity_window)

    # tokens_code: same scoring as `tokens`, but sliced by the live-code mask
    # (drops comments / signatures / imports / preprocessor via tree-sitter).
    # Masks must be row-for-row aligned with `rows` and use the same iteration
    # order as `_collect_for_indices` so the concatenated mask matches all_p.
    if code_only_masks is not None:
        co_chunks: list[np.ndarray] = []
        for i in te_list:
            m = code_only_masks[i]
            if m.shape[0] != len(token_probs[i]):
                # Length mismatch — fall back to keep-all to avoid silent
                # misalignment.
                co_chunks.append(np.ones(len(token_probs[i]), dtype=bool))
            else:
                co_chunks.append(m.astype(bool))
        co_concat = np.concatenate(co_chunks) if co_chunks else np.empty(0, dtype=bool)
        if co_concat.size == scope_y.size and co_concat.any():
            tokens_code_m = _level_metrics(scope_y[co_concat], all_p[co_concat], bootstrap_n)
            tokens_code_m["dropped_fraction"] = float(1.0 - co_concat.mean())
            tokens_code_m["proximity_window"] = int(proximity_window)
        else:
            tokens_code_m = {"n_total": 0, "n_pos": 0}
    else:
        tokens_code_m = {"n_total": 0, "n_pos": 0}

    # tokens_annotated_negative: only tokens with explicit annotation
    # (positive evidence/sink/source/vulnerable_line OR sanitizer-negative).
    # No proximity dilation — the level is defined by annotation, not halo.
    annot_p = all_p[annot_mask]
    annot_y = all_y_exact[annot_mask]
    tokens_ann_neg_m = _level_metrics(annot_y, annot_p, bootstrap_n)

    # example_max: one (row.label, max(probs)) pair per example. Does not
    # consult token spans at score time.
    if ex_max_pairs:
        em_labels = np.array([p[0] for p in ex_max_pairs], dtype=int)
        em_probs = np.array([p[1] for p in ex_max_pairs], dtype=float)
        example_max_m = _level_metrics(em_labels, em_probs, bootstrap_n)
    else:
        example_max_m = {"n_total": 0, "n_pos": 0}

    # Baselines: score each test row at example level, broadcast.
    baseline_aucs: dict[str, dict[str, float]] = {}
    if baselines:
        from sklearn.metrics import roc_auc_score
        rows_te = [rows[i] for i in te_list]
        for bl in baselines:
            try:
                if isinstance(bl, BroadcastProbeBaseline) and sample_X is not None:
                    ex_scores = bl.score(rows_te, X=sample_X[te])
                else:
                    ex_scores = bl.score(rows_te)
            except Exception as e:  # noqa: BLE001
                baseline_aucs[bl.name] = {"error": str(e)}
                continue
            # Broadcast: every token in row i gets ex_scores[i].
            broad_probs = []
            for j, i in enumerate(te_list):
                broad_probs.append(np.full_like(token_probs[i], ex_scores[j], dtype=float))
            broad_arr = np.concatenate(broad_probs) if broad_probs else np.empty(0)
            level: dict[str, float] = {}
            try:
                if len(np.unique(scope_y)) >= 2:
                    level["tokens_auc"] = float(roc_auc_score(scope_y, broad_arr))
                else:
                    level["tokens_auc"] = float("nan")
                if (
                    code_only_masks is not None
                    and co_concat.size == scope_y.size
                    and co_concat.any()
                    and len(np.unique(scope_y[co_concat])) >= 2
                ):
                    level["tokens_code_auc"] = float(
                        roc_auc_score(scope_y[co_concat], broad_arr[co_concat])
                    )
                else:
                    level["tokens_code_auc"] = float("nan")
                if len(np.unique(annot_y)) >= 2 and len(annot_y) > 0:
                    level["tokens_annotated_negative_auc"] = float(
                        roc_auc_score(annot_y, broad_arr[annot_mask])
                    )
                else:
                    level["tokens_annotated_negative_auc"] = float("nan")
                # example_max baseline: one prediction per example = ex_score.
                # Aligned with ex_max_pairs (one-per-example, skipping
                # empty-token rows, same iteration order).
                if ex_max_pairs and len(np.unique([p[0] for p in ex_max_pairs])) >= 2:
                    em_baseline_probs = [
                        ex_scores[j]
                        for j, i in enumerate(te_list)
                        if len(token_probs[i]) > 0
                    ]
                    em_b = np.array(em_baseline_probs, dtype=float)
                    em_lbl = np.array([p[0] for p in ex_max_pairs], dtype=int)
                    level["example_max_auc"] = float(roc_auc_score(em_lbl, em_b))
                else:
                    level["example_max_auc"] = float("nan")
            except Exception as e:  # noqa: BLE001
                level["error"] = str(e)
            baseline_aucs[bl.name] = level

    return TokenSplitReport(
        split_name=split.name,
        n_train_examples=int(len(split.train_idx)),
        n_test_examples=int(len(te)),
        note=split.note,
        tokens_metrics=tokens_m,
        tokens_code_metrics=tokens_code_m,
        tokens_annotated_negative_metrics=tokens_ann_neg_m,
        example_max_metrics=example_max_m,
        baseline_aucs=baseline_aucs,
    )


def full_token_report(
    rows: list[dict],
    token_probs: list[np.ndarray],
    token_spans: list[list[tuple[int, int, int]]],
    *,
    dataset_path: str,
    include: tuple[str, ...] = ("random", "group_repo", "heldout_cwe", "heldout_lang", "heldout_source"),
    seed: int = 7,
    bootstrap_n: int = 1000,
    baselines: Optional[list[Baseline]] = None,
    sample_X: Optional[np.ndarray] = None,
    proximity_window: int = 0,
    code_only_masks: Optional[list[np.ndarray]] = None,
) -> FullTokenReport:
    """Run the canonical split bundle on token-level data.

    Splits are still produced at the *example* level (one row in
    `data/dataset.jsonl` = one example). Token-level evaluation then
    slices the per-row probs / spans by example index.
    """
    y = np.array([int(r.get("label", 0)) for r in rows], dtype=np.int8)
    splits = make_splits(rows, y, include=include, seed=seed)
    reports = [
        evaluate_token_split(
            rows, token_probs, token_spans, sp,
            bootstrap_n=bootstrap_n, baselines=baselines, sample_X=sample_X,
            proximity_window=proximity_window,
            code_only_masks=code_only_masks,
        )
        for sp in splits
    ]
    return FullTokenReport(
        dataset_path=dataset_path,
        n_examples=len(rows),
        n_pos_examples=int(y.sum()),
        splits=reports,
    )
