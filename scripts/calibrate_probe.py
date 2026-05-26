"""Fit Platt (sigmoid) calibration on the layer-17 probe.

Pipeline:
  1. Load `data/activations_v2/activations_layer17.npz`.
  2. Stratified 80/20 split with `random_state=7` — the train portion
     replicates the original probe; the held-out portion is what we calibrate
     against. This matches `src/train_probe.py:fit_layer` so the train probe is
     identical to the shipped weights.
  3. Train `LogisticRegression(max_iter=1000, C=1.0)`.
  4. Get raw logits on val via `clf.decision_function(X_val)`.
  5. Fit Platt parameters T, a by minimising NLL of
     `sigmoid((logit - a) / T)` with L-BFGS-B (init T=1, a=0).
  6. Report Brier + ECE before vs after, plus reliability curves.

Outputs:
  - `data/probe_calibration.json` (T, a, metrics, reliability).
  - Markdown table on stdout.

Calibration is post-hoc — the trained probe weights (`data/probe.npz`) are
NOT touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def _nll(params: np.ndarray, logits: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of Platt-scaled probs.

    `params = [T, a]`. We enforce T > 0 by clamping to a small positive floor
    rather than reparameterising — L-BFGS-B handles bounds directly.
    """
    T, a = float(params[0]), float(params[1])
    z = (logits - a) / max(T, 1e-6)
    # log-sigmoid that is stable for very negative z.
    log_p = -np.logaddexp(0.0, -z)
    log_1mp = -np.logaddexp(0.0, z)
    return float(-(y * log_p + (1.0 - y) * log_1mp).mean())


def expected_calibration_error(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict]]:
    """ECE with equal-width bins on [0, 1] + reliability curve points.

    Returns (ECE, reliability) where each reliability entry has
    `bin_lo, bin_hi, mean_pred, frac_pos, count`. Empty bins are skipped from
    ECE but emitted (count=0) so the curve is well-defined.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    rel: list[dict] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        # right-inclusive on the final bin so p == 1.0 lands somewhere
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rel.append({"bin_lo": float(lo), "bin_hi": float(hi),
                        "mean_pred": None, "frac_pos": None, "count": 0})
            continue
        mp = float(probs[mask].mean())
        fp = float(y[mask].mean())
        ece += (cnt / n) * abs(mp - fp)
        rel.append({"bin_lo": float(lo), "bin_hi": float(hi),
                    "mean_pred": mp, "frac_pos": fp, "count": cnt})
    return float(ece), rel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--acts",
        default="data/activations_v2/activations_layer17.npz",
        help="Layer-17 activation .npz (X, y).",
    )
    ap.add_argument(
        "--out",
        default="data/probe_calibration.json",
        help="Output calibration json.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    npz = np.load(args.acts)
    X, y = npz["X"], npz["y"].astype(np.int64)
    print(f"[calibrate] X={X.shape}  y_pos={int(y.sum())} / {len(y)}",
          file=sys.stderr)

    # Stratified 80/20 split — same recipe as src/train_probe.py:fit_layer.
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # Train the SAME probe (replicates the shipped weights for this layer).
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)

    # Raw logits and pre-calibration probabilities on held-out.
    raw_logits = clf.decision_function(Xval)
    p_before = _sigmoid(raw_logits)

    brier_before = float(brier_score_loss(yval, p_before))
    ece_before, rel_before = expected_calibration_error(p_before, yval, args.n_bins)
    print(f"[calibrate] before: Brier={brier_before:.4f}  ECE={ece_before:.4f}",
          file=sys.stderr)

    # Fit Platt (T, a) by L-BFGS-B on NLL.
    res = minimize(
        _nll,
        x0=np.array([1.0, 0.0]),
        args=(raw_logits, yval.astype(np.float64)),
        method="L-BFGS-B",
        bounds=[(1e-3, 1e3), (-50.0, 50.0)],
    )
    T, a = float(res.x[0]), float(res.x[1])
    print(f"[calibrate] fitted T={T:.4f}  a={a:.4f}  nll={res.fun:.4f}  "
          f"converged={res.success}", file=sys.stderr)

    # Post-calibration probabilities.
    p_after = _sigmoid((raw_logits - a) / T)
    brier_after = float(brier_score_loss(yval, p_after))
    ece_after, rel_after = expected_calibration_error(p_after, yval, args.n_bins)
    print(f"[calibrate] after:  Brier={brier_after:.4f}  ECE={ece_after:.4f}",
          file=sys.stderr)

    payload = {
        "layer": 17,
        "seed": args.seed,
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xval)),
        "T": T,
        "a": a,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "reliability_before": rel_before,
        "reliability_after": rel_after,
        "fit_success": bool(res.success),
        "fit_nll": float(res.fun),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[calibrate] wrote {args.out}", file=sys.stderr)

    # Markdown comparison table for the writeup / PR description.
    print()
    print("| Metric | Before | After | Delta |")
    print("|---|---|---|---|")
    print(f"| Brier | {brier_before:.4f} | {brier_after:.4f} | "
          f"{brier_after - brier_before:+.4f} |")
    print(f"| ECE (10-bin) | {ece_before:.4f} | {ece_after:.4f} | "
          f"{ece_after - ece_before:+.4f} |")
    print(f"| T | — | {T:.4f} | — |")
    print(f"| a | — | {a:.4f} | — |")


if __name__ == "__main__":
    main()
