"""Apply the fitted Platt calibration to both held-out subsets.

Inputs:
  - `data/probe_calibration.json` (T, a) from `scripts/calibrate_probe.py`.
  - `data/activations_v2/activations_layer17.npz` for the full held-out probe
    eval (last-token activations, replicates the original train split).
  - `data/adversarial_eval.json` for T1/T3 adversarial flip-rate change.

Outputs:
  - `data/calibration_eval.json` with AUC/Brier/ECE before vs after on the
    held-out, plus T1/T3 mean-score and flip-rate before vs after on the
    adversarial probes (both last-token and maxpool variants where present).

Adversarial scores are stored as sigmoid outputs, so we invert
(`logit = log(p / (1-p))`) before applying Platt. The flip threshold matches
`scripts/adversarial_test.py` (post-transform score < 0.5).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Local import so the eval uses the same primitive as the streaming server.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.probes.calibration import apply_platt  # noqa: E402

from scripts.calibrate_probe import expected_calibration_error  # noqa: E402


FLIP_THRESHOLD = 0.5  # matches scripts/adversarial_test.py:_flip_rate


def _invert_sigmoid(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Recover raw logits from sigmoid outputs (clamped for stability)."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _flip_rate(scores: np.ndarray, threshold: float = FLIP_THRESHOLD) -> float:
    if len(scores) == 0:
        return 0.0
    return float(np.mean(np.asarray(scores) < threshold))


def evaluate_holdout(acts_path: Path, T: float, a: float, seed: int = 7,
                    n_bins: int = 10) -> dict:
    """Replicate the calibrate-probe split and report before/after metrics."""
    npz = np.load(acts_path)
    X, y = npz["X"], npz["y"].astype(np.int64)
    _, Xval, _, yval = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, C=1.0)
    # Retrain on the same train portion so the eval is self-consistent.
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    clf.fit(Xtr, ytr)
    raw_logits = clf.decision_function(Xval)
    p_before = 1.0 / (1.0 + np.exp(-raw_logits))
    p_after = apply_platt(raw_logits, T, a).astype(np.float64)

    auc_before = float(roc_auc_score(yval, p_before))
    auc_after = float(roc_auc_score(yval, p_after))
    brier_before = float(brier_score_loss(yval, p_before))
    brier_after = float(brier_score_loss(yval, p_after))
    ece_before, _ = expected_calibration_error(p_before, yval, n_bins)
    ece_after, _ = expected_calibration_error(p_after, yval, n_bins)
    return {
        "n_val": int(len(Xval)),
        "auc_before": auc_before,
        "auc_after": auc_after,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }


def evaluate_adversarial(adv_path: Path, T: float, a: float,
                         threshold: float = FLIP_THRESHOLD) -> dict:
    """Recompute T1/T3 mean-score and flip-rate before vs after Platt.

    The adversarial JSON stores final sigmoid outputs, so we invert to get
    raw logits, then re-sigmoid via Platt.
    """
    adv = json.loads(adv_path.read_text())
    per = adv["per_example"]

    def _scores(key: str) -> np.ndarray:
        return np.array([e[key] for e in per], dtype=np.float64)

    def _summarise(p_before_arr: np.ndarray) -> dict:
        logits = _invert_sigmoid(p_before_arr)
        p_after_arr = apply_platt(logits, T, a).astype(np.float64)
        return {
            "mean_before": float(p_before_arr.mean()),
            "mean_after": float(p_after_arr.mean()),
            "flip_rate_before": _flip_rate(p_before_arr, threshold),
            "flip_rate_after": _flip_rate(p_after_arr, threshold),
        }

    return {
        "threshold": threshold,
        "n_examples": len(per),
        "baseline_last": _summarise(_scores("baseline_last")),
        "baseline_max": _summarise(_scores("baseline_max")),
        "t1_last": _summarise(_scores("t1_last")),
        "t1_max": _summarise(_scores("t1_max")),
        "t3_last": _summarise(_scores("t3_last")),
        "t3_max": _summarise(_scores("t3_max")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="data/probe_calibration.json")
    ap.add_argument("--acts", default="data/activations_v2/activations_layer17.npz")
    ap.add_argument("--adversarial", default="data/adversarial_eval.json")
    ap.add_argument("--out", default="data/calibration_eval.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=FLIP_THRESHOLD)
    args = ap.parse_args()

    cal = json.loads(Path(args.calibration).read_text())
    T, a = float(cal["T"]), float(cal["a"])
    print(f"[apply] T={T:.4f}  a={a:.4f}", file=sys.stderr)

    holdout = evaluate_holdout(Path(args.acts), T, a, args.seed, args.n_bins)
    print(
        f"[apply] holdout: AUC {holdout['auc_before']:.4f}->{holdout['auc_after']:.4f}  "
        f"Brier {holdout['brier_before']:.4f}->{holdout['brier_after']:.4f}  "
        f"ECE {holdout['ece_before']:.4f}->{holdout['ece_after']:.4f}",
        file=sys.stderr,
    )

    adv = evaluate_adversarial(Path(args.adversarial), T, a, args.threshold)
    t3 = adv["t3_last"]
    print(
        f"[apply] T3 last: flip {t3['flip_rate_before']:.3f}->{t3['flip_rate_after']:.3f}  "
        f"mean {t3['mean_before']:.3f}->{t3['mean_after']:.3f}",
        file=sys.stderr,
    )

    payload = {
        "T": T,
        "a": a,
        "seed": args.seed,
        "holdout": holdout,
        "adversarial": adv,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[apply] wrote {args.out}", file=sys.stderr)

    # Markdown summary.
    print()
    print("### Held-out (last-token)")
    print()
    print("| Metric | Before | After |")
    print("|---|---|---|")
    print(f"| AUC | {holdout['auc_before']:.4f} | {holdout['auc_after']:.4f} |")
    print(f"| Brier | {holdout['brier_before']:.4f} | {holdout['brier_after']:.4f} |")
    print(f"| ECE | {holdout['ece_before']:.4f} | {holdout['ece_after']:.4f} |")
    print()
    print(f"### Adversarial (threshold={args.threshold})")
    print()
    print("| Subset | mean before | mean after | flip before | flip after |")
    print("|---|---|---|---|---|")
    for k in ("baseline_last", "t1_last", "t3_last",
              "baseline_max", "t1_max", "t3_max"):
        r = adv[k]
        print(f"| {k} | {r['mean_before']:.3f} | {r['mean_after']:.3f} | "
              f"{r['flip_rate_before']:.3f} | {r['flip_rate_after']:.3f} |")


if __name__ == "__main__":
    main()
