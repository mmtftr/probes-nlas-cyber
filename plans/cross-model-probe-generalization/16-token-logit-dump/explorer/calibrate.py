# [ai-generated]
"""Fit Platt calibration per exp-16 probe, locally, on its own held-out split.

For every (model, layer) probe we:
  1. Fit a 2-parameter Platt sigmoid  p = sigmoid((logit - a) / T)  by maximising
     the held-out (is_test) label likelihood — Lin et al. 2007 stable Newton,
     the same routine libsvm/sklearn use. Done at BOTH levels:
       - token   : (token logit, token label y) over held-out tokens
       - example : (example logit_max, example label) over held-out examples
  2. Pick the decision threshold by F1-max on the calibrated held-out scores.

We do NOT reuse the gemmaforge production numbers (T=1.794, a=-0.269,
thr=0.929) — those were fit on a different probe. They are emitted only under
the "production" key as a labelled reference for the UI.

Output: explorer/calibration.json
  { "<model_key>": { "<layer>": { "token": {...}, "example": {...} } },
    "production": { "T":..., "a":..., "threshold":... },
    "_meta": {...} }

Run:  uv run python calibrate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
OUT = HERE / "calibration.json"

PROD = {"T": 1.794, "a": -0.269, "threshold": 0.929082,
        "source": "gemmaforge production probe (F1-max on Platt-calibrated heldout)"}

_DIR_RE = __import__("re").compile(r"^logitdump_(.+)$")


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def fit_platt(deci: np.ndarray, label: np.ndarray, max_iter: int = 100):
    """Lin/Lin/Weng/Chang (2007) Platt fit. Returns (A, B) with p=sigmoid(-(A*x+B)),
    then converts to (T, a) for p = sigmoid((x - a)/T)."""
    deci = np.asarray(deci, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    prior1 = float(label.sum())
    prior0 = float(len(label) - prior1)
    if prior1 == 0 or prior0 == 0:
        return None
    hi = (prior1 + 1.0) / (prior1 + 2.0)
    lo = 1.0 / (prior0 + 2.0)
    t = np.where(label > 0.5, hi, lo)

    A, B = 0.0, np.log((prior0 + 1.0) / (prior1 + 1.0))
    eps = 1e-12

    def obj(A, B):
        fApB = deci * A + B
        # -loglik with stable log-sigmoid
        return np.sum(
            np.where(fApB >= 0, t * fApB + np.log1p(np.exp(-fApB)),
                     (t - 1.0) * fApB + np.log1p(np.exp(fApB)))
        )

    fval = obj(A, B)
    for _ in range(max_iter):
        fApB = deci * A + B
        p = np.where(fApB >= 0, np.exp(-fApB) / (1.0 + np.exp(-fApB)),
                     1.0 / (1.0 + np.exp(fApB)))
        q = 1.0 - p
        d2 = p * q
        h11 = np.sum(deci * deci * d2) + eps
        h22 = np.sum(d2) + eps
        h21 = np.sum(deci * d2)
        d1 = np.sum(deci * (t - p))
        d_1 = np.sum(t - p)
        det = h11 * h22 - h21 * h21
        if abs(det) < 1e-300:
            break
        dA = -(h22 * d1 - h21 * d_1) / det
        dB = -(-h21 * d1 + h11 * d_1) / det
        gd = d1 * dA + d_1 * dB
        # line search
        stepsize = 1.0
        while stepsize >= 1e-10:
            newA, newB = A + stepsize * dA, B + stepsize * dB
            newf = obj(newA, newB)
            if newf < fval + 1e-4 * stepsize * gd:
                A, B, fval = newA, newB, newf
                break
            stepsize /= 2.0
        if stepsize < 1e-10:
            break
        if abs(dA) < 1e-9 and abs(dB) < 1e-9:
            break

    if A >= 0:  # higher logit must map to higher p; degenerate fit otherwise
        return None
    T = -1.0 / A
    a = -B / A
    return T, a


def f1max_threshold(p: np.ndarray, y: np.ndarray):
    """Vectorised F1-max over thresholds = distinct scores. Returns dict."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    P = int(y.sum())
    if P == 0 or P == len(y):
        return None
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    ps = p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    bi = int(np.argmax(f1))
    return {
        "threshold": float(ps[bi]),
        "precision": float(prec[bi]),
        "recall": float(rec[bi]),
        "f1": float(f1[bi]),
        "n": int(len(y)),
        "n_pos": P,
        "base_rate": float(P / len(y)),
    }


def _brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def calibrate_level(deci, label):
    """Fit Platt on held-out (deci,label), return calibration record or None."""
    fit = fit_platt(deci, label)
    if fit is None:
        return None
    T, a = fit
    p_cal = _sigmoid((np.asarray(deci, float) - a) / T)
    p_raw = _sigmoid(deci)
    thr = f1max_threshold(p_cal, label)
    if thr is None:
        return None
    rec = {"T": float(T), "a": float(a)}
    rec.update(thr)
    rec["brier_raw"] = _brier(p_raw, label)
    rec["brier_cal"] = _brier(p_cal, label)
    return rec


def main():
    out = {"production": PROD, "_meta": {
        "method": "Lin et al. 2007 Platt; fit on held-out (is_test) split; "
                  "threshold = F1-max on calibrated heldout scores",
    }}
    for d in sorted(RESULTS_DIR.iterdir()):
        m = _DIR_RE.match(d.name) if d.is_dir() else None
        if not m:
            continue
        key = m.group(1)
        out[key] = {}
        for npz_path in sorted(d.glob("logits_layer*.npz")):
            layer = int(npz_path.stem.split("layer")[-1])
            z = np.load(npz_path, allow_pickle=False)
            is_test = z["is_test"]
            tok_deci = z["logit"][is_test]
            tok_y = z["y"][is_test]
            tok = calibrate_level(tok_deci, tok_y)

            es_path = d / f"example_scores_layer{layer:02d}.json"
            ex = None
            if es_path.is_file():
                es = json.loads(es_path.read_text())
                test = [e for e in es if e.get("is_test")]
                if test:
                    ex_deci = np.array([e["logit_max"] for e in test], float)
                    ex_y = np.array([e["label"] for e in test], int)
                    ex = calibrate_level(ex_deci, ex_y)

            out[key][str(layer)] = {"token": tok, "example": ex}
            tt = f"T={tok['T']:.3f} a={tok['a']:.3f} thr={tok['threshold']:.3f} F1={tok['f1']:.3f}" if tok else "FAIL"
            et = f"thr={ex['threshold']:.3f} F1={ex['f1']:.3f}" if ex else "FAIL"
            print(f"  {key} L{layer:02d}: token[{tt}]  example[{et}]")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[calibrate] wrote {OUT}")


if __name__ == "__main__":
    main()
