"""Post-hoc Platt (sigmoid) calibration for the vulnerability probe.

The shipped binary probe (`data/probe.npz`) returns raw logits via
`sigmoid(w · h + b)`. Raw logits tend to be overconfident on non-risky tokens
(see WRITEUP), so we wrap the score in a 2-parameter Platt transform fit on a
held-out split: `p_calibrated = sigmoid((logit - a) / T)`.

The streaming server imports `apply_platt` and loads `T, a` from
`data/probe_calibration.json`. Calibration is post-hoc; the probe weights
themselves never change.
"""
from __future__ import annotations

import numpy as np


def apply_platt(raw_logits: np.ndarray, T: float, a: float) -> np.ndarray:
    """Apply Platt-scaled sigmoid to raw probe logits.

    Args:
        raw_logits: Probe logits, shape (N,) or scalar — `w · h + b`, NOT a
            sigmoid output.
        T: Temperature (>0). Larger T softens the distribution.
        a: Logit shift. Positive `a` makes the calibrated score more
            conservative (harder to call "risky").

    Returns:
        Calibrated probabilities in [0, 1] with the same shape as input.
    """
    x = np.asarray(raw_logits, dtype=np.float64)
    z = (x - a) / float(T)
    # Numerically stable sigmoid.
    out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return out.astype(np.float32)
