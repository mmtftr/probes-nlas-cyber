"""Measure probe inference overhead on top of the base model forward pass.

The README claims `< 1 ms per token of overhead on top of normal forward
pass`. That number is only honest if we can reproduce it; this module
gives a script you can run on the same machine the demo will run on.

Three measurements:
  1. base_ms_per_token       - model.generate, probe disabled
  2. probe_ms_per_token      - model.generate with probe applied on each token
  3. overhead_ms_per_token   - (2) - (1), with median over `n_runs` trials
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np


@dataclass
class RuntimeReport:
    n_runs: int
    n_tokens: int
    base_ms_per_token_median: float
    probe_ms_per_token_median: float
    overhead_ms_per_token_median: float
    overhead_ms_per_token_p95: float
    base_runs_ms: list[float]
    probe_runs_ms: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    arr = sorted(xs)
    k = (len(arr) - 1) * p
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return arr[f]
    return arr[f] + (arr[c] - arr[f]) * (k - f)


def measure_runtime(
    generate_fn: Callable[[], int],
    apply_probe_fn: Callable[[], int] | None,
    n_runs: int = 5,
    warmup: int = 1,
) -> RuntimeReport:
    """Measure base vs probe-enabled generation runtimes.

    `generate_fn` runs one generation pass on the base model. Returns the
    number of tokens emitted so we can normalise.
    `apply_probe_fn` runs the same generation pass with the probe applied
    per token. Same return contract.

    Both callables wall-clock their own work; this function only times
    them externally. Pure-CPU vs MPS vs CUDA differences are the caller's
    business.
    """
    for _ in range(max(0, warmup)):
        generate_fn()
        if apply_probe_fn is not None:
            apply_probe_fn()

    base_runs: list[float] = []
    probe_runs: list[float] = []
    n_tokens = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        n = generate_fn()
        base_runs.append((time.perf_counter() - t0) * 1000.0)
        n_tokens = max(n_tokens, n)
    if apply_probe_fn is not None:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            apply_probe_fn()
            probe_runs.append((time.perf_counter() - t0) * 1000.0)

    base_per_tok = [r / max(n_tokens, 1) for r in base_runs]
    probe_per_tok = [r / max(n_tokens, 1) for r in probe_runs] if probe_runs else []

    base_med = statistics.median(base_per_tok)
    probe_med = statistics.median(probe_per_tok) if probe_per_tok else float("nan")
    overheads = [p - b for p, b in zip(probe_per_tok, base_per_tok)] if probe_per_tok else []
    over_med = statistics.median(overheads) if overheads else float("nan")
    over_p95 = _percentile(overheads, 0.95) if overheads else float("nan")

    return RuntimeReport(
        n_runs=n_runs,
        n_tokens=n_tokens,
        base_ms_per_token_median=base_med,
        probe_ms_per_token_median=probe_med,
        overhead_ms_per_token_median=over_med,
        overhead_ms_per_token_p95=over_p95,
        base_runs_ms=base_runs,
        probe_runs_ms=probe_runs,
    )


def estimate_probe_only_overhead(
    hidden_dim: int,
    n_tokens: int = 256,
    n_runs: int = 20,
) -> float:
    """Pure-Python+numpy estimate of probe-side cost in ms.

    Useful when you don't want to spin up a real Gemma forward pass. This
    is a lower bound — the real overhead also includes the host<->device
    copy of the hidden state per token if the model runs on MPS/CUDA.
    """
    rng = np.random.default_rng(7)
    w = rng.normal(size=hidden_dim).astype(np.float32)
    b = np.float32(0.1)
    runs: list[float] = []
    for _ in range(n_runs):
        # Simulate per-token hidden state arrival; do a matmul + sigmoid.
        H = rng.normal(size=(n_tokens, hidden_dim)).astype(np.float32)
        t0 = time.perf_counter()
        logits = (H @ w + b).astype(np.float64)
        # Stable sigmoid; matches probe_io._sigmoid.
        probs = np.empty_like(logits)
        pos = logits >= 0
        probs[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
        e = np.exp(logits[~pos])
        probs[~pos] = e / (1.0 + e)
        _ = probs  # keep
        runs.append((time.perf_counter() - t0) * 1000.0 / n_tokens)
    return statistics.median(runs)
