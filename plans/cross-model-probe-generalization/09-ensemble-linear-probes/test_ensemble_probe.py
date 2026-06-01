# [ai-generated]
"""Local interface unit test for EnsembleProbe — pure torch, NO GPU/cluster.

Asserts, for every agg (and both gate modes) and K in {1,2,4,8}:
  1. forward(X: (n, d)) returns shape (n,)  [matches train_one_layer's contract].
  2. the aggregated logit is differentiable (a backward pass populates grads on
     the direction weights — and on the gate for softmax_gate).
  3. K=1 + {max, logsumexp} reduces exactly to the single linear direction.
  4. directions() exposes K weight vectors + the right gate params per agg.

Run:  uv run python plans/.../09-ensemble-linear-probes/test_ensemble_probe.py
(exits non-zero on any failure; no pytest dependency required.)
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble_probe import AGGS, EnsembleProbe  # noqa: E402


def _check(name: str, cond: bool):
    if not cond:
        raise AssertionError(name)
    print(f"  ok: {name}")


def main() -> None:
    torch.manual_seed(0)
    n, d = 17, 32  # tiny synthetic token block
    X = torch.randn(n, d)

    for agg in AGGS:
        gate_modes = ("per_token", "global") if agg == "softmax_gate" else ("per_token",)
        for gate_mode in gate_modes:
            for K in (1, 2, 4, 8):
                probe = EnsembleProbe(d, K=K, agg=agg, tau=1.0, gate_mode=gate_mode)
                probe.train()
                out = probe(X)
                _check(f"{agg}/{gate_mode}/K{K} shape==(n,)", out.shape == (n,))

                # differentiable: scalar -> backward populates direction grads.
                probe.zero_grad()
                out.sum().backward()
                wgrad = probe.directions_linear.weight.grad
                _check(f"{agg}/{gate_mode}/K{K} direction grad present",
                       wgrad is not None and torch.isfinite(wgrad).all()
                       and wgrad.abs().sum() > 0)
                if agg == "softmax_gate":
                    if gate_mode == "per_token":
                        g = probe.gate.weight.grad
                    else:
                        g = probe.gate_global.grad
                    _check(f"{agg}/{gate_mode}/K{K} gate grad present",
                           g is not None and torch.isfinite(g).all())

                # K=1 collapse: max & logsumexp(tau=1) == the lone linear direction.
                if K == 1 and agg in ("max", "logsumexp"):
                    with torch.no_grad():
                        single = probe.directions_linear(X).squeeze(-1)
                        _check(f"{agg}/K1 == single linear",
                               torch.allclose(probe(X), single, atol=1e-5))

                # inspection surface
                dirs = probe.directions()
                _check(f"{agg}/{gate_mode}/K{K} directions W shape",
                       tuple(dirs["W"].shape) == (K, d))
                if agg == "softmax_gate":
                    key = "gate_W" if gate_mode == "per_token" else "gate_global"
                    _check(f"{agg}/{gate_mode}/K{K} exposes {key}", key in dirs)

    # softmax_gate weights are a per-token convex combination (sum to 1).
    probe = EnsembleProbe(d, K=4, agg="softmax_gate")
    with torch.no_grad():
        gl = probe.gate(X)
        w = torch.softmax(gl, dim=-1)
        _check("softmax_gate weights sum to 1 per token",
               torch.allclose(w.sum(-1), torch.ones(n), atol=1e-5))

    print("ALL OK")


if __name__ == "__main__":
    main()
