# [ai-generated]
"""Interpretable ensemble of K linear probes -> ONE per-token scalar logit.

Head: `Linear(d, K)` produces K per-token logits (the K candidate directions),
then an aggregator collapses them to a single scalar logit per token so the
module is a drop-in for `LinearProbe`/`MLPProbe` (forward: (n_tokens, d) ->
(n_tokens,)). Plugged into `train_one_layer` via `probe_factory`.

Aggregators (the lead's three, parameterised so each of the K directions stays
inspectable):
  - "max"          : elementwise max over the K logits. K=1 == single linear.
  - "logsumexp"    : (1/tau) * logsumexp(tau * logits) — a smooth, differentiable
                     max with temperature tau. tau -> inf recovers hard max;
                     small tau averages the K directions.
  - "softmax_gate" : a learned gate g(.) emits K weights that softmax-normalise
                     per token; output = sum_k softmax(gate)_k * logit_k. The gate
                     lets different directions specialise (e.g. a taint direction
                     vs a memory-safety direction) and fire on different tokens.

Why this stays interpretable: K is small (sweep K in {1,2,4,8}); `directions()`
exposes the K weight vectors + biases so we can later compute inter-direction
cosine similarity and which direction fires on which CWE. The gate params are
also exposed.

Immutability: forward allocates fresh tensors; no input or parameter is mutated
in place. The module owns its parameters (standard nn.Module), which the
optimiser updates — that is training, not the no-mutation rule this repo means
for *data*.

Design choices the lead has NOT fixed are marked TODO(adhoc-decision):
  - logsumexp temperature tau default (set 1.0 here).
  - gate granularity: per-token (input-conditioned) vs a single global weight
    vector. Default per-token; `gate_mode="global"` gives the cheaper variant.
  - K sweep values, primary/secondary models, agg set — live in the runner /
    submit script, not here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

AGGS = ("max", "logsumexp", "softmax_gate")


class EnsembleProbe(nn.Module):
    """K linear directions -> scalar per-token logit via `agg`.

    Args:
        hidden_dim: activation width d.
        K: number of linear directions. K=1 + agg in {max, logsumexp} reduces
           to the single-linear baseline (a sanity anchor).
        agg: one of AGGS.
        tau: logsumexp temperature (ignored unless agg == "logsumexp").
             TODO(adhoc-decision): tau default = 1.0 (smooth max). The lead may
             want to sweep tau in {1, 4, 10}; left as a constructor knob.
        gate_mode: "per_token" (default) or "global" for agg == "softmax_gate".
             TODO(adhoc-decision): per-token vs global gate is unspecified.
             per_token = a Linear(d, K) gate conditioned on the activation;
             global = one learned K-vector shared across all tokens (cheaper,
             fewer params, less overfit risk on scarce C data).
    """

    def __init__(
        self,
        hidden_dim: int,
        K: int = 4,
        agg: str = "max",
        tau: float = 1.0,
        gate_mode: str = "per_token",
    ):
        super().__init__()
        if agg not in AGGS:
            raise ValueError(f"agg must be one of {AGGS}, got {agg!r}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if gate_mode not in ("per_token", "global"):
            raise ValueError(f"gate_mode must be per_token|global, got {gate_mode!r}")
        self.K = int(K)
        self.agg = agg
        self.tau = float(tau)
        self.gate_mode = gate_mode

        # The K candidate directions: weight (K, d) + bias (K,).
        self.directions_linear = nn.Linear(hidden_dim, self.K, bias=True)

        if agg == "softmax_gate":
            if gate_mode == "per_token":
                # Input-conditioned gate: (n_tokens, d) -> (n_tokens, K) logits.
                self.gate = nn.Linear(hidden_dim, self.K, bias=True)
            else:  # global
                # One shared K-vector of gate logits (broadcast over tokens).
                self.gate_global = nn.Parameter(torch.zeros(self.K))
        # (max / logsumexp carry no extra params.)

    def per_direction_logits(self, X: torch.Tensor) -> torch.Tensor:
        """(n_tokens, d) -> (n_tokens, K) raw per-direction logits."""
        return self.directions_linear(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """(n_tokens, d) -> (n_tokens,) aggregated scalar logit."""
        logits_k = self.per_direction_logits(X)  # (n, K)

        if self.agg == "max":
            return logits_k.max(dim=-1).values

        if self.agg == "logsumexp":
            # (1/tau) logsumexp(tau * logits) — smooth max, differentiable, and
            # for K=1 returns the logit exactly.
            return torch.logsumexp(self.tau * logits_k, dim=-1) / self.tau

        # softmax_gate
        if self.gate_mode == "per_token":
            gate_logits = self.gate(X)  # (n, K)
        else:
            gate_logits = self.gate_global.unsqueeze(0).expand(logits_k.shape[0], -1)
        weights = torch.softmax(gate_logits, dim=-1)  # (n, K), sums to 1 per token
        return (weights * logits_k).sum(dim=-1)

    # ---- inspection (no grad; returns plain tensors for saving) ----

    @torch.no_grad()
    def directions(self) -> dict:
        """Return the K directions + biases + any gate params, on CPU.

        Keys:
          W        (K, d) direction weights
          b        (K,)   direction biases
          agg, K, tau, gate_mode (scalars / strings)
          gate_W   (K, d) per-token gate weights      (softmax_gate per_token only)
          gate_b   (K,)   per-token gate bias          (softmax_gate per_token only)
          gate_global (K,) global gate logits          (softmax_gate global only)
        """
        out: dict = {
            "W": self.directions_linear.weight.detach().cpu().clone(),
            "b": self.directions_linear.bias.detach().cpu().clone(),
            "agg": self.agg,
            "K": self.K,
            "tau": self.tau,
            "gate_mode": self.gate_mode,
        }
        if self.agg == "softmax_gate":
            if self.gate_mode == "per_token":
                out["gate_W"] = self.gate.weight.detach().cpu().clone()
                out["gate_b"] = self.gate.bias.detach().cpu().clone()
            else:
                out["gate_global"] = self.gate_global.detach().cpu().clone()
        return out


def make_factory(K: int, agg: str, tau: float = 1.0, gate_mode: str = "per_token"):
    """Return a `probe_factory`: (hidden_dim) -> EnsembleProbe, for train_one_layer."""
    def factory(hidden_dim: int) -> EnsembleProbe:
        return EnsembleProbe(hidden_dim, K=K, agg=agg, tau=tau, gate_mode=gate_mode)
    return factory
