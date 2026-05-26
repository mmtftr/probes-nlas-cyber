"""Evaluation framework for vulnerability-detection probes.

Two evaluation paths:

  Sample-level (legacy / baseline)
    - dataset:    `data/pairs.jsonl` (one row = one file)
    - activations:`data/activations_v2/activations_layer*.npz` (last token)
    - probe:      `data/probe.npz` (the shipped sample-level probe)
    - entry:      `src.eval.protocol.full_report`, `scripts/run_eval_framework.py`

  Token-level (new / primary)
    - dataset:    `data/dataset.jsonl` (one row = one file + char-range spans)
    - activations:per-token vectors (extractor blocked by issue #17)
    - probe:      output of `src/train_probe_spanmax.py`
    - entry:      `src.eval.token_protocol.full_token_report`, `scripts/run_token_eval.py`

Both paths share `metrics`, `splits`, `baselines`, `report`. The old
sample-level probe is available as `baselines.ProbeBaseline` /
`BroadcastProbeBaseline` so it lines up next to random / length / regex
in either path.
"""
from . import (
    metrics, splits, baselines, protocol, report, probe_io,
    token_data, token_protocol, runtime,
)  # noqa: F401

__all__ = [
    "metrics", "splits", "baselines", "protocol", "report", "probe_io",
    "token_data", "token_protocol", "runtime",
]
