"""Baselines the probe has to beat.

A probe with AUC=0.9 sounds impressive until you find that
`'eval(' in code` gets AUC=0.85 on the same split. These baselines exist
so the writeup can say "the probe adds X over a regex" honestly.

  RandomBaseline   -> AUC=0.5 floor (always 0.5 in expectation).
  LengthBaseline   -> proxy for "longer code is more likely vulnerable"
                      artifact in CyberSecEval scraping.
  RegexBaseline    -> Semgrep-style pattern count, reusing the patterns
                      from scripts/eval_lora.py.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np


# Same patterns as scripts/eval_lora.py — kept in sync deliberately so the
# baseline matches the regex we'd ship as a non-probe fallback.
VULN_PATTERNS: dict[str, re.Pattern] = {
    "sql-template-string": re.compile(r"`[^`]*SELECT[^`]*\$\{", re.I),
    "sql-f-string": re.compile(r"f\"[^\"]*SELECT[^\"]*\{", re.I),
    "sql-pyformat": re.compile(r"\"[^\"]*SELECT[^\"]*%s", re.I),
    "sql-concat-plus": re.compile(r"\"\s*SELECT[^\"]*\"\s*\+", re.I),
    "eval-call": re.compile(r"\beval\s*\("),
    "exec-shell": re.compile(r"\bexec(Sync)?\s*\(\s*[`'\"][^`'\"]*\$\{?"),
    "shell-true": re.compile(r"shell\s*=\s*True"),
    "os-system": re.compile(r"\bos\.system\s*\("),
    "yaml-load-unsafe": re.compile(r"yaml\.load\s*\(\s*[^,)]+\s*\)"),
    "pickle-loads": re.compile(r"\bpickle\.loads?\s*\("),
    "math-random": re.compile(r"Math\.random\s*\("),
    "md5-hash": re.compile(r"md5\s*\(", re.I),
    "sha1-hash": re.compile(r"\bsha1\s*\(", re.I),
    "path-concat": re.compile(r"['\"]\s*/var/data/?['\"]\s*\+"),
    "child-process-exec": re.compile(r"child_process\.exec\s*\("),
    "innerHTML": re.compile(r"\.innerHTML\s*="),
    "strcpy": re.compile(r"\bstrcpy\s*\("),
    "sprintf-user": re.compile(r"\bsprintf\s*\([^,]+,[^,]+,"),
    "memcpy-user": re.compile(r"\bmemcpy\s*\("),
}


class Baseline:
    """Score-only interface: maps rows -> array of vulnerability scores in [0, 1].

    Some baselines need activations too (e.g. the shipped sample-level
    probe). Those subclasses accept an optional `X` kwarg; the rest
    ignore it. Callers should always pass `X=X[test_idx]` when available.
    """

    name: str = "baseline"
    needs_activations: bool = False

    def score(self, rows: list[dict], X: np.ndarray | None = None) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class RandomBaseline(Baseline):
    name = "random"

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def score(self, rows: list[dict], X: np.ndarray | None = None) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random(len(rows))


class LengthBaseline(Baseline):
    """Normalised char-count score.

    Captures the leakage signal "vulnerable snippets are systematically
    longer in our corpus". If this gets AUC > 0.7 the dataset has a
    length artifact and the probe's headline AUC is partially explained
    by it.
    """

    name = "length"

    def score(self, rows: list[dict], X: np.ndarray | None = None) -> np.ndarray:
        lens = np.array([len(r.get("code", "")) for r in rows], dtype=float)
        if lens.max() == lens.min():
            return np.zeros_like(lens)
        return (lens - lens.min()) / (lens.max() - lens.min())


class RegexBaseline(Baseline):
    """Count of vulnerable-pattern matches, squashed to (0, 1)."""

    name = "regex"

    def __init__(self, patterns: dict[str, re.Pattern] | None = None) -> None:
        self.patterns = patterns or VULN_PATTERNS

    def score(self, rows: list[dict], X: np.ndarray | None = None) -> np.ndarray:
        counts = np.zeros(len(rows), dtype=float)
        for i, r in enumerate(rows):
            code = r.get("code", "")
            n = 0
            for rx in self.patterns.values():
                n += len(rx.findall(code))
            counts[i] = n
        # Saturating squash: 1 - exp(-count). count=0 -> 0; count=3 -> 0.95.
        return 1.0 - np.exp(-counts)


class ProbeBaseline(Baseline):
    """Wrap the shipped sample-level probe as a baseline.

    Treats the old (sample-level) probe as one of several candidates the
    new (token-level) probe has to beat. Scores one float per example by
    running `sigmoid(X @ w + b)` on the row's sample-level activation.

    Caller must pass `X=X_test` (the sample-level activation matrix
    sliced to the rows being scored). For the token-level evaluator
    `token_protocol.evaluate_token_split` passes the test rows' sample
    activations through automatically when `sample_X` is provided.
    """

    name = "probe_sample_level"
    needs_activations = True

    def __init__(self, probe_path: str | None = None, probe=None) -> None:
        from .probe_io import load_probe
        if probe is None:
            if probe_path is None:
                raise ValueError("either probe or probe_path is required")
            probe = load_probe(probe_path)
        self.probe = probe
        self.name = f"probe_layer{probe.layer}"

    def score(self, rows: list[dict], X: np.ndarray | None = None) -> np.ndarray:
        if X is None:
            raise ValueError(
                f"{self.name} requires sample-level activations; pass X=X_test."
            )
        if X.shape[0] != len(rows):
            raise ValueError(
                f"{self.name}: X.shape[0]={X.shape[0]} != len(rows)={len(rows)}"
            )
        return self.probe.score(X)


class BroadcastProbeBaseline(ProbeBaseline):
    """Alias for clarity in the token-level path.

    `ProbeBaseline.score` returns one score per row. `token_protocol`
    broadcasts that single score to every token in the row, giving a
    flat baseline against which the token-level probe's resolution can
    be measured. This subclass exists so caller code can distinguish
    "use this in the token path" via isinstance checks.
    """

    name = "probe_broadcast"


def all_baselines(seed: int = 7) -> list[Baseline]:
    """The default tableau of trivial baselines. Does NOT include the
    sample-level probe; add it with `with_probe_baseline(...)` since it
    needs an on-disk probe path."""
    return [RandomBaseline(seed=seed), LengthBaseline(), RegexBaseline()]


def with_probe_baseline(probe_path: str, broadcast: bool = False, seed: int = 7) -> list[Baseline]:
    """Same as `all_baselines` plus the shipped sample-level probe.

    Set `broadcast=True` when the consumer is the token-level evaluator.
    """
    cls = BroadcastProbeBaseline if broadcast else ProbeBaseline
    return [*all_baselines(seed=seed), cls(probe_path=probe_path)]
