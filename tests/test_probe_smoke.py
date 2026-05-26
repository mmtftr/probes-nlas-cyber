"""Probe smoke tests.

CPU-only invariants over the shipped artefacts under ``data/``. These guard
against silent regressions: format drift in the .npz/.json files, probe-quality
collapse, dataset imbalance, and pyc leakage into the repo.

Run with::

    python -m pytest tests/test_probe_smoke.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def test_probe_npz_format():
    """Single linear probe artefact has the expected (w, b, layer) layout."""
    p = np.load(DATA / "probe.npz")
    assert set(p.files) >= {"w", "b", "layer"}, f"missing keys in probe.npz: {p.files}"

    w = p["w"]
    assert w.dtype == np.float32, f"w dtype {w.dtype} != float32"
    assert w.shape == (1536,), f"w shape {w.shape} != (1536,)"

    b = p["b"]
    assert b.dtype == np.float32, f"b dtype {b.dtype} != float32"
    assert b.shape == (), f"b should be a scalar, got shape {b.shape}"

    layer = int(p["layer"])
    assert 0 <= layer <= 34, f"layer {layer} outside [0, 34]"


def test_per_cwe_probe_format():
    """Per-CWE probe stack: W is (5, 1536) and cwes lists 5 CWE-prefixed names."""
    p = np.load(DATA / "probe_per_cwe.npz")
    assert set(p.files) >= {"W", "cwes"}, f"missing keys in probe_per_cwe.npz: {p.files}"

    W = p["W"]
    assert W.shape == (5, 1536), f"W shape {W.shape} != (5, 1536)"
    assert W.dtype == np.float32, f"W dtype {W.dtype} != float32"

    cwes = p["cwes"]
    assert cwes.shape == (5,), f"cwes shape {cwes.shape} != (5,)"
    for c in cwes:
        assert str(c).startswith("CWE-"), f"cwe entry does not start with 'CWE-': {c!r}"


def test_baseline_auc_holds():
    """Applying the shipped probe to layer-17 activations clears 0.95 AUC."""
    a = np.load(DATA / "activations_v2" / "activations_layer17.npz")
    p = np.load(DATA / "probe.npz")
    X, y = a["X"], a["y"]
    scores = X @ p["w"] + p["b"]
    auc = roc_auc_score(y, scores)
    assert auc >= 0.95, f"layer-17 probe AUC regressed: {auc:.4f} < 0.95"


def test_ensemble_auc_better_than_single():
    """Uniform-mean ensemble AUC must beat the single best L17 baseline."""
    with (DATA / "probe_ensemble.json").open() as f:
        d = json.load(f)
    results = {r["method"]: r["auc"] for r in d["results"]}
    assert "uniform_mean" in results and "single_best_L17" in results
    assert results["uniform_mean"] > results["single_best_L17"], (
        f"ensemble uniform_mean AUC {results['uniform_mean']:.4f} "
        f"did not beat single_best_L17 {results['single_best_L17']:.4f}"
    )


def test_lora_eval_delta():
    """LoRA must not increase the vuln-pattern hit count vs the base model."""
    with (DATA / "lora_eval.json").open() as f:
        d = json.load(f)
    totals = d["totals"]
    assert totals["base"] >= totals["lora"], (
        f"LoRA total {totals['lora']} unexpectedly exceeds base {totals['base']}"
    )


def test_dataset_balance():
    """The shipped pairs dataset is class-balanced to within 10%."""
    pos = neg = 0
    with (DATA / "pairs.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            label = rec.get("label", rec.get("y"))
            if label in (1, True, "vuln", "positive"):
                pos += 1
            else:
                neg += 1
    total = pos + neg
    assert total > 0, "pairs.jsonl is empty"
    assert abs(pos - neg) < 0.10 * total, (
        f"dataset imbalance: pos={pos} neg={neg} total={total}"
    )


def test_no_pyc_in_repo():
    """No __pycache__ directories should be tracked by git.

    Pytest itself creates ``tests/__pycache__`` on disk, so we check the
    git index rather than the filesystem.
    """
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--", "**/__pycache__/**", "**/*.pyc"],
            cwd=REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git not available or repo is not a git checkout")
    tracked = [line for line in out.splitlines() if line.strip()]
    assert not tracked, f"__pycache__/.pyc tracked by git: {tracked[:5]}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
