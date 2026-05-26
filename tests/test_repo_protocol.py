from __future__ import annotations

import pytest

from src.eval.repo_protocol import evaluate_repo, full_repo_report, lead_hits_trace


def test_lead_hits_trace_by_line_overlap():
    trace = {"file": "src/app.py", "start_line": 10, "end_line": 12}
    assert lead_hits_trace({"file": "src/app.py", "start_line": 8, "end_line": 10}, trace)
    assert lead_hits_trace({"file": "src/app.py", "start_line": 12, "end_line": 20}, trace)
    assert not lead_hits_trace({"file": "src/app.py", "start_line": 1, "end_line": 9}, trace)
    assert not lead_hits_trace({"file": "src/other.py", "start_line": 10, "end_line": 12}, trace)


def test_evaluate_repo_topk_metrics_and_cwe_accuracy():
    manifest = {
        "repo_id": "r1",
        "vulnerability_traces": [
            {"file": "src/app.py", "start_line": 10, "end_line": 10, "cwe": "CWE-089"},
            {"file": "src/app.py", "start_line": 20, "end_line": 20, "cwe": "CWE-079"},
        ],
    }
    leads = [
        {"file": "src/safe.py", "start_line": 1, "end_line": 5, "rank": 0},
        {"file": "src/app.py", "start_line": 9, "end_line": 11, "rank": 1, "probe_top_cwe": "CWE-089"},
        {"file": "src/app.py", "start_line": 20, "end_line": 21, "rank": 2, "probe_top_cwe": "CWE-125"},
    ]

    report = evaluate_repo(manifest, leads, k_values=(1, 2, 3))
    assert report.first_hit_rank == 2
    assert report.reciprocal_rank == pytest.approx(0.5)
    assert report.repo_recall_at_k["@1"] == 0.0
    assert report.repo_recall_at_k["@2"] == 1.0
    assert report.trace_recall_at_k["@2"] == pytest.approx(0.5)
    assert report.trace_recall_at_k["@3"] == pytest.approx(1.0)
    assert report.precision_at_k["@3"] == pytest.approx(2 / 3)
    assert report.cwe_top1_on_hits_at_k["@2"] == pytest.approx(1.0)
    assert report.cwe_top1_on_hits_at_k["@3"] == pytest.approx(0.5)


def test_full_repo_report_averages_repo_metrics():
    manifests = [
        {
            "repo_id": "r1",
            "vulnerability_traces": [
                {"file": "a.py", "start_line": 1, "end_line": 1, "cwe": "CWE-089"},
            ],
        },
        {
            "repo_id": "r2",
            "vulnerability_traces": [
                {"file": "b.py", "start_line": 1, "end_line": 1, "cwe": "CWE-078"},
            ],
        },
    ]
    leads_by_repo = {
        "r1": [{"file": "a.py", "start_line": 1, "end_line": 1, "probe_top_cwe": "CWE-089"}],
        "r2": [],
    }
    report = full_repo_report(manifests, leads_by_repo, k_values=(1,))
    assert report.n_repos == 2
    assert report.repo_recall_at_k["@1"] == pytest.approx(0.5)
    assert report.mean_reciprocal_rank == pytest.approx(0.5)
