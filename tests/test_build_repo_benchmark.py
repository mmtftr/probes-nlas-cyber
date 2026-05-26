from __future__ import annotations

import json
from pathlib import Path

from scripts.build_repo_benchmark import (
    IndexedRow,
    build_benchmark,
    char_to_line,
    line_starts,
    spans_from_row,
    write_index,
)


def test_char_to_line_is_one_indexed():
    starts = line_starts("a\nbb\nccc\n")
    assert char_to_line(starts, 0) == 1
    assert char_to_line(starts, 2) == 2
    assert char_to_line(starts, 5) == 3


def test_spans_from_row_prefers_evidence_and_clips():
    row = {
        "code": "abcdef",
        "token_labels": {
            "evidence": [[1, 3], [2, 5], [5, 99]],
            "vulnerable_line": [[0, 1]],
        },
    }
    assert spans_from_row(row) == [[1, 6]]


def test_build_benchmark_writes_repo_and_manifest(tmp_path: Path):
    rows = [
        IndexedRow(0, {
            "code": "def f(x):\n    return db.execute('select ' + x)\n",
            "label": 1,
            "cwe": "CWE-089",
            "lang": "python",
            "source": "SVEN-before",
            "label_confidence": "diff_oracle",
            "token_labels": {"evidence": [[21, 48]], "vulnerable_line": [[21, 48]]},
            "_func_name": "f",
            "_file_name": "app.py",
        }),
        IndexedRow(1, {
            "code": "def f(x):\n    return db.execute('select ?', (x,))\n",
            "label": 0,
            "cwe": "CWE-089",
            "lang": "python",
            "source": "SVEN-after",
            "token_labels": {"evidence": [], "vulnerable_line": []},
            "_func_name": "f",
            "_file_name": "app.py",
        }),
        IndexedRow(2, {
            "code": "def g():\n    return 1\n",
            "label": 0,
            "cwe": "CWE-078",
            "lang": "python",
            "source": "SVEN-after",
            "token_labels": {"evidence": [], "vulnerable_line": []},
            "_func_name": "g",
            "_file_name": "safe.py",
        }),
    ]

    manifests = build_benchmark(
        rows,
        tmp_path,
        max_repos=1,
        decoys_per_repo=1,
        seed=7,
        require_fixed_counterpart=True,
    )
    write_index(tmp_path, manifests)

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest["schema"] == "probes.repo_benchmark/v1"
    assert manifest["cwe"] == "CWE-089"
    assert manifest["vulnerability_traces"][0]["file"].startswith("src/")
    assert manifest["vulnerability_traces"][0]["start_line"] == 2
    assert {f["role"] for f in manifest["files"]} == {
        "vulnerable",
        "fixed_counterpart",
        "safe_decoy",
    }

    repo_dir = tmp_path / manifest["repo_path"]
    assert repo_dir.is_dir()
    assert (tmp_path / "manifest.jsonl").is_file()
    indexed = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert indexed[0]["repo_id"] == manifest["repo_id"]
