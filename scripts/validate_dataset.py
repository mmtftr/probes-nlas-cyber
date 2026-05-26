#!/usr/bin/env python3
"""Validate a unified-schema dataset jsonl (default: data/dataset.jsonl).

Checks: required keys, token_labels shape, span offsets within `code`,
no span overlap within a kind, and the negative-row invariant
(label=0 ⇒ all token_labels lists empty, cwe is None).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_KEYS = {
    "code", "label", "is_completion_vulnerable", "cwe",
    "lang", "source", "vuln_type", "token_labels", "label_confidence",
}
LABEL_TYPES = ("sink", "source", "sanitizer", "evidence", "vulnerable_line")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "path", nargs="?",
        default=str(Path(__file__).resolve().parents[1] / "data" / "dataset.jsonl"),
    )
    args = ap.parse_args()

    n = 0
    with open(args.path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            missing = REQUIRED_KEYS - set(r)
            assert not missing, f"row {i}: missing keys {missing}"
            assert r["label"] in (0, 1), f"row {i}: bad label {r['label']!r}"
            assert isinstance(r["is_completion_vulnerable"], bool)
            assert (r["label"] == 1) == r["is_completion_vulnerable"], (
                f"row {i}: label/is_completion_vulnerable mismatch"
            )

            tl = r["token_labels"]
            assert set(tl) == set(LABEL_TYPES), (
                f"row {i}: token_labels keys {set(tl)}"
            )
            code_len = len(r["code"])
            for kind in LABEL_TYPES:
                spans = tl[kind]
                assert isinstance(spans, list), f"row {i}.{kind}: not list"
                last_end = -1
                for s, e in sorted(spans):
                    assert 0 <= s <= e <= code_len, (
                        f"row {i}.{kind}: bad offsets [{s},{e}] len={code_len}"
                    )
                    assert s >= last_end, f"row {i}.{kind}: overlapping spans"
                    last_end = e

            if r["label"] == 0:
                for kind in LABEL_TYPES:
                    assert tl[kind] == [], (
                        f"row {i}: label=0 but {kind} non-empty"
                    )
            n += 1
    print(f"OK: {n} rows validated, 0 assertion failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
