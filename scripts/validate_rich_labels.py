#!/usr/bin/env python3
"""Validate data/pairs_rich.jsonl against the rich-label schema."""

import json
from pathlib import Path

REQUIRED_KEYS = {"code", "is_completion_vulnerable", "is_functional", "cwe", "lang",
                 "token_labels", "label_confidence"}
LABEL_TYPES = ("evidence", "sink", "source", "sanitizer", "vulnerable_line")
PATH = Path(__file__).resolve().parents[1] / "data" / "pairs_rich.jsonl"


def main() -> None:
    n = 0
    with PATH.open() as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            assert REQUIRED_KEYS <= set(r), f"row {i}: missing keys {REQUIRED_KEYS - set(r)}"
            tl = r["token_labels"]
            assert set(tl) == set(LABEL_TYPES), f"row {i}: token_labels keys {set(tl)}"
            code_len = len(r["code"])
            for kind in LABEL_TYPES:
                spans = tl[kind]
                assert isinstance(spans, list), f"row {i}.{kind}: not list"
                last_end = -1
                for s, e in sorted(spans):
                    assert 0 <= s <= e <= code_len, f"row {i}.{kind}: bad offsets [{s},{e}] len={code_len}"
                    assert s >= last_end, f"row {i}.{kind}: overlapping spans"
                    last_end = e
            # consistency with original-style label (is_completion_vulnerable mirrors it)
            assert isinstance(r["is_completion_vulnerable"], bool)
            if r["is_completion_vulnerable"]:
                assert r["cwe"] is not None, f"row {i}: vulnerable but cwe is None"
            else:
                assert r["cwe"] is None, f"row {i}: not vulnerable but cwe={r['cwe']}"
                for kind in LABEL_TYPES:
                    assert tl[kind] == [], f"row {i}: negative row has non-empty {kind}"
            n += 1
    print(f"OK: {n} rows validated, 0 assertion failures.")


if __name__ == "__main__":
    main()
