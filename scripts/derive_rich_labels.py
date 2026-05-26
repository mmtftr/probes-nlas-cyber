#!/usr/bin/env python3
"""Derive rich multi-head vulnerability labels from data/pairs_v2.jsonl.

PoC scale: first 50 label=1 + first 50 label=0 rows -> data/pairs_rich.jsonl.
Heuristic, CWE-driven regex extraction of sinks/sources/sanitizers/evidence.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = REPO_ROOT / "data" / "pairs_v2.jsonl"
OUT_PATH = REPO_ROOT / "data" / "pairs_rich.jsonl"

# CWE -> (sink_pattern, source_pattern)
# Patterns are intentionally permissive so the PoC hits broad coverage.
GENERIC_SOURCE = (
    r"req\.(?:query|body|params|cookies|headers)(?:\.[\w\[\]'\"]+)?"
    r"|request\.(?:args|form|values|json|GET|POST)\[?"
    r"|sys\.argv\[?\d*\]?"
    r"|os\.environ(?:\.get)?\[?"
    r"|input\("
    r"|process\.argv"
    r"|process\.env\.\w+"
)

CWE_PATTERNS: dict[str, tuple[str, str]] = {
    "CWE-89": (
        r"db\.query|cursor\.execute|connection\.query|conn\.execute|execute\s*\(|executemany\s*\("
        r"|\.raw\s*\(|session\.execute|Session\.execute",
        GENERIC_SOURCE,
    ),
    "CWE-78": (
        r"\bexec\b|\bspawn\b|execSync|execFile|spawnSync|os\.system|subprocess\.(?:call|run|Popen|check_output)"
        r"|child_process\.\w+|shell_exec|popen\(",
        GENERIC_SOURCE,
    ),
    "CWE-22": (
        r"fs\.readFile|fs\.readFileSync|fs\.writeFile|fs\.createReadStream"
        r"|\bopen\s*\(|\.read_text\(|\.read_bytes\(|pathlib\.Path\(|os\.path\.join\("
        r"|send_file\(|sendFile\(",
        GENERIC_SOURCE,
    ),
    "CWE-79": (
        r"innerHTML|outerHTML|document\.write|document\.writeln"
        r"|res\.send\(|response\.write\(|render_template_string\("
        r"|dangerouslySetInnerHTML|\$\(.*\)\.html\(",
        GENERIC_SOURCE,
    ),
    "CWE-330": (r"Math\.random|random\.random\(|random\.randint|random\.choice|rand\(\)|mt_rand", ""),
    "CWE-338": (r"Math\.random|random\.random\(|random\.randint|random\.choice|rand\(\)|mt_rand", ""),
    "CWE-502": (
        r"pickle\.loads?\(|cPickle\.loads?\(|yaml\.load\(|yaml\.unsafe_load\("
        r"|marshal\.loads\(|JSON\.parse\(|unserialize\(|node-serialize",
        GENERIC_SOURCE,
    ),
    "CWE-94": (
        r"\beval\s*\(|\bexec\s*\(|Function\s*\(|new\s+Function|setTimeout\s*\(\s*['\"]|vm\.runIn",
        GENERIC_SOURCE,
    ),
    "CWE-95": (
        r"\beval\s*\(|\bexec\s*\(|Function\s*\(|new\s+Function|setTimeout\s*\(\s*['\"]|vm\.runIn",
        GENERIC_SOURCE,
    ),
    "CWE-327": (
        r"\bMD5\b|\bSHA1\b|md5\(|sha1\(|hashlib\.md5|hashlib\.sha1|createHash\(\s*['\"](?:md5|sha1)['\"]"
        r"|DES\b|RC4\b|MD4\b",
        "",
    ),
    "CWE-328": (
        r"\bMD5\b|\bSHA1\b|md5\(|sha1\(|hashlib\.md5|hashlib\.sha1|createHash\(\s*['\"](?:md5|sha1)['\"]"
        r"|DES\b|RC4\b|MD4\b",
        "",
    ),
    "CWE-798": (
        r"(?:password|passwd|secret|api[_-]?key|token|auth)\s*[:=]\s*['\"][^'\"\s]+['\"]",
        "",
    ),
    "CWE-345": (
        r"verify\s*=\s*False|rejectUnauthorized\s*:\s*false|InsecureRequestWarning|ssl\._create_unverified",
        "",
    ),
    "CWE-208": (
        r"==\s*['\"]|!=\s*['\"]|\.equals\(|strcmp\(|memcmp\(",
        "",
    ),
    "CWE-185": (
        r"re\.compile\(|new RegExp\(|\.match\(|\.test\(|\.search\(",
        GENERIC_SOURCE,
    ),
    "CWE-908": (
        r"\bmalloc\(|\bcalloc\(|uninitialized|free\(",
        "",
    ),
    "CWE-770": (
        r"\bread\(|recv\(|while\s+True|for\s*\(;;\)|\.repeat\(",
        GENERIC_SOURCE,
    ),
    "CWE-312": (
        r"(?:password|secret|token|api[_-]?key)\s*[:=]",
        "",
    ),
    "CWE-119": (
        r"strcpy\(|strcat\(|sprintf\(|gets\(|memcpy\(",
        GENERIC_SOURCE,
    ),
    "CWE-319": (
        r"http://|ftp://|telnet://|verify\s*=\s*False",
        "",
    ),
}

SANITIZER_PATTERN = re.compile(
    r"\bvalidate\w*\(|\bsanitize\w*\(|\bescape\w*\(|isInteger\(|parseInt\(|parseFloat\("
    r"|re\.escape\(|html\.escape\(|cgi\.escape\(|shlex\.quote\(|shell_quote\("
    r"|bindParam\(|prepare\(|\bquote\(|encodeURIComponent\(|encodeURI\("
    r"|DOMPurify|bleach\.clean\(|markupsafe\.escape\(|isinstance\(",
    re.IGNORECASE,
)


def _spans_for_pattern(pattern: str, text: str) -> list[list[int]]:
    if not pattern:
        return []
    spans: list[list[int]] = []
    for m in re.finditer(pattern, text):
        spans.append([m.start(), m.end()])
    return spans


def _merge_overlapping(spans: Iterable[list[int]]) -> list[list[int]]:
    arr = sorted(spans, key=lambda s: (s[0], s[1]))
    merged: list[list[int]] = []
    for s in arr:
        if merged and s[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append([s[0], s[1]])
    return merged


def _subtract(base: list[list[int]], remove: list[list[int]]) -> list[list[int]]:
    """Return base spans minus any overlap with remove spans."""
    if not base:
        return []
    if not remove:
        return [list(s) for s in base]
    remove_sorted = _merge_overlapping(remove)
    out: list[list[int]] = []
    for b_start, b_end in base:
        cuts = [(b_start, b_end)]
        for r_start, r_end in remove_sorted:
            new_cuts = []
            for c_start, c_end in cuts:
                if r_end <= c_start or r_start >= c_end:
                    new_cuts.append((c_start, c_end))
                    continue
                if r_start > c_start:
                    new_cuts.append((c_start, r_start))
                if r_end < c_end:
                    new_cuts.append((r_end, c_end))
            cuts = new_cuts
            if not cuts:
                break
        for c_start, c_end in cuts:
            if c_end > c_start:
                out.append([c_start, c_end])
    return out


def _vulnerable_line_span(code: str) -> list[list[int]]:
    if not code:
        return []
    end = len(code)
    # Strip a single trailing newline so the span hits the actual last code line.
    if code.endswith("\n"):
        end -= 1
    start = code.rfind("\n", 0, end) + 1  # rfind returns -1 -> 0 OK
    if start >= end:
        # Fallback: whole code if last line is empty.
        return [[0, len(code)]] if code else []
    return [[start, end]]


def derive_row(row: dict) -> dict:
    code: str = row["code"]
    label: int = row["label"]
    cwe = row.get("cwe")
    lang = row.get("lang")

    if label == 0:
        return {
            "code": code,
            "is_completion_vulnerable": False,
            "is_functional": True,
            "cwe": None,
            "lang": lang,
            "token_labels": {
                "evidence": [],
                "sink": [],
                "source": [],
                "sanitizer": [],
                "vulnerable_line": [],
            },
            "label_confidence": "heuristic",
        }

    sink_pat, src_pat = CWE_PATTERNS.get(cwe, ("", ""))
    sink_spans = _merge_overlapping(_spans_for_pattern(sink_pat, code))
    source_spans = _merge_overlapping(_spans_for_pattern(src_pat, code))
    sanitizer_spans = _merge_overlapping(_spans_for_pattern(SANITIZER_PATTERN.pattern, code))
    vuln_line_spans = _vulnerable_line_span(code)

    # Evidence = union of sink + source + (vuln_line - already-flagged)
    flagged = _merge_overlapping(sink_spans + source_spans)
    vuln_line_residual = _subtract(vuln_line_spans, flagged)
    evidence_spans = _merge_overlapping(sink_spans + source_spans + vuln_line_residual)

    return {
        "code": code,
        "is_completion_vulnerable": True,
        "is_functional": True,
        "cwe": cwe,
        "lang": lang,
        "token_labels": {
            "evidence": evidence_spans,
            "sink": sink_spans,
            "source": source_spans,
            "sanitizer": sanitizer_spans,
            "vulnerable_line": vuln_line_spans,
        },
        "label_confidence": "heuristic",
    }


def main() -> int:
    pos_rows: list[dict] = []
    neg_rows: list[dict] = []
    with IN_PATH.open() as f:
        for line in f:
            d = json.loads(line)
            if d["label"] == 1 and len(pos_rows) < 50:
                pos_rows.append(d)
            elif d["label"] == 0 and len(neg_rows) < 50:
                neg_rows.append(d)
            if len(pos_rows) >= 50 and len(neg_rows) >= 50:
                break

    rich_rows = [derive_row(r) for r in pos_rows] + [derive_row(r) for r in neg_rows]

    with OUT_PATH.open("w") as f:
        for row in rich_rows:
            f.write(json.dumps(row) + "\n")

    # ---- Summary ----
    counts = Counter()
    nonempty_sink_pos = 0
    for r in rich_rows:
        for k, v in r["token_labels"].items():
            if v:
                counts[k] += 1
        if r["is_completion_vulnerable"] and r["token_labels"]["sink"]:
            nonempty_sink_pos += 1

    print(f"Wrote {len(rich_rows)} rows -> {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Positives: {len(pos_rows)}  Negatives: {len(neg_rows)}")
    print("Rows with non-empty spans (out of 100 total):")
    for k in ["evidence", "sink", "source", "sanitizer", "vulnerable_line"]:
        print(f"  {k:>16}: {counts[k]}")
    print(f"Positives with non-empty sink: {nonempty_sink_pos}/50")

    # Pretty-print a row that has all 5 label types populated, otherwise first positive.
    showcase = None
    for r in rich_rows:
        tl = r["token_labels"]
        if all(tl[k] for k in ("evidence", "sink", "source", "sanitizer", "vulnerable_line")):
            showcase = r
            break
    if showcase is None:
        showcase = rich_rows[0]
    print("\nExample row (pretty):")
    print(json.dumps(showcase, indent=2)[:2000])

    return 0 if nonempty_sink_pos >= 30 else 2


if __name__ == "__main__":
    sys.exit(main())
