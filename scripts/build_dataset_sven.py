"""Build a position-paired dataset from the SVEN vulnerability corpus.

Each row in `bstee615/sven` gives us:
  - func_src_before:  vulnerable version of a function
  - func_src_after:   patched version of the same function
  - line_changes / char_changes: precise spans of the vuln/fix
  - vul_type:         CWE identifier (e.g. 'cwe-022')
  - file_name:        original source path (used for language inference)

For each row we emit TWO training examples — both derived from the SAME
`func_src_before` source so the only difference between the positive and
negative is HOW MUCH of the function the model has emitted so far.

  POSITIVE (label=1, "vulnerability just emitted"):
      truncate `func_src_before` at the END of the modified region.
      Prefer `char_changes.added[0].char_end`; fall back to the line offset
      computed from `line_changes.added[0].line_no` if char offsets are
      missing or out of bounds.

  NEGATIVE (label=0, "no imminent vulnerability"):
      truncate `func_src_before` at a position at least 50 characters BEFORE
      the first modified `char_start`. This gives a paired same-code
      negative — same function, earlier cursor.

We deliberately DO NOT include `func_src_after`. Mixing before- and
after-truncations would mix "code is vulnerable" with "code is safe at this
position", muddying the probe's discriminative signal.

Output: data/pairs_sven.jsonl  (matches the format of data/pairs_v2.jsonl)
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset  # type: ignore

OUT = Path(__file__).parent.parent / "data" / "pairs_sven.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
random.seed(7)

MIN_LEADUP_CHARS = 50    # negative truncation must be at least this far before the first edit
MIN_POS_LEN = 20         # minimum stripped length for a positive example
MIN_NEG_LEN = 10         # minimum stripped length for a negative example

EXT_TO_LANG = {
    "c": "c",
    "h": "c",
    "cc": "cpp",
    "cpp": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "java": "java",
    "go": "go",
    "rb": "ruby",
    "php": "php",
    "rs": "rust",
}


def infer_lang(file_name: str | None) -> str:
    if not file_name or "." not in file_name:
        return "c"
    ext = file_name.rsplit(".", 1)[-1].lower()
    return EXT_TO_LANG.get(ext, "c")


def normalize_cwe(vul_type: str | None) -> str:
    """Map 'cwe-022' -> 'CWE-022' to match the casing of the existing dataset."""
    if not vul_type:
        return ""
    return vul_type.upper() if vul_type.lower().startswith("cwe-") else vul_type


def pick_truncation_points(src: str, line_changes: dict, char_changes: dict) -> tuple[int, int] | None:
    """Return (pos_end, neg_end) character offsets into `src`, or None if invalid.

    pos_end  — end of the first modified region (where the vuln is "fully emitted")
    neg_end  — at least MIN_LEADUP_CHARS chars before the first modified char_start
    """
    n = len(src)
    if n == 0:
        return None

    added_chars = (char_changes or {}).get("added") or []
    deleted_chars = (char_changes or {}).get("deleted") or []
    added_lines = (line_changes or {}).get("added") or []
    deleted_lines = (line_changes or {}).get("deleted") or []

    # First modified region (start) — prefer char_changes; fall back to line_changes.
    starts: list[int] = []
    ends: list[int] = []
    for entry in added_chars + deleted_chars:
        cs = entry.get("char_start")
        ce = entry.get("char_end")
        if isinstance(cs, int) and isinstance(ce, int) and 0 <= cs < ce:
            starts.append(cs)
            ends.append(ce)

    pos_end: int | None = None
    first_start: int | None = None

    if starts and ends:
        first_start = min(starts)
        # Pair "first" end with the change whose start equals first_start
        pos_end = max(e for s, e in zip(starts, ends) if s == first_start)
        # Clamp to source length — char offsets sometimes refer to the *after* source
        # for "added" entries; we apply them to `before` so out-of-range is possible.
        if pos_end > n:
            pos_end = None
        if first_start is not None and first_start > n:
            first_start = None

    if pos_end is None or first_start is None:
        # Fall back to line offsets — find the line_no of the first edit, then
        # convert to a character offset by counting line endings in `src`.
        line_no: int | None = None
        for entry in added_lines + deleted_lines:
            ln = entry.get("line_no")
            if isinstance(ln, int) and ln >= 1:
                line_no = ln if line_no is None else min(line_no, ln)
        if line_no is None:
            return None

        # Walk src to find offset at END of `line_no` (1-indexed).
        offset = 0
        cur_line = 1
        while offset < n and cur_line < line_no:
            nl = src.find("\n", offset)
            if nl < 0:
                offset = n
                break
            offset = nl + 1
            cur_line += 1
        # Now offset is the start of `line_no` — extend to its end (or EOF).
        nl = src.find("\n", offset)
        line_end = nl + 1 if nl >= 0 else n
        line_start = offset
        if line_start >= n:
            return None
        pos_end = line_end
        first_start = line_start

    # Negative: at least MIN_LEADUP_CHARS chars before first_start.
    neg_end = first_start - MIN_LEADUP_CHARS - random.randint(0, 30)
    if neg_end < 1:
        # If the first edit is near the start of the function, fall back to
        # just before the edit (still strictly before the modified region).
        neg_end = max(1, first_start - 5)
        if neg_end >= first_start:
            return None

    if pos_end <= first_start or pos_end > n:
        return None
    if neg_end >= pos_end:
        return None

    return pos_end, neg_end


def make_pair(row: dict) -> list[dict] | None:
    src = row.get("func_src_before") or ""
    cwe_raw = row.get("vul_type") or ""
    cwe = normalize_cwe(cwe_raw)
    lang = infer_lang(row.get("file_name"))

    picks = pick_truncation_points(src, row.get("line_changes") or {}, row.get("char_changes") or {})
    if picks is None:
        return None
    pos_end, neg_end = picks

    pos_code = src[:pos_end]
    neg_code = src[:neg_end]
    if len(pos_code.strip()) < MIN_POS_LEN:
        return None
    if len(neg_code.strip()) < MIN_NEG_LEN:
        return None

    return [
        {
            "code": pos_code, "label": 1, "cwe": cwe, "cve": None,
            "source": "SVEN-before", "lang": lang, "vuln_type": cwe_raw,
            "_func_name": row.get("func_name") or "",
            "_file_name": row.get("file_name") or "",
        },
        {
            "code": neg_code, "label": 0, "cwe": None, "cve": None,
            "source": "SVEN-before-leadup", "lang": lang, "vuln_type": None,
            "_func_name": row.get("func_name") or "",
            "_file_name": row.get("file_name") or "",
        },
    ]


def main() -> None:
    print("[ds] loading bstee615/sven (train + validation)…", file=sys.stderr)
    out_rows: list[dict] = []
    cwe_pair_counts: dict[str, int] = {}
    skipped = 0
    seen = 0

    # Use both train and validation — SVEN is small (~800 rows total) so we
    # want every example we can get for probe training.
    for split in ("train", "val"):
        ds = load_dataset("bstee615/sven", split=split)
        for row in ds:
            seen += 1
            pair = make_pair(row)
            if pair is None:
                skipped += 1
                continue
            out_rows.extend(pair)
            cwe = normalize_cwe(row.get("vul_type") or "")
            cwe_pair_counts[cwe] = cwe_pair_counts.get(cwe, 0) + 1

    random.shuffle(out_rows)
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pos = sum(1 for r in out_rows if r["label"] == 1)
    neg = len(out_rows) - pos
    print(f"[ds] seen={seen} skipped={skipped} pairs={pos}", file=sys.stderr)
    print(f"[ds] per-CWE pair counts: {cwe_pair_counts}", file=sys.stderr)
    print(f"[ds] wrote {OUT}: {pos} positive / {neg} negative", file=sys.stderr)

    # --- Merge with existing CyberSecEval-derived data ---
    v2_path = OUT.parent / "pairs_v2.jsonl"
    merged_path = OUT.parent / "pairs_merged.jsonl"
    merged_rows: list[dict] = []
    if v2_path.exists():
        with v2_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                merged_rows.append(json.loads(line))
    merged_rows.extend(out_rows)
    random.Random(7).shuffle(merged_rows)
    with merged_path.open("w", encoding="utf-8") as f:
        for r in merged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    m_pos = sum(1 for r in merged_rows if r["label"] == 1)
    m_neg = len(merged_rows) - m_pos
    print(f"[ds] wrote {merged_path}: {len(merged_rows)} rows ({m_pos} pos / {m_neg} neg)", file=sys.stderr)


if __name__ == "__main__":
    main()
