"""Build a position-paired dataset from CyberSecEval autocomplete.

Each row in `walledai/CyberSecEval` (autocomplete config) gives us:
  - origin_code: full file context
  - line_number: 1-indexed line containing the vulnerability
  - line_text: the literal vulnerable line content
  - cwe_identifier: e.g. 'CWE-22'

For each row we emit TWO training examples derived from the same source:

  POSITIVE (label=1, "vulnerability just happened"):
      truncate origin_code at END of line_number
      probe sees the hidden state right after the vulnerable line —
      this is the position at which a streaming probe should fire.

  NEGATIVE (label=0, "no imminent vulnerability"):
      truncate origin_code at END of a line that is at least 3 lines
      BEFORE the vulnerable line (skip the immediate lead-up so we
      don't accidentally include the prompt that *led* to the vuln).
      Same code, different position — paired contrast.

This same-code-pair structure gives the probe a clean discriminative
signal: the only thing that changes between pos and neg is "how much
of the file has been emitted so far," so any signal must be about the
*recently emitted vulnerable line*, not about whose code it is.

Languages used: python and javascript (matching the demo prompts).
Output: data/pairs_v2.jsonl
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset  # type: ignore

OUT = Path(__file__).parent.parent / "data" / "pairs_v2.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)
random.seed(7)

LANGS = ["python", "javascript"]
MAX_ROWS_PER_LANG = 800  # cap per language so total is tractable
MIN_LEADUP_LINES = 3      # negative position must be at least this far before the vuln


def make_pair(origin_code: str, line_number: int, line_text: str, cwe: str, lang: str, repo: str) -> list[dict] | None:
    """Build (positive, negative) example pair, or None if we can't."""
    lines = origin_code.splitlines(keepends=True)
    if not lines:
        return None

    # `line_number` is reported relative to the FULL source file, not the
    # snippet excerpt in `origin_code` — frequently far out of bounds for
    # our truncated context. Locate the vulnerable line by FUZZY MATCHING
    # `line_text` inside `origin_code` instead.
    target = (line_text or "").strip()
    vuln_idx = -1
    if target:
        for i, ln in enumerate(lines):
            if target in ln:
                vuln_idx = i
                break
    if vuln_idx < 0:
        return None  # couldn't locate the vulnerable line in the excerpt

    # Positive: truncate up to AND INCLUDING the vulnerable line.
    pos_code = "".join(lines[: vuln_idx + 1])
    if len(pos_code.strip()) < 20:
        return None  # too little context

    # Negative position: a line strictly before MIN_LEADUP_LINES before the vuln line.
    neg_idx = vuln_idx - MIN_LEADUP_LINES - random.randint(0, 4)
    if neg_idx < 1:
        # If the vuln is too close to the start, take just the first line
        # before it as the negative — better than skipping the row entirely.
        neg_idx = max(0, vuln_idx - 1)
        if neg_idx == vuln_idx:
            return None
    neg_code = "".join(lines[: neg_idx + 1])
    if len(neg_code.strip()) < 10:
        return None

    return [
        {
            "code": pos_code, "label": 1, "cwe": cwe, "cve": None,
            "source": "CyberSecEval", "lang": lang, "vuln_type": cwe,
            "_origin_repo": repo, "_line_no": line_number,
        },
        {
            "code": neg_code, "label": 0, "cwe": None, "cve": None,
            "source": "CyberSecEval-leadup", "lang": lang, "vuln_type": None,
            "_origin_repo": repo, "_line_no": neg_idx + 1,
        },
    ]


def main() -> None:
    out_rows: list[dict] = []
    counts: dict[str, int] = {}

    for lang in LANGS:
        print(f"[ds] loading walledai/CyberSecEval autocomplete split={lang}…", file=sys.stderr)
        ds = load_dataset("walledai/CyberSecEval", "autocomplete", split=lang, streaming=True)
        n_lang = 0
        for row in ds:
            pair = make_pair(
                origin_code=row.get("origin_code") or "",
                line_number=int(row.get("line_number") or 0),
                line_text=row.get("line_text") or "",
                cwe=row.get("cwe_identifier") or "",
                lang=lang,
                repo=row.get("repo") or "",
            )
            if pair is None:
                continue
            out_rows.extend(pair)
            n_lang += 1
            if n_lang >= MAX_ROWS_PER_LANG:
                break
        counts[lang] = n_lang
        print(f"[ds] {lang}: {n_lang} pairs collected", file=sys.stderr)

    random.shuffle(out_rows)
    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(1 for r in out_rows if r["label"] == 1)
    neg = len(out_rows) - pos
    print(f"[ds] wrote {OUT}: {pos} positive / {neg} negative  (per-lang {counts})", file=sys.stderr)


if __name__ == "__main__":
    main()
