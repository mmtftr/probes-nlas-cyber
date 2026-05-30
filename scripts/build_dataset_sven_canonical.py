#!/usr/bin/env python3
# [ai-generated]
"""Build the canonical token-labeled `dataset.jsonl` from SVEN in one pass.

The two legacy scripts don't compose into the unified schema that
`src/eval/token_data.py` / `scripts/validate_dataset.py` require:

  - `build_dataset_sven.py` emits position-paired rows but only the
    sample-level keys (`code,label,cwe,source,lang,vuln_type,_func_name,
    _file_name`) — no `token_labels` / `is_completion_vulnerable` /
    `label_confidence`, and it hardcodes the output path + merges with a
    CyberSecEval file that isn't present on a fresh workspace.
  - `derive_rich_labels.py` produces `token_labels` but reads a different
    corpus (`pairs_v2.jsonl`), is capped at 100 rows, and drops the
    sample-level keys.

This driver reuses BOTH modules' logic (no duplication of the truncation
or regex span code) and writes rows in the canonical schema:

    code, label, is_completion_vulnerable, cwe, lang, source, vuln_type,
    token_labels{evidence,vulnerable_line,sink,source,sanitizer},
    label_confidence, _func_name, _file_name

Positives carry rich token_labels; negatives carry all-empty token_labels
and cwe=None (the negative-row invariant checked by validate_dataset.py).

Usage:
    python scripts/build_dataset_sven_canonical.py --out <path/dataset.jsonl>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset  # type: ignore

# Reuse the legacy logic rather than re-implementing it.
from build_dataset_sven import make_pair, normalize_cwe  # noqa: E402
from derive_rich_labels import derive_row  # noqa: E402


def _pattern_key_cwe(padded: str) -> str:
    """Map SVEN's zero-padded 'CWE-022' to the CWE_PATTERNS key form 'CWE-22'.

    SVEN uses 3-digit zero-padding (cwe-022); derive_rich_labels' pattern
    table keys are unpadded (CWE-22). Strip leading zeros from the numeric
    part so the regex lookup hits. Unknown CWEs (no pattern entry) still get
    a vulnerable_line span from derive_row, so positives are never span-less.
    """
    if not padded or not padded.upper().startswith("CWE-"):
        return padded
    num = padded.split("-", 1)[1].lstrip("0") or "0"
    return f"CWE-{num}"


def build(out_path: Path, seed: int = 7) -> None:
    random.seed(seed)
    rows_out: list[dict] = []
    seen = skipped = 0

    for split in ("train", "val"):
        ds = load_dataset("bstee615/sven", split=split)
        for row in ds:
            seen += 1
            pair = make_pair(row)
            if pair is None:
                skipped += 1
                continue
            for base in pair:
                rows_out.append(_to_canonical(base))

    random.shuffle(rows_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pos = sum(1 for r in rows_out if r["label"] == 1)
    neg = len(rows_out) - pos
    print(
        f"[canonical] seen={seen} skipped={skipped} rows={len(rows_out)} "
        f"({pos} pos / {neg} neg) -> {out_path}",
        file=sys.stderr,
    )


def _to_canonical(base: dict) -> dict:
    """Turn a build_dataset_sven pair-row into a canonical schema row."""
    label = int(base["label"])
    lang = base.get("lang")
    source = base.get("source")
    vuln_type = base.get("vuln_type")
    func_name = base.get("_func_name", "")
    file_name = base.get("_file_name", "")

    if label == 0:
        # Negative-row invariant: empty token_labels, cwe None.
        derived = derive_row({"code": base["code"], "label": 0, "cwe": None, "lang": lang})
        cwe_out = None
    else:
        # Positive: feed derive_row the unpadded CWE so CWE_PATTERNS hits.
        cwe_padded = normalize_cwe(base.get("cwe") or "")
        derived = derive_row(
            {
                "code": base["code"],
                "label": 1,
                "cwe": _pattern_key_cwe(cwe_padded),
                "lang": lang,
            }
        )
        # Keep the padded CWE in the canonical row (matches the rest of the corpus).
        cwe_out = cwe_padded or None

    return {
        "code": base["code"],
        "label": label,
        "is_completion_vulnerable": bool(derived["is_completion_vulnerable"]),
        "cwe": cwe_out,
        "lang": lang,
        "source": source,
        "vuln_type": vuln_type,
        "token_labels": derived["token_labels"],
        "label_confidence": derived["label_confidence"],
        "_func_name": func_name,
        "_file_name": file_name,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "dataset.jsonl"),
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    build(Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
