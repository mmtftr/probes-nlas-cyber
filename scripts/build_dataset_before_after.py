#!/usr/bin/env python3
# [ai-generated]
"""Build the SVEN **before/after full-function contrast** `dataset.jsonl`.

Decision: `decisions/0002-dataset-before-after-contrast.md`.
Plan:     `plans/cross-model-probe-generalization/REBUILD-PLAN.md`.

Supersedes the completion-truncation builders (`build_dataset_sven.py`,
`build_dataset_sven_canonical.py`), which cut both rows from `func_src_before`
and so encode a length confound + mid-identifier truncation. Here every row is
a FULL function:

    POSITIVE (label=1) = full `func_src_before`  (the vulnerable version)
    NEGATIVE (label=0) = full `func_src_after`   (the fix)

so a before/after pair holds the task fixed and varies only the vulnerability.

Token labels (positives only)
-----------------------------
`token_labels.vulnerable_line` / `.evidence` mark the **diff'd vulnerable lines**
inside `func_src_before`, derived from SVEN's own diff metadata:

  1. `char_changes.deleted` spans index into `func_src_before` — the parts the
     fix removed/modified. We keep spans whose `chars` text matches
     `before[cs:ce]` (drops the ~25 stale-offset spans in the corpus), then
     expand each to whole lines.
  2. If a row has no usable deleted char-span, fall back to
     `line_changes.deleted` line numbers -> char offsets -> whole lines.
  3. Purely-additive fixes (~33% of SVEN: the fix only *inserts* a guard, so
     `deleted` is empty) have nothing removed in `before`. We locate the
     insertion point by the common prefix/suffix of (before, after) and label
     the `before` line at that boundary — the now-unguarded statement the fix
     wraps. This guarantees EVERY positive carries >=1 vulnerable-line span.

Why the guarantee matters: both trainers derive the example-level label from the
token spans (`ex_y = y[eids==e].max() > 0`), NOT from the row's `label` field.
A positive with zero positive tokens would be **silently relabeled negative**.
So (a) every positive must have a non-empty span, and (b) that span must survive
tokenization within the extractor's `max_length` — hence the `--max-chars` cap.

Length cap
----------
Full functions run to 114k chars (~28k tokens). The extractor tokenizes with
`truncation=True, max_length=L`; any vulnerable line past token L is dropped and
its positive silently becomes a negative. To prevent that we drop whole pairs
where either member exceeds `--max-chars`, model-agnostically, so both models
extract the identical row set with no truncation. `--max-chars 6000` keeps
~88% of pairs and bounds kept functions to <=~1935 tokens (Qwen tokenizer),
matching an extractor `max_length` of 2048.

Negatives carry all-empty token_labels and cwe=None (validate_dataset.py's
negative-row invariant). Schema matches `scripts/validate_dataset.py`.

Usage:
    python scripts/build_dataset_before_after.py \
        --out data/dataset.jsonl --split-out data/sven_split_meta.json \
        --max-chars 6000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset  # type: ignore

# Reuse language inference + CWE normalization from the legacy builder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_sven import infer_lang, normalize_cwe  # noqa: E402

LABEL_TYPES = ("evidence", "sink", "source", "sanitizer", "vulnerable_line")


# --------------------------------------------------------------------------- #
# span helpers
# --------------------------------------------------------------------------- #
def _expand_to_lines(code: str, s: int, e: int) -> tuple[int, int] | None:
    """Expand a char range [s,e) to cover the full lines it touches.

    Returns (line_start, line_end) with line_end including the trailing newline
    (clamped to len(code)), or None if the range is degenerate.
    """
    n = len(code)
    s = max(0, min(s, n))
    e = max(s, min(e, n))
    if n == 0:
        return None
    # Treat a zero-width point as "the line containing offset s".
    probe = min(s, n - 1)
    line_start = code.rfind("\n", 0, probe) + 1  # rfind -1 -> 0
    nl = code.find("\n", max(e - 1, line_start))
    line_end = (nl + 1) if nl >= 0 else n
    if line_end <= line_start:
        return None
    return (line_start, line_end)


def _merge_lines(spans: list[tuple[int, int]]) -> list[list[int]]:
    """Sort + merge overlapping/adjacent line spans into non-overlapping list."""
    if not spans:
        return []
    arr = sorted(spans)
    out: list[list[int]] = [list(arr[0])]
    for s, e in arr[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _line_no_to_char_span(code: str, line_no: int) -> tuple[int, int] | None:
    """1-indexed line_no -> (char_start, char_end_incl_newline) in `code`."""
    if line_no < 1:
        return None
    offset, cur = 0, 1
    n = len(code)
    while offset < n and cur < line_no:
        nl = code.find("\n", offset)
        if nl < 0:
            return None
        offset, cur = nl + 1, cur + 1
    if cur != line_no or offset > n:
        return None
    nl = code.find("\n", offset)
    end = (nl + 1) if nl >= 0 else n
    return (offset, end)


def _insertion_point(before: str, after: str) -> int:
    """Char offset in `before` where `after` diverges (common prefix/suffix)."""
    i = 0
    nb, na = len(before), len(after)
    while i < nb and i < na and before[i] == after[i]:
        i += 1
    j = 0
    while j < (nb - i) and j < (na - i) and before[nb - 1 - j] == after[na - 1 - j]:
        j += 1
    div_start, div_end = i, nb - j  # region of `before` that differs
    if div_end > div_start:
        return div_start  # there IS a modified region in before; point at it
    return min(div_start, max(nb - 1, 0))  # pure insertion: boundary line


def vulnerable_spans(before: str, after: str, char_changes: dict, line_changes: dict) -> list[list[int]]:
    """Diff'd vulnerable-line spans in `before`. Always non-empty (whole-func fallback)."""
    line_spans: list[tuple[int, int]] = []

    # (1) char_changes.deleted, text-verified, expanded to lines.
    for ent in (char_changes or {}).get("deleted") or []:
        cs, ce, txt = ent.get("char_start"), ent.get("char_end"), ent.get("chars")
        if not (isinstance(cs, int) and isinstance(ce, int) and 0 <= cs < ce <= len(before)):
            continue
        if txt is not None and before[cs:ce] != txt:
            continue  # stale offset
        ls = _expand_to_lines(before, cs, ce)
        if ls:
            line_spans.append(ls)

    # (2) fall back to line_changes.deleted.
    if not line_spans:
        for ent in (line_changes or {}).get("deleted") or []:
            ln = ent.get("line_no")
            if isinstance(ln, int):
                cspan = _line_no_to_char_span(before, ln)
                if cspan:
                    ls = _expand_to_lines(before, *cspan)
                    if ls:
                        line_spans.append(ls)

    # (3) purely-additive fix: label the insertion-point line.
    if not line_spans:
        p = _insertion_point(before, after)
        ls = _expand_to_lines(before, p, p)
        if ls:
            line_spans.append(ls)

    merged = _merge_lines(line_spans)
    if not merged:  # degenerate (e.g. before==after): whole function.
        merged = [[0, len(before)]]
    return merged


# --------------------------------------------------------------------------- #
# row construction
# --------------------------------------------------------------------------- #
def _empty_labels() -> dict:
    return {k: [] for k in LABEL_TYPES}


def make_rows(row: dict, max_chars: int) -> list[dict] | None:
    before = row.get("func_src_before") or ""
    after = row.get("func_src_after") or ""
    if not before or not after or before == after:
        return None
    if len(before) > max_chars or len(after) > max_chars:
        return None

    file_name = row.get("file_name") or ""
    func_name = row.get("func_name") or ""
    lang = infer_lang(file_name)
    vul_raw = row.get("vul_type") or ""
    cwe = normalize_cwe(vul_raw) or None

    spans = vulnerable_spans(before, after, row.get("char_changes") or {}, row.get("line_changes") or {})
    pos_labels = _empty_labels()
    pos_labels["vulnerable_line"] = [list(s) for s in spans]
    pos_labels["evidence"] = [list(s) for s in spans]

    common = {"_func_name": func_name, "_file_name": file_name, "lang": lang,
              "label_confidence": "diff-derived"}
    positive = {
        "code": before, "label": 1, "is_completion_vulnerable": True,
        "cwe": cwe, "source": "SVEN-before", "vuln_type": vul_raw or None,
        "token_labels": pos_labels, **common,
    }
    negative = {
        "code": after, "label": 0, "is_completion_vulnerable": False,
        "cwe": None, "source": "SVEN-after", "vuln_type": None,
        "token_labels": _empty_labels(), **common,
    }
    return [positive, negative]


def pair_group_key(row: dict) -> str:
    """Mirror src/remotes/the cluster/train_eval.pair_group_key for split parity."""
    fn = row.get("_file_name") or ""
    func = row.get("_func_name") or ""
    if fn or func:
        return f"func::{fn}::{func}"
    return f"row::{hashlib.sha1((row.get('code') or '').encode()).hexdigest()[:12]}"


def write_split_meta(rows: list[dict], path: Path, frac_heldout: float = 0.2, seed: int = 42) -> None:
    """Deterministic seeded group hold-out, identical to load_or_make_split."""
    groups = sorted({pair_group_key(r) for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_held = max(1, int(round(frac_heldout * len(groups))))
    heldout = sorted(groups[:n_held])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"seed": seed, "frac_heldout": frac_heldout,
         "n_groups": len(groups), "heldout_groups": heldout}, indent=2))
    print(f"[split] {len(groups)} groups, {n_held} held out -> {path}", file=sys.stderr)


def build(out_path: Path, split_path: Path | None, max_chars: int, seed: int = 7) -> None:
    rows_out: list[dict] = []
    seen = skipped_len = skipped_other = 0
    additive = wholefunc = 0

    for split in ("train", "val"):
        ds = load_dataset("bstee615/sven", split=split)
        for row in ds:
            seen += 1
            b = row.get("func_src_before") or ""
            a = row.get("func_src_after") or ""
            if not b or not a or b == a:
                skipped_other += 1
                continue
            if len(b) > max_chars or len(a) > max_chars:
                skipped_len += 1
                continue
            pair = make_rows(row, max_chars)
            if pair is None:
                skipped_other += 1
                continue
            # provenance counters
            cc_del = (row.get("char_changes") or {}).get("deleted") or []
            lc_del = (row.get("line_changes") or {}).get("deleted") or []
            if not cc_del and not lc_del:
                additive += 1
            if pair[0]["token_labels"]["vulnerable_line"] == [[0, len(b)]]:
                wholefunc += 1
            rows_out.extend(pair)

    random.Random(seed).shuffle(rows_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pos = sum(1 for r in rows_out if r["label"] == 1)
    neg = len(rows_out) - pos
    print(
        f"[before-after] seen={seen} dropped(len>{max_chars})={skipped_len} "
        f"dropped(other)={skipped_other} rows={len(rows_out)} ({pos} pos / {neg} neg)\n"
        f"               additive-fix pairs={additive}  whole-func-fallback labels={wholefunc} "
        f"-> {out_path}",
        file=sys.stderr,
    )
    if split_path is not None:
        write_split_meta(rows_out, split_path)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo / "data" / "dataset.jsonl"))
    ap.add_argument("--split-out", default=str(repo / "data" / "sven_split_meta.json"),
                    help="write a fresh seeded group split; '' to skip")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="drop a pair if before or after exceeds this many chars")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    split_path = Path(args.split_out) if args.split_out else None
    build(Path(args.out), split_path, args.max_chars, seed=args.seed)


if __name__ == "__main__":
    main()
