"""Token-level dataset loader for `data/dataset.jsonl`.

The new dataset (SVEN-paired, token-labeled) is the canonical input for
the new probe. Schema per row:

    {
      "code":  "<source code>",
      "label": 1 | 0,
      "cwe":   "CWE-089" | None,
      "lang":  "python" | "c" | "cpp",
      "source":"SVEN-before" | "SVEN-after",
      "_file_name": "...",
      "_func_name": "...",
      "token_labels": {
        "evidence":        [[start_char, end_char], ...],   # positive
        "vulnerable_line": [[start_char, end_char], ...],   # positive
        "sink":            [[start_char, end_char], ...],   # positive
        "source":          [[start_char, end_char], ...],   # positive
        "sanitizer":       [[start_char, end_char], ...],   # negative
      }
    }

`token_labels` ranges are **character offsets** into `code`. At eval time
we map them to token indices using whatever tokenizer the probe was
trained against (caller supplies it — the framework stays tokenizer-
agnostic).

Positive vs negative label semantics (matches the paper, Section 4):
  - positive token = inside any of evidence / vulnerable_line / sink / source
  - negative token = inside sanitizer
  - the union of positive+negative = "annotated span"; everything else is
    out-of-span (used at the `all`-token aggregation level but excluded
    from the `span` and `span_max` levels)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# Span keys treated as positive evidence of vulnerability.
POSITIVE_KEYS: tuple[str, ...] = ("evidence", "vulnerable_line", "sink", "source")
# Span keys treated as negative (mitigated) evidence.
NEGATIVE_KEYS: tuple[str, ...] = ("sanitizer",)


@dataclass
class TokenSpan:
    """A char-range span with a binary label.

    `label`=1 for positive (evidence/sink/source/vulnerable_line),
    `label`=0 for negative (sanitizer).
    """
    start_char: int
    end_char: int
    label: int
    source_key: str  # which token_labels key this span came from


def parse_spans(row: dict) -> list[TokenSpan]:
    """Return all char-range spans in a row, deduplicated.

    The same char range often appears under both `evidence` and
    `vulnerable_line`; we collapse those to a single positive span
    keyed on the first encountered source. Sanitizer spans are kept
    separately as negatives.
    """
    seen: dict[tuple[int, int], TokenSpan] = {}
    tl = row.get("token_labels") or {}
    for key in POSITIVE_KEYS:
        for span in tl.get(key, []) or []:
            s, e = int(span[0]), int(span[1])
            if e < s:
                continue
            k = (s, e)
            if k not in seen:
                seen[k] = TokenSpan(s, e, 1, key)
    for key in NEGATIVE_KEYS:
        for span in tl.get(key, []) or []:
            s, e = int(span[0]), int(span[1])
            if e < s:
                continue
            k = (s, e)
            # Sanitizer overrides positive only if there's no positive at the
            # same range (rare). In practice the dataset doesn't co-emit.
            if k not in seen:
                seen[k] = TokenSpan(s, e, 0, key)
    return list(seen.values())


def load_token_dataset(path: str | Path) -> list[dict]:
    """Load `data/dataset.jsonl`. Returns rows as-is; spans are parsed lazily
    via `parse_spans(row)`.
    """
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def char_spans_to_token_spans(
    char_spans: list[TokenSpan],
    token_offsets: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Map char-range spans to (start_tok, end_tok, label) inclusive tuples.

    `token_offsets[i] == (start_char, end_char)` is the character range
    of the i-th token. Half-open standard from HuggingFace tokenizers.

    A char-range maps to every token whose offset *overlaps* it. Empty
    overlaps yield empty token spans (skipped by callers).
    """
    out: list[tuple[int, int, int]] = []
    for span in char_spans:
        first: int | None = None
        last: int | None = None
        for i, (ts, te) in enumerate(token_offsets):
            if te <= span.start_char:
                continue
            if ts >= span.end_char:
                break
            if first is None:
                first = i
            last = i
        if first is not None and last is not None:
            out.append((first, last, span.label))
    return out


def token_labels_array(
    n_tokens: int,
    token_spans: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, in_span_mask) of length n_tokens.

    - `labels[i] = 1` if token i is inside any positive span, else 0.
    - `in_span_mask[i] = True` if token i is inside any annotated (pos or
      neg) span. Out-of-span tokens get label=0 and mask=False, so they
      contribute to the `all` aggregation but are filtered for `span` and
      `span_max`.
    """
    labels = np.zeros(n_tokens, dtype=np.int8)
    mask = np.zeros(n_tokens, dtype=bool)
    for start, end, lbl in token_spans:
        end_c = min(end, n_tokens - 1)
        if end_c < start:
            continue
        mask[start : end_c + 1] = True
        if lbl == 1:
            labels[start : end_c + 1] = 1
    return labels, mask


def example_label(row: dict) -> int:
    """0/1 example-level label: positive iff `label==1` AND has positive spans."""
    if int(row.get("label", 0)) != 1:
        return 0
    spans = parse_spans(row)
    return 1 if any(s.label == 1 for s in spans) else 0
