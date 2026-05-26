"""AST-based "live code" mask for token-level eval (and training).

Drops tokens that obviously cannot carry a vulnerability — comments,
import / include / use statements, function / class / namespace
*signatures* (keeps the bodies), decorators, C/C++ preprocessor
directives. Keeps everything else, including string literals (SQL
injection lives there).

Used today at eval time as the `code_only_all` aggregation level: same
per-token AUC as `all`, but restricted to live-code tokens. The reason
`all` AUC looks much higher than `span_max` AUC is that ~98% of tokens
are trivially negative (comments, signatures, whitespace) — the probe
just has to keep those at low probability and the AUC inflates. This
mask removes that confound.

NOTE (training-time use, future work): the same mask can drop trivial
negatives from the training set fed into `src/train_probe_spanmax.py`.
Currently the span-max loss treats every out-of-span token as a
negative — including comments and `def` signatures — which gives the
probe an easy way to win that doesn't transfer to harder downstream
distributions. Applying the mask before training would produce a probe
forced to discriminate live-code-positive from live-code-negative
directly. Plumbing for that is *not* wired here yet — only the eval
side.

Backends:
- Python: `tree_sitter_python`
- C:      `tree_sitter_c`
- C++:    `tree_sitter_cpp`

All three ship as standard PyPI wheels (no system compiler needed).
If `tree_sitter` or a language grammar is missing, `code_only_mask`
falls back to a no-op (keeps every token) and emits one warning per
language. The eval framework treats `code_only_all_metrics` as
`{n_total: 0}` in that case, mirroring how `span` behaves on
single-class data.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np


# Node types per language whose char range should be dropped *entirely*.
# Anything not in these sets but inside one of the DROP_PARENT_SIGNATURE_ONLY
# nodes drops only the signature portion (see _build_drop_ranges).
_DROP_WHOLE: dict[str, frozenset[str]] = {
    "python": frozenset({
        "comment",
        "import_statement",
        "import_from_statement",
        "future_import_statement",
        "decorator",
    }),
    "c": frozenset({
        "comment",
        "preproc_include",
        "preproc_def",
        "preproc_function_def",
        "preproc_call",
        "preproc_if",
        "preproc_ifdef",
        "preproc_else",
        "preproc_elif",
    }),
    "cpp": frozenset({
        "comment",
        "preproc_include",
        "preproc_def",
        "preproc_function_def",
        "preproc_call",
        "preproc_if",
        "preproc_ifdef",
        "preproc_else",
        "preproc_elif",
        "using_declaration",
        "using_directive",
        "namespace_alias_definition",
    }),
}


# Definition-style nodes where we want to drop the *signature* but keep
# the body. Maps node_type -> body_field_name (tree-sitter convention).
_SIGNATURE_DROP: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "body",
        "class_definition": "body",
    },
    "c": {
        "function_definition": "body",
    },
    "cpp": {
        "function_definition": "body",
        "namespace_definition": "body",
        # class/struct specifiers: drop the head, keep field list
        "class_specifier": "body",
        "struct_specifier": "body",
    },
}


_LANG_ALIASES = {
    "python": "python", "py": "python",
    "c": "c",
    "cpp": "cpp", "c++": "cpp", "cxx": "cpp", "cc": "cpp",
}


@lru_cache(maxsize=8)
def _get_parser(lang: str):
    """Lazily load a tree-sitter parser. Returns None if unavailable."""
    try:
        from tree_sitter import Language, Parser
    except ImportError:
        warnings.warn(
            "tree-sitter not installed; code_only_mask will keep all tokens. "
            "Install: pip install tree-sitter tree-sitter-python tree-sitter-c tree-sitter-cpp",
            stacklevel=2,
        )
        return None

    try:
        if lang == "python":
            import tree_sitter_python as ts_lang
        elif lang == "c":
            import tree_sitter_c as ts_lang
        elif lang == "cpp":
            import tree_sitter_cpp as ts_lang
        else:
            return None
    except ImportError:
        warnings.warn(
            f"tree-sitter grammar for '{lang}' not installed; "
            f"code_only_mask will keep all tokens for this language.",
            stacklevel=2,
        )
        return None

    language = Language(ts_lang.language())
    parser = Parser(language)
    return parser


def _node_byte_range(node) -> tuple[int, int]:
    """tree-sitter exposes start_byte / end_byte for every node."""
    return node.start_byte, node.end_byte


def _build_drop_ranges(
    root, lang: str, code_bytes: bytes,
) -> list[tuple[int, int]]:
    """Walk the tree-sitter parse tree and collect *byte* ranges to drop.

    Returns a list of half-open `[start, end)` byte ranges. Caller
    converts to char ranges via `bytes.decode` offsetting (since the
    dataset is UTF-8 and offsets are byte offsets in tree-sitter).
    """
    drop_whole = _DROP_WHOLE.get(lang, frozenset())
    sig_drop = _SIGNATURE_DROP.get(lang, {})
    ranges: list[tuple[int, int]] = []

    cursor_stack = [root]
    while cursor_stack:
        node = cursor_stack.pop()
        ntype = node.type

        if ntype in drop_whole:
            ranges.append(_node_byte_range(node))
            continue  # don't descend — the whole subtree is dropped

        if ntype in sig_drop:
            body_field = sig_drop[ntype]
            body = node.child_by_field_name(body_field)
            if body is not None:
                # Signature = everything from this node's start up to the
                # body's start byte.
                start = node.start_byte
                sig_end = body.start_byte
                if sig_end > start:
                    ranges.append((start, sig_end))
                # Descend into the body only.
                cursor_stack.append(body)
                continue
            # No body (rare — forward declaration, abstract method): drop the
            # whole node.
            ranges.append(_node_byte_range(node))
            continue

        # Default: descend.
        cursor_stack.extend(node.children)

    return ranges


def _byte_to_char_ranges(
    byte_ranges: list[tuple[int, int]], code_bytes: bytes,
) -> list[tuple[int, int]]:
    """Convert UTF-8 byte ranges to character offsets in the decoded string.

    The dataset stores `code` as a str and `token_labels` ranges as char
    offsets. tree-sitter emits byte offsets. For pure-ASCII code (most
    of the SVEN corpus), bytes==chars and this is the identity. For
    code with multi-byte chars we walk the byte → char map.
    """
    # Fast path: all ASCII.
    try:
        ascii_only = code_bytes.decode("ascii", errors="strict")
        return list(byte_ranges)  # bytes == chars
    except UnicodeDecodeError:
        pass

    # Slow path: build a byte→char index.
    code_str = code_bytes.decode("utf-8", errors="replace")
    # `i`-th char starts at byte offset byte_starts[i].
    byte_starts = [0] * (len(code_str) + 1)
    pos = 0
    for i, ch in enumerate(code_str):
        byte_starts[i] = pos
        pos += len(ch.encode("utf-8"))
    byte_starts[len(code_str)] = pos

    def b2c(byte_off: int) -> int:
        # Binary search.
        lo, hi = 0, len(byte_starts) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if byte_starts[mid] < byte_off:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return [(b2c(s), b2c(e)) for (s, e) in byte_ranges]


def live_code_char_ranges(code: str, lang: str) -> Optional[list[tuple[int, int]]]:
    """Return a list of `(start_char, end_char)` ranges containing *live code*.

    Live code = `code` minus all dropped ranges (comments, imports,
    signatures, decorators, preprocessor). Adjacent live segments are
    merged. Returns `None` if tree-sitter is unavailable for this
    language — callers should treat that as "keep all tokens".
    """
    lang_norm = _LANG_ALIASES.get((lang or "").lower())
    if lang_norm is None:
        return None

    parser = _get_parser(lang_norm)
    if parser is None:
        return None

    code_bytes = code.encode("utf-8")
    try:
        tree = parser.parse(code_bytes)
    except Exception:
        return None

    drop_byte_ranges = _build_drop_ranges(tree.root_node, lang_norm, code_bytes)
    drop_char_ranges = _byte_to_char_ranges(drop_byte_ranges, code_bytes)

    # Compute complement = live regions.
    n = len(code)
    if not drop_char_ranges:
        return [(0, n)]

    # Sort + merge drop ranges first.
    drop_char_ranges.sort()
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = drop_char_ranges[0]
    for s, e in drop_char_ranges[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    live: list[tuple[int, int]] = []
    prev_end = 0
    for s, e in merged:
        if s > prev_end:
            live.append((prev_end, s))
        prev_end = e
    if prev_end < n:
        live.append((prev_end, n))

    return live


def _token_in_live(
    tok_start: int, tok_end: int, live_ranges: Sequence[tuple[int, int]],
) -> bool:
    """True if the token's char range overlaps any live-code range."""
    for s, e in live_ranges:
        if tok_end <= s:
            return False  # live_ranges is sorted; no later range can overlap
        if tok_start >= e:
            continue
        return True
    return False


def code_only_mask(
    code: str,
    lang: str,
    token_offsets: Iterable[tuple[int, int]],
) -> np.ndarray:
    """Return a `(n_tokens,)` boolean mask where True = "live code token".

    Falls back to all-True if tree-sitter is unavailable, the language
    is unknown, or parsing fails — the eval level then degenerates to
    plain `all`, which is the conservative behaviour.

    Also drops tokens whose char range is entirely whitespace (those
    sometimes survive AST filtering — e.g. between two statements).
    """
    offsets = list(token_offsets)
    n = len(offsets)
    if n == 0:
        return np.zeros(0, dtype=bool)

    live = live_code_char_ranges(code, lang)
    if live is None:
        return np.ones(n, dtype=bool)

    mask = np.zeros(n, dtype=bool)
    for i, (ts, te) in enumerate(offsets):
        if te <= ts:
            continue  # zero-length token (special tokens etc.)
        if not _token_in_live(int(ts), int(te), live):
            continue
        # Drop whitespace-only tokens.
        if code[ts:te].strip() == "":
            continue
        mask[i] = True
    return mask
