"""Smoke tests for the `code_only_mask` tree-sitter live-code mask."""
from __future__ import annotations

import numpy as np
import pytest

from src.eval.code_mask import code_only_mask, live_code_char_ranges


def _per_char_offsets(code: str) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(len(code))]


def _kept_chars(code: str, lang: str) -> str:
    offsets = _per_char_offsets(code)
    mask = code_only_mask(code, lang, offsets)
    return "".join(code[i] for i, keep in enumerate(mask) if keep)


def test_python_drops_comments_imports_signatures_keeps_body():
    code = (
        "# top comment\n"
        "import os\n"
        "from typing import List\n"
        "\n"
        "def foo(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y\n"
    )
    kept = _kept_chars(code, "python")
    assert "comment" not in kept
    assert "import" not in kept
    assert "def" not in kept and "foo" not in kept
    assert "x: int" not in kept and "-> int" not in kept
    assert "y=x+1" in kept.replace(" ", "")
    assert "returny" in kept.replace(" ", "")


def test_python_keeps_string_literals_inside_body():
    code = (
        "def q(name):\n"
        '    sql = f"SELECT * FROM users WHERE name = {name}"\n'
        "    return sql\n"
    )
    kept = _kept_chars(code, "python")
    assert "SELECT" in kept
    assert "WHERE" in kept


def test_python_drops_decorators():
    code = (
        "@cache\n"
        "@retry(3)\n"
        "def f():\n"
        "    return 1\n"
    )
    kept = _kept_chars(code, "python")
    assert "@cache" not in kept
    assert "retry" not in kept
    assert "return1" in kept.replace(" ", "")


def test_c_drops_includes_and_signature_keeps_body():
    code = (
        "#include <stdio.h>\n"
        "// header\n"
        "int main(int argc, char **argv) {\n"
        "    int x = 1;\n"
        "    return x;\n"
        "}\n"
    )
    kept = _kept_chars(code, "c")
    assert "#include" not in kept
    assert "header" not in kept
    assert "main" not in kept and "argc" not in kept
    assert "intx=1" in kept.replace(" ", "")
    assert "returnx" in kept.replace(" ", "")


def test_cpp_drops_namespace_signature_keeps_body():
    code = (
        "#include <string>\n"
        "namespace ns {\n"
        "    int g() {\n"
        "        return 42;\n"
        "    }\n"
        "}\n"
    )
    kept = _kept_chars(code, "cpp")
    assert "#include" not in kept
    assert "namespace" not in kept
    assert "return42" in kept.replace(" ", "")


def test_unknown_lang_falls_back_to_keep_all():
    code = "let x = 1; // comment\n"
    offsets = _per_char_offsets(code)
    mask = code_only_mask(code, "rust", offsets)
    assert mask.all(), "unknown lang must keep all tokens (no-op fallback)"


def test_empty_input():
    assert code_only_mask("", "python", []).shape == (0,)
    assert live_code_char_ranges("", "python") in (None, [(0, 0)])


def test_whitespace_only_tokens_dropped():
    code = "def f():\n    return 1\n"
    offsets = [(0, 4), (4, 5), (5, 6), (6, 8), (8, 9), (9, 13), (13, 19), (19, 20), (20, 21)]
    mask = code_only_mask(code, "python", offsets)
    for i, (s, e) in enumerate(offsets):
        if code[s:e].strip() == "":
            assert not mask[i], f"whitespace-only token {i} ({code[s:e]!r}) should be dropped"


def test_mask_length_matches_offsets():
    code = "def f():\n    x = 1\n    return x\n"
    for n_tokens in (1, 4, 16, 64):
        step = max(1, len(code) // n_tokens)
        offsets = [(i, min(i + step, len(code))) for i in range(0, len(code), step)]
        mask = code_only_mask(code, "python", offsets)
        assert mask.shape == (len(offsets),)


def test_zero_length_tokens_not_kept():
    code = "def f():\n    return 1\n"
    offsets = [(0, 0), (0, 3), (4, 4)] + _per_char_offsets(code)[4:]
    mask = code_only_mask(code, "python", offsets)
    assert not mask[0]
    assert not mask[2]
