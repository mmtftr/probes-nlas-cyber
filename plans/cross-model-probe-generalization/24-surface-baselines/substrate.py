# [ai-generated]
"""exp-24 substrate — load the exp-16 Qwen-32B per-token dump + dataset into a
single aligned token table, plus the surface-feature builders.

The dump (`16-token-logit-dump/results/logitdump_<model>/logits_layer25.npz`)
carries, per token: y, prob, example_id, char_start, char_end, is_test, is_code.
`example_id` indexes `data/dataset.jsonl` (row order); char offsets index that
row's `code`. So the token axis is byte-identical to the probe's, and we derive
every surface feature without re-tokenizing or re-extracting activations.

No GPU, CPU only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DUMP = (REPO / "plans/cross-model-probe-generalization/16-token-logit-dump/"
        "results/logitdump_Qwen_Qwen2.5-Coder-32B-Instruct/logits_layer25.npz")
DATASET = REPO / "data/dataset.jsonl"

INJ = ("CWE-089", "CWE-078", "CWE-022", "CWE-079")
WINDOW = 48


@dataclass
class Substrate:
    # per-token (length N)
    y: np.ndarray          # int8 annotated label
    prob: np.ndarray       # float32 general-probe sigmoid (comparison target)
    eid: np.ndarray        # int32 example id (dataset row)
    cs: np.ndarray         # char_start
    ce: np.ndarray         # char_end
    is_test: np.ndarray    # bool canonical seed-42 hold-out
    is_code: np.ndarray    # bool tree-sitter live-code mask
    tok: list[str]         # token surface string code[cs:ce] (raw, not stripped)
    win: list[str]         # ±48-char window
    lang: np.ndarray       # object per-token language ('python'|'c'|'cpp')
    # per-example (length n_rows, indexed by eid)
    cwe_ex: list           # cwe or None
    clean_ex: np.ndarray   # bool: label==0 and not cwe
    lang_ex: list          # language per example


def load_substrate() -> Substrate:
    d = np.load(DUMP)
    rows = [json.loads(l) for l in DATASET.open()]
    eid = d["example_id"].astype(np.int64)
    cs = d["char_start"].astype(np.int64)
    ce = d["char_end"].astype(np.int64)
    n_rows = len(rows)
    assert eid.max() < n_rows, f"eid {eid.max()} >= n_rows {n_rows}"

    codes = [r["code"] for r in rows]
    lang_ex = [r.get("lang") or "" for r in rows]
    cwe_ex = [r.get("cwe") for r in rows]
    clean_ex = np.array([(r.get("label") == 0 and not r.get("cwe")) for r in rows], bool)

    N = len(eid)
    tok: list[str] = [None] * N
    win: list[str] = [None] * N
    lang = np.empty(N, dtype=object)
    for i in range(N):
        e = eid[i]
        code = codes[e]
        a, b = int(cs[i]), int(ce[i])
        tok[i] = code[a:b]
        win[i] = code[max(0, a - WINDOW): b + WINDOW]
        lang[i] = lang_ex[e]

    return Substrate(
        y=d["y"].astype(np.int8), prob=d["prob"].astype(np.float32), eid=eid,
        cs=cs, ce=ce, is_test=d["is_test"].astype(bool), is_code=d["is_code"].astype(bool),
        tok=tok, win=win, lang=lang,
        cwe_ex=cwe_ex, clean_ex=clean_ex, lang_ex=lang_ex,
    )


# ----------------------------- surface features -----------------------------

# Security lexicon for baseline (c). Grouped, case-insensitive word matches.
LEXICON: dict[str, tuple[str, ...]] = {
    "sql": ("SELECT", "INSERT", "UPDATE", "DELETE", "WHERE", "execute",
            "executemany", "query", "cursor"),
    "cmd": ("system", "popen", "exec", "execve", "execl", "subprocess",
            "shell", "Popen"),
    "path": ("open", "fopen", "realpath", "basename", "dirname"),
    "mem": ("strcpy", "strcat", "memcpy", "memmove", "malloc", "calloc",
            "realloc", "free", "sizeof", "len", "idx", "index", "NULL",
            "alloca", "sprintf", "gets"),
}
_LEX_PATTERNS = {
    grp: re.compile(r"|".join(rf"\b{re.escape(w)}\b" for w in words))
    for grp, words in LEXICON.items()
}
# `../` path traversal handled separately (not a word).
_DOTDOT = re.compile(r"\.\./")


def keyword_counts(windows: list[str]) -> np.ndarray:
    """(N, 5) integer count matrix: [sql, cmd, path, mem, dotdot] hits in window."""
    out = np.zeros((len(windows), 5), dtype=np.float32)
    for i, w in enumerate(windows):
        out[i, 0] = len(_LEX_PATTERNS["sql"].findall(w))
        out[i, 1] = len(_LEX_PATTERNS["cmd"].findall(w))
        out[i, 2] = len(_LEX_PATTERNS["path"].findall(w))
        out[i, 3] = len(_LEX_PATTERNS["mem"].findall(w))
        out[i, 4] = len(_DOTDOT.findall(w))
    return out


def lang_indicator(lang_per_tok: np.ndarray) -> np.ndarray:
    """+1 if token's file is C/C++, else 0. (N,1) float."""
    return np.array([[1.0 if l in ("c", "cpp") else 0.0] for l in lang_per_tok],
                    dtype=np.float32)
