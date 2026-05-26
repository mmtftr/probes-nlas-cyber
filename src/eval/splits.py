"""Leakage-aware evaluation splits for paired vulnerability data.

Refactors `scripts/eval_splits.py` into a reusable, importable module.
Each split function returns a list of `Split` dataclasses; the caller
fits the probe on `train_idx` and evaluates on `test_idx`.

Why these splits and not random?
  Paired safe/vulnerable variants share a `_origin_repo`. A random
  stratified split puts both halves in train AND test, so the probe can
  memorise the repo, not the vulnerability pattern. The paper sidesteps
  this by having long-form generation samples with no pair structure; we
  have to design around it explicitly.

The five splits map to five distinct generalisation claims:
  random_stratified  -> "can fit"            (leaky baseline)
  group_repo         -> "generalises to unseen code"
  heldout_cwe        -> "generalises to unseen vulnerability classes"
  heldout_lang       -> "generalises across programming languages"
  heldout_source     -> "generalises across dataset provenances"
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class Split:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    note: str = ""


def pair_group_key(row: dict) -> str:
    """Canonical group key for paired safe/vulnerable variants.

    Different builders emit different group metadata:
      - CyberSecEval / CyberNative -> `_origin_repo` populated
      - SVEN (`scripts/build_dataset.py`) -> `_origin_repo` is None,
        but `_file_name` and `_func_name` are populated
    To keep `group_repo` meaningful on either corpus, we use the first
    non-empty source in priority order and fall back to a per-row sentinel
    so an unkeyable row becomes its own singleton group (won't leak, won't
    join anything).
    """
    repo = row.get("_origin_repo")
    if repo:
        return f"repo::{repo}"
    f = row.get("_file_name") or ""
    fn = row.get("_func_name") or ""
    if f or fn:
        return f"func::{f}::{fn}"
    # Last resort: keep stable across calls within a process by using id().
    return f"row::{id(row)}"


def attach_paired_cwe(rows: list[dict]) -> list[dict]:
    """Propagate CWE from each pair's positive row down to its paired
    negative, keyed by `pair_group_key` so heldout-CWE keeps pairs together.

    Adds `_paired_cwe` and `_pair_group` to each row in place.
    """
    pair_cwe: dict[str, str | None] = {}
    for r in rows:
        key = pair_group_key(r)
        r["_pair_group"] = key
        if r.get("label") == 1 and r.get("cwe"):
            pair_cwe[key] = r["cwe"]
    for r in rows:
        if r.get("label") == 0 and r.get("cwe") is None:
            r["_paired_cwe"] = pair_cwe.get(r["_pair_group"])
        else:
            r["_paired_cwe"] = r.get("cwe")
    return rows


def split_random(y: np.ndarray, seed: int = 7) -> Split:
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    return Split("random_stratified", tr, te, note="leaky baseline; pairs share group key")


def split_group_repo(rows: list[dict], y: np.ndarray, seed: int = 7) -> Split:
    """Group split on `pair_group_key(row)` — repo when available,
    `(file, func)` for SVEN-style data.
    """
    groups = np.array([pair_group_key(r) for r in rows])
    n_unique = len(set(groups.tolist()))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    (tr, te), = gss.split(np.zeros((len(y), 1)), y, groups=groups)
    note = f"GroupShuffleSplit on pair_group_key (n_groups={n_unique})"
    return Split("group_repo", tr, te, note=note)


def split_heldout_cwe(rows: list[dict], y: np.ndarray, top_k: int = 5) -> list[Split]:
    cwe_arr = np.array([r.get("_paired_cwe") for r in rows], dtype=object)
    pos_cwes = [r.get("_paired_cwe") for r in rows if r.get("label") == 1 and r.get("_paired_cwe")]
    top = [c for c, _ in Counter(pos_cwes).most_common(top_k)]
    out: list[Split] = []
    for cwe in top:
        te = np.where(cwe_arr == cwe)[0]
        tr = np.where(cwe_arr != cwe)[0]
        out.append(Split(f"heldout_cwe::{cwe}", tr, te, note="train all-except / test only"))
    return out


def split_heldout_lang(rows: list[dict], y: np.ndarray, langs: tuple[str, ...] | None = None) -> list[Split]:
    lang_arr = np.array([r.get("lang") for r in rows])
    uniq = sorted({x for x in lang_arr.tolist() if x})
    targets = tuple(langs) if langs else tuple(uniq)
    out: list[Split] = []
    for held in targets:
        te = np.where(lang_arr == held)[0]
        tr = np.where(lang_arr != held)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        # AUC undefined if test set is single-class.
        sub_y = y[te]
        if len(np.unique(sub_y)) < 2:
            out.append(Split(
                f"heldout_lang::test={held}", tr, te,
                note=f"test set single-class ({int(sub_y.sum())}/{len(sub_y)} pos); AUC undefined",
            ))
            continue
        out.append(Split(f"heldout_lang::test={held}", tr, te))
    return out


def split_heldout_source(rows: list[dict], y: np.ndarray) -> list[Split]:
    src_arr = np.array([r.get("source") for r in rows])
    uniq = sorted({x for x in src_arr.tolist() if x})
    out: list[Split] = []
    for held in uniq:
        te = np.where(src_arr == held)[0]
        tr = np.where(src_arr != held)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        sub_y = y[te]
        if len(np.unique(sub_y)) < 2:
            out.append(Split(
                f"heldout_source::{held}", tr, te,
                note=f"single-class test ({int(sub_y.sum())}/{len(sub_y)} pos); skipped",
            ))
            continue
        out.append(Split(f"heldout_source::{held}", tr, te))
    return out


def make_splits(
    rows: list[dict],
    y: np.ndarray,
    include: tuple[str, ...] = ("random", "group_repo", "heldout_cwe", "heldout_lang", "heldout_source"),
    seed: int = 7,
    cwe_top_k: int = 5,
) -> list[Split]:
    """Build the canonical split bundle in one call."""
    attach_paired_cwe(rows)
    splits: list[Split] = []
    if "random" in include:
        splits.append(split_random(y, seed=seed))
    if "group_repo" in include:
        splits.append(split_group_repo(rows, y, seed=seed))
    if "heldout_cwe" in include:
        splits.extend(split_heldout_cwe(rows, y, top_k=cwe_top_k))
    if "heldout_lang" in include:
        splits.extend(split_heldout_lang(rows, y))
    if "heldout_source" in include:
        splits.extend(split_heldout_source(rows, y))
    return splits
