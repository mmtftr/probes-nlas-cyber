#!/usr/bin/env python3
"""Build repo-level scan benchmark fixtures from token-labelled examples.

The existing `data/dataset.jsonl` is function/snippet shaped. That is useful
for probe training, but the scanner's real contract is repo in -> ranked lead
windows out. This script wraps labelled rows into small deterministic repos and
emits manifests with file/line/char traces for later Recall@K / MRR evaluation.

Example:
    uv run python scripts/build_repo_benchmark.py \
        --input data/dataset.jsonl \
        --out-root data/repo_benchmark \
        --max-repos 200 \
        --decoys-per-repo 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "dataset.jsonl"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "repo_benchmark"
SCHEMA = "probes.repo_benchmark/v1"

LANG_EXT = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "c": ".c",
    "cpp": ".cpp",
    "c++": ".cpp",
    "java": ".java",
    "go": ".go",
    "ruby": ".rb",
    "rb": ".rb",
    "php": ".php",
    "rust": ".rs",
    "rs": ".rs",
}


@dataclass(frozen=True)
class IndexedRow:
    idx: int
    row: dict


def load_jsonl(path: Path) -> list[IndexedRow]:
    rows: list[IndexedRow] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(IndexedRow(idx=idx, row=json.loads(line)))
    return rows


def pair_group_key(row: dict) -> str:
    repo = row.get("_origin_repo")
    if repo:
        return f"repo::{repo}"
    file_name = row.get("_file_name") or ""
    func_name = row.get("_func_name") or ""
    if file_name or func_name:
        return f"func::{file_name}::{func_name}"
    return f"row::{hashlib.sha1((row.get('code') or '').encode('utf-8')).hexdigest()[:12]}"


def spans_from_row(row: dict) -> list[list[int]]:
    labels = row.get("token_labels") or {}
    spans = labels.get("evidence") or labels.get("vulnerable_line") or []
    out: list[list[int]] = []
    n = len(row.get("code") or "")
    for span in spans:
        if not isinstance(span, list) or len(span) != 2:
            continue
        s, e = span
        if isinstance(s, int) and isinstance(e, int):
            s2, e2 = max(0, s), min(n, e)
            if s2 < e2:
                out.append([s2, e2])
    return merge_overlapping(out)


def merge_overlapping(spans: Iterable[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for s, e in sorted([list(x) for x in spans if x[0] < x[1]]):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def line_starts(text: str) -> list[int]:
    starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def char_to_line(starts: list[int], offset: int) -> int:
    """Return 1-indexed line number for a char offset."""
    if offset <= 0:
        return 1
    lo, hi = 0, len(starts)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if starts[mid] <= offset:
            lo = mid
        else:
            hi = mid
    return lo + 1


def trace_records(file_path: str, code: str, spans: list[list[int]], cwe: Optional[str]) -> list[dict]:
    starts = line_starts(code)
    traces: list[dict] = []
    for s, e in spans:
        traces.append({
            "file": file_path,
            "start_char": int(s),
            "end_char": int(e),
            "start_line": char_to_line(starts, s),
            "end_line": char_to_line(starts, max(s, e - 1)),
            "cwe": cwe,
            "role": "evidence",
            "label_source": "diff_oracle",
        })
    return traces


def safe_slug(value: str, fallback: str = "item") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-").lower()
    return value[:80] or fallback


def lang_ext(row: dict) -> str:
    file_name = row.get("_file_name") or ""
    suffix = Path(file_name).suffix
    if suffix:
        return suffix
    return LANG_EXT.get(str(row.get("lang") or "").lower(), ".txt")


def code_path(row: dict, *, prefix: str, ordinal: int) -> str:
    raw = row.get("_file_name") or row.get("_func_name") or f"snippet{ordinal}{lang_ext(row)}"
    base = safe_slug(Path(raw).name, fallback=f"snippet{ordinal}{lang_ext(row)}")
    if "." not in base:
        base += lang_ext(row)
    return f"{prefix}/{ordinal:02d}_{base}"


def repo_id_for(row: dict, idx: int) -> str:
    cwe = safe_slug(str(row.get("cwe") or "unknown"))
    lang = safe_slug(str(row.get("lang") or "unknown"))
    source = safe_slug(str(row.get("source") or "dataset"))
    digest_src = f"{idx}:{pair_group_key(row)}:{row.get('cwe')}:{row.get('code', '')[:200]}"
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:10]
    return f"{source}-{lang}-{cwe}-{digest}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def pick_decoys(
    rng: random.Random,
    negatives: list[IndexedRow],
    *,
    lang: Optional[str],
    exclude_groups: set[str],
    count: int,
) -> list[IndexedRow]:
    pool = [
        item for item in negatives
        if pair_group_key(item.row) not in exclude_groups
        and (lang is None or item.row.get("lang") == lang)
        and (item.row.get("code") or "").strip()
    ]
    if len(pool) < count:
        pool = [
            item for item in negatives
            if pair_group_key(item.row) not in exclude_groups
            and (item.row.get("code") or "").strip()
        ]
    if not pool or count <= 0:
        return []
    return rng.sample(pool, k=min(count, len(pool)))


def build_repo(
    out_root: Path,
    pos: IndexedRow,
    *,
    fixed: Optional[IndexedRow],
    decoys: list[IndexedRow],
) -> dict:
    row = pos.row
    repo_id = repo_id_for(row, pos.idx)
    repo_dir = out_root / "repos" / repo_id
    manifest_path = out_root / "manifests" / f"{repo_id}.json"

    vuln_rel = code_path(row, prefix="src", ordinal=0)
    vuln_code = row["code"]
    write_text(repo_dir / vuln_rel, vuln_code)

    files = [{
        "path": vuln_rel,
        "role": "vulnerable",
        "source_row": pos.idx,
        "label": 1,
    }]
    traces = trace_records(vuln_rel, vuln_code, spans_from_row(row), row.get("cwe"))
    negative_regions: list[dict] = []

    if fixed is not None:
        fixed_rel = code_path(fixed.row, prefix="fixed", ordinal=0)
        fixed_code = fixed.row["code"]
        write_text(repo_dir / fixed_rel, fixed_code)
        files.append({
            "path": fixed_rel,
            "role": "fixed_counterpart",
            "source_row": fixed.idx,
            "label": 0,
        })
        negative_regions.append({
            "file": fixed_rel,
            "start_line": 1,
            "end_line": max(1, len(fixed_code.splitlines())),
            "role": "fixed_counterpart",
        })

    for j, decoy in enumerate(decoys, start=1):
        decoy_rel = code_path(decoy.row, prefix="decoys", ordinal=j)
        decoy_code = decoy.row["code"]
        write_text(repo_dir / decoy_rel, decoy_code)
        files.append({
            "path": decoy_rel,
            "role": "safe_decoy",
            "source_row": decoy.idx,
            "label": 0,
        })
        negative_regions.append({
            "file": decoy_rel,
            "start_line": 1,
            "end_line": max(1, len(decoy_code.splitlines())),
            "role": "safe_decoy",
        })

    readme = [
        f"# {repo_id}",
        "",
        "Synthetic repo-level fixture generated from token-labelled examples.",
        f"CWE: {row.get('cwe') or 'unknown'}",
        f"Language: {row.get('lang') or 'unknown'}",
        "",
        "Ground-truth traces live in the adjacent manifest JSON.",
    ]
    write_text(repo_dir / "README.md", "\n".join(readme))

    manifest = {
        "schema": SCHEMA,
        "repo_id": repo_id,
        "repo_path": str(repo_dir.relative_to(out_root)),
        "source": "dataset_microrepo",
        "source_row": pos.idx,
        "source_dataset": row.get("source"),
        "pair_group": pair_group_key(row),
        "language": row.get("lang"),
        "cwe": row.get("cwe"),
        "label_confidence": row.get("label_confidence") or "inherited",
        "files": files,
        "vulnerability_traces": traces,
        "negative_regions": negative_regions,
        "functional_tests": [],
        "security_oracles": [{
            "kind": "diff_trace",
            "description": "Trace comes from token_labels.evidence/vulnerable_line in the source row.",
        }],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def build_benchmark(
    rows: list[IndexedRow],
    out_root: Path,
    *,
    max_repos: Optional[int],
    decoys_per_repo: int,
    seed: int,
    langs: Optional[set[str]] = None,
    cwes: Optional[set[str]] = None,
    require_fixed_counterpart: bool = False,
) -> list[dict]:
    rng = random.Random(seed)
    positives = [
        item for item in rows
        if int(item.row.get("label", 0)) == 1
        and (item.row.get("code") or "").strip()
        and spans_from_row(item.row)
    ]
    negatives = [
        item for item in rows
        if int(item.row.get("label", 0)) == 0
        and (item.row.get("code") or "").strip()
    ]
    if langs:
        positives = [item for item in positives if item.row.get("lang") in langs]
    if cwes:
        positives = [item for item in positives if item.row.get("cwe") in cwes]

    by_group_neg: dict[str, list[IndexedRow]] = {}
    for item in negatives:
        by_group_neg.setdefault(pair_group_key(item.row), []).append(item)

    rng.shuffle(positives)
    manifests: list[dict] = []
    for pos in positives:
        group = pair_group_key(pos.row)
        fixed_candidates = [
            item for item in by_group_neg.get(group, [])
            if item.row.get("cwe") == pos.row.get("cwe") or item.row.get("cwe") is None
        ]
        fixed = fixed_candidates[0] if fixed_candidates else None
        if require_fixed_counterpart and fixed is None:
            continue
        decoys = pick_decoys(
            rng,
            negatives,
            lang=pos.row.get("lang"),
            exclude_groups={group},
            count=decoys_per_repo,
        )
        manifests.append(build_repo(out_root, pos, fixed=fixed, decoys=decoys))
        if max_repos is not None and len(manifests) >= max_repos:
            break
    return manifests


def write_index(out_root: Path, manifests: list[dict]) -> None:
    index_path = out_root / "manifest.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for manifest in manifests:
            f.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    by_cwe: dict[str, int] = {}
    by_lang: dict[str, int] = {}
    for m in manifests:
        by_cwe[m.get("cwe") or "-"] = by_cwe.get(m.get("cwe") or "-", 0) + 1
        by_lang[m.get("language") or "-"] = by_lang.get(m.get("language") or "-", 0) + 1

    summary = {
        "schema": SCHEMA,
        "n_repos": len(manifests),
        "by_cwe": dict(sorted(by_cwe.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_language": dict(sorted(by_lang.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_filter(values: Optional[list[str]]) -> Optional[set[str]]:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        out.update(x.strip() for x in value.split(",") if x.strip())
    return out or None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--max-repos", type=int, default=200)
    ap.add_argument("--decoys-per-repo", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lang", action="append", default=None, help="Language filter; repeat or comma-separate.")
    ap.add_argument("--cwe", action="append", default=None, help="CWE filter; repeat or comma-separate.")
    ap.add_argument(
        "--require-fixed-counterpart",
        action="store_true",
        help="Skip positive rows that do not have a same-group negative/fixed row.",
    )
    args = ap.parse_args(argv)

    rows = load_jsonl(args.input)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifests = build_benchmark(
        rows,
        args.out_root,
        max_repos=args.max_repos,
        decoys_per_repo=args.decoys_per_repo,
        seed=args.seed,
        langs=parse_filter(args.lang),
        cwes=parse_filter(args.cwe),
        require_fixed_counterpart=args.require_fixed_counterpart,
    )
    write_index(args.out_root, manifests)
    print(f"[repo-ds] wrote {len(manifests)} repos under {args.out_root}", file=sys.stderr)
    print(f"[repo-ds] index: {args.out_root / 'manifest.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
