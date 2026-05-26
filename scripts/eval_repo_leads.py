#!/usr/bin/env python3
"""Evaluate repo-level scan lead JSONL files against benchmark manifests.

Expected layout:
  <leads-dir>/<repo_id>.jsonl

Each file should contain records emitted by `python -m src.scan`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.repo_protocol import full_repo_report, load_leads_jsonl, load_manifest_index


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=Path("data/repo_benchmark/manifest.jsonl"))
    ap.add_argument("--leads-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=Path("data/eval/repo_scan_report.json"))
    ap.add_argument("--k", type=int, action="append", default=None)
    args = ap.parse_args(argv)

    manifests = load_manifest_index(args.manifest)
    leads_by_repo = {
        str(m["repo_id"]): load_leads_jsonl(args.leads_dir / f"{m['repo_id']}.jsonl")
        for m in manifests
    }
    k_values = args.k or [1, 5, 10, 25]
    report = full_repo_report(manifests, leads_by_repo, k_values=k_values)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[repo-eval] wrote {args.out_json}")
    print(f"[repo-eval] repo_recall@K: {report.repo_recall_at_k}")
    print(f"[repo-eval] MRR: {report.mean_reciprocal_rank:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
