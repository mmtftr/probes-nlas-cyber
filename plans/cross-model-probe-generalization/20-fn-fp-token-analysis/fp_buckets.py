# [ai-generated]
"""Population-level lexical bucketing of ALL false-positive spans (fp_corpus.json)
into the unified FP taxonomy, so category counts reflect the full FP set rather
than the stratum-capped curated sample the sub-agents categorized.

Heuristic, token-text-based — a coarse cross-check on the hand taxonomy, not a
replacement for it. Precedence: punctuation -> os/path/cmd -> db-api ->
sql-keyword -> identifier/value.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
fp = json.load(open(HERE / "fp_corpus.json"))

PUNCT = re.compile(r"^[^0-9A-Za-z]+$")
SQL_KW = re.compile(r"(?i)\b(select|insert|update|delete|where|from|set|values|"
                    r"order\s+by|group\s+by|like|join|union)\b|insert\s+into")
SQL_CONN = re.compile(r"(?i)(^|\s)(and|or)(\s|$)")
DB_API = re.compile(r"(?i)(cursor|execute|\.query|order_by|fetchone|fetchall|\.format)")
OSPATH = re.compile(r"(?i)(ssh|svctask|svcinfo|svcuser|chown|rmhost|-name|mpol_|/|\.\.)")


def bucket(tok: str) -> str:
    t = tok.strip()
    if t == "" or PUNCT.match(t):
        return "format/punctuation/quote"
    if OSPATH.search(t):
        return "os-command / path / shell string"
    if DB_API.search(t):
        return "DB/cursor/ORM API call"
    if SQL_KW.search(t) or SQL_CONN.search(t):
        return "SQL keyword/clause"
    return "identifier / column / value / placeholder"


by_kind_bucket = defaultdict(Counter)
overall = Counter()
shared = Counter()
for r in fp:
    b = bucket(r["token_text"])
    overall[b] += 1
    by_kind_bucket[r["kind"]][b] += 1
    if r["n_models"] >= 4:
        shared[b] += 1

order = ["SQL keyword/clause", "identifier / column / value / placeholder",
         "format/punctuation/quote", "DB/cursor/ORM API call",
         "os-command / path / shell string"]
total = sum(overall.values())
print(f"ALL FP spans: {total}")
for b in order:
    print(f"  {overall[b]:5d}  {100*overall[b]/total:5.1f}%  {b}")
print("\nby stratum:")
for k in ("safe_alarm", "spread", "misplaced"):
    tot = sum(by_kind_bucket[k].values())
    print(f"  {k} (n={tot}):")
    for b in order:
        c = by_kind_bucket[k][b]
        if c:
            print(f"      {c:5d}  {100*c/tot:5.1f}%  {b}")
print(f"\ncross-model shared (>=4 models, n={sum(shared.values())}):")
for b in order:
    if shared[b]:
        print(f"  {shared[b]:4d}  {b}")

out = {"all": dict(overall), "by_kind": {k: dict(v) for k, v in by_kind_bucket.items()},
       "shared_ge4": dict(shared), "total": total}
(HERE / "fp_buckets.json").write_text(json.dumps(out, indent=2))
