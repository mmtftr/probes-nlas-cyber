# [ai-generated]
"""Merge per-GPU extraction shards into one per-model npz, with hard coverage gates.

The extractor (extract_lasttoken_hidden.py) writes lasttoken_hidden.gpu{id}.npz
shards (rows sharded by eid % n_gpus). This merges them into
lasttoken_hidden_<slug>.npz, sorted by eid, and HARD-FAILS unless:
  - every dataset eid appears exactly once (full, non-overlapping coverage);
  - every held-out test eid is present;
  - npz label == dataset row['label'] for every eid;
  - all hidden states finite.
It also stamps is_test (from the seed-42 split) so downstream never re-derives it.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


# inlined from train_eval (model-independent, no torch) so merge runs on the
# cluster too (its repo lacks src/remotes/train_eval.py).
def pair_group_key(row: dict) -> str:
    repo = row.get("_origin_repo")
    if repo:
        return f"repo::{repo}"
    fn = row.get("_file_name") or ""
    func = row.get("_func_name") or ""
    if fn or func:
        return f"func::{fn}::{func}"
    return f"row::{hashlib.sha1((row.get('code') or '').encode()).hexdigest()[:12]}"


def load_or_make_split(dataset_path, split_path):
    rows = [json.loads(l) for l in Path(dataset_path).open()]
    eid_to_group = {i: pair_group_key(r) for i, r in enumerate(rows)}
    heldout = set(json.loads(Path(split_path).read_text())["heldout_groups"])
    test_eids = {e for e, g in eid_to_group.items() if g in heldout}
    return rows, {e for e in eid_to_group if e not in test_eids}, test_eids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--out", required=True, help="merged npz path")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()

    shards = sorted(Path(args.shard_dir).glob("lasttoken_hidden.gpu*.npz"))
    if not shards:
        raise SystemExit(f"[merge] no shards under {args.shard_dir}")
    parts = [np.load(s, allow_pickle=True) for s in shards]
    H = np.concatenate([p["H"] for p in parts], 0)
    eid = np.concatenate([p["eid"] for p in parts])
    label = np.concatenate([p["label"] for p in parts]).astype(int)
    pack = {k: np.concatenate([p[k] for p in parts]) for k in ("p_yes", "yes_lp", "no_lp", "margin")}

    order = np.argsort(eid, kind="stable")
    H, eid, label = H[order], eid[order], label[order]
    pack = {k: v[order] for k, v in pack.items()}

    rows, _, test_eids = load_or_make_split(args.dataset, args.split)
    # ---- hard gates ----
    assert len(eid) == len(set(eid.tolist())), "duplicate eids across shards"
    assert eid.tolist() == list(range(len(rows))), \
        f"coverage gap: {len(eid)} eids, expected {len(rows)} contiguous"
    assert set(int(e) for e in test_eids).issubset(set(eid.tolist())), "missing test eids"
    true_label = np.array([int(rows[i]["label"]) for i in range(len(rows))])
    assert (label == true_label).all(), "npz label != dataset row['label']"
    assert np.isfinite(H).all(), "non-finite hidden states in merged H"

    is_test = np.array([int(e) in set(int(t) for t in test_eids) for e in eid], bool)
    np.savez_compressed(args.out, H=H.astype(np.float32), eid=eid.astype(np.int32),
                        label=label.astype(np.int8), is_test=is_test, **pack,
                        meta_model=parts[0].get("meta_model", np.array("?")),
                        meta_max_length=parts[0].get("meta_max_length", np.int32(0)))
    print(f"[merge] {args.out}: n={len(eid)} test={int(is_test.sum())} "
          f"H={H.shape} dtype={H.dtype} OK", file=sys.stderr)


if __name__ == "__main__":
    main()
