# [ai-generated]
"""Concatenate K shard outputs (from extract_sharded.py) into ONE acts/ dir that is
byte/array-identical in format to a single-node extract_all_layers.py run.

Shards are concatenated in ASCENDING shard order. Because each shard covers a
contiguous, gap-free, non-overlapping row range (shard_bounds), shard-order
concatenation reproduces the original 1-node row order — therefore the original
per-token order, the original y/example_ids ordering, and the original offsets keys.

Memory-safe: each final layer memmap is pre-allocated on disk at the total token
count, then each shard's layer slab is stream-copied (mmap read -> mmap write) into
its token offset. RAM stays ~one slab-chunk, never the full slab.

Output (under --out):
  layer_{NN}.npy   float32 (T_total, H)   shards in order
  y.npy            int8  (T_total,)
  example_ids.npy  int32 (T_total,)       GLOBAL eids, non-decreasing
  offsets.npz      offsets_row_{eid:04d} for ALL rows (union of shard keys)
  meta.json        {model, n_layers, hidden, n_tokens (sum), n_rows (sum),
                    max_length, pos_tokens (sum), n_shards}
  DONE_EXTRACT     marker (so downstream train/aggregate treat this like a 1-node run)

Asserts (fail loud — a silent merge bug corrupts every downstream probe):
  * n_layers / hidden / model / max_length identical across all shards
  * shard row ranges tile [0, n_rows_total) exactly (sorted, contiguous, no gaps)
  * total tokens == sum of shard tokens
  * concatenated example_ids strictly non-decreasing AND == arange-expansion of rows
  * offsets.npz has exactly one key per row, covering every eid in [0, n_rows_total)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Stream-copy chunk (rows of the (T,H) slab) — bounds peak RAM regardless of H.
COPY_CHUNK = 65536


def _load_shard_meta(shard_dir: Path) -> dict:
    if not (shard_dir / "DONE_SHARD").exists():
        raise FileNotFoundError(f"shard not complete (no DONE_SHARD): {shard_dir}")
    return json.loads((shard_dir / "meta.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", required=True,
                    help="Parent dir containing shard_0 .. shard_{K-1}.")
    ap.add_argument("--out", required=True, help="Final acts/ dir to write.")
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--shard-prefix", default="shard_",
                    help="Per-shard subdir name prefix (default 'shard_').")
    args = ap.parse_args()

    shards_dir = Path(args.shards_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE_EXTRACT").exists():
        print(f"[merge] {out}/DONE_EXTRACT exists — skipping", file=sys.stderr)
        return

    shard_dirs = [shards_dir / f"{args.shard_prefix}{i}" for i in range(args.n_shards)]
    metas = [_load_shard_meta(d) for d in shard_dirs]

    # ---- assert model-level consistency across shards ----
    ref = metas[0]
    for k in ("model", "n_layers", "hidden", "max_length"):
        vals = {m[k] for m in metas}
        if len(vals) != 1:
            raise AssertionError(f"shards disagree on {k!r}: {vals}")
    n_layers = ref["n_layers"]
    hidden = ref["hidden"]

    # ---- assert row ranges tile [0, n_rows_total) exactly, in shard order ----
    n_rows_total = ref["n_rows_total"]
    for m in metas:
        if m["n_rows_total"] != n_rows_total:
            raise AssertionError(
                f"shards disagree on n_rows_total: {m['n_rows_total']} vs {n_rows_total}"
            )
    cursor = 0
    for i, m in enumerate(metas):
        if m["shard_id"] != i or m["n_shards"] != args.n_shards:
            raise AssertionError(
                f"shard {i}: meta shard_id/n_shards = {m['shard_id']}/{m['n_shards']}"
            )
        if m["row_lo"] != cursor:
            raise AssertionError(
                f"shard {i}: row_lo {m['row_lo']} != expected contiguous {cursor}"
            )
        if m["row_hi"] < m["row_lo"]:
            raise AssertionError(f"shard {i}: row_hi {m['row_hi']} < row_lo {m['row_lo']}")
        cursor = m["row_hi"]
    if cursor != n_rows_total:
        raise AssertionError(
            f"shard row ranges end at {cursor}, expected n_rows_total {n_rows_total}"
        )

    # ---- token offsets for each shard in the final array ----
    shard_tokens = [int(m["n_tokens"]) for m in metas]
    total_tokens = sum(shard_tokens)
    tok_start = np.cumsum([0] + shard_tokens[:-1]).tolist()
    n_rows = sum(int(m["n_rows"]) for m in metas)
    if n_rows != n_rows_total:
        raise AssertionError(f"sum of shard n_rows {n_rows} != n_rows_total {n_rows_total}")
    print(f"[merge] {args.n_shards} shards  rows={n_rows}  tokens={total_tokens}  "
          f"layers={n_layers}  hidden={hidden}", file=sys.stderr)

    # ---- per-layer memmaps: pre-allocate final, stream-copy each shard slab in ----
    for li in range(n_layers):
        dst = np.lib.format.open_memmap(
            out / f"layer_{li:02d}.npy", mode="w+", dtype=np.float32,
            shape=(total_tokens, hidden),
        )
        for i, d in enumerate(shard_dirs):
            src = np.load(d / f"layer_{li:02d}.npy", mmap_mode="r")
            t = shard_tokens[i]
            if src.shape != (t, hidden):
                raise AssertionError(
                    f"shard {i} layer {li}: shape {src.shape} != expected {(t, hidden)}"
                )
            base = tok_start[i]
            for c0 in range(0, t, COPY_CHUNK):
                c1 = min(c0 + COPY_CHUNK, t)
                dst[base + c0 : base + c1] = src[c0:c1]
            del src
        dst.flush()
        del dst
        if (li + 1) % 8 == 0 or li == n_layers - 1:
            print(f"[merge] copied layer {li+1}/{n_layers}", file=sys.stderr)

    # ---- y, example_ids: concatenate in shard order ----
    y = np.empty(total_tokens, dtype=np.int8)
    eids = np.empty(total_tokens, dtype=np.int32)
    pos_tokens = 0
    for i, d in enumerate(shard_dirs):
        ys = np.load(d / "y.npy")
        es = np.load(d / "example_ids.npy")
        t = shard_tokens[i]
        if ys.shape[0] != t or es.shape[0] != t:
            raise AssertionError(
                f"shard {i}: y/example_ids len {ys.shape[0]}/{es.shape[0]} != n_tokens {t}"
            )
        base = tok_start[i]
        y[base : base + t] = ys
        eids[base : base + t] = es
        pos_tokens += int((ys == 1).sum())
    np.save(out / "y.npy", y)
    np.save(out / "example_ids.npy", eids)

    # ---- assert example_ids: non-decreasing, exactly one contiguous run per eid ----
    if total_tokens > 0:
        if not np.all(eids[1:] >= eids[:-1]):
            raise AssertionError("merged example_ids not non-decreasing — row order broken")
        present = np.unique(eids)
        expected = np.arange(n_rows_total, dtype=eids.dtype)
        if not np.array_equal(present, expected):
            missing = set(expected.tolist()) - set(present.tolist())
            extra = set(present.tolist()) - set(expected.tolist())
            raise AssertionError(
                f"example_ids don't cover [0,{n_rows_total}) exactly; "
                f"missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}"
            )

    # ---- offsets.npz: union of shard keys; assert exactly one key per row ----
    offsets_out: dict[str, np.ndarray] = {}
    for d in shard_dirs:
        npz = np.load(d / "offsets.npz")
        for k in npz.files:
            if k in offsets_out:
                raise AssertionError(f"duplicate offsets key across shards: {k}")
            offsets_out[k] = np.asarray(npz[k])
    expected_keys = {f"offsets_row_{e:04d}" for e in range(n_rows_total)}
    got_keys = set(offsets_out.keys())
    if got_keys != expected_keys:
        miss = sorted(expected_keys - got_keys)[:8]
        extra = sorted(got_keys - expected_keys)[:8]
        raise AssertionError(f"offsets keys mismatch; missing={miss} extra={extra}")
    # Per-row offsets length must equal that row's token count (cross-check vs eids).
    for e in range(n_rows_total):
        n_off = offsets_out[f"offsets_row_{e:04d}"].shape[0]
        n_tok = int((eids == e).sum())
        if n_off != n_tok:
            raise AssertionError(
                f"row {e}: offsets len {n_off} != token count {n_tok}"
            )
    np.savez(out / "offsets.npz", **offsets_out)

    # ---- meta.json: format-identical to a 1-node run (+ n_shards provenance) ----
    meta = {
        "model": ref["model"], "n_layers": n_layers, "hidden": hidden,
        "n_tokens": int(total_tokens), "n_rows": int(n_rows_total),
        "max_length": ref["max_length"], "pos_tokens": int(pos_tokens),
        "n_shards": args.n_shards,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "DONE_EXTRACT").write_text(json.dumps(meta))
    print(f"[merge] DONE  rows={n_rows_total} tokens={total_tokens} "
          f"pos_tokens={pos_tokens} -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
