# [ai-generated]
"""Sharded copy of 02-.../extract_all_layers.py for 4-node (16-rank) extraction.

Identical extraction logic, format, dtype (float32 storage / bf16 compute), label
+ offset mapping, and VLM-fallback loaders as the single-node extractor — but each
invocation processes only a CONTIGUOUS slice of the dataset rows, controlled by
`--shard-id N --n-shards K`.

Row range for shard N of K over a dataset of R rows:
    lo = floor(N * R / K)
    hi = floor((N + 1) * R / K)
    this shard handles rows[lo:hi]   (the dataset's natural order)

Because the slices are contiguous and partition [0, R) in shard order, simply
concatenating shard outputs in ascending shard order reproduces the original 1-node
row order exactly. `merge_shards.py` does that.

CRITICAL — global identity is preserved:
  - `example_ids.npy` stores the GLOBAL eid (the absolute dataset row index `lo+i`),
    NOT a shard-local index. Downstream splits key on the true eid.
  - `offsets.npz` keys are `offsets_row_{global_eid:04d}` — same key scheme as the
    1-node extractor, so the merged npz has every original key exactly once.

Per-shard output (under --out):
  layer_{NN}.npy   per-layer float32 memmap, shape (T_shard, H)  (this shard's tokens)
  y.npy            int8  (T_shard,)   per-token positive-span label
  example_ids.npy  int32 (T_shard,)   GLOBAL eid per token
  offsets.npz      offsets_row_{global_eid:04d} (T_row, 2) int32, this shard's rows
  meta.json        {model, n_layers, hidden, n_tokens, n_rows (this shard),
                    max_length, pos_tokens, shard_id, n_shards, row_lo, row_hi,
                    n_rows_total, dropped_fraction_canary?}
  DONE_SHARD       marker (json copy of meta)

Idempotent: writes DONE_SHARD when complete; re-running an existing shard is a no-op.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.data.extract_token_activations import _load_tokenizer, _load_model  # noqa: E402
from src.eval.token_data import (  # noqa: E402
    char_spans_to_token_spans,
    parse_spans,
    token_labels_array,
)


def shard_bounds(n_rows_total: int, shard_id: int, n_shards: int) -> tuple[int, int]:
    """Contiguous half-open [lo, hi) row range for shard `shard_id` of `n_shards`.

    floor(N*R/K) : floor((N+1)*R/K). Partitions [0, R) with no gaps/overlaps, so
    concatenation in shard order == original row order. Empty shards (R < K) are
    allowed and produce a valid zero-row shard.
    """
    if not (0 <= shard_id < n_shards):
        raise ValueError(f"shard_id {shard_id} out of range [0,{n_shards})")
    lo = (shard_id * n_rows_total) // n_shards
    hi = ((shard_id + 1) * n_rows_total) // n_shards
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE_SHARD").exists():
        print(f"[extract-shard] {out}/DONE_SHARD exists — skipping", file=sys.stderr)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[extract-shard] shard {args.shard_id}/{args.n_shards} "
          f"device={device} model={args.model}", file=sys.stderr)

    # ---- read ALL rows, then carve this shard's contiguous slice ----
    all_rows = []
    with open(args.pairs) as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))
    n_rows_total = len(all_rows)
    lo, hi = shard_bounds(n_rows_total, args.shard_id, args.n_shards)
    rows = all_rows[lo:hi]
    # GLOBAL eids for this shard: absolute dataset row indices lo..hi-1.
    global_eids = list(range(lo, hi))
    print(f"[extract-shard] total_rows={n_rows_total} shard rows=[{lo}:{hi}) "
          f"({len(rows)} rows)", file=sys.stderr)

    tokenizer = _load_tokenizer(args.model)
    model = _load_model(args.model, dtype)
    model.to(device).eval()

    # ---- pass 1: tokenize this shard's rows; total tokens + per-row slices ----
    enc_ids: list[list[int]] = []
    enc_offsets: list[list[tuple[int, int]]] = []
    row_start: list[int] = []
    total = 0
    for row in rows:
        enc = tokenizer(
            row["code"], return_offsets_mapping=True, truncation=True,
            max_length=args.max_length, return_tensors=None,
        )
        ids = enc["input_ids"]
        enc_ids.append(ids)
        enc_offsets.append(enc["offset_mapping"])
        row_start.append(total)
        total += len(ids)
    print(f"[extract-shard] {len(rows)} rows, {total} tokens (this shard)", file=sys.stderr)

    if len(rows) == 0:
        # Degenerate empty shard (n_shards > n_rows): emit a valid zero-row shard so
        # merge can still assert coverage. n_layers/hidden unknown without a forward;
        # do one tiny dummy forward to record them (keeps shard metas consistent).
        raise SystemExit(
            "[extract-shard] empty shard: n_shards exceeds n_rows; choose n_shards<=n_rows"
        )

    # ---- discover n_layers + hidden from one forward (model-constant) ----
    with torch.inference_mode():
        ids0 = torch.tensor([enc_ids[0]], dtype=torch.long, device=device)
        hs0 = model(ids0, output_hidden_states=True, use_cache=False).hidden_states
    n_layers = len(hs0) - 1  # hidden_states = embeddings + one per layer
    hidden = hs0[-1].shape[-1]
    print(f"[extract-shard] n_layers={n_layers} hidden={hidden}", file=sys.stderr)

    # ---- allocate per-layer memmaps on disk (float32 — do NOT change dtype) ----
    # float32, NOT float16: Gemma-3 / large-model mid layers have massive activations
    # (>65504) that saturate to inf in f16 -> NaN training. Storage sized to THIS
    # shard's token count.
    mmaps = [
        np.lib.format.open_memmap(
            out / f"layer_{li:02d}.npy", mode="w+", dtype=np.float32, shape=(total, hidden)
        )
        for li in range(n_layers)
    ]
    y = np.zeros(total, dtype=np.int8)
    example_ids = np.zeros(total, dtype=np.int32)
    offsets_per_row: list[np.ndarray] = []

    # ---- pass 2: forward each row, write each layer's slice (GLOBAL eid tagging) ----
    t0 = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            geid = global_eids[i]
            ids = enc_ids[i]
            offsets = enc_offsets[i]
            n = len(ids)
            s = row_start[i]
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            hs = model(ids_t, output_hidden_states=True, use_cache=False).hidden_states
            for li in range(n_layers):
                h = hs[li + 1][0].float().cpu().numpy()  # (n, hidden) float32
                mmaps[li][s : s + n] = h
            # labels (identical mapping to the coarse / 1-node extractor)
            tok_spans = char_spans_to_token_spans(parse_spans(row), offsets)
            tok_labels, _mask = token_labels_array(n, tok_spans)
            y[s : s + n] = tok_labels
            example_ids[s : s + n] = geid  # GLOBAL eid, not shard-local
            offsets_per_row.append(np.array(offsets, dtype=np.int32))
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"[extract-shard] {i+1}/{len(rows)} rows  {rate:.2f} ex/s", file=sys.stderr)

    for m in mmaps:
        m.flush()
    np.save(out / "y.npy", y)
    np.save(out / "example_ids.npy", example_ids)
    np.savez(
        out / "offsets.npz",
        **{f"offsets_row_{global_eids[i]:04d}": offsets_per_row[i]
           for i in range(len(offsets_per_row))},
    )
    meta = {
        "model": args.model, "n_layers": n_layers, "hidden": hidden,
        "n_tokens": int(total), "n_rows": len(rows), "max_length": args.max_length,
        "pos_tokens": int(y.sum()),
        # shard provenance (consumed by merge_shards.py for ordering + assertions)
        "shard_id": args.shard_id, "n_shards": args.n_shards,
        "row_lo": lo, "row_hi": hi, "n_rows_total": n_rows_total,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "DONE_SHARD").write_text(json.dumps(meta))
    print(f"[extract-shard] shard {args.shard_id} done in {time.time()-t0:.0f}s  "
          f"pos_tokens={int(y.sum())}", file=sys.stderr)


if __name__ == "__main__":
    main()
