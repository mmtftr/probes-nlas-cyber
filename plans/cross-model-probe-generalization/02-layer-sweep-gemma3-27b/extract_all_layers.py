# [ai-generated]
"""Stream per-token hidden states for EVERY layer of one model to disk.

Step 2 of the cross-model plan (fine-grained layer sweep). The coarse extractor
(`src/data/extract_token_activations.py`) accumulates all captured layers in RAM
then writes once — fine for 4 layers, but all 62 layers of Gemma-3-27B is
~275 GB (413k tokens x 5376 dims x 2 bytes), which won't fit in RAM. So we stream
each layer to its own float16 memmap on scratch, writing one row-slice at a time.

Reuses the exact label/offset logic and robust loaders from the shared pipeline so
the activations match what `train_one_layer` was trained on overnight.

Idempotent: writes `DONE_EXTRACT` when complete; re-running is a no-op.

Output (under --out):
  layer_{NN}.npy   per-layer float16 memmap, shape (T_total, H)
  y.npy            int8  (T_total,)   per-token positive-span label
  example_ids.npy  int32 (T_total,)
  offsets.npz      offsets_row_{NNNN} (T_row, 2)  — parity with the coarse path
  meta.json        {model, n_layers, hidden, n_tokens, n_rows, max_length}
  DONE_EXTRACT     marker
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE_EXTRACT").exists():
        print(f"[extract-all] {out}/DONE_EXTRACT exists — skipping", file=sys.stderr)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[extract-all] device={device} model={args.model}", file=sys.stderr)

    tokenizer = _load_tokenizer(args.model)
    model = _load_model(args.model, dtype)
    model.to(device).eval()

    rows = []
    with open(args.pairs) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # ---- pass 1: tokenize everything; compute total tokens + per-row slices ----
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
    print(f"[extract-all] {len(rows)} rows, {total} tokens total", file=sys.stderr)

    # ---- discover n_layers + hidden from one forward ----
    with torch.inference_mode():
        ids0 = torch.tensor([enc_ids[0]], dtype=torch.long, device=device)
        hs0 = model(ids0, output_hidden_states=True, use_cache=False).hidden_states
    n_layers = len(hs0) - 1  # hidden_states = embeddings + one per layer
    hidden = hs0[-1].shape[-1]
    print(f"[extract-all] n_layers={n_layers} hidden={hidden}", file=sys.stderr)

    # ---- allocate per-layer memmaps on disk (float32) ----
    # float32, NOT float16: Gemma-3 has "massive activations" (>65504) in mid
    # layers that saturate to inf in f16 -> NaN training. The coarse extractor
    # also stores float32. 551 GB on scratch (535 TB free) is fine.
    mmaps = [
        np.lib.format.open_memmap(
            out / f"layer_{li:02d}.npy", mode="w+", dtype=np.float32, shape=(total, hidden)
        )
        for li in range(n_layers)
    ]
    y = np.zeros(total, dtype=np.int8)
    example_ids = np.zeros(total, dtype=np.int32)
    offsets_per_row: list[np.ndarray] = []

    # ---- pass 2: forward each row, write each layer's slice ----
    t0 = time.time()
    with torch.inference_mode():
        for eid, row in enumerate(rows):
            ids = enc_ids[eid]
            offsets = enc_offsets[eid]
            n = len(ids)
            s = row_start[eid]
            ids_t = torch.tensor([ids], dtype=torch.long, device=device)
            hs = model(ids_t, output_hidden_states=True, use_cache=False).hidden_states
            for li in range(n_layers):
                h = hs[li + 1][0].float().cpu().numpy()  # (n, hidden) float32
                mmaps[li][s : s + n] = h
            # labels (identical mapping to the coarse extractor)
            tok_spans = char_spans_to_token_spans(parse_spans(row), offsets)
            tok_labels, _mask = token_labels_array(n, tok_spans)
            y[s : s + n] = tok_labels
            example_ids[s : s + n] = eid
            offsets_per_row.append(np.array(offsets, dtype=np.int32))
            if (eid + 1) % 100 == 0:
                rate = (eid + 1) / (time.time() - t0)
                print(f"[extract-all] {eid+1}/{len(rows)} rows  {rate:.2f} ex/s", file=sys.stderr)

    for m in mmaps:
        m.flush()
    np.save(out / "y.npy", y)
    np.save(out / "example_ids.npy", example_ids)
    np.savez(
        out / "offsets.npz",
        **{f"offsets_row_{i:04d}": offsets_per_row[i] for i in range(len(offsets_per_row))},
    )
    meta = {
        "model": args.model, "n_layers": n_layers, "hidden": hidden,
        "n_tokens": int(total), "n_rows": len(rows), "max_length": args.max_length,
        "pos_tokens": int(y.sum()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "DONE_EXTRACT").write_text(json.dumps(meta))
    print(f"[extract-all] done in {time.time()-t0:.0f}s  pos_tokens={int(y.sum())}", file=sys.stderr)


if __name__ == "__main__":
    main()
