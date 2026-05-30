"""Capture per-token hidden states from Gemma 4, with per-token vulnerability labels.

This is the proper "span-aware" extraction. For each labelled example row:
  1. Tokenize the full code ONCE, with offset_mapping (special tokens included
     so offsets line up with the exact input we feed the model).
  2. Forward-pass through Gemma 4 with `output_hidden_states=True`.
  3. Map the row's `token_labels` char-range fields (evidence / vulnerable_line
     / sink / source for positive, sanitizer for negative) directly to token
     indices via the offset mapping.
  4. For each captured layer, store hidden states at every token position
     along with the per-token binary label.

Output (under `--out`):
  - `token_activations_layer{NN}.npz` per captured layer, each containing:
        X            : float32 (N_total_positions, hidden_dim)
        y            : int8    (N_total_positions,)
        example_ids  : int32   (N_total_positions,)
  - `spans.json` — list of [example_id, tok_start, tok_end] for every positive
    char-range that mapped to ≥1 token (used by the span-max trainer's max-pool).
  - `offsets.npz` — `offsets_row_NNNN` (T,2) int32 arrays, one per row, in
    the same order as the input dataset (the eval CLI reads this).

This fixes #17: the old code looked for `_vuln_line_text` (which the rebuilt
SVEN builder no longer emits) and fell back to labelling the last 5 tokens of
every positive row. It also used `add_special_tokens=False` for offsets while
feeding the model via `tokenizer.encode(...)` — a 1-token offset shift.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.token_data import (  # noqa: E402
    char_spans_to_token_spans,
    parse_spans,
    token_labels_array,
)


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def positive_char_spans(row: dict) -> list[tuple[int, int]]:
    """Return positive (start, end) char ranges from `row['token_labels']`.

    Thin wrapper exposing the upstream API name used by
    `scripts/smoke_extract_token.py`; delegates to `parse_spans`.
    """
    return [(s.start_char, s.end_char) for s in parse_spans(row) if s.label == 1]


def char_spans_to_token_indices(
    char_spans: list[tuple[int, int]],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Map (start_char, end_char) ranges to inclusive (start_tok, end_tok) ranges."""
    from src.eval.token_data import TokenSpan  # local to avoid cycle
    spans = [TokenSpan(s, e, 1, "wrapper") for s, e in char_spans]
    return [(a, b) for (a, b, _lbl) in char_spans_to_token_spans(spans, offsets)]


def _row_label(row: dict) -> int:
    """Best-effort 0/1 example-level label across known row schemas.

    `dataset.jsonl` carries `label`; `pairs_rich.jsonl` only carries
    `is_completion_vulnerable`. Either is acceptable.
    """
    if "label" in row:
        return int(row["label"])
    return int(bool(row.get("is_completion_vulnerable")))


def extract(
    model_id: str,
    jsonl_path: Path,
    out_dir: Path,
    layer_indices: list[int] | None = None,
    max_length: int = 1024,
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    print(f"[token-extract] device={device}  model={model_id}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    t0 = time.time()
    # transformers <5 takes `torch_dtype`; >=5 renamed it to `dtype`. Try the
    # new kwarg first, fall back to the legacy one (we pin 4.46.x on Clariden).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation="eager"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager"
        )
    model.to(device).eval()
    print(f"[token-extract] loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    inner = model
    for attr in ("model", "transformer", "language_model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    layers = getattr(inner, "layers", None) or getattr(getattr(inner, "model", None), "layers", None)
    n_layers = len(layers) if layers is not None else model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})
    print(f"[token-extract] capturing layers {layer_indices} of {n_layers}", file=sys.stderr)

    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[token-extract] processing {len(rows)} rows…", file=sys.stderr)

    per_layer: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}
    labels: list[int] = []
    example_ids: list[int] = []
    spans: list[tuple[int, int, int]] = []  # (example_id, tok_start_incl, tok_end_incl)
    offsets_per_row: list[np.ndarray] = []

    pos_span_lengths: list[int] = []
    pos_rows_with_span = 0
    pos_rows = 0
    t_loop = time.time()

    with torch.inference_mode():
        for eid, row in enumerate(rows):
            code = row["code"]
            label = _row_label(row)
            if label == 1:
                pos_rows += 1

            enc = tokenizer(
                code,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)

            out = model(ids_t, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            n_pos = ids_t.shape[1]

            char_spans = parse_spans(row)
            tok_spans = char_spans_to_token_spans(char_spans, offsets)
            tok_labels, _mask = token_labels_array(n_pos, tok_spans)

            # Record positive token-spans for the span-max trainer.
            row_pos_spans = [(s, e) for (s, e, lbl) in tok_spans if lbl == 1]
            if row_pos_spans:
                pos_rows_with_span += 1
                for s, e in row_pos_spans:
                    spans.append((eid, s, e))
                    pos_span_lengths.append(e - s + 1)

            for li in layer_indices:
                h = hs[li + 1][0, :, :].detach().to("cpu").float().numpy()  # (n_pos, dim)
                per_layer[li].append(h)
            labels.extend(tok_labels.tolist())
            example_ids.extend([eid] * n_pos)
            offsets_per_row.append(np.array(offsets, dtype=np.int32))

            if (eid + 1) % 50 == 0:
                rate = (eid + 1) / (time.time() - t_loop)
                eta = (len(rows) - (eid + 1)) / max(rate, 0.01)
                print(
                    f"[token-extract] {eid+1}/{len(rows)}  {rate:.2f} ex/s  eta {eta:.0f}s",
                    file=sys.stderr,
                )

    y = np.array(labels, dtype=np.int8)
    eid_arr = np.array(example_ids, dtype=np.int32)
    pos_token_count = int(y.sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    for li in layer_indices:
        mat = np.vstack(per_layer[li]).astype(np.float32)
        p = out_dir / f"token_activations_layer{li:02d}.npz"
        np.savez(p, X=mat, y=y, example_ids=eid_arr)
        print(f"[token-extract] saved {p}  X={mat.shape}", file=sys.stderr)

    (out_dir / "spans.json").write_text(json.dumps(spans))
    np.savez(
        out_dir / "offsets.npz",
        **{f"offsets_row_{i:04d}": offsets_per_row[i] for i in range(len(offsets_per_row))},
    )
    print(f"[token-extract] saved spans.json with {len(spans)} positive spans", file=sys.stderr)
    print(f"[token-extract] saved offsets.npz with {len(offsets_per_row)} rows", file=sys.stderr)

    # Smoke-test diagnostic: did spans come out non-trivial?
    if pos_span_lengths:
        arr = np.array(pos_span_lengths)
        print(
            f"[token-extract] positive rows: {pos_rows}; "
            f"rows with ≥1 mapped pos span: {pos_rows_with_span}; "
            f"span tokens: total={pos_token_count} "
            f"median={int(np.median(arr))} p10={int(np.percentile(arr, 10))} "
            f"p90={int(np.percentile(arr, 90))} max={int(arr.max())}",
            file=sys.stderr,
        )
    else:
        print(
            f"[token-extract] WARNING: no positive spans mapped to tokens "
            f"(pos_rows={pos_rows}). Check that the input has populated token_labels.",
            file=sys.stderr,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--pairs", default="data/dataset.jsonl")
    ap.add_argument("--out", default="data/token_activations")
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()
    extract(args.model, Path(args.pairs), Path(args.out), max_length=args.max_length)


if __name__ == "__main__":
    main()
