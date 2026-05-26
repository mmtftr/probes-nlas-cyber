"""Score every token in a JSONL dataset with a linear probe and dump the
per-row probabilities + char-offset arrays consumed by ``scripts/run_token_eval.py``.

This is the missing producer in the token-level eval pipeline:
``src/extract_token_activations.py`` saves token-level hidden states (heavy)
plus an ``offsets.npz``; this script does a single forward pass per row,
applies the probe inline, and writes the two compact .npz files the eval
CLI expects.

Outputs (under ``--out-dir``):
  - ``token_probs.npz``   — ``probs_row_NNNN``   (T,) float32, one per row
  - ``token_offsets.npz`` — ``offsets_row_NNNN`` (T, 2) int32, one per row

Run on a CUDA box (single A100 plenty for E2B/E4B + ~1.2k rows):

    python scripts/extract_token_probs.py \
        --model google/gemma-4-E2B-it \
        --probe data/probe_spanmax.npz \
        --dataset data/dataset.jsonl \
        --out-dir data/token_activations
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.probe_io import load_probe  # noqa: E402


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--probe", default="data/probe_spanmax.npz",
                    help="Path to a probe .npz with keys w/b/layer.")
    ap.add_argument("--dataset", default="data/dataset.jsonl")
    ap.add_argument("--out-dir", default="data/token_activations")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None,
                    help="Truncate to first N rows (for quick smoke runs).")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset)

    probe = load_probe(args.probe)
    print(f"[probs] probe layer={probe.layer} dim={probe.w.shape[0]} b={probe.b:+.4f} source={probe.source}", file=sys.stderr)

    device = _device()
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"[probs] device={device} dtype={dtype} model={args.model}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, attn_implementation="eager"
    )
    model.to(device).eval()
    print(f"[probs] model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    rows: list[dict] = []
    with dataset_path.open() as f:
        for i, line in enumerate(f):
            if args.limit is not None and i >= args.limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[probs] dataset rows={len(rows)} (limit={args.limit})", file=sys.stderr)

    # Cast probe weights to a torch tensor on-device (bf16/fp16 friendly via float math at the end).
    w_t = torch.tensor(probe.w, dtype=torch.float32, device=device)
    b_t = torch.tensor(probe.b, dtype=torch.float32, device=device)

    probs_per_row: dict[str, np.ndarray] = {}
    offsets_per_row: dict[str, np.ndarray] = {}

    t_loop = time.time()
    last_print = t_loop
    with torch.inference_mode():
        for eid, row in enumerate(rows):
            code = row.get("code") or ""
            enc = tokenizer(
                code,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors=None,
            )
            input_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]
            ids_t = torch.tensor([input_ids], dtype=torch.long, device=device)

            out = model(ids_t, output_hidden_states=True, use_cache=False)
            # hidden_states tuple has len = n_layers + 1 (embedding + each block output).
            # The convention matches src/extract_activations.py: probe layer index L
            # reads hidden_states[L + 1].
            h = out.hidden_states[probe.layer + 1][0, :, :].to(torch.float32)
            logits = h @ w_t + b_t
            probs = torch.sigmoid(logits).detach().to("cpu").numpy().astype(np.float32)

            key_probs = f"probs_row_{eid:04d}"
            key_offs = f"offsets_row_{eid:04d}"
            probs_per_row[key_probs] = probs
            offsets_per_row[key_offs] = np.asarray(offsets, dtype=np.int32)

            now = time.time()
            if now - last_print > 5.0:
                rate = (eid + 1) / max(now - t_loop, 0.01)
                eta = (len(rows) - (eid + 1)) / max(rate, 1e-3)
                print(f"[probs] {eid+1}/{len(rows)} {rate:.2f} ex/s eta {eta:.0f}s", file=sys.stderr)
                last_print = now

    elapsed = time.time() - t_loop
    print(f"[probs] forward+score done in {elapsed:.1f}s ({len(rows)/max(elapsed,1e-3):.2f} ex/s)", file=sys.stderr)

    probs_path = out_dir / "token_probs.npz"
    offsets_path = out_dir / "token_offsets.npz"
    np.savez(probs_path, **probs_per_row)
    np.savez(offsets_path, **offsets_per_row)
    print(f"[probs] wrote {probs_path} ({len(probs_per_row)} rows)", file=sys.stderr)
    print(f"[probs] wrote {offsets_path} ({len(offsets_per_row)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
