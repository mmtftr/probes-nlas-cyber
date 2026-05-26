"""Extract per-example hidden states from Gemma 4 for probe training.

Uses HuggingFace transformers (text-only mode) — runs on M1/M2/M3 via the
MPS backend. We pull just the language-model side of the Gemma 4 family
(multimodal wrappers are skipped) to keep the forward pass cheap.

For each (code, label) example in data/pairs.jsonl:
  1. Tokenize the code with Gemma's tokenizer.
  2. Forward pass with `output_hidden_states=True`.
  3. Grab the hidden state at the LAST input token for several candidate
     layers (25%, 50%, 75%, last). The probe will be trained per-layer
     and the best layer picked at training time.
  4. Save (X, y) per layer to data/activations/activations_layer{NN}.npz.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract(model_id: str, jsonl_path: Path, out_dir: Path, layer_indices: list[int] | None = None) -> None:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = _device()
    print(f"[extract] device={device}  model={model_id}", file=sys.stderr)

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    print(f"[extract] model loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    # Inspect layer count.
    inner = model
    for attr in ("model", "transformer", "language_model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    # `inner` should now be the decoder; .layers is the list.
    layers = getattr(inner, "layers", None)
    if layers is None and hasattr(inner, "model"):
        layers = getattr(inner.model, "layers", None)
    n_layers = len(layers) if layers is not None else model.config.num_hidden_layers
    print(f"[extract] decoder has {n_layers} layers", file=sys.stderr)

    if layer_indices is None:
        layer_indices = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})
    print(f"[extract] capturing layers {layer_indices}", file=sys.stderr)

    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[extract] processing {len(rows)} examples…", file=sys.stderr)

    labels = np.array([int(r["label"]) for r in rows], dtype=np.int8)
    activations: dict[int, list[np.ndarray]] = {i: [] for i in layer_indices}

    t_loop = time.time()
    with torch.inference_mode():
        for i, row in enumerate(rows):
            text = row["code"]
            ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model(ids, output_hidden_states=True, use_cache=False)
            # out.hidden_states: tuple of length n_layers+1 (embedding + each layer).
            # We use 1..n (skipping the embedding) so index `li` maps to layer `li`.
            hs = out.hidden_states
            for li in layer_indices:
                # last token, last batch element
                h = hs[li + 1][0, -1, :].detach().to("cpu").float().numpy()
                activations[li].append(h)

            if (i + 1) % 25 == 0:
                rate = (i + 1) / (time.time() - t_loop)
                eta = (len(rows) - (i + 1)) / max(rate, 0.01)
                print(f"[extract] {i+1}/{len(rows)}  {rate:.2f} ex/s  eta {eta:.0f}s", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    for li in layer_indices:
        mat = np.vstack(activations[li]).astype(np.float32)
        out_path = out_dir / f"activations_layer{li:02d}.npz"
        np.savez_compressed(out_path, X=mat, y=labels)
        print(f"[extract] saved {out_path}  shape={mat.shape}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    # Canonical defaults match scripts/build_dataset.py output and
    # src/stream_with_probe.py model — closes #15. Any mismatch here
    # silently invalidates the probe (Gemma-3 activations vs Gemma-4 inference).
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--pairs", default="data/dataset.jsonl")
    ap.add_argument("--out", default="data/activations")
    args = ap.parse_args()

    extract(args.model, Path(args.pairs), Path(args.out))


if __name__ == "__main__":
    main()
