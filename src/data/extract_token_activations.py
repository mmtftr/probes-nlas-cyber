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


def _load_tokenizer(model_id: str):
    """Load a fast tokenizer robustly across model families.

    Bare `AutoTokenizer.from_pretrained` fails for two roster cases:
      - custom-code repos (OpenCoder) that need `trust_remote_code=True`;
        without it the loader hits an interactive `input()` that EOFs in batch.
      - VLM configs whose config type isn't in `TOKENIZER_MAPPING`
        (Mistral3 -> `KeyError`, Tekken-based Devstral -> sentencepiece
        `TypeError`). Those route through `AutoProcessor`, whose `.tokenizer`
        is the fast tokenizer we want (it still serves `return_offsets_mapping`).
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as tok_err:  # noqa: BLE001 — fall back, re-raise if no tokenizer
        # Last resort for VLM repos that only expose the tokenizer through a
        # processor. If that path also fails, re-raise the ORIGINAL tokenizer
        # error — the processor's failure (e.g. a missing image processor) would
        # otherwise mask the real cause.
        tok = None
        try:
            from transformers import AutoProcessor

            proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            tok = getattr(proc, "tokenizer", None)
        except Exception:  # noqa: BLE001
            tok = None
        if tok is None:
            raise tok_err
        print(f"[token-extract] tokenizer via AutoProcessor fallback ({type(proc).__name__})", file=sys.stderr)
        return tok


def _load_model(model_id: str, dtype):
    """Load a causal LM, falling back to the VLM wrapper's text decoder.

    `transformers` <5 takes `torch_dtype`; >=5 renamed it to `dtype` — try the
    new kwarg first, fall back to the legacy one. VLM wrappers (Mistral3,
    gemma-4) aren't registered under `AutoModelForCausalLM`; load them via
    `AutoModelForImageTextToText` and let the layer-walk below find the text
    decoder. A text-only forward (input_ids, no pixel_values) still returns the
    decoder's `hidden_states`.
    """
    from transformers import AutoModelForCausalLM

    def _from(cls):
        try:
            return cls.from_pretrained(
                model_id, dtype=dtype, attn_implementation="eager", trust_remote_code=True
            )
        except TypeError:
            return cls.from_pretrained(
                model_id, torch_dtype=dtype, attn_implementation="eager", trust_remote_code=True
            )

    try:
        return _from(AutoModelForCausalLM)
    except (ValueError, KeyError, OSError):
        # ValueError/KeyError = arch not registered under CausalLM; OSError = the
        # CausalLM checkpoint-shard index references shards the VLM checkpoint
        # doesn't have (e.g. Qwen3.6-27B's `model-000NN-of-00015.safetensors`).
        # Both mean "this isn't a plain CausalLM" -> use the VLM text-decoder path.
        # (Without OSError here, concurrent loads of Qwen3.6 fail nondeterministically
        # depending on a remote-code arch-registration race.)
        from transformers import AutoModelForImageTextToText

        print("[token-extract] CausalLM load failed -> AutoModelForImageTextToText", file=sys.stderr)
        return _from(AutoModelForImageTextToText)


def extract(
    model_id: str,
    jsonl_path: Path,
    out_dir: Path,
    layer_indices: list[int] | None = None,
    max_length: int = 2048,
) -> None:
    device = _device()
    print(f"[token-extract] device={device}  model={model_id}", file=sys.stderr)

    tokenizer = _load_tokenizer(model_id)
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    t0 = time.time()
    model = _load_model(model_id, dtype)
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


def extract_vllm(
    model_id: str,
    jsonl_path: Path,
    out_dir: Path,
    layer_indices: list[int] | None = None,
    max_length: int = 2048,
) -> None:
    """vLLM `extract_hidden_states` backend — ~2.2–2.5× faster than the HF
    forward, byte-identical output format. Reuses the HF tokenizer for
    input_ids/offsets/labels (these don't depend on the forward); vLLM only
    produces the hidden states. Layer convention preserved: repo-layer L =
    hidden_states[L+1] = vLLM aux id (L+1). See docs/vllm-hidden-states-extraction.md.
    """
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # Blackwell guard; harmless on Hopper
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    tokenizer = _load_tokenizer(model_id)
    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # need n_layers to default the layer set + bound aux ids
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers")
    if layer_indices is None:
        layer_indices = sorted({n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers - 1})
    aux_ids = sorted({li + 1 for li in layer_indices})
    print(f"[token-extract:vllm] model={model_id} layers={layer_indices} aux_ids={aux_ids} of {n_layers}", file=sys.stderr)

    # ---- tokenize (identical call to the HF path) ----
    input_ids_list: list[list[int]] = []
    offsets_per_row: list[np.ndarray] = []
    labels: list[int] = []
    example_ids: list[int] = []
    spans: list[tuple[int, int, int]] = []
    pos_span_lengths: list[int] = []
    pos_rows = pos_rows_with_span = 0
    for eid, row in enumerate(rows):
        if _row_label(row) == 1:
            pos_rows += 1
        enc = tokenizer(row["code"], return_offsets_mapping=True, truncation=True,
                        max_length=max_length, return_tensors=None)
        input_ids = enc["input_ids"]; offsets = enc["offset_mapping"]
        n_pos = len(input_ids)
        tok_spans = char_spans_to_token_spans(parse_spans(row), offsets)
        tok_labels, _ = token_labels_array(n_pos, tok_spans)
        row_pos = [(s, e) for (s, e, lbl) in tok_spans if lbl == 1]
        if row_pos:
            pos_rows_with_span += 1
            for s, e in row_pos:
                spans.append((eid, s, e)); pos_span_lengths.append(e - s + 1)
        labels.extend(tok_labels.tolist())
        example_ids.extend([eid] * n_pos)
        offsets_per_row.append(np.array(offsets, dtype=np.int32))
        input_ids_list.append(input_ids)

    # ---- vLLM forward (prefill-only, hidden states via KVConnector) ----
    hs_dir = out_dir / "hs_extract"
    import shutil
    shutil.rmtree(hs_dir, ignore_errors=True); hs_dir.mkdir(parents=True, exist_ok=True)
    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector
    t0 = time.time()
    llm = LLM(
        model=model_id, enable_chunked_prefill=False, enable_prefix_caching=False,
        attention_backend="FLASH_ATTN", max_model_len=max_length + 16, trust_remote_code=True,
        speculative_config={"method": "extract_hidden_states", "num_speculative_tokens": 1,
                            "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": aux_ids}}},
        kv_transfer_config=KVTransferConfig(kv_connector="ExampleHiddenStatesConnector", kv_role="kv_producer",
                                            kv_connector_extra_config={"shared_storage_path": str(hs_dir)}),
    )
    print(f"[token-extract:vllm] engine init {time.time()-t0:.1f}s", file=sys.stderr)
    t1 = time.time()
    outputs = llm.generate([{"prompt_token_ids": ids} for ids in input_ids_list],
                           SamplingParams(max_tokens=1, temperature=0.0))
    print(f"[token-extract:vllm] generate {time.time()-t1:.1f}s "
          f"({len(rows)/(time.time()-t1):.1f} ex/s)", file=sys.stderr)

    # outputs are in request (=input) order; assert + assemble per layer
    aux_index = {aid: j for j, aid in enumerate(aux_ids)}
    per_layer: dict[int, list[np.ndarray]] = {li: [] for li in layer_indices}
    for eid, o in enumerate(outputs):
        obj = example_hidden_states_connector.load_hidden_states(o.kv_transfer_params["hidden_states_path"])
        hs = obj["hidden_states"]
        hs = hs.float().cpu().numpy() if hasattr(hs, "float") else np.asarray(hs, dtype=np.float32)
        if hs.shape[0] != len(input_ids_list[eid]):
            raise SystemExit(f"vllm token mismatch eid {eid}: hs={hs.shape[0]} ids={len(input_ids_list[eid])}")
        for li in layer_indices:
            per_layer[li].append(hs[:, aux_index[li + 1], :].astype(np.float32))
    shutil.rmtree(hs_dir, ignore_errors=True)

    # ---- write identical output format ----
    y = np.array(labels, dtype=np.int8)
    eid_arr = np.array(example_ids, dtype=np.int32)
    out_dir.mkdir(parents=True, exist_ok=True)
    for li in layer_indices:
        mat = np.vstack(per_layer[li]).astype(np.float32)
        p = out_dir / f"token_activations_layer{li:02d}.npz"
        np.savez(p, X=mat, y=y, example_ids=eid_arr)
        print(f"[token-extract:vllm] saved {p}  X={mat.shape}", file=sys.stderr)
    (out_dir / "spans.json").write_text(json.dumps(spans))
    np.savez(out_dir / "offsets.npz",
             **{f"offsets_row_{i:04d}": offsets_per_row[i] for i in range(len(offsets_per_row))})
    print(f"[token-extract:vllm] spans={len(spans)} rows={len(offsets_per_row)} "
          f"pos_rows={pos_rows} with_span={pos_rows_with_span} pos_tokens={int(y.sum())}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--pairs", default="data/dataset.jsonl")
    ap.add_argument("--out", default="data/token_activations")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm",
                    help="vllm (default, ~2.2-2.5x faster, needs vllm on PYTHONPATH) "
                         "or hf (transformers output_hidden_states fallback).")
    ap.add_argument("--layers", default=None,
                    help="Comma-separated explicit layer indices to capture "
                         "(e.g. '23,24,25,26,27'). Default: auto {n/4,n/2,3n/4,n-1}. "
                         "Layer L == hidden_states[L+1] (output of block L).")
    args = ap.parse_args()
    layer_indices = None
    if args.layers:
        layer_indices = sorted({int(x) for x in args.layers.split(",") if x.strip() != ""})
    fn = extract_vllm if args.backend == "vllm" else extract
    fn(args.model, Path(args.pairs), Path(args.out),
       layer_indices=layer_indices, max_length=args.max_length)


if __name__ == "__main__":
    main()
