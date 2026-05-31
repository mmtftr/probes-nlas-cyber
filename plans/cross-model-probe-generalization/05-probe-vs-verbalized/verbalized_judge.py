# [ai-generated]
"""Verbalized vulnerability judgment: ask the model itself, read P(yes).

For each SVEN dataset row, build a NEUTRAL input-stream prompt with the code
BEFORE the eliciting question (so the code is encoded the same way the probe's
cached activations were — on raw code, no question conditioning the encoding),
forward once, and read the model's own yes/no answer probability at the first
assistant-token position.

    PREAMBLE  = "Here is a code snippet:"
    QUESTION  = "Does the code above contain a security vulnerability? "
                "Answer with a single word: yes or no."
    content   = f"{PREAMBLE}\n\n```\n{code}\n```\n\n{QUESTION}"

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, return_tensors="pt")

P(yes) = sigmoid( logsumexp(logprob[yes_ids]) - logsumexp(logprob[no_ids]) )
where logprob = log_softmax(logits[-1]) (the next-token distribution).

This step LOADS THE MODEL (unlike the cached-acts experiments 03/04) — it is the
forward-pass half of the comparison. The comparison itself (probe vs. these
scores) is done WITHOUT the model in compare_probe_vs_verbalized.py.

GPU-sharded by dataset row index (idx % n_gpus == gpu_id). Each worker writes
its own shard `verbalized_scores.gpu{gpu_id}.json` into the --out directory and
skips it if it already exists (resumable). The aggregator merges all shards.

Model loading mirrors src/data/extract_token_activations.py
(`_load_model` / `_load_tokenizer`, dtype = bfloat16 on cuda).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.data.extract_token_activations import _load_model, _load_tokenizer  # noqa: E402


PREAMBLE = "Here is a code snippet:"
QUESTION = (
    "Does the code above contain a security vulnerability? "
    "Answer with a single word: yes or no."
)


def build_content(code: str) -> str:
    """Code BEFORE the question, minimal neutral preamble, fenced code block."""
    return f"{PREAMBLE}\n\n```\n{code}\n```\n\n{QUESTION}"


def resolve_yes_no_ids(tokenizer):
    """First-token ids for yes/no across common casings/leading-space variants."""
    yes_words = ["Yes", "yes", " Yes", " yes", "YES"]
    no_words = ["No", "no", " No", " no", "NO"]

    def first_ids(ws):
        ids = set()
        for w in ws:
            enc = tokenizer.encode(w, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
        return sorted(ids)

    return first_ids(yes_words), first_ids(no_words)


def p_yes_from_logits(logits_last, yes_ids, no_ids):
    """P(yes) over the yes/no token sets from a single next-token logit vector.

    logits_last: 1-D array/tensor of vocab logits at the last position.
    Returns a python float in (0, 1).
    """
    import torch

    if not torch.is_tensor(logits_last):
        logits_last = torch.as_tensor(np.asarray(logits_last, dtype=np.float64))
    logprobs = torch.log_softmax(logits_last.to(torch.float64), dim=-1)
    yes_lp = torch.logsumexp(logprobs[yes_ids], dim=-1)
    no_lp = torch.logsumexp(logprobs[no_ids], dim=-1)
    return float(torch.sigmoid(yes_lp - no_lp).item())


def _row_label(row: dict) -> int:
    return int(row.get("label", row.get("vulnerable", 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True, help="dataset.jsonl")
    ap.add_argument("--out", required=True,
                    help="OUTPUT DIRECTORY; shard files verbalized_scores.gpu{id}.json land here")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"verbalized_scores.gpu{args.gpu_id}.json"
    if shard.exists():
        print(f"[verbalized] gpu {args.gpu_id}: shard exists, skip ({shard})", file=sys.stderr)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = _load_tokenizer(args.model)
    model = _load_model(args.model, dtype)
    model.to(device).eval()
    print(f"[verbalized] gpu {args.gpu_id}: device={device} dtype={dtype} model={args.model}",
          file=sys.stderr)

    yes_ids, no_ids = resolve_yes_no_ids(tokenizer)
    print(f"[verbalized] yes_ids={yes_ids} -> {[tokenizer.decode([i]) for i in yes_ids]!r}",
          file=sys.stderr)
    print(f"[verbalized] no_ids ={no_ids} -> {[tokenizer.decode([i]) for i in no_ids]!r}",
          file=sys.stderr)
    if not yes_ids or not no_ids:
        raise SystemExit("[verbalized] could not resolve yes/no token ids")

    rows = [json.loads(l) for l in Path(args.pairs).open() if l.strip()]
    mine = [i for i in range(len(rows)) if i % args.n_gpus == args.gpu_id]
    print(f"[verbalized] gpu {args.gpu_id}/{args.n_gpus}: {len(mine)}/{len(rows)} rows",
          file=sys.stderr)

    results = []
    printed_template = False
    with torch.inference_mode():
        for eid in mine:
            row = rows[eid]
            code = row["code"]
            # Truncate the CODE tokenization to max_length (the question + chat
            # scaffolding are short and always kept).
            code_ids = tokenizer.encode(code, add_special_tokens=False,
                                        truncation=True, max_length=args.max_length)
            code_trunc = tokenizer.decode(code_ids)
            content = build_content(code_trunc)
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True, return_tensors="pt",
            ).to(device)

            if not printed_template:
                rendered = tokenizer.decode(input_ids[0].tolist())
                print("[verbalized] ===== first rendered chat template (decoded) =====",
                      file=sys.stderr)
                print(rendered, file=sys.stderr)
                print("[verbalized] ===== end template =====", file=sys.stderr)
                printed_template = True

            out = model(input_ids, use_cache=False)
            logits_last = out.logits[0, -1, :]
            p_yes = p_yes_from_logits(logits_last, yes_ids, no_ids)
            results.append({"eid": int(eid), "p_yes": float(p_yes), "label": _row_label(row)})

            if len(results) % 50 == 0:
                print(f"[verbalized] gpu {args.gpu_id}: {len(results)}/{len(mine)}",
                      file=sys.stderr)

    shard.write_text(json.dumps(results))
    print(f"[verbalized] gpu {args.gpu_id}: wrote {len(results)} scores -> {shard}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
