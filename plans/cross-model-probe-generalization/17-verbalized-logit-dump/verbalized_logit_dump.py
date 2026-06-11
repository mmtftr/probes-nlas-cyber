# [ai-generated]
"""Verbalized vulnerability judgment — and PERSIST every logit (exp-05's gap).

Exp-05's `verbalized_judge.py` forwarded the model over each SVEN row and saved
only the scalar `p_yes` per example; the raw yes/no first-token logits were
thrown away. That is exactly the gap exp-16 closed for the PROBE side (memory
`persist-token-level-predictions`). This script re-runs the identical verbalized
forward and persists every logit needed to recompute the score.

The forward, chat-template rendering and Qwen3 thinking-mode guard are copied
VERBATIM from exp-05 `verbalized_judge.py` so the score is bit-identical to the
historical run; only the persistence is widened. For each SVEN row we build the
NEUTRAL input-stream prompt (code BEFORE the eliciting question, minimal preamble),
forward once, and read the next-token distribution at the first assistant-token
position. Per example we persist:

    p_yes   = sigmoid(yes_lp - no_lp)                  # exp-05's score, unchanged
    yes_lp  = logsumexp(logprob[yes_ids])              # post-softmax log-prob mass
    no_lp   = logsumexp(logprob[no_ids])
    margin  = yes_lp - no_lp                            # the logit fed to sigmoid
    yes_logits_raw = logits[-1][yes_ids]               # raw PRE-softmax token logits
    no_logits_raw  = logits[-1][no_ids]
    topk_ids, topk_logits                              # top-K first-token logits

where logprob = log_softmax(logits[-1]). This step LOADS THE MODEL (like exp-05);
it is layer-independent (reads the final next-token logits, not a hidden layer) —
so there is NO probe, no cached activations, no layer argument here.

GPU-sharded by row index (idx % n_gpus == gpu_id). Each worker writes its own npz
shard `verbalized_logits.gpu{id}.npz` (skipped if present, resumable) plus a tiny
`verbalized_scores.gpu{id}.json` ({eid, p_yes, label}) for drop-in compatibility
with exp-05's `compare_probe_vs_verbalized.py`. The aggregator merges all shards.

Model loading mirrors src/data/extract_token_activations.py (`_load_model` /
`_load_tokenizer`, dtype = bfloat16 on cuda) — the same load path extraction used.
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
# Exact wording carried VERBATIM from exp-05 verbalized_judge.py (single source of
# truth there). Demands one word so the first assistant token is yes/no. Logged
# into the metrics record's `question` field for provenance.
QUESTION = (
    "Does the code above contain a security vulnerability? "
    'Respond with ONLY one word — "yes" or "no" — and nothing else.'
)


def build_content(code: str) -> str:
    """Code BEFORE the question, minimal neutral preamble, fenced code block."""
    return f"{PREAMBLE}\n\n```\n{code}\n```\n\n{QUESTION}"


def is_qwen3_model(model_name: str) -> bool:
    """True for Qwen3 family, which defaults to a <think> block and REQUIRES
    enable_thinking=False (else the first assistant token is <think>, not yes/no)."""
    return "Qwen3" in str(model_name)


def render_chat(tokenizer, content: str, device, model_name: str):
    """Render the user turn to model-ready input tensors (copied from exp-05).

    THINKING-MODE GUARD (Qwen3): try add_generation_prompt=True with
    enable_thinking=False so the assistant turn opens directly (no <think> block).
    On TypeError: a NON-Qwen3 template ignores the kwarg, so fall back (caller
    WARNs); a Qwen3 template that rejects it ABORTS — without it the first token is
    <think>, which silently invalidates the yes/no read.

    Returns (enc_dict, used_enable_thinking_kwarg: bool).
    """
    import torch

    base_kwargs = dict(add_generation_prompt=True, return_tensors="pt",
                       return_dict=True)
    msgs = [{"role": "user", "content": content}]
    used_kwarg = True
    try:
        enc = tokenizer.apply_chat_template(msgs, enable_thinking=False, **base_kwargs)
    except TypeError as e:
        if is_qwen3_model(model_name):
            raise RuntimeError(
                f"[verbalized] ABORT: model {model_name!r} is a Qwen3 model but "
                "its chat template rejected enable_thinking=False (TypeError). "
                "Without it the first assistant token is <think>, which would "
                "silently invalidate the yes/no read — refusing to produce scores."
            ) from e
        used_kwarg = False
        enc = tokenizer.apply_chat_template(msgs, **base_kwargs)
    if torch.is_tensor(enc):  # robustness across transformers versions
        enc = {"input_ids": enc}
    enc = {k: v.to(device) for k, v in dict(enc).items()}
    return enc, used_kwarg


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


def verbalized_record(logits_last, yes_ids, no_ids, topk=10):
    """Score + EVERY logit needed to recompute it, from one next-token logit vector.

    Returns a dict of python/np scalars + small arrays:
      p_yes, yes_lp, no_lp, margin   — exp-05's score and its components
      yes_logits_raw [|yes_ids|]     — raw PRE-softmax logits at the yes ids
      no_logits_raw  [|no_ids|]      — raw PRE-softmax logits at the no ids
      topk_ids [topk], topk_logits [topk] — top-K first-token logits (yes/no should win)
    p_yes is bit-identical to exp-05's p_yes_from_logits (same float64 formula).
    """
    import torch

    if not torch.is_tensor(logits_last):
        logits_last = torch.as_tensor(np.asarray(logits_last, dtype=np.float64))
    lf = logits_last.to(torch.float64)
    logprobs = torch.log_softmax(lf, dim=-1)
    yes_lp = torch.logsumexp(logprobs[yes_ids], dim=-1)
    no_lp = torch.logsumexp(logprobs[no_ids], dim=-1)
    margin = yes_lp - no_lp
    p_yes = torch.sigmoid(margin)
    k = min(topk, lf.shape[-1])
    topv, topi = torch.topk(lf, k=k)
    return {
        "p_yes": float(p_yes.item()),
        "yes_lp": float(yes_lp.item()),
        "no_lp": float(no_lp.item()),
        "margin": float(margin.item()),
        "yes_logits_raw": lf[yes_ids].to(torch.float32).cpu().numpy(),
        "no_logits_raw": lf[no_ids].to(torch.float32).cpu().numpy(),
        "topk_ids": topi.to(torch.int32).cpu().numpy(),
        "topk_logits": topv.to(torch.float32).cpu().numpy(),
    }


def _row_label(row: dict) -> int:
    return int(row.get("label", row.get("vulnerable", 0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True, help="dataset.jsonl")
    ap.add_argument("--out", required=True,
                    help="OUTPUT DIRECTORY; shard files verbalized_logits.gpu{id}.npz land here")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_npz = out_dir / f"verbalized_logits.gpu{args.gpu_id}.npz"
    shard_json = out_dir / f"verbalized_scores.gpu{args.gpu_id}.json"
    if shard_npz.exists():
        print(f"[verbalized] gpu {args.gpu_id}: shard exists, skip ({shard_npz})", file=sys.stderr)
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

    # Accumulators (one row per example, this shard's examples only).
    eid_col, label_col = [], []
    p_yes_col, yes_lp_col, no_lp_col, margin_col = [], [], [], []
    yes_raw_col, no_raw_col, topk_ids_col, topk_logits_col = [], [], [], []
    scores_json = []
    printed_template = False
    with torch.inference_mode():
        for eid in mine:
            row = rows[eid]
            code = row["code"]
            # Truncate the CODE tokenization to max_length (question + scaffolding short).
            code_ids = tokenizer.encode(code, add_special_tokens=False,
                                        truncation=True, max_length=args.max_length)
            code_trunc = tokenizer.decode(code_ids)
            content = build_content(code_trunc)
            enc, used_thinking_kwarg = render_chat(tokenizer, content, device, args.model)
            input_ids = enc["input_ids"]

            out = model(**enc, use_cache=False)
            logits_last = out.logits[0, -1, :]

            # MANDATORY DEBUG PRINT (once per worker): rendered tail + top-5 argmax,
            # so a human preflight verifies (a) the tail is the assistant turn-start
            # with NO <think> token and (b) yes/no dominate the first-token argmax.
            if not printed_template:
                ids = input_ids[0].tolist()
                rendered = tokenizer.decode(ids)
                tail = tokenizer.decode(ids[-40:])
                top5 = torch.topk(logits_last.float(), k=5)
                top5_tok = [(int(i), tokenizer.decode([int(i)]))
                            for i in top5.indices.tolist()]
                if not used_thinking_kwarg:
                    print("[verbalized] WARNING: tokenizer rejected enable_thinking="
                          "False (TypeError); fell back to plain apply_chat_template. "
                          "If this is a Qwen3 model, VERIFY the tail below has NO "
                          "<think> token.", file=sys.stderr)
                print(f"[verbalized] enc keys={list(enc.keys())} "
                      f"input_ids.shape={tuple(input_ids.shape)} "
                      f"enable_thinking_kwarg_used={used_thinking_kwarg}", file=sys.stderr)
                print("[verbalized] ===== rendered TAIL (last ~40 tokens) — must be "
                      "the assistant turn-start, NO <think> =====", file=sys.stderr)
                print(repr(tail), file=sys.stderr)
                print("[verbalized] ===== top-5 first-token argmax (decoded) — yes/no "
                      "should dominate =====", file=sys.stderr)
                for tid, dec in top5_tok:
                    print(f"[verbalized]   id={tid} -> {dec!r}", file=sys.stderr)
                # HARD THINKING-MODE ASSERTION: refuse to proceed if the top-1
                # first-token argmax begins a <think> block (invalid yes/no read).
                top1_dec = top5_tok[0][1].strip()
                if top1_dec == "<think>" or "think>" in top1_dec:
                    raise RuntimeError(
                        f"[verbalized] ABORT: model {args.model!r} top-1 first-token "
                        f"argmax is {top1_dec!r}, which begins a <think> reasoning "
                        "block — the yes/no read is invalid. Refusing to produce scores.")
                printed_template = True

            rec = verbalized_record(logits_last, yes_ids, no_ids, topk=args.topk)
            lab = _row_label(row)
            eid_col.append(int(eid)); label_col.append(lab)
            p_yes_col.append(rec["p_yes"]); yes_lp_col.append(rec["yes_lp"])
            no_lp_col.append(rec["no_lp"]); margin_col.append(rec["margin"])
            yes_raw_col.append(rec["yes_logits_raw"]); no_raw_col.append(rec["no_logits_raw"])
            topk_ids_col.append(rec["topk_ids"]); topk_logits_col.append(rec["topk_logits"])
            scores_json.append({"eid": int(eid), "p_yes": rec["p_yes"], "label": lab})

            if len(eid_col) % 50 == 0:
                print(f"[verbalized] gpu {args.gpu_id}: {len(eid_col)}/{len(mine)}",
                      file=sys.stderr)

    np.savez_compressed(
        shard_npz,
        eid=np.asarray(eid_col, np.int32),
        label=np.asarray(label_col, np.int8),
        p_yes=np.asarray(p_yes_col, np.float64),
        yes_lp=np.asarray(yes_lp_col, np.float64),
        no_lp=np.asarray(no_lp_col, np.float64),
        margin=np.asarray(margin_col, np.float64),
        yes_logits_raw=np.asarray(yes_raw_col, np.float32),
        no_logits_raw=np.asarray(no_raw_col, np.float32),
        topk_ids=np.asarray(topk_ids_col, np.int32),
        topk_logits=np.asarray(topk_logits_col, np.float32),
        yes_ids=np.asarray(yes_ids, np.int32),
        no_ids=np.asarray(no_ids, np.int32),
    )
    shard_json.write_text(json.dumps(scores_json))
    print(f"[verbalized] gpu {args.gpu_id}: wrote {len(eid_col)} records -> {shard_npz}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
