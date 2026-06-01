# [ai-generated]
"""Verbalized vulnerability judgment: ask the model itself, read P(yes).

For each SVEN dataset row, build a NEUTRAL input-stream prompt with the code
BEFORE the eliciting question (so the code is encoded the same way the probe's
cached activations were — on raw code, no question conditioning the encoding),
forward once, and read the model's own yes/no answer probability at the first
assistant-token position.

    PREAMBLE  = "Here is a code snippet:"
    QUESTION  = "Does the code above contain a security vulnerability? Respond
                with ONLY one word — \"yes\" or \"no\" — and nothing else."
    content   = f"{PREAMBLE}\n\n```\n{code}\n```\n\n{QUESTION}"

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, return_tensors="pt")

P(yes) = sigmoid( logsumexp(logprob[yes_ids]) - logsumexp(logprob[no_ids]) )
where logprob = log_softmax(logits[-1]) (the next-token distribution).

This step LOADS THE MODEL (unlike the cached-acts experiments 03/04) — it is the
forward-pass half of the comparison. The comparison itself (probe vs. these
scores) is done WITHOUT the model in compare_belief_audit.py.

GPU-sharded by dataset row index (idx % n_gpus == gpu_id). Each worker writes
its own shard `verbalized_scores.gpu{gpu_id}.json` into the --out directory and
skips it if it already exists (resumable). The aggregator merges all shards.

Model loading mirrors src/data/extract_token_activations.py
(`_load_model` / `_load_tokenizer`, dtype = bfloat16 on cuda). `_load_model`'s
CausalLM -> ImageTextToText fallback chain transparently loads the TEXT DECODER
of a VLM (Qwen3.6-27B), so a text-only forward (input_ids, no pixel_values)
returns the decoder's next-token logits — the same load path extraction uses.

[ai-generated] BELIEF-AUDIT extension (2026-06-01)
---------------------------------------------------
Two changes for the 4-model belief audit (see BELIEF-AUDIT.md):

  * THINKING MODE GUARD. Qwen3 models (Qwen3-32B, Qwen3.6-27B) default to a
    <think> reasoning block: with thinking on, the FIRST assistant token is the
    start of <think>, NOT yes/no — which would INVALIDATE the P(yes) read. We
    therefore call apply_chat_template(..., enable_thinking=False); if the
    tokenizer rejects that kwarg we fall back (no kwarg) and WARN loudly. See
    `render_chat`.
  * MANDATORY DEBUG PRINT (once per worker): the decoded rendered-prompt TAIL
    (last ~40 tokens) and the top-5 first-token argmax (decoded). A human
    preflight uses this to VERIFY (a) the tail is the assistant turn-start with
    NO <think> token, and (b) yes/no dominate the first-token distribution.
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
# [ai-generated] TODO(adhoc-decision): exact wording of the single-word demand.
# The lead asked for a clearer yes/no-only instruction than the old
# "Answer with a single word: yes or no." This phrasing demands ONE word
# unambiguously ("Respond with ONLY one word ... and nothing else"). If the lead
# prefers different wording, change it HERE — it is the single source of truth and
# is logged into every metrics record's `question` field for provenance.
QUESTION = (
    "Does the code above contain a security vulnerability? "
    'Respond with ONLY one word — "yes" or "no" — and nothing else.'
)


def build_content(code: str) -> str:
    """Code BEFORE the question, minimal neutral preamble, fenced code block."""
    return f"{PREAMBLE}\n\n```\n{code}\n```\n\n{QUESTION}"


def is_qwen3_model(model_name: str) -> bool:
    """[ai-generated] True for Qwen3 family (Qwen3-32B, Qwen3.6-27B), which
    default to a <think> reasoning block and REQUIRE enable_thinking=False."""
    return "Qwen3" in str(model_name)


def render_chat(tokenizer, content: str, device, model_name: str):
    """[ai-generated] Render the user turn to model-ready input tensors.

    THINKING-MODE GUARD (Qwen3): try add_generation_prompt=True with
    enable_thinking=False so the assistant turn opens directly (no <think>
    block). transformers' apply_chat_template forwards unknown kwargs into the
    Jinja template; a tokenizer whose template does NOT accept enable_thinking
    raises TypeError (and some non-Qwen templates ignore it silently).

    On TypeError: for a NON-Qwen3 model the kwarg is irrelevant, so we fall back
    to the plain call (caller WARNs). For a Qwen3 model we ABORT — without
    enable_thinking=False the first assistant token is <think>, which silently
    invalidates the yes/no read; this runs autonomously with no human to catch a
    warning, so we must fail loudly rather than emit wrong scores.

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
        # Non-Qwen3: kwarg is irrelevant — fall back, WARN once via caller.
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
    ap.add_argument("--max-length", type=int, default=2048)
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
            # transformers 5.9.0's apply_chat_template returns a BatchEncoding
            # (dict-like: input_ids + attention_mask), NOT a bare tensor; render_chat
            # forces the dict form and applies the Qwen3 thinking-mode guard
            # (enable_thinking=False, with TypeError fallback). Stay robust across
            # versions that return a bare tensor.
            enc, used_thinking_kwarg = render_chat(tokenizer, content, device, args.model)
            input_ids = enc["input_ids"]

            out = model(**enc, use_cache=False)
            logits_last = out.logits[0, -1, :]

            # [ai-generated] MANDATORY DEBUG PRINT (once per worker). A human
            # preflight verifies from this: (a) the rendered TAIL is the assistant
            # turn-start with NO <think> token; (b) yes/no dominate the first-token
            # argmax distribution. Printed AFTER the first forward so the top-5
            # first-token decode is real model output.
            if not printed_template:
                ids = input_ids[0].tolist()
                rendered = tokenizer.decode(ids)
                tail_ids = ids[-40:]
                tail = tokenizer.decode(tail_ids)
                top5 = torch.topk(logits_last.float(), k=5)
                top5_tok = [(int(i), tokenizer.decode([int(i)]))
                            for i in top5.indices.tolist()]
                if not used_thinking_kwarg:
                    print("[verbalized] WARNING: tokenizer rejected "
                          "enable_thinking=False (TypeError); fell back to plain "
                          "apply_chat_template. If this is a Qwen3 model, VERIFY "
                          "the rendered tail below has NO <think> token.",
                          file=sys.stderr)
                print(f"[verbalized] enc keys={list(enc.keys())} "
                      f"input_ids.shape={tuple(input_ids.shape)} "
                      f"enable_thinking_kwarg_used={used_thinking_kwarg}",
                      file=sys.stderr)
                print("[verbalized] ===== full rendered chat template (decoded) =====",
                      file=sys.stderr)
                print(rendered, file=sys.stderr)
                print("[verbalized] ===== rendered TAIL (last ~40 tokens) — must be "
                      "the assistant turn-start, NO <think> =====", file=sys.stderr)
                print(repr(tail), file=sys.stderr)
                print("[verbalized] ===== top-5 first-token argmax (decoded) — yes/no "
                      "should dominate =====", file=sys.stderr)
                for tid, dec in top5_tok:
                    print(f"[verbalized]   id={tid} -> {dec!r}", file=sys.stderr)
                print("[verbalized] ===== end debug =====", file=sys.stderr)

                # [ai-generated] HARD THINKING-MODE ASSERTION (once per worker).
                # Even if enable_thinking=False was ACCEPTED, a buggy template
                # could still open a <think> block. Decode the top-1 first-token
                # argmax and refuse to proceed if it begins a thinking block —
                # this runs autonomously, so a silently-still-thinking template
                # must crash here rather than emit invalid yes/no scores. Printed
                # debug above gives the human the rendered tail + top-5 context.
                top1_dec = top5_tok[0][1].strip()
                if top1_dec == "<think>" or "think>" in top1_dec:
                    raise RuntimeError(
                        f"[verbalized] ABORT: model {args.model!r} top-1 "
                        f"first-token argmax is {top1_dec!r}, which begins a "
                        "<think> reasoning block — the yes/no read is invalid. "
                        "The chat template is still in thinking mode despite the "
                        "guard; refusing to produce scores."
                    )
                printed_template = True

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
