# [ai-generated]
"""Extract the LAST-PROMPT-TOKEN hidden state in the verbalized QA format.

The genuine NXT2b position: the `Assistant:` turn-boundary token (index -1 of the
exp-17 verbalized input) — the exact position whose next-token logits give
verbalized P(yes). exp-17 saved only those vocab logits; this saves the hidden
state there, for ALL layers, so a linear probe can be trained on it.

The forward + chat rendering + Qwen3 thinking-guard + yes/no id resolution are
imported VERBATIM from exp-17 verbalized_logit_dump.py (single source of truth),
so the persisted p_yes is bit-identical to exp-17 and gates against it. We only
add output_hidden_states=True and capture hidden_states[i][0, -1, :] for every i.

Layer convention: hidden_states[L+1] == output of repo-block L (hidden_states[0]
= embeddings). We persist the full tuple (n_layers+1 vectors); the trainer indexes
repo-layer L = column L+1.

GPU-sharded by row index (idx % n_gpus == gpu_id); resumable per shard. Output:
  lasttoken_hidden.gpu{id}.npz — H[n, n_layers+1, hidden] FLOAT32 (fp16 overflows
      on late-layer massive activations), eid int32, label int8,
      p_yes/yes_lp/no_lp/margin float64 (exp-17 gate columns), + meta_* provenance.
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
# reuse exp-17's exact template/guard/id logic — do NOT re-implement
E17 = REPO / "plans/cross-model-probe-generalization/17-verbalized-logit-dump"
sys.path.insert(0, str(E17))
from verbalized_logit_dump import (  # noqa: E402
    render_chat, resolve_yes_no_ids, verbalized_record, _row_label)

# default question = exp-17's vulnerability prompt (primed). Override with
# --question for a NEUTRAL prompt ("What do you think about this code?", exp-31).
DEFAULT_QUESTION = ('Does the code above contain a security vulnerability? '
                    'Respond with ONLY one word — "yes" or "no" — and nothing else.')


def build_content(code: str, preamble: str, question: str) -> str:
    return f"{preamble}\n\n```\n{code}\n```\n\n{question}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True, help="dataset.jsonl")
    ap.add_argument("--out", required=True, help="output dir for shard npz")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--preamble", default="Here is a code snippet:")
    ap.add_argument("--question", default=DEFAULT_QUESTION,
                    help='neutral override: "What do you think about this code?"')
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    shard = out_dir / f"lasttoken_hidden.gpu{args.gpu_id}.npz"
    if shard.exists():
        print(f"[lasttok] gpu {args.gpu_id}: shard exists, skip ({shard})", file=sys.stderr)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = _load_tokenizer(args.model)
    model = _load_model(args.model, dtype); model.to(device).eval()
    yes_ids, no_ids = resolve_yes_no_ids(tokenizer)
    if not yes_ids or not no_ids:
        raise SystemExit("[lasttok] could not resolve yes/no token ids")

    rows = [json.loads(l) for l in Path(args.pairs).open() if l.strip()]
    mine = [i for i in range(len(rows)) if i % args.n_gpus == args.gpu_id]
    print(f"[lasttok] gpu {args.gpu_id}/{args.n_gpus}: {len(mine)}/{len(rows)} rows "
          f"model={args.model}", file=sys.stderr)

    H_col, eid_col, lab_col = [], [], []
    pyes_col, yeslp_col, nolp_col, margin_col = [], [], [], []
    printed = False
    with torch.inference_mode():
        for eid in mine:
            row = rows[eid]
            code_ids = tokenizer.encode(row["code"], add_special_tokens=False,
                                        truncation=True, max_length=args.max_length)
            content = build_content(tokenizer.decode(code_ids), args.preamble, args.question)
            enc, used_kwarg = render_chat(tokenizer, content, device, args.model)
            out = model(**enc, use_cache=False, output_hidden_states=True)
            logits_last = out.logits[0, -1, :]

            if not printed:  # exp-17 preflight: tail must be assistant turn-start, no <think>
                ids = enc["input_ids"][0].tolist()
                top5 = torch.topk(logits_last.float(), k=5)
                print(f"[lasttok] enable_thinking_used={used_kwarg} "
                      f"n_hidden_layers={len(out.hidden_states)} hidden={out.hidden_states[0].shape[-1]}",
                      file=sys.stderr)
                print("[lasttok] TAIL:", repr(tokenizer.decode(ids[-40:])), file=sys.stderr)
                for i in top5.indices.tolist():
                    print(f"[lasttok]   argmax {i!r} -> {tokenizer.decode([i])!r}", file=sys.stderr)
                top1 = tokenizer.decode([int(top5.indices[0])]).strip()
                if top1 == "<think>" or "think>" in top1:
                    raise RuntimeError(f"[lasttok] ABORT: top-1 first token {top1!r} is a <think> "
                                       "block — yes/no read invalid.")
                printed = True

            # last-prompt-token hidden state for every layer -> [n_layers+1, hidden].
            # float32 (NOT float16): late-layer "massive activations" exceed fp16's
            # 65504 max -> inf -> StandardScaler nan -> silent layer drop (review C2).
            hs = torch.stack([h[0, -1, :] for h in out.hidden_states], dim=0)
            H_col.append(hs.to(torch.float32).cpu().numpy())
            rec = verbalized_record(logits_last, yes_ids, no_ids)
            eid_col.append(int(eid)); lab_col.append(_row_label(row))
            pyes_col.append(rec["p_yes"]); yeslp_col.append(rec["yes_lp"])
            nolp_col.append(rec["no_lp"]); margin_col.append(rec["margin"])
            if len(eid_col) % 100 == 0:
                print(f"[lasttok] gpu {args.gpu_id}: {len(eid_col)}/{len(mine)}", file=sys.stderr)

    H = np.asarray(H_col, np.float32)                  # [n, n_layers+1, hidden]
    if not np.isfinite(H).all():
        raise SystemExit(f"[lasttok] gpu {args.gpu_id}: non-finite hidden states "
                         f"({(~np.isfinite(H)).sum()} entries) — refusing to save")
    np.savez_compressed(
        shard,
        H=H,
        eid=np.asarray(eid_col, np.int32),
        label=np.asarray(lab_col, np.int8),
        p_yes=np.asarray(pyes_col, np.float64),
        yes_lp=np.asarray(yeslp_col, np.float64),
        no_lp=np.asarray(nolp_col, np.float64),
        margin=np.asarray(margin_col, np.float64),
        # provenance (review M-meta): review the artifact without the logs
        meta_model=np.array(args.model), meta_max_length=np.int32(args.max_length),
        meta_yes_ids=np.asarray(yes_ids, np.int32), meta_no_ids=np.asarray(no_ids, np.int32))
    print(f"[lasttok] gpu {args.gpu_id}: wrote {len(eid_col)} rows -> {shard}", file=sys.stderr)


if __name__ == "__main__":
    main()
