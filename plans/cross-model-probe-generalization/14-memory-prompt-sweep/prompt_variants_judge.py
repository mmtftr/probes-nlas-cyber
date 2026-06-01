# [ai-generated]
"""Prompt-sensitivity sweep: score verbalized P(yes) under SEVERAL prompt framings.

The belief audit (exp-05 / compare_belief_audit.py) found the model's VERBALIZED
memory-safety judgment is at/below chance (example-AUC 0.39-0.55) under a GENERIC
prompt ("Does the code above contain a security vulnerability? yes/no"). The
lead's hypothesis: a MEMORY-SPECIFIC prompt may massively raise the memory
example-AUC — i.e. the model CAN judge memory-safety when explicitly asked, and
the generic prompt just elicits its injection-biased prior.

This script tests that. It REUSES verbalized_judge.py's verified first-assistant-
token read EXACTLY — model load (incl. the Qwen3.6 VLM text-decoder fallback),
render_chat (enable_thinking=False + Qwen3 abort-guard), p_yes_from_logits,
resolve_yes_no_ids, build_content's CODE-BEFORE-QUESTION neutral framing, and the
chat-template path. The ONLY thing that changes across variants is the QUESTION
string appended after the fenced code block. Everything else is byte-for-byte the
same machinery, so each variant remains a VALID first-assistant-token P(yes) read
directly comparable to the belief audit's V0 reference.

For EACH variant, P(yes) is scored over the SVEN rows, sharded over the node's
GPUs (idx % n_gpus == gpu_id, mirroring verbalized_judge), writing per-variant
per-example scores to:
    $WORK/runs/promptsweep_<slug>/variant_<id>.gpu<g>.json   (eid, p_yes, label)
Skip-if-exists per (variant, shard) — resumable. The aggregator
(analyze_prompt_sweep.py) merges shards per variant and computes memory +
injection example-AUC.

MANDATORY DEBUG PRINT. For the FIRST forward of EACH variant we emit the same
preflight verbalized_judge does (rendered tail + top-5 first-token argmax + the
hard <think> abort assertion), so a human can verify the read stays valid under
the new question wording. Done per variant because the question text differs.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _load_verbalized_judge():
    """[ai-generated] Load the sibling 05-.../verbalized_judge.py by file path so
    we REUSE its EXACT machinery (model load, render_chat, p_yes_from_logits,
    resolve_yes_no_ids, build_content, PREAMBLE). Loading by path keeps it robust
    to cwd and avoids copy-paste drift: if verbalized_judge's read changes, this
    sweep inherits the change. The module imports torch at call time only, so the
    import itself is light."""
    p = (REPO / "plans" / "cross-model-probe-generalization"
         / "05-probe-vs-verbalized" / "verbalized_judge.py")
    if not p.exists():
        raise SystemExit(f"[promptsweep] cannot find verbalized_judge.py at {p}")
    spec = importlib.util.spec_from_file_location("verbalized_judge", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verbalized_judge"] = mod  # so its `from __future__` etc. resolve
    spec.loader.exec_module(mod)
    return mod


# [ai-generated] TODO(adhoc-decision): exact wording of every memory-specific
# variant below is the lead's to set. The CONSTRAINTS are load-bearing and must
# NOT change: code BEFORE the question (set by verbalized_judge.build_content's
# PREAMBLE + fenced block), neutral framing, and a one-word yes/no demand
# IDENTICAL in form to V0 so the first-assistant-token read stays valid and the
# single-word instruction does not itself shift the prior. V0 is copied VERBATIM
# from verbalized_judge.QUESTION (asserted at startup) — it is the reference, the
# == current belief-audit prompt. If the lead prefers different memory wording,
# change it HERE; this list is the single source of truth and each variant's
# question text is logged into every metrics record for provenance.
PROMPT_VARIANTS = [
    {
        "id": "V0_generic",
        "question": (
            "Does the code above contain a security vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V1_memory",
        "question": (
            "Does the code above contain a memory-safety vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V2_memory_detailed",
        "question": (
            "Does the code above contain a memory-safety bug such as a "
            "use-after-free, NULL-pointer dereference, out-of-bounds read/write, "
            "or buffer overflow? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V3_memory_cwe",
        "question": (
            "Does the code above contain CWE-416 (use-after-free), CWE-476 "
            "(NULL dereference), CWE-125 (out-of-bounds read), or CWE-787 "
            "(out-of-bounds write)? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    # [ai-generated] exp-15 ENSEMBLE-COMPARISON extension (2026-06-01).
    # The symmetric probe-vs-verbalized matrix (15-ensemble-comparison) needs a
    # VERBALIZED analogue for every probe member. A specialized PROBE's analogue is
    # a specialized PROMPT. These add:
    #   * V_injection      — the injection-family analogue of V1_memory (the
    #                        'injection' member's verbalized prompt).
    #   * V_cwe<NNN> (x9)  — one per-INDIVIDUAL-CWE prompt, the verbalized analogue
    #                        of each per-CWE probe (the ind-ensemble members).
    # Wording is HELD IN FORM identical to V0/V1 (code-before-question, neutral
    # preamble, one-word yes/no demand) so every read stays a valid first-assistant-
    # token P(yes) directly comparable to the rest of the sweep. ONLY the named
    # vulnerability changes. The per-(variant,shard) skip-if-exists means a re-run
    # computes ONLY these new variants (and fills any missing GPU shards, e.g.
    # Qwen3.6's gpu0). TODO(adhoc-decision): exact phrasing of V_injection and the
    # 9 per-CWE descriptions is the lead's to confirm; this list is the single
    # source of truth and each question text is logged per-variant for provenance.
    {
        "id": "V_injection",
        "question": (
            "Does the code above contain an injection vulnerability such as SQL "
            "injection, OS command injection, path traversal, or cross-site "
            "scripting? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    # --- per-INDIVIDUAL-CWE prompts (verbalized analogue of the per-CWE probes) ---
    # memory family
    {
        "id": "V_cwe416",
        "question": (
            "Does the code above contain a use-after-free vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V_cwe476",
        "question": (
            "Does the code above contain a NULL-pointer dereference "
            "vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V_cwe125",
        "question": (
            "Does the code above contain an out-of-bounds read vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V_cwe787",
        "question": (
            "Does the code above contain an out-of-bounds write vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    # injection family
    {
        "id": "V_cwe089",
        "question": (
            "Does the code above contain a SQL injection vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V_cwe078",
        "question": (
            "Does the code above contain an OS command injection vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V_cwe022",
        "question": (
            "Does the code above contain a path traversal vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V_cwe079",
        "question": (
            "Does the code above contain a cross-site scripting (XSS) "
            "vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
    {
        "id": "V_cwe190",
        "question": (
            "Does the code above contain an integer overflow vulnerability? "
            'Respond with ONLY one word — "yes" or "no" — and nothing else.'
        ),
    },
]


def build_content_for_question(vj, code: str, question: str) -> str:
    """[ai-generated] Identical to verbalized_judge.build_content but with the
    QUESTION swapped for this variant. We keep the PREAMBLE and the fenced
    code-before-question layout EXACTLY as the reference so only the question
    text differs across variants.

        vj.PREAMBLE + "\\n\\n```\\n" + code + "\\n```\\n\\n" + question
    """
    return f"{vj.PREAMBLE}\n\n```\n{code}\n```\n\n{question}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True, help="dataset.jsonl")
    ap.add_argument("--out", required=True,
                    help="OUTPUT DIRECTORY; shard files variant_<id>.gpu{id}.json land here")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    args = ap.parse_args()

    import torch

    vj = _load_verbalized_judge()

    # Provenance + guard: V0 MUST be byte-identical to the belief-audit prompt
    # (verbalized_judge.QUESTION) so it is a faithful reference column. Fail loud
    # if they ever drift — a drifted V0 would silently invalidate the comparison.
    v0 = next(v for v in PROMPT_VARIANTS if v["id"] == "V0_generic")
    if v0["question"] != vj.QUESTION:
        raise SystemExit(
            "[promptsweep] ABORT: V0_generic question is NOT byte-identical to "
            "verbalized_judge.QUESTION (the belief-audit reference prompt). They "
            "must match for V0 to be the reference column.\n"
            f"  V0  : {v0['question']!r}\n"
            f"  ref : {vj.QUESTION!r}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which variants still need this shard BEFORE loading the model;
    # if every variant's shard already exists, skip the (expensive) model load.
    pending = []
    for v in PROMPT_VARIANTS:
        shard = out_dir / f"variant_{v['id']}.gpu{args.gpu_id}.json"
        if shard.exists():
            print(f"[promptsweep] gpu {args.gpu_id}: variant {v['id']} shard "
                  f"exists, skip ({shard})", file=sys.stderr)
        else:
            pending.append((v, shard))
    if not pending:
        print(f"[promptsweep] gpu {args.gpu_id}: all {len(PROMPT_VARIANTS)} "
              "variant shards present, nothing to do", file=sys.stderr)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Model + tokenizer load: EXACT reuse of verbalized_judge's load path
    # (src.data.extract_token_activations _load_model/_load_tokenizer via vj's
    # imports). Same CausalLM -> ImageTextToText fallback for the Qwen3.6 VLM
    # text-decoder.
    tokenizer = vj._load_tokenizer(args.model)
    model = vj._load_model(args.model, dtype)
    model.to(device).eval()
    print(f"[promptsweep] gpu {args.gpu_id}: device={device} dtype={dtype} "
          f"model={args.model}", file=sys.stderr)

    yes_ids, no_ids = vj.resolve_yes_no_ids(tokenizer)
    print(f"[promptsweep] yes_ids={yes_ids} -> "
          f"{[tokenizer.decode([i]) for i in yes_ids]!r}", file=sys.stderr)
    print(f"[promptsweep] no_ids ={no_ids} -> "
          f"{[tokenizer.decode([i]) for i in no_ids]!r}", file=sys.stderr)
    if not yes_ids or not no_ids:
        raise SystemExit("[promptsweep] could not resolve yes/no token ids")

    rows = [json.loads(l) for l in Path(args.pairs).open() if l.strip()]
    mine = [i for i in range(len(rows)) if i % args.n_gpus == args.gpu_id]
    print(f"[promptsweep] gpu {args.gpu_id}/{args.n_gpus}: {len(mine)}/{len(rows)} "
          f"rows x {len(pending)} pending variant(s)", file=sys.stderr)

    # Pre-tokenize/truncate each row's CODE ONCE (variant-independent — only the
    # question changes), then reuse across variants. The code is truncated to
    # max_length exactly as verbalized_judge does (question + scaffolding short).
    code_by_eid = {}
    for eid in mine:
        code = rows[eid]["code"]
        code_ids = tokenizer.encode(code, add_special_tokens=False,
                                    truncation=True, max_length=args.max_length)
        code_by_eid[eid] = tokenizer.decode(code_ids)

    with torch.inference_mode():
        for v, shard in pending:
            vid, question = v["id"], v["question"]
            print(f"[promptsweep] gpu {args.gpu_id}: === variant {vid} ===",
                  file=sys.stderr)
            results = []
            printed_template = False  # debug print ONCE PER VARIANT
            for eid in mine:
                code_trunc = code_by_eid[eid]
                content = build_content_for_question(vj, code_trunc, question)
                # render_chat applies the Qwen3 thinking-mode guard
                # (enable_thinking=False, TypeError fallback) — IDENTICAL to the
                # reference; the question text does not touch this path.
                enc, used_thinking_kwarg = vj.render_chat(
                    tokenizer, content, device, args.model)
                input_ids = enc["input_ids"]

                out = model(**enc, use_cache=False)
                logits_last = out.logits[0, -1, :]

                # [ai-generated] MANDATORY DEBUG PRINT (once per variant per
                # worker). Mirrors verbalized_judge EXACTLY so a human preflight
                # verifies, FOR THIS VARIANT'S WORDING: (a) the rendered tail is
                # the assistant turn-start with NO <think> token; (b) yes/no
                # dominate the first-token argmax. Printed after the first forward
                # so the top-5 decode is real model output.
                if not printed_template:
                    ids = input_ids[0].tolist()
                    rendered = tokenizer.decode(ids)
                    tail = tokenizer.decode(ids[-40:])
                    top5 = torch.topk(logits_last.float(), k=5)
                    top5_tok = [(int(i), tokenizer.decode([int(i)]))
                                for i in top5.indices.tolist()]
                    if not used_thinking_kwarg:
                        print("[promptsweep] WARNING: tokenizer rejected "
                              "enable_thinking=False (TypeError); fell back to "
                              "plain apply_chat_template. If this is a Qwen3 "
                              "model, VERIFY the rendered tail below has NO "
                              "<think> token.", file=sys.stderr)
                    print(f"[promptsweep] variant={vid} enc keys={list(enc.keys())} "
                          f"input_ids.shape={tuple(input_ids.shape)} "
                          f"enable_thinking_kwarg_used={used_thinking_kwarg}",
                          file=sys.stderr)
                    print(f"[promptsweep] variant={vid} QUESTION={question!r}",
                          file=sys.stderr)
                    print("[promptsweep] ===== full rendered chat template "
                          "(decoded) =====", file=sys.stderr)
                    print(rendered, file=sys.stderr)
                    print("[promptsweep] ===== rendered TAIL (last ~40 tokens) — "
                          "must be the assistant turn-start, NO <think> =====",
                          file=sys.stderr)
                    print(repr(tail), file=sys.stderr)
                    print("[promptsweep] ===== top-5 first-token argmax (decoded) "
                          "— yes/no should dominate =====", file=sys.stderr)
                    for tid, dec in top5_tok:
                        print(f"[promptsweep]   id={tid} -> {dec!r}", file=sys.stderr)
                    print("[promptsweep] ===== end debug =====", file=sys.stderr)

                    # [ai-generated] HARD THINKING-MODE ASSERTION (once per variant
                    # per worker). Same guard as verbalized_judge: even if
                    # enable_thinking=False was accepted, a buggy template could
                    # still open a <think> block. Refuse to proceed if the top-1
                    # first-token argmax begins a thinking block — this runs
                    # autonomously, so a silently-still-thinking template must
                    # crash here rather than emit invalid yes/no scores.
                    top1_dec = top5_tok[0][1].strip()
                    if top1_dec == "<think>" or "think>" in top1_dec:
                        raise RuntimeError(
                            f"[promptsweep] ABORT: model {args.model!r} variant "
                            f"{vid} top-1 first-token argmax is {top1_dec!r}, which "
                            "begins a <think> reasoning block — the yes/no read is "
                            "invalid. The chat template is still in thinking mode "
                            "despite the guard; refusing to produce scores."
                        )
                    printed_template = True

                p_yes = vj.p_yes_from_logits(logits_last, yes_ids, no_ids)
                results.append({"eid": int(eid), "p_yes": float(p_yes),
                                "label": vj._row_label(rows[eid])})

                if len(results) % 50 == 0:
                    print(f"[promptsweep] gpu {args.gpu_id} {vid}: "
                          f"{len(results)}/{len(mine)}", file=sys.stderr)

            shard.write_text(json.dumps(results))
            print(f"[promptsweep] gpu {args.gpu_id}: variant {vid} wrote "
                  f"{len(results)} scores -> {shard}", file=sys.stderr)


if __name__ == "__main__":
    main()
