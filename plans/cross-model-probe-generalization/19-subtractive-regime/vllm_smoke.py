# [ai-generated]
"""Numerical smoke: vLLM extract_hidden_states vs HF output_hidden_states for the
SAME tokens, same model, same layer. Pass = mean per-token cos >= 0.999 (bf16 +
FlashAttn vs eager differ slightly; the guide reports ~0.9999 at the matched
index). Validates the L+1 aux-id convention end-to-end.

All work is under `if __name__ == "__main__"` — vLLM v1 spawns its engine-core
process which re-imports this module, so top-level must stay import-only.
  python vllm_smoke.py <MODEL> <REPO_LAYER> [N_ROWS]
"""
import json, os, sys
import numpy as np
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")


def main():
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-1b-it"
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    DATA = os.environ.get("DATASET", "./data/dataset.jsonl")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    rows = [json.loads(l) for l in open(DATA)][:N]
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    ids = [tok(r["code"], truncation=True, max_length=2048)["input_ids"] for r in rows]
    print(f"[smoke] {MODEL} layer {L}  rows={N}  tok_lens={[len(i) for i in ids]}", flush=True)

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="eager", trust_remote_code=True).cuda().eval()
    hf = []
    with torch.inference_mode():
        for i in ids:
            o = m(torch.tensor([i]).cuda(), output_hidden_states=True, use_cache=False)
            hf.append(o.hidden_states[L + 1][0].float().cpu().numpy())
    del m; torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector
    HS = "/tmp/hs_smoke"; os.system(f"rm -rf {HS}; mkdir -p {HS}")
    llm = LLM(model=MODEL, enable_chunked_prefill=False, enable_prefix_caching=False,
              attention_backend="FLASH_ATTN", max_model_len=2064, trust_remote_code=True,
              speculative_config={"method": "extract_hidden_states", "num_speculative_tokens": 1,
                                  "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": [L + 1]}}},
              kv_transfer_config=KVTransferConfig(kv_connector="ExampleHiddenStatesConnector", kv_role="kv_producer",
                                                  kv_connector_extra_config={"shared_storage_path": HS}))
    outs = llm.generate([{"prompt_token_ids": i} for i in ids], SamplingParams(max_tokens=1, temperature=0.0))

    cos_all = []
    for r, o in enumerate(outs):
        obj = example_hidden_states_connector.load_hidden_states(o.kv_transfer_params["hidden_states_path"])
        hs = obj["hidden_states"]
        hs = hs.float().cpu().numpy() if hasattr(hs, "float") else np.asarray(hs, dtype=np.float32)
        v = hs[:, 0, :]
        h = hf[r]
        if v.shape != h.shape:
            print(f"[smoke] row {r} SHAPE MISMATCH vllm={v.shape} hf={h.shape}", flush=True); continue
        cos = (v * h).sum(-1) / (np.linalg.norm(v, axis=-1) * np.linalg.norm(h, axis=-1) + 1e-8)
        cos_all.append(cos)
        print(f"[smoke] row {r}: n_tok={len(cos)} mean_cos={cos.mean():.5f} min_cos={cos.min():.5f}", flush=True)

    allc = np.concatenate(cos_all)
    print(f"[smoke] OVERALL mean_cos={allc.mean():.5f} min_cos={allc.min():.5f} "
          f"-> {'PASS' if allc.mean() >= 0.999 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
