# [ai-generated]
# vLLM hidden-state extraction benchmark. Run as a SUBPROCESS (not in the IPython
# kernel) to avoid vLLM's stdout.fileno() crash.
#
# Usage: python bench_vllm.py <MODEL> [N_PROMPTS]
# Emits JSON on the last stdout line: {"phase_s": {...}, "tok_s": ..., ...}
import json, os, sys, time, shutil, glob

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
HS_DIR = "/dev/shm/hs_extract"
CACHE_DIR = "/root/.cache/vllm"

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# vllm 0.22 bundles flashinfer; its sampler JIT-compiles kernels whose
# check_cuda_arch() wrongly rejects sm120 (Blackwell). Force the native torch
# sampler so extraction never touches flashinfer.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from transformers import AutoConfig, AutoTokenizer

cfg = AutoConfig.from_pretrained(MODEL)
n_layers = cfg.num_hidden_layers
hidden = cfg.hidden_size
# 4 evenly spaced aux layer ids in [1, n_layers] (exact semantics irrelevant for timing)
aux_ids = sorted({max(1, round(n_layers * f)) for f in (0.2, 0.4, 0.6, 0.8)})

tok = AutoTokenizer.from_pretrained(MODEL)
# Build a deterministic token pool from a code-like seed, then slice to target lengths.
seed = ("def f(x):\n    # compute something\n    y = x * 2 + 1\n    return y\n"
        "class A:\n    def __init__(self, n):\n        self.n = n\n") * 400
pool = tok(seed, add_special_tokens=False)["input_ids"]
lengths = [128, 256, 512, 1024, 2048]
prompts = []
for i in range(N):
    L = lengths[i % len(lengths)]
    start = (i * 37) % max(1, len(pool) - L)
    prompts.append(pool[start:start + L])
total_tokens = sum(len(p) for p in prompts)

# fresh hidden-state dir
shutil.rmtree(HS_DIR, ignore_errors=True)
os.makedirs(HS_DIR, exist_ok=True)

cache_present_before = os.path.isdir(CACHE_DIR) and bool(os.listdir(CACHE_DIR)) if os.path.isdir(CACHE_DIR) else False

from vllm import LLM, SamplingParams
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import example_hidden_states_connector

t0 = time.time()
llm = LLM(
    model=MODEL,
    enable_chunked_prefill=False,
    enable_prefix_caching=False,
    attention_backend="FLASH_ATTN",
    max_model_len=4096,
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": aux_ids}},
    },
    kv_transfer_config=KVTransferConfig(
        kv_connector="ExampleHiddenStatesConnector",
        kv_role="kv_producer",
        kv_connector_extra_config={"shared_storage_path": HS_DIR},
    ),
)
t_init = time.time() - t0

sp = SamplingParams(max_tokens=1, temperature=0.0)
t1 = time.time()
outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
t_extract = time.time() - t1

# verify extraction actually happened on a sample
n_files = len(glob.glob(os.path.join(HS_DIR, "**", "*.safetensors"), recursive=True))
sample_shape = None
try:
    path = outputs[0].kv_transfer_params["hidden_states_path"]
    obj = example_hidden_states_connector.load_hidden_states(path)
    sample_shape = list(obj["hidden_states"].shape)
except Exception as e:
    sample_shape = f"verify_failed: {e}"

result = {
    "model": MODEL,
    "n_layers": n_layers,
    "hidden": hidden,
    "aux_ids": aux_ids,
    "n_prompts": N,
    "total_tokens": total_tokens,
    "compile_cache_present_before_init": cache_present_before,
    "phase_s": {"engine_init": round(t_init, 1), "extract": round(t_extract, 1)},
    "ex_s": round(N / t_extract, 1),
    "tok_s": round(total_tokens / t_extract, 0),
    "hs_files_written": n_files,
    "sample_hs_shape": sample_shape,
}
print("BENCH_JSON " + json.dumps(result))
