#!/usr/bin/env bash
# [ai-generated]
# Parallelized cold-start setup for vLLM hidden-state extraction on Colab G4.
# Overlaps: (a) HF model download, (b) vllm+cu130 torch install, (c) optional
# torch.compile cache pull. Prints per-phase wall-clock so we can see what the
# critical path actually is.
#
# Usage: bash cold_start.sh <MODEL> [CACHE_HF_DATASET]
set -uo pipefail

MODEL="${1:?need model id}"
CACHE_REF="${2:-}"          # optional HF dataset repo holding a prebuilt vllm compile cache
LOG=/content/coldlog
mkdir -p "$LOG"
: "${HF_TOKEN:?HF_TOKEN must be exported}"

now() { date +%s.%N; }
elapsed() { awk "BEGIN{printf \"%.1f\", $2-$1}"; }

T0=$(now)

# ---- bootstrap: uv + hf_transfer + huggingface_hub (small, fast) ----
pip install -q uv hf_transfer "huggingface_hub>=0.34" >"$LOG/bootstrap.log" 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$HOME/.local/bin:$PATH"
T_BOOT=$(now)
echo "[phase] bootstrap            $(elapsed $T0 $T_BOOT)s"

# ---- parallel phase ----
# (a) model download into the default HF cache so LLM(model=...) finds it
( S=$(now)
  hf download "$MODEL" --token "$HF_TOKEN" >"$LOG/download.log" 2>&1
  echo "$(elapsed $S $(now))" >"$LOG/download.time"
) &
PID_DL=$!

# (b) vllm + cu130 torch install
( S=$(now)
  uv pip install --system --torch-backend=cu130 "vllm==0.22.0" hf_transfer >"$LOG/install.log" 2>&1 \
  && uv pip install --system --torch-backend=cu130 --force-reinstall "torch==2.11.0" >>"$LOG/install.log" 2>&1 \
  && uv pip uninstall --system torchvision torchaudio >>"$LOG/install.log" 2>&1
  echo "$(elapsed $S $(now))" >"$LOG/install.time"
) &
PID_INSTALL=$!

# (c) optional compile-cache pull (runs concurrently; cheap if present)
if [ -n "$CACHE_REF" ]; then
  ( S=$(now)
    hf download "$CACHE_REF" --repo-type dataset --token "$HF_TOKEN" \
       --local-dir /root/.cache/vllm >"$LOG/cache.log" 2>&1
    echo "$(elapsed $S $(now))" >"$LOG/cache.time"
  ) &
  PID_CACHE=$!
fi

wait $PID_DL;      DL_OK=$?
wait $PID_INSTALL; IN_OK=$?
[ -n "$CACHE_REF" ] && { wait $PID_CACHE; }

T_END=$(now)
echo "[phase] download   (parallel) $(cat $LOG/download.time 2>/dev/null)s  exit=$DL_OK"
echo "[phase] install    (parallel) $(cat $LOG/install.time  2>/dev/null)s  exit=$IN_OK"
[ -n "$CACHE_REF" ] && echo "[phase] cachepull  (parallel) $(cat $LOG/cache.time 2>/dev/null)s"
echo "[phase] SETUP TOTAL (wall)     $(elapsed $T0 $T_END)s"
[ $DL_OK -ne 0 ] && { echo "DOWNLOAD FAILED"; tail -20 "$LOG/download.log"; exit 1; }
[ $IN_OK -ne 0 ] && { echo "INSTALL FAILED";  tail -30 "$LOG/install.log";  exit 1; }
echo "SETUP_OK"
