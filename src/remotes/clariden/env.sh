#!/bin/bash
# [ai-generated]
# Container-side environment for Clariden probe-sweep jobs. Sourced INSIDE the
# `--environment=alps3` srun step (NGC PyTorch image: has torch+CUDA, lacks HF).
# Idempotent: installs missing Python deps into $PYTHON_DEPS_DIR once, guarded
# by a flock so the 4 per-GPU workers in a node-job don't race.
set -euo pipefail

WORK=${WORK:-$HOME/scratch/probes}
export REPO=${REPO:-$WORK/repo}
export DATA=${DATA:-$WORK/data}
export RUNS=${RUNS:-$WORK/runs}
export PYTHON_DEPS_DIR=${PYTHON_DEPS_DIR:-$WORK/.python_deps}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$WORK/.uv_cache}
export HF_HOME=${HF_HOME:-$WORK/hf_cache}
export HF_HUB_ENABLE_HF_TRANSFER=1
export TRITON_CACHE_DIR=/tmp/probes-${SLURM_JOB_ID:-0}/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/probes-${SLURM_JOB_ID:-0}/inductor
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"

# HF token (gated models). User drops it here as a one-line file.
if [ -f "$WORK/secrets/hf_token" ]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "$WORK/secrets/hf_token")"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Repo on PYTHONPATH so `import src...` works; deps dir first.
export PYTHONPATH="$PYTHON_DEPS_DIR:$REPO:${PYTHONPATH:-}"

PYBIN="$(command -v python || command -v python3)"
DEPS="transformers>=4.50 accelerate safetensors sentencepiece hf_transfer scikit-learn"
# torch, numpy already in the NGC image — installed --no-deps to avoid clobbering.
if [ ! -f "$PYTHON_DEPS_DIR/.deps_ok" ]; then
    mkdir -p "$PYTHON_DEPS_DIR"
    flock "$PYTHON_DEPS_DIR/.lock" bash -c '
        set -e
        if [ ! -f "'"$PYTHON_DEPS_DIR"'/.deps_ok" ]; then
            echo "[env] installing deps into '"$PYTHON_DEPS_DIR"'" >&2
            uv pip install --target "'"$PYTHON_DEPS_DIR"'" --python "'"$PYBIN"'" --no-deps '"$DEPS"'
            touch "'"$PYTHON_DEPS_DIR"'/.deps_ok"
        fi'
fi
