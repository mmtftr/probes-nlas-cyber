[ai-generated]

# Clariden (CSCS) cluster access

Practical lessons from running the cross-model sweeps on Clariden (Alps).

## Hardware / partition
- Node = 4× GH200 (96 GB each), 288 logical CPUs, ~460 GB usable RAM.
- `debug` partition: **90 min walltime**, and **debug-qos allows only ONE
  submitted job per user** (`QOSMaxSubmitJobPerUserLimit` on a 2nd `sbatch`).
  → run jobs sequentially. For a multi-model sweep, put a small `nohup`
  orchestrator on the login node that waits on `squeue -j <jid>`, submits the
  next when the slot frees (retry loop past the QOS error), and logs `DONE`.
- Account: `lsaie-ss26`. Scratch: `~/scratch/probes` =
  `/iopsstor/scratch/cscs/course_00136` (~535 TB free — float32 acts are fine).

## Containers
- Use `--environment=alps3` (NGC PyTorch image: py3.12, torch 2.10.0a0, CUDA 13.1).
- `srun` flags that work: `srun -lu --mpi=pmi2 --environment=alps3
  --cpus-per-task $SLURM_CPUS_PER_TASK bash -c '...'`.
  **`--mpi=pmi2` is required** — without it pyxis throws a pmix mount error on
  interactive `srun`. Match `--ntasks-per-node=1` between srun and the sbatch.
- Pin memory to the GPU's NUMA node: `CUDA_VISIBLE_DEVICES=$g numactl
  --membind=$g python ...`, one process per GPU, `wait` for the fan-out.

## Provenance / workflow
- `git push` locally **before** `ssh clariden 'cd repo && git pull'` — easy to
  forget; the cluster silently stays on the old commit otherwise.
- The repo lives at `$WORK/repo`; `source $REPO/src/remotes/clariden/env.sh`
  sets the dep stack + HF env inside the container.
- Submit scripts here are idempotent/resumable (DONE markers, per-unit JSON that
  is skipped if present) so a 90-min timeout just means re-submit.

## Dependency management (the bump that bit us)
- The image ships old `transformers`; we need ≥5.9.0 for new model archs. Install
  the new stack with `pip install --no-deps ... --target $WORK/.python_deps5` and
  put it FIRST on `PYTHONPATH` so it shadows the image libs **without touching
  torch/numpy** (those stay from the image). See [[probe-activation-extraction]].
- hub 1.17's Xet downloader vs the image's `hf_xet` mismatch →
  `download_files() got an unexpected keyword argument 'request_headers'`. Fix:
  `export HF_HUB_DISABLE_XET=1`. Isolating the new stack in its own `--target`
  dir meant already-finished runs were never at risk from the bump.
