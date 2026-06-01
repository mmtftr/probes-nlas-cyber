# [ai-generated]
"""Assert a 4-node-then-merged acts/ dir equals a reference 1-node acts/ dir for the
SAME model + SAME rows. No GPU needed — pure array/byte comparison on scratch.

This is the lead's correctness gate: extract gemma-3-1b-it 4-node, merge, then diff
against the existing 1-node acts at
  ~/scratch/probes/runs/layersweep_google_gemma-3-1b-it/acts/

Validation procedure (what this checks, in order):
  1. meta.json — model, n_layers, hidden, n_tokens, n_rows, max_length, pos_tokens
     must match exactly. (n_shards is merge-only provenance, ignored.)
  2. y.npy            — exact array equality (int8).
  3. example_ids.npy  — exact array equality (int32). This is the row-order check:
                        if shards merged out of order, eids would differ here.
  4. offsets.npz      — same key set; each offsets_row array exactly equal.
  5. layer_NN.npy     — float32 activations. The extractor is deterministic
     (inference_mode, no sampling, identical tokenization), so 4-node-then-merge
     should be BIT-IDENTICAL to 1-node modulo nothing — the per-row forward pass is
     the same computation regardless of which rank ran it. We assert exact equality
     by default. `--rtol/--atol` switch to np.allclose if the lead observes
     hardware-nondeterminism (e.g. different GPU SKUs across nodes); document the
     tolerance used in the run log if so.

     Layers checked: by default a sample (`--layer-sample N`, evenly spaced incl.
     first/last). `--all-layers` checks every layer (slower, full guarantee).
     Each checked layer is compared in row-chunks (mmap) to bound RAM.

Exit code 0 = identical (within tolerance); non-zero on first mismatch with a
diagnostic. Designed to be run inside the container OR on any box with numpy + the
scratch FS mounted (no torch / no transformers needed).

Usage:
  python validate_merge.py --merged <merged_acts> --reference <1node_acts> \
      [--layer-sample 8 | --all-layers] [--rtol 0 --atol 0] [--chunk 65536]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

META_FIELDS = ("model", "n_layers", "hidden", "n_tokens", "n_rows",
               "max_length", "pos_tokens")


def fail(msg: str) -> None:
    print(f"[validate] FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def cmp_arrays(name: str, a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> None:
    if a.shape != b.shape:
        fail(f"{name}: shape {a.shape} != reference {b.shape}")
    if a.dtype != b.dtype:
        fail(f"{name}: dtype {a.dtype} != reference {b.dtype}")
    if rtol == 0.0 and atol == 0.0:
        if not np.array_equal(a, b):
            n_diff = int((a != b).sum())
            idx = np.argwhere(a != b)
            fail(f"{name}: {n_diff} elements differ (exact); first at {tuple(idx[0])}: "
                 f"{a[tuple(idx[0])]} vs {b[tuple(idx[0])]}")
    else:
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            fail(f"{name}: not within rtol={rtol} atol={atol}; max|Δ|={d.max():.3e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--layer-sample", type=int, default=8,
                    help="Evenly-spaced layers to check (incl first/last). Ignored if --all-layers.")
    ap.add_argument("--all-layers", action="store_true")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=65536, help="Row-chunk for layer cmp.")
    args = ap.parse_args()

    m = Path(args.merged)
    r = Path(args.reference)

    # 1. meta.json
    mm = json.loads((m / "meta.json").read_text())
    rm = json.loads((r / "meta.json").read_text())
    for k in META_FIELDS:
        if mm.get(k) != rm.get(k):
            fail(f"meta.json[{k!r}] {mm.get(k)} != reference {rm.get(k)}")
    print(f"[validate] meta.json OK  ({ {k: mm[k] for k in META_FIELDS} })", file=sys.stderr)

    # 2. y.npy
    cmp_arrays("y.npy", np.load(m / "y.npy"), np.load(r / "y.npy"), 0.0, 0.0)
    print("[validate] y.npy OK (exact)", file=sys.stderr)

    # 3. example_ids.npy  (row-order check)
    cmp_arrays("example_ids.npy", np.load(m / "example_ids.npy"),
               np.load(r / "example_ids.npy"), 0.0, 0.0)
    print("[validate] example_ids.npy OK (exact — row order matches)", file=sys.stderr)

    # 4. offsets.npz
    mo = np.load(m / "offsets.npz")
    ro = np.load(r / "offsets.npz")
    if set(mo.files) != set(ro.files):
        miss = sorted(set(ro.files) - set(mo.files))[:8]
        extra = sorted(set(mo.files) - set(ro.files))[:8]
        fail(f"offsets.npz keys differ; missing={miss} extra={extra}")
    for k in ro.files:
        cmp_arrays(f"offsets.npz[{k}]", np.asarray(mo[k]), np.asarray(ro[k]), 0.0, 0.0)
    print(f"[validate] offsets.npz OK ({len(ro.files)} rows, exact)", file=sys.stderr)

    # 5. layer_NN.npy
    n_layers = int(mm["n_layers"])
    if args.all_layers:
        layers = list(range(n_layers))
    else:
        k = max(1, min(args.layer_sample, n_layers))
        layers = sorted(set(np.linspace(0, n_layers - 1, k).round().astype(int).tolist()))
    print(f"[validate] checking layers {layers} "
          f"(rtol={args.rtol} atol={args.atol})", file=sys.stderr)
    for li in layers:
        ma = np.load(m / f"layer_{li:02d}.npy", mmap_mode="r")
        ra = np.load(r / f"layer_{li:02d}.npy", mmap_mode="r")
        if ma.shape != ra.shape:
            fail(f"layer_{li:02d}: shape {ma.shape} != reference {ra.shape}")
        if ma.dtype != ra.dtype:
            fail(f"layer_{li:02d}: dtype {ma.dtype} != reference {ra.dtype}")
        t = ma.shape[0]
        for c0 in range(0, t, args.chunk):
            c1 = min(c0 + args.chunk, t)
            cmp_arrays(f"layer_{li:02d}[{c0}:{c1}]",
                       np.asarray(ma[c0:c1]), np.asarray(ra[c0:c1]),
                       args.rtol, args.atol)
        print(f"[validate] layer_{li:02d} OK  shape={tuple(ma.shape)}", file=sys.stderr)
        del ma, ra

    print(f"[validate] PASS — merged {m} matches reference {r} "
          f"({'all' if args.all_layers else len(layers)} layers checked)", file=sys.stderr)


if __name__ == "__main__":
    main()
