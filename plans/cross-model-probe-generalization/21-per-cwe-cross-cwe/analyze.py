# [ai-generated]
"""exp-21 analysis: render the CWE×CWE transfer matrices + injection/memory block
summary with Wilson CIs, for each model. Method-agnostic (reads matrix.json)."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = {"qwen32b": "Qwen2.5-Coder-32B (L25)", "gemma1b": "Gemma-3-1b-it (L25)"}
INJ = ["CWE-022", "CWE-078", "CWE-079", "CWE-089"]   # injection
MEM = ["CWE-125", "CWE-190", "CWE-416", "CWE-476", "CWE-787"]  # memory/other
ORDER = INJ + MEM


def wilson(p, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def block_ci(mat, npairs, rows, cols):
    """pooled pair-acc + Wilson CI over all (row-probe,col-test) pairs in block."""
    succ = tot = 0
    for c in rows:
        for cp in cols:
            v, n = mat[c].get(cp), npairs.get(cp, 0)
            if v is not None and v == v and n:
                succ += round(v * n); tot += n
    p = succ / tot if tot else float("nan")
    lo, hi = wilson(p, tot)
    return p, lo, hi, tot


def show(key):
    m = json.load(open(HERE / "results" / key / "matrix.json"))
    cwes = [c for c in ORDER if c in m["cwes"]]
    pa = m["pairacc_natural"]; pb = m["pairacc_balanced"]
    nt = m["n_test_pairs"]; ntr = m["n_train_pairs"]
    print(f"\n{'='*70}\n{MODELS[key]}   (natural train sizes; pair-accuracy, vuln vs its own patch)")
    print("rows = TRAIN-CWE probe, cols = TEST-CWE.  [inj | mem]")
    hdr = "train\\test   " + " ".join(f"{c[-3:]:>5s}" for c in cwes) + "   self"
    print(hdr)
    for c in cwes:
        cells = " ".join(
            (f"{pa[c][cp]:5.2f}" if pa[c].get(cp) is not None and pa[c][cp] == pa[c][cp] else "   . ")
            for cp in cwes)
        tag = "INJ" if c in INJ else "mem"
        print(f"{c} {tag} {cells}   {pa[c][c]:.2f}  (ntr={ntr[c]},nte={nt[c]})")
    print("\nBlock pair-acc [natural] with 95% Wilson CI (pooled over test pairs):")
    for gr, rows in (("inj", INJ), ("mem", MEM)):
        for gc, cols in (("inj", INJ), ("mem", MEM)):
            p, lo, hi, tot = block_ci(pa, nt, [c for c in rows if c in cwes], [c for c in cols if c in cwes])
            print(f"  {gr}->{gc}:  {p:.3f}  [{lo:.3f}, {hi:.3f}]  (n={tot})")
    print("Block pair-acc [balanced-15]:")
    for gr, rows in (("inj", INJ), ("mem", MEM)):
        for gc, cols in (("inj", INJ), ("mem", MEM)):
            p, lo, hi, tot = block_ci(pb, nt, [c for c in rows if c in cwes], [c for c in cols if c in cwes])
            print(f"  {gr}->{gc}:  {p:.3f}  [{lo:.3f}, {hi:.3f}]  (n={tot})")
    # diagonal (self) vs off-diagonal within injection
    inj = [c for c in INJ if c in cwes]
    self_inj = np.nanmean([pa[c][c] for c in inj])
    off_inj = np.nanmean([pa[c][cp] for c in inj for cp in inj if cp != c])
    mem = [c for c in MEM if c in cwes]
    self_mem = np.nanmean([pa[c][c] for c in mem])
    print(f"\ninjection: self-detect={self_inj:.3f}  cross(off-diag)={off_inj:.3f}  -> "
          f"{'feature shared' if off_inj>0.52 else 'idiosyncratic'}")
    print(f"memory:    self-detect={self_mem:.3f}  -> "
          f"{'has signal' if self_mem>0.55 else 'NO self signal (≈chance)'}")


if __name__ == "__main__":
    for k in MODELS:
        show(k)
