# [ai-generated]
"""Fast CPU self-test of train_cross.py's NEW logic (no cluster, no real acts).

Fakes two `state` dicts with small random activations and exercises:
  - train_probe()  (the train_one_layer invocation + <2-class guard)
  - eval_on()      (token-AUC, g-mean^2, pairAcc on CODE tokens; cross-apply w)
  - ex_max_prob / pair_stats / max_gmean wiring

Goal: catch integration bugs (signatures, masks, NaNs, shape errors) before an
unattended overnight cluster run. It does NOT validate make_state/load_layer
(those are reused unchanged from the proven exp-19 train_grid).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_cross as tc


def fake_state(name, n_ex=80, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    # n_ex examples, ~12 tokens each; half vuln (label via eid parity)
    toks_per = rng.integers(8, 16, n_ex)
    eids = np.repeat(np.arange(n_ex, dtype=np.int32), toks_per)
    N = len(eids)
    # a STRONG, trivially-separable signal so the sanity check is unambiguous:
    # positive code tokens get a large push along a fixed direction.
    w_true = rng.normal(size=dim)
    w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(N, dim)).astype(np.float32)
    is_vuln_ex = (np.arange(n_ex) % 2 == 0)
    isc = rng.random(N) > 0.25                      # ~75% code tokens
    truth = np.zeros(N, bool)
    for e in range(n_ex):
        idx = np.where(eids == e)[0]
        if is_vuln_ex[e]:
            # a few positive code tokens carry the signal
            pos = idx[isc[idx]][:3]
            truth[pos] = True
            X[pos] += 8.0 * w_true
    ypos = (truth & isc).astype(np.int8)
    # pairs: vuln eid e -> safe eid e+1
    v2s = {e: e + 1 for e in range(0, n_ex, 2)}
    sub_set = set(range(n_ex))                      # all subtractive (for test)
    test_eids = set(range(n_ex // 2, n_ex))         # second half = test
    is_test = np.fromiter((int(e) in test_eids for e in eids), bool, N)
    in_sub = np.ones(N, bool)
    langs = np.array(["c", "cpp", "python"])[rng.integers(0, 3, n_ex)]
    cpp_tok = np.fromiter((langs[int(e)] in tc.CPP for e in eids), bool, N)
    ds = [{"lang": langs[e]} for e in range(n_ex)]
    return dict(name=name, ds=ds, X=X, eids=eids, isc=isc, truth=truth, ypos=ypos,
                v2s=v2s, sub_set=sub_set, test_set=test_eids,
                is_test=is_test, in_sub=in_sub, cpp_tok=cpp_tok)


def main():
    sven = fake_state("sven", seed=1)
    pv = fake_state("pv", seed=2)
    # train a probe on sven (subtractive ∩ train)
    train_mask = sven["in_sub"] & ~sven["is_test"]
    wb = tc.train_probe(sven, train_mask, epochs=20, device="cpu")
    assert wb is not None, "train_probe returned None on 2-class data"
    w, b = wb
    assert w.shape[0] == sven["X"].shape[1] and np.isfinite(b)
    # eval same probe on sven test and CROSS on pv test (apply w to pv X)
    for st, cpp in ((sven, True), (pv, False)):
        sub_p, add_p = tc.eval_set_pairs(st, cpp_only=cpp)
        m = tc.eval_on(w, b, st, st["is_test"] & (st["cpp_tok"] if cpp else np.ones(len(st["eids"]), bool)),
                       sub_p, add_p)
        print(f"eval on {st['name']:5s} cpp={cpp}: tokAUC={m['token_code_auc']:.3f} "
              f"g2={m['gmean_sq']:.3f} pairAcc[sub={m['pair_acc_sub']:.2f} add={m['pair_acc_add']:.2f}] "
              f"n_tok={m['n_eval_tok']} n_pos={m['n_eval_pos']} n_subp={m['n_sub_pairs']}")
        assert 0.0 <= m["gmean_sq"] <= 1.0 or m["gmean_sq"] != m["gmean_sq"]  # in [0,1] or NaN
        assert m["token_code_auc"] != m["token_code_auc"] or 0.0 <= m["token_code_auc"] <= 1.0
    # the in-domain sven probe should beat chance on its own signal
    sub_p, add_p = tc.eval_set_pairs(sven, cpp_only=False)
    m = tc.eval_on(w, b, sven, sven["is_test"], sub_p, add_p)
    assert m["token_code_auc"] > 0.6, f"in-domain AUC too low: {m['token_code_auc']}"
    print(f"\nIN-DOMAIN sven AUC={m['token_code_auc']:.3f} (>0.6 expected) — harness logic OK")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
