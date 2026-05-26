"""Leakage-aware re-evaluation of the linear probe.

The headline AUC=0.963 on layer 17 (CyberSecEval position-paired data)
is suspicious because the safe/vulnerable variants of each snippet share
the same `_origin_repo`. A random `train_test_split` puts paired examples
in BOTH train and test, so the probe can memorise the repo rather than
learn a generalisable vulnerability signature.

This script re-runs the layer-17 probe under five splits of the SAME data
and writes a Markdown comparison to data/eval_splits.md. It also saves a
strict probe (worst credible split) to data/probe_strict.npz in the same
format src/stream_with_probe.py expects.

Splits:
  (a) Random stratified            -- the current baseline
  (b) Pair / repo group split      -- GroupShuffleSplit on _origin_repo
  (c) Held-out CWE (per top-5)     -- one CWE class held out at a time
  (d) Held-out language            -- python<->javascript
  (e) Held-out source              -- noop here (only one source in v2)

Constraints:
  - Reads data/activations_v2/activations_layer17.npz lockstep with
    data/pairs.jsonl (row i in jsonl == row i in npz).
  - Touches NOTHING in src/. Lives under scripts/.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(__file__).resolve().parent.parent
ACTS = ROOT / "data" / "activations_v2" / "activations_layer17.npz"
PAIRS = ROOT / "data" / "pairs.jsonl"
OUT_MD = ROOT / "data" / "eval_splits.md"
OUT_PROBE = ROOT / "data" / "probe_strict.npz"
LAYER = 17
SEED = 7


def load_data() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    npz = np.load(ACTS)
    X, y = npz["X"], npz["y"].astype(int)
    rows = [json.loads(line) for line in PAIRS.read_text().splitlines() if line.strip()]
    if len(rows) != len(X):
        raise SystemExit(f"row count mismatch: jsonl={len(rows)} npz={len(X)}")
    # Propagate CWE from each repo's positive row down to its paired negative,
    # so held-out-CWE splits keep pairs together by CWE rather than dumping
    # all `None`-CWE negatives into the training set.
    repo_cwe: dict[str, str | None] = {}
    for r in rows:
        if r["label"] == 1 and r.get("cwe"):
            repo_cwe[r["_origin_repo"]] = r["cwe"]
    for r in rows:
        if r["label"] == 0 and r.get("cwe") is None:
            r["_paired_cwe"] = repo_cwe.get(r["_origin_repo"])
        else:
            r["_paired_cwe"] = r.get("cwe")
    return X, y, rows


def fit_eval(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[float, float, np.ndarray, float, int, int]:
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan"), float("nan"), np.zeros(X.shape[1], dtype=np.float32), 0.0, len(train_idx), len(test_idx)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = clf.predict(Xte)
    auc = float(roc_auc_score(yte, prob))
    acc = float(accuracy_score(yte, pred))
    return auc, acc, clf.coef_[0].astype(np.float32), float(clf.intercept_[0]), len(train_idx), len(test_idx)


def split_random(X, y, rows):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    return [("random_stratified", tr, te)]


def split_group_repo(X, y, rows):
    groups = np.array([r["_origin_repo"] for r in rows])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    (tr, te), = gss.split(X, y, groups=groups)
    return [("group_repo", tr, te)]


def split_heldout_cwe(X, y, rows, top_k: int = 5):
    cwe_arr = np.array([r["_paired_cwe"] for r in rows], dtype=object)
    # Top CWE classes among POSITIVES (negatives inherit the paired CWE).
    pos_cwes = [r["_paired_cwe"] for r in rows if r["label"] == 1 and r["_paired_cwe"]]
    top = [c for c, _ in Counter(pos_cwes).most_common(top_k)]
    out = []
    for cwe in top:
        te = np.where(cwe_arr == cwe)[0]
        tr = np.where(cwe_arr != cwe)[0]
        out.append((f"heldout_cwe::{cwe}", tr, te))
    return out


def split_heldout_lang(X, y, rows):
    lang_arr = np.array([r["lang"] for r in rows])
    out = []
    for held in ("python", "javascript"):
        te = np.where(lang_arr == held)[0]
        tr = np.where(lang_arr != held)[0]
        out.append((f"heldout_lang::test={held}", tr, te))
    return out


def split_heldout_source(X, y, rows):
    src_arr = np.array([r["source"] for r in rows])
    uniq = sorted(set(src_arr.tolist()))
    # Both labels actually live under different `source` strings here
    # (`CyberSecEval` for positives, `CyberSecEval-leadup` for negatives),
    # so a leave-one-source-out split would have a single-class test set
    # and AUC is undefined. We surface that fact instead of running it.
    return [], uniq


def main() -> None:
    X, y, rows = load_data()
    print(f"loaded X={X.shape}  y_pos={int(y.sum())}  rows={len(rows)}")

    results: list[dict] = []

    for name, tr, te in split_random(X, y, rows):
        auc, acc, w, b, ntr, nte = fit_eval(X, y, tr, te)
        results.append(dict(split=name, auc=auc, acc=acc, n_train=ntr, n_test=nte, w=w, b=b))

    for name, tr, te in split_group_repo(X, y, rows):
        auc, acc, w, b, ntr, nte = fit_eval(X, y, tr, te)
        results.append(dict(split=name, auc=auc, acc=acc, n_train=ntr, n_test=nte, w=w, b=b))

    for name, tr, te in split_heldout_cwe(X, y, rows):
        auc, acc, w, b, ntr, nte = fit_eval(X, y, tr, te)
        results.append(dict(split=name, auc=auc, acc=acc, n_train=ntr, n_test=nte, w=w, b=b))

    for name, tr, te in split_heldout_lang(X, y, rows):
        auc, acc, w, b, ntr, nte = fit_eval(X, y, tr, te)
        results.append(dict(split=name, auc=auc, acc=acc, n_train=ntr, n_test=nte, w=w, b=b))

    src_splits, src_uniq = split_heldout_source(X, y, rows)
    for name, tr, te in src_splits:
        auc, acc, w, b, ntr, nte = fit_eval(X, y, tr, te)
        results.append(dict(split=name, auc=auc, acc=acc, n_train=ntr, n_test=nte, w=w, b=b))

    # ---- pretty print ----
    lines: list[str] = []
    lines.append("# Leakage-aware re-evaluation of the layer-17 probe")
    lines.append("")
    lines.append(f"Data: `data/activations_v2/activations_layer17.npz` (N={len(y)}, dim={X.shape[1]}, pos={int(y.sum())})")
    lines.append("")
    lines.append(f"Probe: `LogisticRegression(max_iter=1000, C=1.0)` on layer {LAYER}.")
    lines.append("")
    lines.append("| Split | AUC | ACC | n_train | n_test |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in results:
        auc = "n/a" if np.isnan(r["auc"]) else f"{r['auc']:.3f}"
        acc = "n/a" if np.isnan(r["acc"]) else f"{r['acc']:.3f}"
        lines.append(f"| `{r['split']}` | {auc} | {acc} | {r['n_train']} | {r['n_test']} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- `random_stratified` is the leaky baseline used by `src/train_probe.py`. The paired safe/vulnerable variants of a snippet share `_origin_repo`, so a random split puts both halves of the same pair in train AND test.")
    lines.append(f"- `group_repo` uses `GroupShuffleSplit` on `_origin_repo` -- the same repo never appears in both partitions.")
    lines.append(f"- `heldout_cwe::*` trains on every CWE except the named one and tests only on that CWE (negatives inherit their pair's CWE so held-out CWE pulls both halves out together).")
    lines.append(f"- `heldout_lang::*` trains on the other language and tests on the named one.")
    lines.append(f"- `heldout_source`: skipped. In `pairs.jsonl` the v2 `source` field is `CyberSecEval` for positives and `CyberSecEval-leadup` for negatives ({sorted(src_uniq)}), so a leave-one-source-out split would have a single-class test set and AUC would be undefined.")
    lines.append("")

    # Identify the worst credible split. We treat held-out-CWE and held-out-lang
    # as the most credible "does this generalise to a new vulnerability class
    # or a new language" stress tests.
    credible = [r for r in results if r["split"].startswith("heldout_cwe::") or r["split"].startswith("heldout_lang::")]
    worst = min(credible, key=lambda r: (float("inf") if np.isnan(r["auc"]) else r["auc"]))
    random_baseline = next(r for r in results if r["split"] == "random_stratified")
    group_repo = next(r for r in results if r["split"] == "group_repo")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Random stratified (leaky baseline): **AUC={random_baseline['auc']:.3f}** / ACC={random_baseline['acc']:.3f}")
    lines.append(f"- Repo-grouped: **AUC={group_repo['auc']:.3f}** / ACC={group_repo['acc']:.3f}")
    lines.append(f"- Worst credible split (`{worst['split']}`): **AUC={worst['auc']:.3f}** / ACC={worst['acc']:.3f}")
    lines.append("")
    lines.append(f"Recommended writeup number: **AUC={worst['auc']:.3f}** under `{worst['split']}` (the strictest credible stress test).")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")

    # ---- save strict probe ----
    np.savez_compressed(
        OUT_PROBE,
        w=worst["w"],
        b=np.float32(worst["b"]),
        layer=np.int32(LAYER),
    )
    print(f"saved {OUT_PROBE}  (split={worst['split']}  AUC={worst['auc']:.3f})")

    # Companion card so the strict probe is self-documenting.
    card_path = OUT_PROBE.with_suffix(".json")
    card_path.write_text(json.dumps({
        "layer": LAYER,
        "split": worst["split"],
        "auc": worst["auc"],
        "acc": worst["acc"],
        "n_train": worst["n_train"],
        "n_test": worst["n_test"],
        "source_activations": str(ACTS.relative_to(ROOT)),
        "source_pairs": str(PAIRS.relative_to(ROOT)),
        "all_splits": [
            {"split": r["split"], "auc": r["auc"], "acc": r["acc"], "n_train": r["n_train"], "n_test": r["n_test"]}
            for r in results
        ],
    }, indent=2))
    print(f"saved {card_path}")


if __name__ == "__main__":
    main()
