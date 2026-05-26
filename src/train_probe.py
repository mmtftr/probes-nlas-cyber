"""Train a linear probe on Gemma 4 hidden-state activations.

Reads data/activations/activations_layer*.npz, trains a LogisticRegression
per layer, reports AUC + accuracy, saves the best probe as data/probe.npz
plus a small `probe_card.json` with metrics.

The "probe" is just (w, b) for w @ activation + b → vulnerability logit.
Apply sigmoid to get a 0..1 risk score.
"""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split


# Lazy + guarded wandb import. `wandb` may not be installed; in that case
# every call below becomes a no-op and the script keeps running.
try:
    import wandb  # type: ignore
    _HAS_WANDB = True
except Exception:  # ImportError or stale install
    wandb = None  # type: ignore
    _HAS_WANDB = False


def _wandb_init(args: argparse.Namespace) -> "object | None":
    """Initialise a wandb run if --wandb-project was passed.

    Returns the run handle (or None). All wandb interactions are
    try/except-wrapped so a broken wandb install can't kill training.
    """
    if not args.wandb_project:
        return None
    if not _HAS_WANDB:
        print("[train_probe] wandb not installed, skipping logging", file=sys.stderr)
        return None
    try:
        mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"probe-sample-{ts}"
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            mode=mode,
            config=vars(args),
            job_type="probe-train",
        )
        print(f"[train_probe] wandb run: {run_name} (mode={mode})", file=sys.stderr)
        return run
    except Exception as e:
        print(f"[train_probe] wandb init failed: {e}", file=sys.stderr)
        return None


def _wandb_log(run, payload: dict, step: int | None = None) -> None:
    if run is None:
        return
    try:
        if step is not None:
            wandb.log(payload, step=step)
        else:
            wandb.log(payload)
    except Exception as e:
        print(f"[train_probe] wandb.log failed: {e}", file=sys.stderr)


def _wandb_artifact(run, path: Path, name: str, art_type: str, metadata: dict) -> None:
    if run is None or not path.exists():
        return
    try:
        art = wandb.Artifact(name=name, type=art_type, metadata=metadata)
        art.add_file(str(path))
        run.log_artifact(art)
    except Exception as e:
        print(f"[train_probe] wandb artifact failed: {e}", file=sys.stderr)


def _wandb_finish(run) -> None:
    if run is None:
        return
    try:
        wandb.finish()
    except Exception:
        pass


def fit_layer(X: np.ndarray, y: np.ndarray, seed: int = 7, groups: np.ndarray | None = None) -> dict:
    """Fit a LogReg probe on activations.

    If `groups` is provided, uses `GroupShuffleSplit` so paired safe/vulnerable
    siblings (same `_origin_repo` / `(_file_name, _func_name)`) cannot land in
    both train and test — closes the training-side leakage in issue #18. If
    not provided, falls back to the legacy stratified split.
    """
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(X, y, groups=groups))
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        split_kind = "group_shuffle"
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        split_kind = "random_stratified"
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)
    yhat_prob = clf.predict_proba(Xte)[:, 1]
    yhat_cls = clf.predict(Xte)
    auc = roc_auc_score(yte, yhat_prob)
    acc = accuracy_score(yte, yhat_cls)
    return {
        "auc": float(auc),
        "acc": float(acc),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "pos_rate": float(y.mean()),
        "w": clf.coef_[0].astype(np.float32),
        "b": float(clf.intercept_[0]),
        "split_kind": split_kind,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="data/activations")
    ap.add_argument("--out", default="data/probe.npz")
    ap.add_argument("--card", default="data/probe_card.json")
    ap.add_argument(
        "--pairs",
        default=None,
        help="Optional path to pairs.jsonl row-aligned with activations; "
        "when set, uses GroupShuffleSplit on pair_group_key() to avoid "
        "sibling leakage during layer selection (issue #18).",
    )
    ap.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name; omit to disable wandb logging",
    )
    args = ap.parse_args()

    acts_dir = Path(args.acts_dir)
    files = sorted(acts_dir.glob("activations_layer*.npz"))
    if not files:
        print(f"[train_probe] no activation files in {acts_dir}", file=sys.stderr)
        sys.exit(1)

    groups: np.ndarray | None = None
    if args.pairs:
        try:
            from src.eval.splits import pair_group_key  # type: ignore
        except Exception as e:
            print(f"[train_probe] could not import pair_group_key ({e}); falling back to random split", file=sys.stderr)
        else:
            rows = []
            with open(args.pairs) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            groups = np.array([pair_group_key(r) for r in rows])
            print(f"[train_probe] group split enabled — {len(set(groups.tolist()))} unique groups across {len(rows)} rows", file=sys.stderr)

    run = _wandb_init(args)

    results: list[dict] = []
    for f in files:
        layer = int(f.stem.split("layer")[-1])
        npz = np.load(f)
        X, y = npz["X"], npz["y"]
        print(f"[train_probe] layer {layer:02d}  X={X.shape}  y_pos={int(y.sum())}", file=sys.stderr)
        # If groups were loaded and row counts mismatch, fall back to
        # random — better than crashing on a misaligned dataset.
        gr = groups if (groups is not None and len(groups) == X.shape[0]) else None
        r = fit_layer(X, y, groups=gr)
        r["layer"] = layer
        results.append(r)
        print(f"[train_probe] layer {layer:02d}  AUC={r['auc']:.3f}  ACC={r['acc']:.3f}  split={r['split_kind']}", file=sys.stderr)
        # One log row per layer, indexed by layer id for a clean x-axis.
        _wandb_log(
            run,
            {
                "layer/auc": r["auc"],
                "layer/acc": r["acc"],
                "layer/n_train": r["n_train"],
                "layer/n_test": r["n_test"],
                "layer/pos_rate": r["pos_rate"],
                "layer": layer,
            },
            step=layer,
        )

    best = max(results, key=lambda r: r["auc"])
    print(f"[train_probe] best layer = {best['layer']}  AUC={best['auc']:.3f}", file=sys.stderr)

    np.savez_compressed(
        args.out,
        w=best["w"],
        b=np.float32(best["b"]),
        layer=np.int32(best["layer"]),
    )
    card = {
        "best_layer": best["layer"],
        "best_auc": best["auc"],
        "best_acc": best["acc"],
        "all_layers": [
            {"layer": r["layer"], "auc": r["auc"], "acc": r["acc"], "n_train": r["n_train"], "n_test": r["n_test"]}
            for r in results
        ],
        "pos_rate": best["pos_rate"],
    }
    Path(args.card).write_text(json.dumps(card, indent=2))
    print(f"[train_probe] saved {args.out} + {args.card}", file=sys.stderr)

    # Summary metrics + a per-layer table for quick comparison.
    if run is not None:
        try:
            table = wandb.Table(columns=["layer", "auc", "acc", "n_train", "n_test"])
            for r in results:
                table.add_data(r["layer"], r["auc"], r["acc"], r["n_train"], r["n_test"])
            wandb.log({"per_layer_table": table})
            wandb.summary["best_layer"] = best["layer"]
            wandb.summary["best_auc"] = best["auc"]
            wandb.summary["best_acc"] = best["acc"]
            wandb.summary["pos_rate"] = best["pos_rate"]
        except Exception as e:
            print(f"[train_probe] wandb summary failed: {e}", file=sys.stderr)

    # Save probe weights as a reproducible artifact.
    _wandb_artifact(
        run,
        Path(args.out),
        name="probe-weights",
        art_type="model",
        metadata={
            "best_layer": best["layer"],
            "best_auc": best["auc"],
            "best_acc": best["acc"],
        },
    )
    _wandb_finish(run)


if __name__ == "__main__":
    main()
