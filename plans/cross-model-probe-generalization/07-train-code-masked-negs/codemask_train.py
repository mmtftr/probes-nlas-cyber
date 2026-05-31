# [ai-generated]
"""exp-07: does masking trivial (non-live-code) negatives OUT of the span-max
loss change the honest live-code token AUC?

This is the paired-comparison runner for ONE layer. For that layer it trains the
SAME span-max probe twice per seed:

    mode="none"       baseline — every out-of-span token is a negative (the
                      `train_probe_spanmax.train_one_layer` default).
    mode="code_only"  trivial negatives (~live_code & y==0: comments, imports,
                      signatures, whitespace) are EXCLUDED from the loss
                      entirely (see train_one_layer(mask_negatives="code_only")
                      and src/eval/code_mask.py for the motivation).

Both modes are then evaluated on the SAME held-out TEST tokens with the honest
contrast (`honest_token_aucs`): the inflated all-token `tokens_auc`, the
live-code-only `tokens_code_auc`, and `dropped_fraction`. Example-AUC rides
along (max-pool, the canonical `example_scores`). The headline output is
`delta_tokens_code_auc = mean(code_only) - mean(none)`: positive means masking
trivial negatives helped the probe discriminate live-code-positive from
live-code-negative.

Structure mirrors 06/train_all_layers.py (same REPO insert, same split loading
via train_eval.load_or_make_split, same offsets/dataset loaders). Differences:
single layer per invocation (--layer), two mask modes x 5 seeds, aggregate to
one metrics.json.

The TRAIN-split live-code mask reuses `honest_scoring.build_code_mask` (the
exact same per-eid `code_only_mask` logic the TEST honest contrast uses) so the
train-time filter and the eval-time honest metric never diverge.

--dry-run fabricates a tiny synthetic acts dir + dataset + split in a tmpdir and
runs the FULL path on layer 0 with epochs=1, asserting both modes land in
metrics.json. It exercises the real train_one_layer (needs torch).
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    build_code_mask,
    honest_token_aucs,
    load_dataset_rows,
    load_offsets_npz,
)
from sklearn.metrics import roc_auc_score  # noqa: E402

MODES = ("none", "code_only")
SEEDS = (42, 43, 44, 45, 46)


def _load_train_eval():
    """Import the canonical train_eval module by path (no __init__ in its dir)."""
    p = REPO / "src" / "remotes" / "clariden" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("clariden_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ms(vals):
    """(mean, std, n) over the finite, non-None values."""
    a = np.array([v for v in vals if v is not None and v == v], dtype=float)
    return (float(a.mean()), float(a.std(ddof=0)), int(a.size)) if a.size else (None, None, 0)


def _train_eval_one_cell(
    Xfull, y, eids, tr, te, li, mode, seed, epochs, device,
    offsets_by_eid, dataset_rows_by_eid, te_mod,
):
    """Train one (mode, seed) cell on a single layer; return the per-seed record.

    Returns {mode, seed, tokens_code_auc, tokens_auc, dropped_fraction, ex_auc}.
    On a failure (degenerate labels, divergence, non-finite acts) returns the
    same dict with NaN AUCs and an `error`/`skipped` note — never raises, so the
    paired comparison still produces a row.
    """
    rec = {"mode": mode, "seed": seed}
    try:
        Xtr = np.asarray(Xfull[tr], dtype=np.float32)
        ytr, etr = y[tr], eids[tr]
        if len(np.unique(ytr)) < 2 or te.sum() == 0:
            return {**rec, "tokens_code_auc": float("nan"), "tokens_auc": float("nan"),
                    "dropped_fraction": float("nan"), "ex_auc": float("nan"),
                    "skipped": "degenerate labels"}
        if not np.isfinite(Xtr).all():
            return {**rec, "tokens_code_auc": float("nan"), "tokens_auc": float("nan"),
                    "dropped_fraction": float("nan"), "ex_auc": float("nan"),
                    "error": "non-finite activations"}

        # TRAIN-split live-code mask, aligned to the TRAIN tokens (etr order).
        # Built with the SAME build_code_mask the TEST honest contrast uses, so
        # train-time filtering and eval-time metric never diverge.
        train_code_mask = None
        if mode == "code_only":
            train_code_mask = build_code_mask(etr, offsets_by_eid, dataset_rows_by_eid)

        # train_one_layer accepts `seed` directly (drives torch.manual_seed +
        # np.random.seed + the internal val split). Pass it through; vary the
        # OUTER test split via `seed` upstream (caller's masks), same as exp-03.
        r = train_one_layer(
            Xtr, ytr, etr, epochs=epochs, device=device, verbose=False, seed=seed,
            mask_negatives=mode,
            code_mask=(train_code_mask if mode == "code_only" else None),
        )
        w, b = np.asarray(r["w"], np.float32), float(r["b"])
        if not (np.isfinite(w).all() and np.isfinite(b)):
            return {**rec, "tokens_code_auc": float("nan"), "tokens_auc": float("nan"),
                    "dropped_fraction": float("nan"), "ex_auc": float("nan"),
                    "error": "diverged (NaN probe weights)"}

        Xte = np.asarray(Xfull[te], dtype=np.float32)
        tok_p = 1.0 / (1.0 + np.exp(-(Xte @ w + b)))
        tok_y, te_eids = y[te], eids[te]
        honest = honest_token_aucs(
            tok_p, tok_y, te_eids, offsets_by_eid, dataset_rows_by_eid,
        )
        ex_ids, ex_p = te_mod.example_scores(tok_p, te_eids)
        ex_y = np.array([int(y[(eids == e)].max() > 0) for e in ex_ids])
        ex_auc = (float(roc_auc_score(ex_y, ex_p))
                  if len(np.unique(ex_y)) > 1 else float("nan"))
        return {**rec,
                "tokens_code_auc": honest["tokens_code_auc"],
                "tokens_auc": honest["tokens_auc"],
                "dropped_fraction": honest["dropped_fraction"],
                "ex_auc": ex_auc,
                "n_pos_code": honest["n_pos_code"],
                "n_total_code": honest["n_total_code"]}
    except Exception as e:  # noqa: BLE001 — record + continue, never abort the sweep
        return {**rec, "tokens_code_auc": float("nan"), "tokens_auc": float("nan"),
                "dropped_fraction": float("nan"), "ex_auc": float("nan"),
                "error": f"{type(e).__name__}: {str(e)[:200]}"}


def run(acts_dir, dataset, split, out, layer, offsets, epochs, model, device, te_mod):
    """Train+eval both mask modes x all seeds on one layer; write metrics.json."""
    acts = Path(acts_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    offsets_path = Path(offsets) if offsets else (acts / "offsets.npz")
    offsets_by_eid = load_offsets_npz(offsets_path)
    dataset_rows_by_eid = load_dataset_rows(Path(dataset))

    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")

    _rows, train_eids, _test_eids = te_mod.load_or_make_split(
        Path(dataset), Path(split)
    )
    tr = np.fromiter((int(e) in train_eids for e in eids), bool, len(eids))
    te = ~tr

    Xfull = np.load(acts / f"layer_{layer:02d}.npy", mmap_mode="r")

    # Per-seed cells for both modes.
    per_seed = {m: [] for m in MODES}
    for mode in MODES:
        for seed in SEEDS:
            cell = _train_eval_one_cell(
                Xfull, y, eids, tr, te, layer, mode, seed, epochs, device,
                offsets_by_eid, dataset_rows_by_eid, te_mod,
            )
            per_seed[mode].append(cell)
            note = cell.get("error") or cell.get("skipped") or ""
            print(f"[codemask] L{layer:02d} {mode} s{seed}  "
                  f"tokens_code_auc={cell['tokens_code_auc']!r} "
                  f"tokens_auc={cell['tokens_auc']!r} "
                  f"dropped={cell['dropped_fraction']!r} ex={cell['ex_auc']!r} {note}",
                  file=sys.stderr)

    # Aggregate per mode.
    rows = []
    code_auc_means = {}
    for mode in MODES:
        cells = per_seed[mode]
        cm, cs, n = _ms([c["tokens_code_auc"] for c in cells])
        tm, ts, _ = _ms([c["tokens_auc"] for c in cells])
        em, es, _ = _ms([c["ex_auc"] for c in cells])
        code_auc_means[mode] = cm
        rows.append({
            "mode": mode,
            "tokens_code_auc_mean": cm, "tokens_code_auc_std": cs,
            "tokens_auc_mean": tm, "tokens_auc_std": ts,
            "ex_auc_mean": em, "ex_auc_std": es,
            "n": n,
            "per_seed": [[c["seed"], c["tokens_code_auc"]] for c in cells],
        })

    # delta = mean(code_only) - mean(none) on the headline honest metric.
    cm_code = code_auc_means.get("code_only")
    cm_none = code_auc_means.get("none")
    delta = (cm_code - cm_none) if (cm_code is not None and cm_none is not None) else None

    record = {
        "model": model,
        "layer": layer,
        "rows": rows,
        "delta_tokens_code_auc": delta,
    }
    (out / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"[codemask] L{layer:02d} delta_tokens_code_auc={delta!r}  "
          f"-> {out / 'metrics.json'}", file=sys.stderr)
    return record


def _make_synthetic(tmp: Path):
    """Fabricate a tiny acts dir + dataset.jsonl + split.json for --dry-run.

    A handful of short Python snippets (some vulnerable, some clean) with
    per-char offsets, one captured layer (layer_00.npy), y/example_ids, meta.json,
    a matching mini dataset.jsonl, and a split.json holding out some groups.
    Returns (acts_dir, dataset_path, split_path).
    """
    rng = np.random.default_rng(0)
    acts = tmp / "acts"
    acts.mkdir(parents=True, exist_ok=True)

    # Mini dataset: row index == eid. Mix of langs/labels; distinct _file_name so
    # pair_group_key spreads them across groups (some land train, some test).
    snippets = [
        ("# c0\nimport os\ndef f(x):\n    y = x + 1\n    return y\n", "python", 1),
        ("# c1\nimport sys\ndef g(z):\n    w = z * 2\n    return w\n", "python", 0),
        ("# c2\nfrom a import b\ndef h(p):\n    q = p - 3\n    return q\n", "python", 1),
        ("# c3\nimport re\ndef k(m):\n    n = m / 4\n    return n\n", "python", 0),
        ("# c4\nimport io\ndef j(a):\n    c = a + 5\n    return c\n", "python", 1),
        ("# c5\nimport abc\ndef l(d):\n    e = d - 6\n    return e\n", "python", 0),
    ]
    rows = []
    for i, (code, lang, lab) in enumerate(snippets):
        rows.append({"code": code, "lang": lang, "label": lab,
                     "_file_name": f"file_{i}.py", "_func_name": f"fn_{i}"})
    dataset_path = tmp / "dataset.jsonl"
    dataset_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    # Per-char offsets per row (offsets_row_NNNN), flat tokens, labels.
    # Positive token: the assigned var char in the body (a crude vuln stand-in)
    # for label==1 rows only. Live-code mask will keep the body, drop comment/import.
    offsets_dict = {}
    X_parts, y_parts, eid_parts = [], [], []
    hidden = 4
    for eid, (code, lang, lab) in enumerate(snippets):
        offs = np.array([(c, c + 1) for c in range(len(code))], dtype=np.int32)
        offsets_dict[f"offsets_row_{eid:04d}"] = offs
        T = len(code)
        ytok = np.zeros(T, dtype=np.int64)
        if lab == 1:
            # mark the body assignment char(s) positive: the line "    y = x + 1"
            body_marker = code.find("    ", code.find("def "))
            if body_marker >= 0:
                # the first non-space char after the body indent
                pos = body_marker + 4
                if pos < T:
                    ytok[pos] = 1
        X = rng.standard_normal((T, hidden)).astype(np.float32)
        # Make positives mildly separable so AUC is defined / non-degenerate.
        X[ytok == 1] += 1.5
        X_parts.append(X)
        y_parts.append(ytok)
        eid_parts.append(np.full(T, eid, dtype=np.int64))

    np.savez(acts / "offsets.npz", **offsets_dict)
    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    eid_all = np.concatenate(eid_parts, axis=0)
    np.save(acts / "layer_00.npy", X_all)
    np.save(acts / "y.npy", y_all)
    np.save(acts / "example_ids.npy", eid_all)
    (acts / "meta.json").write_text(json.dumps({"n_layers": 1}))

    # Split: hold out a couple of groups so train + test are both non-empty and
    # each carries at least one positive (eids 0,2,4 positive; 1,3,5 clean).
    # pair_group_key -> "func::file_i.py::fn_i". Hold out groups for eids 4,5.
    heldout = ["func::file_4.py::fn_4", "func::file_5.py::fn_5"]
    split_path = tmp / "split.json"
    split_path.write_text(json.dumps(
        {"seed": 42, "frac_heldout": 0.33, "n_groups": 6, "heldout_groups": heldout}))
    return acts, dataset_path, split_path


def _dry_run() -> int:
    """Self-contained smoke test of the full path on synthetic data."""
    te_mod = _load_train_eval()
    try:
        import torch  # noqa: F401
        device = "cpu"
    except Exception as e:  # pragma: no cover
        print(f"[codemask] dry-run: torch unavailable ({e})", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        acts, dataset_path, split_path = _make_synthetic(tmp)
        out = tmp / "out"
        record = run(
            acts_dir=acts, dataset=dataset_path, split=split_path, out=out,
            layer=0, offsets=None, epochs=1, model="dry-run", device=device,
            te_mod=te_mod,
        )
        mj = out / "metrics.json"
        assert mj.exists(), "metrics.json not written"
        loaded = json.loads(mj.read_text())
        modes_present = {r["mode"] for r in loaded["rows"]}
        assert modes_present == set(MODES), f"expected both modes, got {modes_present}"
        assert "delta_tokens_code_auc" in loaded
        for r in loaded["rows"]:
            assert "tokens_code_auc_mean" in r and "per_seed" in r
            assert len(r["per_seed"]) == len(SEEDS)
    print("OK")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir")
    ap.add_argument("--dataset")
    ap.add_argument("--split")
    ap.add_argument("--out")
    ap.add_argument("--layer", type=int, help="single layer index (required unless --dry-run)")
    ap.add_argument("--offsets", default=None,
                    help="Per-row char offsets npz (offsets_row_NNNN keys). "
                         "Defaults to <acts-dir>/offsets.npz.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the full path on a tiny synthetic acts dir; assert "
                         "both modes land in metrics.json; print OK; exit 0.")
    args = ap.parse_args()

    if args.dry_run:
        sys.exit(_dry_run())

    missing = [n for n in ("acts_dir", "dataset", "split", "out") if getattr(args, n) is None]
    if missing or args.layer is None:
        ap.error(f"--{'/--'.join(missing + (['layer'] if args.layer is None else []))} required")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    te_mod = _load_train_eval()
    run(
        acts_dir=args.acts_dir, dataset=args.dataset, split=args.split, out=args.out,
        layer=args.layer, offsets=args.offsets, epochs=args.epochs, model=args.model,
        device=device, te_mod=te_mod,
    )


if __name__ == "__main__":
    main()
