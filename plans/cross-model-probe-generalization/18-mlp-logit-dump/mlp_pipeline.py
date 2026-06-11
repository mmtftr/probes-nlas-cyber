# [ai-generated]
"""Resumable, 16-GPU MLP-probe logit dump (exp-18).

Third logit-dump variant: exp-16 dumped the LINEAR span-max probe's logits,
exp-17 the verbalized read; this dumps the MLP-probe family (exp-12). The MLP
scores as sigmoid(probe(X)) (NOT the linear X·w+b), so it is a separate dump.

Six instruct models, both heads (mlp256, mlp512):
  * 2 have a known exp-12 MLP best-layer (gemma-3-27b-it L21, Qwen2.5-Coder-32B
    L37, same layer for both heads) -> "fixed": extract just that layer + dump.
  * 4 have NO MLP sweep (gemma-3 1b/4b/12b-it, Qwen2.5-Coder-7B) -> "sweep":
    extract ALL layers, MLP-sweep every layer (val_tokens_code_auc-selected,
    identical carve to exp-12), then dump at each head's best layer.

ONE worker per GPU (rank in [0, world), passed via --rank/--world-size). Work is
decomposed into per-unit files with skip-if-exists, so a worker that hits a time
limit just gets re-run and continues. Three stages, dependency-gated by file presence:

  1. extract : (model, shard) — example-sharded (eid % EXTRACT_SHARDS). Each shard
     forwards its ~1/16 of rows ONCE, writes layer{NN}_shard{R}.npz (X,y,eids) for
     every captured layer + offsets_shard{R}.npz, then EXTRACT_shard{R}.DONE.
     EXTRACT_SHARDS is FIXED (16) so the shard files are identical regardless of
     how many GPUs (world) actually run — a 4-GPU preflight and a 16-GPU run
     produce the same 16 shards.
  2. sweep   : (model, layer, head) — concat the 16 shard files for the layer,
     carve fit/val (15% groups, VAL_SEED=42) exactly as exp-12, train MLP on FIT,
     write sweep_<head>/layer{NN}.json with val/test honest tokens_code_auc.
     (sweep models only.)
  3. dump    : (model, head) — pick best layer (max val_tokens_code_auc; for fixed
     models the known layer), retrain the MLP on FIT (deterministic seed=7 -> same
     probe as the sweep), dump every per-token logit = probe(X) + prob, max-pool
     per example, with char offsets + live-code mask + is_test. Gate: test
     tokens_code_auc vs exp-12 (fixed models) or vs the sweep cell (sweep models).

The MLP training (train_one_layer + MLPProbe via probe_factory, epochs=30, seed=7)
and the fit/val carve are byte-for-byte exp-12's, so the gate reproduces its tc.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import MLPProbe, train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs, load_offsets_npz, load_dataset_rows,
)
from src.eval.code_mask import code_only_mask  # noqa: E402
from src.data.extract_token_activations import (  # noqa: E402
    _load_model, _load_tokenizer, _row_label,
)
from src.eval.token_data import (  # noqa: E402
    char_spans_to_token_spans, parse_spans, token_labels_array,
)

# ---- plan ---------------------------------------------------------------------

EXTRACT_SHARDS = 16            # FIXED example-shard count (eid % 16)
HEADS = ["mlp256", "mlp512"]
VAL_FRAC = 0.15               # exp-12 selection-val carve
VAL_SEED = 42
EPOCHS = 30

# exp-12 val-selected MLP best layers + their TEST tokens_code_auc (the gate).
HIST_MLP = {
    "google/gemma-3-27b-it": {"layer": 21, "mlp256": 0.822, "mlp512": 0.824},
    "Qwen/Qwen2.5-Coder-32B-Instruct": {"layer": 37, "mlp256": 0.817, "mlp512": 0.816},
}

MODELS = [
    {"id": "google/gemma-3-1b-it", "mode": "sweep", "n_layers": 26},
    {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "mode": "sweep", "n_layers": 28},
    {"id": "google/gemma-3-4b-it", "mode": "sweep", "n_layers": 34},
    {"id": "google/gemma-3-12b-it", "mode": "sweep", "n_layers": 48},
    {"id": "google/gemma-3-27b-it", "mode": "fixed", "layers": [21]},
    {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "mode": "fixed", "layers": [37]},
]


def slug(model_id: str) -> str:
    out = []
    for ch in model_id:
        out.append(ch if (ch.isalnum() or ch in "._-") else "_")
    return "".join(out)


def model_layers(m: dict) -> list[int]:
    return list(range(m["n_layers"])) if m["mode"] == "sweep" else list(m["layers"])


def _load_train_eval():
    p = REPO / "src" / "remotes" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- stage 1: extraction ------------------------------------------------------

def extract_shard(m: dict, shard: int, dataset: Path, out_root: Path, log) -> None:
    """Forward this shard's rows (eid % EXTRACT_SHARDS == shard) once, capturing
    every layer in model_layers(m). Resumable via EXTRACT_shard{R}.DONE."""
    import torch

    acts = out_root / slug(m["id"]) / "acts"
    acts.mkdir(parents=True, exist_ok=True)
    done = acts / f"EXTRACT_shard{shard:02d}.DONE"
    if done.exists():
        return
    layers = model_layers(m)

    rows = [json.loads(l) for l in dataset.open() if l.strip()]
    mine = [eid for eid in range(len(rows)) if eid % EXTRACT_SHARDS == shard]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = _load_tokenizer(m["id"])
    model = _load_model(m["id"], dtype)
    model.to(device).eval()
    log(f"extract {m['id']} shard {shard}: {len(mine)} rows, layers={layers[:3]}"
        f"{'...' if len(layers) > 3 else ''} ({len(layers)})")

    per_layer = {li: [] for li in layers}
    y_all, eid_all, offs = [], [], {}
    with torch.inference_mode():
        for eid in mine:
            row = rows[eid]
            enc = tok(row["code"], return_offsets_mapping=True, truncation=True,
                      max_length=2048, return_tensors=None)
            ids = torch.tensor([enc["input_ids"]], dtype=torch.long, device=device)
            offsets = enc["offset_mapping"]
            out = model(ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states
            n_pos = ids.shape[1]
            tok_spans = char_spans_to_token_spans(parse_spans(row), offsets)
            tok_labels, _ = token_labels_array(n_pos, tok_spans)
            for li in layers:
                per_layer[li].append(hs[li + 1][0].detach().to("cpu").float().numpy())
            y_all.extend(tok_labels.tolist())
            eid_all.extend([eid] * n_pos)
            offs[f"offsets_row_{eid:04d}"] = np.array(offsets, dtype=np.int32)

    y = np.array(y_all, dtype=np.int8)
    eid_arr = np.array(eid_all, dtype=np.int32)
    for li in layers:
        mat = np.vstack(per_layer[li]).astype(np.float32) if per_layer[li] else np.zeros((0, 1), np.float32)
        np.savez(acts / f"layer{li:02d}_shard{shard:02d}.npz", X=mat, y=y, example_ids=eid_arr)
    np.savez(acts / f"offsets_shard{shard:02d}.npz", **offs)
    done.write_text("")
    log(f"extract {m['id']} shard {shard}: DONE ({len(eid_arr)} tokens)")


def extract_complete(m: dict, out_root: Path) -> bool:
    acts = out_root / slug(m["id"]) / "acts"
    return all((acts / f"EXTRACT_shard{s:02d}.DONE").exists() for s in range(EXTRACT_SHARDS))


def load_layer(m: dict, li: int, out_root: Path):
    """Concat the 16 shard files for layer li -> (X, y, eids), sorted by eid."""
    acts = out_root / slug(m["id"]) / "acts"
    files = sorted(glob.glob(str(acts / f"layer{li:02d}_shard*.npz")))
    Xs, ys, es = [], [], []
    for f in files:
        d = np.load(f)
        Xs.append(d["X"]); ys.append(d["y"]); es.append(d["example_ids"])
    X = np.concatenate(Xs); y = np.concatenate(ys); eids = np.concatenate(es)
    order = np.argsort(eids, kind="stable")
    return X[order], y[order].astype(np.int8), eids[order].astype(np.int32)


def load_offsets_all(m: dict, out_root: Path) -> dict:
    acts = out_root / slug(m["id"]) / "acts"
    merged = {}
    for f in sorted(glob.glob(str(acts / "offsets_shard*.npz"))):
        merged.update(load_offsets_npz(f))
    return merged


# ---- fit/val/test carve (verbatim exp-12) -------------------------------------

def carve(te_mod, dataset: Path, split: Path):
    rows, train_eids, test_eids = te_mod.load_or_make_split(dataset, split)
    tr_eid_to_group = {e: te_mod.pair_group_key(rows[e]) for e in train_eids}
    tr_groups = sorted(set(tr_eid_to_group.values()))
    rng = np.random.default_rng(VAL_SEED)
    rng.shuffle(tr_groups)
    n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
    val_groups = set(tr_groups[:n_val])
    val_eids = {e for e, g in tr_eid_to_group.items() if g in val_groups}
    fit_eids = train_eids - val_eids
    return rows, fit_eids, val_eids, test_eids


def _factory(head: str):
    H = int(head[3:])
    return lambda d, H=H: MLPProbe(d, hidden=H)


# ---- stage 2: sweep -----------------------------------------------------------

def sweep_unit(m: dict, li: int, head: str, dataset: Path, split: Path,
               out_root: Path, log) -> None:
    import torch
    base = out_root / slug(m["id"]) / f"sweep_{head}"
    base.mkdir(parents=True, exist_ok=True)
    dst = base / f"layer{li:02d}.json"
    if dst.exists():
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    te_mod = _load_train_eval()
    rows, fit_eids, val_eids, test_eids = carve(te_mod, dataset, split)
    offsets_by_eid = load_offsets_all(m, out_root)
    rows_by_eid = load_dataset_rows(dataset)

    X, y, eids = load_layer(m, li, out_root)
    fit = np.fromiter((int(e) in fit_eids for e in eids), bool, len(eids))
    val = np.fromiter((int(e) in val_eids for e in eids), bool, len(eids))
    te = np.fromiter((int(e) in test_eids for e in eids), bool, len(eids))
    if len(np.unique(y[fit])) < 2 or val.sum() == 0 or te.sum() == 0:
        dst.write_text(json.dumps({"layer": li, "head": head, "skipped": "degenerate"}))
        return
    r = train_one_layer(X[fit], y[fit], eids[fit], epochs=EPOCHS, device=device,
                        verbose=False, probe_factory=_factory(head))
    probe = r["probe"].to(device).eval()

    def score(mask):
        Xs = torch.from_numpy(np.ascontiguousarray(X[mask])).to(device)
        with torch.no_grad():
            return torch.sigmoid(probe(Xs)).detach().cpu().numpy()

    val_h = honest_token_aucs(score(val), y[val], eids[val], offsets_by_eid, rows_by_eid)
    test_h = honest_token_aucs(score(te), y[te], eids[te], offsets_by_eid, rows_by_eid)
    dst.write_text(json.dumps({
        "layer": li, "head": head,
        "val_tokens_code_auc": val_h["tokens_code_auc"],
        "tokens_code_auc": test_h["tokens_code_auc"],
        "tokens_auc": test_h["tokens_auc"],
        "dropped_fraction": test_h["dropped_fraction"],
        "val_ex_auc": float(r["ex_auc"])}))
    log(f"sweep {m['id']} L{li:02d} {head}: val_tc={val_h['tokens_code_auc']:.3f} "
        f"test_tc={test_h['tokens_code_auc']:.3f}")


def sweep_complete(m: dict, head: str, out_root: Path) -> bool:
    base = out_root / slug(m["id"]) / f"sweep_{head}"
    return all((base / f"layer{li:02d}.json").exists() for li in model_layers(m))


def best_layer(m: dict, head: str, out_root: Path) -> int:
    if m["mode"] == "fixed":
        return m["layers"][0]
    base = out_root / slug(m["id"]) / f"sweep_{head}"
    layers = model_layers(m)
    cells = [(li, json.loads((base / f"layer{li:02d}.json").read_text())) for li in layers]
    # Prefer the exp-12 selection signal; fall back through alternatives so a few
    # degenerate (nan) layers never blank out the choice. Last resort: middle layer.
    for key in ("val_tokens_code_auc", "tokens_code_auc", "tokens_auc"):
        valid = [(li, d.get(key)) for li, d in cells
                 if isinstance(d.get(key), (int, float)) and d.get(key) == d.get(key)]
        if valid:
            return max(valid, key=lambda t: t[1])[0]
    return layers[len(layers) // 2]


# ---- stage 3: dump ------------------------------------------------------------

def dump_unit(m: dict, head: str, dataset: Path, split: Path, out_root: Path, log) -> None:
    import torch
    out = out_root / slug(m["id"]) / f"dump_{head}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        return
    li = best_layer(m, head, out_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    te_mod = _load_train_eval()
    rows, fit_eids, val_eids, test_eids = carve(te_mod, dataset, split)
    test_set = set(int(e) for e in test_eids)
    offsets_by_eid = load_offsets_all(m, out_root)
    rows_by_eid = load_dataset_rows(dataset)

    X, y, eids = load_layer(m, li, out_root)
    fit = np.fromiter((int(e) in fit_eids for e in eids), bool, len(eids))
    r = train_one_layer(X[fit], y[fit], eids[fit], epochs=EPOCHS, device=device,
                        verbose=False, probe_factory=_factory(head))
    probe = r["probe"].to(device).eval()
    with torch.no_grad():
        logit = probe(torch.from_numpy(np.ascontiguousarray(X)).to(device)).detach().cpu().numpy().astype(np.float32)
    prob = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)

    # per-token char offsets + live-code mask, aligned to the eid-sorted token axis
    char_start = np.empty(len(eids), np.int32)
    char_end = np.empty(len(eids), np.int32)
    is_code = np.zeros(len(eids), bool)
    is_test = np.fromiter((int(e) in test_set for e in eids), bool, len(eids))
    cur = 0
    for eid in np.unique(eids):
        n_tok = int((eids == eid).sum())
        o = offsets_by_eid[int(eid)]
        if o.shape[0] != n_tok:
            raise SystemExit(f"{m['id']} L{li}: offset/token mismatch eid {eid}: {o.shape[0]} vs {n_tok}")
        char_start[cur:cur + n_tok] = o[:, 0]
        char_end[cur:cur + n_tok] = o[:, 1]
        mask = code_only_mask(rows[int(eid)].get("code", ""), rows[int(eid)].get("lang", "") or "", o)
        is_code[cur:cur + n_tok] = mask.astype(bool)
        cur += n_tok

    te = is_test
    def auc(yy, ss):
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(yy, ss)) if len(np.unique(yy)) > 1 else float("nan")
    ex_ids, ex_p = te_mod.example_scores(prob[te], eids[te])
    ex_y = np.array([int(y[eids == e].max() > 0) for e in ex_ids])
    # Gate metric via the SAME honest mask exp-12/the sweep use (not the inline
    # code_only_mask), so test_tokens_code_auc reproduces exp-12 exactly.
    test_h = honest_token_aucs(prob[te], y[te], eids[te], offsets_by_eid, rows_by_eid)

    np.savez_compressed(out / "logits_mlp.npz",
                        logit=logit, prob=prob, y=y, example_id=eids,
                        char_start=char_start, char_end=char_end,
                        is_test=is_test, is_code=is_code,
                        layer=np.int32(li), head=np.array(head))
    all_ids, all_p = te_mod.example_scores(prob, eids)
    ex_records = [{"eid": int(e), "score": float(p),
                   "logit_max": float(logit[eids == e].max()),
                   "label": int(np.asarray(y[eids == e]).max() > 0),
                   "cwe": rows[int(e)].get("cwe"), "lang": rows[int(e)].get("lang"),
                   "is_test": int(e) in test_set}
                  for e, p in zip(all_ids.tolist(), all_p.tolist())]
    (out / "example_scores_mlp.json").write_text(json.dumps(ex_records))

    test_tc = test_h["tokens_code_auc"]
    hist = HIST_MLP.get(m["id"], {}).get(head)
    gate = None
    if hist is not None:
        gate = {"hist_tokens_code_auc": hist, "this_tokens_code_auc": test_tc,
                "delta": round(test_tc - hist, 4), "ok_within_0.02": bool(abs(test_tc - hist) <= 0.02)}
    summary = {"model": m["id"], "head": head, "layer": int(li), "mode": m["mode"],
               "n_tokens": int(len(eids)), "n_tokens_code": int(is_code.sum()),
               "test_tokens_auc": test_h["tokens_auc"],
               "test_tokens_code_auc": test_tc,
               "test_example_auc": auc(ex_y, ex_p),
               "val_ex_auc": float(r["ex_auc"]),
               "n_test_examples": int(len(ex_ids)), "hist_gate": gate}
    (out / "metrics_mlp.json").write_text(json.dumps(summary, indent=2))
    (out / "DONE").write_text("")
    g = f" [gate hist={hist:.3f} d={gate['delta']:+.3f} ok={gate['ok_within_0.02']}]" if gate else ""
    log(f"dump {m['id']} {head} L{li}: test_tc={test_tc:.3f} ex_auc={summary['test_example_auc']:.3f}{g}")


# ---- worker -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=int, default=1200, help="seconds before clean exit")
    ap.add_argument("--only-model", default=None, help="restrict to one model id (preflight)")
    args = ap.parse_args()

    rank, world = args.rank, args.world_size
    out_root = Path(args.out)
    dataset, split = Path(args.dataset), Path(args.split)
    deadline = time.time() + args.time_budget

    def log(msg):
        print(f"[mlp-pipe r{rank}/{world}] {msg}", file=sys.stderr, flush=True)

    models = MODELS if not args.only_model else [m for m in MODELS if m["id"] == args.only_model]

    # static unit lists (global, identical on every worker)
    extract_units = [(mi, s) for mi, _ in enumerate(models) for s in range(EXTRACT_SHARDS)]
    sweep_units = [(mi, li, h) for mi, m in enumerate(models) if m["mode"] == "sweep"
                   for li in model_layers(m) for h in HEADS]
    dump_units = [(mi, h) for mi, _ in enumerate(models) for h in HEADS]

    def mine(units, idx):
        return idx % world == rank

    while time.time() < deadline:
        progressed = False
        # stage 1: extraction
        for idx, (mi, s) in enumerate(extract_units):
            if not mine(extract_units, idx) or time.time() >= deadline:
                continue
            acts_done = (out_root / slug(models[mi]["id"]) / "acts" / f"EXTRACT_shard{s:02d}.DONE")
            if acts_done.exists():
                continue
            extract_shard(models[mi], s, dataset, out_root, log); progressed = True
        # stage 2: sweep (deps: model fully extracted)
        for idx, (mi, li, h) in enumerate(sweep_units):
            if not mine(sweep_units, idx) or time.time() >= deadline:
                continue
            if (out_root / slug(models[mi]["id"]) / f"sweep_{h}" / f"layer{li:02d}.json").exists():
                continue
            if not extract_complete(models[mi], out_root):
                continue
            sweep_unit(models[mi], li, h, dataset, split, out_root, log); progressed = True
        # stage 3: dump (deps: fixed->extract; sweep->all layers swept)
        for idx, (mi, h) in enumerate(dump_units):
            if not mine(dump_units, idx) or time.time() >= deadline:
                continue
            m = models[mi]
            if (out_root / slug(m["id"]) / f"dump_{h}" / "DONE").exists():
                continue
            ready = extract_complete(m, out_root) and (m["mode"] == "fixed" or sweep_complete(m, h, out_root))
            if not ready:
                continue
            dump_unit(m, h, dataset, split, out_root, log); progressed = True

        # all my units done?
        all_done = (
            all((out_root / slug(models[mi]["id"]) / "acts" / f"EXTRACT_shard{s:02d}.DONE").exists()
                for idx, (mi, s) in enumerate(extract_units) if mine(extract_units, idx))
            and all((out_root / slug(models[mi]["id"]) / f"sweep_{h}" / f"layer{li:02d}.json").exists()
                    for idx, (mi, li, h) in enumerate(sweep_units) if mine(sweep_units, idx))
            and all((out_root / slug(models[mi]["id"]) / f"dump_{h}" / "DONE").exists()
                    for idx, (mi, h) in enumerate(dump_units) if mine(dump_units, idx)))
        if all_done:
            log("all my units done"); break
        if not progressed:
            time.sleep(20)  # wait for peers' dependencies to land


if __name__ == "__main__":
    main()
