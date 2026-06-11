# [ai-generated]
"""Exp-10: POST-HOC ENSEMBLE of interpretable specialist probes vs the opaque MLP.

Question (the lead's): a single linear probe is beaten by an MLP on some models
(e.g. gemma-3-27b-it: linear 0.770 vs MLP 0.814 honest `tokens_code`, a +0.044
nonlinear gap). Per-CWE / per-family SPECIALIZED linear probes each strongly beat
the GENERAL probe on their own slice (e.g. UAF 0.43->0.77). Does combining the
specialists' per-token scores POST-HOC give a SINGLE interpretable detector that
approaches the MLP? And do MLP specialists ensembled go further?

This runner, per model at its single best layer on CACHED acts, with the EXACT
exp-10 splits (test hold-out + 15% VAL carve, group-aware, VAL_SEED=42):

  1. Trains a SPECIALIST SET (`--spec-head {linear,mlp256,mlp512}`):
       - individual per-CWE specialists: for each CWE with >= --min-train-pos
         (default 10) fit positives, a probe on
         {CWE-X fit positives} u {ALL fit negatives}   (neg_pool == "all").
       - family specialists: ONE pooled `memory` probe + ONE pooled `injection`
         probe (all family fit positives u all fit negatives).
     `--spec-set {family,individual,both}` chooses which subset is ensembled.
  2. Scores every specialist on the FULL test token set AND the FULL val token
     set -> aligned per-token sigmoid vectors, stacked (in a FIXED specialist
     order) into [n_tokens x n_specialists] matrices S_test, S_val.
  3. Combines the specialist columns into ONE per-token score, three ways:
       max     = rowwise max over specialists (union: any specialist fires)
       mean    = rowwise mean
       learned = sklearn LogisticRegression fit on S_val -> val token labels
                 y[val], then predict_proba(S_test)[:,1]. HONEST: fit on VAL,
                 eval on TEST (val/test are group-disjoint). Torch fallback if
                 sklearn LR is unavailable.
  4. Reference points (same run, same split): GENERAL LINEAR probe (the floor)
     and a GENERAL MLP (`--ref-mlp`, default mlp512 — the MLP ceiling).
  5. Eval: overall honest `tokens_code` for {general_linear, general_mlp,
     ens_max, ens_mean, ens_learned}, plus per-language (python/c/cpp) overall
     `tokens_code` for each combiner (cheap, via a test subset mask).

HONEST-EVAL contract (mirrors per_cwe_probe.py exactly):
  - Specialists trained ONLY on fit_eids (train minus the 15% VAL carve).
  - The learned combiner is fit ONLY on val tokens.
  - Everything is evaluated ONLY on test tokens.
  - All specialist score vectors are aligned to the SAME ordered token set: the
    test/val boolean masks are built ONCE over the stored `example_ids` order,
    and each specialist is scored on those exact masks, so column i of S_test and
    column i of S_val are the SAME specialist over the SAME tokens.

Output: ONE JSON per (model x spec_head x spec_set) at --out, resumable (skip if
the file exists). Carries: model, layer, spec_head, spec_set, n_specialists, the
specialist names + their SOLO overall `tokens_code`, the reference
{general_linear, general_mlp} overall, and for each combiner {max,mean,learned}
the overall + by_lang `tokens_code`.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.training.train_probe_spanmax import MLPProbe, train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    honest_token_aucs, load_dataset_rows, load_offsets_npz,
)

# Parity with train_all_layers.py / per_cwe_probe.py: 15% group-aware VAL carve
# of TRAIN, seed 42. The learned combiner is fit on these val tokens; specialists
# never see them.
VAL_FRAC = 0.15
VAL_SEED = 42

# CWE -> family (verbatim from per_cwe_probe.py; keep the two in sync).
# TODO(adhoc-decision): CWE-190 family is debatable (per_cwe_probe.py docstring);
# this map is the only place to change it.
FAMILY = {
    "CWE-089": "injection", "CWE-078": "injection", "CWE-022": "injection",
    "CWE-079": "injection", "CWE-190": "injection",
    "CWE-125": "memory", "CWE-476": "memory", "CWE-416": "memory",
    "CWE-787": "memory",
}
FAMILIES = ("memory", "injection")


def _load_train_eval():
    p = REPO / "src" / "remotes" / "train_eval.py"
    spec = importlib.util.spec_from_file_location("remote_train_eval", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _factory_for(head: str):
    """head -> probe_factory for train_one_layer. None == LinearProbe (default).
    'mlp256','mlp512' -> MLPProbe(hidden=H). Mirrors train_head_baseline._factory_for."""
    if head == "linear":
        return None
    m = re.fullmatch(r"mlp(\d+)", head)
    if not m:
        raise ValueError(f"bad head {head!r} (expected linear|mlpH)")
    H = int(m.group(1))
    return lambda d, H=H: MLPProbe(d, hidden=H)


def _fit_probe(Xmm, fit_mask, y, eids, *, epochs, head, device):
    """Fit one span-max probe on the tokens selected by fit_mask. Returns the
    train_one_layer result dict. Inputs are never mutated (a float32 fit copy is
    built)."""
    Xfit = np.asarray(Xmm[fit_mask], np.float32)
    return train_one_layer(
        Xfit, y[fit_mask], eids[fit_mask],
        epochs=epochs, device=device, verbose=False,
        probe_factory=_factory_for(head),
    )


def _score_tokens(probe_result, X, device) -> np.ndarray:
    """Per-token sigmoid prob from a train_one_layer result on activations X.

    Linear heads expose (w, b); non-linear heads expose the torch module under
    "probe" with w=None. Score uniformly via the module (works for both), with a
    cheap explicit (w, b) fast path for the linear case. Never mutates X."""
    import torch
    if probe_result.get("w") is not None:
        w = np.asarray(probe_result["w"], np.float32)
        b = float(probe_result["b"])
        return 1.0 / (1.0 + np.exp(-(np.asarray(X, np.float32) @ w + b)))
    probe = probe_result["probe"].to(device).eval()
    with torch.no_grad():
        logits = probe(torch.from_numpy(np.asarray(X, np.float32)).to(device))
        return torch.sigmoid(logits).detach().cpu().numpy()


def combine(S: np.ndarray, how: str, *, clf=None) -> np.ndarray:
    """Combine an [n_tokens x n_specialists] specialist-score matrix into one
    per-token score vector (length n_tokens).

      max  -> rowwise max  (union: any specialist fires)
      mean -> rowwise mean
      learned -> clf.predict_proba(S)[:,1] for an already-fit combiner `clf`.

    `clf` is required only for how == "learned"; it is fit elsewhere (on VAL).
    Never mutates S."""
    S = np.asarray(S, np.float32)
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D [n_tokens x n_specialists], got {S.shape}")
    if how == "max":
        return S.max(axis=1)
    if how == "mean":
        return S.mean(axis=1)
    if how == "learned":
        if clf is None:
            raise ValueError("combine(how='learned') requires a fit clf")
        return clf.predict_proba(S)[:, 1]
    raise ValueError(f"unknown combiner {how!r}")


class _TorchLogisticCombiner:
    """Tiny torch logistic-regression fallback when sklearn LR is unavailable.

    Exposes the sklearn `.predict_proba(S)[:,1]` interface used by `combine`.
    Fit on VAL only; never on TEST."""
    def __init__(self, n_features: int, device: str, *, epochs: int = 300, lr: float = 0.05):
        import torch
        self.device = device
        self.lin = torch.nn.Linear(n_features, 1).to(device)
        self.epochs = epochs
        self.lr = lr

    def fit(self, S: np.ndarray, y: np.ndarray) -> "_TorchLogisticCombiner":
        import torch
        Xt = torch.from_numpy(np.asarray(S, np.float32)).to(self.device)
        yt = torch.from_numpy(np.asarray(y, np.float32)).to(self.device)
        opt = torch.optim.Adam(self.lin.parameters(), lr=self.lr, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss()
        self.lin.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logit = self.lin(Xt).squeeze(-1)
            loss = lossf(logit, yt)
            loss.backward()
            opt.step()
        self.lin.eval()
        return self

    def predict_proba(self, S: np.ndarray) -> np.ndarray:
        import torch
        Xt = torch.from_numpy(np.asarray(S, np.float32)).to(self.device)
        with torch.no_grad():
            p1 = torch.sigmoid(self.lin(Xt).squeeze(-1)).detach().cpu().numpy()
        return np.stack([1.0 - p1, p1], axis=1)


def _fit_learned_combiner(S_val: np.ndarray, y_val: np.ndarray, device: str):
    """Fit a logistic-regression combiner on VAL specialist scores -> val labels.

    Tries sklearn LogisticRegression; falls back to a tiny torch logistic layer.
    Returns (clf, backend_name). If val is single-class, returns (None, reason) —
    the learned combiner is then skipped and reported as NaN."""
    y_val = np.asarray(y_val)
    if len(np.unique(y_val)) < 2:
        return None, "val single-class"
    try:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(np.asarray(S_val, np.float32), y_val)
        return clf, "sklearn"
    except Exception as e:  # noqa: BLE001 — fall back, but record why.
        clf = _TorchLogisticCombiner(S_val.shape[1], device).fit(S_val, y_val)
        return clf, f"torch_fallback({type(e).__name__})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--spec-head", choices=("linear", "mlp256", "mlp512"),
                    default="linear",
                    help="Head for the SPECIALIST probes. linear is most "
                         "interpretable; mlpH is the 'MLP specialists' arm.")
    ap.add_argument("--spec-set", choices=("family", "individual", "both"),
                    default="both",
                    help="Which specialists to ensemble. family = the 2 pooled "
                         "family probes only (most interpretable); individual = "
                         "per-CWE only; both = union.")
    ap.add_argument("--ref-mlp", choices=("mlp256", "mlp512"), default="mlp512",
                    help="The opaque-MLP ceiling reference (general probe). "
                         "TODO(adhoc-decision): mlp512 mirrors exp-09's default "
                         "ceiling; the lead may prefer mlp256.")
    ap.add_argument("--min-train-pos", type=int, default=10,
                    help="Skip an individual-CWE specialist with fewer fit "
                         "positives (post-VAL-carve). TODO(adhoc-decision): "
                         "default 10 mirrors per_cwe_probe / breakdown.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[posthoc] {out} exists, skipping", file=sys.stderr)
        return

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    te_mod = _load_train_eval()
    acts = Path(args.acts_dir)
    offsets_by_eid = load_offsets_npz(
        Path(args.offsets) if args.offsets else acts / "offsets.npz")
    rows_by_eid = load_dataset_rows(Path(args.dataset))

    y = np.load(acts / "y.npy")
    eids = np.load(acts / "example_ids.npy")
    rows, train_eids, test_eids = te_mod.load_or_make_split(
        Path(args.dataset), Path(args.split))

    # --- group-aware 15% VAL carve of TRAIN (exact parity with per_cwe_probe) ---
    tr_eid_to_group = {e: te_mod.pair_group_key(rows[e]) for e in train_eids}
    tr_groups = sorted(set(tr_eid_to_group.values()))
    rng = np.random.default_rng(VAL_SEED)
    rng.shuffle(tr_groups)
    n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
    val_groups = set(tr_groups[:n_val])
    val_eids = {e for e, g in tr_eid_to_group.items() if g in val_groups}
    fit_eids = train_eids - val_eids  # specialist + general fit pool (eids)

    def cwe_of(e):
        return rows[e].get("cwe")

    def lang_of(e):
        return (rows[e].get("lang") or "").lower()

    neg_fit = {e for e in fit_eids if not cwe_of(e)}  # all negative fit eids

    # --- token masks built ONCE over the stored example_ids order. Every
    # specialist is scored on these SAME masks so the stacked S_* columns align. ---
    n_tok = len(eids)
    fit_mask = np.fromiter((int(e) in fit_eids for e in eids), bool, n_tok)
    val_mask = np.fromiter((int(e) in val_eids for e in eids), bool, n_tok)
    test_mask = np.fromiter((int(e) in test_eids for e in eids), bool, n_tok)
    print(f"[posthoc] {args.model} L{args.best_layer} spec_head={args.spec_head} "
          f"spec_set={args.spec_set} fit_tok={fit_mask.sum()} "
          f"val_tok={val_mask.sum()} test_tok={test_mask.sum()}", file=sys.stderr)

    if test_mask.sum() == 0 or val_mask.sum() == 0 or fit_mask.sum() == 0:
        out.write_text(json.dumps({"model": args.model, "layer": args.best_layer,
                                   "spec_head": args.spec_head, "spec_set": args.spec_set,
                                   "skipped": "degenerate split (empty fit/val/test)"}))
        return

    Xmm = np.load(acts / f"layer_{args.best_layer:02d}.npy", mmap_mode="r")
    # Materialize the test/val activation blocks once (scored repeatedly).
    Xte = np.asarray(Xmm[test_mask], np.float32)
    Xval = np.asarray(Xmm[val_mask], np.float32)
    if not (np.isfinite(Xte).all() and np.isfinite(Xval).all()):
        out.write_text(json.dumps({"model": args.model, "layer": args.best_layer,
                                   "spec_head": args.spec_head, "spec_set": args.spec_set,
                                   "error": "non-finite activations"}))
        return

    y_te, e_te = y[test_mask], eids[test_mask]
    y_val = y[val_mask]

    def honest_tc(token_probs, mask_e, mask_y, sub=None):
        """Overall (or subset) honest tokens_code_auc for a per-token score vector
        aligned to the test (or val) token order. `sub` is an optional boolean
        mask over those tokens (for per-language)."""
        p, ee, yy = token_probs, mask_e, mask_y
        if sub is not None:
            p, ee, yy = p[sub], ee[sub], yy[sub]
        if len(np.unique(yy)) < 2:
            return float("nan")
        h = honest_token_aucs(p, yy, ee, offsets_by_eid, rows_by_eid)
        return h["tokens_code_auc"]

    # ---------- specialist set ----------
    # Build the ordered specialist list FIRST (fixed order -> fixed columns).
    specs: list[tuple[str, set]] = []  # (name, fit_eids_for_this_specialist)

    want_family = args.spec_set in ("family", "both")
    want_individual = args.spec_set in ("individual", "both")

    if want_family:
        for fam in FAMILIES:  # fixed order: memory, injection
            pos_fit = {e for e in fit_eids if FAMILY.get(cwe_of(e)) == fam}
            if pos_fit and neg_fit:
                specs.append((f"family:{fam}", pos_fit | neg_fit))
            else:
                print(f"[posthoc] family {fam}: empty pos/neg fit pool -> skip",
                      file=sys.stderr)

    if want_individual:
        train_pos_counts = Counter(cwe_of(e) for e in fit_eids if cwe_of(e))
        # most_common -> deterministic order (count desc, then insertion); fix
        # ties by sorting on (-count, cwe) for stable reproducibility.
        ordered = sorted(train_pos_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        for cwe, n in ordered:
            if n < args.min_train_pos:
                continue
            pos_fit = {e for e in fit_eids if cwe_of(e) == cwe}
            if pos_fit and neg_fit:
                specs.append((f"cwe:{cwe}", pos_fit | neg_fit))

    if not specs:
        out.write_text(json.dumps({"model": args.model, "layer": args.best_layer,
                                   "spec_head": args.spec_head, "spec_set": args.spec_set,
                                   "error": "no specialists met --min-train-pos"}))
        return

    spec_names = [name for name, _ in specs]
    print(f"[posthoc] {len(specs)} specialists: {spec_names}", file=sys.stderr)

    # ---------- train specialists, score on test + val (aligned columns) ----------
    S_test_cols, S_val_cols = [], []
    spec_solo = {}  # name -> solo overall tokens_code on test
    for name, spec_fit_eids in specs:
        sfm = np.fromiter((int(e) in spec_fit_eids for e in eids), bool, n_tok)
        res = _fit_probe(Xmm, sfm, y, eids, epochs=args.epochs,
                         head=args.spec_head, device=device)
        p_te = _score_tokens(res, Xte, device)
        p_val = _score_tokens(res, Xval, device)
        S_test_cols.append(p_te)
        S_val_cols.append(p_val)
        spec_solo[name] = honest_tc(p_te, e_te, y_te)
        solo = spec_solo[name]
        print(f"[posthoc]   {name} solo tc={solo:.3f}" if solo == solo
              else f"[posthoc]   {name} solo tc=nan", file=sys.stderr)

    S_test = np.stack(S_test_cols, axis=1)  # [n_test_tok x n_spec]
    S_val = np.stack(S_val_cols, axis=1)    # [n_val_tok  x n_spec]

    # ---------- reference points: general LINEAR (floor) + general MLP (ceiling) ----------
    gen_lin_res = _fit_probe(Xmm, fit_mask, y, eids, epochs=args.epochs,
                             head="linear", device=device)
    gen_lin_p = _score_tokens(gen_lin_res, Xte, device)
    general_linear = honest_tc(gen_lin_p, e_te, y_te)

    gen_mlp_res = _fit_probe(Xmm, fit_mask, y, eids, epochs=args.epochs,
                             head=args.ref_mlp, device=device)
    gen_mlp_p = _score_tokens(gen_mlp_res, Xte, device)
    general_mlp = honest_tc(gen_mlp_p, e_te, y_te)

    # ---------- learned combiner: fit on VAL only ----------
    clf, clf_backend = _fit_learned_combiner(S_val, y_val, device)

    # ---------- per-language test masks (over the test token order) ----------
    lang_te = np.array([lang_of(int(e)) for e in e_te])
    lang_masks = {lang: (lang_te == lang) for lang in ("python", "c", "cpp")}

    combiners = {}
    for how in ("max", "mean", "learned"):
        if how == "learned" and clf is None:
            combiners[how] = {"overall": float("nan"), "by_lang": {},
                              "note": f"learned skipped: {clf_backend}"}
            continue
        comb_te = combine(S_test, how, clf=clf if how == "learned" else None)
        overall = honest_tc(comb_te, e_te, y_te)
        by_lang = {}
        for lang, lm in lang_masks.items():
            if lm.sum() == 0:
                continue
            by_lang[lang] = honest_tc(comb_te, e_te, y_te, sub=lm)
        cell = {"overall": overall, "by_lang": by_lang}
        if how == "learned":
            cell["backend"] = clf_backend
        combiners[how] = cell
        ov = f"{overall:.3f}" if overall == overall else "nan"
        print(f"[posthoc]   ens_{how} overall tc={ov}", file=sys.stderr)

    rec = {
        "model": args.model,
        "layer": args.best_layer,
        "spec_head": args.spec_head,
        "spec_set": args.spec_set,
        "ref_mlp": args.ref_mlp,
        "min_train_pos": args.min_train_pos,
        "n_specialists": len(specs),
        "specialists": spec_names,
        "specialist_solo_tokens_code": spec_solo,
        "reference": {
            "general_linear": general_linear,
            "general_mlp": general_mlp,
        },
        "combiners": combiners,
        "n_test_tokens": int(test_mask.sum()),
        "n_val_tokens": int(val_mask.sum()),
        "n_fit_tokens": int(fit_mask.sum()),
    }
    out.write_text(json.dumps(rec, indent=2))
    gl = f"{general_linear:.3f}" if general_linear == general_linear else "nan"
    gm = f"{general_mlp:.3f}" if general_mlp == general_mlp else "nan"
    print(f"[posthoc] wrote {out}  general_linear={gl} general_mlp={gm} "
          f"n_spec={len(specs)}", file=sys.stderr)


if __name__ == "__main__":
    main()
