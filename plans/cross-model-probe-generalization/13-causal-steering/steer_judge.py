# [ai-generated]
"""Exp-13 Tier-1 CAUSAL STEERING (v2): does ADDING a direction to the residual
stream raise the model's VERBALIZED P("yes, vulnerable")? — and is the effect
SPECIFIC to the memory-family probe direction, or a generic yes-bias?

The belief audit (exp-05) asks whether the model VERBALIZES memory-vuln. This
experiment asks the CAUSAL question: if we steer the residual stream along a
unit direction `d_hat` at the best layer, does the model's stated P(yes) move?
A monotone rise on +alpha that is STRONGER for the memory direction than for a
real control direction (injection) and for random directions is evidence the
memory-family direction CAUSALLY and SPECIFICALLY drives the stated belief — a
MoC-style linear correction (Marks & Tegmark; ITI, Li et al. 2023; activation
steering, Turner et al. 2023) — rather than an epiphenomenal correlate or a
generic "say yes" axis.

=== WHAT CHANGED FROM v1 (the lead's two fixes) =====================
1. MAGNITUDE (A). v1 scaled by `median |h|_2` (the activation L2 norm). For
   Gemma-3 that is HUGE (~8367; massive mid-layer activations concentrated in a
   few outlier dims that are largely ORTHOGONAL to the probe direction), so
   |alpha|>=0.5 DESTROYED the model and the effect looked global. v2 scales by
   the PROJECTION-STD: `scale_d = std over code tokens of (h . d_hat)`, computed
   per direction from the CACHED acts at the best layer. A steer of
   `alpha * scale_d * d_hat` therefore moves the residual `alpha` STANDARD
   DEVIATIONS ALONG that direction — interpretable, model-agnostic, and immune
   to the orthogonal outlier-norm blow-up. `scale_def = "proj_std"`.
2. SPECIFICITY CONTROL (B). v1 steered ONLY the memory direction, so a global
   rise (P(yes) up on negatives too) could not be told apart from a
   memory-specific effect. v2 steers FOUR directions, each scaled by ITS OWN
   proj-std (a fair `alpha`-std push along each):
     - memory    : unit-normed pooled MEMORY-family probe weight   (the TEST direction)
     - injection : unit-normed pooled INJECTION-family probe weight (a REAL control)
     - random_0  : a random unit direction (np.random.default_rng(0))  (RANDOM control)
     - random_1  : a random unit direction (np.random.default_rng(1))  (RANDOM control)
   Memory-specific causation => memory rises MORE than injection (different real
   axis) and MUCH more than random.

ALSO in v2:
- ALPHA GRID in std units (C): default {-4,-2,-1,0,1,2,4} (--alphas). 0 is
  MANDATORY (it gates the self-check); the runner refuses a grid without it.
- DEGRADATION GUARD (D): at each (direction, alpha), if the steered first-token
  distribution no longer answers yes/no — yes+no probability mass
  P(yes)+P(no) < DEGRADE_THRESH (0.05) — flag `degraded: true` for that cell.
  P(yes) is still recorded; analysis EXCLUDES degraded cells. This catches the
  v1 Gemma blow-up without crashing.
- INCREMENTAL OUTPUT (E): v1 wrote one JSON at the very end, so a 22-min timeout
  lost everything. v2 writes the output JSON after EACH direction's sweep
  completes and SKIPS a direction already present on a re-run — resumable
  per-direction.

PIPELINE (one model, one node, 1 GPU is enough):
  1. DIRECTIONS. Fit the pooled MEMORY-family and INJECTION-family linear probes
     at the model's best layer on cached acts (reuse exp-10's FAMILY map + the
     pooled-family fit recipe via train_one_layer). Unit-normalize each head
     weight -> d_hat. Plus two random unit directions (fixed seeds 0, 1).
  2. SCALE (per direction). scale_d = std over CODE tokens of (h . d_hat) at
     layer L, from cached acts (== hidden_states[L+1], the probe's training
     tensor). Code tokens via the offsets.npz live-code mask (keep-all fallback).
  3. HOOK. Forward hook on decoder layer L adds `alpha * scale_d * d_hat` to that
     layer's residual-stream OUTPUT (output[0]) at ALL token positions. The
     cached acts are hidden_states[L+1] = the OUTPUT of model...layers[L], so the
     hook perturbs EXACTLY the tensor the probe trained on. Module path resolved
     by `_resolve_decoder_layers` mirroring _load_model's walk.
  4. SWEEP alpha in std units. alpha=0 MUST reproduce the unsteered P(yes)
     EXACTLY (the idle hook adds 0 -> identity). ASSERTED with a per-example
     abs-diff < 1e-4 self-check vs a no-hook baseline forward; ABORT on failure.
  5. MEASURE. Reuse verbalized_judge.py's prompt construction (apply_chat_template
     add_generation_prompt=True, enable_thinking=False for Qwen3 with the
     abort-guard) + p_yes_from_logits. On a SUBSET (cheap): up to N memory-family
     positives, N injection-family positives, N negatives (cwe==null), from the
     leakage-free TEST split. Per (direction, alpha, subset), record mean P(yes)
     and a per-cell degraded flag.

OUTPUT JSON (incremental; see schema in main()):
  {model, layer, scale_def:"proj_std", alpha_grid, directions:{<dir>:{scale,...}},
   by_direction:{<dir>:{by_subset:{<sub>:{p_yes:[...], degraded:[...]}}, n_per_subset}},
   selfcheck:{ok, max_abs_diff, ...}}
plus each direction at <out>.<dir>.dir.pt.

REUSE: verbalized_judge.py (model load via _load_model, render_chat with the
Qwen3 thinking guard, p_yes_from_logits, resolve_yes_no_ids, QUESTION,
build_content); per_cwe_probe.py (FAMILY map, pooled-family fit); train_eval.py
(load_or_make_split, pair_group_key) for the leakage-free TEST split;
honest_scoring.py (build_code_mask / load_offsets_npz / load_dataset_rows) for
the live-code mask used by the proj-std scale.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.data.extract_token_activations import _load_model, _load_tokenizer  # noqa: E402
from src.training.train_probe_spanmax import train_one_layer  # noqa: E402
from src.eval.honest_scoring import (  # noqa: E402
    build_code_mask, load_offsets_npz, load_dataset_rows,
)

# Reuse the FAMILY map + the verbalized-judge prompt/scoring helpers by loading
# those experiment modules by path (they live under plans/, not an importable
# package). Done once at import.
_EXP = REPO / "plans" / "cross-model-probe-generalization"


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vj = _load_by_path(
    "exp05_verbalized_judge", _EXP / "05-probe-vs-verbalized" / "verbalized_judge.py")
_pc = _load_by_path(
    "exp10_per_cwe_probe", _EXP / "10-per-cwe-probes" / "per_cwe_probe.py")
_te = _load_by_path(
    "remote_train_eval", REPO / "src" / "remotes" / "train_eval.py")

FAMILY = _pc.FAMILY                  # CWE -> "injection" | "memory"
build_content = _vj.build_content
render_chat = _vj.render_chat
resolve_yes_no_ids = _vj.resolve_yes_no_ids
p_yes_from_logits = _vj.p_yes_from_logits

# Parity with per_cwe_probe.py: 15% group-aware VAL carve of TRAIN, seed 42.
# We exclude VAL tokens from the direction fit so the steering direction is
# trained on EXACTLY the fit pool exp-10's family probe used (no leakage from
# the held-out selection groups into the direction we then test on).
VAL_FRAC = _pc.VAL_FRAC
VAL_SEED = _pc.VAL_SEED

# alpha grid in PROJECTION-STD units (each step moves the residual `alpha` std
# ALONG the steered direction). TODO(adhoc-decision): the grid is the lead's to
# settle; this is the briefed v2 default {-4,-2,-1,0,1,2,4}. 0 is MANDATORY (it
# gates the self-check) — keep it whatever else changes.
DEFAULT_ALPHAS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
SELFCHECK_TOL = 1e-4  # abs P(yes) diff at alpha=0 vs no-hook baseline

# DEGRADATION GUARD threshold: if the yes+no probability mass at the first token
# drops below this, the model no longer answers yes/no (it's emitting some other
# token — the v1 Gemma blow-up) and the cell is flagged `degraded`. P(yes) is
# still recorded; analysis EXCLUDES degraded cells. TODO(adhoc-decision): 0.05
# is the briefed default; the lead may tighten/loosen it. It is the same for all
# directions/alphas and is logged in the output JSON for provenance.
DEGRADE_THRESH = 0.05

# Random-direction seeds (the RANDOM control). Fixed for reproducibility across
# reruns. np.random is fine on the cluster (no torch RNG-state coupling).
# TODO(adhoc-decision): two random dirs (seeds 0,1) is the briefed default; more
# would tighten the random-control band at linear extra cost.
RANDOM_SEEDS = (0, 1)

# Direction order: memory (TEST) first, then the controls. Sweep + incremental
# write follow this order, so a timeout keeps the most important direction.
DIRECTION_ORDER = ("memory", "injection", "random_0", "random_1")


# ---------------------------------------------------------------------------
# DIRECTIONS
# ---------------------------------------------------------------------------
def _fit_eids(rows, train_eids):
    """The exp-10 fit pool: TRAIN minus the group-aware 15% VAL carve (seed 42).
    Returns the set of fit eids. Inputs never mutated."""
    tr_eid_to_group = {e: _te.pair_group_key(rows[e]) for e in train_eids}
    tr_groups = sorted(set(tr_eid_to_group.values()))
    rng = np.random.default_rng(VAL_SEED)
    rng.shuffle(tr_groups)
    n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
    val_groups = set(tr_groups[:n_val])
    val_eids = {e for e, g in tr_eid_to_group.items() if g in val_groups}
    return train_eids - val_eids


def train_family_direction(family: str, acts_dir: Path, rows, train_eids,
                           best_layer: int, *, epochs: int, device: str):
    """Train the pooled `family` ("memory"|"injection") linear probe at
    `best_layer` on cached acts; return (d_hat unit-normalized float32 (d,),
    raw_w_norm, n_fit_pos, n_fit_neg). Mirrors per_cwe_probe.py's pooled-family
    fit (family positives plus ALL cwe==null negatives, VAL groups excluded).
    Inputs never mutated."""
    fit_eids = _fit_eids(rows, train_eids)

    def cwe_of(e):
        return rows[e].get("cwe")

    # Pooled family: every positive whose CWE maps to `family`, plus ALL
    # cwe==null negatives in the fit pool (matches per_cwe_probe.py --neg-pool all).
    pos_fit = {e for e in fit_eids if FAMILY.get(cwe_of(e)) == family}
    neg_fit = {e for e in fit_eids if not cwe_of(e)}
    spec_fit_eids = pos_fit | neg_fit
    if not pos_fit:
        raise SystemExit(f"[steer] no {family}-family positives in fit pool — "
                         f"cannot train a {family} direction")

    y = np.load(acts_dir / "y.npy")
    eids = np.load(acts_dir / "example_ids.npy")
    Xmm = np.load(acts_dir / f"layer_{best_layer:02d}.npy", mmap_mode="r")

    fit_mask = np.fromiter((int(e) in spec_fit_eids for e in eids), bool, len(eids))
    Xfit = np.asarray(Xmm[fit_mask], np.float32)
    res = train_one_layer(Xfit, y[fit_mask], eids[fit_mask],
                          epochs=epochs, device=device, verbose=False)
    w = np.asarray(res["w"], np.float32).reshape(-1)
    raw_norm = float(np.linalg.norm(w))
    if raw_norm == 0.0:
        raise SystemExit(f"[steer] degenerate {family} probe: ||w|| == 0")
    d_hat = (w / raw_norm).astype(np.float32)
    return d_hat, raw_norm, len(pos_fit), len(neg_fit)


def random_direction(seed: int, dim: int):
    """A random unit direction (np.random.default_rng(seed)). The RANDOM control.
    Deterministic per seed; immutable."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    n = float(np.linalg.norm(v))
    if n == 0.0:  # astronomically unlikely; guard anyway
        raise SystemExit(f"[steer] degenerate random dir (seed={seed}): norm 0")
    return (v / n).astype(np.float32)


def _code_token_mask(acts_dir: Path, dataset: Path, eids: np.ndarray):
    """Live-code boolean mask aligned to the cached-acts token order, via the
    offsets.npz live-code mask. Returns (mask (n,), used_code_mask: bool). Falls
    back to keep-all (all True) if offsets.npz is absent so the proj-std still
    computes (over ALL tokens) rather than crashing — logged either way.

    TODO(adhoc-decision): proj-std is over LIVE-CODE tokens (the briefed
    intent — the same population the honest token-AUC uses, excluding trivial
    comment/scaffolding tokens). On a keep-all fallback it is over ALL cached
    tokens; the chosen population is recorded as `scale_token_pop` per direction
    for provenance."""
    offs_path = acts_dir / "offsets.npz"
    if not offs_path.exists():
        print(f"[steer] WARNING: {offs_path} absent — proj-std over ALL cached "
              "tokens (keep-all), not live-code only.", file=sys.stderr)
        return np.ones(eids.shape[0], dtype=bool), False
    offsets_by_eid = load_offsets_npz(offs_path)
    rows_by_eid = load_dataset_rows(dataset)
    mask = build_code_mask(eids, offsets_by_eid, rows_by_eid)
    return mask, True


def proj_std_scale(acts_dir: Path, best_layer: int, d_hat: np.ndarray,
                   code_mask: np.ndarray) -> float:
    """scale_d = std over CODE tokens of the scalar projection (h . d_hat) at
    layer L, from cached acts (== hidden_states[L+1], the probe's training
    tensor). A steer of `alpha * scale_d * d_hat` moves the residual `alpha`
    standard-deviations ALONG d_hat — interpretable and model-agnostic (immune
    to Gemma's orthogonal outlier-norm blow-up). Read-only mmap; no mutation.

    Streamed in chunks (cached acts can be tens of GB); float64 accumulation of
    the projection, then a single std over the masked projections."""
    Xmm = np.load(acts_dir / f"layer_{best_layer:02d}.npy", mmap_mode="r")
    n = Xmm.shape[0]
    if code_mask.shape[0] != n:
        raise SystemExit(f"[steer] code_mask length {code_mask.shape[0]} != "
                         f"n_tokens {n} at layer {best_layer}")
    d64 = d_hat.astype(np.float64)
    proj = np.empty(n, dtype=np.float64)
    step = 100_000
    for s in range(0, n, step):
        chunk = np.asarray(Xmm[s:s + step], np.float64)
        proj[s:s + step] = chunk @ d64
    sel = proj[code_mask]
    if sel.size < 2:
        raise SystemExit(f"[steer] proj-std needs >=2 code tokens; got {sel.size}")
    return float(np.std(sel))


# ---------------------------------------------------------------------------
# HOOK
# ---------------------------------------------------------------------------
def _resolve_decoder_layers(model):
    """Return the decoder layer ModuleList, mirroring _load_model's inner walk
    (model -> transformer -> language_model, then .layers, with a nested
    .model.layers fallback for VLM text-decoder wrappers).

    For these roster models the residual stream at layer L is output[0] of
    layers[L]; the cached acts (extract_all_layers.py) store hidden_states[L+1],
    which IS that output. So a forward hook on layers[L] that perturbs output[0]
    perturbs exactly the tensor the probe was trained on."""
    inner = model
    for attr in ("model", "transformer", "language_model"):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)
    layers = getattr(inner, "layers", None)
    if layers is None:
        layers = getattr(getattr(inner, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError(
            "[steer] could not resolve decoder layer list; module walk found "
            f"{type(inner).__name__} with no .layers")
    return layers


class _SteerState:
    """Mutable holder for the hook's current delta vector. The hook reads this,
    so we change the additive vector between (direction, alpha) WITHOUT
    re-registering (cheaper, and keeps one hook handle for clean removal). The
    *state object* is mutated by design (a control register); the model tensors
    are not."""

    def __init__(self):
        self.delta = None  # torch tensor (d,) on model device/dtype, or None


def _make_hook(state: _SteerState):
    """Forward hook adding state.delta to the layer OUTPUT[0] residual stream at
    ALL token positions. Returns a NEW output tuple (does not mutate the model's
    tensors in place beyond the standard hook-return contract).

    TODO(adhoc-decision): intervention is ADDITIVE along d_hat at ALL positions
    (briefed default, unchanged from v1). Alternatives the lead may want:
    code-only positions (needs the per-token code mask threaded into the forward
    — not wired here), or projection-removal (subtract the d_hat component then
    add alpha*scale). This is additive + all-positions."""

    def hook(_module, _inputs, output):
        if state.delta is None:
            return output  # alpha==0 fast path is handled by delta=None
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + state.delta.to(dtype=hs.dtype, device=hs.device)
            return (hs,) + tuple(output[1:])
        hs = output
        return hs + state.delta.to(dtype=hs.dtype, device=hs.device)

    return hook


# ---------------------------------------------------------------------------
# MEASURE
# ---------------------------------------------------------------------------
def pick_subsets(rows, test_eids, n_per_subset: int):
    """Pick up to n_per_subset eids per subset from the leakage-free TEST split:
    memory-family positives, injection-family positives, cwe==null negatives.
    Deterministic (sorted eids, no shuffle) so reruns are reproducible."""
    def cwe_of(e):
        return rows[e].get("cwe")

    te_sorted = sorted(test_eids)
    mem = [e for e in te_sorted if FAMILY.get(cwe_of(e)) == "memory"]
    inj = [e for e in te_sorted if FAMILY.get(cwe_of(e)) == "injection"]
    neg = [e for e in te_sorted if not cwe_of(e)]
    return {
        "memory_pos": mem[:n_per_subset],
        "injection_pos": inj[:n_per_subset],
        "negative": neg[:n_per_subset],
    }


def _load_existing(out_path: Path):
    """Load a partial result JSON for per-direction resume, or None if absent /
    unparseable. A corrupt partial is treated as absent (we re-run from scratch
    rather than trust a half-written file)."""
    if not out_path.exists():
        return None
    try:
        return json.loads(out_path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"[steer] WARNING: {out_path} unparseable — re-running from scratch",
              file=sys.stderr)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--acts-dir", required=True,
                    help="layersweep_<slug>/acts (layer_NN.npy + y.npy + "
                         "example_ids.npy [+ offsets.npz] live here)")
    ap.add_argument("--dataset", required=True, help="dataset.jsonl")
    ap.add_argument("--split", required=True, help="sven_split_meta.json")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--n-per-subset", type=int, default=40)
    ap.add_argument("--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS))
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    import torch

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    acts_dir = Path(args.acts_dir)
    dataset = Path(args.dataset)
    split = Path(args.split)
    L = args.best_layer

    alphas = list(args.alphas)
    if 0.0 not in alphas:
        # alpha=0 is the validity gate; refuse to run without it.
        raise SystemExit("[steer] alpha grid MUST contain 0.0 (the self-check "
                         f"gate); got {alphas}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit_device = "cuda" if torch.cuda.is_available() else "cpu"

    rows, train_eids, test_eids = _te.load_or_make_split(dataset, split)

    # --- BUILD DIRECTIONS (cached acts only — no model needed yet) ---
    eids = np.load(acts_dir / "example_ids.npy")
    dim = int(np.load(acts_dir / f"layer_{L:02d}.npy", mmap_mode="r").shape[1])
    code_mask, used_code_mask = _code_token_mask(acts_dir, dataset, eids)
    scale_token_pop = "live_code" if used_code_mask else "all_tokens(keep-all fallback)"
    n_code = int(code_mask.sum())
    print(f"[steer] proj-std token population: {scale_token_pop} "
          f"({n_code}/{code_mask.shape[0]} tokens)", file=sys.stderr)

    # memory + injection real directions (pooled-family probe weights), then 2
    # random unit dirs. Each carries its OWN proj-std scale.
    directions: dict[str, dict] = {}
    dir_vecs: dict[str, np.ndarray] = {}
    for fam in ("memory", "injection"):
        d_hat, raw_norm, n_fit_pos, n_fit_neg = train_family_direction(
            fam, acts_dir, rows, train_eids, L, epochs=args.epochs, device=fit_device)
        scale_d = proj_std_scale(acts_dir, L, d_hat, code_mask)
        dir_vecs[fam] = d_hat
        directions[fam] = {
            "kind": "probe", "family": fam, "scale": scale_d,
            "raw_w_norm": raw_norm, "n_fit_pos": n_fit_pos, "n_fit_neg": n_fit_neg,
            "scale_token_pop": scale_token_pop,
        }
        print(f"[steer] dir={fam}: ||w||={raw_norm:.4f} fit_pos={n_fit_pos} "
              f"fit_neg={n_fit_neg} proj_std(scale)={scale_d:.4f}", file=sys.stderr)
    for seed in RANDOM_SEEDS:
        name = f"random_{seed}"
        d_hat = random_direction(seed, dim)
        scale_d = proj_std_scale(acts_dir, L, d_hat, code_mask)
        dir_vecs[name] = d_hat
        directions[name] = {
            "kind": "random", "seed": seed, "scale": scale_d,
            "scale_token_pop": scale_token_pop,
        }
        print(f"[steer] dir={name}: proj_std(scale)={scale_d:.4f}", file=sys.stderr)

    # Save each direction next to the JSON (provenance + reuse by Tier-2).
    for name, d_hat in dir_vecs.items():
        dir_path = Path(f"{out_path}.{name}.dir.pt")
        meta = {k: v for k, v in directions[name].items()}
        torch.save({"d_hat": torch.from_numpy(d_hat), "layer": L,
                    "model": args.model, "name": name, **meta}, dir_path)

    # --- LOAD MODEL + register hook on layer L ---
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = _load_tokenizer(args.model)
    model = _load_model(args.model, dtype)
    model.to(device).eval()
    layers = _resolve_decoder_layers(model)
    n_layers = len(layers)
    if not (0 <= L < n_layers):
        raise SystemExit(f"[steer] best-layer {L} out of range for "
                         f"{n_layers}-layer decoder")
    target_module = layers[L]
    print(f"[steer] hook target = model...layers[{L}] "
          f"({type(target_module).__name__}); n_layers={n_layers}",
          file=sys.stderr)

    state = _SteerState()
    handle = target_module.register_forward_hook(_make_hook(state))

    yes_ids, no_ids = resolve_yes_no_ids(tokenizer)
    if not yes_ids or not no_ids:
        handle.remove()
        raise SystemExit("[steer] could not resolve yes/no token ids")

    subsets = pick_subsets(rows, test_eids, args.n_per_subset)
    n_per = {k: len(v) for k, v in subsets.items()}
    print(f"[steer] subset sizes (test split): {n_per}", file=sys.stderr)

    # Pre-render each subset eid's input ONCE (the prompt is direction/alpha
    # independent; only the residual-stream perturbation differs). Cache encs to
    # avoid re-tokenizing across the sweep.
    enc_by_eid: dict[int, dict] = {}
    printed_debug = False
    for sub_eids in subsets.values():
        for eid in sub_eids:
            if eid in enc_by_eid:
                continue
            code = rows[eid]["code"]
            code_ids = tokenizer.encode(code, add_special_tokens=False,
                                        truncation=True, max_length=args.max_length)
            code_trunc = tokenizer.decode(code_ids)
            content = build_content(code_trunc)
            enc, used_kwarg = render_chat(tokenizer, content, device, args.model)
            enc_by_eid[eid] = enc
            if not used_kwarg and not printed_debug:
                print("[steer] WARNING: tokenizer rejected enable_thinking=False "
                      "(non-Qwen3 fallback). For Qwen3 render_chat would have "
                      "ABORTED.", file=sys.stderr)
                printed_debug = True

    def forward_pyesno(enc):
        """Return (p_yes, p_yes_plus_no_mass). The mass is the probability the
        first token is ANY yes/no variant — the degradation guard reads it."""
        out = model(**enc, use_cache=False)
        logits_last = out.logits[0, -1, :]
        p_yes = p_yes_from_logits(logits_last, yes_ids, no_ids)
        lp = torch.log_softmax(logits_last.to(torch.float64), dim=-1)
        mass = float(torch.exp(torch.logsumexp(lp[yes_ids + no_ids], dim=-1)).item())
        return p_yes, mass

    # Per-eid alpha=0 P(yes) under the hook (delta=None) for the self-check.
    selfcheck = {"tol": SELFCHECK_TOL, "max_abs_diff": 0.0, "ok": True,
                 "n_checked": 0, "worst": None}

    def make_result(by_direction):
        """Assemble the (possibly partial) result dict for an incremental write.
        Pure — builds a fresh dict from current state; mutates nothing."""
        return {
            "model": args.model,
            "layer": L,
            "scale_def": "proj_std",
            "scale_def_long": ("std over code tokens of (h . d_hat) at layer L "
                               "(== hidden_states[L+1]); a steer of alpha*scale*"
                               "d_hat moves the residual alpha std along d_hat"),
            "scale_token_pop": scale_token_pop,
            "degrade_thresh": DEGRADE_THRESH,
            "intervention": "additive along unit d_hat, ALL token positions, "
                            "on decoder layer L output[0]",
            "alpha_grid": alphas,
            "directions": directions,
            "by_direction": by_direction,
            "n_per_subset": n_per,
            "selfcheck": selfcheck,
            "question": _vj.QUESTION,
        }

    with torch.inference_mode():
        # --- alpha=0 self-check: hook present but delta=None MUST equal a true
        #     no-hook forward to < tol. Run on a small check set (first eid of
        #     each non-empty subset) and ABORT on mismatch. ---
        check_eids = [s[0] for s in subsets.values() if s]
        state.delta = None
        with_hook_idle = {e: forward_pyesno(enc_by_eid[e])[0] for e in check_eids}
        handle.remove()  # truly no hook now
        no_hook = {e: forward_pyesno(enc_by_eid[e])[0] for e in check_eids}
        for e in check_eids:
            d = abs(with_hook_idle[e] - no_hook[e])
            selfcheck["n_checked"] += 1
            if d > selfcheck["max_abs_diff"]:
                selfcheck["max_abs_diff"] = d
                selfcheck["worst"] = {"eid": int(e),
                                      "with_hook_idle": with_hook_idle[e],
                                      "no_hook": no_hook[e]}
        selfcheck["ok"] = selfcheck["max_abs_diff"] < SELFCHECK_TOL
        print(f"[steer] alpha=0 self-check: max_abs_diff="
              f"{selfcheck['max_abs_diff']:.2e} tol={SELFCHECK_TOL} "
              f"ok={selfcheck['ok']}", file=sys.stderr)
        if not selfcheck["ok"]:
            raise SystemExit(
                "[steer] ABORT: alpha=0 self-check FAILED — the idle hook "
                "(delta=None) does not reproduce the no-hook P(yes) within "
                f"{SELFCHECK_TOL}. The hook target/output handling is wrong; "
                "refusing to emit steering scores.")

        # Re-register for the actual sweep (the self-check removed it).
        handle = target_module.register_forward_hook(_make_hook(state))

        # --- resume scaffolding: carry over already-completed directions ---
        existing = _load_existing(out_path)
        by_direction: dict[str, dict] = {}
        if existing and existing.get("by_direction"):
            for k, v in existing["by_direction"].items():
                if k in DIRECTION_ORDER:
                    by_direction[k] = v
            if by_direction:
                print(f"[steer] resume: directions already present "
                      f"{sorted(by_direction)} — skipping them", file=sys.stderr)

        # base steering vectors (scale_d * d_hat) on the model device; per-alpha
        # per-direction delta = alpha * base[dir]. Built once.
        base_vec = {
            name: directions[name]["scale"]
                  * torch.from_numpy(d_hat).to(device=device, dtype=torch.float32)
            for name, d_hat in dir_vecs.items()
        }

        # --- per-direction sweep (incremental write after EACH direction) ---
        for name in DIRECTION_ORDER:
            if name in by_direction:
                continue  # resumed — already on disk
            cell = {sub: {"p_yes": [], "degraded": []} for sub in subsets}
            for alpha in alphas:
                state.delta = None if alpha == 0.0 else (alpha * base_vec[name])
                for sub, sub_eids in subsets.items():
                    if not sub_eids:
                        cell[sub]["p_yes"].append(float("nan"))
                        cell[sub]["degraded"].append(False)
                        continue
                    pys, masses = [], []
                    for e in sub_eids:
                        py, mass = forward_pyesno(enc_by_eid[e])
                        pys.append(py)
                        masses.append(mass)
                    cell[sub]["p_yes"].append(float(np.mean(pys)))
                    # DEGRADATION GUARD: degraded if the MEAN yes+no mass over the
                    # subset has collapsed below threshold (the model stopped
                    # answering yes/no at this (dir, alpha)). P(yes) still recorded.
                    cell[sub]["degraded"].append(bool(np.mean(masses) < DEGRADE_THRESH))
                row = " ".join(
                    f"{s}={cell[s]['p_yes'][-1]:.3f}"
                    f"{'*' if cell[s]['degraded'][-1] else ''}" for s in subsets)
                print(f"[steer] dir={name} alpha={alpha:+.1f}  {row}", file=sys.stderr)
            by_direction[name] = {"by_subset": cell, "n_per_subset": n_per}
            # INCREMENTAL WRITE — persist after this direction so a timeout keeps it.
            out_path.write_text(json.dumps(make_result(by_direction), indent=2))
            print(f"[steer] wrote (incremental, +{name}) {out_path}", file=sys.stderr)

    handle.remove()

    # Final write (idempotent with the last incremental one; ensures complete).
    out_path.write_text(json.dumps(make_result(by_direction), indent=2))
    print(f"[steer] DONE -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
