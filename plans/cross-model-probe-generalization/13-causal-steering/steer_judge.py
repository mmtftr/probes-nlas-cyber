# [ai-generated]
"""Exp-13 Tier-1 CAUSAL STEERING: does ADDING the memory-family probe direction
to the residual stream raise the model's VERBALIZED P("yes, vulnerable")?

The belief audit (exp-05) asks whether the model VERBALIZES memory-vuln. This
experiment asks the CAUSAL question: if we steer the residual stream along the
memory-safety probe direction `w_hat`, does the model's stated P(yes) move? A
monotone rise on +alpha (especially for memory-family positives) is evidence the
probe direction CAUSALLY drives the stated belief — a MoC-style linear
correction (Marks & Tegmark; ITI, Li et al. 2023; activation steering, Turner
et al. 2023) — rather than an epiphenomenal correlate.

PIPELINE (one model, one node, 1 GPU is enough):
  1. DIRECTION. Train the exp-10 POOLED MEMORY-FAMILY linear probe at the model's
     best layer on cached acts (reuse 10-per-cwe-probes/per_cwe_probe.py's FAMILY
     map + pooled-family fit recipe via train_one_layer on the memory subset).
     Take the linear head weight w (d-dim), unit-normalize -> w_hat. Save it.
  2. SCALE. scale = median over tokens of the L2 norm of the layer-L hidden state
     (computed once from cached acts at layer L). This makes alpha interpretable
     across models with very different activation magnitudes. NOTE: Gemma-3 has
     MASSIVE mid-layer activations (>65504, see extract_all_layers.py) — the
     activation-norm scale is what keeps a fixed alpha grid meaningful for it;
     without it alpha=+1.0 would be a rounding error on Gemma's residual stream.
  3. HOOK. Register a forward hook on decoder layer L that ADDS
     alpha * scale * w_hat to that layer's residual-stream OUTPUT (output[0])
     at ALL token positions. The cached acts are hidden_states[L+1] = the OUTPUT
     of model...layers[L] (see extract_all_layers.py line 123: hs[li+1]); the
     probe was trained on exactly that tensor, so steering MUST add to the SAME
     tensor — the layer's output[0], NOT its input. Confirmed module path:
     `model.model.layers[L]` (Qwen2.5 / Qwen3 / Gemma-3 text decoder), resolved
     robustly by `_resolve_decoder_layers` mirroring _load_model's walk.
  4. SWEEP alpha in {-1, -0.5, 0, +0.5, +1} (x scale). alpha=0 MUST reproduce the
     unsteered P(yes) EXACTLY (the hook adds 0 -> identity). We ASSERT this with
     a per-example abs-diff < 1e-4 self-check vs a no-hook baseline forward and
     ABORT on failure (a mismatch means the hook target is wrong).
  5. MEASURE. Reuse verbalized_judge.py's prompt construction (apply_chat_template
     add_generation_prompt=True, enable_thinking=False for Qwen3 with the
     abort-guard) + p_yes_from_logits. On a SUBSET (cheap): up to N memory-family
     positives, N injection-family positives, N negatives (cwe==null), from the
     leakage-free TEST split. Per alpha, record mean P(yes) per subset.

OUTPUT JSON:
  {model, layer, scale, alpha_grid, by_subset: {memory_pos, injection_pos,
   negative} -> [mean P(yes) per alpha], n_per_subset, baseline_pyes_alpha0_selfcheck}
plus the direction at <out>.dir.pt.

REUSE: verbalized_judge.py (model load via _load_model, render_chat with the
Qwen3 thinking guard, p_yes_from_logits, resolve_yes_no_ids, QUESTION,
build_content); per_cwe_probe.py (FAMILY map, pooled-family fit); train_eval.py
(load_or_make_split, pair_group_key) for the leakage-free TEST split.
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
    "remote_train_eval", REPO / "src" / "remotes" / "the cluster" / "train_eval.py")

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

# alpha grid (units of `scale`). TODO(adhoc-decision): the grid is the lead's to
# settle; this is the briefed default {-1,-0.5,0,+0.5,+1}. 0 is MANDATORY (it
# gates the self-check) — keep it whatever else changes.
DEFAULT_ALPHAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
SELFCHECK_TOL = 1e-4  # abs P(yes) diff at alpha=0 vs no-hook baseline


# ---------------------------------------------------------------------------
# DIRECTION
# ---------------------------------------------------------------------------
def train_memory_direction(acts_dir: Path, dataset: Path, split: Path,
                           best_layer: int, *, epochs: int, device: str):
    """Train the pooled MEMORY-family linear probe at `best_layer` on cached
    acts; return (w_hat unit-normalized float32 (d,), raw_w_norm, n_fit_pos,
    n_fit_neg). Mirrors per_cwe_probe.py's pooled-family fit (memory positives
    plus ALL cwe==null negatives, VAL groups excluded). Inputs never mutated."""
    rows, train_eids, test_eids = _te.load_or_make_split(dataset, split)

    # group-aware 15% VAL carve of TRAIN (exact parity with exp-10).
    tr_eid_to_group = {e: _te.pair_group_key(rows[e]) for e in train_eids}
    tr_groups = sorted(set(tr_eid_to_group.values()))
    rng = np.random.default_rng(VAL_SEED)
    rng.shuffle(tr_groups)
    n_val = max(1, int(round(VAL_FRAC * len(tr_groups))))
    val_groups = set(tr_groups[:n_val])
    val_eids = {e for e, g in tr_eid_to_group.items() if g in val_groups}
    fit_eids = train_eids - val_eids

    def cwe_of(e):
        return rows[e].get("cwe")

    # Pooled memory family: every positive whose CWE maps to "memory", plus ALL
    # cwe==null negatives in the fit pool (matches per_cwe_probe.py --neg-pool all).
    pos_fit = {e for e in fit_eids if FAMILY.get(cwe_of(e)) == "memory"}
    neg_fit = {e for e in fit_eids if not cwe_of(e)}
    spec_fit_eids = pos_fit | neg_fit
    if not pos_fit:
        raise SystemExit("[steer] no memory-family positives in fit pool — "
                         "cannot train a memory direction")

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
        raise SystemExit("[steer] degenerate probe: ||w|| == 0")
    w_hat = (w / raw_norm).astype(np.float32)
    return w_hat, raw_norm, len(pos_fit), len(neg_fit)


def layer_hidden_norm_scale(acts_dir: Path, best_layer: int) -> float:
    """scale = median over tokens of the L2 norm of the layer-L hidden state,
    from the cached acts at layer L (hidden_states[L+1] = the layer's output —
    the SAME tensor the hook adds to). Read-only mmap; no mutation.

    TODO(adhoc-decision): scale unit. Default = activation-RMS (token-L2-norm
    median). Alternatives the lead may prefer: probe-margin std (std of w_hat . h
    over tokens), or a fixed constant. Whatever is chosen, it must be the SAME
    unit the alpha grid is interpreted in, and documented in the output JSON."""
    Xmm = np.load(acts_dir / f"layer_{best_layer:02d}.npy", mmap_mode="r")
    n = Xmm.shape[0]
    # Stream the norm in chunks (cached acts can be tens of GB); float64 accum.
    norms = np.empty(n, dtype=np.float64)
    step = 100_000
    for s in range(0, n, step):
        chunk = np.asarray(Xmm[s:s + step], np.float64)
        norms[s:s + step] = np.linalg.norm(chunk, axis=1)
    return float(np.median(norms))


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
    so we change the additive vector between alpha values WITHOUT re-registering
    (cheaper, and keeps one hook handle for clean removal). The *state object* is
    mutated by design (a control register); the model tensors are not."""

    def __init__(self):
        self.delta = None  # torch tensor (d,) on model device/dtype, or None


def _make_hook(state: _SteerState):
    """Forward hook adding state.delta to the layer OUTPUT[0] residual stream at
    ALL token positions. Returns a NEW output tuple (does not mutate the model's
    tensors in place beyond the standard hook-return contract).

    TODO(adhoc-decision): intervention is ADDITIVE along w_hat at ALL positions
    (briefed default). Alternatives the lead may want: code-only positions
    (needs the per-token code mask threaded into the forward — not wired here),
    or projection-removal (subtract the w_hat component then add alpha*scale).
    This is additive + all-positions."""

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--best-layer", type=int, required=True)
    ap.add_argument("--acts-dir", required=True,
                    help="layersweep_<slug>/acts (layer_NN.npy + y.npy + "
                         "example_ids.npy live here)")
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
    if out_path.exists():
        print(f"[steer] {out_path} exists — skip", file=sys.stderr)
        return
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

    # 1. DIRECTION (cached acts only — no model needed yet).
    w_hat, raw_norm, n_fit_pos, n_fit_neg = train_memory_direction(
        acts_dir, dataset, split, L, epochs=args.epochs, device=fit_device)
    # 2. SCALE.
    scale = layer_hidden_norm_scale(acts_dir, L)
    print(f"[steer] direction: ||w||={raw_norm:.4f} fit_pos={n_fit_pos} "
          f"fit_neg={n_fit_neg}; scale(median |h|_2 at L={L})={scale:.3f}",
          file=sys.stderr)

    # Save the direction next to the JSON (provenance + reuse by Tier-2).
    dir_path = Path(str(out_path) + ".dir.pt")
    torch.save({"w_hat": torch.from_numpy(w_hat),
                "raw_norm": raw_norm, "scale": scale, "layer": L,
                "model": args.model, "family": "memory"}, dir_path)

    # 3. LOAD MODEL + register hook on layer L.
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

    # base steering vector (scale * w_hat) on the model device; per-alpha delta
    # = alpha * base. Built once.
    w_hat_t = torch.from_numpy(w_hat).to(device=device, dtype=torch.float32)
    base_vec = (scale * w_hat_t)

    rows, train_eids, test_eids = _te.load_or_make_split(dataset, split)
    subsets = pick_subsets(rows, test_eids, args.n_per_subset)
    n_per = {k: len(v) for k, v in subsets.items()}
    print(f"[steer] subset sizes (test split): {n_per}", file=sys.stderr)

    # Pre-render each subset eid's input ONCE (the prompt is alpha-independent;
    # only the residual-stream perturbation differs). Cache encs to avoid
    # re-tokenizing across the alpha sweep.
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

    def forward_pyes(enc) -> float:
        out = model(**enc, use_cache=False)
        logits_last = out.logits[0, -1, :]
        return p_yes_from_logits(logits_last, yes_ids, no_ids)

    # by_subset[name] = list aligned to `alphas` of mean P(yes).
    by_subset: dict[str, list[float]] = {k: [] for k in subsets}
    # per-eid alpha=0 P(yes) under the hook (delta=None) for the self-check.
    selfcheck = {"tol": SELFCHECK_TOL, "max_abs_diff": 0.0, "ok": True,
                 "n_checked": 0, "worst": None}

    with torch.inference_mode():
        # --- alpha=0 self-check: hook present but delta=None MUST equal a true
        #     no-hook forward to < tol. Run on a small check set (first eid of
        #     each non-empty subset) and ABORT on mismatch. ---
        check_eids = []
        for sub_eids in subsets.values():
            if sub_eids:
                check_eids.append(sub_eids[0])
        state.delta = None
        with_hook_idle = {e: forward_pyes(enc_by_eid[e]) for e in check_eids}
        handle.remove()  # truly no hook now
        no_hook = {e: forward_pyes(enc_by_eid[e]) for e in check_eids}
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

        # --- alpha sweep ---
        for alpha in alphas:
            state.delta = None if alpha == 0.0 else (alpha * base_vec)
            for name, sub_eids in subsets.items():
                if not sub_eids:
                    by_subset[name].append(float("nan"))
                    continue
                ps = [forward_pyes(enc_by_eid[e]) for e in sub_eids]
                by_subset[name].append(float(np.mean(ps)))
            row = " ".join(f"{n}={by_subset[n][-1]:.3f}" for n in subsets)
            print(f"[steer] alpha={alpha:+.2f}  {row}", file=sys.stderr)

    handle.remove()

    result = {
        "model": args.model,
        "layer": L,
        "scale": scale,
        "scale_def": "median over tokens of L2 norm of layer-L hidden state "
                     "(== hidden_states[L+1], the probe's training tensor)",
        "raw_w_norm": raw_norm,
        "intervention": "additive along unit w_hat, ALL token positions, "
                        "on decoder layer L output[0]",
        "alpha_grid": alphas,
        "by_subset": by_subset,
        "n_per_subset": n_per,
        "n_fit_pos": n_fit_pos,
        "n_fit_neg": n_fit_neg,
        "baseline_pyes_alpha0_selfcheck": selfcheck,
        "question": _vj.QUESTION,
        "direction_path": str(dir_path),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[steer] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
