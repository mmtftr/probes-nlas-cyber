"""Train a linear probe with the span-max loss from Obeso, Arditi et al. 2025.

Reference: arXiv 2509.03531 §3 "Probe loss".

The loss is:
    L_probe = (1-omega) * sum_{i in T} w_i * BCE(y_i, p_i)
            + omega     * sum_{s in S} BCE(y_s, max_{i in s} p_i)

where:
  - T = all token positions across all training examples
  - S = the set of annotated vulnerable-line spans (positive examples)
  - y_i = 1 if token i is inside any span, else 0
  - y_s = example-level label for span s (we use 1 here; spans are recorded only
    for positive examples, see src/extract_token_activations.py)
  - p_i = sigmoid(w . h_i + b)
  - w_i = alpha=10 if token i is inside a span, else 1
  - omega is annealed 0 -> 1 linearly over training steps

We train one probe per captured layer, evaluate token-level AND example-level
AUC on a held-out 10% example split (stratified by example label), and save
the best layer's (w, b, layer) in the same npz format as src/train_probe.py
so `data/probe_spanmax.npz` is a drop-in replacement for `data/probe.npz`.

Run:
    python -m src.train_probe_spanmax
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# -------- model --------

class LinearProbe(nn.Module):
    """w . h + b  -- single scalar logit per token."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (n_tokens, hidden_dim) -> (n_tokens,)
        return self.linear(X).squeeze(-1)


class MLPProbe(nn.Module):
    """Two-layer MLP head: in_dim -> hidden -> 1 logit per token."""
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
    def forward(self, X: torch.Tensor) -> torch.Tensor:  # (n_tokens, in_dim) -> (n_tokens,)
        return self.net(X).squeeze(-1)


# -------- data --------

def _group_by_example(X: np.ndarray, y: np.ndarray, example_ids: np.ndarray):
    """Return a dict {eid: (X_eid, y_eid)} preserving order."""
    order = np.argsort(example_ids, kind="stable")
    X = X[order]
    y = y[order]
    eids = example_ids[order]

    # find run boundaries
    boundaries = np.concatenate(([0], np.where(np.diff(eids) != 0)[0] + 1, [len(eids)]))
    groups: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        eid = int(eids[a])
        groups[eid] = (X[a:b], y[a:b])
    return groups


def _example_label(y_tokens: np.ndarray) -> int:
    """Example is positive iff it has any positive token (matches extractor logic)."""
    return int(y_tokens.any())


# -------- loss --------

def soft_labels_triangular(
    y_hard: np.ndarray,
    window: int,
) -> np.ndarray:
    """Triangular decay around hard 0/1 token labels (closes #26).

    Inside any positive run: 1.0. Outside, linear ramp `1 - d/window` to 0
    over `window` tokens, then 0. Overlapping tails resolve via max.
    `window=0` returns the hard labels unchanged.
    """
    if window <= 0:
        return y_hard.astype(np.float32)
    n = len(y_hard)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    soft = y_hard.astype(np.float32).copy()
    # Identify contiguous positive runs.
    edges = np.diff(np.concatenate(([0], y_hard, [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1  # inclusive
    for s, e in zip(starts, ends):
        # Left tail.
        for d in range(1, window + 1):
            i = s - d
            if i < 0:
                break
            v = 1.0 - d / window
            if v > soft[i]:
                soft[i] = v
        # Right tail.
        for d in range(1, window + 1):
            i = e + d
            if i >= n:
                break
            v = 1.0 - d / window
            if v > soft[i]:
                soft[i] = v
    return soft


def span_max_loss(
    logits: torch.Tensor,
    y_tok: torch.Tensor,
    is_positive: bool,
    omega: float,
    alpha: float = 10.0,
    y_soft: torch.Tensor | None = None,
    neg_incl: bool = False,
) -> torch.Tensor:
    """Per-example contribution to L_probe.

    Args:
        logits: (n_pos,) raw logits for one example's tokens
        y_tok:  (n_pos,) 0/1 per-token HARD labels (drives alpha-weighting and
                the span-max pool — we still want one peak per annotated span).
        is_positive: whether this example has an annotated span (y_s=1).
        omega: annealed weight in [0, 1].
        alpha: up-weighting factor for entity tokens in the per-token term.
        y_soft: optional float (n_pos,) target distribution in [0, 1] used by
                the per-token BCE term. Defaults to `y_tok.float()`. Soft
                labels with a triangular tail (#26) let the probe also fire
                on context tokens near the annotated diff range without being
                punished by hard-0 BCE.
        neg_incl: if True, NEGATIVE examples also contribute a span term
                BCE(y_s=0, max_i p_i) over ALL their tokens — see
                `span_max_loss_neg_incl`. Default False = paper-faithful span-max
                (negatives shaped only by the per-token term).

    Returns:
        Scalar loss contribution (NOT yet averaged; caller sums then averages).
    """
    if y_soft is None:
        y_soft = y_tok.float()
    # Per-token weighted BCE against the soft target.
    bce_per_tok = nn.functional.binary_cross_entropy_with_logits(
        logits, y_soft, reduction="none"
    )
    w = torch.where(y_tok > 0, torch.full_like(bce_per_tok, alpha), torch.ones_like(bce_per_tok))
    token_term = (w * bce_per_tok).sum()

    # Span term: BCE(y_s=1, max_i p_i) over the HARD-span tokens. Keep this on
    # hard labels so the max-pool tracks the actual annotated vuln region, not
    # the softened halo around it.
    if is_positive:
        span_logits = logits[y_tok > 0]
        if span_logits.numel() == 0:
            span_term = logits.new_tensor(0.0)
        else:
            max_logit = span_logits.max()
            y_s = logits.new_tensor(1.0)
            span_term = nn.functional.binary_cross_entropy_with_logits(
                max_logit, y_s, reduction="sum"
            )
    elif neg_incl and logits.numel() > 0:
        # Negative example: push its SINGLE highest-scoring token toward 0. This
        # directly optimizes the example-level score we evaluate on (example prob
        # = max_i sigmoid(logit_i)). Baseline span-max leaves this term at 0 and
        # shapes negatives only through the per-token BCE.
        max_logit = logits.max()
        y_s = logits.new_tensor(0.0)
        span_term = nn.functional.binary_cross_entropy_with_logits(
            max_logit, y_s, reduction="sum"
        )
    else:
        span_term = logits.new_tensor(0.0)

    return (1.0 - omega) * token_term + omega * span_term


def span_max_loss_neg_incl(
    logits: torch.Tensor,
    y_tok: torch.Tensor,
    is_positive: bool,
    omega: float,
    alpha: float = 10.0,
    y_soft: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative-inclusive span-max: like `span_max_loss` but negative examples
    also contribute BCE(y_s=0, max_i p_i) over all their tokens.

    Rationale: the eval metric is the per-example max token sigmoid, so a clean
    example is penalised exactly by its highest false-alarm token. The baseline
    span term is one-sided (positives only), pulling the in-span max up but never
    explicitly pushing clean code's max down — which the per-token BCE does only
    weakly once ω anneals the token term away. This restores the symmetry."""
    return span_max_loss(logits, y_tok, is_positive, omega,
                          alpha=alpha, y_soft=y_soft, neg_incl=True)


# -------- training --------

def train_one_layer(
    X: np.ndarray,
    y: np.ndarray,
    example_ids: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_examples: int = 8,
    weight_decay: float = 1e-4,
    seed: int = 7,
    device: str = "cpu",
    verbose: bool = True,
    label_window: int = 0,
    alpha: float = 10.0,
    neg_incl: bool = False,
    probe_factory: "Callable[[int], nn.Module] | None" = None,
    mask_negatives: str = "none",
    code_mask: "np.ndarray | None" = None,
) -> dict:
    """Train one probe with the span-max loss and return metrics + (w, b).

    alpha: in-span up-weight for the per-token BCE term (loss sweep knob).
    neg_incl: use the negative-inclusive span term (`span_max_loss_neg_incl`).
    probe_factory: optional callable (hidden_dim) -> nn.Module to build a custom
        probe head (e.g. MLPProbe). Defaults to LinearProbe. Non-linear heads
        return w=None, b=None; the trained module is always in the "probe" key.
    mask_negatives: ADDITIVE train-time negative filter. "none" (default) is the
        original behavior — every out-of-span token is a negative. "code_only"
        EXCLUDES tokens that are NOT live-code AND not positive (`~code_mask &
        (y==0)`) from the loss entirely — they are neither positive nor
        negative, not down-weighted. See `src/eval/code_mask.py` for the
        motivation (trivial comment/signature negatives inflate the easy win).
    code_mask: (n_tokens,) bool array aligned to X/y/example_ids, True = live
        code. Required when mask_negatives=="code_only"; ignored otherwise. The
        EVAL split (internal 10% val) is scored on the SAME filtered tokens so
        train and val see a consistent negative set.

    Immutability: input X/y/example_ids/code_mask are never mutated; filtered
    copies are built when mask_negatives=="code_only"."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if mask_negatives not in ("none", "code_only"):
        raise ValueError(
            f"mask_negatives must be 'none' or 'code_only', got {mask_negatives!r}"
        )
    if mask_negatives == "code_only":
        if code_mask is None:
            raise ValueError("mask_negatives='code_only' requires code_mask")
        code_mask = np.asarray(code_mask, dtype=bool)
        if code_mask.shape[0] != X.shape[0]:
            raise ValueError(
                f"code_mask length {code_mask.shape[0]} != n_tokens {X.shape[0]}"
            )
        # Keep a token iff it is live-code OR positive. Drop ~code & negative.
        keep = code_mask | (np.asarray(y) != 0)
        # Build fresh filtered arrays (no in-place mutation of inputs).
        X = X[keep]
        y = np.asarray(y)[keep]
        example_ids = np.asarray(example_ids)[keep]

    groups = _group_by_example(X, y, example_ids)
    all_eids = sorted(groups.keys())
    ex_labels = np.array([_example_label(groups[e][1]) for e in all_eids], dtype=np.int64)

    # 90/10 split, stratified by example-level label.
    try:
        eids_tr, eids_te = train_test_split(
            all_eids, test_size=0.1, random_state=seed, stratify=ex_labels
        )
    except ValueError:
        eids_tr, eids_te = train_test_split(all_eids, test_size=0.1, random_state=seed)

    hidden_dim = X.shape[1]
    probe = (probe_factory(hidden_dim) if probe_factory is not None else LinearProbe(hidden_dim)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max(1, epochs * ((len(eids_tr) + batch_examples - 1) // batch_examples))
    step = 0

    # Pre-move tensors to device per example to amortize host->device copies.
    # cache: (X, y_hard, y_soft, is_positive). y_soft == y_hard.float() when
    # --label-window is 0; otherwise carries the triangular tail (#26).
    cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]] = {}
    for eid in all_eids:
        Xi, yi = groups[eid]
        yi_soft = soft_labels_triangular(yi, label_window)
        cache[eid] = (
            torch.from_numpy(Xi).to(device),
            torch.from_numpy(yi).long().to(device),
            torch.from_numpy(yi_soft).float().to(device),
            bool(_example_label(yi)),
        )

    best_eval_token_auc = -1.0
    best_eval_example_auc = -1.0
    best_state = None
    history: list[dict] = []

    for ep in range(epochs):
        probe.train()
        rng = np.random.default_rng(seed + ep)
        order = list(eids_tr)
        rng.shuffle(order)

        running = 0.0
        n_examples_seen = 0
        for i in range(0, len(order), batch_examples):
            batch = order[i : i + batch_examples]
            opt.zero_grad()
            omega = min(1.0, step / max(1, total_steps - 1))
            loss_accum = torch.zeros((), device=device)
            n_tok_total = 0
            for eid in batch:
                Xi, yi, yi_soft, is_pos = cache[eid]
                logits = probe(Xi)
                contrib = span_max_loss(logits, yi, is_pos, omega=omega, y_soft=yi_soft,
                                        alpha=alpha, neg_incl=neg_incl)
                loss_accum = loss_accum + contrib
                n_tok_total += yi.shape[0]
            # Average per token so the scale is comparable across batches.
            loss = loss_accum / max(1, n_tok_total)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n_examples_seen += len(batch)
            step += 1

        # ---- eval ----
        probe.eval()
        with torch.no_grad():
            tok_y_all = []
            tok_p_all = []
            ex_y_all = []
            ex_p_all = []
            for eid in eids_te:
                Xi, yi, _y_soft, is_pos = cache[eid]
                logits = probe(Xi)
                p = torch.sigmoid(logits).detach().to("cpu").numpy()
                yi_np = yi.detach().to("cpu").numpy()
                tok_p_all.append(p)
                tok_y_all.append(yi_np)
                ex_y_all.append(int(is_pos))
                ex_p_all.append(float(p.max()))

            tok_y = np.concatenate(tok_y_all)
            tok_p = np.concatenate(tok_p_all)
            ex_y = np.array(ex_y_all)
            ex_p = np.array(ex_p_all)

            try:
                tok_auc = float(roc_auc_score(tok_y, tok_p)) if len(np.unique(tok_y)) > 1 else float("nan")
            except ValueError:
                tok_auc = float("nan")
            try:
                ex_auc = float(roc_auc_score(ex_y, ex_p)) if len(np.unique(ex_y)) > 1 else float("nan")
            except ValueError:
                ex_auc = float("nan")

        history.append({"epoch": ep, "loss": running / max(1, len(order)), "tok_auc": tok_auc, "ex_auc": ex_auc, "omega_end": omega})
        if verbose:
            print(
                f"[spanmax]   ep {ep+1:02d}/{epochs}  loss={running/max(1,len(order)):.4f}  "
                f"tok_AUC={tok_auc:.3f}  ex_AUC={ex_auc:.3f}  omega={omega:.2f}",
                file=sys.stderr,
            )

        # Select the best epoch by example-level AUC (the paper's headline metric),
        # falling back to token-level if example-level is undefined.
        score = ex_auc if not np.isnan(ex_auc) else tok_auc
        best_score = best_eval_example_auc if not np.isnan(best_eval_example_auc) else best_eval_token_auc
        if not np.isnan(score) and score > (best_score if not np.isnan(best_score) else -1):
            best_eval_token_auc = tok_auc
            best_eval_example_auc = ex_auc
            best_state = {k: v.detach().to("cpu").clone() for k, v in probe.state_dict().items()}

    # Restore best.
    if best_state is not None:
        probe.load_state_dict(best_state)

    if isinstance(probe, LinearProbe):
        w = probe.linear.weight.detach().to("cpu").numpy().astype(np.float32).reshape(-1)
        b = float(probe.linear.bias.detach().to("cpu").numpy().reshape(-1)[0])
    else:
        w = None
        b = None
    return {
        "w": w,
        "b": b,
        "probe": probe.to("cpu"),
        "tok_auc": best_eval_token_auc,
        "ex_auc": best_eval_example_auc,
        "n_train_examples": len(eids_tr),
        "n_eval_examples": len(eids_te),
        "history": history,
    }


# -------- driver --------

def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts-dir", default="data/token_activations")
    ap.add_argument("--out", default="data/probe_spanmax.npz")
    ap.add_argument("--card", default="data/probe_spanmax_card.json")
    ap.add_argument("--layers", nargs="*", type=int, default=[8, 17, 26, 34])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-examples", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--label-window", type=int, default=0,
        help="Soft-label triangular decay window in tokens (#26). 0 = "
        "hard 0/1 labels (legacy). 8 = label ramps 1 → 0 over 8 tokens "
        "outside each annotated span.",
    )
    args = ap.parse_args()

    acts_dir = Path(args.acts_dir)
    expected = [acts_dir / f"token_activations_layer{li:02d}.npz" for li in args.layers]
    missing = [p for p in expected if not p.exists()]
    if missing:
        print(
            "[span-max] waiting for data/token_activations/*.npz — run src/extract_token_activations.py first",
            file=sys.stderr,
        )
        for p in missing:
            print(f"[span-max]   missing: {p}", file=sys.stderr)
        sys.exit(0)

    device = args.device or _device()
    print(f"[span-max] device={device}", file=sys.stderr)

    results: list[dict] = []
    for li, path in zip(args.layers, expected):
        npz = np.load(path)
        X = npz["X"]
        y = npz["y"]
        example_ids = npz["example_ids"]
        n_examples = int(np.unique(example_ids).size)
        print(
            f"[span-max] layer {li:02d}  X={X.shape}  tokens_pos={int(y.sum())}/{len(y)}  "
            f"examples={n_examples}",
            file=sys.stderr,
        )
        r = train_one_layer(
            X, y, example_ids,
            epochs=args.epochs,
            lr=args.lr,
            batch_examples=args.batch_examples,
            device=device,
            label_window=args.label_window,
        )
        r["layer"] = li
        results.append(r)
        print(
            f"[span-max] layer {li:02d}  best tok_AUC={r['tok_auc']:.3f}  best ex_AUC={r['ex_auc']:.3f}",
            file=sys.stderr,
        )

    # Pick winner by example-level AUC (the task we actually care about at serve time).
    def _score(r: dict) -> float:
        v = r["ex_auc"]
        return v if not np.isnan(v) else -1.0

    best = max(results, key=_score)
    print(
        f"[span-max] best layer = {best['layer']}  ex_AUC={best['ex_auc']:.3f}  tok_AUC={best['tok_auc']:.3f}",
        file=sys.stderr,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        w=best["w"].astype(np.float32),
        b=np.float32(best["b"]),
        layer=np.int32(best["layer"]),
    )

    card = {
        "loss": "span-max (Obeso, Arditi et al. 2025, arXiv 2509.03531 §3)",
        "alpha": 10.0,
        "omega_schedule": "linear 0->1 over all training steps",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_examples": args.batch_examples,
        "label_window": args.label_window,
        "best_layer": best["layer"],
        "best_token_auc": best["tok_auc"],
        "best_example_auc": best["ex_auc"],
        "previous_sklearn_logreg_layer8_auc": 0.846,
        "all_layers": [
            {
                "layer": r["layer"],
                "token_auc": r["tok_auc"],
                "example_auc": r["ex_auc"],
                "n_train_examples": r["n_train_examples"],
                "n_eval_examples": r["n_eval_examples"],
            }
            for r in results
        ],
    }
    Path(args.card).write_text(json.dumps(card, indent=2))
    print(f"[span-max] saved {out_path} + {args.card}", file=sys.stderr)
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
