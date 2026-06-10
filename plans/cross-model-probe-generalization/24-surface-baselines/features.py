# [ai-generated]
"""exp-24 feature matrices + baseline classifiers (CPU, sparse).

Precompute, once over all tokens, the surface feature blocks:
  - Umat : one-hot token-identity (vocab = train-observed token strings)
  - Hmat : hashed char 3-5-gram of the ±48 window (HashingVectorizer 2^18)
  - Kmat : (N,5) security-lexicon hit counts in window
  - Lmat : (N,1) language indicator (C/C++ = 1)
Each baseline slices rows for its train pool, fits a LogisticRegression, and
emits a per-token score over ALL tokens (eval just indexes in).

Training negatives are capped (NEG_CAP) for runtime; EVAL pools are always full.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

from substrate import Substrate, keyword_counts, lang_indicator

NEG_CAP = 60_000          # max negative train tokens per fit
NEG_POS_RATIO = 25        # and at most this many negs per positive
SEED = 42
HASH_BITS = 18


def build_feature_blocks(s: Substrate, train_idx: np.ndarray) -> dict:
    """Build (and cache) the four feature blocks. `train_idx` defines the
    unigram vocabulary (train-observed token strings only)."""
    # --- token-identity one-hot (vocab from train) ---
    vocab: dict[str, int] = {}
    for i in train_idx:
        t = s.tok[i].strip()
        if t and t not in vocab:
            vocab[t] = len(vocab)
    V = len(vocab)
    rows, cols = [], []
    for i in range(len(s.tok)):
        t = s.tok[i].strip()
        j = vocab.get(t)
        if j is not None:
            rows.append(i)
            cols.append(j)
    Umat = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(s.tok), V))

    # --- hashed char 3-5-gram of window ---
    hv = HashingVectorizer(analyzer="char", ngram_range=(3, 5),
                           n_features=2 ** HASH_BITS, alternate_sign=False,
                           norm=None, dtype=np.float32)
    Hmat = hv.transform(s.win)

    # --- keyword counts + language indicator ---
    Kmat = sp.csr_matrix(keyword_counts(s.win))
    Lmat = sp.csr_matrix(lang_indicator(s.lang))

    return {"U": Umat, "H": Hmat, "K": Kmat, "L": Lmat, "vocab_size": V}


def _subsample_negs(pos_idx: np.ndarray, neg_idx: np.ndarray, rng) -> np.ndarray:
    cap = min(NEG_CAP, max(NEG_POS_RATIO * max(len(pos_idx), 1), 2000))
    if len(neg_idx) > cap:
        neg_idx = rng.choice(neg_idx, cap, replace=False)
    return neg_idx


def fit_lr_score(feat: sp.spmatrix, pos_idx: np.ndarray, neg_idx: np.ndarray,
                 rng) -> np.ndarray:
    """Fit LR on pos+subsampled-neg rows of `feat`; return per-token score (N,)."""
    neg_idx = _subsample_negs(pos_idx, neg_idx, rng)
    tr = np.concatenate([pos_idx, neg_idx])
    ytr = np.concatenate([np.ones(len(pos_idx), np.int8),
                          np.zeros(len(neg_idx), np.int8)])
    if len(np.unique(ytr)) < 2:
        return np.full(feat.shape[0], np.nan, np.float32)
    clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    clf.fit(feat[tr], ytr)
    return clf.decision_function(feat).astype(np.float32)


def hstack(blocks: dict, keys: tuple[str, ...]) -> sp.spmatrix:
    return sp.hstack([blocks[k] for k in keys], format="csr")
