# [ai-generated]
"""Tiny local unit test of the P(yes) math + resolve_yes_no_ids, with a FAKE
tokenizer stub and a fake logit vector. Does NOT load any real model/tokenizer.

    uv run python test_pyes_math.py
"""
from __future__ import annotations
import sys

import numpy as np

import verbalized_judge as vj


class FakeTokenizer:
    """Deterministic stub: maps each whitespace-stripped word to a fixed id.

    Yes-variants -> id in {10, 11, 12}, No-variants -> id in {20, 21, 22},
    so resolve_yes_no_ids must recover those two disjoint id sets.
    """
    _MAP = {
        "yes": 10, "Yes": 11, "YES": 12,
        "no": 20, "No": 21, "NO": 22,
    }

    def encode(self, w, add_special_tokens=False):  # noqa: ARG002
        key = w.strip()
        if key in self._MAP:
            return [self._MAP[key]]
        return []


def main() -> None:
    tok = FakeTokenizer()

    # --- resolve_yes_no_ids ---
    yes_ids, no_ids = vj.resolve_yes_no_ids(tok)
    assert yes_ids == [10, 11, 12], yes_ids
    assert no_ids == [20, 21, 22], no_ids
    assert set(yes_ids).isdisjoint(no_ids)
    print(f"[test] resolve_yes_no_ids: yes={yes_ids} no={no_ids}  OK")

    # --- p_yes_from_logits: in (0,1) and monotone increasing in the yes logit ---
    vocab = 30
    base = np.zeros(vocab, dtype=np.float64)
    base[yes_ids] = 0.0
    base[no_ids] = 0.0
    p_mid = vj.p_yes_from_logits(base, yes_ids, no_ids)
    assert 0.0 < p_mid < 1.0, p_mid
    # equal yes/no logits -> P(yes) == 0.5 (sigmoid(0)).
    assert abs(p_mid - 0.5) < 1e-9, p_mid

    prev = None
    for bump in [-2.0, 0.0, 2.0, 5.0]:
        logits = base.copy()
        logits[yes_ids] += bump  # raise every yes-id logit
        p = vj.p_yes_from_logits(logits, yes_ids, no_ids)
        assert 0.0 < p < 1.0, (bump, p)
        if prev is not None:
            assert p > prev, f"P(yes) not increasing: bump={bump} p={p} prev={prev}"
        prev = p
        print(f"[test] yes-logit bump {bump:+.1f} -> P(yes)={p:.4f}")

    # Sanity: raising the NO logit instead should DECREASE P(yes).
    logits = base.copy()
    logits[no_ids] += 5.0
    p_no_up = vj.p_yes_from_logits(logits, yes_ids, no_ids)
    assert p_no_up < 0.5, p_no_up
    print(f"[test] no-logit bump +5.0 -> P(yes)={p_no_up:.4f} (< 0.5)  OK")

    # --- build_content places code BEFORE the question ---
    content = vj.build_content("int x = 0;")
    assert content.index("int x = 0;") < content.index(vj.QUESTION)
    assert content.startswith(vj.PREAMBLE)
    print("[test] build_content: code precedes question  OK")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
