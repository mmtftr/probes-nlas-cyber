# [ai-generated]
"""Token-level FN/FP error analysis of the pooled span-max vulnerability probe
on the SVEN-SUBTRACTIVE subset, across all 7 model variants.

Rides on the exp-16 token-logit dumps (per-token logit + char offsets + is_code
mask, persisted at each model's operating layer) and the exp-19 subtractive
membership. Recomputes the HONEST label (tight difflib delete/replace span in
`before` ∩ tree-sitter live-code), picks a per-model operating threshold
(F1-max on the subtractive TRAIN code tokens), and on the held-out subtractive
TEST tokens computes:

  - per-token confusion (TP/FP/FN) over code tokens, honest labels;
  - per-(model, vuln-example) detection bit = does ANY tight vulnerable code
    token clear threshold;  -> cross-model detect/miss matrix (Qwen vs Gemma);
  - FN corpus: every subtractive TEST vuln example, marked vulnerable span +
    per-model detect bits (so the same corpus answers "consistently detected"
    AND "consistently missed");
  - FP corpus: spurious high-scoring code-token spans (label-0 tokens clearing
    threshold), merged into contiguous spans, ranked, deduped across models.

Outputs (this dir):
  analysis.json   — all numbers (thresholds, confusion, detection matrix).
  fn_corpus.json  — [{eid, cwe, lang, n_detect, detect{model:bit}, excerpt, ...}]
  fp_corpus.json  — [{eid, is_safe, cwe, lang, models:[...], excerpt, logit, ...}]

Methodology choices (TODO(adhoc-decision), see RESULTS.md):
  - operating point = per-model F1-max threshold on subtractive TRAIN code tokens;
  - regime = code-only (is_code), tight∩is_code positives (ADR-0004 honest label);
  - example "detected" iff >=1 tight vulnerable CODE token clears threshold.
"""
from __future__ import annotations

import difflib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DS_PATH = REPO / "data" / "dataset.jsonl"
LOGITS = HERE.parent / "16-token-logit-dump" / "results"
MEMBERSHIP = HERE.parent / "19-subtractive-regime" / "subtractive_membership.json"

# operating layer per model dir (from exp-16 relabel_recompute.py)
OP_LAYER = {
    "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct": 25,
    "logitdump_Qwen_Qwen2.5-Coder-7B-Instruct": 16,
    "logitdump_google_gemma-3-1b-it": 25,
    "logitdump_google_gemma-3-4b-it": 7,
    "logitdump_google_gemma-3-12b-it": 15,
    "logitdump_google_gemma-3-27b-it": 19,
    "logitdump_google_gemma-3-12b-pt": 13,
}
# short labels + family
MODELS = {
    "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct": ("qwen-32b", "qwen"),
    "logitdump_Qwen_Qwen2.5-Coder-7B-Instruct": ("qwen-7b", "qwen"),
    "logitdump_google_gemma-3-27b-it": ("gemma-27b-it", "gemma"),
    "logitdump_google_gemma-3-12b-it": ("gemma-12b-it", "gemma"),
    "logitdump_google_gemma-3-12b-pt": ("gemma-12b-pt", "gemma"),
    "logitdump_google_gemma-3-4b-it": ("gemma-4b-it", "gemma"),
    "logitdump_google_gemma-3-1b-it": ("gemma-1b-it", "gemma"),
}
FLAGSHIP = {"qwen": "logitdump_Qwen_Qwen2.5-Coder-32B-Instruct",
            "gemma": "logitdump_google_gemma-3-27b-it"}


def tight_spans(before: str, after: str):
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    return [(i1, i2) for tag, i1, i2, j1, j2 in sm.get_opcodes()
            if tag in ("replace", "delete") and i2 > i1]


def f1max_threshold(y, s):
    """Threshold maximizing F1 over score s; returns (thr, f1, prec, rec)."""
    p, r, t = precision_recall_curve(y, s)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    k = int(np.argmax(f1[:-1])) if len(t) else 0
    return float(t[k]), float(f1[k]), float(p[k]), float(r[k])


def line_window(code: str, spans, ctx=5):
    """Return (excerpt_str, start_line) covering the lines that `spans`
    (char ranges) touch, padded by ctx lines, with each span wrapped «...»."""
    if not spans:
        return "", 0
    # char index -> line number
    line_starts = [0]
    for i, ch in enumerate(code):
        if ch == "\n":
            line_starts.append(i + 1)
    n_lines = len(line_starts)

    def line_of(ci):
        lo, hi = 0, n_lines - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= ci:
                lo = mid
            else:
                hi = mid - 1
        return lo

    touched = set()
    for s, e in spans:
        for ln in range(line_of(s), line_of(max(s, e - 1)) + 1):
            touched.add(ln)
    lo = max(0, min(touched) - ctx)
    hi = min(n_lines - 1, max(touched) + ctx)
    # build marked code by inserting markers at char positions
    marks = []
    for s, e in spans:
        marks.append((s, "«"))
        marks.append((e, "»"))
    marks.sort(key=lambda x: (x[0], x[1] == "«"))
    out = []
    cur = 0
    for pos, m in marks:
        out.append(code[cur:pos])
        out.append(m)
        cur = pos
    out.append(code[cur:])
    marked = "".join(out)
    # re-split marked code by lines; map original line indices (markers don't add lines)
    mlines = marked.split("\n")
    excerpt = "\n".join(f"{i+1:4d}| {mlines[i]}" for i in range(lo, min(hi + 1, len(mlines))))
    return excerpt, lo + 1


def merge_contiguous(idx_sorted, cs, ce, gap=2):
    """Merge token indices (already sorted by char_start) whose char spans are
    adjacent (<= gap chars apart) into [(c_start, c_end, [idxs])]."""
    spans = []
    for i in idx_sorted:
        if spans and cs[i] <= spans[-1][1] + gap:
            spans[-1][1] = max(spans[-1][1], int(ce[i]))
            spans[-1][2].append(i)
        else:
            spans.append([int(cs[i]), int(ce[i]), [i]])
    return spans


def main():
    ds = [json.loads(l) for l in open(DS_PATH)]
    mem = json.loads(MEMBERSHIP.read_text())
    pairs = mem["pairs"]                       # [[vuln, safe], ...]
    sub_vuln = set(mem["subtractive_vuln"])
    sub_safe = {s for v, s in pairs}
    sub_all = sub_vuln | sub_safe
    vuln_to_safe = {v: s for v, s in pairs}
    safe_to_vuln = {s: v for v, s in pairs}
    tspans = {v: tight_spans(ds[v]["code"], ds[s]["code"]) for v, s in pairs}

    def pair_cwe(e):
        """CWE for any subtractive example (safe rows carry no cwe -> use vuln pair)."""
        if e in safe_to_vuln:
            return ds[safe_to_vuln[e]].get("cwe")
        return ds[e].get("cwe")

    per_model = {}        # dir -> dict of arrays/metrics
    test_vuln_sets = {}
    for d_name, L in OP_LAYER.items():
        z = np.load(LOGITS / d_name / f"logits_layer{L:02d}.npz")
        logit = z["logit"].astype(np.float64)
        eid = z["example_id"].astype(int)
        cs, ce = z["char_start"].astype(int), z["char_end"].astype(int)
        te = z["is_test"].astype(bool)
        isc = z["is_code"].astype(bool)

        # honest tight∩is_code per-token positive label
        y_tok = np.zeros(len(eid), dtype=int)
        for v, spans in tspans.items():
            if not spans:
                continue
            idx = np.where(eid == v)[0]
            if not len(idx):
                continue
            s_, e_ = cs[idx], ce[idx]
            ov = np.zeros(len(idx), dtype=bool)
            for (i1, i2) in spans:
                ov |= (s_ < i2) & (e_ > i1)
            y_tok[idx] = (ov & isc[idx]).astype(int)

        in_sub = np.isin(eid, list(sub_all))
        code = isc & in_sub                      # subtractive code tokens
        tr = code & ~te
        ev = code & te
        # operating threshold: F1-max on subtractive TRAIN code tokens
        thr, f1, prec, rec = f1max_threshold(y_tok[tr], logit[tr])
        pred = logit >= thr

        # token confusion on subtractive TEST code tokens (honest labels)
        yv, pv = y_tok[ev], pred[ev]
        tp = int(((yv == 1) & pv).sum()); fn = int(((yv == 1) & ~pv).sum())
        fp = int(((yv == 0) & pv).sum()); tn = int(((yv == 0) & ~pv).sum())
        tok_auc = (float(roc_auc_score(yv, logit[ev]))
                   if len(np.unique(yv)) > 1 else float("nan"))

        per_model[d_name] = dict(
            logit=logit, eid=eid, cs=cs, ce=ce, te=te, isc=isc, y_tok=y_tok,
            pred=pred, thr=thr, f1=f1, prec=prec, rec=rec, layer=L,
            conf=dict(tp=tp, fn=fn, fp=fp, tn=tn, tok_auc=tok_auc,
                      precision=tp / (tp + fp) if tp + fp else 0.0,
                      recall=tp / (tp + fn) if tp + fn else 0.0))
        test_vuln_sets[d_name] = {int(v) for v in sub_vuln
                                  if (eid == v).any() and te[eid == v].any()}

    # test vuln eids = consensus across models (should be identical)
    test_vuln = sorted(set.intersection(*test_vuln_sets.values()))
    label, family = {}, {}
    for d, (lab, fam) in MODELS.items():
        label[d] = lab; family[d] = fam

    # ---- per-example detection matrix ----
    detect = {}     # eid -> {dir: bit}
    for v in test_vuln:
        spans = tspans[v]
        row = {}
        for d, pm in per_model.items():
            idx = np.where(pm["eid"] == v)[0]
            s_, e_ = pm["cs"][idx], pm["ce"][idx]
            pos = np.zeros(len(idx), bool)
            for (i1, i2) in spans:
                pos |= (s_ < i2) & (e_ > i1)
            pos &= pm["isc"][idx]
            row[d] = bool(pos.any() and pm["pred"][idx][pos].any())
        detect[v] = row

    def n_det(v):
        return sum(detect[v].values())

    def fam_det(v, fam):
        return [label[d] for d in per_model if family[d] == fam and detect[v][d]]

    # ---- FN corpus: every test vuln example, marked vuln span + detect bits ----
    fn_corpus = []
    for v in test_vuln:
        spans = tspans[v]
        excerpt, ln0 = line_window(ds[v]["code"], spans, ctx=5)
        # flagship: what did it fire on instead (top code token text)?
        top = {}
        for fam, d in FLAGSHIP.items():
            pm = per_model[d]
            idx = np.where((pm["eid"] == v) & pm["isc"])[0]
            if len(idx):
                j = idx[np.argmax(pm["logit"][idx])]
                top[fam] = dict(text=ds[v]["code"][int(pm["cs"][j]):int(pm["ce"][j])],
                                logit=round(float(pm["logit"][j]), 3),
                                thr=round(pm["thr"], 3))
        fn_corpus.append(dict(
            eid=v, cwe=ds[v].get("cwe"), lang=ds[v].get("lang"),
            n_detect=n_det(v),
            detect={label[d]: detect[v][d] for d in per_model},
            qwen_detect=fam_det(v, "qwen"), gemma_detect=fam_det(v, "gemma"),
            removed_span=[ds[v]["code"][i1:i2] for i1, i2 in spans][:6],
            flagship_top=top,
            excerpt=excerpt))

    # ---- FP corpus: spurious high-scoring code-token spans on subtractive test ----
    # build per (eid, char-span) FP records, tag which models fire, rank.
    fp_map = {}      # (eid, cstart, cend) -> record
    for d, pm in per_model.items():
        ev = pm["isc"] & pm["te"] & np.isin(pm["eid"], list(sub_all))
        fp_mask = ev & (pm["y_tok"] == 0) & pm["pred"]
        idx = np.where(fp_mask)[0]
        # group by eid, merge contiguous tokens into spans
        by_eid = defaultdict(list)
        for i in idx:
            by_eid[int(pm["eid"][i])].append(i)
        for e, ii in by_eid.items():
            ii_sorted = sorted(ii, key=lambda k: pm["cs"][k])
            for cstart, cend, members in merge_contiguous(ii_sorted, pm["cs"], pm["ce"]):
                key = (e, cstart, cend)
                maxlogit = max(float(pm["logit"][m]) for m in members)
                margin = round(maxlogit - pm["thr"], 3)
                rec = fp_map.get(key)
                if rec is None:
                    # distance (chars) to nearest true tight vuln span in same
                    # example; None for safe examples (no true span exists).
                    nd = None
                    if e in vuln_to_safe and tspans[e]:
                        nd = min(0 if (cstart < i2 and cend > i1)
                                 else min(abs(cstart - i2), abs(i1 - cend))
                                 for i1, i2 in tspans[e])
                    rec = dict(eid=e, is_safe=(e in sub_safe),
                               cwe=pair_cwe(e), lang=ds[e].get("lang"),
                               cstart=cstart, cend=cend,
                               token_text=ds[e]["code"][cstart:cend],
                               nearest_vuln_dist=nd,
                               models={}, max_margin=margin)
                    fp_map[key] = rec
                rec["models"][label[d]] = margin
                rec["max_margin"] = max(rec["max_margin"], margin)
    # attach excerpts + cross-model count + stratum, sort by (n_models, margin)
    SPREAD_CHARS = 40
    fp_corpus = []
    for rec in fp_map.values():
        rec["n_models"] = len(rec["models"])
        rec["families"] = sorted({family[d] for d in per_model
                                  if label[d] in rec["models"]})
        nd = rec["nearest_vuln_dist"]
        if rec["is_safe"]:
            rec["kind"] = "safe_alarm"      # fires in patched/safe code (no real vuln)
        elif nd is not None and nd <= SPREAD_CHARS:
            rec["kind"] = "spread"          # adjacent to true span (sink spillover)
        else:
            rec["kind"] = "misplaced"       # vuln example, far from any true span
        rec["excerpt"], _ = line_window(ds[rec["eid"]]["code"],
                                        [(rec["cstart"], rec["cend"])], ctx=4)
        fp_corpus.append(rec)
    fp_corpus.sort(key=lambda r: (r["n_models"], r["max_margin"]), reverse=True)

    # curated FP sample for hand-categorization: top-by-(n_models,margin) within
    # each stratum, deduped, capped — captures the variety without 6k rows.
    def stratum_sample(kind, cap, min_models=2):
        rs = [r for r in fp_corpus if r["kind"] == kind and r["n_models"] >= min_models]
        return rs[:cap]
    fp_sample = (stratum_sample("safe_alarm", 50)
                 + stratum_sample("spread", 45)
                 + stratum_sample("misplaced", 35, min_models=1))
    # de-dup by key just in case, keep order
    seen = set(); fp_sample_u = []
    for r in fp_sample:
        k = (r["eid"], r["cstart"], r["cend"])
        if k not in seen:
            seen.add(k); fp_sample_u.append(r)
    fp_sample = fp_sample_u

    # ---- summary ----
    consistently_detected = [v for v in test_vuln if n_det(v) >= 6]
    consistently_missed = [v for v in test_vuln if n_det(v) <= 1]
    qwen_only = [v for v in test_vuln
                 if all(detect[v][d] for d in per_model if family[d] == "qwen")
                 and sum(detect[v][d] for d in per_model if family[d] == "gemma") <= 1]
    gemma_only = [v for v in test_vuln
                  if not any(detect[v][d] for d in per_model if family[d] == "qwen")
                  and sum(detect[v][d] for d in per_model if family[d] == "gemma") >= 3]

    def cwe_breakdown(eids):
        c = defaultdict(int)
        for v in eids:
            c[ds[v].get("cwe")] += 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    def cwe_breakdown_recs(recs):
        c = defaultdict(int)
        for r in recs:
            c[r.get("cwe")] += 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    analysis = dict(
        n_test_vuln=len(test_vuln),
        models={label[d]: dict(family=family[d], layer=per_model[d]["layer"],
                               thr=round(per_model[d]["thr"], 4),
                               train_f1=round(per_model[d]["f1"], 3),
                               train_prec=round(per_model[d]["prec"], 3),
                               train_rec=round(per_model[d]["rec"], 3),
                               **{k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in per_model[d]["conf"].items()},
                               n_detect_examples=sum(detect[v][d] for v in test_vuln))
                for d in per_model},
        detect_hist={k: sum(1 for v in test_vuln if n_det(v) == k) for k in range(8)},
        consistently_detected=dict(n=len(consistently_detected),
                                   cwe=cwe_breakdown(consistently_detected),
                                   eids=consistently_detected),
        consistently_missed=dict(n=len(consistently_missed),
                                 cwe=cwe_breakdown(consistently_missed),
                                 eids=consistently_missed),
        qwen_only=dict(n=len(qwen_only), cwe=cwe_breakdown(qwen_only), eids=qwen_only),
        gemma_only=dict(n=len(gemma_only), cwe=cwe_breakdown(gemma_only), eids=gemma_only),
        cwe_overall=cwe_breakdown(test_vuln),
        n_fp_spans=len(fp_corpus),
        fp_stats=dict(
            total=len(fp_corpus),
            by_kind={k: sum(1 for r in fp_corpus if r["kind"] == k)
                     for k in ("safe_alarm", "spread", "misplaced")},
            by_n_models={n: sum(1 for r in fp_corpus if r["n_models"] == n)
                         for n in range(1, 8)},
            # cross-model-shared (>=4 models) breakdown — the robust spurious patterns
            shared_ge4=dict(
                total=sum(1 for r in fp_corpus if r["n_models"] >= 4),
                by_kind={k: sum(1 for r in fp_corpus
                                if r["n_models"] >= 4 and r["kind"] == k)
                         for k in ("safe_alarm", "spread", "misplaced")},
                cwe=cwe_breakdown_recs([r for r in fp_corpus if r["n_models"] >= 4])),
            safe_alarm_cwe=cwe_breakdown_recs([r for r in fp_corpus
                                               if r["kind"] == "safe_alarm"
                                               and r["n_models"] >= 2]),
            sample_size=len(fp_sample),
            sample_by_kind={k: sum(1 for r in fp_sample if r["kind"] == k)
                            for k in ("safe_alarm", "spread", "misplaced")},
        ),
    )

    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (HERE / "fn_corpus.json").write_text(json.dumps(fn_corpus, indent=2))
    (HERE / "fp_corpus.json").write_text(json.dumps(fp_corpus, indent=2))
    (HERE / "fp_sample.json").write_text(json.dumps(fp_sample, indent=2))

    print(json.dumps({k: analysis[k] for k in
                      ("n_test_vuln", "detect_hist", "n_fp_spans")}, indent=2))
    print("\nper-model:")
    for lab in (label[d] for d in per_model):
        m = analysis["models"][lab]
        print(f"  {lab:14s} L{m['layer']:<3d} thr={m['thr']:+.3f} "
              f"tokAUC={m['tok_auc']:.3f} P={m['precision']:.2f} R={m['recall']:.2f} "
              f"TP={m['tp']} FP={m['fp']} FN={m['fn']} det_ex={m['n_detect_examples']}/{analysis['n_test_vuln']}")
    print(f"\nconsistently detected (>=6/7): {analysis['consistently_detected']['n']}  "
          f"{analysis['consistently_detected']['cwe']}")
    print(f"consistently missed (<=1/7):   {analysis['consistently_missed']['n']}  "
          f"{analysis['consistently_missed']['cwe']}")
    print(f"qwen-only: {analysis['qwen_only']['n']}  gemma-only: {analysis['gemma_only']['n']}")
    fs = analysis["fp_stats"]
    print(f"\nFP spans: {fs['total']}  by_kind={fs['by_kind']}")
    print(f"  shared>=4: {fs['shared_ge4']['total']}  by_kind={fs['shared_ge4']['by_kind']}  cwe={fs['shared_ge4']['cwe']}")
    print(f"  safe_alarm(>=2models) cwe-of-pair={fs['safe_alarm_cwe']}")
    print(f"  curated sample: {fs['sample_size']}  by_kind={fs['sample_by_kind']}")


if __name__ == "__main__":
    main()
