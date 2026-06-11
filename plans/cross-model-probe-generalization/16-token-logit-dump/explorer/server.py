# [ai-generated]
"""Logit explorer — browse exp-16 span-max probe predictions over the SVEN dataset.

Serves a single-page UI to browse dataset samples with per-token probe logits,
defaulting to the Platt-calibrated decision threshold. Selectors: model variant,
layer (= the per-layer trained probe), token/example view. Threshold + Platt
(T, a) are adjustable; all colouring is recomputed client-side from raw logits.

Zero new deps: stdlib http.server + numpy (already a project dep).

Data sources (all under this experiment dir / repo data/):
  results/logitdump_<MODEL>/
    metrics_logitdump.json        — per-layer AUCs, best layer
    example_scores_layer<NN>.json — per-example {eid, score, logit_max, label, cwe, lang, is_test}
    logits_layer<NN>.npz          — per-token {logit, prob, y, example_id, char_start, char_end, is_test, is_code}
  <repo>/data/dataset.jsonl       — eid-th line is example eid: {code, label, cwe, lang, token_labels{evidence,...}, _func_name, _file_name}

Run:  uv run python server.py [--host 0.0.0.0] [--port 8011]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict, defaultdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "results"
# repo root = .../probes-nlas-cyber ; this file is at plans/.../16-token-logit-dump/explorer/
REPO_ROOT = HERE.parents[3]
DATASET = REPO_ROOT / "data" / "dataset.jsonl"
INDEX_HTML = HERE / "index.html"

# Canonical Platt defaults — from the gemmaforge production probe (F1-max on
# Platt-calibrated heldout scores). NOT refit per exp-16 probe; surfaced as a
# caveat in the UI and adjustable via sliders.
DEFAULT_T = 1.794
DEFAULT_A = -0.269
DEFAULT_THRESHOLD = 0.929082

_DIR_RE = re.compile(r"^logitdump_(.+)$")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_dataset() -> list[dict]:
    rows: list[dict] = []
    with open(DATASET) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def load_calibration() -> dict:
    p = HERE / "calibration.json"
    if p.is_file():
        return json.loads(p.read_text())
    return {"production": {"T": DEFAULT_T, "a": DEFAULT_A, "threshold": DEFAULT_THRESHOLD}}


@lru_cache(maxsize=1)
def discover_models() -> dict:
    """Return {key: {dir, name, layers:[int], best_layer, metrics:{layer:{...}}}}."""
    out: dict[str, dict] = {}
    if not RESULTS_DIR.is_dir():
        return out
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = _DIR_RE.match(d.name)
        if not m:
            continue
        key = m.group(1)
        layers = sorted(
            int(p.stem.split("layer")[-1])
            for p in d.glob("example_scores_layer*.json")
        )
        if not layers:
            continue
        metrics_path = d / "metrics_logitdump.json"
        name, best_layer, per_layer = key, layers[0], {}
        if metrics_path.is_file():
            mj = json.loads(metrics_path.read_text())
            name = mj.get("model", key)
            best_layer = mj.get("best_layer", layers[0])
            for ly in mj.get("layers", []):
                per_layer[int(ly["layer"])] = {
                    k: ly.get(k)
                    for k in (
                        "test_tokens_auc",
                        "test_tokens_code_auc",
                        "test_example_auc",
                        "val_ex_auc",
                        "n_tokens",
                        "n_tokens_code",
                    )
                }
        # layers with token-level logits present on disk (gitignored heavy npz)
        has_tokens = {
            ly for ly in layers if (d / f"logits_layer{ly:02d}.npz").is_file()
        }
        out[key] = {
            "dir": str(d),
            "name": name,
            "layers": layers,
            "best_layer": best_layer,
            "metrics": per_layer,
            "token_layers": sorted(has_tokens),
        }
    return out


@lru_cache(maxsize=1)
def build_pairs() -> list[dict]:
    """SVEN before/after pairs: examples sharing (_file_name, _func_name), one
    vulnerable + one fixed. Multi-version groups are zipped vuln[i]<->safe[i]
    in dataset order. Pairs are split-stable (grouped split keeps them together)."""
    ds = load_dataset()
    groups: dict[tuple, list[int]] = defaultdict(list)
    for eid, r in enumerate(ds):
        groups[(r.get("_file_name"), r.get("_func_name"))].append(eid)
    pairs = []
    for (f, fn), eids in groups.items():
        vs = [e for e in eids if ds[e].get("label") == 1]
        ss = [e for e in eids if ds[e].get("label") == 0]
        for i in range(min(len(vs), len(ss))):
            ve = ds[vs[i]]
            pairs.append({
                "pair_key": f"{f}::{fn}#{i}",
                "file": f, "func": fn,
                "lang": ve.get("lang"), "cwe": ve.get("cwe"),
                "vuln_eid": vs[i], "safe_eid": ss[i],
            })
    return pairs


def pairs_for(key: str, layer: int) -> list[dict]:
    es = {e["eid"]: e for e in _example_scores(key, layer)}
    out = []
    for p in build_pairs():
        v, s = es.get(p["vuln_eid"]), es.get(p["safe_eid"])
        if v is None or s is None:
            continue
        out.append({
            **p,
            "is_test": bool(v.get("is_test")),
            "vuln": {"eid": p["vuln_eid"], "logit_max": v["logit_max"]},
            "safe": {"eid": p["safe_eid"], "logit_max": s["logit_max"]},
        })
    return out


def _example_scores(key: str, layer: int) -> list[dict]:
    info = discover_models().get(key)
    if not info:
        raise KeyError(f"unknown model {key!r}")
    path = Path(info["dir"]) / f"example_scores_layer{layer:02d}.json"
    if not path.is_file():
        raise KeyError(f"no example_scores for {key} layer {layer}")
    return json.loads(path.read_text())


# small LRU over loaded token-logit npz arrays (each ~13 MB resident)
class _NpzCache:
    def __init__(self, maxsize: int = 4):
        self.maxsize = maxsize
        self._d: OrderedDict[tuple, dict] = OrderedDict()

    def get(self, key: str, layer: int):
        ck = (key, layer)
        if ck in self._d:
            self._d.move_to_end(ck)
            return self._d[ck]
        info = discover_models().get(key)
        if not info:
            raise KeyError(f"unknown model {key!r}")
        path = Path(info["dir"]) / f"logits_layer{layer:02d}.npz"
        if not path.is_file():
            raise KeyError(f"no token logits on disk for {key} layer {layer}")
        z = np.load(path, allow_pickle=False)
        arrs = {
            "logit": z["logit"],
            "y": z["y"],
            "example_id": z["example_id"],
            "char_start": z["char_start"],
            "char_end": z["char_end"],
            "is_code": z["is_code"],
        }
        self._d[ck] = arrs
        self._d.move_to_end(ck)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)
        return arrs


_NPZ = _NpzCache()


def example_tokens(key: str, layer: int, eid: int) -> list[dict]:
    arrs = _NPZ.get(key, layer)
    mask = arrs["example_id"] == eid
    s = arrs["char_start"][mask]
    e = arrs["char_end"][mask]
    logit = arrs["logit"][mask]
    y = arrs["y"][mask]
    is_code = arrs["is_code"][mask]
    order = np.argsort(s, kind="stable")
    toks = []
    for i in order:
        cs, ce = int(s[i]), int(e[i])
        if ce <= cs:  # skip zero-width (BOS / template) tokens
            continue
        toks.append(
            {
                "s": cs,
                "e": ce,
                "logit": round(float(logit[i]), 4),
                "y": int(y[i]),
                "code": bool(is_code[i]),
            }
        )
    return toks


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/config":
                self._json(
                    {
                        "T": DEFAULT_T,
                        "a": DEFAULT_A,
                        "threshold": DEFAULT_THRESHOLD,
                        "platt_source": "gemmaforge production probe (F1-max on "
                        "Platt-calibrated heldout scores); not refit per exp-16 probe",
                    }
                )
            elif u.path == "/api/models":
                self._json(discover_models())
            elif u.path == "/api/calibration":
                self._json(load_calibration())
            elif u.path == "/api/examples":
                self._json(_example_scores(q["model"], int(q["layer"])))
            elif u.path == "/api/pairs":
                self._json(pairs_for(q["model"], int(q["layer"])))
            elif u.path == "/api/example":
                key, layer, eid = q["model"], int(q["layer"]), int(q["eid"])
                ds = load_dataset()
                if not (0 <= eid < len(ds)):
                    self._json({"error": f"eid {eid} out of range"}, 404)
                    return
                row = ds[eid]
                tl = row.get("token_labels") or {}
                self._json(
                    {
                        "eid": eid,
                        "code": row.get("code", ""),
                        "label": row.get("label"),
                        "cwe": row.get("cwe"),
                        "lang": row.get("lang"),
                        "func": row.get("_func_name"),
                        "file": row.get("_file_name"),
                        "source": row.get("source"),
                        "evidence": tl.get("evidence", []),
                        "vulnerable_line": tl.get("vulnerable_line", []),
                        "tokens": example_tokens(key, layer, eid),
                    }
                )
            else:
                self._json({"error": "not found"}, 404)
        except KeyError as ex:
            self._json({"error": str(ex)}, 404)
        except Exception as ex:  # surface, don't swallow
            self._json({"error": f"{type(ex).__name__}: {ex}"}, 500)

    do_HEAD = do_GET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8011)
    args = ap.parse_args()

    models = discover_models()
    print(f"[explorer] dataset: {DATASET} ({len(load_dataset())} examples)")
    print(f"[explorer] models: {len(models)} variants")
    for k, v in models.items():
        tl = v["token_layers"]
        print(f"  - {v['name']}: layers {v['layers']} (token logits: {tl or 'none'})")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[explorer] serving on http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
