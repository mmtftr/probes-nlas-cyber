"""Hand-curated eval: run the probe directly on known-good and known-bad code
snippets to see if the trained probe actually discriminates outside its
training distribution.

This is the demo-aligned eval: if the probe doesn't put red on the SQL
injection snippet and green on the parameterised version, the demo won't
work no matter how nice the UI is.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"
PROBE_PATH = Path("data/probe.npz")


def _sigmoid(x):
    if x >= 0: return 1.0 / (1.0 + float(np.exp(-x)))
    e = float(np.exp(x)); return e / (1.0 + e)


def main():
    npz = np.load(PROBE_PATH)
    w, b, layer = npz["w"].astype(np.float32), float(npz["b"]), int(npz["layer"])
    print(f"probe: layer={layer}  dim={len(w)}  b={b:.3f}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, attn_implementation="eager").to(device).eval()

    samples = [
        # (label, name, code)
        ("VULN", "sql-injection",
            "function getUserById(id) {\n  return db.query(`SELECT * FROM users WHERE id=${id}`);\n}"),
        ("SAFE", "sql-parameterised",
            "function getUserById(id) {\n  return db.query('SELECT * FROM users WHERE id=?', [id]);\n}"),
        ("VULN", "path-traversal",
            "function readUserFile(filename) {\n  return fs.readFileSync('/var/data/' + filename, 'utf8');\n}"),
        ("SAFE", "path-safe",
            "function readUserFile(filename) {\n  const safe = path.resolve('/var/data', path.basename(filename));\n  if (!safe.startsWith('/var/data/')) throw new Error('bad path');\n  return fs.readFileSync(safe, 'utf8');\n}"),
        ("VULN", "cmd-inject",
            "function runDiag(host) {\n  return execSync(`ping -c 1 ${host}`).toString();\n}"),
        ("SAFE", "cmd-safe",
            "function runDiag(host) {\n  if (!/^[a-z0-9.-]+$/.test(host)) throw new Error('bad host');\n  return execFileSync('ping', ['-c', '1', host]).toString();\n}"),
        ("VULN", "weak-random",
            "function newToken() { return Math.random().toString(36).slice(2); }"),
        ("SAFE", "strong-random",
            "function newToken() { return crypto.randomBytes(32).toString('hex'); }"),
        ("VULN", "md5-pwd",
            "function hashPassword(pw) { return crypto.createHash('md5').update(pw).digest('hex'); }"),
        ("SAFE", "argon2-pwd",
            "async function hashPassword(pw) { return await argon2.hash(pw, { type: argon2.argon2id }); }"),
        ("NEUTRAL", "plain-greeting",
            "Hello! How can I help you today?"),
        ("NEUTRAL", "markdown-doc",
            "### Prerequisites\n\nFirst, make sure you have the necessary packages installed:"),
    ]

    print(f"\n{'TAG':8s}  {'NAME':22s}  {'RISK %':>7s}  TOKEN_COUNT")
    print("-" * 60)
    with torch.inference_mode():
        for tag, name, code in samples:
            ids = tok.encode(code, return_tensors="pt").to(device)
            out = model(ids, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[layer + 1][0, -1, :].detach().to("cpu").float().numpy()
            risk = _sigmoid(float(np.dot(w, h) + b))
            n = ids.shape[1]
            print(f"{tag:8s}  {name:22s}  {risk*100:6.1f}   {n}")


if __name__ == "__main__":
    main()
