# [ai-generated]
"""exp-27 — print the RESULTS.md markdown tables from results/*.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEM = ("CWE-125", "CWE-416", "CWE-476")
INJ = ("CWE-089", "CWE-078", "CWE-022", "CWE-079")
UNTRUSTED = ("CWE-190", "CWE-787")
REG = ("allclean", "conly", "matchedpatch")


def fmt(cell, ci=True):
    a = cell["auc"]
    if a != a:
        return "nan"
    s = f"{a:.3f}"
    if ci and "ci" in cell and cell["ci"][0] == cell["ci"][0]:
        s += f" [{cell['ci'][0]:.3f},{cell['ci'][1]:.3f}]"
    return s


def p25(row, rg):
    p = row["exp25_probe"][rg]
    return f"**{p['auc']:.3f} [{p['ci'][0]:.3f},{p['ci'][1]:.3f}]**"


def main():
    for slug in ("qwen32b", "gemma1b"):
        d = json.load(open(HERE / f"results/exp27_{slug}_axis.json"))
        rows = d["rows"]
        print(f"\n### {slug} axis — memory CWEs (probe = exp-25 specialized, cited)\n")
        print("| CWE | regime | probe (exp-25) | char-ngram | combined | "
              "unigram | conlytr-char | conlytr-comb | probeG |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c in MEM:
            r = rows[c]
            for rg in REG:
                cell = r["regimes"][rg]
                print(f"| {c} (n={r['n_test_vuln_ex']}) | {rg} | {p25(r, rg)} | "
                      f"{fmt(cell['char_ngram_lr'])} | {fmt(cell['combined_abd_lr'])} | "
                      f"{fmt(cell['token_unigram_lr'])} | "
                      f"{fmt(cell['conlytrained_char_ngram_lr'])} | "
                      f"{fmt(cell['conlytrained_combined_abd_lr'])} | "
                      f"{fmt(cell['probe_general'])} |")
        print(f"\n### {slug} axis — injection positive controls (matchedpatch only)\n")
        print("| CWE | probe (exp-25) | char-ngram | combined | conlytr-char |")
        print("|---|---|---|---|---|")
        for c in INJ:
            r = rows[c]
            cell = r["regimes"]["matchedpatch"]
            print(f"| {c} (n={r['n_test_vuln_ex']}) | {p25(r, 'matchedpatch')} | "
                  f"{fmt(cell['char_ngram_lr'])} | {fmt(cell['combined_abd_lr'])} | "
                  f"{fmt(cell['conlytrained_char_ngram_lr'], ci=False)} |")
        print(f"\n### {slug} axis — untrusted cells (n<10), matchedpatch\n")
        for c in UNTRUSTED:
            if c not in rows:
                continue
            r = rows[c]
            cell = r["regimes"]["matchedpatch"]
            print(f"- {c} (n={r['n_test_vuln_ex']}): char {fmt(cell['char_ngram_lr'])} "
                  f"vs probe {p25(r, 'matchedpatch')}")
        # secondary mp columns for memory
        print(f"\n### {slug} secondary surface columns, memory × matchedpatch\n")
        for c in MEM:
            cell = rows[c]["regimes"]["matchedpatch"]
            print(f"- {c}: unigram {fmt(cell['token_unigram_lr'])}, "
                  f"keyword-LR {fmt(cell['keyword_lr'])}, "
                  f"keyword-untrained {fmt(cell['keyword_untrained'])}, "
                  f"lang-indicator {fmt(cell['lang_indicator'], ci=False)}")
        # paired probeG - surface deltas
        print(f"\n### {slug} paired probeG−surface Δ (same resamples), memory\n")
        print("| CWE | regime | variant | Δ 95% CI | frac Δ>0 |")
        print("|---|---|---|---|---|")
        for c in MEM:
            for rg in ("conly", "matchedpatch"):
                pair = rows[c]["regimes"][rg].get("probeG_minus_surface_delta")
                if not pair:
                    continue
                for nm, v in pair.items():
                    if v[0] != v[0]:
                        continue
                    print(f"| {c} | {rg} | {nm} | [{v[0]:+.3f},{v[1]:+.3f}] | {v[3]:.3f} |")


if __name__ == "__main__":
    main()
