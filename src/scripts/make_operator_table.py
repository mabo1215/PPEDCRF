"""The operator arm, tabulated.

The manuscript's operator section says its three results are ``tabulated per
operator in the Supplementary Material''. They were not: the supplement
carried the operator *definitions* and no results table, so a referee checking
the claim that the null survives every operator at the operating point, and
that the placement effect reverses sign between operators at a large budget,
had a sentence and nowhere to check it. This writes that table.

Nothing is re-run. The rows are the committed per-query exports of the
operator study, read under the protocol's own unit of inference: the three
seeds are averaged within a query before pairing, so each cell is 400 paired
differences rather than 1,200, and the interval is bootstrapped over the 277
place clusters the manifest labels.
"""
from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "scripts"))
from _pvalue import fmt_p  # noqa: E402
from analyze_placement_equivalence import tost  # noqa: E402
from make_msls_placement_table import boot, per_query  # noqa: E402

# The four operators of the manuscript's operator section, in the order it
# introduces them. The export names each run "<sigma><operator>".
OPERATORS = [
    ("gaussian", "additive Gaussian"),
    ("correlated", "correlated Gaussian"),
    ("blur", "selective low-pass"),
    ("mosaic", "block quantisation"),
]

# The two budgets the study was run at: the retrieval operating point, and the
# large budget where the manuscript reports the sign reversal.
BUDGETS = [
    ("sigma8", r"Operating point, $\sigma_0=8$"),
    ("sigma32", r"Large budget, $\sigma_0=32$ ($15.4\times$ the delivered MSE)"),
]


def cell(rows: Sequence[dict], place_of: Dict[str, str], margin: float):
    """Edge placement against the uniform control, under both units."""
    ref = per_query(rows, "uniform")
    cur = per_query(rows, "edge")
    if not ref or not cur:
        return None
    qs = sorted(set(cur) & set(ref))
    d = np.array([cur[q] - ref[q] for q in qs])
    places = [place_of.get(q, q) for q in qs]
    nz = d[d != 0]
    # The paper's one signed-rank convention: zeros dropped, discordant pairs
    # enumerated, so this column means the same thing it means in the
    # placement tables.
    p = float(wilcoxon(nz).pvalue) if nz.size else 1.0
    return {
        "ref": float(np.mean([ref[q] for q in qs])),
        "edge": float(np.mean([cur[q] for q in qs])),
        "delta": float(d.mean()),
        "ci": boot(d, places),
        "p": p,
        "disc": int(nz.size),
        "tost": tost(d, places, margin, 10000, 1234),
        "n": len(qs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", default=str(REPO / "src/exports/operator_study"))
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"))
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--out",
                    default=str(REPO / "paper/generated/tab_operators.tex"))
    args = ap.parse_args()

    place_of = {}
    with open(args.places, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            place_of[r["query_id"]] = r["correct_place"]

    study = Path(args.study)
    body: List[str] = []
    n_q = n_p = 0
    for prefix, heading in BUDGETS:
        block: List[str] = []
        for key, label in OPERATORS:
            path = study / f"{prefix}_{key}" / "per_query.csv"
            if not path.is_file():
                continue
            rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
            got = cell(rows, place_of, args.margin)
            if got is None:
                continue
            if not n_q:
                n_q = got["n"]
                n_p = len({place_of.get(q, q) for q in per_query(rows, "uniform")})
            inside = max(abs(got["ci"][0]), abs(got["ci"][1])) <= args.margin
            verdict = "negligible" if inside else (
                "separates" if got["ci"][0] * got["ci"][1] > 0 else "none det.")
            block.append(
                rf"{label} & {got['ref']:.4f} & {got['edge']:.4f} & "
                rf"${got['delta']:+.4f}$ & "
                rf"[{got['ci'][0]:+.3f},{got['ci'][1]:+.3f}] & "
                rf"{fmt_p(got['p'])}\,({got['disc']}) & {got['tost']:.3f} & "
                rf"{verdict} \\")
        if block:
            body.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{heading}}}}} \\")
            body.extend(block)
            body.append(r"\hline")

    if not body:
        raise SystemExit("no operator rows found")
    body = body[:-1]

    caption = (
        rf"The operator arm: what the budget is spent on, with where it goes "
        rf"held fixed. Each row releases the most concentrated placement rule "
        rf"(edge magnitude) and the uniform control under one operator at the "
        rf"same delivered MSE, solved per frame by bisection through the pixel "
        rf"clamp, on the primary place-labelled manifest --- {n_q} query "
        rf"clusters over {n_p} places, three seeds averaged within a query "
        rf"before pairing. $\Delta$ is edge minus uniform, so ``positive means "
        rf"worse privacy''; the interval is bootstrapped over places, the "
        rf"figure beside $p$ counts discordant pairs, and "
        rf"$p_{{\mathrm{{TOST}}}}$ is the equivalence test at "
        rf"$\pm{args.margin:.2f}$. At the operating point no operator lets "
        rf"placement separate, so the null is a property of the regime rather "
        rf"than of the additive-Gaussian operator this family inherits; at the "
        rf"large budget the effect reverses sign between operators --- edges "
        rf"help under additive noise, hurt under low-pass and block "
        rf"quantisation --- so a rule tuned on one can harm another. The "
        rf"uniform column carries the threefold spread at that budget.")

    lines = [
        "% Generated by src/scripts/make_operator_table.py from",
        "% src/exports/operator_study/. Do not edit by hand.",
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:operators}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.2pt}",
        # Eight columns do not fit the IEEEtran column even at scriptsize,
        # and the document's hand-written wide tables solve it the same way.
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Operator & uniform & edge & $\Delta$ & place 95\% CI & $p$ "
        r"& $p_{\mathrm{TOST}}$ & Verdict \\",
        r"\hline",
        *body,
        r"\hline", r"\end{tabular}%", r"}", r"\end{table}",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": write_text otherwise uses the platform separator, so
    # regenerating on Windows rewrites every line of a committed LF table and
    # buries the real change in the diff.
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[done] wrote {out}")
    for line in body:
        print("  " + line.replace(r"\\", "").replace("&", " ").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
