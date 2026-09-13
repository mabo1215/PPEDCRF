"""The two unfreeze budgets, pooled over seeds.

Cells are pooled across the four seeds rather than printed per seed. At one
seed the split is 100 held-out queries and individual cells swing far enough to
flip sign, so a per-seed table would invite exactly the reading the pooling
exists to prevent. The seed count is stated so the pooling is visible.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import re
import sys
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "scripts"))
from make_msls_placement_table import boot  # noqa: E402

EXPOSURES = [("isotropic", "isotropic noise"),
             ("direction", "the direction release"),
             ("hardened_direction", "the hardened release")]
LABEL = {"stock": "fixed index", "rebuilt": "re-indexed"}


def per_query(path, cond):
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        if r["condition"] == cond:
            acc[r["query_id"]].append(1.0 if int(r["correct_rank"]) == 1 else 0.0)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(REPO / "src/outputs/n3_eval"))
    ap.add_argument("--condition", default="transfer_3")
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"))
    ap.add_argument("--out",
                    default=str(REPO / "paper/generated/tab_unfreeze.tex"))
    args = ap.parse_args()

    place_of = {}
    if Path(args.places).is_file():
        for r in csv.DictReader(open(args.places, newline="", encoding="utf-8")):
            place_of[r["query_id"]] = r["correct_place"]

    runs = {}
    for p in glob.glob(str(Path(args.root) / "n3_*.csv")):
        m = re.match(r".*n3_k(\d)_(?:s(\d+)_)?(isotropic|direction|hardened_direction)_(stock|rebuilt)\.csv$",
                     p.replace("\\", "/"))
        if m:
            runs[(int(m.group(1)), m.group(2) or "1234", m.group(3), m.group(4))] = p
    base = {}
    for p in glob.glob(str(Path(args.root) / "baseline_*.csv")):
        m = re.search(r"baseline_s(\d+)", Path(p).stem)
        base[m.group(1) if m else "1234"] = p
    seeds = sorted({k[1] for k in runs})
    print(f"{len(runs)} runs over seeds {', '.join(seeds)}")

    lines = []
    for mode in ("stock", "rebuilt"):
        lines.append(r"\multicolumn{6}{l}{\textit{" + LABEL[mode] + r"}} \\")
        for exp, pretty in EXPOSURES:
            d_all, k1s, k5s, bs = [], [], [], []
            for sd in seeds:
                a, b = runs.get((1, sd, exp, mode)), runs.get((5, sd, exp, mode))
                if not a or not b or sd not in base:
                    continue
                x1, x5 = per_query(a, args.condition), per_query(b, args.condition)
                bb = per_query(base[sd], args.condition)
                qs = sorted(set(x1) & set(x5) & set(bb))
                if not qs:
                    continue
                k1s.append(np.mean([x1[q] for q in qs]))
                k5s.append(np.mean([x5[q] for q in qs]))
                bs.append(np.mean([bb[q] for q in qs]))
                d_all.append(([x5[q] - x1[q] for q in qs],
                              [place_of.get(q, q) for q in qs]))
            if not d_all:
                continue
            d = np.concatenate([np.array(x[0]) for x in d_all])
            g = [q for x in d_all for q in x[1]]
            ci = boot(d, g)
            pos = sum(1 for x in d_all if np.mean(x[0]) > 0)
            n = len(d_all)
            p = sum(comb(n, i) for i in range(pos, n + 1)) / 2 ** n
            lines.append(f"\\quad adapted to {pretty} & {np.mean(bs):.3f} & "
                         f"{np.mean(k1s):.3f} & {np.mean(k5s):.3f} & "
                         f"${d.mean():+.3f}$ & [{ci[0]:+.3f},{ci[1]:+.3f}] "
                         f"({pos}/{n}) \\\\")
            print(f"  {mode:8s} {exp:20s} base={np.mean(bs):.3f} "
                  f"k1={np.mean(k1s):.3f} k5={np.mean(k5s):.3f} "
                  f"d={d.mean():+.3f} {pos}/{n} p={p:.5f}")

    caption = (
        r"The adaptive attacker at two unfreeze budgets, under the directional "
        r"release. $k{=}1$ fine-tunes the last residual block, the setting "
        r"the main text reports; $k{=}5$ unfreezes every residual stage and "
        r"the stem, the only fully-unfrozen setting. Both budgets were run in "
        r"one batch under one protocol rather than compared across runs: each "
        r"adapted model is scored on the held-out split its own checkpoint was "
        f"trained against and differenced against an unadapted baseline on that "
        f"same split, over {len(seeds)} seeds. Cells pool the seeds, because at "
        r"one seed the split is 100 queries and individual cells swing far "
        r"enough to flip sign; the figure in parentheses counts how many of the "
        r"seeds moved in the direction of the pooled mean, and the interval is "
        r"bootstrapped over places. The result is not that capacity helps: with "
        r"the gallery held at the pretrained index the whole encoder buys "
        r"nothing, and the same capacity becomes worth about half of what the "
        r"perturbation removed only once the attacker may also re-embed the "
        r"gallery.")

    out = ["% Generated by src/scripts/make_unfreeze_table.py. Do not edit.",
           r"\begin{table}[t]", r"\centering",
           r"\caption{" + caption + "}", r"\label{tab:unfreeze}",
           r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
           r"\resizebox{\columnwidth}{!}{%",
           r"\begin{tabular}{lccccc}", r"\hline",
           r"Attacker & unadapted & $k{=}1$ & $k{=}5$ & $\Delta$ & place 95\% CI (seeds) \\",
           r"\hline", *lines, r"\hline", r"\end{tabular}%", r"}",
           r"\end{table}", ""]
    Path(args.out).write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
