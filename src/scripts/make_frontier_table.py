"""Render the privacy-utility frontier table with uncertainty on every drop.

Utility comes from src/exports/tifs_a5 (or src/outputs/tifs_a5): dataset-level
mIoU over 200 VOC images, intersections and unions pooled per class before the
ratio. The drop is mIoU(clean) minus mIoU(condition) on the same images, and
its 95% interval is a paired bootstrap that resamples images and recomputes
both terms on the same resample -- the marginal interval of each mIoU is far
wider and is not the estimand a tolerance is judged against.

Privacy comes from the tifs6_a8_mse*_s2 exports: seeds 5678 and 9012 over all
400 queries, at every budget, with a query bootstrap after averaging the seeds
within a query.

The earlier seed-1234 runs are deliberately not pooled in. Their run_config
files show that two of the four budgets, MSE 5.0 and MSE 60, were produced with
a random start of 8.0 while the operating point and MSE 241.5 used 1.0, so that
sweep was not one configuration across budgets. The runs read here are 1.0
throughout. Where the earlier run does share the configuration -- MSE 15.68 and
MSE 241.5 -- it agrees closely, which is the check that the two sets are
otherwise comparable.

Admissibility against the declared tolerance is reported in three states, not
two: a cell whose interval lies within the tolerance, one whose interval
straddles it, and one whose interval lies beyond it.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from typing import Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BUDGETS = [(5.0, "tifs6_a8_mse5.0_s2"), (15.68, "tifs6_a8_mse15.68_s2"),
           (60.0, "tifs6_a8_mse60.0_s2"), (241.5, "tifs6_a8_mse241.5_s2")]
# (export condition name, printed label). The hardened arm is printed as
# "hardened" rather than "direction (EOT)": with six columns the longer label
# is what pushes the table past the text column, and the caption defines it.
CONDS = [("isotropic", "isotropic"), ("direction", "direction"),
         ("direction (EOT)", "hardened")]
CANON = {"hardened": "direction (EOT)", "hardened_direction": "direction (EOT)", "direction_eot": "direction (EOT)"}


def root_for(name: str) -> Path:
    for base in (REPO / "src" / "outputs", REPO / "src" / "exports"):
        if (base / name).is_dir():
            return base / name
    raise FileNotFoundError(name)


def miou(rows) -> float:
    inter, union = defaultdict(int), defaultdict(int)
    for r in rows:
        for cls, (i, u) in (r.get("seg_iu") or {}).items():
            inter[cls] += int(i)
            union[cls] += int(u)
    vals = [inter[c] / union[c] for c in union if union[c]]
    return float(np.mean(vals))


def utility(budget: float, rng, n=2000):
    seen = {}
    for tree in ("tifs_a5", "tifs6_a5_s2"):
        try:
            base = root_for(tree)
        except FileNotFoundError:
            continue
        match = [x for x in base.glob("segmentation_mse*")
                 if abs(float(x.name.rsplit("mse", 1)[1]) - budget) < 1e-6]
        if not match:
            continue
        for line in (match[0] / "per_image.jsonl").open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                seen[(r["image_id"], r["condition"], r.get("seed"))] = r
    grouped = defaultdict(lambda: defaultdict(list))
    for r in seen.values():
        grouped[CANON.get(r["condition"], r["condition"])][r["image_id"]].append(r)
    by = defaultdict(dict)
    for cond, per_image in grouped.items():
        for image_id, rows in per_image.items():
            # One entry per image: intersections and unions summed over the
            # seeds, which averages the seeds inside the ratio taken later.
            merged = {"seg_iu": defaultdict(lambda: [0, 0])}
            for r in rows:
                for cls, (i_val, u_val) in (r.get("seg_iu") or {}).items():
                    merged["seg_iu"][cls][0] += int(i_val)
                    merged["seg_iu"][cls][1] += int(u_val)
            merged["seg_iu"] = {k: tuple(v) for k, v in merged["seg_iu"].items()}
            by[cond][image_id] = merged
    clean = by["clean"]
    out = {"clean": (miou(list(clean.values())), None, None)}
    for cond, _ in CONDS:
        cr = by.get(cond)
        if not cr:
            continue
        ids = sorted(set(clean) & set(cr))
        point = miou([clean[i] for i in ids]) - miou([cr[i] for i in ids])
        boots = []
        for _ in range(n):
            pick = [ids[j] for j in rng.integers(0, len(ids), len(ids))]
            boots.append(miou([clean[i] for i in pick]) - miou([cr[i] for i in pick]))
        out[cond] = (miou([cr[i] for i in ids]), point, (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))))
    return out


def privacy(trees: Sequence[str], rng, n=5000):
    """Top-1 on the float release, averaged over seeds within each query.

    Only the queries every tree shares are used, so each seed contributes the
    same sample; the published run covers 200 of the 400 the later seeds cover.
    """
    per_tree = []
    for tree in trees:
        try:
            path = root_for(tree) / "serialized_release.csv"
        except FileNotFoundError:
            continue
        rows = [r for r in csv.DictReader(path.open(newline="", encoding="utf-8"))
                if r["serialisation"] == "float"]
        if rows:
            per_tree.append(rows)
    shared = set.intersection(*[{r["query_id"] for r in rows} for rows in per_tree])
    out = {}
    conds = sorted({CANON.get(r["condition"], r["condition"])
                    for rows in per_tree for r in rows})
    for cond in conds:
        by_query = defaultdict(list)
        for rows in per_tree:
            for r in rows:
                if CANON.get(r["condition"], r["condition"]) != cond:
                    continue
                if r["query_id"] in shared:
                    by_query[r["query_id"]].append(float(r["top1"]))
        if not by_query:
            continue
        v = np.array([float(np.mean(x)) for x in by_query.values()])
        boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
        out[cond] = (float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    out["_n_queries"] = len(shared)
    out["_n_seeds"] = len(per_tree)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="The tolerance declared in advance. Its scale is "
                         "calibrated by measure_segmenter_spread.py: 0.05 is "
                         "the widest gap between two adjacent published "
                         "segmenters on these same images.")
    ap.add_argument("--strict-tolerance", dest="strict_tolerance", type=float,
                    default=0.016,
                    help="A second, stricter tolerance, reported alongside the "
                         "first so a reader can see which verdicts depend on "
                         "the choice. The default is the median gap between "
                         "adjacent published segmenters on these images.")
    ap.add_argument("--output", default=str(REPO / "paper" / "generated" / "tab_frontier.tex"))
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tol, strict = args.tolerance, args.strict_tolerance

    blocks, clean_vals = [], []
    n_q = 0
    for budget, tree in BUDGETS:
        u, p = utility(budget, rng), privacy([tree], rng)
        n_q = p.pop("_n_queries", n_q)
        p.pop("_n_seeds", None)
        clean_vals.append(u["clean"][0])
        rows = []
        for cond, label in CONDS:
            if cond not in u or cond not in p:
                continue
            m, drop, (lo, hi) = u[cond]
            t1, t1lo, t1hi = p[cond]
            verdicts = [
                r"$\checkmark$" if hi <= t else (r"$\sim$" if lo < t else "---")
                for t in (tol, strict)
            ]
            rows.append(
                rf"{label:<15} & {t1:.3f} [{t1lo:.3f},{t1hi:.3f}] & {m:.4f}"
                rf" & {drop:.4f} [{lo:.3f},{hi:.3f}]"
                rf" & {verdicts[0]} & {verdicts[1]} \\")
        blocks.append((budget, rows))
    clean = float(np.mean(clean_vals))

    lines = [
        "% Generated by src/scripts/make_frontier_table.py from tifs_a5 (utility) and",
        "% the tifs6_a8_mse*_s2 privacy trees. Do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Privacy against measured downstream utility over four delivered-MSE",
        rf"budgets. Clean mIoU is ${clean:.4f}$ throughout. Utility averages three",
        rf"seeds over 200 images; privacy averages two over {n_q} queries, from the",
        r"runs that share one optimiser configuration at every budget. Drop is",
        r"mIoU lost on those images, with a",
        r"paired image-bootstrap 95\% interval; Top-1 carries a query-bootstrap",
        r"interval. ``Hardened'' is the direction optimised in expectation over",
        r"attacker-side transforms. A drop interval lies within a tolerance",
        r"($\checkmark$), straddles it ($\sim$), or lies beyond it (---). The",
        rf"declared tolerance is ${tol:.2f}$ and ${strict:.3f}$ is the stricter",
        r"alternative of \S\ref{sec:direction}; both are calibrated against the",
        r"spread between published segmenters on these images. Clean Top-1",
        r"for this attacker is $0.197$. Retrieval is on MSLS and utility on VOC,",
        r"paired by budget, not by image.}",
        r"\label{tab:frontier}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"& \multicolumn{1}{c}{Top-1 $\downarrow$} & \multicolumn{1}{c}{mIoU $\uparrow$}"
        r" & & \multicolumn{2}{c}{Admissible} \\",
        rf"Condition & [95\% CI] & & Drop [95\% CI] & ${tol:.2f}$ & ${strict:.3f}$ \\",
        r"\hline",
    ]
    for budget, rows in blocks:
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{delivered MSE {budget:g}}}}} \\")
        lines += rows
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
