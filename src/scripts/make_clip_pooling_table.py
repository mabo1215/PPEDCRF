"""Render the clip-pooling table the manuscript prints, from the raw rows.

The question the table has to answer in one glance is whether an attacker who
holds the clip recovers what the single-frame numbers report. So it is laid out
by clip length, with the isotropic control beside the directional arms at every
length, and the paired contrast carried on the place-clustered unit this
protocol prescribes.

Also exports the slim per-query rows the released artifact carries, and fails
loudly if a condition present in the run is missing from the table -- the same
completeness check the second dataset's generator runs, for the same reason:
a condition that is run but not printed is invisible to every other check.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
CONTROL = "isotropic"
LABEL = {"isotropic": "isotropic control", "direction": "direction",
         "hardened": "hardened direction", "white_box": "white-box bound"}
ORDER = ["isotropic", "direction", "hardened", "white_box"]
POOL_LABEL = {"first": "first frame", "mean": "mean", "max": "max",
              "best_frame": "best frame"}


def places(d6_dir: Path) -> dict:
    out = {}
    for path in sorted(glob.glob(str(d6_dir / "*.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "correct_place" not in (reader.fieldnames or []):
                continue
            for row in reader:
                out.setdefault(row["query_id"], row["correct_place"])
    return out


def boot(diff, ids, n, rng):
    uniq, inv = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    stats = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        stats[b] = diff[np.concatenate([groups[i] for i in pick])].mean()
    return np.percentile(stats, [2.5, 97.5])


def p_str(p: float) -> str:
    return "$<$0.001" if p < 5e-4 else f"{p:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default=str(REPO / "src" / "outputs" /
                                          "clip_pooling" / "clip_shard*.csv"))
    ap.add_argument("--d6_dir", default=str(REPO / "src" / "exports" / "tifs_d6"))
    ap.add_argument("--export", default=str(REPO / "src" / "exports" /
                                            "clip_pooling" / "per_query.csv"))
    ap.add_argument("--out", default=str(REPO / "paper" / "generated" /
                                         "tab_clip_pooling.tex"))
    ap.add_argument("--poolings", nargs="+", default=["mean", "best_frame"])
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    # A row can appear twice: ten shards were relaunched after dying early and
    # their replacements recomputed a few groups. The recomputation is not
    # bit-identical -- the backward pass through the surrogates is not
    # deterministic -- so these are two draws of the same condition rather
    # than copies, and the export keeps the first.
    raw, cells, seen = [], defaultdict(lambda: defaultdict(list)), set()
    for path in sorted(glob.glob(args.rows)):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (r["query_id"], r["condition"], r["seed"],
                       r["clip_len"], r["pooling"])
                if key in seen:
                    continue
                seen.add(key)
                raw.append(r)
                cells[(r["condition"], int(r["clip_len"]), r["pooling"])][
                    r["query_id"]].append(float(int(r["correct_rank"]) == 1))
    if not raw:
        print(f"[FAIL] no rows matched {args.rows}")
        return 1
    cells = {k: {q: float(np.mean(v)) for q, v in d.items()}
             for k, d in cells.items()}
    # Restrict to the queries that reach every clip length, so a Top-1 that
    # rises with k is the attacker gaining rather than the sample changing.
    common = set.intersection(*[set(v) for v in cells.values()])
    cells = {k: {q: v for q, v in d.items() if q in common}
             for k, d in cells.items()}

    ran = {k[0] for k in cells}
    listed = set(ORDER)
    missing = sorted(ran - listed)
    if missing:
        print(f"[FAIL] run but not in the table: {', '.join(missing)}")
        return 1

    Path(args.export).parent.mkdir(parents=True, exist_ok=True)
    keep = ["query_id", "condition", "seed", "clip_len", "pooling",
            "correct_rank", "correct_place", "mean_mse"]
    with open(args.export, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        for r in raw:
            w.writerow(r)
    print(f"[export] {len(raw)} rows -> {args.export}")

    pm = places(Path(args.d6_dir))
    lengths = sorted({k[1] for k in cells})
    conds = [c for c in ORDER if c in ran]
    rng = np.random.default_rng(0)
    lines, stats = [], []
    # One row per (pooling, condition), one column per clip length: the
    # question is whether Top-1 rises with k, and a row is where a reader can
    # see that. The paired contrast is carried at the longest clip, which is
    # the case that decides it.
    for pooling in args.poolings:
        lines.append(r"\hline")
        lines.append(rf"\multicolumn{{{len(lengths) + 3}}}{{l}}{{\textit{{pooling: "
                     rf"{POOL_LABEL.get(pooling, pooling)}}}}} \\")
        for cond in conds:
            tops = []
            for k in lengths:
                arm = cells.get((cond, k, pooling))
                tops.append(f"{np.mean(list(arm.values())):.4f}" if arm
                            else "---")
            arm = cells.get((cond, max(lengths), pooling))
            ref = cells.get((CONTROL, max(lengths), pooling))
            if cond == CONTROL or not arm or not ref:
                lines.append(rf"{LABEL[cond]:<18} & " + " & ".join(tops)
                             + r" & --- & --- \\")
                stats.append({"condition": cond, "pooling": pooling,
                              "top1": [float(t) for t in tops if t != "---"]})
                continue
            qs = sorted(set(arm) & set(ref) & set(pm))
            d = np.array([arm[q] - ref[q] for q in qs])
            # A stream per cell, seeded from the cell's own name: an interval
            # is then reproducible on its own rather than only as part of the
            # sequence of draws that produced the whole table.
            cell_rng = np.random.default_rng(
                zlib.crc32(f"{pooling}|{cond}".encode()) & 0x7FFFFFFF)
            lo, hi = boot(d, np.array([pm[q] for q in qs]), args.n_boot,
                          cell_rng)
            try:
                pv = float(wilcoxon(d).pvalue)
            except ValueError:
                pv = 1.0
            lines.append(rf"{LABEL[cond]:<18} & " + " & ".join(tops)
                         + rf" & ${d.mean():+.4f}$ & $[{lo:+.3f},{hi:+.3f}]$ \\")
            stats.append({"condition": cond, "pooling": pooling,
                          "top1": [float(t) for t in tops if t != "---"],
                          "delta": float(d.mean()), "ci": [lo, hi], "p": pv,
                          "n": len(qs)})

    any_cell = next(iter(cells.values()))
    n_q = len(any_cell)
    n_places = len({pm[q] for q in any_cell if q in pm})
    head = [
        "% Generated by src/scripts/make_clip_pooling_table.py from",
        "% src/exports/clip_pooling. Do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{An attacker holding the clip. Every frame is released under",
        r"the same condition, optimiser, seed and delivered distortion, and the",
        r"attacker pools $k$ of them before ranking the same 2,000-image",
        rf"gallery ({n_q} queries over {n_places} places, 3 seeds). The $k=1$",
        r"column is the single-frame result. $\Delta$ is against the isotropic",
        r"control at $k=7$ under the same pooling, so it isolates what pooling",
        r"does to a direction rather than what it does to having more frames;",
        r"the interval resamples places. Queries that cannot reach seven frames",
        r"inside their own place are excluded from every column, so a Top-1",
        r"that rises with $k$ is the attacker gaining and not the sample",
        r"changing.}",
        r"\label{tab:clip_pooling}", r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        rf"\begin{{tabular}}{{l{'c' * len(lengths)}cc}}",
        r"\hline",
        r"& \multicolumn{" + str(len(lengths)) + r"}{c}{Top-1 $\downarrow$ at clip length} & & \\",
        "Condition & " + " & ".join(f"$k={k}$" for k in lengths)
        + r" & $\Delta$ at $k=7$ & 95\% CI \\",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + lines + [r"\hline", r"\end{tabular}",
                                             r"\end{table}"]) + "\n",
                   encoding="utf-8")
    print(f"[table] {out}")
    for st in stats:
        top = " ".join(f"{t:.4f}" for t in st["top1"])
        extra = (f" delta {st['delta']:+.4f} "
                 f"[{st['ci'][0]:+.3f},{st['ci'][1]:+.3f}] p={st['p']:.2g}"
                 if "delta" in st else "")
        print(f"[done] {st['pooling']:11s}{st['condition']:11s}{top}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
