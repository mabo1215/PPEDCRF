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

    raw, cells = [], defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(args.rows)):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
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
    for pooling in args.poolings:
        lines.append(r"\hline")
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{pooling: "
                     rf"{POOL_LABEL.get(pooling, pooling)}}}}} \\")
        for cond in conds:
            for k in lengths:
                arm = cells.get((cond, k, pooling))
                ref = cells.get((CONTROL, k, pooling))
                if not arm:
                    continue
                top1 = float(np.mean(list(arm.values())))
                if cond == CONTROL:
                    lines.append(rf"{LABEL[cond]:<18} & {k} & {top1:.4f} & "
                                 rf"--- & --- & --- \\")
                    stats.append({"condition": cond, "clip_len": k,
                                  "pooling": pooling, "top1": top1})
                    continue
                qs = sorted(set(arm) & set(ref) & set(pm))
                d = np.array([arm[q] - ref[q] for q in qs])
                lo, hi = boot(d, np.array([pm[q] for q in qs]), args.n_boot, rng)
                try:
                    pv = float(wilcoxon(d).pvalue)
                except ValueError:
                    pv = 1.0
                lines.append(rf"{LABEL[cond]:<18} & {k} & {top1:.4f} & "
                             rf"${d.mean():+.4f}$ & $[{lo:+.3f},{hi:+.3f}]$ & "
                             rf"{p_str(pv)} \\")
                stats.append({"condition": cond, "clip_len": k,
                              "pooling": pooling, "top1": top1,
                              "delta": float(d.mean()), "ci": [lo, hi],
                              "p": pv, "n": len(qs)})

    n_places = len({pm[q] for q in cells[(conds[0], lengths[0],
                                          args.poolings[0])] if q in pm})
    n_q = len(cells[(conds[0], lengths[0], args.poolings[0])])
    head = [
        "% Generated by src/scripts/make_clip_pooling_table.py from",
        "% src/exports/clip_pooling. Do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{An attacker holding the clip. Every frame of the clip is",
        r"released under the same condition, optimiser, seed and delivered",
        rf"distortion, and the attacker pools $k$ released frames before",
        rf"ranking the same 2,000-image gallery ({n_q} queries over",
        rf"{n_places} places, 3 seeds). $\Delta$ is against the isotropic",
        r"control at the same clip length and pooling, so the comparison",
        r"isolates what pooling does to a direction rather than what it does",
        r"to having more frames; the interval resamples places.}",
        r"\label{tab:clip_pooling}", r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"Condition & $k$ & Top-1 & $\Delta$ & 95\% CI & $p$ \\",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + lines + [r"\hline", r"\end{tabular}",
                                             r"\end{table}"]) + "\n",
                   encoding="utf-8")
    print(f"[table] {out}")
    for s in stats:
        if "delta" in s and s["clip_len"] == max(lengths):
            print(f"[done] {s['pooling']:10s} k={s['clip_len']} "
                  f"{s['condition']:10s} top1 {s['top1']:.4f} "
                  f"delta {s['delta']:+.4f} "
                  f"[{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
