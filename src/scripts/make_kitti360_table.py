"""Export the KITTI-360 replication and render its table.

Why this exists. Every real-place result in the manuscript came from one
dataset, which is the most natural thing for a referee to attack in a paper
whose thesis is that a design axis does not work. This is the second dataset:
a place-labelled retrieval benchmark built from KITTI-360 revisits, run through
the same protocol, the same energy gate and the same unit of inference as the
MSLS one, so a difference between them is a difference of dataset and not of
method.

What is reported. The placement arm gives every energy-matched rule against
the uniform reference; the direction arm gives the surrogate-optimised
direction against the isotropic control at the same delivered distortion. Each
query is collapsed to its seed-averaged hit rate before pairing, the interval
is a query bootstrap and the test a Wilcoxon signed-rank, exactly as on MSLS.

One difference from MSLS worth stating in the caption rather than burying: a
place here is a stretch of road the drive revisits, so a hit means the right
stretch, which is coarser than MSLS's 25 m ball. That changes the absolute
Top-1 level and not the paired contrasts, which is all either arm reads.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]

KEEP_PLACE = ["query_id", "placement", "seed", "correct_rank"]
KEEP_DIR = ["query_id", "condition", "seed", "correct_rank"]
BOOT_DRAWS = 10000
BOOT_SEED = 20260910
MARGIN = 0.01

# (export key, printed label). Uniform is the reference and prints first.
PLACEMENTS = [
    ("uniform", "uniform (ref.)"), ("learned", "learned support"),
    ("oracle_grad", "score-gradient"), ("anti_oracle_grad", "anti-score-grad."),
    ("saliency", "saliency"), ("center", "centre bias"),
    ("random_fixed", "fixed random"), ("edge", "edge magnitude"),
    ("segmentation", "DeepLabV3 scene"), ("segmentation_fcn", "FCN scene"),
    ("segmentation_ade", "SegFormer scene"),
]


def slim(source: Path, target: Path, keep: List[str]) -> int:
    rows = []
    with source.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            rows.append({k: r[k] for k in keep})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=keep)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def per_query(rows, column: str, value: str) -> Dict[str, float]:
    hits = defaultdict(list)
    for r in rows:
        if r[column] == value:
            hits[r["query_id"]].append(float(int(r["correct_rank"]) == 1))
    return {q: float(np.mean(v)) for q, v in hits.items()}


def compare(arm, ref, rng):
    shared = sorted(set(arm) & set(ref))
    diff = np.array([arm[q] - ref[q] for q in shared])
    draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                      for _ in range(BOOT_DRAWS)])
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    nz = diff[diff != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    return (float(np.mean([arm[q] for q in shared])), float(diff.mean()),
            lo, hi, p, len(shared))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--placement",
                    default=str(REPO / "src" / "outputs" / "kitti360_placement"
                                / "per_query.csv"))
    ap.add_argument("--direction",
                    default=str(REPO / "src" / "outputs" / "kitti360_direction"
                                / "per_query.csv"))
    ap.add_argument("--export-dir", dest="export_dir",
                    default=str(REPO / "src" / "exports" / "kitti360_rows"))
    ap.add_argument("--output", default=str(REPO / "paper" / "generated"
                                            / "tab_kitti360.tex"))
    args = ap.parse_args()
    ex = Path(args.export_dir)

    for src, name, keep in ((args.placement, "placement.csv", KEEP_PLACE),
                            (args.direction, "direction.csv", KEEP_DIR)):
        if Path(src).is_file():
            print(f"[export] {slim(Path(src), ex / name, keep)} rows -> {ex / name}")
    place_rows = list(csv.DictReader((ex / "placement.csv").open(
        newline="", encoding="utf-8")))
    dir_rows = list(csv.DictReader((ex / "direction.csv").open(
        newline="", encoding="utf-8")))

    rng = np.random.default_rng(BOOT_SEED)
    ref = per_query(place_rows, "placement", "uniform")
    seeds = sorted({r["seed"] for r in place_rows})
    stats, lines = [], []
    worst = 0.0
    for key, label in PLACEMENTS:
        arm = per_query(place_rows, "placement", key)
        if not arm:
            continue
        if key == "uniform":
            lines.append(rf"{label:<18} & {np.mean(list(ref.values())):.4f} & --- & --- & --- \\")
            continue
        top1, delta, lo, hi, p, n = compare(arm, ref, rng)
        worst = max(worst, abs(delta))
        lines.append(rf"{label:<18} & {top1:.4f} & ${delta:+.4f}$ & "
                     rf"$[{lo:+.3f},{hi:+.3f}]$ & {p:.2f} \\")
        stats.append({"arm": "placement", "name": key, "top1": top1,
                      "delta": delta, "ci": [lo, hi], "p": p, "n": n})

    iso = per_query(dir_rows, "condition", "isotropic")
    dtop1, ddelta, dlo, dhi, dp, dn = compare(
        per_query(dir_rows, "condition", "transfer_3"), iso, rng)
    stats.append({"arm": "direction", "name": "transfer_3", "top1": dtop1,
                  "delta": ddelta, "ci": [dlo, dhi], "p": dp, "n": dn})

    out = [
        "% Generated by src/scripts/make_kitti360_table.py from",
        "% src/exports/kitti360_rows. Do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{The same two comparisons on a second dataset: a",
        rf"place-labelled benchmark of {len(ref)} queries over a 2,000-image",
        r"gallery built from KITTI-360 revisits, each query at least 600 frames",
        r"from any of its positives. Placement is against the uniform reference",
        r"and direction against the isotropic control at the same delivered",
        rf"distortion, over {len(seeds)} seeds, with a query bootstrap interval",
        r"and a Wilcoxon signed-rank test. A place here is a stretch of road the",
        r"drive revisits, so a hit means the right stretch --- coarser than the",
        r"25\,m ball MSLS uses, which moves the absolute level but not the",
        r"paired contrasts these rows read.}",
        r"\label{tab:kitti360}", r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lcccc}", r"\hline",
        r"Condition & Top-1 & $\Delta$ & 95\% CI & $p$ \\",
        r"\hline",
        r"\multicolumn{5}{l}{\textit{Allocation: energy-matched placements}} \\",
    ] + lines + [
        r"\hline",
        r"\multicolumn{5}{l}{\textit{Direction: surrogate ensemble, same budget}} \\",
        rf"isotropic (ref.)   & {np.mean(list(iso.values())):.4f} & --- & --- & --- \\",
        rf"3 surrogates       & {dtop1:.4f} & ${ddelta:+.4f}$ & "
        rf"$[{dlo:+.3f},{dhi:+.3f}]$ & {dp:.3f} \\",
        r"\hline", r"\end{tabular}", r"\end{table}", ""]
    Path(args.output).write_text("\n".join(out), encoding="utf-8")

    (ex / "kitti360_summary.json").write_text(json.dumps(
        {"queries": len(ref), "seeds": seeds, "bootstrap_draws": BOOT_DRAWS,
         "bootstrap_seed": BOOT_SEED, "equivalence_margin": MARGIN,
         "largest_placement_abs_delta": worst, "rows": stats},
        indent=2) + "\n", encoding="utf-8")
    print(f"[done] largest |placement delta| {worst:.4f}; "
          f"direction {ddelta:+.4f} [{dlo:+.3f},{dhi:+.3f}] p={dp:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
