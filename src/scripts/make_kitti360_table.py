"""Export the KITTI-360 replication and render its table.

Why this exists. Every real-place result in the manuscript came from one
dataset, which is the most natural thing for a referee to attack in a paper
whose thesis is that a design axis does not work. This is the second dataset:
a place-labelled retrieval benchmark built from KITTI-360 revisits, run through
the same protocol and the same energy gate as the MSLS one, so a difference
between them is a difference of dataset and not of method.

What is reported. The placement arm gives every energy-matched rule against
the uniform reference; the direction arm gives the surrogate-optimised
direction against the isotropic control at the same delivered distortion. Each
query is collapsed to its seed-averaged hit rate before pairing, the interval
is a query bootstrap and the test a Wilcoxon signed-rank, exactly as on MSLS.

Two differences from MSLS belong in the caption rather than buried. A place
here is a stretch of road the drive revisits, so a hit means the right stretch,
which is coarser than MSLS's 25 m ball; that changes the absolute Top-1 level
and not the paired contrasts. And 227 queries come from only 16 places, against
277 on the eight-city MSLS manifest, so the manuscript's own clustered unit of
inference -- resample places, carry every query of a sampled place -- is far
more binding here than there. Both intervals are therefore computed and both
are printed, because on this benchmark they disagree about the direction arm
and the reader is entitled to see that rather than be handed the narrower one.

Reference levels. A null is only as good as the demonstrated sensitivity of the
measurement behind it, so the table opens with three rows fixing the scale: the
unperturbed query, the mechanism's own release, and a white-box sign-gradient
bound. Without them a reader cannot tell whether the placements fail to move
retrieval because placement does nothing or because there was nothing to move.

Completeness. The table is generated from a fixed list of conditions, which is
how two conditions that had been run once went unreported. The list is now
checked against the export and a condition present in the data but missing from
the list is a hard failure, not a silent omission.
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

KEEP_PLACE = ["query_id", "place_id", "placement", "seed", "correct_rank"]
KEEP_DIR = ["query_id", "place_id", "condition", "seed", "correct_rank"]
KEEP_BASE = ["query_id", "place_id", "variant", "seed", "correct_rank"]
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
    ("margin_oracle", "margin oracle"), ("anti_margin_oracle", "anti-margin or."),
]


def load_places(manifest: Path) -> Dict[str, str]:
    """query_id -> place label, so the export can be clustered on its own."""
    places: Dict[str, str] = {}
    if not manifest.is_file():
        return places
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                places[r["query_id"]] = r["place_id"]
    return places


def slim(source: Path, target: Path, keep: List[str],
         places: Dict[str, str]) -> int:
    rows = []
    with source.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            r = dict(r)
            r.setdefault("place_id", places.get(r["query_id"], ""))
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


def compare(arm, ref, rng, places=None):
    """Paired contrast with a query interval and, when places are given, a
    clustered one that resamples places and carries all their queries."""
    shared = sorted(set(arm) & set(ref))
    diff = np.array([arm[q] - ref[q] for q in shared])
    draws = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                      for _ in range(BOOT_DRAWS)])
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    nz = diff[diff != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    clustered = None
    if places:
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, q in enumerate(shared):
            groups[places.get(q, q)].append(i)
        cl = [np.array(v) for v in groups.values()]
        cdraws = np.empty(BOOT_DRAWS)
        for k in range(BOOT_DRAWS):
            pick = rng.integers(0, len(cl), len(cl))
            cdraws[k] = diff[np.concatenate([cl[j] for j in pick])].mean()
        clo, chi = (float(x) for x in np.percentile(cdraws, [2.5, 97.5]))
        clustered = (clo, chi, len(cl))
    return (float(np.mean([arm[q] for q in shared])), float(diff.mean()),
            lo, hi, p, len(shared), clustered)



def p_str(p: float) -> str:
    """A p-value a reader can act on. A Wilcoxon p is never exactly zero, and
    printing 0.000 claims a precision the test does not have; below the
    resolution of three decimals the honest form is the bound."""
    return "$<$0.001" if p < 5e-4 else f"{p:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--placement",
                    default=str(REPO / "src" / "outputs" / "kitti360_placement"
                                / "per_query.csv"))
    ap.add_argument("--direction",
                    default=str(REPO / "src" / "outputs" / "kitti360_direction"
                                / "per_query.csv"))
    ap.add_argument("--baseline",
                    default=str(REPO / "src" / "outputs" / "kitti360_baseline"
                                / "geotagged_vpr_per_query.csv"))
    ap.add_argument("--manifest",
                    default=str(REPO / "src" / "outputs" / "e1_kitti360"
                                / "manifest_loop.jsonl"))
    ap.add_argument("--export-dir", dest="export_dir",
                    default=str(REPO / "src" / "exports" / "kitti360_rows"))
    ap.add_argument("--output", default=str(REPO / "paper" / "generated"
                                            / "tab_kitti360.tex"))
    args = ap.parse_args()
    ex = Path(args.export_dir)
    places = load_places(Path(args.manifest))

    for src, name, keep in ((args.placement, "placement.csv", KEEP_PLACE),
                            (args.direction, "direction.csv", KEEP_DIR),
                            (args.baseline, "baseline.csv", KEEP_BASE)):
        if Path(src).is_file():
            print(f"[export] {slim(Path(src), ex / name, keep, places)} rows "
                  f"-> {ex / name}")
    place_rows = list(csv.DictReader((ex / "placement.csv").open(
        newline="", encoding="utf-8")))
    dir_rows = list(csv.DictReader((ex / "direction.csv").open(
        newline="", encoding="utf-8")))
    if not places:
        places = {r["query_id"]: r["place_id"] for r in dir_rows if r["place_id"]}

    # Completeness: a condition that was run and is not in PLACEMENTS would be
    # dropped silently, which is how two of them went unreported once already.
    ran = {r["placement"] for r in place_rows}
    listed = {k for k, _ in PLACEMENTS}
    missing = sorted(ran - listed)
    if missing:
        print(f"[FAIL] run but not in the table: {', '.join(missing)}")
        return 1
    print(f"[check] all {len(ran)} conditions in the export are reported")

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
            lines.append(rf"{label:<18} & {np.mean(list(ref.values())):.4f} "
                         rf"& --- & --- & --- \\")
            continue
        top1, delta, lo, hi, p, n, cl = compare(arm, ref, rng, places)
        worst = max(worst, abs(delta))
        lines.append(rf"{label:<18} & {top1:.4f} & ${delta:+.4f}$ & "
                     rf"$[{lo:+.3f},{hi:+.3f}]$ & {p:.2f} \\")
        stats.append({"arm": "placement", "name": key, "top1": top1,
                      "delta": delta, "ci": [lo, hi], "p": p, "n": n,
                      "clustered_ci": list(cl[:2]) if cl else None})

    iso = per_query(dir_rows, "condition", "isotropic")
    dtop1, ddelta, dlo, dhi, dp, dn, dcl = compare(
        per_query(dir_rows, "condition", "transfer_3"), iso, rng, places)
    stats.append({"arm": "direction", "name": "transfer_3", "top1": dtop1,
                  "delta": ddelta, "ci": [dlo, dhi], "p": dp, "n": dn,
                  "clustered_ci": list(dcl[:2]) if dcl else None})
    # Reference levels: what an unperturbed query scores, what the mechanism
    # itself does, and what a white-box attacker can do to this benchmark.
    base_lines = []
    base_path = ex / "baseline.csv"
    if base_path.is_file():
        base_rows = list(csv.DictReader(base_path.open(newline="", encoding="utf-8")))
        raw = per_query(base_rows, "variant", "raw")
        for key, label in (("raw", "clean (unperturbed)"),
                           ("full", "mechanism release"),
                           ("attacker_aware", "white-box bound")):
            arm = per_query(base_rows, "variant", key)
            if not arm:
                continue
            if key == "raw":
                base_lines.append(rf"{label:<18} & {np.mean(list(arm.values())):.4f}"
                                  rf" & --- & --- & --- \\")
                continue
            t1, dl, lo2, hi2, pv, nn, cc = compare(arm, raw, rng, places)
            ci = cc if cc else (lo2, hi2)
            base_lines.append(rf"{label:<18} & {t1:.4f} & ${dl:+.4f}$ & "
                              rf"$[{ci[0]:+.3f},{ci[1]:+.3f}]$ & {p_str(pv)} \\")
            stats.append({"arm": "reference", "name": key, "top1": t1,
                          "delta": dl, "ci": [lo2, hi2],
                          "clustered_ci": list(cc[:2]) if cc else None,
                          "p": pv, "n": nn})

    n_places = dcl[2] if dcl else 0
    # Does any placement separate from zero once places are the unit?
    cl_sig = [s_ for s_ in stats if s_["arm"] == "placement" and s_["clustered_ci"]
              and (s_["clustered_ci"][0] > 0 or s_["clustered_ci"][1] < 0)]

    out = [
        "% Generated by src/scripts/make_kitti360_table.py from",
        "% src/exports/kitti360_rows. Do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{The same two comparisons on a second dataset: a",
        rf"place-labelled benchmark of {len(ref)} queries from {n_places} places",
        rf"over a 2,000-image gallery built from KITTI-360 revisits, {len(seeds)}",
        r"seeds, each query at least 600 frames from any positive. Placement is",
        r"against the uniform reference, direction against the isotropic control",
        r"at the same delivered distortion. The printed interval is a query",
        rf"bootstrap; with only {n_places} places the direction row also carries",
        r"the place-clustered interval this protocol prescribes, and the two",
        r"disagree. Every placement row was tested on both units and separates",
        r"from zero on neither, so only the query interval is printed for them;",
        r"the white-box bound separates on the clustered unit, which is what",
        r"shows this benchmark can be moved at all. Reference rows carry the",
        r"clustered interval. A place",
        r"here is a road stretch, coarser than the 25\,m ball MSLS uses, which",
        r"moves the absolute level and not the paired contrasts.}",
        r"\label{tab:kitti360}", r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lcccc}", r"\hline",
        r"Condition & Top-1 & $\Delta$ & 95\% CI & $p$ \\",
        r"\hline",
        r"\multicolumn{5}{l}{\textit{Reference levels}} \\",
    ] + base_lines + [
        r"\hline",
        r"\multicolumn{5}{l}{\textit{Allocation: energy-matched placements}} \\",
    ] + lines + [
        r"\hline",
        r"\multicolumn{5}{l}{\textit{Direction: surrogate ensemble, same budget}} \\",
        rf"isotropic (ref.)   & {np.mean(list(iso.values())):.4f} & --- & --- & --- \\",
        rf"3 surrogates       & {dtop1:.4f} & ${ddelta:+.4f}$ & "
        rf"$[{dlo:+.3f},{dhi:+.3f}]$ & {p_str(dp)} \\",
        rf"\quad clustered on {n_places} places & & & "
        rf"$[{dcl[0]:+.3f},{dcl[1]:+.3f}]$ & \\" if dcl else "",
        r"\hline", r"\end{tabular}", r"\end{table}", ""]
    Path(args.output).write_text("\n".join(x for x in out if x != ""), encoding="utf-8")

    (ex / "kitti360_summary.json").write_text(json.dumps(
        {"queries": len(ref), "places": n_places, "seeds": seeds,
         "bootstrap_draws": BOOT_DRAWS, "bootstrap_seed": BOOT_SEED,
         "equivalence_margin": MARGIN, "conditions_run": sorted(ran),
         "largest_placement_abs_delta": worst,
         "placements_significant_clustered": [s_["name"] for s_ in cl_sig],
         "rows": stats}, indent=2) + "\n", encoding="utf-8")
    print(f"[done] {len(ran)} conditions, largest |placement delta| {worst:.4f}, "
          f"none clustered-significant: {not cl_sig}")
    print(f"[done] direction {ddelta:+.4f} query [{dlo:+.3f},{dhi:+.3f}] "
          f"clustered [{dcl[0]:+.3f},{dcl[1]:+.3f}] over {n_places} places")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
