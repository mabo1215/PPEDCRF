"""Render the supplement's two real-place (E1) retrieval tables from the exports.

Why this exists. These two tables were the last ones in either document still
maintained by hand in the LaTeX source, and they had drifted: every point
estimate reproduced exactly from the exports, but two of the eighteen printed
interval endpoints sat about 0.005 outside anything the bootstrap they describe
produces, under any seed and either seed-averaging convention. The script that
made them was never committed, so the discrepancy could not be traced -- which
is the argument for generating them here instead.

What is reported. For each manifest and attacker backbone: Top-1 on the raw
query, Top-1 on the mechanism's release, and their paired difference. Each query
is collapsed to its seed-averaged hit rate before averaging over queries, so a
condition that happened to run an extra seed cannot outvote one that did not.

The interval. Queries are not independent -- the 200 queries of a two-city
manifest fall into 155, 81 or 54 of the dataset's own place clusters -- so the
interval resamples *places* and carries every query of a sampled place. It is a
Monte-Carlo quantity: 10,000 draws are used rather than 2,000 because at 54
clusters the 2,000-draw endpoint still moves by about 0.002 between seeds, which
is the same order as the differences the table is read for. The seed is fixed
and printed in the caption's source so the column is reproducible rather than
merely plausible.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]

BACKBONES = [("resnet18", "ResNet18"), ("resnet50", "ResNet50"),
             ("vgg16", "VGG16"), ("cosplace", "CosPlace"),
             ("mixvpr", "MixVPR"), ("patchnetvlad", "Patch-NetVLAD")]
MANIFESTS = [("primary", "primary"), ("old_to_new", r"old$\to$new"),
             ("new_to_old", r"new$\to$old")]

BOOT_DRAWS = 10000
BOOT_SEED = 20260910


def read_cell(root: Path, scale: str, manifest: str, backbone: str):
    """Raw and released per-query hit rates for one cell, with place labels."""
    path = root / scale / manifest / f"{backbone}.csv"
    if not path.is_file():
        return None
    raw: Dict[str, float] = {}
    released: Dict[str, List[float]] = defaultdict(list)
    place: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            hit = float(int(r["correct_rank"]) == 1)
            place[r["query_id"]] = r["place_id"]
            if r["variant"] == "raw":
                raw[r["query_id"]] = hit
            else:
                released[r["query_id"]].append(hit)
    shared = sorted(set(raw) & set(released))
    if not shared:
        return None
    return (np.array([raw[q] for q in shared]),
            np.array([float(np.mean(released[q])) for q in shared]),
            [place[q] for q in shared])


def place_interval(diff: np.ndarray, places: Sequence[str]) -> Tuple[float, float]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, p in enumerate(places):
        groups[p].append(i)
    clusters = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.empty(BOOT_DRAWS)
    for k in range(BOOT_DRAWS):
        pick = rng.integers(0, len(clusters), len(clusters))
        draws[k] = diff[np.concatenate([clusters[j] for j in pick])].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def is_negative(value: float, places: int = 4) -> bool:
    """Negative as the table prints it, not as the float happens to be.

    One difference lands at -2.2e-18, which is zero to every decimal the table
    shows; counting it as negative would make a caption disagree with the column
    a reader is looking at.
    """
    return round(value, places) < 0


def excludes_zero(lo: float, hi: float, places: int = 3) -> bool:
    """Whether the interval excludes zero at the precision it is printed to."""
    return round(lo, places) > 0 or round(hi, places) < 0


def signed(value: float, places: int = 4) -> str:
    """Print a difference with its sign, and mark an exact zero as such."""
    if abs(value) < 0.5 * 10 ** (-places):
        return r"$\pm%s$" % format(0.0, f".{places}f")
    return "$%+.*f$" % (places, value)


def build(root: Path, scale: str, with_interval: bool):
    rows, stats = [], []
    for manifest, label in MANIFESTS:
        if rows:
            rows.append(r"\hline")
        for backbone, name in BACKBONES:
            cell = read_cell(root, scale, manifest, backbone)
            if cell is None:
                continue
            raw, released, places = cell
            diff = released - raw
            entry = {"manifest": manifest, "backbone": backbone,
                     "raw": float(raw.mean()), "released": float(released.mean()),
                     "delta": float(diff.mean()), "n": len(places),
                     "places": len(set(places))}
            if with_interval:
                lo, hi = place_interval(diff, places)
                entry["ci"] = [lo, hi]
                mark = r"^\ast" if excludes_zero(lo, hi) else ""
                rows.append(
                    rf"{label:<11} & {name:<14} & {entry['raw']:.4f} & "
                    rf"{entry['released']:.4f} & {signed(entry['delta'])} & "
                    rf"$[{lo:+.3f},{hi:+.3f}]{mark}$ \\")
            else:
                rows.append(
                    rf"{label:<11} & {name:<14} & {entry['raw']:.4f} & "
                    rf"{entry['released']:.4f} & {signed(entry['delta'])} \\")
            stats.append(entry)
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports",
                    default=str(REPO / "src" / "exports" / "e1_msls_rows"))
    ap.add_argument("--out-dir", dest="out_dir",
                    default=str(REPO / "paper" / "generated"))
    args = ap.parse_args()
    root, out_dir = Path(args.exports), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = ["% Generated by src/scripts/make_e1_tables.py from",
              "% src/exports/e1_msls_rows. Do not edit by hand."]

    wide_rows, wide_stats = build(root, "wide8", with_interval=False)
    negative = sum(1 for s in wide_stats if is_negative(s["delta"]))
    lines = header + [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Wider 8-city extension of the E1 manifests: the primary",
        rf"manifest and the two official cross-time subtasks, {wide_stats[0]['n']}",
        r"queries and a 2,000-image gallery each, over six attacker backbones.",
        r"Top-1 on the raw query against the mechanism's release, each query",
        r"averaged over three seeds first. Negative $\Delta$ is the protective",
        rf"direction: {negative} of {len(wide_stats)} cells are negative. The",
        r"earlier two-city run is in the extended report.}",
        r"\label{tab:e1_wide8}", r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccc}", r"\hline",
        r"Manifest & Backbone & raw & PPEDCRF & $\Delta$ \\", r"\hline",
    ] + wide_rows + [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "tab_e1_wide8.tex").write_text("\n".join(lines), encoding="utf-8")

    two_rows, two_stats = build(root, "two_city", with_interval=True)
    negative = sum(1 for s in two_stats if is_negative(s["delta"]))
    excluding = sum(1 for s in two_stats if excludes_zero(*s["ci"]))
    places = sorted({s["places"] for s in two_stats}, reverse=True)
    lines = header + [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{E1 multi-backbone extension, the earlier two-city run at",
        rf"{two_stats[0]['n']} queries over a 1,000-image gallery: Top-1 raw",
        r"against the mechanism's release, by manifest and backbone, each with a",
        rf"place-cluster bootstrap 95\% interval on the difference ({BOOT_DRAWS:,}",
        r"resamples over place ids;",
        rf"{', '.join(str(p) for p in places[:-1])} and {places[-1]} place",
        r"identities for the three manifests). Negative $\Delta$ is the",
        rf"protective direction: {negative} of {len(two_stats)} cells are",
        rf"negative, and $\ast$ marks the {excluding} whose interval excludes",
        r"zero.}",
        r"\label{tab:e1_multibackbone}", r"\footnotesize",
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\begin{tabular}{llcccc}", r"\hline",
        r"Manifest & Backbone & raw & PPEDCRF & $\Delta$ & 95\% CI \\", r"\hline",
    ] + two_rows + [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (out_dir / "tab_e1_multibackbone.tex").write_text("\n".join(lines),
                                                      encoding="utf-8")

    summary = {"bootstrap_draws": BOOT_DRAWS, "bootstrap_seed": BOOT_SEED,
               "wide8": wide_stats, "two_city": two_stats,
               "two_city_intervals_excluding_zero": excluding}
    (Path(args.exports) / "e1_table_stats.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote both E1 tables; two-city intervals excluding zero: "
          f"{excluding} of {len(two_stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
