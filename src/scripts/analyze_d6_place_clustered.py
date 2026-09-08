"""Place-clustered re-analysis of the D6 transfer exports (review item R5, R10).

The published transfer table uses the query as the unit of inference, which
already fixes the earlier pair-level pooling. It does not address the second
layer of dependence the review names: the 400 queries of the eight-city
manifest carry only 277 distinct ``unique_cluster`` place identities, and one
place can back as many as a dozen queries. Two queries of the same place share
their gallery neighbourhood, so they are not independent draws.

This script reports, for every condition in the D6 plain exports:

* Top-1, Top-5 and Top-10 accuracy, so a ranking claim is not made from Top-1
  alone (R10);
* the paired effect against the isotropic control with the query as the unit,
  seeds averaged within a query before pairing;
* a query-level bootstrap interval (the published procedure) beside a
  place-clustered bootstrap interval that resamples places and carries every
  query of a sampled place, which is the interval the dependence structure
  asks for;
* a Wilcoxon signed-rank p-value over the query-level differences, and its
  Holm-corrected value within a declared primary family;
* an equivalence verdict against a caller-supplied margin, so a "no benefit"
  sentence rests on an interval lying inside a margin rather than on a failure
  to reject equality.

Usage:

    python src/scripts/analyze_d6_place_clustered.py \
        --d6_dir src/outputs/tifs_d6 \
        --sanitizer none --margin 0.01 \
        --json src/outputs/tifs_d6/place_clustered.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# The condition every other condition is paired against.
CONTROL = "isotropic"

# The primary family for the multiplicity correction: one test per
# (backbone, condition) contrast at the reported operating point. Holm is
# applied within this family and the family is named in the output so a reader
# is not left to guess what was corrected against what.
PRIMARY_CONDITIONS = ("transfer_3", "transfer_4", "white_box")

BACKBONE_FILES = {
    "resnet18": "d6_r18_plain.csv",
    "mixvpr": "d6_mix_plain.csv",
}


def load_rows(path: Path, sanitizer: str):
    """Return {(condition, query_id): {seed: correct_rank}} for one sanitizer."""
    table: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    places: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["sanitizer"] != sanitizer:
                continue
            key = (row["condition"], row["query_id"])
            seed = row["seed"]
            if seed in table[key]:
                raise ValueError(f"duplicate row for {key} seed {seed} in {path}")
            table[key][seed] = int(row["correct_rank"])
            places.setdefault(row["query_id"], row["correct_place"])
    return table, places


def hit_matrix(table, condition: str, queries, k: int) -> np.ndarray:
    """Seed-averaged Top-k indicator per query, in the order of `queries`."""
    out = np.empty(len(queries), dtype=float)
    for i, qid in enumerate(queries):
        seeds = table[(condition, qid)]
        out[i] = float(np.mean([1.0 if r <= k else 0.0 for r in seeds.values()]))
    return out


def bootstrap_ci(diff: np.ndarray, groups: np.ndarray | None, n_boot: int,
                 rng: np.random.Generator) -> tuple[float, float]:
    """Percentile interval for the mean of `diff`.

    With `groups` given, the resampling unit is the group: a sampled group
    contributes all of its rows, which is what carries the within-place
    dependence into the interval.
    """
    if groups is None:
        idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        means = diff[idx].mean(axis=1)
    else:
        order = np.argsort(groups, kind="stable")
        sorted_groups = groups[order]
        sorted_diff = diff[order]
        bounds = np.searchsorted(sorted_groups, np.unique(sorted_groups))
        members = np.split(sorted_diff, bounds[1:])
        n_groups = len(members)
        means = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            pick = rng.integers(0, n_groups, size=n_groups)
            sample = np.concatenate([members[j] for j in pick])
            means[b] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down correction over a named family."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def analyse(d6_dir: Path, sanitizer: str, margin: float, n_boot: int,
            seed: int) -> dict:
    rng = np.random.default_rng(seed)
    report: dict = {
        "sanitizer": sanitizer,
        "equivalence_margin_top1": margin,
        "n_bootstrap": n_boot,
        "rng_seed": seed,
        "primary_family": [],
        "backbones": {},
    }
    raw_p: dict[str, float] = {}

    for backbone, filename in BACKBONE_FILES.items():
        table, places = load_rows(d6_dir / filename, sanitizer)
        conditions = sorted({cond for cond, _ in table})
        queries = sorted({qid for _, qid in table})
        place_ids = np.array([places[q] for q in queries])
        n_places = len(set(place_ids))
        sizes = np.unique(place_ids, return_counts=True)[1]

        entry = {
            "n_queries": len(queries),
            "n_places": n_places,
            "max_queries_per_place": int(sizes.max()),
            "mean_queries_per_place": float(sizes.mean()),
            "conditions": {},
        }

        control = {k: hit_matrix(table, CONTROL, queries, k) for k in (1, 5, 10)}
        for cond in conditions:
            hits = {k: hit_matrix(table, cond, queries, k) for k in (1, 5, 10)}
            cell = {f"top{k}": float(hits[k].mean()) for k in (1, 5, 10)}
            if cond != CONTROL:
                for k in (1, 5, 10):
                    diff = hits[k] - control[k]
                    q_lo, q_hi = bootstrap_ci(diff, None, n_boot, rng)
                    p_lo, p_hi = bootstrap_ci(diff, place_ids, n_boot, rng)
                    nonzero = diff[diff != 0.0]
                    wil = (float(stats.wilcoxon(nonzero).pvalue)
                           if nonzero.size else float("nan"))
                    cell[f"delta_top{k}"] = float(diff.mean())
                    cell[f"query_ci_top{k}"] = [q_lo, q_hi]
                    cell[f"place_ci_top{k}"] = [p_lo, p_hi]
                    cell[f"wilcoxon_p_top{k}"] = wil
                    cell[f"ci_widening_top{k}"] = (
                        (p_hi - p_lo) / (q_hi - q_lo) if q_hi > q_lo else float("nan")
                    )
                    if k == 1:
                        # An equivalence verdict needs the interval inside the
                        # margin on both sides; a CI that merely covers zero is
                        # not evidence of no effect.
                        cell["equivalent_top1"] = bool(
                            p_lo > -margin and p_hi < margin
                        )
                        cell["separated_from_zero_top1"] = bool(
                            p_hi < 0.0 or p_lo > 0.0
                        )
                if cond in PRIMARY_CONDITIONS:
                    raw_p[f"{backbone}:{cond}"] = cell["wilcoxon_p_top1"]
            entry["conditions"][cond] = cell
        report["backbones"][backbone] = entry

    report["primary_family"] = sorted(raw_p)
    adjusted = holm(raw_p)
    for name, value in adjusted.items():
        backbone, cond = name.split(":")
        report["backbones"][backbone]["conditions"][cond]["holm_p_top1"] = value
    return report


def render(report: dict) -> str:
    lines = [
        f"D6 place-clustered re-analysis (sanitizer={report['sanitizer']}, "
        f"{report['n_bootstrap']} resamples, RNG seed {report['rng_seed']})",
        f"Equivalence margin on Top-1: +/-{report['equivalence_margin_top1']}",
        f"Primary family for Holm: {', '.join(report['primary_family'])}",
        "",
    ]
    for backbone, entry in report["backbones"].items():
        lines.append(
            f"== {backbone}: {entry['n_queries']} queries, {entry['n_places']} "
            f"places, up to {entry['max_queries_per_place']} queries per place"
        )
        header = (f"{'condition':<12}{'top1':>8}{'top5':>8}{'top10':>8}"
                  f"{'d.top1':>9}{'query 95% CI':>22}{'place 95% CI':>22}"
                  f"{'wilcoxon':>11}{'holm':>11}")
        lines.append(header)
        for cond, cell in entry["conditions"].items():
            row = (f"{cond:<12}{cell['top1']:>8.4f}{cell['top5']:>8.4f}"
                   f"{cell['top10']:>8.4f}")
            if "delta_top1" in cell:
                qci = cell["query_ci_top1"]
                pci = cell["place_ci_top1"]
                row += (f"{cell['delta_top1']:>9.4f}"
                        f"{f'[{qci[0]:+.4f},{qci[1]:+.4f}]':>22}"
                        f"{f'[{pci[0]:+.4f},{pci[1]:+.4f}]':>22}"
                        f"{cell['wilcoxon_p_top1']:>11.2e}")
                row += (f"{cell['holm_p_top1']:>11.2e}"
                        if "holm_p_top1" in cell else f"{'--':>11}")
            lines.append(row)
        lines.append("")
        for cond, cell in entry["conditions"].items():
            if "delta_top5" not in cell:
                continue
            lines.append(
                f"   {cond}: Top-5 {cell['delta_top5']:+.4f} "
                f"[{cell['place_ci_top5'][0]:+.4f},{cell['place_ci_top5'][1]:+.4f}], "
                f"Top-10 {cell['delta_top10']:+.4f} "
                f"[{cell['place_ci_top10'][0]:+.4f},{cell['place_ci_top10'][1]:+.4f}], "
                f"place interval {cell['ci_widening_top1']:.2f}x the query interval"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d6_dir", default="src/outputs/tifs_d6")
    ap.add_argument("--sanitizer", default="none",
                    help="preprocessing condition to analyse (default: none)")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="equivalence margin on Top-1")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    report = analyse(Path(args.d6_dir), args.sanitizer, args.margin,
                     args.n_boot, args.seed)
    text = render(report)
    print(text)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                       encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
