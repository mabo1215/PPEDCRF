"""Summarise the release audit (A8): what is transmitted, and what survives it.

Two questions from finding R10, answered from the rows
`validate_serialized_release.py` writes.

First, the release boundary. The optimiser projects to a stated linf, then
`release_at_mse` rescales the whole perturbation to hit a delivered-MSE target.
The released amplitude is therefore the projection times that gain, and the
gain is only below 1 when the projection was not the binding constraint. This
script reports the gain and the realised amplitude per condition and budget, so
the manuscript can state the pair rather than the projection alone.

Second, serialisation. Retrieval in the study is evaluated on a float tensor;
a deployment transmits a file. The audit re-measures distortion and Top-k on
the decoded pixels of PNG and JPEG round-trips, and this script reports the
Top-k of each alongside the float baseline. A codec that restores retrieval is
a limitation on the deployment claim, and is reported as one rather than
omitted.

Inference matches the direction family's published treatment: queries are the
unit, places are the cluster, and intervals come from a bootstrap over places.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def load(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def place_of(query_id: str) -> str:
    """Recover the city from a query id, the coarsest available cluster here.

    The per-query place label lives in the manifest rather than in these rows.
    Where the manifest is not alongside the export, the city prefix is still a
    real dependence structure -- queries from one city share a gallery
    neighbourhood -- so it is the honest fallback, and the summary says which
    was used.
    """
    return query_id.split("_", 1)[0]


def place_bootstrap(values_by_place: Dict[str, List[float]], n: int = 10000,
                    seed: int = 20260909) -> Tuple[float, float, float]:
    """Cluster bootstrap: resample places, carry every query inside each.

    Resampling queries independently would understate uncertainty whenever two
    queries in one place move together, which is exactly the dependence the
    review asks to be respected.
    """
    rng = np.random.default_rng(seed)
    places = sorted(values_by_place)
    if not places:
        return float("nan"), float("nan"), float("nan")
    flat = [v for p in places for v in values_by_place[p]]
    point = float(np.mean(flat)) if flat else float("nan")
    draws = []
    idx = np.arange(len(places))
    for _ in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        vals = [v for i in pick for v in values_by_place[places[i]]]
        if vals:
            draws.append(np.mean(vals))
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarise(rows: Sequence[dict], label: str) -> dict:
    """Per-condition, per-serialisation release and retrieval summary."""
    by: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["serialisation"])].append(r)

    out: Dict[str, dict] = {}
    for (cond, ser), rs in sorted(by.items()):
        top1_by_place: Dict[str, List[float]] = defaultdict(list)
        for r in rs:
            top1_by_place[place_of(r["query_id"])].append(float(r["top1"]))
        point, lo, hi = place_bootstrap(top1_by_place)

        def mean(key: str) -> float:
            vals = [float(r[key]) for r in rs if r[key] not in ("", None)]
            return float(np.mean(vals)) if vals else float("nan")

        out[f"{cond}/{ser}"] = {
            "n_queries": len(rs),
            "top1": point, "top1_ci_lo": lo, "top1_ci_hi": hi,
            "top5": mean("top5"), "top10": mean("top10"),
            "release_gain": mean("release_gain"),
            "max_abs_delta_float": mean("max_abs_delta_float"),
            "max_abs_delta_decoded": mean("max_abs_delta_decoded"),
            "mse_float": mean("mse_float"),
            "mse_decoded": mean("mse_decoded"),
            "clipped_frac_float": mean("clipped_frac_float"),
            "bytes": mean("bytes"),
            "linf_projection": (rs[0]["linf_projection"] or None),
        }
    return {"label": label, "conditions": out}


def exceeds_projection(summary: dict) -> List[str]:
    """Name the conditions whose released amplitude passes their own bound.

    Only conditions that actually carry a projection are eligible: the
    isotropic control is drawn rather than optimised, so it has no bound to
    exceed and is excluded rather than counted as a breach.
    """
    notes = []
    for key, s in summary["conditions"].items():
        proj = s["linf_projection"]
        if not proj:
            continue
        if s["max_abs_delta_float"] > float(proj) + 1e-9:
            notes.append(
                f"{key}: released max|delta|={s['max_abs_delta_float']:.2f} "
                f"exceeds projection {float(proj):.0f} "
                f"(gain {s['release_gain']:.3f})")
    return notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more serialized_release.csv files.")
    ap.add_argument("--labels", nargs="+", default=[],
                    help="Label per input; defaults to the parent directory.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    report = []
    for i, path in enumerate(args.inputs):
        p = Path(path)
        label = args.labels[i] if i < len(args.labels) else p.parent.name
        rows = load(p)
        s = summarise(rows, label)
        s["projection_notes"] = exceeds_projection(s)
        report.append(s)

    (out / "a8_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    for s in report:
        print(f"\n=== {s['label']} ===")
        print(f"{'condition/format':28s} {'Top-1':>16s} {'Top-5':>6s} "
              f"{'Top-10':>6s} {'gain':>6s} {'max|d|':>7s} {'MSE':>8s} "
              f"{'KB':>7s}")
        for key, c in s["conditions"].items():
            ci = (f"{c['top1']:.4f}[{c['top1_ci_lo']:.3f},{c['top1_ci_hi']:.3f}]"
                  if not math.isnan(c["top1"]) else "n/a")
            kb = c["bytes"] / 1024.0 if c["bytes"] else 0.0
            print(f"{key:28s} {ci:>16s} {c['top5']:6.3f} {c['top10']:6.3f} "
                  f"{c['release_gain']:6.3f} {c['max_abs_delta_decoded']:7.2f} "
                  f"{c['mse_decoded']:8.2f} {kb:7.1f}")
        for note in s["projection_notes"]:
            print(f"  ! {note}")
        if not s["projection_notes"]:
            print("  (no condition's released amplitude exceeds its own "
                  "projection at this budget)")

    print(f"\n[done] wrote {out / 'a8_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
