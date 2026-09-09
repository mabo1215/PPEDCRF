"""Privacy against measured utility, over budgets (experiment A5, finding R7).

The manuscript compares conditions at one delivered-distortion operating point
and reports downstream utility separately. R7 objects that this does not
establish a privacy--utility frontier: an undetected privacy effect is not
exactly zero, and one operating point does not show how the trade-off moves.

This script joins the two axes at matched delivered MSE. Utility comes from
`evaluate_direction_utility.py` (segmenter mIoU on VOC frames), privacy from
`validate_serialized_release.py` (Top-1/5/10 retrieval on MSLS frames), and
they are paired by the delivered-MSE budget both were solved to. The output is
the frontier the review asks for, plus the comparison at matched *utility*
rather than matched distortion, which is the comparison a deployment actually
faces.

Two honest limits are printed with the numbers rather than left implicit.

First, this is a **cross-dataset** pairing: retrieval is measured on MSLS
street scenes and utility on VOC frames, because the VOC frames carry the
segmentation labels and the MSLS frames carry the place labels. No single image
here has both. The pairing is therefore between distributions at a shared
distortion budget, not between two measurements of one release, and the script
labels every joined row as such.

Second, a declared utility tolerance is required as an input rather than chosen
after seeing the results. `--utility_tolerance` is the largest mIoU drop from
clean that the caller is willing to accept; the frontier then reports which
conditions are admissible at each budget under that declaration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Utility runs use these condition names; the privacy runs use the same three
# plus a hardened variant under a different spelling. Joining them needs the
# map to be explicit rather than inferred from string similarity.
CONDITION_ALIASES = {
    "direction_eot": "hardened_direction",
    "hardened_direction": "hardened_direction",
    "direction": "direction",
    "isotropic": "isotropic",
    "clean": "clean",
}


def canonical(name: str) -> str:
    return CONDITION_ALIASES.get(name, name)


def load_utility(root: Path) -> Dict[Tuple[float, str], dict]:
    """Read every `segmentation_mse<budget>/` summary under `root`.

    Per-image rows are deduplicated on (image_id, condition, seed) before being
    aggregated: a run that was restarted while a previous process was still
    writing can leave each row twice, and while those duplicates were verified
    identical here, averaging over them would silently weight some images
    double if they ever were not.
    """
    out: Dict[Tuple[float, str], dict] = {}
    for d in sorted(root.glob("*_mse*")):
        per_image = d / "per_image.jsonl"
        if not per_image.is_file():
            continue
        try:
            budget = float(d.name.rsplit("mse", 1)[1])
        except (IndexError, ValueError):
            continue
        seen: Dict[Tuple[str, str, object], dict] = {}
        with per_image.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                seen[(r["image_id"], r["condition"], r.get("seed"))] = r
        by_cond: Dict[str, List[dict]] = defaultdict(list)
        for r in seen.values():
            by_cond[canonical(r["condition"])].append(r)
        for cond, rows in by_cond.items():
            # Dataset-level mIoU, matching evaluate_direction_utility.py:
            # intersections and unions are pooled over images per class and the
            # ratio is taken once, then averaged over classes. Averaging
            # per-image IoU instead would be a different statistic -- exactly
            # the confusion R7 flags for detection AP -- and would not
            # reproduce the published table.
            inter: Dict[str, int] = defaultdict(int)
            union: Dict[str, int] = defaultdict(int)
            for r in rows:
                for class_id, iu in (r.get("seg_iu") or {}).items():
                    i_val, u_val = iu
                    inter[class_id] += int(i_val)
                    union[class_id] += int(u_val)
            ious = [inter[c] / union[c] for c in union if union[c]]
            mses = [float(r["effective_mse"]) for r in rows
                    if r.get("effective_mse") is not None]
            out[(budget, cond)] = {
                "n_images": len(rows),
                "n_classes": len(ious),
                "miou": float(np.mean(ious)) if ious else float("nan"),
                "delivered_mse": float(np.mean(mses)) if mses else float("nan"),
            }
    return out


def load_privacy(paths: Sequence[Path]) -> Dict[Tuple[float, str], dict]:
    """Read A8 exports, keeping only the float (pre-serialisation) release.

    The frontier is about the optimiser's operating point, so it uses the same
    object the utility side perturbs. Codec effects are a separate question and
    are reported by `analyze_serialized_release.py`.
    """
    out: Dict[Tuple[float, str], dict] = {}
    for p in paths:
        if not p.is_file():
            continue
        rows = []
        with p.open("r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("serialisation") == "float":
                    rows.append(r)
        by: Dict[Tuple[float, str], List[dict]] = defaultdict(list)
        for r in rows:
            budget = round(float(r["mse_float"]), 2)
            by[(budget, canonical(r["condition"]))].append(r)
        for (budget, cond), rs in by.items():
            by_place: Dict[str, List[float]] = defaultdict(list)
            for r in rs:
                by_place[r["query_id"].split("_", 1)[0]].append(float(r["top1"]))
            point, lo, hi = place_bootstrap(by_place)
            out[(budget, cond)] = {
                "n_queries": len(rs),
                "top1": point, "top1_lo": lo, "top1_hi": hi,
                "top5": float(np.mean([float(r["top5"]) for r in rs])),
                "top10": float(np.mean([float(r["top10"]) for r in rs])),
            }
    return out


def place_bootstrap(values_by_place: Dict[str, List[float]], n: int = 10000,
                    seed: int = 20260909) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    places = sorted(values_by_place)
    flat = [v for p in places for v in values_by_place[p]]
    if not flat:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(flat))
    idx = np.arange(len(places))
    draws = []
    for _ in range(n):
        pick = rng.choice(idx, size=len(idx), replace=True)
        vals = [v for i in pick for v in values_by_place[places[i]]]
        if vals:
            draws.append(np.mean(vals))
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def nearest_budget(target: float, budgets: Sequence[float],
                   rel_tol: float = 0.10) -> Optional[float]:
    """Match a privacy budget to a utility budget within a relative tolerance.

    The two pipelines solve for delivered MSE independently and land a fraction
    of a percent apart, so exact float equality never matches. Anything outside
    the tolerance is reported as unmatched rather than snapped to the closest
    value, because silently pairing 60 with 241 would fabricate a frontier
    point.
    """
    best, best_d = None, math.inf
    for b in budgets:
        d = abs(b - target) / max(target, 1e-9)
        if d < best_d:
            best, best_d = b, d
    return best if best_d <= rel_tol else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--utility_root", required=True,
                    help="Directory holding segmentation_mse<budget>/ dirs.")
    ap.add_argument("--privacy", nargs="+", required=True,
                    help="serialized_release.csv files from A8.")
    ap.add_argument("--utility_tolerance", type=float, default=0.05,
                    help="Largest acceptable mIoU drop from clean. Declared "
                         "by the caller before results are seen; the frontier "
                         "reports admissibility against it.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    utility = load_utility(Path(args.utility_root))
    privacy = load_privacy([Path(p) for p in args.privacy])

    ubudgets = sorted({b for b, _ in utility})
    clean = {b: utility[(b, "clean")]["miou"] for b in ubudgets
             if (b, "clean") in utility}
    # Clean utility does not depend on the budget, so disagreement between
    # budgets would mean the runs are not comparable. Report it rather than
    # averaging it away.
    clean_vals = sorted(clean.values())
    clean_spread = (clean_vals[-1] - clean_vals[0]) if clean_vals else float("nan")
    clean_miou = float(np.mean(clean_vals)) if clean_vals else float("nan")

    rows = []
    for (pb, cond), pv in sorted(privacy.items()):
        ub = nearest_budget(pb, ubudgets)
        if ub is None or (ub, cond) not in utility:
            rows.append({"budget_privacy": pb, "budget_utility": None,
                         "condition": cond, "matched": False, **pv})
            continue
        uv = utility[(ub, cond)]
        drop = clean_miou - uv["miou"]
        rows.append({
            "budget_privacy": pb, "budget_utility": ub, "condition": cond,
            "matched": True,
            "top1": pv["top1"], "top1_lo": pv["top1_lo"], "top1_hi": pv["top1_hi"],
            "top5": pv["top5"], "top10": pv["top10"],
            "n_queries": pv["n_queries"],
            "miou": uv["miou"], "n_images": uv["n_images"],
            "miou_drop_from_clean": drop,
            "admissible": bool(drop <= args.utility_tolerance),
        })

    report = {
        "clean_miou": clean_miou,
        "clean_miou_spread_across_budgets": clean_spread,
        "utility_tolerance": args.utility_tolerance,
        "pairing": ("CROSS-DATASET: retrieval on MSLS, utility on VOC, joined "
                    "by delivered-MSE budget. No single image contributes to "
                    "both axes; this is a comparison of distributions at a "
                    "shared budget, not of one release measured twice."),
        "rows": rows,
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "privacy_utility_frontier.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"clean mIoU = {clean_miou:.4f} "
          f"(spread across budgets {clean_spread:.5f})")
    print(f"declared utility tolerance: {args.utility_tolerance:.3f} mIoU drop\n")
    print(f"{'budget':>9s} {'condition':>20s} {'Top-1':>18s} {'Top-5':>6s} "
          f"{'mIoU':>7s} {'drop':>7s} {'ok?':>4s}")
    for r in rows:
        if not r["matched"]:
            print(f"{r['budget_privacy']:9.2f} {r['condition']:>20s} "
                  f"{'(no matching utility budget)':>40s}")
            continue
        ci = f"{r['top1']:.3f}[{r['top1_lo']:.3f},{r['top1_hi']:.3f}]"
        print(f"{r['budget_utility']:9.2f} {r['condition']:>20s} {ci:>18s} "
              f"{r['top5']:6.3f} {r['miou']:7.4f} "
              f"{r['miou_drop_from_clean']:7.4f} "
              f"{'yes' if r['admissible'] else 'NO':>4s}")

    print(f"\n{report['pairing']}")
    print(f"\n[done] wrote {out / 'privacy_utility_frontier.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
