"""Do `unique_cluster` positives agree with geographic positives?

The benchmark scores a retrieval as correct when the retrieved gallery image
carries the query's `unique_cluster` label. The MSLS paper instead defines a
single-image positive geographically: within a distance threshold of the query,
and within a viewing-angle threshold of its heading. Finding R9 asks for these
to be checked against each other before cluster agreement is treated as
equivalent to geographically accurate localisation.

This script does that check on the manifest itself. It needs no model, no GPU
and no imagery -- only the coordinates the manifest already carries -- so it can
run before or alongside any GPU work.

What can and cannot be audited here
-----------------------------------
The manifest carries `latitude` and `longitude` for every query and gallery
entry, so the **distance** criterion is reproduced exactly, and swept over
several thresholds so its sensitivity is visible rather than assumed.

It does **not** carry a numeric compass heading. It carries a categorical
`viewpoint` field (for example "Forward"), which is not the same thing: two
forward-facing images can point in opposite directions along the same street.
The 40-degree viewing-angle half of the MSLS criterion therefore cannot be
reproduced from this manifest, and this script does not pretend otherwise. It
reports viewpoint agreement as a separate, weaker categorical statistic and
says so in its output. Closing that gap needs the raw MSLS `postprocessed.csv`
heading column, which is not in the staged subset.

Reading the output
------------------
The categories turn on the *intersection* of the two positive sets, not on
either set alone. Asking only whether some gallery entry is nearby would pass a
query whose cluster positive sits 200 m away merely because an unrelated image
happens to be next door, which is not the question. For each distance
threshold, over queries:

  cluster_supported    a cluster positive exists and at least one of them lies
                       within the threshold -- the two criteria agree
  cluster_unsupported  cluster positives exist but all lie beyond it -- the
                       benchmark would score a hit geography does not support
  geo_only             no cluster positive, but something lies within the
                       threshold -- geography supports a hit scored wrong
  neither              no cluster positive and nothing within the threshold
  jaccard              mean per-query overlap of the two positive sets

Reported alongside, and deliberately not part of that partition because a
query can be both: `has_unlabelled_neighbour`, the fraction of queries with a
gallery entry inside the threshold carrying a *different* cluster label.

A high `cluster_unsupported` rate at a small threshold is the finding that
would matter most: it would mean the reported accuracies are not measuring
geographically accurate localisation. A high `geo_only` or
`has_unlabelled_neighbour` rate means the benchmark is conservative, which
understates the attacker rather than overstating it.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

EARTH_RADIUS_M = 6371008.8

# The first four are mutually exclusive and partition the queries; the fifth is
# an independent flag and is deliberately excluded from the partition total.
_CATEGORIES = ("cluster_supported", "cluster_unsupported", "geo_only",
               "neither", "has_unlabelled_neighbour")


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres.

    Haversine rather than a planar approximation: the benchmark spans eight
    cities from Zurich to Manila, and a flat-earth approximation tuned to one
    latitude would misstate the threshold at the others.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2.0) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2)
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def load(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def has_coords(item: dict) -> bool:
    """A record is usable only if both coordinates are present and finite.

    Missing coordinates are counted and reported rather than silently treated
    as distance zero or dropped, because either would bias the agreement rate
    in a direction the reader cannot see.
    """
    try:
        lat, lon = float(item["latitude"]), float(item["longitude"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon)


def audit(records: Sequence[dict], thresholds: Sequence[float]) -> dict:
    gallery_index: Dict[str, dict] = {}
    for rec in records:
        for g in rec["gallery"]:
            gallery_index[g["gallery_id"]] = g
    gallery_ids = sorted(gallery_index)

    missing_query = sum(1 for r in records if not has_coords(r))
    missing_gallery = sum(1 for g in gallery_index.values() if not has_coords(g))

    usable_gallery = [g for g in gallery_ids if has_coords(gallery_index[g])]

    per_threshold: Dict[str, dict] = {}
    nearest_cluster_positive: List[float] = []

    for thr in thresholds:
        counts = Counter()
        jaccards: List[float] = []
        for rec in records:
            if not has_coords(rec):
                continue
            qlat, qlon = float(rec["latitude"]), float(rec["longitude"])
            cluster_pos = {g for g in gallery_ids
                           if gallery_index[g].get("place_id") == rec["place_id"]}
            geo_pos = set()
            for g in usable_gallery:
                gi = gallery_index[g]
                d = haversine_m(qlat, qlon,
                                float(gi["latitude"]), float(gi["longitude"]))
                if d <= thr:
                    geo_pos.add(g)
            # The question is not whether *something* is nearby, but whether
            # the entry the benchmark would score as correct is nearby. So the
            # categories turn on the intersection, not on either set alone: a
            # query whose cluster positive sits 200 m away is a disagreement
            # even if some unrelated gallery image happens to be next door.
            agree = cluster_pos & geo_pos
            if cluster_pos and agree:
                counts["cluster_supported"] += 1
            elif cluster_pos and not agree:
                counts["cluster_unsupported"] += 1
            elif not cluster_pos and geo_pos:
                counts["geo_only"] += 1
            else:
                counts["neither"] += 1
            # Independent of the above: nearby gallery entries that carry a
            # different cluster label, which the benchmark scores as wrong.
            if geo_pos - cluster_pos:
                counts["has_unlabelled_neighbour"] += 1
            union = cluster_pos | geo_pos
            if union:
                jaccards.append(len(agree) / len(union))
        n = sum(counts[k] for k in _CATEGORIES[:4])
        per_threshold[f"{thr:g}m"] = {
            "queries": n,
            **{k: counts[k] for k in _CATEGORIES},
            **{f"{k}_frac": (counts[k] / n if n else float("nan"))
               for k in _CATEGORIES},
            "mean_jaccard": (sum(jaccards) / len(jaccards)
                             if jaccards else float("nan")),
        }

    # Distance from each query to its nearest cluster-labelled positive: the
    # distribution that says what "same place" means in metres for this subset.
    viewpoint_agree = Counter()
    for rec in records:
        if not has_coords(rec):
            continue
        qlat, qlon = float(rec["latitude"]), float(rec["longitude"])
        best = math.inf
        for g in usable_gallery:
            gi = gallery_index[g]
            if gi.get("place_id") != rec["place_id"]:
                continue
            best = min(best, haversine_m(qlat, qlon, float(gi["latitude"]),
                                         float(gi["longitude"])))
            same = (gi.get("viewpoint", "") == rec.get("viewpoint", ""))
            viewpoint_agree["same" if same else "different"] += 1
        if math.isfinite(best):
            nearest_cluster_positive.append(best)

    nearest_cluster_positive.sort()

    def pct(p: float) -> float:
        if not nearest_cluster_positive:
            return float("nan")
        k = min(len(nearest_cluster_positive) - 1,
                max(0, int(round(p * (len(nearest_cluster_positive) - 1)))))
        return nearest_cluster_positive[k]

    return {
        "queries": len(records),
        "gallery": len(gallery_ids),
        "queries_missing_coordinates": missing_query,
        "gallery_missing_coordinates": missing_gallery,
        "per_threshold": per_threshold,
        "nearest_cluster_positive_distance_m": {
            "n": len(nearest_cluster_positive),
            "p05": pct(0.05), "median": pct(0.50),
            "p95": pct(0.95),
            "max": (nearest_cluster_positive[-1]
                    if nearest_cluster_positive else float("nan")),
        },
        "viewpoint_agreement_among_cluster_positives": dict(viewpoint_agree),
        "heading_criterion": (
            "NOT AUDITED: the manifest carries a categorical 'viewpoint' field "
            "but no numeric compass heading, so the MSLS 40-degree "
            "viewing-angle criterion cannot be reproduced from it. The "
            "viewpoint counts above are a weaker categorical proxy and must "
            "not be reported as the angular criterion."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[10.0, 25.0, 50.0, 100.0],
                    help="Distance thresholds in metres; 25 is the MSLS "
                         "single-image positive criterion.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    records = load(Path(args.manifest))
    result = audit(records, args.thresholds)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cluster_vs_geographic.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    print(f"queries={result['queries']} gallery={result['gallery']} "
          f"missing coords: q={result['queries_missing_coordinates']} "
          f"g={result['gallery_missing_coordinates']}")
    print(f"\n{'threshold':>10s} {'supported':>10s} {'unsupported':>12s} "
          f"{'geo_only':>9s} {'neither':>8s} {'unlab_nbr':>10s} "
          f"{'jaccard':>8s}")
    for thr, st in result["per_threshold"].items():
        print(f"{thr:>10s} {st['cluster_supported_frac']:10.3f} "
              f"{st['cluster_unsupported_frac']:12.3f} "
              f"{st['geo_only_frac']:9.3f} {st['neither_frac']:8.3f} "
              f"{st['has_unlabelled_neighbour_frac']:10.3f} "
              f"{st['mean_jaccard']:8.3f}")
    d = result["nearest_cluster_positive_distance_m"]
    print(f"\nnearest cluster positive (m): p05={d['p05']:.1f} "
          f"median={d['median']:.1f} p95={d['p95']:.1f} max={d['max']:.1f}")
    print(f"\n{result['heading_criterion']}")
    print(f"\n[done] wrote {out / 'cluster_vs_geographic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
