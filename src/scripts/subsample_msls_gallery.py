r"""Build gallery-size variants of a place-labelled manifest.

The audit reports every real-data number at one gallery size, which leaves open
whether either axis behaves differently when the attacker's database grows. A
sweep needs manifests that differ in gallery size and in nothing else.

This subsamples the gallery of an existing manifest rather than rebuilding from
the dataset, for a reason worth stating: rebuilding would need the corpus
metadata that carries place labels, and adding images whose place is unknown
would let a true positive be scored as a miss. Every image kept here keeps the
label it already had, so smaller galleries are exactly the original with some
\emph{negatives} removed.

Positives are never dropped. A query whose positives alone exceed the target is
kept with all of them, so the realised gallery can exceed the target slightly;
the script reports that rather than silently truncating a positive set.
"""
from __future__ import annotations

import argparse
import json
import zlib
from collections import OrderedDict
from pathlib import Path


def city_of(gallery_id: str) -> str:
    return gallery_id.split("_")[0]


def nested(args) -> int:
    """Hold the query set fixed and grow the gallery by adding whole cities.

    The gallery of this benchmark contains no pure distractors: every image is
    the answer to some query, so it cannot be shrunk without taking a query's
    answer away. It can be grown safely in one direction only -- by other
    cities, because a place never spans two.
    """
    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    shared = OrderedDict()
    for r in records:
        for g in r["gallery"]:
            shared.setdefault(g["gallery_id"], g)
    by_city = OrderedDict()
    for gid, g in shared.items():
        by_city.setdefault(city_of(gid), []).append(g)
    order = sorted(by_city)
    print(f"cities: " + ", ".join(f"{c}={len(by_city[c])}" for c in order))

    for qcity in args.nested_cities:
        queries = [r for r in records if r["query_id"].split("_")[0] == qcity]
        if not queries:
            print(f"[skip] {qcity}: no queries")
            continue
        others = [c for c in order if c != qcity]
        for level in args.levels:
            cities = [qcity] + others[: max(0, level - 1)]
            keep = {g["gallery_id"] for c in cities for g in by_city[c]}
            out = Path(f"{args.out_prefix}{qcity}_L{level}.jsonl")
            with out.open("w", encoding="utf-8") as fh:
                for r in queries:
                    sub = [g for g in r["gallery"] if g["gallery_id"] in keep]
                    assert any(g["place_id"] == r["place_id"] for g in sub), r["query_id"]
                    rec = dict(r); rec["gallery"] = sub
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
            print(f"[done] {out.name}: {len(queries)} queries, gallery {len(keep)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--nested_cities", nargs="*", default=[],
                    help="Query cities for the nested sweep. Place labels are "
                         "scoped to a city and no place spans two, so images "
                         "from other cities are distractors that cannot be "
                         "the answer -- which is what makes a larger gallery "
                         "larger without making it wrong.")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="Cities in the gallery at each level, the query's own "
                         "city first.")
    args = ap.parse_args()
    if args.nested_cities:
        return nested(args)

    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    # One shared gallery across queries, which is how the benchmark reads it.
    shared = OrderedDict()
    for r in records:
        for g in r["gallery"]:
            shared.setdefault(g["gallery_id"], g)
    positives = {g["gallery_id"] for r in records for g in r["gallery"]
                 if g["place_id"] == r["place_id"]}
    negatives = [k for k in shared if k not in positives]
    # Deterministic order that is not the file order, so a smaller gallery is a
    # representative subset rather than a prefix of however the file was built.
    negatives.sort(key=lambda k: zlib.crc32(k.encode()))
    print(f"source: {len(records)} queries, {len(shared)} gallery "
          f"({len(positives)} positive, {len(negatives)} negative)")

    for size in sorted(args.sizes):
        if size < len(positives):
            print(f"[skip] {size}: fewer than the {len(positives)} positives")
            continue
        keep = set(positives) | set(negatives[: size - len(positives)])
        out = Path(f"{args.out_prefix}{size}.jsonl")
        n_rows = 0
        with out.open("w", encoding="utf-8") as fh:
            for r in records:
                sub = [g for g in r["gallery"] if g["gallery_id"] in keep]
                # Every query must keep at least its own positives.
                assert any(g["place_id"] == r["place_id"] for g in sub), r["query_id"]
                rec = dict(r); rec["gallery"] = sub
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
                n_rows += 1
        print(f"[done] {out.name}: {n_rows} queries, gallery {len(keep)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
