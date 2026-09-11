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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()

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
