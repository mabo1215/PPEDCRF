"""Build clips of neighbouring frames for the MSLS queries.

Why this exists. Every retrieval number in this work is measured on a single
released frame, but the scenario that motivates the work uploads video. An
attacker holding the clip can embed several released frames and pool them
before ranking, and a perturbation whose direction is re-derived per frame
does not point the same way in each of them: pooling averages differently
oriented displacements while the location signal they hide is common to every
frame. Whether that recovers the retrieval accuracy the frame-level numbers
report is an empirical question, and answering it needs the neighbours of each
query frame.

MSLS supplies them. ``seq_info.csv`` gives every image its sequence key and
frame number, so the frames adjacent in capture order to a query frame are
recoverable without any new imagery. This script writes, for each query of an
existing manifest, an ordered clip: the query frame first, then its nearest
neighbours in the same sequence by absolute frame distance.

Neighbours are restricted by default to frames carrying the query's own
``unique_cluster`` place label. The restriction is not cosmetic: a frame a few
seconds away can belong to the neighbouring place, and pooling it in would let
the attacker's top-1 move to a place that is arguably correct while the
protocol scores it as a miss, understating the attacker this experiment exists
to strengthen. Keeping the clip inside one place keeps the ground truth
unambiguous. ``--allow-cross-place`` relaxes it and the manifest records which
rule each clip satisfied, so the choice can be audited.

The output carries identifiers and relative paths only; no imagery is copied.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_side(msls_root: Path, city: str, side: str):
    """(key -> sequence, frame number) and (key -> place cluster) for one side."""
    seq, place = {}, {}
    sinfo = msls_root / city / side / "seq_info.csv"
    post = msls_root / city / side / "postprocessed.csv"
    with open(sinfo, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            seq[row["key"]] = (row["sequence_key"], int(row["frame_number"]))
    with open(post, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            place[row["key"]] = row["unique_cluster"]
    return seq, place


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="Existing query manifest (JSONL) to build clips for.")
    ap.add_argument("--msls_root", required=True,
                    help="Extracted MSLS root containing train_val/.")
    ap.add_argument("--split", default="train_val")
    ap.add_argument("--clip_len", type=int, default=7,
                    help="Frames per clip, including the query frame.")
    ap.add_argument("--allow-cross-place", action="store_true",
                    help="Admit neighbours that carry a different place label.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.msls_root) / args.split
    records = [json.loads(line) for line in
               open(args.manifest, encoding="utf-8")]
    cities = sorted({r["city"] for r in records})
    seq, place = {}, {}
    for city in cities:
        s, p = read_side(root, city, "query")
        seq.update({k: (city, *v) for k, v in s.items()})
        place.update(p)

    by_sequence = defaultdict(list)
    for key, (city, skey, frame) in seq.items():
        by_sequence[(city, skey)].append((frame, key))
    for frames in by_sequence.values():
        frames.sort()

    clips, sizes, no_meta = {}, defaultdict(int), 0
    for rec in records:
        key = Path(rec["query_path"]).stem
        if key not in seq:
            no_meta += 1
            continue
        city, skey, frame = seq[key]
        neighbours = by_sequence[(city, skey)]
        # Nearest in capture order, the query frame itself first (distance 0).
        ordered = sorted(neighbours, key=lambda fk: (abs(fk[0] - frame), fk[0]))
        if not args.allow_cross_place:
            ordered = [fk for fk in ordered
                       if place.get(fk[1]) == place.get(key)]
        chosen = ordered[: args.clip_len]
        clips[rec["query_id"]] = {
            "frames": [f"{args.split}/{city}/query/images/{k}.jpg"
                       for _, k in chosen],
            "frame_offsets": [f - frame for f, _ in chosen],
            "sequence_key": skey,
            "same_place_only": not args.allow_cross_place,
        }
        sizes[len(chosen)] += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"clip_len": args.clip_len,
                   "same_place_only": not args.allow_cross_place,
                   "clips": clips}, fh, indent=1, sort_keys=True)

    print(f"[clips] {len(clips)} queries with clips, {no_meta} without "
          f"sequence metadata")
    for n in sorted(sizes, reverse=True):
        print(f"[clips]   {sizes[n]:4d} queries reach {n} frame(s)")
    full = sum(v for n, v in sizes.items() if n >= args.clip_len)
    print(f"[clips] {full} of {len(records)} queries support the full "
          f"{args.clip_len}-frame clip")
    print(f"[clips] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
