"""Build a place-labelled retrieval manifest from KITTI-360 poses.

Why this exists. Every real-place result in the manuscript comes from one
dataset, MSLS, under one capture condition. A negative result about a design
axis needs to survive a change of dataset, and this builds the second one from
data the project already holds.

Why loop closure rather than a second traversal. The cleanest analogue of MSLS
-- separate query and database traversals of the same streets -- is not
available here: of the eleven KITTI-360 drives, only 0000 and 0002 have
imagery on disk, and those two are spatially disjoint. Every cross-traversal
overlap the poses show involves a drive whose images were never downloaded.
What is available in abundance is within-drive revisit: in drive 0000 roughly
5,600 frames have another frame within 25 m at least a minute away in time.

How a query is kept honest. A place is a cell of the ground plane that the
drive enters twice with a long gap in between. The query is taken from the
later visit and its positives from the earlier one, and every query-positive
pair is separated by at least ``--min-gap`` frames, so no query can be matched
by a frame that merely sits next to it in the sequence. Splitting the drive at
one global time instead -- early block as gallery, late block as queries --
was tried first and is much worse: it keeps only the revisits that happen to
straddle the cut, which on these two drives is five or six places out of the
274 the data actually contains.

Place labels. MSLS ships its own cluster label; here there is none, so the
cell is the label, and a query shares it with the earlier-visit frames of the
same cell by construction. A cell is coarser than a radius, so the label and
the geometry do not agree perfectly at cell boundaries; the disagreement is
measured against the positive radius and reported, exactly as the manuscript
audits the MSLS cluster label against distance.

Both drives are combined into one benchmark: the two are spatially disjoint,
so their clusters cannot collide, and the union gives a query count and gallery
size comparable with the wide MSLS manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]

DRIVES = ["2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync"]
IMAGE_TEMPLATE = "data_2d_raw/{drive}/image_00/data_rect/{frame:010d}.png"


def load_poses(root: Path, drive: str) -> Tuple[np.ndarray, np.ndarray]:
    """Frame indices and world (x, y) for one drive."""
    raw = np.loadtxt(root / "data_poses" / drive / "poses.txt")
    if raw.ndim == 1:
        raw = raw[None]
    frames = raw[:, 0].astype(int)
    xy = raw[:, 1:].reshape(-1, 3, 4)[:, :2, 3]
    return frames, xy


def label_by_nearest_query(g_xy: np.ndarray, q_xy: np.ndarray,
                           radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Give each gallery frame the nearest query's place, or its own.

    A frame within ``radius`` of some query belongs to the nearest of them; one
    outside every radius is a distractor with a label nothing else shares. The
    only way a geographic positive can then miss its query's label is by lying
    closer to a different query, which query spacing controls.
    """
    d2 = ((g_xy[:, None, :] - q_xy[None, :, :]) ** 2).sum(-1)
    nearest = d2.argmin(1)
    within = d2[np.arange(len(g_xy)), nearest] < radius ** 2
    return nearest, within


def merge_adjacent(keys: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """Union revisited cells that touch, so one physical place is one label.

    Two revisited cells side by side are the same street corner seen from a few
    metres apart. Left separate they generate each other's false negatives: a
    query in one cell has the other's frames well inside the positive radius
    and scores them wrong. Merging is a correction, not a coarsening.
    """
    parent = {k: k for k in keys}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    key_set = set(keys)
    for (cx, cy) in keys:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                other = (cx + dx, cy + dy)
                if other in key_set:
                    ra, rb = find((cx, cy)), find(other)
                    if ra != rb:
                        parent[ra] = rb
    roots = {}
    out = {}
    for k in keys:
        r = find(k)
        out[k] = roots.setdefault(r, len(roots))
    return out


def build_drive(root: Path, drive: str, args):
    """Queries and gallery entries for one drive, paired by revisit."""
    frames, xy = load_poses(root, drive)
    exists = np.array([(root / IMAGE_TEMPLATE.format(drive=drive, frame=int(f))).is_file()
                       for f in frames])
    frames, xy = frames[exists], xy[exists]
    if not len(frames):
        return [], [], [], frames, xy, drive

    cell = np.floor(xy / args.cell).astype(int)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i, (cx, cy) in enumerate(cell):
        buckets.setdefault((int(cx), int(cy)), []).append(i)

    revisited = [k for k, m in buckets.items()
                 if frames[max(m, key=lambda i: frames[i])]
                 - frames[min(m, key=lambda i: frames[i])] >= args.min_gap]
    groups = merge_adjacent(revisited)

    members: Dict[int, List[int]] = {}
    for k in revisited:
        members.setdefault(groups[k], []).extend(buckets[k])

    gallery: List[dict] = []
    queries: List[dict] = []
    used = set()
    for gid, idxs in sorted(members.items()):
        order = sorted(idxs, key=lambda i: frames[i])
        # A merged place is a stretch of road the drive enters twice. Split it
        # at its largest temporal gap: the later visit supplies queries, the
        # earlier one supplies the gallery positives, and the gap between them
        # is what stops a query matching a neighbouring frame.
        f = frames[order]
        if len(order) < 2:
            continue
        cut = int(np.argmax(np.diff(f))) + 1
        early, late = order[:cut], order[cut:]
        if not early or not late or frames[late[0]] - frames[early[-1]] < args.min_gap:
            continue
        place = f"kitti360:{drive[-9:-5]}_p{gid}"
        pos = early[::max(1, len(early) // args.positives_per_place)]
        pos = pos[:args.positives_per_place]
        for i in pos:
            gallery.append({"drive": drive, "frame": int(frames[i]),
                            "xy": xy[i], "place_id": place})
        # queries spaced along the later visit so they are distinct viewpoints
        chosen: List[int] = []
        for i in late:
            if chosen and (((xy[i] - xy[chosen[-1]]) ** 2).sum()
                           < args.query_spacing ** 2):
                continue
            chosen.append(i)
            if len(chosen) >= args.queries_per_place:
                break
        for i in chosen:
            queries.append({"drive": drive, "frame": int(frames[i]),
                            "xy": xy[i], "place_id": place,
                            "n_positives": len(pos),
                            "gap": int(frames[i] - frames[pos[-1]])})
        used.update(idxs)

    pool = [i for i in range(len(frames)) if i not in used][::args.distractor_stride]
    print(f"[{drive[-14:]}] {len(revisited)} revisited cells -> {len(queries)} "
          f"merged places, {len(gallery)} positives, {len(pool)} distractor candidates")
    return gallery, queries, pool, frames, xy, drive


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/mnt/g/work/datasets/kitti360")
    ap.add_argument("--output", default=str(REPO / "src" / "outputs" / "e1_kitti360"
                                            / "manifest_loop.jsonl"))
    ap.add_argument("--radius", type=float, default=25.0,
                    help="Positive radius the label is audited against; 25 m is "
                         "the threshold MSLS uses.")
    ap.add_argument("--cell", type=float, default=25.0,
                    help="Side of the ground cell that defines a place.")
    ap.add_argument("--min-gap", dest="min_gap", type=int, default=600,
                    help="Minimum frame separation between a query and any of "
                         "its positives; 600 frames is about a minute.")
    ap.add_argument("--positives-per-place", dest="positives_per_place", type=int,
                    default=8)
    ap.add_argument("--queries-per-place", dest="queries_per_place", type=int,
                    default=25)
    ap.add_argument("--query-spacing", dest="query_spacing", type=float,
                    default=6.0,
                    help="Minimum metres between queries of one place, so they "
                         "are distinct viewpoints rather than the same frame.")
    ap.add_argument("--gallery-size", dest="gallery_size", type=int, default=2000)
    ap.add_argument("--distractor-stride", dest="distractor_stride", type=int,
                    default=7)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()
    root = Path(args.root)
    rng = np.random.default_rng(args.seed)

    gallery: List[dict] = []
    queries: List[dict] = []
    distractors: List[dict] = []
    for drive in DRIVES:
        g, q, pool, frames, xy, name = build_drive(root, drive, args)
        gallery += g
        queries += q
        distractors += [{"drive": name, "frame": int(frames[i]), "xy": xy[i],
                         "place_id": f"kitti360:{name[-9:-5]}_d{int(frames[i])}"}
                        for i in pool]
    if not queries:
        print("[fail] no revisited place found")
        return 1

    # Fill the gallery to its target with distractors that are not near any query.
    q_xy = np.stack([q["xy"] for q in queries])
    room = max(0, args.gallery_size - len(gallery))
    if room and distractors:
        d_xy = np.stack([d["xy"] for d in distractors])
        far = (((d_xy[:, None, :] - q_xy[None, :, :]) ** 2).sum(-1)
               >= args.radius ** 2).all(1)
        eligible = [d for d, ok in zip(distractors, far) if ok]
        pick = rng.permutation(len(eligible))[:room]
        gallery += [eligible[i] for i in sorted(pick)]
        print(f"[fill ] {len(eligible)} distractors clear of every query, "
              f"{min(room, len(eligible))} added")

    g_xy = np.stack([g["xy"] for g in gallery])
    d2 = ((g_xy[:, None, :] - q_xy[None, :, :]) ** 2).sum(-1)
    geo = d2 < args.radius ** 2
    same = np.array([[g["place_id"] == q["place_id"] for q in queries]
                     for g in gallery])
    total_geo, agree = int(geo.sum()), int((geo & same).sum())
    label_only = int((same & ~geo).sum())
    per_query = same.sum(0)
    print(f"[audit] {len(queries)} queries, {len(gallery)} gallery images, "
          f"{len(set(q['place_id'] for q in queries))} places")
    print(f"[audit] labelled positives per query: median "
          f"{int(np.median(per_query))}, min {int(per_query.min())}")
    print(f"[audit] of {total_geo} positives within {args.radius:g} m, {agree} "
          f"({100.0 * agree / max(total_geo, 1):.1f}%) share the query's label")
    print(f"[audit] {label_only} labelled positives lie beyond {args.radius:g} m "
          f"(the cell is coarser than the radius)")
    gaps = [q["gap"] for q in queries]
    print(f"[audit] query-to-positive frame gap: min {min(gaps)}, "
          f"median {int(np.median(gaps))}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    gallery_entries = [{
        "gallery_id": f"{g['drive'][-9:-5]}_{g['frame']:010d}",
        "path": IMAGE_TEMPLATE.format(drive=g["drive"], frame=g["frame"]),
        "place_id": g["place_id"], "frame_number": str(g["frame"]),
        "latitude": float(g["xy"][0]), "longitude": float(g["xy"][1]),
        "sequence_id": g["drive"], "viewpoint": "Forward", "illumination": "day",
        "season": "", "weather": "",
    } for g in gallery]
    with out.open("w", encoding="utf-8") as handle:
        for q in queries:
            handle.write(json.dumps({
                "query_id": f"{q['drive'][-9:-5]}_query_{q['frame']:010d}",
                "query_path": IMAGE_TEMPLATE.format(drive=q["drive"], frame=q["frame"]),
                "place_id": q["place_id"], "frame_number": str(q["frame"]),
                "latitude": float(q["xy"][0]), "longitude": float(q["xy"][1]),
                "sequence_id": q["drive"], "subtask": "loop",
                "viewpoint": "Forward", "illumination": "day",
                "season": "", "weather": "",
                "gallery": gallery_entries,
            }) + "\n")
    meta = {
        "dataset": "KITTI-360", "drives": DRIVES,
        "construction": "within-drive revisit; query from the later visit of a "
                        "ground cell, positives from the earlier visit",
        "positive_radius_m": args.radius, "place_cell_m": args.cell,
        "min_query_positive_frame_gap": args.min_gap,
        "query_count": len(queries), "gallery_count": len(gallery),
        "place_count": len(set(q["place_id"] for q in queries)),
        "positives_within_radius": total_geo,
        "positives_within_radius_sharing_label": agree,
        "labelled_positives_beyond_radius": label_only,
        "place_labels": "ground cell of side place_cell_m",
        "coordinates": "KITTI-360 world frame, metres; the latitude/longitude "
                       "fields carry x and y so the schema matches the MSLS "
                       "manifest the runner already reads",
        "image_native_size": "1408x376, resized by the pipeline to its working "
                             "resolution like every other frame",
        "image_files_are_not_copied": True,
    }
    Path(str(out).replace(".jsonl", ".metadata.json")).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
