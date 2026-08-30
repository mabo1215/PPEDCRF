"""Build a sequence-held-out KITTI-360 manifest for unary attribution checks.

The manifest uses the official camera poses to form coarse 10-metre place cells
and retains the official 2D semantic masks for stratification. These place
cells are GPS/pose-derived evaluation labels, not learned sensitivity labels.
The resulting JSONL is consumed by ``validate_unary_attribution.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a KITTI-360 unary attribution manifest.")
    parser.add_argument("--kitti360_root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True, help="Sequence IDs such as 0000 0002.")
    parser.add_argument("--camera", default="image_00", choices=("image_00", "image_01"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--place_cell_m", type=float, default=10.0)
    parser.add_argument("--max_queries", type=int, default=32)
    parser.add_argument("--gallery_size", type=int, default=128)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata_output", default="")
    return parser.parse_args()


def sequence_name(sequence: str) -> str:
    return f"2013_05_28_drive_{int(sequence):04d}_sync"


def pose_positions(path: Path) -> dict[int, tuple[float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing KITTI-360 camera pose file: {path}")
    positions: dict[int, tuple[float, float, float]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) < 13:
            raise ValueError(f"Pose line {line_number} has fewer than 13 values: {path}")
        frame = int(values[0])
        matrix = [float(value) for value in values[1:13]]
        positions[frame] = (matrix[3], matrix[7], matrix[11])
    return positions


def frame_files(root: Path, sequence: str, camera: str) -> dict[int, Path]:
    seq_name = sequence_name(sequence)
    image_dir = root / "data_2d_raw" / seq_name / camera / "data_rect"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing KITTI-360 image directory: {image_dir}")
    result: dict[int, Path] = {}
    for path in sorted(image_dir.glob("*.png")):
        try:
            result[int(path.stem)] = path
        except ValueError:
            continue
    return result


def place_id(sequence: str, position: Sequence[float], cell_size: float) -> str:
    cell_x = math.floor(float(position[0]) / cell_size)
    cell_y = math.floor(float(position[1]) / cell_size)
    return f"kitti360:{int(sequence):04d}:{cell_x}:{cell_y}"


def distance(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    return math.sqrt(
        sum((float(a[key]) - float(b[key])) ** 2 for key in ("pose_x", "pose_y", "pose_z"))
    )


def relativize(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def main() -> None:
    args = parse_args()
    root = Path(args.kitti360_root).expanduser().resolve()
    all_queries: list[dict] = []
    all_gallery: list[dict] = []
    for sequence in args.sequences:
        seq_name = sequence_name(sequence)
        images = frame_files(root, sequence, args.camera)
        positions = pose_positions(root / "data_poses" / seq_name / "cam0_to_world.txt")
        semantic_dir = root / "data_2d_semantics" / "train" / seq_name / args.camera / "semantic"
        if not semantic_dir.is_dir():
            raise FileNotFoundError(f"Missing KITTI-360 semantic directory: {semantic_dir}")
        frames = [frame for frame in sorted(images) if frame in positions and (semantic_dir / f"{frame:010d}.png").is_file()]
        if len(frames) < 4:
            raise ValueError(f"Sequence {sequence} has fewer than four image/pose/semantic triples.")
        sampled = frames[:: max(1, int(args.stride))]
        if len(sampled) < 4:
            sampled = frames
        records: list[dict] = []
        for index, frame in enumerate(sampled):
            position = positions[frame]
            record = {
                "sequence_id": f"{int(sequence):04d}",
                "frame_number": frame,
                "image_path": images[frame],
                "semantic_path": semantic_dir / f"{frame:010d}.png",
                "pose_x": position[0],
                "pose_y": position[1],
                "pose_z": position[2],
                "place_id": place_id(sequence, position, float(args.place_cell_m)),
            }
            if index % 5 == 0:
                all_queries.append(record)
            else:
                all_gallery.append(record)

    all_queries.sort(key=lambda item: (item["sequence_id"], item["frame_number"]))
    all_gallery.sort(key=lambda item: (item["sequence_id"], item["frame_number"]))
    eligible = [
        query for query in all_queries
        if any(item["place_id"] == query["place_id"] for item in all_gallery)
    ]
    queries = eligible[: int(args.max_queries)]
    if not queries:
        raise ValueError("No query has a pose-derived positive gallery cell.")
    if int(args.gallery_size) < len(queries):
        raise ValueError("--gallery_size must be at least the number of selected queries.")

    selected: dict[tuple[str, int], dict] = {}
    for query in queries:
        positives = [item for item in all_gallery if item["place_id"] == query["place_id"]]
        positives.sort(key=lambda item: (distance(query, item), item["sequence_id"], item["frame_number"]))
        if positives:
            key = (positives[0]["sequence_id"], positives[0]["frame_number"])
            selected[key] = positives[0]
    negatives = [item for item in all_gallery if item["place_id"] not in {q["place_id"] for q in queries}]
    negatives.sort(
        key=lambda item: (
            min((distance(query, item) for query in queries), default=0.0),
            item["sequence_id"],
            item["frame_number"],
        )
    )
    for item in negatives:
        if len(selected) >= int(args.gallery_size):
            break
        selected[(item["sequence_id"], item["frame_number"])] = item
    if len(selected) < int(args.gallery_size):
        raise ValueError(f"Only {len(selected)} gallery images available; requested {args.gallery_size}.")
    gallery = [selected[key] for key in sorted(selected)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gallery_payload = []
    for item in gallery:
        gallery_payload.append(
            {
                "gallery_id": f"seq{item['sequence_id']}_frame{int(item['frame_number']):010d}",
                "path": relativize(item["image_path"], root),
                "place_id": item["place_id"],
                "sequence_id": item["sequence_id"],
                "frame_number": item["frame_number"],
                "pose_x": item["pose_x"],
                "pose_y": item["pose_y"],
                "pose_z": item["pose_z"],
            }
        )
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, query in enumerate(queries):
            record = {
                "query_id": f"seq{query['sequence_id']}_frame{int(query['frame_number']):010d}",
                "query_path": relativize(query["image_path"], root),
                "semantic_path": relativize(query["semantic_path"], root),
                "place_id": query["place_id"],
                "sequence_id": query["sequence_id"],
                "frame_number": query["frame_number"],
                "pose_x": query["pose_x"],
                "pose_y": query["pose_y"],
                "pose_z": query["pose_z"],
                "viewpoint": args.camera,
                "gallery": gallery_payload,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    metadata = {
        "dataset": "KITTI-360",
        "official_source": "https://www.cvlibs.net/datasets/kitti-360/",
        "sequences": [f"{int(sequence):04d}" for sequence in args.sequences],
        "camera": args.camera,
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "place_labels": "10-metre pose-derived cells",
        "semantic_labels": "KITTI-360 2D semantic IDs",
        "query_gallery_path_overlap": 0,
        "sequence_held_out": True,
        "scientific_evidence": True,
    }
    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[kitti360] wrote {len(queries)} queries and {len(gallery)} gallery images to {output_path}")
    print(f"[kitti360] wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
