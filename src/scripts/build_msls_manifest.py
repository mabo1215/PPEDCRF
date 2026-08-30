"""Build a ground-truth MSLS manifest for the PPEDCRF geotagged VPR runner.

The official MSLS release stores query and database metadata next to the image
files. This adapter converts one city/subtask into the repository's JSONL
manifest without copying or redistributing dataset images. The supplied
``unique_cluster`` field is retained as the place label; if it is unavailable,
the builder fails instead of silently inventing a geographic label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SUBTASK_ALIASES = {
    "all": "all",
    "summer2winter": "s2w",
    "winter2summer": "w2s",
    "old2new": "o2n",
    "new2old": "n2o",
    "day2night": "d2n",
    "night2day": "n2d",
    "s2w": "s2w",
    "w2s": "w2s",
    "o2n": "o2n",
    "n2o": "n2o",
    "d2n": "d2n",
    "n2d": "n2d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest from official MSLS metadata.")
    parser.add_argument("--msls_root", required=True, help="Extracted MSLS root containing train_val/ and/or test/.")
    parser.add_argument("--split", choices=("train_val", "test"), default="train_val")
    parser.add_argument("--cities", nargs="+", required=True)
    parser.add_argument("--subtask", default="all", help="all, s2w, w2s, o2n, n2o, d2n, or n2d.")
    parser.add_argument("--max_queries", type=int, default=200)
    parser.add_argument("--max_gallery", type=int, default=1000)
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--metadata_output", default="", help="Optional JSON metadata sidecar.")
    parser.add_argument("--include_panos", action="store_true")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing MSLS metadata file: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def row_key(row: Mapping[str, str]) -> str:
    for name in ("key", "image_id", "id"):
        value = row.get(name, "")
        if value:
            return str(value)
    first = next(iter(row), "")
    value = row.get(first, "") if first else ""
    if not value:
        raise ValueError("MSLS metadata row has no usable key.")
    return str(value)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def as_float(value: object) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def pick(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value not in (None, ""):
            return str(value)
    return ""


def find_image(images_dir: Path, key: str) -> Path:
    direct = images_dir / (key if Path(key).suffix else f"{key}.jpg")
    if direct.is_file():
        return direct
    matches = sorted(images_dir.glob(f"{Path(key).name}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve MSLS image key {key!r} under {images_dir}")


def load_side(city_root: Path, side: str, split: str, requested_subtask: str, include_panos: bool) -> list[dict]:
    side_root = city_root / side
    post_rows = read_csv_rows(side_root / "postprocessed.csv")
    raw_rows = read_csv_rows(side_root / "raw.csv")
    seq_rows = read_csv_rows(side_root / "seq_info.csv")
    index_path = side_root / "subtask_index.csv"
    subtask_rows = read_csv_rows(index_path) if index_path.is_file() else []
    subtask_column = SUBTASK_ALIASES.get(requested_subtask.lower())
    if subtask_column is None:
        raise ValueError(f"Unknown MSLS subtask: {requested_subtask}")

    raw_by_key = {row_key(row): row for row in raw_rows}
    seq_by_key = {row_key(row): row for row in seq_rows}
    records: list[dict] = []
    for index, post in enumerate(post_rows):
        key = row_key(post)
        raw = raw_by_key.get(key, raw_rows[index] if index < len(raw_rows) else {})
        seq = seq_by_key.get(key, seq_rows[index] if index < len(seq_rows) else {})
        if subtask_column != "all" and subtask_rows:
            flag = subtask_rows[index].get(subtask_column, "") if index < len(subtask_rows) else ""
            if not as_bool(flag):
                continue
        if not include_panos and as_bool(raw.get("pano", "")):
            continue

        cluster = pick(post, "unique_cluster", "cluster", "place_id")
        if not cluster:
            raise ValueError(
                f"MSLS row {key} has no unique_cluster. Refusing to create a synthetic place label."
            )
        latitude = as_float(pick(raw, "lat", "latitude"))
        longitude = as_float(pick(raw, "lon", "longitude", "lng"))
        easting = as_float(pick(post, "easting", "utm_easting"))
        northing = as_float(pick(post, "northing", "utm_northing"))
        records.append(
            {
                "key": key,
                "image_path": str(find_image(side_root / "images", key)),
                "place_id": f"{city_root.name}:{cluster}",
                "latitude": latitude,
                "longitude": longitude,
                "easting": easting,
                "northing": northing,
                "sequence_id": pick(seq, "sequence_id", "seq_id"),
                "frame_number": pick(seq, "frame_number", "frame"),
                "captured_at": pick(raw, "captured_at", "timestamp", "date"),
                "viewpoint": pick(post, "view_direction", "viewpoint"),
                "illumination": "night" if as_bool(post.get("night", "")) else "day",
                "season": pick(post, "season"),
                "weather": pick(post, "weather"),
                "city": city_root.name,
                "side": side,
            }
        )
    return records


def distance(a: Mapping[str, object], b: Mapping[str, object]) -> float:
    ax, ay = a.get("easting"), a.get("northing")
    bx, by = b.get("easting"), b.get("northing")
    if all(isinstance(value, (int, float)) for value in (ax, ay, bx, by)):
        return math.hypot(float(ax) - float(bx), float(ay) - float(by))
    return 0.0


def choose_gallery(queries: Sequence[dict], database: Sequence[dict], max_gallery: int) -> list[dict]:
    if max_gallery < len(queries):
        raise ValueError("--max_gallery must be at least --max_queries after query filtering.")
    query_places = {query["place_id"] for query in queries}
    positives_by_query: dict[str, list[dict]] = {}
    for query in queries:
        positives = [item for item in database if item["place_id"] == query["place_id"]]
        if not positives:
            continue
        positives_by_query[query["key"]] = sorted(positives, key=lambda item: (distance(query, item), item["key"]))

    selected: dict[str, dict] = {}
    for query in queries:
        positives = positives_by_query.get(query["key"], [])
        if positives:
            selected[positives[0]["key"]] = positives[0]
    if len(selected) > max_gallery:
        raise ValueError("The gallery budget cannot contain one positive for every query.")

    remaining_positive = [
        item for item in database if item["place_id"] in query_places and item["key"] not in selected
    ]
    remaining_positive.sort(key=lambda item: item["key"])
    for item in remaining_positive:
        if len(selected) >= max_gallery:
            break
        selected[item["key"]] = item

    negatives = [item for item in database if item["place_id"] not in query_places]
    negatives.sort(
        key=lambda item: (
            min((distance(query, item) for query in queries), default=0.0),
            item["key"],
        )
    )
    for item in negatives:
        if len(selected) >= max_gallery:
            break
        selected[item["key"]] = item
    if len(selected) < max_gallery:
        raise ValueError(f"Only {len(selected)} valid gallery images are available; requested {max_gallery}.")
    return [selected[key] for key in sorted(selected)]


def relativize(path: str, root: Path) -> str:
    value = Path(path).resolve()
    try:
        return value.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(value)


def main() -> None:
    args = parse_args()
    root = Path(args.msls_root).expanduser().resolve()
    split_root = root / args.split
    all_queries: list[dict] = []
    all_database: list[dict] = []
    for city in args.cities:
        city_root = split_root / city
        if not city_root.is_dir():
            raise FileNotFoundError(f"Missing MSLS city directory: {city_root}")
        all_queries.extend(load_side(city_root, "query", args.split, args.subtask, args.include_panos))
        all_database.extend(load_side(city_root, "database", args.split, args.subtask, args.include_panos))

    all_queries.sort(key=lambda item: (item["city"], item["key"]))
    eligible = [query for query in all_queries if any(item["place_id"] == query["place_id"] for item in all_database)]
    if not eligible:
        raise ValueError("No query has a GPS-backed positive in the database split.")
    queries = eligible[: int(args.max_queries)]
    gallery = choose_gallery(queries, all_database, int(args.max_gallery))
    gallery_by_id = {
        f"{item['city']}_database_{item['key']}": item
        for item in gallery
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for query in queries:
            query_id = f"{query['city']}_query_{query['key']}"
            record = {
                "query_id": query_id,
                "query_path": relativize(query["image_path"], root),
                "place_id": query["place_id"],
                "latitude": query["latitude"],
                "longitude": query["longitude"],
                "sequence_id": query["sequence_id"],
                "frame_number": query["frame_number"],
                "viewpoint": query["viewpoint"],
                "illumination": query["illumination"],
                "season": query["season"],
                "weather": query["weather"],
                "subtask": SUBTASK_ALIASES[args.subtask.lower()],
                "gallery": [
                    {
                        "gallery_id": gallery_id,
                        "path": relativize(item["image_path"], root),
                        "place_id": item["place_id"],
                        "latitude": item["latitude"],
                        "longitude": item["longitude"],
                        "sequence_id": item["sequence_id"],
                        "frame_number": item["frame_number"],
                        "viewpoint": item["viewpoint"],
                        "illumination": item["illumination"],
                        "season": item["season"],
                        "weather": item["weather"],
                    }
                    for gallery_id, item in gallery_by_id.items()
                ],
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    metadata = {
        "dataset": "Mapillary Street-Level Sequences",
        "official_source": "https://github.com/mapillary/mapillary_sls",
        "download_source": "https://www.mapillary.com/dataset/places",
        "split": args.split,
        "cities": args.cities,
        "subtask": SUBTASK_ALIASES[args.subtask.lower()],
        "query_count": len(queries),
        "gallery_count": len(gallery),
        "place_labels": "official unique_cluster values from MSLS postprocessed.csv",
        "gps_fields": ["latitude", "longitude"],
        "query_gallery_path_overlap": 0,
        "image_files_are_not_copied": True,
        "scientific_evidence": True,
    }
    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[msls] wrote {len(queries)} queries and {len(gallery)} gallery images to {output_path}")
    print(f"[msls] wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
