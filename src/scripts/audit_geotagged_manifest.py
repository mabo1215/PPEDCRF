"""Audit a geotagged VPR manifest before any GPU inference.

The audit reuses the strict manifest loader, reports place and condition
coverage, and records whether the manifest is suitable for an ICME revision
run. It never loads a neural model and never relabels a proxy benchmark as
geographic evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from run_geotagged_vpr_benchmark import load_manifest


CONDITION_FIELDS = ("city", "subtask", "viewpoint", "illumination", "season", "weather")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an ICME geotagged VPR manifest.")
    parser.add_argument("--mode", choices=("manifest", "smoke"), default="manifest")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--root", default="")
    parser.add_argument(
        "--output",
        default="src/outputs/icme2027_manifest_audit/manifest_gate.json",
    )
    return parser.parse_args()


def value_counts(records: list[Mapping[str, Any]], field: str) -> dict[str, int]:
    values = [str(record.get(field, "")) for record in records]
    return dict(sorted(Counter(value for value in values if value).items()))


def audit_records(records: list[Mapping[str, Any]], gallery_count: int) -> dict[str, Any]:
    positive_counts = []
    for record in records:
        place_id = str(record["place_id"])
        gallery = record["gallery"]
        positive_counts.append(
            sum(1 for item in gallery if str(item["place_id"]) == place_id)
        )

    coverage = {field: value_counts(records, field) for field in CONDITION_FIELDS}
    issues: list[str] = []
    if len(coverage["city"]) < 2:
        issues.append("fewer than two city strata are represented")
    if not coverage["viewpoint"]:
        issues.append("viewpoint metadata are absent")
    if not coverage["illumination"]:
        issues.append("illumination metadata are absent")
    if min(positive_counts, default=0) < 1:
        issues.append("at least one query has no positive gallery item")

    return {
        "review_cycle": "ICME-2027",
        "manifest_validated": True,
        "query_count": len(records),
        "gallery_count": int(gallery_count),
        "unique_query_ids": len({str(record["query_id"]) for record in records}) == len(records),
        "unique_place_ids": len({str(record["place_id"]) for record in records}),
        "positive_gallery_min": min(positive_counts, default=0),
        "positive_gallery_max": max(positive_counts, default=0),
        "condition_coverage": coverage,
        "coverage_gate_passed": not issues,
        "coverage_gate_issues": issues,
        "scientific_evidence": False,
        "note": "This audit is a pre-inference gate; it is not retrieval evidence.",
    }


def run_manifest(args: argparse.Namespace) -> Path:
    if not args.manifest:
        raise ValueError("--manifest is required in manifest mode.")
    records, gallery_by_id = load_manifest(args.manifest, args.root)
    payload = audit_records(records, len(gallery_by_id))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[audit] manifest valid={payload['manifest_validated']} "
        f"coverage_gate={payload['coverage_gate_passed']} "
        f"queries={payload['query_count']} gallery={payload['gallery_count']} "
        f"output={output}"
    )
    return output


def run_smoke(args: argparse.Namespace) -> Path:
    records = [
        {
            "query_id": "smoke_q0",
            "place_id": "place_0",
            "city": "city_a",
            "subtask": "all",
            "viewpoint": "forward",
            "illumination": "day",
            "season": "summer",
            "weather": "clear",
            "gallery": [{"place_id": "place_0"}],
        },
        {
            "query_id": "smoke_q1",
            "place_id": "place_1",
            "city": "city_b",
            "subtask": "o2n",
            "viewpoint": "forward",
            "illumination": "night",
            "season": "winter",
            "weather": "rain",
            "gallery": [{"place_id": "place_1"}],
        },
    ]
    payload = audit_records(records, gallery_count=2)
    if not payload["coverage_gate_passed"]:
        raise RuntimeError("ICME-M3 manifest audit smoke gate failed.")
    output = Path(args.output).with_name("manifest_gate_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[smoke] ICME-M3 audit gate passed; output={output}")
    return output


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_manifest(args)


if __name__ == "__main__":
    main()
