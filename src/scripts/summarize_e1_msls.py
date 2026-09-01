"""Summarize the MSLS real-place benchmark and cross-condition coverage.

The script consumes only the manifest and per-query outputs produced by the
official-source MSLS runs. It writes compact paper-facing CSV/JSON summaries
without copying or redistributing the dataset images.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MSLS E1 runs.")
    parser.add_argument("--base_dir", default="src/outputs/e1_msls")
    parser.add_argument("--runs", nargs="+", default=["all", "o2n", "n2o"])
    parser.add_argument("--output_csv", default="src/outputs/e1_msls/e1_cross_condition_summary.csv")
    parser.add_argument("--output_json", default="src/outputs/e1_msls/e1_cross_condition_summary.json")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_row(rows: list[dict[str, str]], run: str, city: str, variant: str, seed: str) -> dict[str, Any]:
    ranks = np.asarray([int(float(row["correct_rank"])) for row in rows], dtype=np.int64)
    quality_rows = [row for row in rows if row.get("psnr_mean", "") not in ("", "nan")]
    return {
        "run": run,
        "city": city,
        "variant": variant,
        "seed": seed,
        "n": int(len(rows)),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "psnr": float(np.mean([float(row["psnr_mean"]) for row in quality_rows])) if quality_rows else "",
        "ssim": float(np.mean([float(row["ssim_mean"]) for row in quality_rows])) if quality_rows else "",
        "effective_mse": float(np.mean([float(row["effective_mse"]) for row in quality_rows])) if quality_rows else "",
    }


def summarize_run(base_dir: Path, run: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = base_dir / ("manifest_all.jsonl" if run == "all" else f"manifest_{run}.jsonl")
    output_path = base_dir / ("geotagged_vpr" if run == "all" else f"geotagged_{run}") / "geotagged_vpr_per_query.csv"
    rows = read_csv(output_path)
    manifest_records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_rows: list[dict[str, Any]] = []
    for city in sorted({str(record.get("city", record["query_id"].split("_", 1)[0])) for record in manifest_records}):
        city_rows = [row for row in rows if row["query_id"].startswith(f"{city}_")]
        for variant in sorted({row["variant"] for row in city_rows}):
            variant_rows = [row for row in city_rows if row["variant"] == variant]
            for seed in sorted({str(row["seed"]) for row in variant_rows}):
                seed_rows = [row for row in variant_rows if str(row["seed"]) == seed]
                summary_rows.append(metric_row(seed_rows, run, city, variant, seed))

    condition: dict[str, Any] = {"run": run, "query_count": len(manifest_records), "cities": {}}
    for city in sorted({str(record.get("city", record["query_id"].split("_", 1)[0])) for record in manifest_records}):
        city_records = [record for record in manifest_records if str(record.get("city", record["query_id"].split("_", 1)[0])) == city]
        query_dates = sorted(str(record.get("captured_at", "")) for record in city_records if record.get("captured_at"))
        gallery_items = [item for record in city_records for item in record.get("gallery", []) if str(item.get("city", "")) == city]
        gallery_dates = sorted(str(item.get("captured_at", "")) for item in gallery_items if item.get("captured_at"))
        condition["cities"][city] = {
            "query_count": len(city_records),
            "subtasks": sorted({str(record.get("subtask", "")) for record in city_records}),
            "query_viewpoints": sorted({str(record.get("viewpoint", "")) for record in city_records}),
            "query_illuminations": sorted({str(record.get("illumination", "")) for record in city_records}),
            "query_seasons": sorted({str(record.get("season", "")) for record in city_records}),
            "query_weather": sorted({str(record.get("weather", "")) for record in city_records}),
            "query_capture_min": query_dates[0] if query_dates else "",
            "query_capture_max": query_dates[-1] if query_dates else "",
            "gallery_capture_min": gallery_dates[0] if gallery_dates else "",
            "gallery_capture_max": gallery_dates[-1] if gallery_dates else "",
        }
    return summary_rows, condition


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "city", "variant", "seed", "n", "top1", "top5", "top10", "psnr", "ssim", "effective_mse"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    requested_runs = ", ".join(str(run) for run in args.runs)
    all_rows: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    for run in args.runs:
        rows, condition = summarize_run(base_dir, run)
        all_rows.extend(rows)
        conditions.append(condition)

    aggregate: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["seed"] != "raw":
            grouped[(row["run"], row["city"], row["variant"])].append(row)
    for (run, city, variant), rows in sorted(grouped.items()):
        aggregate.append(
            {
                "run": run,
                "city": city,
                "variant": variant,
                "seed": "mean_std",
                "n": int(sum(row["n"] for row in rows)),
                "top1": f"{np.mean([row['top1'] for row in rows]):.6f} +/- {np.std([row['top1'] for row in rows], ddof=0):.6f}",
                "top5": f"{np.mean([row['top5'] for row in rows]):.6f} +/- {np.std([row['top5'] for row in rows], ddof=0):.6f}",
                "top10": f"{np.mean([row['top10'] for row in rows]):.6f} +/- {np.std([row['top10'] for row in rows], ddof=0):.6f}",
                "psnr": f"{np.mean([row['psnr'] for row in rows if row['psnr'] != '']):.6f}",
                "ssim": f"{np.mean([row['ssim'] for row in rows if row['ssim'] != '']):.6f}",
                "effective_mse": f"{np.mean([row['effective_mse'] for row in rows if row['effective_mse'] != '']):.6f}",
            }
        )

    output_csv = Path(args.output_csv)
    write_csv(output_csv, all_rows + aggregate)
    payload = {
        "runs": args.runs,
        "per_seed_and_city": all_rows,
        "aggregate_seed_mean_std": aggregate,
        "condition_coverage": conditions,
        "scientific_evidence": True,
        "interpretation": (
            f"The requested MSLS runs ({requested_runs}) use official subtasks. The available subset "
            "still has limited viewpoint and illumination diversity, so these results "
            "do not establish broad all-condition robustness."
        ),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[e1] wrote {len(all_rows) + len(aggregate)} summary rows to {output_csv}")
    print(f"[e1] wrote condition coverage to {output_json}")


if __name__ == "__main__":
    main()
