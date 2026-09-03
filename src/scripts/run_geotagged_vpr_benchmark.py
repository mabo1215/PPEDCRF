"""Evaluate PPEDCRF on a manifest-driven geotagged VPR benchmark.

The loader intentionally does not assume a particular public VPR directory
layout. Each JSONL record contains one query and its gallery:

    {"query_id": "q0", "query_path": "...", "place_id": "p0",
     "gallery": [{"gallery_id": "g0", "path": "...", "place_id": "p0"}, ...],
     "viewpoint": "front", "illumination": "night"}

At least one gallery item must share the query's place_id. The script reports
true place-level retrieval, not the synthetic paired-scene proxy metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from datasets.driving_clip_dataset import _read_image, _resize_if_needed  # noqa: E402
from eval.metrics import psnr_torch  # noqa: E402
from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
)
from main import load_sensnet_checkpoint  # noqa: E402
from run_tomm_review_proxy import (  # noqa: E402
    VARIANT_LABELS,
    detailed_retrieval,
    normalized_embeddings,
    optimize_attacker_aware_query,
    protect_review_clip,
)
from utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geotagged VPR benchmark for PPEDCRF.")
    parser.add_argument("--mode", choices=("smoke", "geotagged"), default="smoke")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--root", default="")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    parser.add_argument("--output_dir", default="src/outputs/tomm_geotagged_vpr")
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    parser.add_argument("--backbones", nargs="+", default=["resnet18"])
    parser.add_argument("--variants", nargs="+", default=["full"])
    parser.add_argument("--smoke_queries", type=int, default=3)
    parser.add_argument("--smoke_size", type=int, default=64)
    parser.add_argument(
        "--include_attacker_aware",
        action="store_true",
        help=(
            "Add a white-box sign-gradient variant (same optimizer as the "
            "proxy benchmark's attacker_aware). Reported as a separate "
            "diagnostic threat model, never pooled with the black-box rows."
        ),
    )
    parser.add_argument("--attacker_backbone", default="resnet18")
    parser.add_argument("--attacker_steps", type=int, default=20)
    parser.add_argument("--attacker_step_size", type=float, default=1.0)
    parser.add_argument("--attacker_linf", type=float, default=8.0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(path_value: str, root: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(root) / path).resolve() if root else path.resolve()


def load_manifest(path: str, root: str) -> Tuple[List[dict], Dict[str, dict]]:
    records: List[dict] = []
    gallery_by_id: Dict[str, dict] = {}
    query_ids: set[str] = set()
    query_paths: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("query_id", "query_path", "place_id", "gallery"):
                if key not in item:
                    raise ValueError(f"Manifest line {line_number} is missing '{key}'.")
            query_id = str(item["query_id"])
            if query_id in query_ids:
                raise ValueError(f"Duplicate query_id: {query_id}")
            query_ids.add(query_id)
            query_path = resolve_path(str(item["query_path"]), root)
            if not query_path.is_file():
                raise FileNotFoundError(f"Query image does not exist: {query_path}")
            query_paths.add(str(query_path))
            gallery = item["gallery"]
            if not isinstance(gallery, list) or not gallery:
                raise ValueError(f"Query {query_id} has an empty gallery.")
            normalized_gallery: List[dict] = []
            for gallery_item in gallery:
                for key in ("gallery_id", "path", "place_id"):
                    if key not in gallery_item:
                        raise ValueError(f"Query {query_id} gallery item is missing '{key}'.")
                gallery_id = str(gallery_item["gallery_id"])
                gallery_path = resolve_path(str(gallery_item["path"]), root)
                gallery_record = {
                    "gallery_id": gallery_id,
                    "path": str(gallery_path),
                    "place_id": str(gallery_item["place_id"]),
                }
                # The manifest format shares one gallery pool across every query
                # line (see build_msls_manifest.py / build_kitti360_unary_manifest.py),
                # so the same gallery_id legitimately recurs across queries. Only
                # reject it if a later occurrence disagrees with the first one.
                existing = gallery_by_id.get(gallery_id)
                if existing is not None and existing != gallery_record:
                    raise ValueError(
                        f"Gallery id {gallery_id} maps to inconsistent records: "
                        f"{existing} vs {gallery_record}"
                    )
                if existing is None:
                    if not gallery_path.is_file():
                        raise FileNotFoundError(f"Gallery image does not exist: {gallery_path}")
                    gallery_by_id[gallery_id] = gallery_record
                normalized_gallery.append(gallery_record)
            place_id = str(item["place_id"])
            if not any(g["place_id"] == place_id for g in normalized_gallery):
                raise ValueError(f"Query {query_id} has no same-place gallery positive.")
            record = dict(item)
            record["query_id"] = query_id
            record["query_path"] = str(query_path)
            record["place_id"] = place_id
            record["gallery"] = normalized_gallery
            records.append(record)
    if not records:
        raise ValueError("Manifest contains no query records.")
    gallery_paths = {record["path"] for record in gallery_by_id.values()}
    overlap = query_paths & gallery_paths
    if overlap:
        raise ValueError(f"Query/gallery path overlap detected ({len(overlap)} files).")
    return records, gallery_by_id


def load_image(path: str, resize_hw: Tuple[int, int]) -> torch.Tensor:
    return _resize_if_needed(_read_image(path), resize_hw)


def run_geotagged(args: argparse.Namespace) -> Path:
    if not args.manifest:
        raise ValueError("--manifest is required in geotagged mode.")
    records, gallery_by_id = load_manifest(args.manifest, args.root)
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
    resize_hw = (int(args.resize_h), int(args.resize_w))

    gallery_ids = sorted(gallery_by_id)
    gallery_images = torch.stack([load_image(gallery_by_id[g]["path"], resize_hw) for g in gallery_ids])
    query_ids = [record["query_id"] for record in records]
    query_images = torch.stack([load_image(record["query_path"], resize_hw) for record in records])
    positive_place = {record["query_id"]: record["place_id"] for record in records}
    gallery_place = {gallery_id: record["place_id"] for gallery_id, record in gallery_by_id.items()}

    sanitized: Dict[Tuple[str, int], torch.Tensor] = {}
    quality: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = {}
    for variant in args.variants:
        if variant not in VARIANT_LABELS:
            raise ValueError(f"Unknown variant '{variant}'.")
        for seed in args.seeds:
            images: List[torch.Tensor] = []
            q_quality: Dict[str, Dict[str, float]] = {}
            for index, query_id in enumerate(query_ids):
                clip, metrics = protect_review_clip(
                    query_images[index].unsqueeze(0),
                    sensnet,
                    cfg,
                    device,
                    variant,
                    int(seed),
                )
                images.append(clip[0])
                q_quality[query_id] = metrics
            sanitized[(variant, int(seed))] = torch.stack(images)
            quality[(variant, int(seed))] = q_quality

    rows: List[Dict[str, object]] = []
    for backbone in args.backbones:
        rcfg = RetrievalConfig(
            backbone=backbone,
            device=str(device),
            normalize=True,
            input_size=default_input_size_for_backbone(backbone),
            topk=(1, 5, 10),
        )
        embedder = make_default_embedder(rcfg).eval().to(device)
        gallery_tensor = gallery_images
        raw_rows = detailed_retrieval(
            query_images,
            query_ids,
            gallery_tensor,
            gallery_ids,
            embedder,
            device,
            rcfg.input_size,
            positive_place_by_query=positive_place,
            gallery_place_by_id=gallery_place,
        )
        for row, record in zip(raw_rows, records):
            row.update(
                {
                    "variant": "raw",
                    "label": "raw query",
                    "seed": "raw",
                    "backbone": backbone,
                    "gallery_size": len(gallery_ids),
                    "place_id": record["place_id"],
                    "city": record.get("city", ""),
                    "subtask": record.get("subtask", ""),
                    "side": record.get("side", ""),
                    "captured_at": record.get("captured_at", ""),
                    "viewpoint": record.get("viewpoint", ""),
                    "illumination": record.get("illumination", ""),
                    "season": record.get("season", ""),
                    "weather": record.get("weather", ""),
                }
            )
        rows.extend(raw_rows)
        for variant in args.variants:
            for seed in args.seeds:
                qrows = detailed_retrieval(
                    sanitized[(variant, int(seed))],
                    query_ids,
                    gallery_tensor,
                    gallery_ids,
                    embedder,
                    device,
                    rcfg.input_size,
                    quality_by_query=quality[(variant, int(seed))],
                    positive_place_by_query=positive_place,
                    gallery_place_by_id=gallery_place,
                )
                for row, record in zip(qrows, records):
                    row.update(
                        {
                            "variant": variant,
                            "label": VARIANT_LABELS[variant],
                            "seed": int(seed),
                            "backbone": backbone,
                            "gallery_size": len(gallery_ids),
                            "place_id": record["place_id"],
                            "city": record.get("city", ""),
                            "subtask": record.get("subtask", ""),
                            "side": record.get("side", ""),
                            "captured_at": record.get("captured_at", ""),
                            "viewpoint": record.get("viewpoint", ""),
                            "illumination": record.get("illumination", ""),
                            "season": record.get("season", ""),
                            "weather": record.get("weather", ""),
                        }
                    )
                rows.extend(qrows)

        if args.include_attacker_aware and backbone == args.attacker_backbone:
            with torch.no_grad():
                gallery_emb = normalized_embeddings(embedder, gallery_tensor, device, rcfg.input_size)
            aware_images: List[torch.Tensor] = []
            aware_quality: Dict[str, Dict[str, float]] = {}
            for index, query_id in enumerate(query_ids):
                target_place = positive_place[query_id]
                positive_indices = [
                    j for j, gallery_id in enumerate(gallery_ids) if gallery_place[gallery_id] == target_place
                ]
                if not positive_indices:
                    raise ValueError(f"No positive gallery item for attacker-aware query {query_id}.")
                aware = optimize_attacker_aware_query(
                    query_images[index],
                    gallery_emb[positive_indices[0]],
                    embedder,
                    device,
                    rcfg.input_size,
                    args.attacker_steps,
                    args.attacker_step_size,
                    args.attacker_linf,
                )
                aware_images.append(aware)
                aware_quality[query_id] = {
                    "psnr_mean": psnr_torch(query_images[index], aware),
                    "effective_mse": float(torch.mean((aware - query_images[index]).float().square()).item()),
                    "support_mse": float(torch.mean((aware - query_images[index]).float().square()).item()),
                    "support_coverage": 1.0,
                }
            aware_rows = detailed_retrieval(
                torch.stack(aware_images),
                query_ids,
                gallery_tensor,
                gallery_ids,
                embedder,
                device,
                rcfg.input_size,
                quality_by_query=aware_quality,
                positive_place_by_query=positive_place,
                gallery_place_by_id=gallery_place,
            )
            for row, record in zip(aware_rows, records):
                row.update(
                    {
                        "variant": "attacker_aware",
                        "label": VARIANT_LABELS["attacker_aware"],
                        "seed": "deterministic",
                        "backbone": backbone,
                        "gallery_size": len(gallery_ids),
                        "place_id": record["place_id"],
                        "city": record.get("city", ""),
                        "subtask": record.get("subtask", ""),
                        "side": record.get("side", ""),
                        "captured_at": record.get("captured_at", ""),
                        "viewpoint": record.get("viewpoint", ""),
                        "illumination": record.get("illumination", ""),
                        "season": record.get("season", ""),
                        "weather": record.get("weather", ""),
                        "attacker_steps": int(args.attacker_steps),
                        "attacker_step_size": float(args.attacker_step_size),
                        "attacker_linf": float(args.attacker_linf),
                    }
                )
            rows.extend(aware_rows)

    write_csv(output_dir / "geotagged_vpr_per_query.csv", rows)
    write_json(
        output_dir / "manifest_gate.json",
        {
            "passed": True,
            "query_count": len(records),
            "gallery_count": len(gallery_by_id),
            "query_gallery_path_overlap": 0,
            "unique_query_ids": True,
            "unique_gallery_ids": True,
            "true_place_labels": True,
        },
    )
    write_json(
        output_dir / "run_metadata.json",
        {
            "review_cycle": "ICME-2027",
            "mode": "geotagged",
            "device": str(device),
            "manifest": str(Path(args.manifest).resolve()),
            "backbones": args.backbones,
            "variants": args.variants,
            "seeds": args.seeds,
            "attacker_aware_requested": bool(args.include_attacker_aware),
            "attacker_backbone": args.attacker_backbone if args.include_attacker_aware else None,
            "attacker_steps": int(args.attacker_steps) if args.include_attacker_aware else None,
            "attacker_step_size": float(args.attacker_step_size) if args.include_attacker_aware else None,
            "attacker_linf": float(args.attacker_linf) if args.include_attacker_aware else None,
            "attacker_aware_note": (
                "White-box sign-gradient diagnostic against a fixed target gallery "
                "embedding; a separate threat model, not comparable to the black-box "
                "backbone-transfer rows."
                if args.include_attacker_aware
                else None
            ),
            "scientific_evidence": True,
        },
    )
    print(f"[geotagged] wrote {len(rows)} per-query rows to {output_dir}")
    return output_dir


class TinyEmbedder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, (4, 4)).flatten(1)


def run_smoke(args: argparse.Namespace) -> Path:
    from run_train import SensitiveRegionNet
    from run_tomm_review_proxy import detailed_retrieval, protect_review_clip

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).with_name("tomm_geotagged_smoke")
    ensure_dir(output_dir)
    size = int(args.smoke_size)
    count = int(args.smoke_queries)
    torch.manual_seed(20260830)
    queries = torch.rand(count, 3, size, size) * 255.0
    gallery = torch.rand(count + 2, 3, size, size) * 255.0
    query_ids = [f"q_{i}" for i in range(count)]
    gallery_ids = [f"g_{i}" for i in range(count + 2)]
    records = [{"query_id": q, "place_id": f"p_{i}"} for i, q in enumerate(query_ids)]
    gallery_place = {gallery_ids[i]: f"p_{i}" if i < count else f"negative_{i}" for i in range(count + 2)}
    positive_place = {record["query_id"]: record["place_id"] for record in records}
    model = SensitiveRegionNet().to(device).eval()
    embedder = TinyEmbedder().to(device).eval()
    cfg = load_yaml(args.config)
    rows: List[Dict[str, object]] = []
    for variant in ("full", "unary_only", "no_dcrf"):
        protected = []
        quality = {}
        for index, query_id in enumerate(query_ids):
            clip, metrics = protect_review_clip(
                queries[index].unsqueeze(0), model, cfg, device, variant, 1234,
                allow_ssim_fallback=True,
            )
            protected.append(clip[0])
            quality[query_id] = metrics
        qrows = detailed_retrieval(
            torch.stack(protected), query_ids, gallery, gallery_ids, embedder,
            device, size, quality_by_query=quality,
            positive_place_by_query=positive_place,
            gallery_place_by_id=gallery_place,
        )
        for row in qrows:
            row.update({"variant": variant, "device": str(device)})
        rows.extend(qrows)
    write_csv(output_dir / "per_query.csv", rows)
    write_json(output_dir / "smoke_metadata.json", {"device": str(device), "rows": len(rows), "scientific_evidence": False})
    if len(rows) != count * 3 or not all(np.isfinite(float(r["retrieval_margin"])) for r in rows):
        raise RuntimeError("Geotagged smoke test failed.")
    print(f"[smoke] geotagged loader/evaluator passed on {device}; wrote {len(rows)} rows")
    return output_dir


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_geotagged(args)


if __name__ == "__main__":
    main()
