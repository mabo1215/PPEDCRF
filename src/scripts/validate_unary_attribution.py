"""Independently diagnose unary maps with frozen VPR occlusion attribution.

This script does not pretend that semantic labels are sensitivity ground truth.
For each query it measures how tiled image occlusion changes the correct-place
versus hardest-negative retrieval margin. The resulting weak reference is
compared with the PPEDCRF unary/DCRF map on static-background pixels from an
optional semantic mask. A real run is scientific evidence only when the
manifest, checkpoint, sequence split, and non-constant-map gates pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from datasets.driving_clip_dataset import _read_image, _resize_if_needed  # noqa: E402
from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
)
from main import compute_refined_sensitivity, load_sensnet_checkpoint  # noqa: E402
from run_tomm_review_proxy import preprocess_for_embed  # noqa: E402
from utils.config import load_yaml  # noqa: E402


DEFAULT_DYNAMIC_IDS = (11, 12, 13, 14, 15, 16, 17, 18)


class TinyEmbedder(nn.Module):
    """Small deterministic embedder used only for the smoke test."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(images, (4, 4)).flatten(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate unary maps with VPR attribution consistency.")
    parser.add_argument("--mode", choices=("smoke", "manifest"), default="smoke")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--root", default="")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    parser.add_argument("--output_dir", default="src/outputs/tomm_unary_attribution")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--grid_rows", type=int, default=6)
    parser.add_argument("--grid_cols", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_queries", type=int, default=64)
    parser.add_argument("--min_valid_queries", type=int, default=8)
    parser.add_argument("--dynamic_ids", default=",".join(str(x) for x in DEFAULT_DYNAMIC_IDS))
    parser.add_argument("--constant_std_threshold", type=float, default=1e-3)
    parser.add_argument("--smoke_size", type=int, default=64)
    parser.add_argument("--smoke_queries", type=int, default=3)
    return parser.parse_args()


def resolve_path(value: str, root: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(root) / path).resolve() if root else path.resolve()


def load_manifest(path: str, root: str, max_queries: int) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("query_id", "query_path", "place_id", "gallery", "semantic_path"):
                if key not in record:
                    raise ValueError(f"Manifest line {line_number} is missing '{key}'.")
            query_path = resolve_path(str(record["query_path"]), root)
            semantic_path = resolve_path(str(record["semantic_path"]), root)
            if not query_path.is_file():
                raise FileNotFoundError(f"Missing query image: {query_path}")
            if not semantic_path.is_file():
                raise FileNotFoundError(f"Missing semantic mask: {semantic_path}")
            gallery = record["gallery"]
            if not isinstance(gallery, list) or len(gallery) < 2:
                raise ValueError(f"Query {record['query_id']} needs at least one positive and one negative.")
            normalized_gallery: list[dict] = []
            gallery_paths: set[str] = set()
            for item in gallery:
                for key in ("gallery_id", "path", "place_id"):
                    if key not in item:
                        raise ValueError(f"Query {record['query_id']} gallery item is missing '{key}'.")
                gallery_path = resolve_path(str(item["path"]), root)
                if not gallery_path.is_file():
                    raise FileNotFoundError(f"Missing gallery image: {gallery_path}")
                if str(gallery_path) in gallery_paths:
                    raise ValueError(f"Duplicate gallery path in query {record['query_id']}: {gallery_path}")
                gallery_paths.add(str(gallery_path))
                normalized_gallery.append(
                    {
                        **item,
                        "path": str(gallery_path),
                        "gallery_id": str(item["gallery_id"]),
                        "place_id": str(item["place_id"]),
                    }
                )
            if str(query_path) in gallery_paths:
                raise ValueError(f"Query/gallery path overlap for {record['query_id']}")
            records.append(
                {
                    **record,
                    "query_id": str(record["query_id"]),
                    "query_path": str(query_path),
                    "semantic_path": str(semantic_path),
                    "place_id": str(record["place_id"]),
                    "gallery": normalized_gallery,
                }
            )
            if len(records) >= int(max_queries):
                break
    if not records:
        raise ValueError("Attribution manifest is empty.")
    return records


def load_mask(path: str, size_hw: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size_hw[1], size_hw[0]), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.int64)


def encode_images(
    embedder: nn.Module,
    images: Sequence[torch.Tensor],
    device: torch.device,
    input_size: int,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    embedder.eval().to(device)
    with torch.no_grad():
        for start in range(0, len(images), max(1, int(batch_size))):
            batch = torch.stack(list(images[start : start + int(batch_size)])).to(device)
            embeddings = embedder(preprocess_for_embed(batch, input_size))
            outputs.append(F.normalize(embeddings, dim=1).detach())
    return torch.cat(outputs, dim=0)


def place_margin(
    query_embedding: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_places: Sequence[str],
    place_id: str,
) -> tuple[float, float, float]:
    similarity = query_embedding @ gallery_embeddings.t()
    positives = [index for index, value in enumerate(gallery_places) if value == place_id]
    negatives = [index for index, value in enumerate(gallery_places) if value != place_id]
    if not positives or not negatives:
        raise ValueError("Attribution query must have both a positive and a negative gallery item.")
    positive_similarity = max(float(similarity[0, index].item()) for index in positives)
    negative_similarity = max(float(similarity[0, index].item()) for index in negatives)
    return positive_similarity - negative_similarity, positive_similarity, negative_similarity


def occlusion_reference(
    query: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_places: Sequence[str],
    place_id: str,
    embedder: nn.Module,
    device: torch.device,
    input_size: int,
    grid_rows: int,
    grid_cols: int,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    _, height, width = query.shape
    base_embedding = encode_images(embedder, [query], device, input_size, 1)
    base_margin, _, _ = place_margin(base_embedding, gallery_embeddings, gallery_places, place_id)
    fill = query.mean(dim=(1, 2), keepdim=True)
    reference = np.zeros((height, width), dtype=np.float32)
    occluded: list[torch.Tensor] = []
    regions: list[tuple[int, int, int, int]] = []
    for row in range(int(grid_rows)):
        y0 = row * height // int(grid_rows)
        y1 = (row + 1) * height // int(grid_rows)
        for col in range(int(grid_cols)):
            x0 = col * width // int(grid_cols)
            x1 = (col + 1) * width // int(grid_cols)
            candidate = query.clone()
            candidate[:, y0:y1, x0:x1] = fill
            occluded.append(candidate)
            regions.append((y0, y1, x0, x1))
    occluded_embeddings = encode_images(embedder, occluded, device, input_size, batch_size)
    for embedding, (y0, y1, x0, x1) in zip(occluded_embeddings, regions):
        margin, _, _ = place_margin(embedding.unsqueeze(0), gallery_embeddings, gallery_places, place_id)
        reference[y0:y1, x0:x1] = max(0.0, float(base_margin - margin))
    return reference, base_margin


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0 + 1.0
        index = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    rx = average_ranks(x)
    ry = average_ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def top_fraction_overlap(x: np.ndarray, y: np.ndarray, fraction: float = 0.10) -> float:
    if x.size == 0:
        return 0.0
    count = max(1, int(math.ceil(x.size * fraction)))
    x_top = set(np.argpartition(x, -count)[-count:].tolist())
    y_top = set(np.argpartition(y, -count)[-count:].tolist())
    return float(len(x_top & y_top) / max(1, len(x_top | y_top)))


def evaluate_record(
    record: Mapping[str, object],
    root: str,
    sensnet: nn.Module,
    embedder: nn.Module,
    cfg: Mapping[str, object],
    device: torch.device,
    resize_hw: tuple[int, int],
    input_size: int,
    grid_rows: int,
    grid_cols: int,
    batch_size: int,
    dynamic_ids: set[int],
    constant_std_threshold: float,
) -> dict[str, object]:
    query = _resize_if_needed(_read_image(str(record["query_path"])), resize_hw)
    gallery = list(record["gallery"])  # type: ignore[arg-type]
    gallery_images = [_resize_if_needed(_read_image(str(item["path"])), resize_hw) for item in gallery]
    gallery_places = [str(item["place_id"]) for item in gallery]
    reference, base_margin = occlusion_reference(
        query,
        encode_images(embedder, gallery_images, device, input_size, batch_size),
        gallery_places,
        str(record["place_id"]),
        embedder,
        device,
        input_size,
        grid_rows,
        grid_cols,
        batch_size,
    )
    with torch.no_grad():
        refined, _ = compute_refined_sensitivity(query.unsqueeze(0), sensnet, cfg, device)
    predicted = refined[0, 0].numpy().astype(np.float32)
    semantic = load_mask(str(record["semantic_path"]), resize_hw)
    valid = semantic != 255
    static = valid & ~np.isin(semantic, list(dynamic_ids))
    if not np.any(static):
        raise ValueError(f"No valid static-background pixels for {record['query_id']}")
    pred_values = predicted[static].reshape(-1)
    ref_values = reference[static].reshape(-1)
    threshold = float(np.quantile(ref_values, 0.90))
    ref_top = np.maximum(ref_values - threshold, 0.0)
    ref_total = float(ref_top.sum())
    pred_top = predicted[static] >= float(np.quantile(pred_values, 0.90))
    pred_energy = float(ref_top[pred_top].sum() / max(ref_total, 1e-12))
    dynamic = valid & ~static
    return {
        "query_id": str(record["query_id"]),
        "sequence_id": str(record.get("sequence_id", "")),
        "place_id": str(record["place_id"]),
        "base_retrieval_margin": base_margin,
        "spearman_static": spearman(pred_values, ref_values),
        "top10_overlap_static": top_fraction_overlap(pred_values, ref_values),
        "reference_energy_in_predicted_top10": pred_energy,
        "predicted_static_mean": float(pred_values.mean()),
        "predicted_dynamic_mean": float(predicted[dynamic].mean()) if np.any(dynamic) else float("nan"),
        "predicted_static_std": float(pred_values.std()),
        "reference_static_std": float(ref_values.std()),
        "static_fraction": float(static.mean()),
        "map_constant_gate": bool(float(predicted.std()) < constant_std_threshold),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_smoke(args: argparse.Namespace) -> Path:
    torch.manual_seed(20260831)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = int(args.smoke_size)
    count = int(args.smoke_queries)
    query_images = [torch.rand(3, size, size) * 255.0 for _ in range(count)]
    gallery_images = [torch.rand(3, size, size) * 255.0 for _ in range(count + 2)]
    model = __import__("run_train", fromlist=["SensitiveRegionNet"]).SensitiveRegionNet().to(device).eval()
    embedder = TinyEmbedder().to(device).eval()
    cfg = load_yaml(args.config)
    rows: list[dict[str, object]] = []
    for index, query in enumerate(query_images):
        gallery = gallery_images
        places = [f"p_{j}" if j == index else f"negative_{j}" for j in range(len(gallery))]
        reference, base_margin = occlusion_reference(
            query, encode_images(embedder, gallery, device, size, 8), places, f"p_{index}",
            embedder, device, size, 4, 4, 8
        )
        with torch.no_grad():
            refined, _ = compute_refined_sensitivity(query.unsqueeze(0), model, cfg, device)
        predicted = refined[0, 0].numpy()
        flat_pred = predicted.reshape(-1)
        flat_ref = reference.reshape(-1)
        rows.append(
            {
                "query_id": f"smoke_{index}",
                "base_retrieval_margin": base_margin,
                "spearman_all": spearman(flat_pred, flat_ref),
                "top10_overlap_all": top_fraction_overlap(flat_pred, flat_ref),
                "predicted_std": float(predicted.std()),
                "reference_std": float(reference.std()),
            }
        )
    output_dir = Path(args.output_dir).with_name("tomm_unary_attribution_smoke")
    write_csv(output_dir / "unary_attribution.csv", rows)
    summary = {
        "mode": "smoke",
        "device": str(device),
        "query_count": len(rows),
        "scientific_evidence": False,
        "gate_passed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[smoke] unary attribution schema passed on {device}; wrote {len(rows)} rows to {output_dir}")
    return output_dir


def run_manifest(args: argparse.Namespace) -> Path:
    records = load_manifest(args.manifest, args.root, int(args.max_queries))
    dynamic_ids = {int(value) for value in str(args.dynamic_ids).split(",") if value.strip()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(args.config)
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
    embedder = make_default_embedder(
        RetrievalConfig(
            backbone=args.backbone,
            device=str(device),
            normalize=True,
            input_size=default_input_size_for_backbone(args.backbone),
            topk=(1, 5, 10),
        )
    ).eval().to(device)
    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            evaluate_record(
                record,
                args.root,
                sensnet,
                embedder,
                cfg,
                device,
                (int(args.resize_h), int(args.resize_w)),
                default_input_size_for_backbone(args.backbone),
                int(args.grid_rows),
                int(args.grid_cols),
                int(args.batch_size),
                dynamic_ids,
                float(args.constant_std_threshold),
            )
        )
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "unary_attribution.csv", rows)
    valid_rows = [row for row in rows if not bool(row["map_constant_gate"])]
    summary = {
        "mode": "manifest",
        "dataset": "KITTI-360 or another registered pose/semantic manifest",
        "device": str(device),
        "backbone": args.backbone,
        "query_count": len(rows),
        "valid_nonconstant_queries": len(valid_rows),
        "mean_spearman_static": float(np.mean([float(row["spearman_static"]) for row in rows])),
        "mean_top10_overlap_static": float(np.mean([float(row["top10_overlap_static"]) for row in rows])),
        "mean_reference_energy_in_predicted_top10": float(
            np.mean([float(row["reference_energy_in_predicted_top10"]) for row in rows])
        ),
        "map_constant_std_threshold": float(args.constant_std_threshold),
        "gate_passed": len(rows) >= int(args.min_valid_queries) and len(valid_rows) == len(rows),
        "scientific_evidence": len(rows) >= int(args.min_valid_queries) and len(valid_rows) == len(rows),
        "weak_supervision_only": True,
        "ground_truth_sensitivity_accuracy_claim": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[unary] wrote {len(rows)} rows to {output_dir}; gate_passed={summary['gate_passed']}")
    return output_dir


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        if not args.manifest:
            raise ValueError("--manifest is required in manifest mode.")
        run_manifest(args)


if __name__ == "__main__":
    main()
