"""Evaluate frame-pooling and best-frame attacks on sanitized clips.

This ICME 2027 revision-cycle runner addresses the frame-versus-sequence
threat-model gap. It evaluates clip lengths 1, 2, 4, and 8 with first-frame,
mean-pooling, max-pooling, and best-frame retrieval. The real mode uses the
controlled monitoring proxy and is therefore mechanism evidence, not
geographic ground-truth evidence. The smoke mode is download-free.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from datasets.monitoring_clip_dataset import MonitoringClipDataset
from eval.metrics import flicker_score, perturbation_stability, psnr_torch, ssim_grayscale_np
from eval.retrieval_attack import RetrievalConfig, default_input_size_for_backbone, make_default_embedder
from main import load_sensnet_checkpoint
from run_controlled_retrieval_benchmark import discover_paired_locations
from run_tomm_review_proxy import protect_review_clip
from utils.config import load_yaml


POOLINGS = ("first", "mean", "max", "best_frame")
DEFAULT_VARIANTS = ("full", "global_noise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ICME 2027 sequence retrieval benchmark.")
    parser.add_argument("--mode", choices=("smoke", "proxy"), default="smoke")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--monitoring_root", default=r"F:workdatasetsmonitoringimages")
    parser.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    parser.add_argument("--output_dir", default="src/outputs/icme2027_sequence_retrieval")
    parser.add_argument("--num_queries", type=int, default=12)
    parser.add_argument("--pair_pool_size", type=int, default=240)
    parser.add_argument("--max_gallery", type=int, default=48)
    parser.add_argument("--clip_lengths", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--base_clip_len", type=int, default=8)
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--gallery_size", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    parser.add_argument("--backbones", nargs="+", default=["resnet18"])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--smoke_queries", type=int, default=3)
    parser.add_argument("--smoke_size", type=int, default=64)
    parser.add_argument("--smoke_clip_len", type=int, default=8)
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


def center_clip(frames: torch.Tensor, length: int) -> torch.Tensor:
    """Return a deterministic centered clip, padding only as a last resort."""
    requested = max(1, int(length))
    if frames.size(0) >= requested:
        start = (frames.size(0) - requested) // 2
        return frames[start : start + requested]
    if frames.size(0) == 0:
        raise ValueError("Cannot select a clip from zero frames.")
    repeats = requested - frames.size(0)
    tail = frames[-1:].repeat(repeats, 1, 1, 1)
    return torch.cat([frames, tail], dim=0)


def encode_clips(
    embedder: nn.Module,
    clips: Mapping[str, torch.Tensor],
    device: torch.device,
    input_size: int,
) -> Dict[str, torch.Tensor]:
    """Encode every frame once and return normalized frame embeddings."""
    keys = list(clips)
    flattened = torch.cat([clips[key] for key in keys], dim=0).to(device)
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, flattened.size(0), 32):
            batch = flattened[start : start + 32]
            batch = batch / 255.0
            if batch.size(-1) != input_size or batch.size(-2) != input_size:
                batch = F.interpolate(batch, size=(input_size, input_size), mode="bilinear", align_corners=False)
            embedding = embedder(batch)
            outputs.append(F.normalize(embedding, dim=1).cpu())
    frame_embeddings = torch.cat(outputs, dim=0)
    result: Dict[str, torch.Tensor] = {}
    offset = 0
    for key in keys:
        count = clips[key].size(0)
        result[key] = frame_embeddings[offset : offset + count]
        offset += count
    return result


def pooled_embedding(frame_embeddings: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "first":
        value = frame_embeddings[0]
    elif pooling == "mean":
        value = frame_embeddings.mean(dim=0)
    elif pooling == "max":
        value = frame_embeddings.max(dim=0).values
    else:
        raise ValueError(f"Pooling {pooling} is not a fixed pooled descriptor.")
    return F.normalize(value.unsqueeze(0), dim=1).squeeze(0)


def retrieval_row(
    query_frames: torch.Tensor,
    gallery_frames: Mapping[str, torch.Tensor],
    query_id: str,
    place_id: str,
    pooling: str,
    query_embeddings: Mapping[str, torch.Tensor],
    gallery_embeddings: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    gallery_ids = sorted(gallery_frames)
    gallery_places = {gallery_id: (gallery_id if gallery_id.startswith("loc_") else "negative") for gallery_id in gallery_ids}
    if place_id not in gallery_places:
        raise ValueError(f"Missing positive gallery for {query_id}: {place_id}")

    if pooling == "best_frame":
        query_embedding = query_embeddings[query_id]
        gallery_matrix = torch.stack([pooled_embedding(gallery_embeddings[item], "mean") for item in gallery_ids])
        similarities = query_embedding @ gallery_matrix.t()
        scores = similarities.max(dim=0).values
    else:
        query_embedding = pooled_embedding(query_embeddings[query_id], pooling)
        gallery_matrix = torch.stack([pooled_embedding(gallery_embeddings[item], pooling) for item in gallery_ids])
        scores = query_embedding @ gallery_matrix.t()

    order = torch.argsort(scores, descending=True).tolist()
    positive_index = gallery_ids.index(place_id)
    rank = 1 + order.index(positive_index)
    negative_indices = [index for index, item in enumerate(gallery_ids) if item != place_id]
    hardest_negative = max(negative_indices, key=lambda index: float(scores[index].item()))
    margin = float(scores[positive_index].item() - scores[hardest_negative].item())
    return {
        "query_id": query_id,
        "positive_gallery_id": place_id,
        "pooling": pooling,
        "correct_rank": rank,
        "top1_hit": int(rank <= 1),
        "top5_hit": int(rank <= 5),
        "top10_hit": int(rank <= 10),
        "retrieval_margin": margin,
        "correct_similarity": float(scores[positive_index].item()),
        "hardest_negative_similarity": float(scores[hardest_negative].item()),
    }


def annotate_quality(
    row: Dict[str, object], original: torch.Tensor, protected: torch.Tensor, metrics: Mapping[str, float]
) -> None:
    row.update(metrics)
    row["flicker_score"] = flicker_score(protected)
    row["perturbation_stability"] = perturbation_stability(original, protected)
    row["psnr_mean"] = float(np.mean([psnr_torch(original[i], protected[i]) for i in range(original.size(0))]))
    row["ssim_mean"] = float(
        np.mean(
            [
                ssim_grayscale_np(
                    original[i].clamp(0, 255).byte().numpy().transpose(1, 2, 0),
                    protected[i].clamp(0, 255).byte().numpy().transpose(1, 2, 0),
                )
                for i in range(original.size(0))
            ]
        )
    )


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, int, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["variant"]),
                str(row["backbone"]),
                int(row["clip_len"]),
                str(row["pooling"]),
            )
        ].append(row)
    numeric = (
        "correct_rank",
        "top1_hit",
        "top5_hit",
        "top10_hit",
        "retrieval_margin",
        "psnr_mean",
        "ssim_mean",
        "effective_mse",
        "flicker_score",
        "perturbation_stability",
    )
    summaries: List[Dict[str, object]] = []
    for key, members in sorted(groups.items()):
        variant, backbone, clip_len, pooling = key
        summary: Dict[str, object] = {
            "variant": variant,
            "backbone": backbone,
            "clip_len": clip_len,
            "pooling": pooling,
            "num_queries": len(members),
        }
        for name in numeric:
            values = np.asarray([float(item[name]) for item in members], dtype=np.float64)
            summary[f"{name}_mean"] = float(values.mean())
            summary[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def evaluate_clips(
    query_clips: Mapping[str, torch.Tensor],
    gallery_clips: Mapping[str, torch.Tensor],
    sensnet: nn.Module,
    embedder: nn.Module,
    cfg: Mapping[str, object],
    device: torch.device,
    args: argparse.Namespace,
    backbone: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    input_size = default_input_size_for_backbone(backbone)
    for clip_len in sorted(set(int(value) for value in args.clip_lengths)):
        query_subset = {key: center_clip(value, clip_len) for key, value in query_clips.items()}
        gallery_subset = {key: center_clip(value, clip_len) for key, value in gallery_clips.items()}
        raw_query_embeddings = encode_clips(embedder, query_subset, device, input_size)
        raw_gallery_embeddings = encode_clips(embedder, gallery_subset, device, input_size)
        for variant in args.variants:
            for seed in args.seeds:
                protected: Dict[str, torch.Tensor] = {}
                quality: Dict[str, Dict[str, float]] = {}
                for query_id, original in query_subset.items():
                    sanitized, metrics = protect_review_clip(
                        original,
                        sensnet,
                        cfg,
                        device,
                        variant,
                        int(seed),
                    )
                    protected[query_id] = sanitized
                    quality[query_id] = metrics
                sanitized_embeddings = encode_clips(embedder, protected, device, input_size)
                for pooling in POOLINGS:
                    for query_id, original in query_subset.items():
                        row = retrieval_row(
                            protected[query_id],
                            gallery_subset,
                            query_id,
                            query_id,
                            pooling,
                            sanitized_embeddings,
                            raw_gallery_embeddings,
                        )
                        row.update(
                            {
                                "variant": variant,
                                "seed": int(seed),
                                "backbone": backbone,
                                "clip_len": clip_len,
                            }
                        )
                        annotate_quality(row, original, protected[query_id], quality[query_id])
                        rows.append(row)
        raw_gallery = raw_gallery_embeddings
        for pooling in POOLINGS:
            for query_id, original in query_subset.items():
                row = retrieval_row(
                    original,
                    gallery_subset,
                    query_id,
                    query_id,
                    pooling,
                    raw_query_embeddings,
                    raw_gallery,
                )
                row.update(
                    {
                        "variant": "raw",
                        "seed": "raw",
                        "backbone": backbone,
                        "clip_len": clip_len,
                        "flicker_score": flicker_score(original),
                        "perturbation_stability": 0.0,
                        "psnr_mean": 100.0,
                        "ssim_mean": 1.0,
                        "effective_mse": 0.0,
                    }
                )
                rows.append(row)
    return rows


def build_proxy_sequences(args: argparse.Namespace, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict]:
    pairs, distractor_ids, hard_meta = discover_paired_locations(
        root=args.monitoring_root,
        resize_hw=(int(args.resize_h), int(args.resize_w)),
        num_queries=int(args.num_queries),
        max_gallery=int(args.max_gallery),
        min_frames=int(args.min_frames),
        pair_pool_size=int(args.pair_pool_size),
        device=device,
    )
    query_clip_ids = [item["query_clip_id"] for item in pairs]
    positive_clip_ids = [item["gallery_clip_id"] for item in pairs]
    query_ds = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=query_clip_ids,
        view="query",
        clip_len=int(args.base_clip_len),
        resize_hw=(int(args.resize_h), int(args.resize_w)),
        min_frames=int(args.min_frames),
    )
    positive_ds = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=positive_clip_ids,
        view="gallery",
        clip_len=int(args.base_clip_len),
        resize_hw=(int(args.resize_h), int(args.resize_w)),
        min_frames=int(args.min_frames),
    )
    query_by_clip = {sample.clip_id: sample.frames for sample in query_ds}
    positive_by_clip = {sample.clip_id: sample.frames for sample in positive_ds}
    query_clips = {item["location_id"]: query_by_clip[item["query_clip_id"]] for item in pairs}
    gallery_clips = {
        item["location_id"]: positive_by_clip[item["gallery_clip_id"]] for item in pairs
    }
    all_distractor_ids = [clip_id for clip_id in distractor_ids if clip_id not in set(query_clip_ids)]
    distractor_ds = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=all_distractor_ids[: max(0, int(args.gallery_size) - len(gallery_clips))],
        view="gallery",
        clip_len=int(args.base_clip_len),
        resize_hw=(int(args.resize_h), int(args.resize_w)),
        min_frames=int(args.min_frames),
    )
    for sample in distractor_ds:
        gallery_clips[sample.clip_id] = sample.frames
    gallery_clips = dict(list(gallery_clips.items())[: int(args.gallery_size)])
    metadata = {
        "pairs": pairs,
        "hard_distractors": hard_meta,
        "query_count": len(query_clips),
        "gallery_count": len(gallery_clips),
        "benchmark": "controlled paired-scene proxy",
    }
    return query_clips, gallery_clips, metadata


class TinyEmbedder(nn.Module):
    """Download-free embedder used only by the smoke test."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(images, (4, 4)).flatten(1)


def run_smoke(args: argparse.Namespace) -> Path:
    from run_train import SensitiveRegionNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).with_name("icme2027_sequence_retrieval_smoke")
    ensure_dir(output_dir)
    torch.manual_seed(20270903)
    count = int(args.smoke_queries)
    clip_len = int(args.smoke_clip_len)
    size = int(args.smoke_size)
    query_clips = {
        f"loc_{index:03d}": torch.rand(clip_len, 3, size, size) * 255.0
        for index in range(count)
    }
    gallery_clips = dict(query_clips)
    gallery_clips.update(
        {
            "negative_0": torch.rand(clip_len, 3, size, size) * 255.0,
            "negative_1": torch.rand(clip_len, 3, size, size) * 255.0,
        }
    )
    model = SensitiveRegionNet().to(device).eval()
    embedder = TinyEmbedder().to(device).eval()
    cfg = load_yaml(args.config)
    smoke_args = argparse.Namespace(
        clip_lengths=[1, 2, 4, min(8, clip_len)],
        variants=list(DEFAULT_VARIANTS),
        seeds=[int(args.seeds[0])],
    )
    rows = evaluate_clips(
        query_clips, gallery_clips, model, embedder, cfg, device, smoke_args, "tiny"
    )
    finite = all(
        math.isfinite(float(row["retrieval_margin"]))
        and math.isfinite(float(row["flicker_score"]))
        and math.isfinite(float(row["perturbation_stability"]))
        for row in rows
    )
    expected = count * len(set(smoke_args.clip_lengths)) * (len(DEFAULT_VARIANTS) + 1) * len(POOLINGS)
    if len(rows) != expected or not finite:
        raise RuntimeError("ICME-M4 sequence smoke gate failed.")
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "summary.csv", aggregate_rows(rows))
    write_json(
        output_dir / "run_metadata.json",
        {
            "review_cycle": "ICME-2027",
            "mode": "smoke",
            "device": str(device),
            "clip_lengths": smoke_args.clip_lengths,
            "poolings": list(POOLINGS),
            "rows": len(rows),
            "scientific_evidence": False,
        },
    )
    print(f"[smoke] ICME-M4 gate passed on {device}; wrote {len(rows)} rows to {output_dir}")
    return output_dir


def run_proxy(args: argparse.Namespace) -> Path:
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_clips, gallery_clips, selection = build_proxy_sequences(args, device)
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
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
        rows.extend(
            evaluate_clips(query_clips, gallery_clips, sensnet, embedder, cfg, device, args, backbone)
        )
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "summary.csv", aggregate_rows(rows))
    write_json(
        output_dir / "selection.json",
        {
            "review_cycle": "ICME-2027",
            "mode": "proxy",
            **selection,
            "clip_lengths": args.clip_lengths,
            "poolings": list(POOLINGS),
            "variants": args.variants,
            "backbones": args.backbones,
            "seeds": args.seeds,
            "scientific_evidence": False,
        },
    )
    write_json(
        output_dir / "run_metadata.json",
        {
            "review_cycle": "ICME-2027",
            "mode": "proxy",
            "device": str(device),
            "clip_lengths": args.clip_lengths,
            "poolings": list(POOLINGS),
            "variants": args.variants,
            "backbones": args.backbones,
            "seeds": args.seeds,
            "scientific_evidence": False,
            "note": "Proxy output is sequence-threat-model evidence only, not geographic ground truth.",
        },
    )
    print(f"[proxy] wrote {len(rows)} sequence-level rows to {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_proxy(args)


if __name__ == "__main__":
    main()
