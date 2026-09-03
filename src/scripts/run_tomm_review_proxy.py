"""Run the ICME 2027 review-cycle proxy ablation and retrieval diagnostics.

This script addresses reviewer requests R2-2 and R3-3/R3-7/R3-8 on the
controlled paired-scene proxy benchmark.  It deliberately keeps proxy results
separate from the geotagged benchmark requested in R2-1.  The ``smoke`` mode
uses synthetic tensors and a local tiny embedder, so it never downloads model
weights or creates paper evidence.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from datasets.monitoring_clip_dataset import MonitoringClipDataset  # noqa: E402
from eval.metrics import psnr_torch, ssim_grayscale_np  # noqa: E402
from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    build_gallery_embeddings,
    default_input_size_for_backbone,
    make_default_embedder,
    preprocess_for_embed,
)
from main import load_sensnet_checkpoint  # noqa: E402
from models.dynamic_crf import DynamicCRF, DynamicCRFConfig  # noqa: E402
from privacy.NCP import NCPAllocator, NCPConfig  # noqa: E402
from privacy.noise_injector import NoiseConfig, NoiseInjector  # noqa: E402
from scripts.run_controlled_retrieval_benchmark import (  # noqa: E402
    build_external_distractor_frames,
    build_gallery_tensor,
    collect_external_image_paths,
    discover_paired_locations,
    select_eval_frame,
    tensor_to_uint8_image,
)
from utils.config import load_yaml  # noqa: E402


VARIANT_LABELS = {
    "full": "PPEDCRF",
    "no_temporal": "w/o temporal consistency",
    "no_ncp": "w/o NCP (fixed strength)",
    "unary_only": "unary-only + NCP",
    "no_dcrf": "no-DCRF + fixed strength",
    "masked_blur": "mask-guided blur",
    "masked_mosaic": "mask-guided mosaic",
    "global_noise": "global Gaussian noise",
    "attacker_aware": "attacker-aware feature suppression",
}

REVIEW_VARIANTS = (
    "full",
    "no_temporal",
    "no_ncp",
    "unary_only",
    "no_dcrf",
    "masked_blur",
    "masked_mosaic",
    "global_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICME 2027 review-cycle proxy ablation and per-query diagnostics."
    )
    parser.add_argument("--mode", choices=("smoke", "proxy"), default="smoke")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument(
        "--monitoring_root",
        default=r"F:\work\datasets\monitoring\images",
    )
    parser.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    parser.add_argument("--output_dir", default="src/outputs/tomm_review_proxy")
    parser.add_argument("--num_queries", type=int, default=12)
    parser.add_argument("--pair_pool_size", type=int, default=240)
    parser.add_argument("--max_gallery", type=int, default=48)
    parser.add_argument("--gallery_sizes", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--clip_len", type=int, default=4)
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--min_frames", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["resnet18"],
        help="Attacker backbones. Use all eight names for the full 4c rerun.",
    )
    parser.add_argument("--coco_root", default="")
    parser.add_argument("--digica_root", default="")
    parser.add_argument("--max_external_distractors", type=int, default=0)
    parser.add_argument("--include_attacker_aware", action="store_true")
    parser.add_argument("--attacker_steps", type=int, default=20)
    parser.add_argument("--attacker_step_size", type=float, default=1.0)
    parser.add_argument("--attacker_linf", type=float, default=8.0)
    parser.add_argument("--smoke_queries", type=int, default=3)
    parser.add_argument("--smoke_size", type=int, default=64)
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Override config noise sigma for every noise-based variant in this "
        "run (used for the matched-operating-point sigma sweep, R2-2/R3-3/R3-4). "
        "Leave unset to use the config's default sigma.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


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


def tensor_to_image(frame: torch.Tensor) -> np.ndarray:
    return tensor_to_uint8_image(frame)


def apply_masked_blur(frame: torch.Tensor, mask: torch.Tensor, kernel_size: int = 21) -> torch.Tensor:
    pad = kernel_size // 2
    blurred = F.avg_pool2d(
        F.pad(frame, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )
    return torch.clamp(frame * (1.0 - mask) + blurred * mask, 0.0, 255.0)


def apply_masked_mosaic(frame: torch.Tensor, mask: torch.Tensor, block_size: int = 12) -> torch.Tensor:
    _, _, height, width = frame.shape
    small_h = max(1, height // block_size)
    small_w = max(1, width // block_size)
    coarse = F.interpolate(frame, size=(small_h, small_w), mode="bilinear", align_corners=False)
    mosaic = F.interpolate(coarse, size=(height, width), mode="nearest")
    return torch.clamp(frame * (1.0 - mask) + mosaic * mask, 0.0, 255.0)


def _variant_mask_and_strength(
    unary: torch.Tensor,
    crf: DynamicCRF,
    ncp: NCPAllocator,
    prev_prob: torch.Tensor | None,
    variant: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return mask, strength, and the next temporal prior for one frame."""
    if variant in {"unary_only", "no_dcrf"}:
        refined_prob = torch.sigmoid(unary)
        next_prev = None
    elif variant == "no_temporal":
        refined_prob, _ = crf.refine(unary, prev_prob=None, flow=None)
        next_prev = None
    else:
        refined_prob, next_prev = crf.refine(unary, prev_prob=prev_prob, flow=None)

    if variant in {"no_ncp", "no_dcrf"}:
        strength = torch.ones_like(refined_prob)
    else:
        strength = ncp.allocate(refined_prob)
    return refined_prob, strength, next_prev


@torch.no_grad()
def protect_review_clip(
    frames: torch.Tensor,
    sensnet: nn.Module,
    cfg: Mapping[str, object],
    device: torch.device,
    variant: str,
    seed: int,
    allow_ssim_fallback: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Protect a clip and return quality/energy diagnostics."""
    dcfg = cfg["ppedcrf"]["dynamic_crf"]  # type: ignore[index]
    ncfg = cfg["ppedcrf"]["ncp"]  # type: ignore[index]
    pcfg = cfg["ppedcrf"]["noise"]  # type: ignore[index]
    temporal_weight = 0.0 if variant == "no_temporal" else float(dcfg["temporal_weight"])
    crf = DynamicCRF(
        DynamicCRFConfig(
            n_iters=int(dcfg["n_iters"]),
            spatial_weight=float(dcfg["spatial_weight"]),
            temporal_weight=temporal_weight,
            smooth_kernel=int(dcfg["smooth_kernel"]),
        )
    )
    ncp = NCPAllocator(NCPConfig(alpha=float(ncfg.get("alpha", 1.0))))
    injector = NoiseInjector(
        NoiseConfig(
            mode=str(pcfg["mode"]),
            sigma=float(pcfg["sigma"]),
            clamp_min=float(pcfg["clamp_min"]),
            clamp_max=float(pcfg["clamp_max"]),
            seed=int(seed),
        )
    )

    protected_frames: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    prev_prob: torch.Tensor | None = None
    for t in range(frames.size(0)):
        frame = frames[t : t + 1].to(device)
        unary = sensnet(frame)
        mask, strength, next_prev = _variant_mask_and_strength(
            unary, crf, ncp, prev_prob, variant
        )
        if variant == "masked_blur":
            protected = apply_masked_blur(frame, mask)
        elif variant == "masked_mosaic":
            protected = apply_masked_mosaic(frame, mask)
        elif variant == "global_noise":
            protected = injector.apply(frame, torch.ones_like(mask), torch.ones_like(mask), t_index=t)
        else:
            protected = injector.apply(frame, mask, strength, t_index=t)
        protected_frames.append(protected.squeeze(0).cpu())
        masks.append(mask.squeeze(0).cpu())
        prev_prob = next_prev

    protected_clip = torch.stack(protected_frames, dim=0)
    original = frames.detach().cpu().float()
    delta = protected_clip.float() - original
    full_mse = float(torch.mean(delta.square()).item())
    support = torch.stack(masks, dim=0).float()
    support_coverage = float(support.mean().item())
    support_den = float((support.sum() * 3.0).item())
    support_mse = float((delta.square() * support).sum().item() / max(support_den, 1.0))
    frame_psnr = float(np.mean([psnr_torch(original[i], protected_clip[i]) for i in range(original.size(0))]))
    try:
        frame_ssim = float(
            np.mean(
                [
                    ssim_grayscale_np(tensor_to_image(original[i]), tensor_to_image(protected_clip[i]))
                    for i in range(original.size(0))
                ]
            )
        )
    except ImportError:
        if not allow_ssim_fallback:
            raise
        # Smoke-only fallback. Formal runs must install scikit-image and use
        # the repository's real SSIM implementation.
        frame_ssim = 0.0
    return protected_clip, {
        "psnr_mean": frame_psnr,
        "ssim_mean": frame_ssim,
        "effective_mse": full_mse,
        "support_mse": support_mse,
        "support_coverage": support_coverage,
    }


def normalized_embeddings(
    embedder: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    input_size: int,
    batch_size: int = 32,
) -> torch.Tensor:
    """Embed images in fixed-size chunks rather than one forward pass.

    Some attacker backbones (e.g. Patch-NetVLAD, whose patch-level dense
    descriptors are far larger per image than a global-pooled CNN embedding)
    OOM on a single unbatched forward pass once the gallery reaches official
    MSLS scale (1000 images), even though the same call is fine at the
    smaller proxy-benchmark gallery sizes (<=100). Chunking keeps peak
    activation memory bounded by batch_size regardless of total gallery size.
    """
    chunks = []
    for start in range(0, images.size(0), batch_size):
        x = preprocess_for_embed(images[start : start + batch_size].to(device), input_size)
        chunks.append(embedder(x))
    emb = torch.cat(chunks, dim=0)
    return F.normalize(emb, dim=1)


def detailed_retrieval(
    query_images: torch.Tensor,
    query_ids: Sequence[str],
    gallery_images: torch.Tensor,
    gallery_ids: Sequence[str],
    embedder: nn.Module,
    device: torch.device,
    input_size: int,
    quality_by_query: Mapping[str, Mapping[str, float]] | None = None,
    positive_place_by_query: Mapping[str, str] | None = None,
    gallery_place_by_id: Mapping[str, str] | None = None,
) -> List[Dict[str, object]]:
    """Return per-query ranks and correct-versus-hardest-negative margins."""
    embedder.eval().to(device)
    with torch.no_grad():
        query_emb = normalized_embeddings(embedder, query_images, device, input_size)
        gallery_emb = normalized_embeddings(embedder, gallery_images, device, input_size)
    similarity = query_emb @ gallery_emb.t()
    rows: List[Dict[str, object]] = []
    for i, query_id in enumerate(query_ids):
        sims = similarity[i]
        order = torch.argsort(sims, descending=True).tolist()
        if positive_place_by_query is None:
            positive = [j for j, gallery_id in enumerate(gallery_ids) if gallery_id == query_id]
        else:
            target_place = positive_place_by_query[query_id]
            positive = [
                j
                for j, gallery_id in enumerate(gallery_ids)
                if gallery_place_by_id is not None and gallery_place_by_id.get(gallery_id) == target_place
            ]
        if not positive:
            raise ValueError(f"No positive gallery item found for query {query_id}.")
        positive_idx = max(positive, key=lambda j: float(sims[j].item()))
        rank = 1 + order.index(positive_idx)
        negative = [j for j in range(len(gallery_ids)) if j not in positive]
        hardest_idx = max(negative, key=lambda j: float(sims[j].item())) if negative else positive_idx
        correct_similarity = float(sims[positive_idx].item())
        hardest_similarity = float(sims[hardest_idx].item())
        row: Dict[str, object] = {
            "query_id": query_id,
            "correct_gallery_id": gallery_ids[positive_idx],
            "correct_rank": rank,
            "correct_similarity": correct_similarity,
            "hardest_negative_id": gallery_ids[hardest_idx],
            "hardest_negative_similarity": hardest_similarity,
            "retrieval_margin": correct_similarity - hardest_similarity,
            "top1_gallery_id": gallery_ids[order[0]],
            "top5_hit": int(rank <= 5),
            "top10_hit": int(rank <= 10),
        }
        if quality_by_query is not None and query_id in quality_by_query:
            row.update(quality_by_query[query_id])
        rows.append(row)
    return rows


def optimize_attacker_aware_query(
    query_image: torch.Tensor,
    positive_gallery_embedding: torch.Tensor,
    embedder: nn.Module,
    device: torch.device,
    input_size: int,
    steps: int,
    step_size: float,
    linf_bound: float,
) -> torch.Tensor:
    """Suppress similarity to one fixed gallery embedding with projected updates."""
    original = query_image.detach().float().to(device).unsqueeze(0)
    candidate = original.clone()
    frozen_flags = [parameter.requires_grad for parameter in embedder.parameters()]
    for parameter in embedder.parameters():
        parameter.requires_grad_(False)
    target = positive_gallery_embedding.detach().to(device).view(1, -1)
    try:
        for _ in range(max(1, int(steps))):
            candidate.requires_grad_(True)
            query_embedding = normalized_embeddings(embedder, candidate, device, input_size)
            loss = (query_embedding * target).sum()
            gradient = torch.autograd.grad(loss, candidate, only_inputs=True)[0]
            candidate = candidate - float(step_size) * gradient.sign()
            delta = (candidate - original).clamp(-float(linf_bound), float(linf_bound))
            candidate = (original + delta).clamp(0.0, 255.0).detach()
    finally:
        for parameter, flag in zip(embedder.parameters(), frozen_flags):
            parameter.requires_grad_(flag)
    return candidate.squeeze(0).cpu()


def aggregate_rows(rows: Sequence[Mapping[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(group_key) for group_key in group_keys)
        groups.setdefault(key, []).append(row)
    outputs: List[Dict[str, object]] = []
    numeric = (
        "correct_rank",
        "correct_similarity",
        "hardest_negative_similarity",
        "retrieval_margin",
        "top5_hit",
        "top10_hit",
        "psnr_mean",
        "ssim_mean",
        "effective_mse",
        "support_mse",
        "support_coverage",
    )
    for key, members in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        output: Dict[str, object] = dict(zip(group_keys, key))
        output["num_queries"] = len(members)
        output["top1"] = float(np.mean([int(float(m["correct_rank"])) == 1 for m in members]))
        for name in numeric:
            values = np.asarray(
                [float(m[name]) for m in members if m.get(name) is not None],
                dtype=np.float64,
            )
            if len(values) == 0:
                output[f"{name}_mean"] = None
                output[f"{name}_std"] = None
            else:
                output[f"{name}_mean"] = float(values.mean())
                output[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        outputs.append(output)
    return outputs


def _build_proxy_data(args: argparse.Namespace, device: torch.device):
    resize_hw = (int(args.resize_h), int(args.resize_w))
    pairs, distractors, hard_meta = discover_paired_locations(
        root=args.monitoring_root,
        resize_hw=resize_hw,
        num_queries=int(args.num_queries),
        max_gallery=int(args.max_gallery),
        min_frames=int(args.min_frames),
        pair_pool_size=int(args.pair_pool_size),
        device=device,
    )
    query_ids = [item["location_id"] for item in pairs]
    query_clip_ids = [item["query_clip_id"] for item in pairs]
    gallery_clip_ids = [item["gallery_clip_id"] for item in pairs]

    external_by_id: Dict[str, torch.Tensor] = {}
    external_ids: List[str] = []
    if args.max_external_distractors > 0:
        paths: List[Path] = []
        for root in (args.coco_root, args.digica_root):
            if root:
                paths.extend(collect_external_image_paths(root, max_items=args.max_external_distractors))
        external_by_id = build_external_distractor_frames(
            paths, resize_hw=resize_hw, max_items=int(args.max_external_distractors)
        )
        external_ids = sorted(external_by_id)

    query_ds = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=query_clip_ids,
        view="query",
        clip_len=int(args.clip_len),
        resize_hw=resize_hw,
        min_frames=int(args.min_frames),
    )
    positive_ds = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=gallery_clip_ids,
        view="gallery",
        clip_len=int(args.clip_len),
        resize_hw=resize_hw,
        min_frames=int(args.min_frames),
    )
    distractor_ds = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=distractors,
        view="gallery",
        clip_len=int(args.clip_len),
        resize_hw=resize_hw,
        min_frames=int(args.min_frames),
    )
    query_by_clip = {sample.clip_id: sample.frames for sample in query_ds}
    positive_by_clip = {sample.clip_id: select_eval_frame(sample.frames) for sample in positive_ds}
    distractor_by_clip = {sample.clip_id: select_eval_frame(sample.frames) for sample in distractor_ds}

    query_clips = {item["location_id"]: query_by_clip[item["query_clip_id"]] for item in pairs}
    raw_queries = {location_id: select_eval_frame(clip) for location_id, clip in query_clips.items()}
    gallery_by_id: Dict[str, torch.Tensor] = {
        item["location_id"]: positive_by_clip[item["gallery_clip_id"]] for item in pairs
    }
    for label, clip_id in zip(distractors, distractors):
        if clip_id in distractor_by_clip:
            gallery_by_id[clip_id] = distractor_by_clip[clip_id]
    gallery_by_id.update(external_by_id)
    all_distractors = list(distractors) + external_ids
    return pairs, hard_meta, query_ids, query_clips, raw_queries, gallery_by_id, all_distractors


def run_proxy(args: argparse.Namespace) -> Path:
    cfg = load_yaml(args.config)
    if args.sigma is not None:
        cfg = copy.deepcopy(cfg)
        cfg["ppedcrf"]["noise"] = dict(cfg["ppedcrf"]["noise"])
        cfg["ppedcrf"]["noise"]["sigma"] = float(args.sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    print(f"[proxy] device={device}")
    data = _build_proxy_data(args, device)
    pairs, hard_meta, query_ids, query_clips, raw_queries, gallery_by_id, distractors = data
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)

    selection = {
        "review_cycle": "ICME-2027",
        "benchmark": "controlled paired-scene proxy",
        "pairs": pairs,
        "hard_distractors": hard_meta,
        "query_count": len(query_ids),
        "gallery_sizes": args.gallery_sizes,
        "seeds": args.seeds,
        "variants": list(REVIEW_VARIANTS),
        "scientific_evidence": False,
    }
    write_json(output_dir / "selection.json", selection)

    protected: Dict[Tuple[str, int], torch.Tensor] = {}
    quality: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = {}
    for variant in REVIEW_VARIANTS:
        for seed in args.seeds:
            images: List[torch.Tensor] = []
            quality_for_variant: Dict[str, Dict[str, float]] = {}
            for query_id in query_ids:
                clip, metrics = protect_review_clip(
                    query_clips[query_id], sensnet, cfg, device, variant, int(seed)
                )
                frame = select_eval_frame(clip)
                images.append(frame)
                quality_for_variant[query_id] = metrics
            protected[(variant, int(seed))] = torch.stack(images, dim=0)
            quality[(variant, int(seed))] = quality_for_variant

    per_query: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []
    raw_query_tensor = torch.stack([raw_queries[query_id] for query_id in query_ids], dim=0)
    all_variants = list(REVIEW_VARIANTS)
    if args.include_attacker_aware:
        all_variants.append("attacker_aware")
    for backbone in args.backbones:
        rcfg = RetrievalConfig(
            backbone=backbone,
            device=str(device),
            normalize=True,
            input_size=default_input_size_for_backbone(backbone),
            topk=(1, 5, 10),
        )
        embedder = make_default_embedder(rcfg).eval().to(device)
        for gallery_size in sorted(args.gallery_sizes):
            gallery_tensor, gallery_ids = build_gallery_tensor(
                gallery_frame_by_id=gallery_by_id,
                query_ids=query_ids,
                distractor_ids=distractors,
                gallery_size=int(gallery_size),
            )
            gallery_emb = build_gallery_embeddings(rcfg, embedder, gallery_tensor)
            raw_rows = detailed_retrieval(
                raw_query_tensor,
                query_ids,
                gallery_tensor,
                gallery_ids,
                embedder,
                device,
                rcfg.input_size,
            )
            for row in raw_rows:
                row.update({"variant": "raw", "label": "raw query", "seed": "raw"})
                row.update({"backbone": backbone, "gallery_size": int(gallery_size)})
                per_query.append(row)
            for variant in all_variants:
                if variant == "attacker_aware":
                    if backbone != "resnet18":
                        continue
                    aware_images: List[torch.Tensor] = []
                    aware_quality: Dict[str, Dict[str, float]] = {}
                    for index, query_id in enumerate(query_ids):
                        positive_indices = [j for j, gallery_id in enumerate(gallery_ids) if gallery_id == query_id]
                        if not positive_indices:
                            raise ValueError(f"No positive gallery item for attacker-aware query {query_id}.")
                        aware = optimize_attacker_aware_query(
                            raw_query_tensor[index],
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
                            "psnr_mean": psnr_torch(raw_query_tensor[index], aware),
                            "effective_mse": float(torch.mean((aware - raw_query_tensor[index]).float().square()).item()),
                            "support_mse": float(torch.mean((aware - raw_query_tensor[index]).float().square()).item()),
                            "support_coverage": 1.0,
                        }
                    q_rows = detailed_retrieval(
                        torch.stack(aware_images),
                        query_ids,
                        gallery_tensor,
                        gallery_ids,
                        embedder,
                        device,
                        rcfg.input_size,
                        quality_by_query=aware_quality,
                    )
                    for row in q_rows:
                        row.update(
                            {
                                "variant": "attacker_aware",
                                "label": VARIANT_LABELS["attacker_aware"],
                                "seed": "deterministic",
                                "backbone": backbone,
                                "gallery_size": int(gallery_size),
                                "attacker_steps": int(args.attacker_steps),
                                "attacker_linf": float(args.attacker_linf),
                            }
                        )
                    per_query.extend(q_rows)
                    continue
                for seed in args.seeds:
                    q_rows = detailed_retrieval(
                        protected[(variant, int(seed))],
                        query_ids,
                        gallery_tensor,
                        gallery_ids,
                        embedder,
                        device,
                        rcfg.input_size,
                        quality_by_query=quality[(variant, int(seed))],
                    )
                    for row in q_rows:
                        row.update(
                            {
                                "variant": variant,
                                "label": VARIANT_LABELS[variant],
                                "seed": int(seed),
                                "backbone": backbone,
                                "gallery_size": int(gallery_size),
                            }
                        )
                    per_query.extend(q_rows)
            summary.extend(
                aggregate_rows(
                    [
                        row
                        for row in per_query
                        if row.get("backbone") == backbone and row.get("gallery_size") == int(gallery_size)
                    ],
                    ("variant", "backbone", "gallery_size"),
                )
            )

    write_csv(output_dir / "per_query.csv", per_query)
    write_csv(output_dir / "summary.csv", summary)
    write_json(
        output_dir / "run_metadata.json",
        {
            "review_cycle": "ICME-2027",
            "mode": "proxy",
            "device": str(device),
            "backbones": args.backbones,
            "seeds": args.seeds,
            "sigma": float(cfg["ppedcrf"]["noise"]["sigma"]),
            "variants": list(REVIEW_VARIANTS),
            "scientific_evidence": False,
            "attacker_aware_requested": bool(args.include_attacker_aware),
            "note": "Promote to paper only after completion and checksum gates pass.",
        },
    )
    print(f"[proxy] wrote {len(per_query)} per-query rows and {len(summary)} summary rows to {output_dir}")
    return output_dir


class TinyEmbedder(nn.Module):
    """Download-free embedder used only by the smoke test."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, (4, 4))
        return pooled.flatten(1)


def run_smoke(args: argparse.Namespace) -> Path:
    from run_train import SensitiveRegionNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).with_name("tomm_review_smoke")
    ensure_dir(output_dir)
    torch.manual_seed(20260830)
    size = int(args.smoke_size)
    query_count = int(args.smoke_queries)
    frames = torch.rand(query_count, 4, 3, size, size) * 255.0
    gallery = torch.rand(query_count + 3, 3, size, size) * 255.0
    query_ids = [f"loc_{i:03d}" for i in range(query_count)]
    gallery_ids = list(query_ids) + [f"neg_{i:03d}" for i in range(3)]
    model = SensitiveRegionNet().to(device).eval()
    embedder = TinyEmbedder().to(device).eval()
    cfg = load_yaml(args.config)
    rows: List[Dict[str, object]] = []
    for variant in REVIEW_VARIANTS:
        protected_images: List[torch.Tensor] = []
        quality_by_query: Dict[str, Dict[str, float]] = {}
        for index, query_id in enumerate(query_ids):
            protected, metrics = protect_review_clip(
                frames[index], model, cfg, device, variant, seed=1234,
                allow_ssim_fallback=True,
            )
            protected_images.append(select_eval_frame(protected))
            quality_by_query[query_id] = metrics
        q_rows = detailed_retrieval(
            torch.stack(protected_images),
            query_ids,
            gallery,
            gallery_ids,
            embedder,
            device,
            input_size=size,
            quality_by_query=quality_by_query,
        )
        for row in q_rows:
            row.update({"variant": variant, "label": VARIANT_LABELS[variant], "device": str(device)})
        rows.extend(q_rows)
    summary = aggregate_rows(rows, ("variant",))
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(
        output_dir / "smoke_metadata.json",
        {
            "mode": "smoke",
            "device": str(device),
            "query_count": query_count,
            "size": size,
            "variants": list(REVIEW_VARIANTS),
            "scientific_evidence": False,
        },
    )
    finite = all(
        np.isfinite(float(row["retrieval_margin"])) and np.isfinite(float(row["effective_mse"]))
        for row in rows
    )
    if not finite or len(rows) != query_count * len(REVIEW_VARIANTS):
        raise RuntimeError("Smoke test produced non-finite values or missing variant rows.")
    print(f"[smoke] passed on {device}; wrote {len(rows)} rows to {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_proxy(args)


if __name__ == "__main__":
    main()
