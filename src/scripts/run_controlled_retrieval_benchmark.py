from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

from datasets.monitoring_clip_dataset import MonitoringClipDataset, iter_clip_ids
from datasets.driving_clip_dataset import _read_image, _resize_if_needed
from eval.metrics import flicker_score, perturbation_stability, psnr_torch, ssim_grayscale_np
from eval.retrieval_attack import RetrievalConfig, build_gallery_embeddings, make_default_embedder, query_topk
from main import load_sensnet_checkpoint
from models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from privacy.NCP import NCPAllocator, NCPConfig
from privacy.noise_injector import NoiseConfig, NoiseInjector
from utils.config import load_yaml


VARIANT_LABELS = {
    "ppedcrf": "PPEDCRF",
    "no_temporal": "w/o temporal consistency",
    "no_ncp": "w/o NCP (fixed sigma)",
    "masked_blur": "mask-guided blur",
    "masked_mosaic": "mask-guided mosaic",
    "random_mask": "random mask",
    "full_frame": "global Gaussian noise",
    "raw": "raw query",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled retrieval benchmark for PPEDCRF.")
    parser.add_argument("--config", type=str, default="src/config/config.yaml")
    parser.add_argument(
        "--monitoring_root",
        type=str,
        default=r"F:\work\datasets\monitoring\images",
        help="Directory that stores monitoring frames named as <clip_id>_frame<number>.jpg",
    )
    parser.add_argument("--checkpoint", type=str, default="src/outputs/sensnet_final.pt")
    parser.add_argument(
        "--coco_root",
        type=str,
        default=r"C:\work\datasets\Coco",
        help="Optional external image pool root (COCO style) used for extra hard distractors.",
    )
    parser.add_argument(
        "--digica_root",
        type=str,
        default=r"C:\work\datasets\digica\digica_v4.3",
        help="Optional external image pool root (Digica style) used for extra hard distractors.",
    )
    parser.add_argument(
        "--max_external_distractors",
        type=int,
        default=256,
        help="Maximum number of external-image distractors to consider from COCO/Digica pools.",
    )
    parser.add_argument("--num_queries", type=int, default=12)
    parser.add_argument("--max_gallery", type=int, default=48)
    parser.add_argument("--pair_pool_size", type=int, default=240)
    parser.add_argument("--gallery_sizes", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--clip_len", type=int, default=4)
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--min_frames", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    parser.add_argument("--backbones", type=str, nargs="+", default=["resnet18", "resnet50"])
    parser.add_argument("--frontier_sigmas", type=float, nargs="+", default=[8.0, 16.0, 24.0])
    parser.add_argument("--ablation_sigmas", type=float, nargs="+", default=[8.0, 16.0, 24.0, 32.0],
                        help="Sigma values for the full ablation table (DCRF/NCP comparison)")
    parser.add_argument("--matched_psnr_targets", type=float, nargs="+", default=[30.0, 33.0, 36.0],
                        help="PSNR targets for matched-utility comparison (dB)")
    parser.add_argument("--blur_kernel_sizes", type=int, nargs="+", default=[5, 11, 21, 31],
                        help="Kernel sizes for blur parameter sweep (m7)")
    parser.add_argument("--mosaic_block_sizes", type=int, nargs="+", default=[4, 8, 12, 20],
                        help="Block sizes for mosaic parameter sweep (m7)")
    parser.add_argument("--output_dir", type=str, default="src/outputs/controlled_retrieval")
    return parser.parse_args()


def clone_cfg(cfg: dict) -> dict:
    return copy.deepcopy(cfg)


def build_variant_modules(cfg: dict, variant: str, device: torch.device, seed: int) -> Tuple[DynamicCRF, NCPAllocator, NoiseInjector]:
    dcfg = cfg["ppedcrf"]["dynamic_crf"]
    ncfg = cfg["ppedcrf"]["ncp"]
    pcfg = cfg["ppedcrf"]["noise"]

    temporal_weight = float(dcfg["temporal_weight"])
    if variant == "no_temporal":
        temporal_weight = 0.0

    crf = DynamicCRF(
        DynamicCRFConfig(
            n_iters=int(dcfg["n_iters"]),
            spatial_weight=float(dcfg["spatial_weight"]),
            temporal_weight=temporal_weight,
            smooth_kernel=int(dcfg["smooth_kernel"]),
        )
    )
    ncp = NCPAllocator(
        NCPConfig(alpha=float(ncfg.get("alpha", 1.0))),
        class_sensitivity=ncfg.get("class_sensitivity", {}),
    )
    injector = NoiseInjector(
        NoiseConfig(
            mode=str(pcfg["mode"]),
            sigma=float(pcfg["sigma"]),
            clamp_min=float(pcfg["clamp_min"]),
            clamp_max=float(pcfg["clamp_max"]),
            seed=seed,
        )
    )
    return crf, ncp, injector


def make_random_mask_like(sens_map: Tensor, seed: int, t_index: int) -> Tensor:
    coverage = float(sens_map.mean().item())
    coverage = max(0.01, min(0.99, coverage))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 9973 * (t_index + 1))
    rand = torch.rand(sens_map.shape, generator=generator, device="cpu").to(sens_map.device)
    return (rand < coverage).float()


def mean_temporal_iou(masks: List[Tensor]) -> float:
    if len(masks) < 2:
        return 0.0

    scores = []
    for prev, cur in zip(masks[:-1], masks[1:]):
        prev_b = (prev > 0.5).float()
        cur_b = (cur > 0.5).float()
        intersection = (prev_b * cur_b).sum().item()
        union = ((prev_b + cur_b) > 0).float().sum().item()
        scores.append(intersection / union if union > 0 else 1.0)
    return float(sum(scores) / len(scores))


def apply_mask_guided_blur(frame: Tensor, mask: Tensor, kernel_size: int = 11) -> Tensor:
    """
    A simple task-aligned baseline that blurs only the inferred sensitive region.

    The mask remains continuous so that transition boundaries stay smooth and
    the baseline uses the same spatial support as PPEDCRF.
    """
    pad = kernel_size // 2
    blurred = F.avg_pool2d(F.pad(frame, (pad, pad, pad, pad), mode="reflect"), kernel_size=kernel_size, stride=1)
    return torch.clamp(frame * (1.0 - mask) + blurred * mask, 0.0, 255.0)


def apply_mask_guided_mosaic(frame: Tensor, mask: Tensor, block_size: int = 12) -> Tensor:
    """
    A second task-aligned baseline that pixelates only the inferred sensitive
    region while leaving the rest of the frame untouched.
    """
    _, _, height, width = frame.shape
    small_h = max(1, height // block_size)
    small_w = max(1, width // block_size)
    coarse = F.interpolate(frame, size=(small_h, small_w), mode="bilinear", align_corners=False)
    mosaic = F.interpolate(coarse, size=(height, width), mode="nearest")
    return torch.clamp(frame * (1.0 - mask) + mosaic * mask, 0.0, 255.0)


@torch.no_grad()
def protect_clip_variant(
    frames: Tensor,
    sensnet,
    cfg: dict,
    device: torch.device,
    variant: str,
    seed: int,
) -> Tuple[Tensor, float]:
    crf, ncp, injector = build_variant_modules(cfg, variant=variant, device=device, seed=seed)
    prev_prob = None
    protected_frames = []
    masks = []

    for t in range(frames.size(0)):
        frame = frames[t : t + 1].to(device)
        unary = sensnet(frame)

        if variant == "no_temporal":
            refined_prob, _ = crf.refine(unary, prev_prob=None, flow=None)
            prev_prob = None
        else:
            refined_prob, prev_prob = crf.refine(unary, prev_prob=prev_prob, flow=None)

        if variant == "random_mask":
            mask = make_random_mask_like(refined_prob, seed=seed, t_index=t)
            strength = torch.ones_like(refined_prob)
        elif variant == "masked_blur":
            mask = refined_prob
            protected = apply_mask_guided_blur(frame, mask)
            protected_frames.append(protected.squeeze(0).cpu())
            masks.append(mask.squeeze(0).cpu())
            continue
        elif variant == "masked_mosaic":
            mask = refined_prob
            protected = apply_mask_guided_mosaic(frame, mask)
            protected_frames.append(protected.squeeze(0).cpu())
            masks.append(mask.squeeze(0).cpu())
            continue
        elif variant == "full_frame":
            mask = torch.ones_like(refined_prob)
            strength = torch.ones_like(refined_prob)
        else:
            mask = refined_prob
            if variant == "no_ncp":
                strength = torch.ones_like(refined_prob)
            else:
                strength = ncp.allocate(refined_prob)

        protected = injector.apply(frame, mask, strength, foreground_mask=None, t_index=t)
        protected_frames.append(protected.squeeze(0).cpu())
        masks.append(mask.squeeze(0).cpu())

    return torch.stack(protected_frames, dim=0), mean_temporal_iou(masks)


def select_eval_frame(frames: Tensor) -> Tensor:
    return frames[frames.size(0) // 2]


def tensor_to_uint8_image(frame: Tensor) -> np.ndarray:
    return frame.detach().cpu().clamp(0, 255).byte().numpy().transpose(1, 2, 0)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def plot_frontier(frontier_rows: List[dict], output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    metric_pairs = [("R@1_mean", "R@1"), ("R@5_mean", "R@5")]
    colors = {"ppedcrf": "#1f77b4", "full_frame": "#d62728"}

    for ax, (metric_key, metric_label) in zip(axes, metric_pairs):
        for variant in ("ppedcrf", "full_frame"):
            rows = [row for row in frontier_rows if row["variant"] == variant]
            rows = sorted(rows, key=lambda item: float(item["sigma"]))
            xs = [row["psnr_mean"] for row in rows]
            ys = [row[metric_key] for row in rows]
            ax.plot(xs, ys, marker="o", linewidth=2, label=VARIANT_LABELS[variant], color=colors[variant])
            for row, x, y in zip(rows, xs, ys):
                ax.annotate(f"$\\sigma={row['sigma']:.0f}$", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

        ax.set_xlabel("PSNR (dB)")
        ax.set_ylabel(f"{metric_label} retrieval accuracy")
        ax.set_title(f"Privacy-utility frontier ({metric_label})")
        ax.grid(alpha=0.25)

    axes[1].legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(robustness_rows: List[dict], output_path: str) -> None:
    backbones = sorted({row["backbone"] for row in robustness_rows})
    fig, axes = plt.subplots(1, len(backbones), figsize=(5.2 * len(backbones), 4.2), constrained_layout=True)
    if len(backbones) == 1:
        axes = [axes]

    colors = {"raw": "#7f7f7f", "ppedcrf": "#1f77b4"}
    for ax, backbone in zip(axes, backbones):
        for variant in ("raw", "ppedcrf"):
            rows = [row for row in robustness_rows if row["backbone"] == backbone and row["variant"] == variant]
            rows = sorted(rows, key=lambda item: int(item["gallery_size"]))
            xs = [int(row["gallery_size"]) for row in rows]
            ys = [row["R@1_mean"] for row in rows]
            yerr = [row["R@1_std"] for row in rows]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=2,
                capsize=3,
                label=VARIANT_LABELS[variant],
                color=colors[variant],
            )

        ax.set_title(f"Retrieval robustness ({backbone})")
        ax.set_xlabel("Gallery size")
        ax.set_ylabel("R@1 retrieval accuracy")
        ax.set_xticks(sorted({int(row['gallery_size']) for row in robustness_rows}))
        ax.grid(alpha=0.25)

    axes[-1].legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_gallery_tensor(
    gallery_frame_by_id: Dict[str, Tensor],
    query_ids: Sequence[str],
    distractor_ids: Sequence[str],
    gallery_size: int,
) -> Tuple[Tensor, List[str]]:
    assert gallery_size >= len(query_ids)
    chosen_ids = list(query_ids) + list(distractor_ids[: gallery_size - len(query_ids)])
    gallery_images = torch.stack([gallery_frame_by_id[clip_id] for clip_id in chosen_ids], dim=0)
    return gallery_images, chosen_ids


def aggregate_metric_dict(runs: List[Dict[str, float]]) -> Dict[str, float]:
    keys = list(runs[0].keys())
    output = {}
    for key in keys:
        values = [run[key] for run in runs]
        output[f"{key}_mean"] = float(np.mean(values))
        output[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return output


def discover_paired_locations(
    root: str,
    resize_hw: Tuple[int, int],
    num_queries: int,
    max_gallery: int,
    min_frames: int,
    pair_pool_size: int,
    device: torch.device,
) -> Tuple[List[dict], List[str], List[dict]]:
    """
    Build a harder benchmark by pairing different but visually similar sequences.

    Each selected pair contributes one gallery sequence and one query sequence that
    share a synthetic location label. Remaining unused sequences serve as gallery
    distractors.
    """
    ordered_ids = list(iter_clip_ids(root=root, min_frames=min_frames))
    if len(ordered_ids) < max(max_gallery + num_queries, pair_pool_size):
        pair_pool_size = len(ordered_ids)

    candidate_ids = ordered_ids[:pair_pool_size]
    candidate_dataset = MonitoringClipDataset(
        root=root,
        clip_ids=candidate_ids,
        view="full",
        clip_len=1,
        resize_hw=resize_hw,
        min_frames=min_frames,
    )

    candidate_frames = []
    candidate_clip_ids = []
    for sample in candidate_dataset:
        candidate_clip_ids.append(sample.clip_id)
        candidate_frames.append(select_eval_frame(sample.frames))

    rcfg = RetrievalConfig(
        backbone="resnet18",
        device=str(device),
        normalize=True,
        input_size=224,
        topk=(1, 5, 10),
    )
    embedder = make_default_embedder(rcfg).eval().to(device)
    candidate_tensor = torch.stack(candidate_frames, dim=0)
    candidate_emb = build_gallery_embeddings(rcfg, embedder, candidate_tensor)
    sim = candidate_emb @ candidate_emb.t()

    upper = torch.triu_indices(sim.size(0), sim.size(1), offset=1)
    pair_scores = sim[upper[0], upper[1]].cpu().numpy()
    sorted_pair_indices = np.argsort(pair_scores)[::-1]

    used: set[str] = set()
    pairs: List[dict] = []
    for pair_rank in sorted_pair_indices:
        i = int(upper[0, pair_rank].item())
        j = int(upper[1, pair_rank].item())
        gallery_clip_id = candidate_clip_ids[i]
        query_clip_id = candidate_clip_ids[j]
        if gallery_clip_id in used or query_clip_id in used:
            continue

        location_id = f"loc_{len(pairs):03d}"
        pairs.append(
            {
                "location_id": location_id,
                "gallery_clip_id": gallery_clip_id,
                "query_clip_id": query_clip_id,
                "pair_similarity": float(pair_scores[pair_rank]),
            }
        )
        used.add(gallery_clip_id)
        used.add(query_clip_id)
        if len(pairs) >= num_queries:
            break

    if len(pairs) < num_queries:
        raise RuntimeError(f"Unable to discover {num_queries} paired locations from {pair_pool_size} candidates.")

    distractor_needed = max(0, max_gallery - num_queries)
    clip_to_index = {clip_id: idx for idx, clip_id in enumerate(candidate_clip_ids)}
    used_indices = [clip_to_index[clip_id] for clip_id in used]
    hard_distractors = []
    for clip_id in candidate_clip_ids:
        if clip_id in used:
            continue
        clip_index = clip_to_index[clip_id]
        hardness = float(sim[clip_index, used_indices].max().item()) if used_indices else 0.0
        hard_distractors.append({"clip_id": clip_id, "hardness": hardness})
    hard_distractors.sort(key=lambda item: item["hardness"], reverse=True)
    selected_hard_distractors = hard_distractors[:distractor_needed]
    distractor_ids = [item["clip_id"] for item in selected_hard_distractors]

    if len(distractor_ids) < distractor_needed:
        existing = set(distractor_ids)
        remaining = [clip_id for clip_id in ordered_ids[pair_pool_size:] if clip_id not in used and clip_id not in existing]
        distractor_ids.extend(remaining[: distractor_needed - len(distractor_ids)])

    if len(distractor_ids) < distractor_needed:
        raise RuntimeError(
            f"Needed {distractor_needed} distractor sequences, but only found {len(distractor_ids)}."
        )

    return pairs, distractor_ids, selected_hard_distractors


def collect_external_image_paths(root: str) -> List[Path]:
    root_path = Path(root)
    if not root_path.is_dir():
        return []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths: List[Path] = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            image_paths.append(path)
    return sorted(image_paths)


def build_external_distractor_frames(
    image_paths: Sequence[Path],
    resize_hw: Tuple[int, int],
    max_items: int,
) -> Dict[str, Tensor]:
    frames: Dict[str, Tensor] = {}
    for idx, path in enumerate(image_paths[: max(0, int(max_items))]):
        try:
            frame = _read_image(str(path))
            frame = _resize_if_needed(frame, resize_hw)
        except Exception:
            continue
        frames[f"ext_{idx:04d}"] = frame
    return frames


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    ensure_dir(args.output_dir)
    paper_image_dir = os.path.join(PROJECT_ROOT, "paper", "figs")
    ensure_dir(paper_image_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resize_hw = (int(args.resize_h), int(args.resize_w))
    paired_locations, distractor_clip_ids, hard_distractor_meta = discover_paired_locations(
        root=args.monitoring_root,
        resize_hw=resize_hw,
        num_queries=args.num_queries,
        max_gallery=args.max_gallery,
        min_frames=args.min_frames,
        pair_pool_size=args.pair_pool_size,
        device=device,
    )
    pair_similarities = [float(item["pair_similarity"]) for item in paired_locations]
    distractor_hardness = [float(item["hardness"]) for item in hard_distractor_meta]
    paired_count = len(paired_locations)
    distractor_hardness_by_gallery = {}
    for gallery_size in sorted(args.gallery_sizes):
        num_distractors = max(0, int(gallery_size) - paired_count)
        if num_distractors == 0:
            continue
        selected = distractor_hardness[:num_distractors]
        distractor_hardness_by_gallery[str(gallery_size)] = {
            "count": num_distractors,
            "mean": float(np.mean(selected)),
            "min": float(np.min(selected)),
            "max": float(np.max(selected)),
        }
    query_ids = [item["location_id"] for item in paired_locations]
    query_clip_ids = [item["query_clip_id"] for item in paired_locations]
    gallery_positive_clip_ids = [item["gallery_clip_id"] for item in paired_locations]
    distractor_ids = [f"dist_{idx:03d}" for idx in range(len(distractor_clip_ids))]
    distractor_label_to_clip = {label: clip_id for label, clip_id in zip(distractor_ids, distractor_clip_ids)}

    coco_image_paths = collect_external_image_paths(args.coco_root)
    digica_image_paths = collect_external_image_paths(args.digica_root)
    external_paths = coco_image_paths + digica_image_paths
    external_frames_by_id = build_external_distractor_frames(
        image_paths=external_paths,
        resize_hw=resize_hw,
        max_items=args.max_external_distractors,
    )
    external_distractor_ids = list(external_frames_by_id.keys())

    selection_path = os.path.join(args.output_dir, "selection.json")
    save_json(
        selection_path,
        {
            "monitoring_root": args.monitoring_root,
            "paired_locations": paired_locations,
            "query_ids": query_ids,
            "query_clip_ids": query_clip_ids,
            "gallery_positive_clip_ids": gallery_positive_clip_ids,
            "distractor_ids": distractor_ids,
            "distractor_clip_ids": distractor_clip_ids,
            "external_sources": {
                "coco_root": args.coco_root,
                "coco_image_count": len(coco_image_paths),
                "digica_root": args.digica_root,
                "digica_image_count": len(digica_image_paths),
                "external_distractors_loaded": len(external_distractor_ids),
                "max_external_distractors": int(args.max_external_distractors),
            },
            "hard_distractors": hard_distractor_meta,
            "pair_similarity_stats": {
                "mean": float(np.mean(pair_similarities)),
                "min": float(np.min(pair_similarities)),
                "max": float(np.max(pair_similarities)),
            },
            "distractor_hardness_stats": {
                "mean": float(np.mean(distractor_hardness)),
                "min": float(np.min(distractor_hardness)),
                "max": float(np.max(distractor_hardness)),
            },
            "distractor_hardness_stats_by_gallery": distractor_hardness_by_gallery,
            "resize_hw": list(resize_hw),
            "clip_len": args.clip_len,
            "seeds": args.seeds,
            "backbones": args.backbones,
            "gallery_sizes": args.gallery_sizes,
            "frontier_sigmas": args.frontier_sigmas,
        },
    )

    query_dataset = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=query_clip_ids,
        view="query",
        clip_len=args.clip_len,
        resize_hw=resize_hw,
        min_frames=args.min_frames,
    )
    gallery_positive_dataset = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=gallery_positive_clip_ids,
        view="gallery",
        clip_len=args.clip_len,
        resize_hw=resize_hw,
        min_frames=args.min_frames,
    )
    distractor_dataset = MonitoringClipDataset(
        root=args.monitoring_root,
        clip_ids=distractor_clip_ids,
        view="gallery",
        clip_len=args.clip_len,
        resize_hw=resize_hw,
        min_frames=args.min_frames,
    )

    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
    sensnet.eval()

    query_clip_frames_by_actual_id: Dict[str, Tensor] = {}
    for sample in query_dataset:
        query_clip_frames_by_actual_id[sample.clip_id] = sample.frames

    gallery_positive_frames_by_actual_id: Dict[str, Tensor] = {}
    for sample in gallery_positive_dataset:
        gallery_positive_frames_by_actual_id[sample.clip_id] = select_eval_frame(sample.frames)

    distractor_frames_by_actual_id: Dict[str, Tensor] = {}
    for sample in distractor_dataset:
        distractor_frames_by_actual_id[sample.clip_id] = select_eval_frame(sample.frames)

    query_clips: Dict[str, Tensor] = {}
    raw_query_images: Dict[str, Tensor] = {}
    gallery_frames_by_id: Dict[str, Tensor] = {}
    for item in paired_locations:
        location_id = item["location_id"]
        query_clip_id = item["query_clip_id"]
        gallery_clip_id = item["gallery_clip_id"]
        query_frames = query_clip_frames_by_actual_id[query_clip_id]
        query_clips[location_id] = query_frames
        raw_query_images[location_id] = select_eval_frame(query_frames)
        gallery_frames_by_id[location_id] = gallery_positive_frames_by_actual_id[gallery_clip_id]

    for label, clip_id in distractor_label_to_clip.items():
        gallery_frames_by_id[label] = distractor_frames_by_actual_id[clip_id]

    for ext_id, ext_frame in external_frames_by_id.items():
        gallery_frames_by_id[ext_id] = ext_frame

    # Prefer monitoring hard negatives first, then use external pools to support larger galleries.
    distractor_ids = distractor_ids + external_distractor_ids

    protected_query_images: Dict[str, Dict[int, Tensor]] = {}
    quality_summary_rows: List[dict] = []
    variants = ["ppedcrf", "no_temporal", "no_ncp", "masked_blur", "masked_mosaic", "random_mask", "full_frame"]

    for variant in variants:
        protected_query_images[variant] = {}
        per_seed_psnr = []
        per_seed_ssim = []
        per_seed_mask_iou = []

        for seed in args.seeds:
            cfg_seed = clone_cfg(cfg)
            cfg_seed["ppedcrf"]["noise"] = dict(cfg_seed["ppedcrf"]["noise"])
            cfg_seed["ppedcrf"]["noise"]["seed"] = int(seed)

            protected_frames = []
            psnr_scores = []
            ssim_scores = []
            mask_ious = []

            for clip_id in query_ids:
                protected_clip, mask_iou = protect_clip_variant(
                    query_clips[clip_id],
                    sensnet,
                    cfg_seed,
                    device=device,
                    variant=variant,
                    seed=int(seed),
                )
                protected_frame = select_eval_frame(protected_clip)
                original_frame = raw_query_images[clip_id]

                psnr_scores.append(psnr_torch(original_frame, protected_frame))
                ssim_scores.append(
                    ssim_grayscale_np(
                        tensor_to_uint8_image(original_frame),
                        tensor_to_uint8_image(protected_frame),
                    )
                )
                mask_ious.append(mask_iou)
                protected_frames.append(protected_frame)

            protected_query_images[variant][int(seed)] = torch.stack(protected_frames, dim=0)
            per_seed_psnr.append(float(np.mean(psnr_scores)))
            per_seed_ssim.append(float(np.mean(ssim_scores)))
            per_seed_mask_iou.append(float(np.mean(mask_ious)))

        quality_summary_rows.append(
            {
                "variant": variant,
                "label": VARIANT_LABELS[variant],
                "psnr_mean": float(np.mean(per_seed_psnr)),
                "psnr_std": float(np.std(per_seed_psnr, ddof=1)) if len(per_seed_psnr) > 1 else 0.0,
                "ssim_mean": float(np.mean(per_seed_ssim)),
                "ssim_std": float(np.std(per_seed_ssim, ddof=1)) if len(per_seed_ssim) > 1 else 0.0,
                "mask_iou_mean": float(np.mean(per_seed_mask_iou)),
                "mask_iou_std": float(np.std(per_seed_mask_iou, ddof=1)) if len(per_seed_mask_iou) > 1 else 0.0,
            }
        )

    raw_query_tensor = torch.stack([raw_query_images[clip_id] for clip_id in query_ids], dim=0)

    ablation_rows: List[dict] = []
    robustness_rows: List[dict] = []
    frontier_rows: List[dict] = []

    default_gallery_size = max(args.gallery_sizes)
    default_backbone = args.backbones[0]

    for backbone in args.backbones:
        rcfg = RetrievalConfig(
            backbone=backbone,
            device=str(device),
            normalize=True,
            input_size=224,
            topk=(1, 5, 10),
        )
        embedder = make_default_embedder(rcfg).eval().to(device)

        for gallery_size in sorted(args.gallery_sizes):
            gallery_tensor, gallery_ids = build_gallery_tensor(
                gallery_frame_by_id=gallery_frames_by_id,
                query_ids=query_ids,
                distractor_ids=distractor_ids,
                gallery_size=gallery_size,
            )
            gallery_emb = build_gallery_embeddings(rcfg, embedder, gallery_tensor)

            raw_result = query_topk(rcfg, raw_query_tensor, list(query_ids), gallery_emb, gallery_ids, embedder)
            raw_metrics = {f"{key}_mean": value for key, value in raw_result.items()}
            raw_metrics.update({f"{key}_std": 0.0 for key in raw_result})
            raw_row = {
                "variant": "raw",
                "label": VARIANT_LABELS["raw"],
                "backbone": backbone,
                "gallery_size": gallery_size,
                **raw_metrics,
            }
            robustness_rows.append(raw_row)

            if backbone == default_backbone and gallery_size == default_gallery_size:
                ablation_rows.append(raw_row)

            for variant in variants:
                runs = []
                for seed in args.seeds:
                    query_tensor = protected_query_images[variant][int(seed)]
                    runs.append(query_topk(rcfg, query_tensor, list(query_ids), gallery_emb, gallery_ids, embedder))
                aggregated = aggregate_metric_dict(runs)
                row = {
                    "variant": variant,
                    "label": VARIANT_LABELS[variant],
                    "backbone": backbone,
                    "gallery_size": gallery_size,
                    **aggregated,
                }
                if variant == "ppedcrf":
                    robustness_rows.append(row)
                if backbone == default_backbone and gallery_size == default_gallery_size:
                    qrow = next(item for item in quality_summary_rows if item["variant"] == variant)
                    ablation_rows.append({**row, **qrow})

    for sigma in args.frontier_sigmas:
        for variant in ("ppedcrf", "full_frame"):
            runs = []
            psnr_runs = []
            ssim_runs = []

            for seed in args.seeds:
                cfg_sigma = clone_cfg(cfg)
                cfg_sigma["ppedcrf"]["noise"] = dict(cfg_sigma["ppedcrf"]["noise"])
                cfg_sigma["ppedcrf"]["noise"]["seed"] = int(seed)
                cfg_sigma["ppedcrf"]["noise"]["sigma"] = float(sigma)

                protected_frames = []
                psnr_scores = []
                ssim_scores = []
                for clip_id in query_ids:
                    protected_clip, _ = protect_clip_variant(
                        query_clips[clip_id],
                        sensnet,
                        cfg_sigma,
                        device=device,
                        variant=variant,
                        seed=int(seed),
                    )
                    protected_frame = select_eval_frame(protected_clip)
                    original_frame = raw_query_images[clip_id]
                    protected_frames.append(protected_frame)
                    psnr_scores.append(psnr_torch(original_frame, protected_frame))
                    ssim_scores.append(
                        ssim_grayscale_np(
                            tensor_to_uint8_image(original_frame),
                            tensor_to_uint8_image(protected_frame),
                        )
                    )

                protected_tensor = torch.stack(protected_frames, dim=0)
                rcfg = RetrievalConfig(
                    backbone=default_backbone,
                    device=str(device),
                    normalize=True,
                    input_size=224,
                    topk=(1, 5, 10),
                )
                embedder = make_default_embedder(rcfg).eval().to(device)
                gallery_tensor, gallery_ids = build_gallery_tensor(
                    gallery_frame_by_id=gallery_frames_by_id,
                    query_ids=query_ids,
                    distractor_ids=distractor_ids,
                    gallery_size=default_gallery_size,
                )
                gallery_emb = build_gallery_embeddings(rcfg, embedder, gallery_tensor)
                runs.append(query_topk(rcfg, protected_tensor, list(query_ids), gallery_emb, gallery_ids, embedder))
                psnr_runs.append(float(np.mean(psnr_scores)))
                ssim_runs.append(float(np.mean(ssim_scores)))

            frontier_rows.append(
                {
                    "variant": variant,
                    "label": VARIANT_LABELS[variant],
                    "sigma": float(sigma),
                    **aggregate_metric_dict(runs),
                    "psnr_mean": float(np.mean(psnr_runs)),
                    "psnr_std": float(np.std(psnr_runs, ddof=1)) if len(psnr_runs) > 1 else 0.0,
                    "ssim_mean": float(np.mean(ssim_runs)),
                    "ssim_std": float(np.std(ssim_runs, ddof=1)) if len(ssim_runs) > 1 else 0.0,
                }
            )

    # ---- Blur/mosaic parameter sweep (m7) ----
    # Sweep blur kernel sizes and mosaic block sizes to build comparable frontiers.
    baseline_sweep_rows: List[dict] = []
    for ks in args.blur_kernel_sizes:
        psnr_scores_bk = []
        ssim_scores_bk = []
        prot_frames_bk = []
        for clip_id in query_ids:
            clip = query_clips[clip_id]
            prot_clip_frames = []
            for t in range(clip.size(0)):
                frame = clip[t:t+1].to(device)
                unary = sensnet(frame)
                refined_prob, _ = DynamicCRF(
                    DynamicCRFConfig(
                        n_iters=int(cfg["ppedcrf"]["dynamic_crf"]["n_iters"]),
                        spatial_weight=float(cfg["ppedcrf"]["dynamic_crf"]["spatial_weight"]),
                        temporal_weight=float(cfg["ppedcrf"]["dynamic_crf"]["temporal_weight"]),
                        smooth_kernel=int(cfg["ppedcrf"]["dynamic_crf"]["smooth_kernel"]),
                    )
                ).refine(unary, prev_prob=None, flow=None)
                prot_f = apply_mask_guided_blur(frame, refined_prob, kernel_size=ks)
                prot_clip_frames.append(prot_f.squeeze(0).cpu())
            prot_clip_t = torch.stack(prot_clip_frames, dim=0)
            eval_frame = select_eval_frame(prot_clip_t)
            orig_frame = raw_query_images[clip_id]
            prot_frames_bk.append(eval_frame)
            psnr_scores_bk.append(psnr_torch(orig_frame, eval_frame))
            ssim_scores_bk.append(ssim_grayscale_np(
                tensor_to_uint8_image(orig_frame), tensor_to_uint8_image(eval_frame)))
        prot_tensor_bk = torch.stack(prot_frames_bk, dim=0)
        rcfg_bk = RetrievalConfig(
            backbone=default_backbone, device=str(device),
            normalize=True, input_size=224, topk=(1, 5, 10))
        emb_bk = make_default_embedder(rcfg_bk).eval().to(device)
        gt_bk, gids_bk = build_gallery_tensor(
            gallery_frame_by_id=gallery_frames_by_id,
            query_ids=query_ids, distractor_ids=distractor_ids,
            gallery_size=default_gallery_size)
        ge_bk = build_gallery_embeddings(rcfg_bk, emb_bk, gt_bk)
        retr_bk = query_topk(rcfg_bk, prot_tensor_bk, list(query_ids), ge_bk, gids_bk, emb_bk)
        row_bk = {
            "variant": "masked_blur",
            "label": f"blur (k={ks})",
            "param": ks,
            **retr_bk,
            "psnr_mean": float(np.mean(psnr_scores_bk)),
            "psnr_std": float(np.std(psnr_scores_bk, ddof=1)) if len(psnr_scores_bk) > 1 else 0.0,
            "ssim_mean": float(np.mean(ssim_scores_bk)),
            "ssim_std": float(np.std(ssim_scores_bk, ddof=1)) if len(ssim_scores_bk) > 1 else 0.0,
        }
        baseline_sweep_rows.append(row_bk)
        frontier_rows.append({
            "variant": "masked_blur",
            "label": f"blur (k={ks})",
            "sigma": float(ks),
            **retr_bk,
            "psnr_mean": row_bk["psnr_mean"],
            "psnr_std": row_bk["psnr_std"],
            "ssim_mean": row_bk["ssim_mean"],
            "ssim_std": row_bk["ssim_std"],
        })

    for bs in args.mosaic_block_sizes:
        psnr_scores_ms = []
        ssim_scores_ms = []
        prot_frames_ms = []
        for clip_id in query_ids:
            clip = query_clips[clip_id]
            prot_clip_frames = []
            for t in range(clip.size(0)):
                frame = clip[t:t+1].to(device)
                unary = sensnet(frame)
                refined_prob, _ = DynamicCRF(
                    DynamicCRFConfig(
                        n_iters=int(cfg["ppedcrf"]["dynamic_crf"]["n_iters"]),
                        spatial_weight=float(cfg["ppedcrf"]["dynamic_crf"]["spatial_weight"]),
                        temporal_weight=float(cfg["ppedcrf"]["dynamic_crf"]["temporal_weight"]),
                        smooth_kernel=int(cfg["ppedcrf"]["dynamic_crf"]["smooth_kernel"]),
                    )
                ).refine(unary, prev_prob=None, flow=None)
                prot_f = apply_mask_guided_mosaic(frame, refined_prob, block_size=bs)
                prot_clip_frames.append(prot_f.squeeze(0).cpu())
            prot_clip_t = torch.stack(prot_clip_frames, dim=0)
            eval_frame = select_eval_frame(prot_clip_t)
            orig_frame = raw_query_images[clip_id]
            prot_frames_ms.append(eval_frame)
            psnr_scores_ms.append(psnr_torch(orig_frame, eval_frame))
            ssim_scores_ms.append(ssim_grayscale_np(
                tensor_to_uint8_image(orig_frame), tensor_to_uint8_image(eval_frame)))
        prot_tensor_ms = torch.stack(prot_frames_ms, dim=0)
        rcfg_ms = RetrievalConfig(
            backbone=default_backbone, device=str(device),
            normalize=True, input_size=224, topk=(1, 5, 10))
        emb_ms = make_default_embedder(rcfg_ms).eval().to(device)
        gt_ms, gids_ms = build_gallery_tensor(
            gallery_frame_by_id=gallery_frames_by_id,
            query_ids=query_ids, distractor_ids=distractor_ids,
            gallery_size=default_gallery_size)
        ge_ms = build_gallery_embeddings(rcfg_ms, emb_ms, gt_ms)
        retr_ms = query_topk(rcfg_ms, prot_tensor_ms, list(query_ids), ge_ms, gids_ms, emb_ms)
        row_ms = {
            "variant": "masked_mosaic",
            "label": f"mosaic (b={bs})",
            "param": bs,
            **retr_ms,
            "psnr_mean": float(np.mean(psnr_scores_ms)),
            "psnr_std": float(np.std(psnr_scores_ms, ddof=1)) if len(psnr_scores_ms) > 1 else 0.0,
            "ssim_mean": float(np.mean(ssim_scores_ms)),
            "ssim_std": float(np.std(ssim_scores_ms, ddof=1)) if len(ssim_scores_ms) > 1 else 0.0,
        }
        baseline_sweep_rows.append(row_ms)
        frontier_rows.append({
            "variant": "masked_mosaic",
            "label": f"mosaic (b={bs})",
            "sigma": float(bs),
            **retr_ms,
            "psnr_mean": row_ms["psnr_mean"],
            "psnr_std": row_ms["psnr_std"],
            "ssim_mean": row_ms["ssim_mean"],
            "ssim_std": row_ms["ssim_std"],
        })

    # ---- Matched-operating-point comparison ----
    # For each PSNR target, find the sigma that brings PPEDCRF and global Gaussian
    # noise closest, then compare their retrieval accuracy at that operating point.
    matched_rows: List[dict] = []
    matched_variants = ["ppedcrf", "full_frame", "masked_blur", "masked_mosaic"]
    search_sigmas = np.arange(2.0, 50.0, 2.0).tolist()
    for target_psnr in args.matched_psnr_targets:
        for variant in matched_variants:
            # Deterministic baselines: blur/mosaic have fixed PSNR regardless of sigma
            if variant in ("masked_blur", "masked_mosaic"):
                # Use default sigma; PSNR is fixed by the method
                best_sigma = float(cfg["ppedcrf"]["noise"]["sigma"])
                best_psnr = None
                for qrow in quality_summary_rows:
                    if qrow["variant"] == variant:
                        best_psnr = qrow["psnr_mean"]
                        break
                if best_psnr is None:
                    continue
                # Find cached retrieval result
                for arow in ablation_rows:
                    if arow["variant"] == variant:
                        matched_rows.append({
                            "target_psnr": target_psnr,
                            "variant": variant,
                            "label": VARIANT_LABELS[variant],
                            "actual_sigma": best_sigma,
                            "actual_psnr": best_psnr,
                            "R@1_mean": arow.get("R@1_mean", float("nan")),
                            "R@1_std": arow.get("R@1_std", 0.0),
                        })
                        break
                continue

            best_sigma = None
            best_psnr_diff = float("inf")
            best_result = None
            for sigma in search_sigmas:
                psnr_runs_s = []
                retrieval_runs_s = []
                for seed in args.seeds:
                    cfg_s = clone_cfg(cfg)
                    cfg_s["ppedcrf"]["noise"] = dict(cfg_s["ppedcrf"]["noise"])
                    cfg_s["ppedcrf"]["noise"]["seed"] = int(seed)
                    cfg_s["ppedcrf"]["noise"]["sigma"] = float(sigma)
                    psnr_scores_s = []
                    prot_frames_s = []
                    for clip_id in query_ids:
                        prot_clip, _ = protect_clip_variant(
                            query_clips[clip_id], sensnet, cfg_s,
                            device=device, variant=variant, seed=int(seed),
                        )
                        prot_frame = select_eval_frame(prot_clip)
                        orig_frame = raw_query_images[clip_id]
                        psnr_scores_s.append(psnr_torch(orig_frame, prot_frame))
                        prot_frames_s.append(prot_frame)
                    psnr_runs_s.append(float(np.mean(psnr_scores_s)))
                    prot_tensor = torch.stack(prot_frames_s, dim=0)
                    rcfg_m = RetrievalConfig(
                        backbone=default_backbone, device=str(device),
                        normalize=True, input_size=224, topk=(1, 5, 10),
                    )
                    emb_m = make_default_embedder(rcfg_m).eval().to(device)
                    gt, gids = build_gallery_tensor(
                        gallery_frame_by_id=gallery_frames_by_id,
                        query_ids=query_ids, distractor_ids=distractor_ids,
                        gallery_size=default_gallery_size,
                    )
                    ge = build_gallery_embeddings(rcfg_m, emb_m, gt)
                    retrieval_runs_s.append(query_topk(rcfg_m, prot_tensor, list(query_ids), ge, gids, emb_m))

                mean_psnr = float(np.mean(psnr_runs_s))
                diff = abs(mean_psnr - target_psnr)
                if diff < best_psnr_diff:
                    best_psnr_diff = diff
                    best_sigma = sigma
                    agg = aggregate_metric_dict(retrieval_runs_s)
                    best_result = {"actual_psnr": mean_psnr, **agg}

            if best_result is not None:
                matched_rows.append({
                    "target_psnr": target_psnr,
                    "variant": variant,
                    "label": VARIANT_LABELS[variant],
                    "actual_sigma": best_sigma,
                    "actual_psnr": best_result["actual_psnr"],
                    "R@1_mean": best_result["R@1_mean"],
                    "R@1_std": best_result["R@1_std"],
                })

    # ---- Temporal consistency metrics ----
    temporal_rows: List[dict] = []
    for variant in variants:
        per_seed_flicker = []
        per_seed_stability = []
        for seed in args.seeds:
            cfg_t = clone_cfg(cfg)
            cfg_t["ppedcrf"]["noise"] = dict(cfg_t["ppedcrf"]["noise"])
            cfg_t["ppedcrf"]["noise"]["seed"] = int(seed)
            clip_flickers = []
            clip_stabilities = []
            for clip_id in query_ids:
                orig_clip = query_clips[clip_id]
                prot_clip, _ = protect_clip_variant(
                    orig_clip, sensnet, cfg_t,
                    device=device, variant=variant, seed=int(seed),
                )
                clip_flickers.append(flicker_score(prot_clip))
                clip_stabilities.append(perturbation_stability(orig_clip, prot_clip))
            per_seed_flicker.append(float(np.mean(clip_flickers)))
            per_seed_stability.append(float(np.mean(clip_stabilities)))

        temporal_rows.append({
            "variant": variant,
            "label": VARIANT_LABELS[variant],
            "flicker_mean": float(np.mean(per_seed_flicker)),
            "flicker_std": float(np.std(per_seed_flicker, ddof=1)) if len(per_seed_flicker) > 1 else 0.0,
            "perturbation_stability_mean": float(np.mean(per_seed_stability)),
            "perturbation_stability_std": float(np.std(per_seed_stability, ddof=1)) if len(per_seed_stability) > 1 else 0.0,
        })

    save_csv(
        os.path.join(args.output_dir, "quality_summary.csv"),
        quality_summary_rows,
        fieldnames=[
            "variant",
            "label",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
            "mask_iou_mean",
            "mask_iou_std",
        ],
    )
    save_csv(
        os.path.join(args.output_dir, "ablation_summary.csv"),
        ablation_rows,
        fieldnames=[
            "variant",
            "label",
            "backbone",
            "gallery_size",
            "R@1_mean",
            "R@1_std",
            "R@5_mean",
            "R@5_std",
            "R@10_mean",
            "R@10_std",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
            "mask_iou_mean",
            "mask_iou_std",
        ],
    )
    save_csv(
        os.path.join(args.output_dir, "robustness_summary.csv"),
        robustness_rows,
        fieldnames=[
            "variant",
            "label",
            "backbone",
            "gallery_size",
            "R@1_mean",
            "R@1_std",
            "R@5_mean",
            "R@5_std",
            "R@10_mean",
            "R@10_std",
        ],
    )
    save_csv(
        os.path.join(args.output_dir, "frontier_summary.csv"),
        frontier_rows,
        fieldnames=[
            "variant",
            "label",
            "sigma",
            "R@1_mean",
            "R@1_std",
            "R@5_mean",
            "R@5_std",
            "R@10_mean",
            "R@10_std",
            "psnr_mean",
            "psnr_std",
            "ssim_mean",
            "ssim_std",
        ],
    )

    # Save new experiment outputs
    if matched_rows:
        save_csv(
            os.path.join(args.output_dir, "matched_operating_point.csv"),
            matched_rows,
            fieldnames=[
                "target_psnr", "variant", "label", "actual_sigma",
                "actual_psnr", "R@1_mean", "R@1_std",
            ],
        )
    if temporal_rows:
        save_csv(
            os.path.join(args.output_dir, "temporal_consistency.csv"),
            temporal_rows,
            fieldnames=[
                "variant", "label",
                "flicker_mean", "flicker_std",
                "perturbation_stability_mean", "perturbation_stability_std",
            ],
        )
    if baseline_sweep_rows:
        save_csv(
            os.path.join(args.output_dir, "baseline_sweep.csv"),
            baseline_sweep_rows,
            fieldnames=[
                "variant", "label", "param",
                "R@1_mean", "R@1_std", "R@5_mean", "R@5_std",
                "R@10_mean", "R@10_std",
                "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
            ],
        )

    plot_frontier(frontier_rows, os.path.join(paper_image_dir, "privacy_utility_tradeoff.pdf"))
    plot_robustness(robustness_rows, os.path.join(paper_image_dir, "retrieval_robustness_topk.pdf"))

    report_path = os.path.join(args.output_dir, "summary.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Controlled Retrieval Benchmark Summary\n\n")
        handle.write(f"- Monitoring root: `{args.monitoring_root}`\n")
        handle.write(f"- Query sequences: {len(query_ids)}\n")
        handle.write(f"- Gallery sizes: {args.gallery_sizes}\n")
        handle.write(f"- Backbones: {args.backbones}\n")
        handle.write(f"- Seeds: {args.seeds}\n")
        handle.write(f"- Resize: {resize_hw}\n")
        handle.write(
            f"- External pools: coco={len(coco_image_paths)} images, "
            f"digica={len(digica_image_paths)} images, "
            f"loaded external distractors={len(external_distractor_ids)}\n"
        )
        handle.write(
            "- Pair similarity (ResNet18 cosine): "
            f"mean={np.mean(pair_similarities):.4f}, min={np.min(pair_similarities):.4f}, max={np.max(pair_similarities):.4f}\n"
        )
        handle.write(
            "- Hard distractor max-similarity to paired locations: "
            f"mean={np.mean(distractor_hardness):.4f}, min={np.min(distractor_hardness):.4f}, max={np.max(distractor_hardness):.4f}\n"
        )
        for gallery_size in sorted(args.gallery_sizes):
            num_distractors = max(0, int(gallery_size) - len(query_ids))
            if num_distractors == 0:
                continue
            selected = distractor_hardness[:num_distractors]
            handle.write(
                f"- Gallery {gallery_size}: first {num_distractors} distractors have "
                f"mean max-similarity {np.mean(selected):.4f}, min {np.min(selected):.4f}, max {np.max(selected):.4f}\n"
            )
        handle.write("\n## Default ablation setting\n\n")
        handle.write(f"- Backbone: `{default_backbone}`\n")
        handle.write(f"- Gallery size: `{default_gallery_size}`\n")
        handle.write("\nThe detailed numeric outputs are stored in the CSV files in this directory.\n")

    print(f"Saved benchmark outputs to: {args.output_dir}")
    print(f"Saved figure: {os.path.join(paper_image_dir, 'privacy_utility_tradeoff.pdf')}")
    print(f"Saved figure: {os.path.join(paper_image_dir, 'retrieval_robustness_topk.pdf')}")


if __name__ == "__main__":
    main()
