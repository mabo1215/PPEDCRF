"""Run the ICME 2027 energy-preserving spatial intervention experiment.

The experiment tests whether the spatial placement of the PPEDCRF effective
weight changes retrieval beyond the total perturbation energy. It compares
the released support against a same-energy uniform map, a deterministic
spatial roll, and a deterministic flattened permutation. The smoke mode uses
synthetic tensors and a download-free embedder; neither smoke output nor the
monitoring proxy is geographic ground-truth evidence.
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

from eval.metrics import psnr_torch, ssim_grayscale_np
from eval.retrieval_attack import (
    RetrievalConfig,
    build_gallery_embeddings,
    default_input_size_for_backbone,
    make_default_embedder,
)
from main import load_sensnet_checkpoint
from models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from privacy.NCP import NCPAllocator, NCPConfig
from privacy.noise_injector import NoiseConfig, NoiseInjector
from run_controlled_retrieval_benchmark import (
    build_gallery_tensor,
    select_eval_frame,
    tensor_to_uint8_image,
)
from run_tomm_review_proxy import (
    VARIANT_LABELS,
    _build_proxy_data,
    detailed_retrieval,
)
from utils.config import load_yaml


CONTROL_LABELS = {
    "full": "learned support",
    "uniform_energy": "uniform same-energy support",
    "rolled_energy": "rolled same-energy support",
    "permuted_energy": "permuted same-energy support",
}
CONTROL_VARIANTS = tuple(CONTROL_LABELS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICME 2027 energy-preserving spatial intervention benchmark."
    )
    parser.add_argument("--mode", choices=("smoke", "proxy"), default="smoke")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--monitoring_root", default=r"F:workdatasetsmonitoringimages")
    parser.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    parser.add_argument("--output_dir", default="src/outputs/icme2027_mask_intervention")
    parser.add_argument("--num_queries", type=int, default=12)
    parser.add_argument("--pair_pool_size", type=int, default=240)
    parser.add_argument("--max_gallery", type=int, default=48)
    parser.add_argument("--gallery_sizes", type=int, nargs="+", default=[48])
    parser.add_argument("--clip_len", type=int, default=4)
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--min_frames", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    parser.add_argument("--backbones", nargs="+", default=["resnet18"])
    parser.add_argument("--coco_root", default="")
    parser.add_argument("--digica_root", default="")
    parser.add_argument("--max_external_distractors", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--energy_tolerance", type=float, default=1e-6)
    parser.add_argument("--smoke_queries", type=int, default=3)
    parser.add_argument("--smoke_size", type=int, default=64)
    parser.add_argument("--smoke_clip_len", type=int, default=4)
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


def build_modules(
    cfg: Mapping[str, object], device: torch.device, seed: int
) -> Tuple[DynamicCRF, NCPAllocator, NoiseInjector]:
    dcfg = cfg["ppedcrf"]["dynamic_crf"]  # type: ignore[index]
    ncfg = cfg["ppedcrf"]["ncp"]  # type: ignore[index]
    pcfg = cfg["ppedcrf"]["noise"]  # type: ignore[index]
    crf = DynamicCRF(
        DynamicCRFConfig(
            n_iters=int(dcfg["n_iters"]),
            spatial_weight=float(dcfg["spatial_weight"]),
            temporal_weight=float(dcfg["temporal_weight"]),
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
    return crf, ncp, injector


def intervention_weight(
    effective: torch.Tensor, control: str, seed: int, t_index: int
) -> torch.Tensor:
    """Create a control map with the same sum of squared effective weights."""
    if control == "full":
        return effective
    if control == "uniform_energy":
        amplitude = torch.sqrt(torch.mean(effective.square()).clamp_min(0.0))
        return torch.full_like(effective, amplitude)
    if control == "rolled_energy":
        height, width = effective.shape[-2:]
        shift_y = int(seed + 17 * (t_index + 1)) % max(1, height)
        shift_x = int(3 * seed + 31 * (t_index + 1)) % max(1, width)
        return torch.roll(effective, shifts=(shift_y, shift_x), dims=(-2, -1))
    if control == "permuted_energy":
        flat = effective.detach().cpu().reshape(-1)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1009 * (t_index + 1))
        permutation = torch.randperm(flat.numel(), generator=generator)
        return flat[permutation].to(effective.device).reshape_as(effective)
    raise ValueError(f"Unknown intervention control: {control}")


@torch.no_grad()
def protect_intervention_clip(
    frames: torch.Tensor,
    sensnet: nn.Module,
    cfg: Mapping[str, object],
    device: torch.device,
    control: str,
    seed: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Protect a clip and report intervention and clipping diagnostics."""
    if control not in CONTROL_VARIANTS:
        raise ValueError(f"Unknown intervention control: {control}")
    crf, ncp, injector = build_modules(cfg, device, seed)
    protected_frames: List[torch.Tensor] = []
    effective_maps: List[torch.Tensor] = []
    original = frames.detach().float().cpu()
    previous: torch.Tensor | None = None
    for t_index in range(frames.size(0)):
        frame = frames[t_index : t_index + 1].to(device)
        unary = sensnet(frame)
        refined, previous_next = crf.refine(unary, prev_prob=previous, flow=None)
        strength = ncp.allocate(refined)
        effective = (refined * strength).clamp_min(0.0)
        selected = intervention_weight(effective, control, seed, t_index)
        if control == "full":
            protected = injector.apply(frame, refined, strength, t_index=t_index)
        else:
            protected = injector.apply(
                frame,
                torch.ones_like(selected),
                selected,
                t_index=t_index,
            )
        protected_frames.append(protected.squeeze(0).cpu())
        effective_maps.append(selected.squeeze(0).cpu())
        previous = previous_next

    protected_clip = torch.stack(protected_frames, dim=0)
    effective_stack = torch.stack(effective_maps, dim=0).float()
    delta = protected_clip.float() - original
    range_min = float(cfg["ppedcrf"]["noise"]["clamp_min"])  # type: ignore[index]
    range_max = float(cfg["ppedcrf"]["noise"]["clamp_max"])  # type: ignore[index]
    clipped = ((protected_clip <= range_min + 1e-6) | (protected_clip >= range_max - 1e-6)).float()
    full_mse = float(delta.square().mean().item())
    effective_energy = float(effective_stack.square().mean().item())
    weighted_mse = float((delta.square() * effective_stack).mean().item())
    quality = {
        "psnr_mean": float(
            np.mean([psnr_torch(original[i], protected_clip[i]) for i in range(original.size(0))])
        ),
        "ssim_mean": float(
            np.mean(
                [
                    ssim_grayscale_np(
                        tensor_to_uint8_image(original[i]),
                        tensor_to_uint8_image(protected_clip[i]),
                    )
                    for i in range(original.size(0))
                ]
            )
        ),
        "effective_mse": full_mse,
        "effective_weight_energy": effective_energy,
        "effective_weight_mean": float(effective_stack.mean().item()),
        "weighted_delta_mse": weighted_mse,
        "support_coverage": float((effective_stack > 1e-6).float().mean().item()),
        "clipping_fraction": float(clipped.mean().item()),
        "map_mean": float(effective_stack.mean().item()),
        "map_std": float(effective_stack.std(unbiased=False).item()),
    }
    return protected_clip, quality


def aggregate_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, int], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["control"]), str(row["backbone"]), int(row["gallery_size"]))
        ].append(row)
    numeric = (
        "correct_rank",
        "retrieval_margin",
        "top5_hit",
        "top10_hit",
        "psnr_mean",
        "ssim_mean",
        "effective_mse",
        "effective_weight_energy",
        "weighted_delta_mse",
        "support_coverage",
        "clipping_fraction",
        "energy_relative_error",
    )
    output: List[Dict[str, object]] = []
    for (control, backbone, gallery_size), members in sorted(groups.items()):
        row: Dict[str, object] = {
            "control": control,
            "label": CONTROL_LABELS.get(control, control),
            "backbone": backbone,
            "gallery_size": gallery_size,
            "num_rows": len(members),
            "top1_mean": float(np.mean([int(float(item["correct_rank"])) == 1 for item in members])),
        }
        for name in numeric:
            values = np.asarray(
                [float(item[name]) for item in members if item.get(name) is not None],
                dtype=np.float64,
            )
            row[f"{name}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(row)
    return output


def run_smoke(args: argparse.Namespace) -> Path:
    from run_train import SensitiveRegionNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).with_name("icme2027_mask_intervention_smoke")
    ensure_dir(output_dir)
    torch.manual_seed(20270903)
    query_count = int(args.smoke_queries)
    size = int(args.smoke_size)
    clip_len = int(args.smoke_clip_len)
    query_clips = torch.rand(query_count, clip_len, 3, size, size) * 255.0
    gallery = torch.rand(query_count + 2, 3, size, size) * 255.0
    query_ids = [f"loc_{index:03d}" for index in range(query_count)]
    gallery_ids = list(query_ids) + ["negative_0", "negative_1"]
    model = SensitiveRegionNet().to(device).eval()
    embedder = TinyEmbedder().to(device).eval()
    cfg = load_yaml(args.config)
    seed = int(args.seeds[0])
    protected: Dict[str, torch.Tensor] = {}
    quality: Dict[str, Dict[str, float]] = {}
    for control in CONTROL_VARIANTS:
        clips: List[torch.Tensor] = []
        for index, query_id in enumerate(query_ids):
            clip, metrics = protect_intervention_clip(
                query_clips[index], model, cfg, device, control, seed
            )
            clips.append(select_eval_frame(clip))
            quality[query_id + "::" + control] = metrics
        protected[control] = torch.stack(clips)

    rows: List[Dict[str, object]] = []
    for control in CONTROL_VARIANTS:
        qrows = detailed_retrieval(
            protected[control],
            query_ids,
            gallery,
            gallery_ids,
            embedder,
            device,
            input_size=size,
            quality_by_query={
                query_id: quality[query_id + "::" + control]
                for query_id in query_ids
            },
        )
        for qrow in qrows:
            qrow.update(
                {
                    "control": control,
                    "label": CONTROL_LABELS[control],
                    "seed": seed,
                    "backbone": "tiny",
                    "gallery_size": len(gallery_ids),
                    "energy_relative_error": 0.0,
                }
            )
        rows.extend(qrows)
    summary = aggregate_rows(rows)
    energy_ok = check_energy_consistency(rows, args.energy_tolerance)
    if len(rows) != query_count * len(CONTROL_VARIANTS) or not energy_ok:
        raise RuntimeError("ICME-M2 smoke gate failed.")
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(
        output_dir / "selection.json",
        {
            "review_cycle": "ICME-2027",
            "mode": "smoke",
            "controls": list(CONTROL_VARIANTS),
            "query_count": query_count,
            "clip_len": clip_len,
            "energy_tolerance": float(args.energy_tolerance),
            "scientific_evidence": False,
        },
    )
    write_json(
        output_dir / "run_metadata.json",
        {
            "device": str(device),
            "seed": seed,
            "rows": len(rows),
            "intervention_gate_passed": True,
            "scientific_evidence": False,
        },
    )
    print(f"[smoke] ICME-M2 gate passed on {device}; wrote {len(rows)} rows to {output_dir}")
    return output_dir


def check_energy_consistency(
    rows: Sequence[Mapping[str, object]], tolerance: float
) -> bool:
    by_query: Dict[str, float] = {}
    for row in rows:
        if "effective_weight_energy" not in row:
            continue
        query_id = str(row["query_id"])
        control = str(row["control"])
        energy = float(row["effective_weight_energy"])
        if control == "full":
            by_query[query_id] = energy
    if not by_query:
        return False
    for row in rows:
        if "effective_weight_energy" not in row:
            continue
        reference = by_query.get(str(row["query_id"]))
        if reference is None:
            return False
        relative_error = abs(float(row["effective_weight_energy"]) - reference) / max(
            abs(reference), 1e-12
        )
        if relative_error > float(tolerance):
            return False
    return True


def run_proxy(args: argparse.Namespace) -> Path:
    cfg = load_yaml(args.config)
    if args.sigma is not None:
        cfg = json.loads(json.dumps(cfg))
        cfg["ppedcrf"]["noise"]["sigma"] = float(args.sigma)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    pairs, hard_meta, query_ids, query_clips, raw_queries, gallery_by_id, distractors = _build_proxy_data(
        args, device
    )
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
    protected: Dict[Tuple[str, int], torch.Tensor] = {}
    quality: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = {}
    for control in CONTROL_VARIANTS:
        for seed in args.seeds:
            images: List[torch.Tensor] = []
            qquality: Dict[str, Dict[str, float]] = {}
            for query_id in query_ids:
                clip, metrics = protect_intervention_clip(
                    query_clips[query_id], sensnet, cfg, device, control, int(seed)
                )
                images.append(select_eval_frame(clip))
                qquality[query_id] = metrics
            protected[(control, int(seed))] = torch.stack(images)
            quality[(control, int(seed))] = qquality

    rows: List[Dict[str, object]] = []
    raw_tensor = torch.stack([raw_queries[query_id] for query_id in query_ids])
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
                gallery_by_id, query_ids, distractors, int(gallery_size)
            )
            gallery_embeddings = build_gallery_embeddings(rcfg, embedder, gallery_tensor)
            raw_rows = detailed_retrieval(
                raw_tensor, query_ids, gallery_tensor, gallery_ids,
                embedder, device, rcfg.input_size
            )
            for row in raw_rows:
                row.update(
                    {
                        "control": "raw",
                        "label": "raw query",
                        "seed": "raw",
                        "backbone": backbone,
                        "gallery_size": int(gallery_size),
                        "energy_relative_error": 0.0,
                    }
                )
            rows.extend(raw_rows)
            for control in CONTROL_VARIANTS:
                for seed in args.seeds:
                    qrows = detailed_retrieval(
                        protected[(control, int(seed))],
                        query_ids,
                        gallery_tensor,
                        gallery_ids,
                        embedder,
                        device,
                        rcfg.input_size,
                        quality_by_query=quality[(control, int(seed))],
                    )
                    for row in qrows:
                        reference = float(
                            quality[("full", int(seed))][str(row["query_id"])]
                            ["effective_weight_energy"]
                        )
                        current = float(row["effective_weight_energy"])
                        row.update(
                            {
                                "control": control,
                                "label": CONTROL_LABELS[control],
                                "seed": int(seed),
                                "backbone": backbone,
                                "gallery_size": int(gallery_size),
                                "energy_relative_error": abs(current - reference)
                                / max(abs(reference), 1e-12),
                            }
                        )
                    rows.extend(qrows)

    energy_ok = check_energy_consistency(rows, args.energy_tolerance)
    finite_ok = all(
        math.isfinite(float(row["retrieval_margin"]))
        and math.isfinite(float(row["energy_relative_error"]))
        and (
            str(row["control"]) == "raw"
            or math.isfinite(float(row["effective_weight_energy"]))
        )
        for row in rows
    )
    gate_passed = bool(energy_ok and finite_ok)
    summary = aggregate_rows(rows)
    write_csv(output_dir / "per_query.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_json(
        output_dir / "selection.json",
        {
            "review_cycle": "ICME-2027",
            "mode": "proxy",
            "benchmark": "controlled paired-scene proxy",
            "pairs": pairs,
            "hard_distractors": hard_meta,
            "query_count": len(query_ids),
            "gallery_sizes": args.gallery_sizes,
            "seeds": args.seeds,
            "backbones": args.backbones,
            "controls": list(CONTROL_VARIANTS),
            "energy_tolerance": float(args.energy_tolerance),
            "intervention_gate_passed": gate_passed,
            "scientific_evidence": False,
        },
    )
    write_json(
        output_dir / "run_metadata.json",
        {
            "review_cycle": "ICME-2027",
            "mode": "proxy",
            "device": str(device),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "backbones": args.backbones,
            "seeds": args.seeds,
            "sigma": float(cfg["ppedcrf"]["noise"]["sigma"]),
            "controls": list(CONTROL_VARIANTS),
            "intervention_gate_passed": gate_passed,
            "scientific_evidence": False,
            "note": "Proxy output is mechanism evidence only, not geographic ground truth.",
        },
    )
    if not gate_passed:
        raise RuntimeError("ICME-M2 intervention gate failed.")
    print(f"[proxy] ICME-M2 gate passed on {device}; wrote {len(rows)} rows to {output_dir}")
    return output_dir


class TinyEmbedder(nn.Module):
    """Download-free embedder used only by the smoke test."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(images, (4, 4)).flatten(1)


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_proxy(args)


if __name__ == "__main__":
    main()
