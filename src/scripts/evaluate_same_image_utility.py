"""Evaluate detector mAP and segmentation mIoU on the same sanitized images.

The manifest format is intentionally simple and dataset-neutral. Each JSONL
record contains an image and optional labels:

    {"image_id": "x", "image_path": "...",
     "boxes": [[x0, y0, x1, y1]], "labels": [1],
     "segmentation_path": "masks/x.png"}

This track is utility evidence only. It must not be used as geographic
ground-truth evidence for the VPR benchmark.
"""

from __future__ import annotations

import argparse
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
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from datasets.driving_clip_dataset import _read_image, _resize_if_needed  # noqa: E402
from main import load_sensnet_checkpoint  # noqa: E402
from run_tomm_review_proxy import protect_review_clip  # noqa: E402
from utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-image detector/segmenter utility evaluation.")
    parser.add_argument("--mode", choices=("smoke", "manifest"), default="smoke")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--root", default="")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    parser.add_argument("--output_dir", default="src/outputs/tomm_same_image_utility")
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--variants", nargs="+", default=["full", "global_noise"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--smoke_size", type=int, default=64)
    return parser.parse_args()


def resolve_path(value: str, root: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path(root) / path if root else path).resolve()


def load_manifest(path: str, root: str) -> List[dict]:
    records: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "image_id" not in record or "image_path" not in record:
                raise ValueError(f"Line {line_number} requires image_id and image_path.")
            image_path = resolve_path(str(record["image_path"]), root)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            boxes = record.get("boxes", [])
            labels = record.get("labels", [])
            if len(boxes) != len(labels):
                raise ValueError(f"Line {line_number} has mismatched boxes and labels.")
            normalized = dict(record)
            normalized["image_path"] = str(image_path)
            if record.get("segmentation_path"):
                mask_path = resolve_path(str(record["segmentation_path"]), root)
                if not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                normalized["segmentation_path"] = str(mask_path)
            records.append(normalized)
    if not records:
        raise ValueError("Utility manifest is empty.")
    return records


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-12)


def average_precision_detections(
    predictions: Sequence[Mapping[str, object]],
    targets: Sequence[Mapping[str, object]],
    iou_threshold: float = 0.5,
) -> float:
    """Compute class-averaged AP at a fixed IoU threshold."""
    classes = sorted(
        set(int(label) for target in targets for label in target.get("labels", []))
        | set(int(label) for prediction in predictions for label in prediction.get("labels", []))
    )
    if not classes:
        return 0.0
    aps: List[float] = []
    for class_id in classes:
        gt_by_image: Dict[int, List[Sequence[float]]] = {}
        total_gt = 0
        for image_index, target in enumerate(targets):
            boxes = target.get("boxes", [])
            labels = target.get("labels", [])
            gt = [box for box, label in zip(boxes, labels) if int(label) == class_id]
            gt_by_image[image_index] = gt
            total_gt += len(gt)
        scored: List[Tuple[float, int, Sequence[float]]] = []
        for image_index, prediction in enumerate(predictions):
            for box, label, score in zip(
                prediction.get("boxes", []),
                prediction.get("labels", []),
                prediction.get("scores", []),
            ):
                if int(label) == class_id:
                    scored.append((float(score), image_index, box))
        scored.sort(key=lambda item: item[0], reverse=True)
        matched: Dict[int, set[int]] = {index: set() for index in gt_by_image}
        tp = np.zeros(len(scored), dtype=np.float64)
        fp = np.zeros(len(scored), dtype=np.float64)
        for index, (_, image_index, box) in enumerate(scored):
            candidates = gt_by_image[image_index]
            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(candidates):
                if j in matched[image_index]:
                    continue
                overlap = iou_xyxy(box, gt_box)
                if overlap > best_iou:
                    best_iou, best_j = overlap, j
            if best_iou >= iou_threshold and best_j >= 0:
                matched[image_index].add(best_j)
                tp[index] = 1.0
            else:
                fp[index] = 1.0
        if total_gt == 0:
            continue
        precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1e-12)
        recall = np.cumsum(tp) / float(total_gt)
        recall_grid = np.linspace(0.0, 1.0, 101)
        ap = 0.0
        for recall_value in recall_grid:
            ap += float(np.max(precision[recall >= recall_value])) if np.any(recall >= recall_value) else 0.0
        aps.append(ap / 101.0)
    return float(np.mean(aps)) if aps else 0.0


def mean_iou(pred_masks: Sequence[np.ndarray], target_masks: Sequence[np.ndarray], ignore_index: int = 255) -> float:
    classes: set[int] = set()
    for mask in target_masks:
        classes.update(int(x) for x in np.unique(mask) if int(x) != ignore_index)
    if not classes:
        return 0.0
    values: List[float] = []
    for class_id in sorted(classes):
        intersection = 0
        union = 0
        for prediction, target in zip(pred_masks, target_masks):
            valid = target != ignore_index
            pred_class = prediction == class_id
            target_class = target == class_id
            intersection += int(np.logical_and(np.logical_and(pred_class, target_class), valid).sum())
            union += int(np.logical_and(np.logical_or(pred_class, target_class), valid).sum())
        if union:
            values.append(intersection / union)
    return float(np.mean(values)) if values else 0.0


def build_detector(device: torch.device):
    import torchvision
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    return model.to(device).eval()


def build_segmenter(device: torch.device):
    import torchvision
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=DeepLabV3_ResNet50_Weights.DEFAULT
    )
    return model.to(device).eval()


@torch.no_grad()
def predict_detector(model, image: torch.Tensor, device: torch.device, threshold: float) -> dict:
    result = model([image.to(device)])[0]
    keep = result["scores"] >= threshold
    return {
        "boxes": result["boxes"][keep].detach().cpu().tolist(),
        "labels": result["labels"][keep].detach().cpu().tolist(),
        "scores": result["scores"][keep].detach().cpu().tolist(),
    }


@torch.no_grad()
def predict_segmenter(model, image: torch.Tensor, device: torch.device) -> np.ndarray:
    output = model((image / 255.0).unsqueeze(0).to(device))["out"]
    return output.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)


def _read_class_index_mask(path: str) -> np.ndarray:
    """Read a palette-indexed segmentation PNG as raw class-id values.

    Must not go through ``_read_image``: that function decodes to an RGB
    photo (via ``cv2.IMREAD_COLOR`` or ``Image.convert('RGB')``), which maps
    each palette index through to an arbitrary display colour and destroys
    the class-id semantics that VOC-style masks (0..20, 255=ignore) rely on.
    """
    from PIL import Image

    with Image.open(path) as image:
        return np.array(image, dtype=np.int64)


def load_target(record: Mapping[str, object], root: str, resize_hw: Tuple[int, int]) -> Tuple[dict, np.ndarray | None]:
    target = {"boxes": record.get("boxes", []), "labels": record.get("labels", [])}
    mask = None
    mask_path = record.get("segmentation_path")
    if mask_path:
        mask_image = _read_class_index_mask(str(resolve_path(str(mask_path), root)))
        # Nearest-neighbour resize preserves class IDs.
        import cv2

        mask = cv2.resize(mask_image, (resize_hw[1], resize_hw[0]), interpolation=cv2.INTER_NEAREST)
    return target, mask


class ToyDetector:
    def __call__(self, images: Sequence[torch.Tensor]) -> List[dict]:
        height, width = images[0].shape[-2:]
        return [{"boxes": torch.tensor([[0, 0, width, height]], device=images[0].device, dtype=torch.float32),
                 "labels": torch.tensor([1], device=images[0].device),
                 "scores": torch.tensor([0.9], device=images[0].device)}]


class ToySegmenter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, images: torch.Tensor) -> dict:
        batch, _, height, width = images.shape
        output = torch.zeros(batch, 2, height, width, device=images.device)
        output[:, 1] = images[:, 0] - images[:, 1]
        return {"out": output}


def toy_prediction(model, image: torch.Tensor, device: torch.device, threshold: float) -> dict:
    result = model([image.to(device)])[0]
    keep = result["scores"] >= threshold
    return {key: result[key][keep].detach().cpu().tolist() for key in ("boxes", "labels", "scores")}


def run_smoke(args: argparse.Namespace) -> Path:
    from run_train import SensitiveRegionNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).with_name("tomm_same_image_utility_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    size = int(args.smoke_size)
    torch.manual_seed(20260830)
    image = torch.rand(3, size, size, device="cpu") * 255.0
    cfg = load_yaml(args.config)
    sanitizer = SensitiveRegionNet().to(device).eval()
    detector = ToyDetector()
    segmenter = ToySegmenter().to(device).eval()
    target = {"boxes": [[0, 0, size, size]], "labels": [1]}
    target_mask = np.zeros((size, size), dtype=np.int64)
    target_mask[:, size // 2 :] = 1
    rows = []
    for variant in ("full", "global_noise"):
        protected, _ = protect_review_clip(
            image.unsqueeze(0), sanitizer, cfg, device, variant, int(args.seed), allow_ssim_fallback=True
        )
        protected_image = protected[0]
        prediction = toy_prediction(detector, protected_image, device, 0.05)
        segment_prediction = segmenter((protected_image / 255.0).unsqueeze(0).to(device))["out"].argmax(1)[0]
        rows.append(
            {
                "variant": variant,
                "map50": average_precision_detections([prediction], [target]),
                "miou": mean_iou([segment_prediction.detach().cpu().numpy()], [target_mask]),
            }
        )
    (output_dir / "utility_summary.json").write_text(json.dumps({"device": str(device), "rows": rows, "scientific_evidence": False}, indent=2), encoding="utf-8")
    if not all(np.isfinite(float(row["map50"])) and np.isfinite(float(row["miou"])) for row in rows):
        raise RuntimeError("Same-image utility smoke test failed.")
    print(f"[smoke] same-image utility metrics passed on {device}; wrote {output_dir}")
    return output_dir


def run_manifest(args: argparse.Namespace) -> Path:
    if not args.manifest:
        raise ValueError("--manifest is required in manifest mode.")
    records = load_manifest(args.manifest, args.root)
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resize_hw = (int(args.resize_h), int(args.resize_w))
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
    detector = build_detector(device)
    segmenter = build_segmenter(device)
    original_det: List[dict] = []
    original_seg: List[np.ndarray] = []
    targets: List[dict] = []
    target_masks: List[np.ndarray] = []
    images: List[torch.Tensor] = []
    for record in records:
        image = _resize_if_needed(_read_image(record["image_path"]), resize_hw)
        target, mask = load_target(record, args.root, resize_hw)
        if not target["boxes"] and mask is None:
            raise ValueError(f"Record {record['image_id']} has neither detection nor segmentation labels.")
        images.append(image)
        targets.append(target)
        original_det.append(predict_detector(detector, image / 255.0, device, float(args.score_threshold)))
        if mask is not None:
            original_seg.append(predict_segmenter(segmenter, image, device))
            target_masks.append(mask)

    if target_masks and len(target_masks) != len(records):
        raise ValueError("Segmentation labels must be present for every manifest record or for none of them.")

    rows: List[dict] = []
    for variant in args.variants:
        protected_images = []
        for image in images:
            protected, _ = protect_review_clip(image.unsqueeze(0), sensnet, cfg, device, variant, int(args.seed))
            protected_images.append(protected[0])
        protected_det = [predict_detector(detector, image / 255.0, device, float(args.score_threshold)) for image in protected_images]
        det_row = {
            "variant": variant,
            "map50_original": average_precision_detections(original_det, targets),
            "map50_sanitized": average_precision_detections(protected_det, targets),
        }
        if target_masks:
            protected_seg = [predict_segmenter(segmenter, image, device) for image in protected_images]
            det_row["miou_original"] = mean_iou(original_seg, target_masks)
            det_row["miou_sanitized"] = mean_iou(protected_seg, target_masks)
        rows.append(det_row)
    (output_dir / "utility_summary.json").write_text(json.dumps({"rows": rows, "device": str(device), "scientific_evidence": True}, indent=2), encoding="utf-8")
    print(f"[utility] wrote {len(rows)} rows to {output_dir}")
    return output_dir


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_manifest(args)


if __name__ == "__main__":
    main()
