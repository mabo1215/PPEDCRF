"""Build the same-image utility manifest for E4 (detection mAP + segmentation mIoU).

Detection records come from COCO val2017 (boxes/labels match the raw COCO
category ids used by torchvision's COCO-pretrained Faster R-CNN). Segmentation
records come from the official PASCAL VOC 2012 segmentation validation split
(class-indexed masks match torchvision's COCO-with-VOC-labels DeepLabV3).

Detection boxes are rescaled here into the (resize_h, resize_w) coordinate
space that ``evaluate_same_image_utility.py`` resizes images into, since that
script does not rescale manifest boxes itself (it only rescales segmentation
masks). Segmentation masks are left at their native resolution; the evaluation
script nearest-neighbour-resizes them.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the E4 same-image utility manifest.")
    parser.add_argument("--coco_root", default="/mnt/f/work/datasets/coco")
    parser.add_argument("--voc_root", default="/mnt/f/work/datasets/VOC")
    parser.add_argument("--num_detection", type=int, default=200)
    parser.add_argument("--num_segmentation", type=int, default=200)
    parser.add_argument("--resize_h", type=int, default=192)
    parser.add_argument("--resize_w", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_detection_records(coco_root: Path, num: int, seed: int, resize_hw) -> List[dict]:
    ann_path = coco_root / "annotations" / "instances_val2017.json"
    with ann_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    images_by_id = {image["id"]: image for image in data["images"]}
    anns_by_image: Dict[int, List[dict]] = {}
    for ann in data["annotations"]:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    eligible = sorted(image_id for image_id in anns_by_image if anns_by_image[image_id])
    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[: int(num)]

    resize_h, resize_w = resize_hw
    records: List[dict] = []
    for image_id in chosen:
        image_meta = images_by_id[image_id]
        orig_w, orig_h = float(image_meta["width"]), float(image_meta["height"])
        sx, sy = resize_w / orig_w, resize_h / orig_h
        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            x0, y0, x1, y1 = x * sx, y * sy, (x + w) * sx, (y + h) * sy
            boxes.append([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)])
            labels.append(int(ann["category_id"]))
        if not boxes:
            continue
        records.append(
            {
                "image_id": f"coco_{image_id}",
                "image_path": str((coco_root / "val2017" / image_meta["file_name"]).resolve()),
                "boxes": boxes,
                "labels": labels,
                "source": "coco_val2017",
            }
        )
    return records


def build_segmentation_records(voc_root: Path, num: int, seed: int) -> List[dict]:
    split_path = voc_root / "ImageSets" / "Segmentation" / "val.txt"
    ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed + 1)
    rng.shuffle(ids)
    chosen = ids[: int(num)]

    records: List[dict] = []
    for image_id in chosen:
        image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        mask_path = voc_root / "SegmentationClass" / f"{image_id}.png"
        if not image_path.is_file() or not mask_path.is_file():
            continue
        records.append(
            {
                "image_id": f"voc_{image_id}",
                "image_path": str(image_path.resolve()),
                "boxes": [],
                "labels": [],
                "segmentation_path": str(mask_path.resolve()),
                "source": "voc2012_segmentation_val",
            }
        )
    return records


def main() -> None:
    args = parse_args()
    resize_hw = (int(args.resize_h), int(args.resize_w))
    detection_records = build_detection_records(
        Path(args.coco_root), int(args.num_detection), int(args.seed), resize_hw
    )
    segmentation_records = build_segmentation_records(
        Path(args.voc_root), int(args.num_segmentation), int(args.seed)
    )
    if not detection_records and not segmentation_records:
        raise RuntimeError("No manifest records were built; check --coco_root/--voc_root.")

    # evaluate_same_image_utility.py requires every record in one manifest to
    # carry segmentation labels, or none of them, so detection (COCO) and
    # segmentation (VOC) tracks are written as two separate manifests rather
    # than interleaved in one file.
    output_path = Path(args.output)
    detection_path = output_path.with_name(f"{output_path.stem}_detection{output_path.suffix}")
    segmentation_path = output_path.with_name(f"{output_path.stem}_segmentation{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, records: List[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    if detection_records:
        _write(detection_path, detection_records)
        print(f"[manifest] wrote {len(detection_records)} detection records to {detection_path}")
    if segmentation_records:
        _write(segmentation_path, segmentation_records)
        print(f"[manifest] wrote {len(segmentation_records)} segmentation records to {segmentation_path}")


if __name__ == "__main__":
    main()
