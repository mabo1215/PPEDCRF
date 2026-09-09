"""Measure the mIoU spread between published segmenters on the utility images.

Why this exists. The privacy-utility frontier judges a mechanism against a
declared tolerance on segmentation mIoU. A tolerance stated without a reference
scale is arbitrary: the reader has no way to tell whether losing 0.05 mIoU is a
rounding error or a catastrophe. One defensible reference scale is the spread
between segmenters that the community already treats as interchangeable choices
for the same task -- if two published designs differ by about that much on the
very images used here, then a perturbation that costs the same amount costs the
downstream consumer about as much as swapping one published segmenter for
another.

What it measures. Every model is a torchvision segmentation architecture with
released COCO-with-VOC-labels weights, evaluated with its own official
preprocessing transform against the VOC ground-truth masks of the same image
subset the utility experiments use. mIoU is dataset-level and identical in
definition to the utility pipeline: per-class intersections and unions are
pooled over images before the ratio is taken, the class set is the set of
classes appearing in any ground-truth mask, and ignore pixels (255) are
excluded from both intersection and union.

Gate. The DeepLabV3-ResNet50 row must reproduce the clean mIoU already exported
by the utility runs (0.6974 on this subset). That cell is the only one whose
answer is known in advance, so it is the check that this script's preprocessing,
mask decoding and pooling agree with the pipeline whose tolerance it calibrates.
A mismatch there invalidates every other row. Reaching it requires the same
working resolution as the utility runs: frames are resized to 192x320 with
antialiasing and labels with nearest-neighbour interpolation, exactly as the
pipeline does, and the image-loading and resize helpers are imported from the
pipeline rather than reimplemented. Scored at native resolution instead, the
same weights read 0.782 -- a scale on which the declared tolerance would mean
something different, which is why the gate is checked before anything is
quoted.

Uncertainty. Each adjacent pair in the mIoU ranking gets a paired image
bootstrap: images are resampled with replacement and both models' pooled mIoU
are recomputed on the same resample, so the interval is on the gap rather than
on either endpoint. The marginal interval of a single mIoU is much wider and is
not the quantity a tolerance is compared against.

Resumable. Predictions are appended one (image, model) row at a time and
flushed to disk, and a restart skips rows already present in the output.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

IGNORE_INDEX = 255

# Published torchvision segmenters with COCO-with-VOC-labels weights, in the
# order they are reported. Each entry is (label, builder attribute, weights
# enum name). The label is what appears in the manuscript and in the export.
MODELS: List[Tuple[str, str, str]] = [
    ("FCN-ResNet50", "fcn_resnet50", "FCN_ResNet50_Weights"),
    ("FCN-ResNet101", "fcn_resnet101", "FCN_ResNet101_Weights"),
    ("DeepLabV3-ResNet50", "deeplabv3_resnet50", "DeepLabV3_ResNet50_Weights"),
    ("DeepLabV3-ResNet101", "deeplabv3_resnet101", "DeepLabV3_ResNet101_Weights"),
    ("DeepLabV3-MobileNetV3", "deeplabv3_mobilenet_v3_large",
     "DeepLabV3_MobileNet_V3_Large_Weights"),
    ("LR-ASPP-MobileNetV3", "lraspp_mobilenet_v3_large",
     "LRASPP_MobileNet_V3_Large_Weights"),
]

# The manifest was written on Windows and records absolute drive paths. Reading
# it from a POSIX mount needs the drive letter mapped, not the manifest edited:
# the manifest is an experiment record and rewriting it would break the
# provenance of every export that names it.
_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def localize(path: str) -> Path:
    """Return a path readable on this platform without altering the manifest."""
    text = str(path)
    match = _DRIVE.match(text)
    if match and os.name != "nt":
        drive, rest = match.groups()
        return Path(f"/mnt/{drive.lower()}") / rest.replace("\\", "/")
    if os.name != "nt":
        return Path(text.replace("\\", "/"))
    return Path(text)


def read_frame(path: Path, resize_hw: Tuple[int, int]):
    """Load and resize a frame through the pipeline's own helpers.

    Importing them rather than reimplementing them is what keeps this
    measurement on the same footing as the utility runs it calibrates: the
    colour conversion, the float range and the antialiased resize are the ones
    the exported mIoU was produced with.
    """
    from datasets.driving_clip_dataset import _read_image, _resize_if_needed

    return _resize_if_needed(_read_image(str(path)), resize_hw)


def read_label(path: Path, resize_hw: Tuple[int, int]) -> np.ndarray:
    """Load a palette-indexed VOC label at the pipeline's working resolution.

    The mask must not be decoded as a photo: an RGB conversion maps each
    palette index to its display colour and destroys the class-id semantics.
    Nearest-neighbour resize is what preserves those ids.
    """
    import cv2
    from PIL import Image

    with Image.open(path) as image:
        mask = np.array(image, dtype=np.int64)
    return cv2.resize(mask, (resize_hw[1], resize_hw[0]),
                      interpolation=cv2.INTER_NEAREST)


def build_model(builder: str, weights_name: str, device):
    """Instantiate one published segmenter with its own official transform."""
    import torchvision
    from torchvision.models import segmentation as seg

    weights_enum = getattr(seg, weights_name)
    weights = weights_enum.DEFAULT
    model = getattr(torchvision.models.segmentation, builder)(weights=weights)
    return model.to(device).eval(), weights.transforms(), str(weights)


def per_image_iu(prediction: np.ndarray, target: np.ndarray) -> Dict[str, List[int]]:
    """Intersection and union per class on one image, ignore pixels excluded.

    Only classes with a non-empty union are stored, which is what makes the
    pooled ratio over a set of images equal to the dataset-level mIoU the
    utility pipeline reports.
    """
    valid = target != IGNORE_INDEX
    out: Dict[str, List[int]] = {}
    for class_id in np.union1d(np.unique(prediction[valid]), np.unique(target[valid])):
        class_id = int(class_id)
        pred_class = prediction == class_id
        target_class = target == class_id
        union = int(np.logical_and(np.logical_or(pred_class, target_class), valid).sum())
        if not union:
            continue
        intersection = int(
            np.logical_and(np.logical_and(pred_class, target_class), valid).sum()
        )
        out[str(class_id)] = [intersection, union]
    return out


def pooled_miou(rows: Iterable[dict], classes: Sequence[int]) -> float:
    """Dataset-level mIoU: pool per-class I and U over images, then average.

    ``classes`` is the ground-truth class set, so a class the model predicts but
    that never appears in a label is not scored -- the same restriction the
    utility pipeline applies.
    """
    inter: Dict[int, int] = defaultdict(int)
    union: Dict[int, int] = defaultdict(int)
    for row in rows:
        for key, (i_val, u_val) in row["seg_iu"].items():
            inter[int(key)] += int(i_val)
            union[int(key)] += int(u_val)
    values = [inter[c] / union[c] for c in classes if union[c]]
    return float(np.mean(values)) if values else 0.0


def predict_all(args, records, out_path: Path) -> None:
    """Run every requested model over every image, appending rows as they land."""
    import torch

    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    done.add((row["image_id"], row["model"]))
    if done:
        print(f"[resume] {len(done)} rows already present", flush=True)

    device = torch.device(args.device)
    stream = out_path.open("a", encoding="utf-8")
    try:
        for label, builder, weights_name in MODELS:
            if args.models and label not in args.models:
                continue
            pending = [r for r in records if (r["image_id"], label) not in done]
            if not pending:
                print(f"[skip] {label}: complete", flush=True)
                continue
            try:
                model, preprocess, weights_id = build_model(builder, weights_name, device)
            except Exception as exc:  # noqa: BLE001 - a missing download is not fatal
                print(f"[skip] {label}: weights unavailable ({exc})", flush=True)
                continue
            print(f"[run ] {label}: {len(pending)} images, weights {weights_id}", flush=True)
            resize_hw = (int(args.resize_h), int(args.resize_w))
            for index, record in enumerate(pending, start=1):
                image = read_frame(localize(record["image_path"]), resize_hw)
                target = read_label(localize(record["segmentation_path"]), resize_hw)
                original_size = tuple(int(v) for v in image.shape[-2:])
                with torch.no_grad():
                    model_input = preprocess(image / 255.0).unsqueeze(0).to(device)
                    logits = model(model_input)["out"]
                    logits = torch.nn.functional.interpolate(
                        logits, size=original_size, mode="bilinear", align_corners=False
                    )
                    prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
                valid = target != IGNORE_INDEX
                row = {
                    "image_id": record["image_id"],
                    "model": label,
                    "weights": weights_id,
                    "gt_classes": sorted(int(c) for c in np.unique(target[valid])),
                    "seg_iu": per_image_iu(prediction, target),
                }
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                if index % 50 == 0 or index == len(pending):
                    print(f"       {index}/{len(pending)}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        stream.close()


def analyse(out_path: Path, summary_path: Path, gate: float, tolerance: float,
            n_boot: int, seed: int) -> dict:
    """Rank the models, gate against the known cell, and interval each gap."""
    rows = [json.loads(l) for l in out_path.open(encoding="utf-8") if l.strip()]
    by_model: Dict[str, Dict[str, dict]] = defaultdict(dict)
    gt_classes: set = set()
    for row in rows:
        by_model[row["model"]][row["image_id"]] = row
        gt_classes.update(row["gt_classes"])
    classes = sorted(gt_classes)

    shared = None
    for per_image in by_model.values():
        shared = set(per_image) if shared is None else shared & set(per_image)
    image_ids = sorted(shared or [])

    scores = {
        model: pooled_miou([per_image[i] for i in image_ids], classes)
        for model, per_image in by_model.items()
    }
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    gate_model = "DeepLabV3-ResNet50"
    gate_value = scores.get(gate_model)
    gate_ok = gate_value is not None and abs(gate_value - gate) <= 5e-4

    rng = np.random.default_rng(seed)
    index = np.arange(len(image_ids))
    resamples = [rng.integers(0, len(image_ids), len(image_ids)) for _ in range(n_boot)]

    def boot_gap(model_a: str, model_b: str) -> Tuple[float, float]:
        a, b = by_model[model_a], by_model[model_b]
        draws = []
        for pick in resamples:
            ids = [image_ids[j] for j in pick]
            draws.append(
                pooled_miou([a[i] for i in ids], classes)
                - pooled_miou([b[i] for i in ids], classes)
            )
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    adjacent = []
    for (model_a, score_a), (model_b, score_b) in zip(ranking, ranking[1:]):
        lo, hi = boot_gap(model_a, model_b)
        adjacent.append({
            "better": model_a,
            "worse": model_b,
            "miou_better": round(score_a, 6),
            "miou_worse": round(score_b, 6),
            "gap": round(score_a - score_b, 6),
            "ci95": [round(lo, 6), round(hi, 6)],
        })

    gaps = [entry["gap"] for entry in adjacent]
    full_span = ranking[0][1] - ranking[-1][1] if len(ranking) > 1 else 0.0
    summary = {
        "n_images": len(image_ids),
        "n_classes": len(classes),
        "n_bootstrap": n_boot,
        "seed": seed,
        "gate": {
            "model": gate_model,
            "expected": gate,
            "observed": None if gate_value is None else round(gate_value, 6),
            "passed": bool(gate_ok),
        },
        "miou": {model: round(value, 6) for model, value in ranking},
        "adjacent_gaps": adjacent,
        "median_adjacent_gap": round(float(np.median(gaps)), 6) if gaps else None,
        "max_adjacent_gap": round(float(np.max(gaps)), 6) if gaps else None,
        "full_span": round(float(full_span), 6),
        "declared_tolerance": tolerance,
        "note": (
            "mIoU is dataset-level over the ground-truth class set, ignore "
            "pixels excluded, pooled before the ratio -- the definition the "
            "utility pipeline uses. Intervals are paired image bootstraps on "
            "the gap, not marginal intervals on either mIoU."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest",
                    default=str(REPO / "src" / "outputs" / "e4_audit"
                                / "utility_manifest_segmentation.jsonl"),
                    help="The segmentation utility manifest the frontier uses.")
    ap.add_argument("--output-dir",
                    default=str(REPO / "src" / "exports" / "segmenter_spread"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resize-h", dest="resize_h", type=int, default=192,
                    help="Working resolution of the utility runs; changing it "
                         "changes the scale the tolerance is read on.")
    ap.add_argument("--resize-w", dest="resize_w", type=int, default=320)
    ap.add_argument("--models", nargs="*", default=None,
                    help="Restrict to these labels; default is all available.")
    ap.add_argument("--gate", type=float, default=0.697352,
                    help="Exported clean mIoU the paper's segmenter must reproduce.")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.manifest).open(encoding="utf-8")
               if l.strip()]
    records = [r for r in records if r.get("segmentation_path")]
    print(f"[data] {len(records)} labelled images from {args.manifest}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_image.jsonl"

    if not args.analyse_only:
        predict_all(args, records, out_path)

    summary = analyse(out_path, out_dir / "segmenter_spread_summary.json",
                      args.gate, args.tolerance, args.bootstrap, args.seed)

    gate = summary["gate"]
    print(f"\n[gate] {gate['model']}: observed {gate['observed']} vs "
          f"expected {gate['expected']} -> {'PASS' if gate['passed'] else 'FAIL'}",
          flush=True)
    print(f"[data] {summary['n_images']} images, {summary['n_classes']} classes")
    for model, value in summary["miou"].items():
        print(f"       {model:<24} {value:.4f}")
    print("[gaps] adjacent pairs in the ranking:")
    for entry in summary["adjacent_gaps"]:
        print(f"       {entry['better']:<24} - {entry['worse']:<24} "
              f"{entry['gap']:+.4f} [{entry['ci95'][0]:+.4f},{entry['ci95'][1]:+.4f}]")
    print(f"[span] median adjacent gap {summary['median_adjacent_gap']}, "
          f"max {summary['max_adjacent_gap']}, full span {summary['full_span']}, "
          f"declared tolerance {summary['declared_tolerance']}")
    if not gate["passed"]:
        print("[gate] FAILED: the pipeline does not reproduce the known cell; "
              "no other row here should be quoted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
