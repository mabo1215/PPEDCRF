"""What does the direction axis cost the downstream task?

The placement and operator studies report detector mAP and segmenter mIoU for
the mechanism's own additive perturbation, which is the *allocation* axis. The
paper's positive result lives on the *direction* axis, and its only quality
evidence there is a distortion figure (delivered MSE and PSNR). PSNR is not
utility, and an adversarially directed perturbation is precisely the family for
which the two can diverge: a perturbation optimised to move a retrieval
embedding may damage a detector or segmenter far more than isotropic noise of
identical energy.

This script closes that gap by running the same frozen detector and segmenter
used for the allocation-axis utility table over the same co-audited manifests,
with three conditions compared at identical delivered MSE:

  clean       the unperturbed reference
  isotropic   the operating-point control: same delivered MSE, no direction
  direction   the deployable, gallery-free surrogate-ensemble direction

The direction condition uses the "self" objective -- the perturbation is
steered away from the frame's own clean embedding under each surrogate -- so it
needs nothing but the frame in hand. That is also the only objective available
here: these are COCO and VOC frames with no retrieval gallery, so a
positive-targeted direction is undefined for them.

Per-image results are written incrementally and the run resumes by skipping
(image_id, condition) pairs already present, so it can be interrupted the
moment the GPU is needed elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
)
from datasets.driving_clip_dataset import _read_image, _resize_if_needed  # noqa: E402
from scripts.evaluate_same_image_utility import (  # noqa: E402
    average_precision_detections,
    build_detector,
    build_segmenter,
    load_manifest as load_utility_manifest,
    load_target,
    predict_detector,
    predict_segmenter_batch,
)
from scripts.run_direction_transfer_study import (  # noqa: E402
    directional_delta,
    normalised_embedding,
    release_at_mse,
)
from eval.sanitizers import SANITIZERS  # noqa: E402

IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Downstream utility of the direction perturbation.")
    ap.add_argument("--manifest", required=True,
                    help="Utility manifest (detection or segmentation), the "
                         "same file used for the allocation-axis table.")
    ap.add_argument("--root", default="",
                    help="Root for relative manifest paths.")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"],
                    help="Surrogate ensemble the direction is optimised "
                         "against. No attacker model is involved: utility is "
                         "measured on the released frame itself.")
    ap.add_argument("--target_mse", type=float, default=15.68,
                    help="Delivered MSE every perturbed condition is matched "
                         "to; the operating point used throughout the paper.")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--random_start", type=float, default=1.0,
                    help="uniform displacement (pixel units) before the first "
                         "sign-gradient step. Required for this objective: "
                         "the clean frame is a stationary point of the "
                         "self-similarity objective, so a zero start leaves "
                         "some frames unperturbed entirely.")
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--resize_h", type=int, default=192)
    ap.add_argument("--resize_w", type=int, default=320)
    ap.add_argument("--score_threshold", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eot_sanitizers", nargs="*", default=[],
                    help="if given, add a 'hardened_direction' condition "
                         "optimised in expectation over these attacker-side "
                         "transforms (same names as the transfer study), "
                         "alongside the unhardened 'direction' so the two are "
                         "paired within one run.")
    ap.add_argument("--eot_samples", type=int, default=2)
    ap.add_argument("--output_dir", required=True)
    return ap.parse_args()


def seg_intersection_union(pred: np.ndarray, target: np.ndarray,
                           classes: Sequence[int]) -> Dict[str, List[int]]:
    """Per-class intersection and union contributions for one image.

    Storing these instead of the mask itself keeps the resume cache small
    while remaining exactly aggregatable: mIoU sums intersections and unions
    over images before dividing, so per-image contributions are sufficient.
    """
    valid = target != IGNORE_INDEX
    out: Dict[str, List[int]] = {}
    for class_id in classes:
        pred_class = pred == class_id
        target_class = target == class_id
        inter = int(np.logical_and(np.logical_and(pred_class, target_class),
                                   valid).sum())
        union = int(np.logical_and(np.logical_or(pred_class, target_class),
                                   valid).sum())
        if inter or union:
            out[str(class_id)] = [inter, union]
    return out


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[utility-direction] device={device}", flush=True)

    records = load_utility_manifest(args.manifest, args.root)
    if args.limit:
        records = records[: args.limit]
    resize_hw = (int(args.resize_h), int(args.resize_w))

    images: List[torch.Tensor] = []
    targets: List[dict] = []
    masks: List[np.ndarray | None] = []
    for record in records:
        image = _resize_if_needed(_read_image(record["image_path"]), resize_hw)
        target, mask = load_target(record, args.root, resize_hw)
        images.append(image)
        targets.append(target)
        masks.append(mask)
    has_det = any(bool(t.get("boxes")) for t in targets)
    has_seg = any(m is not None for m in masks)
    if has_seg and any(m is None for m in masks):
        raise ValueError("Segmentation labels must be present for every "
                         "record or for none of them.")
    print(f"[utility-direction] {len(records)} images, detection={has_det}, "
          f"segmentation={has_seg}", flush=True)

    seg_classes: List[int] = []
    if has_seg:
        found: set[int] = set()
        for mask in masks:
            found.update(int(x) for x in np.unique(mask)
                         if int(x) != IGNORE_INDEX)
        seg_classes = sorted(found)

    detector = build_detector(device) if has_det else None
    segmenter, seg_preprocess = build_segmenter(device) if has_seg else (None, None)

    embedders, sizes = {}, {}
    for b in args.surrogates:
        cfg = RetrievalConfig(backbone=b,
                              input_size=default_input_size_for_backbone(b))
        embedders[b] = make_default_embedder(cfg).eval().to(device)
        sizes[b] = cfg.input_size
        print(f"[utility-direction] surrogate ready: {b}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "per_image.jsonl"
    done: set[tuple[str, str, str]] = set()
    if cache.is_file():
        with open(cache, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                done.add((row["image_id"], row["condition"], str(row["seed"])))
        print(f"[utility-direction] resuming, {len(done)} rows present",
              flush=True)
    sink = open(cache, "a", encoding="utf-8")

    conditions = ["clean", "isotropic", "direction"]
    for name in args.eot_sanitizers:
        if name not in SANITIZERS:
            raise SystemExit(f"unknown sanitizer for EOT: {name!r}; "
                             f"available: {sorted(SANITIZERS)}")
    eot_ops = [SANITIZERS[n] for n in args.eot_sanitizers]
    if eot_ops:
        conditions.append("hardened_direction")
    print(f"[utility-direction] conditions={conditions} "
          f"eot={args.eot_sanitizers or 'off'}", flush=True)

    def emit(row: Mapping[str, object]) -> None:
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        os.fsync(sink.fileno())

    for idx, record in enumerate(records, 1):
        image = images[idx - 1]
        frame = image.unsqueeze(0).to(device)
        for seed in args.seeds:
            for cond in conditions:
                key = (record["image_id"], cond, str(seed))
                if key in done:
                    continue
                if cond == "clean":
                    released = frame
                elif cond == "isotropic":
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    delta = torch.randn(frame.shape, generator=g).to(device)
                    released = release_at_mse(frame, delta, args.target_mse)
                else:
                    with torch.no_grad():
                        tgts = [normalised_embedding(
                            embedders[b], frame, sizes[b]).detach()
                            for b in args.surrogates]
                    gstart = torch.Generator(device="cpu").manual_seed(
                        zlib.crc32(
                            f"{record['image_id']}|{seed}".encode())
                        & 0x7FFFFFFF)
                    # The hardened and unhardened directions share the same
                    # random start, so their utility difference is a paired
                    # comparison of the objective, not of the start.
                    with torch.enable_grad():
                        delta = directional_delta(
                            frame, tgts,
                            [embedders[b] for b in args.surrogates],
                            [sizes[b] for b in args.surrogates],
                            args.steps, args.step_size, args.linf,
                            random_start=args.random_start,
                            generator=gstart,
                            eot_ops=(eot_ops if cond == "hardened_direction"
                                     else ()),
                            eot_samples=args.eot_samples)
                    if float(delta.abs().max()) == 0.0:
                        raise RuntimeError(
                            f"Zero perturbation for {record['image_id']}: the "
                            f"random start is not doing its job.")
                    released = release_at_mse(frame, delta, args.target_mse)
                mse = float((released - frame).square().mean())
                row: Dict[str, object] = {
                    "image_id": record["image_id"],
                    "condition": cond,
                    "seed": seed,
                    "effective_mse": round(mse, 6),
                    "psnr": round(float(
                        10.0 * np.log10(255.0 ** 2 / max(mse, 1e-9))), 4),
                }
                released_chw = released[0]
                if has_det:
                    pred = predict_detector(detector, released_chw / 255.0,
                                            device, float(args.score_threshold))
                    row["det"] = {
                        "boxes": [[float(v) for v in b]
                                  for b in pred.get("boxes", [])],
                        "labels": [int(v) for v in pred.get("labels", [])],
                        "scores": [float(v) for v in pred.get("scores", [])],
                    }
                if has_seg:
                    pred_mask = predict_segmenter_batch(
                        segmenter, [released_chw.cpu()], device,
                        seg_preprocess)[0]
                    row["seg_iu"] = seg_intersection_union(
                        pred_mask, masks[idx - 1], seg_classes)
                emit(row)
        if idx % 25 == 0:
            print(f"[utility-direction] {idx}/{len(records)} images",
                  flush=True)
    sink.close()

    # Aggregate. Detection AP is global over the sorted score list, so
    # predictions are reassembled in manifest order; mIoU sums the cached
    # per-image intersections and unions per class.
    rows_by_key: Dict[tuple[str, str, str], dict] = {}
    with open(cache, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_key[(row["image_id"], row["condition"],
                         str(row["seed"]))] = row

    summary = []
    for cond in conditions:
        for seed in args.seeds:
            present = [rows_by_key.get((r["image_id"], cond, str(seed)))
                       for r in records]
            if any(p is None for p in present):
                print(f"[utility-direction] skipping incomplete cell "
                      f"{cond}/{seed}", flush=True)
                continue
            cell: Dict[str, object] = {"condition": cond, "seed": seed}
            mses = [p["effective_mse"] for p in present]
            cell["mean_effective_mse"] = round(float(np.mean(mses)), 6)
            cell["mean_psnr"] = round(float(np.mean(
                [p["psnr"] for p in present])), 4)
            if has_det:
                preds = [p["det"] for p in present]
                cell["map50"] = round(
                    average_precision_detections(preds, targets), 6)
            if has_seg:
                inter: Dict[str, int] = {}
                union: Dict[str, int] = {}
                for p in present:
                    for class_id, (i_val, u_val) in p["seg_iu"].items():
                        inter[class_id] = inter.get(class_id, 0) + i_val
                        union[class_id] = union.get(class_id, 0) + u_val
                ious = [inter[c] / union[c] for c in union if union[c]]
                cell["miou"] = round(float(np.mean(ious)) if ious else 0.0, 6)
            summary.append(cell)

    (out_dir / "utility_direction_summary.json").write_text(
        json.dumps({"rows": summary,
                    "manifest": args.manifest,
                    "surrogates": args.surrogates,
                    "target_mse": args.target_mse,
                    "steps": args.steps,
                    "device": str(device),
                    "n_images": len(records)}, indent=2),
        encoding="utf-8")
    for cell in summary:
        print(f"[utility-direction] {cell}", flush=True)
    print(f"[utility-direction] wrote {out_dir/'utility_direction_summary.json'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
