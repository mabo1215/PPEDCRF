"""Downstream utility measured on the very frames the retrieval attack sees.

The privacy-utility frontier pairs retrieval on MSLS with detector and
segmenter scores on VOC and COCO. Nothing is wrong with either measurement, but
no image contributes to both, so the pairing is between distributions at a
shared budget rather than two readings of one release. A reviewer is entitled
to ask whether the utility cost the frontier reports is the cost of the frames
whose retrieval it reports.

This script answers that on the same 400 MSLS query frames. Those frames carry
no segmentation ground truth -- MSLS is a place-recognition dataset -- so the
reference here is not an annotation but the frozen segmenter's own prediction
on the *clean* frame. What is measured is therefore agreement:

    mIoU( prediction on the released frame , prediction on the clean frame )

read as "how much of what a downstream model saw in this frame survives
release". That is a weaker quantity than accuracy against ground truth and is
reported as such; it is also the only joint quantity these frames admit, and it
is the one the frontier's structural criticism actually asks for. A perfect
score means the released frame is indistinguishable to the segmenter, not that
the segmenter was right.

Conditions match the frontier exactly: clean (trivially 1.0 and kept as the
gate that the pipeline is deterministic), isotropic noise at the same delivered
MSE, the gallery-free surrogate direction, and the EOT-hardened direction.
Rows are written one at a time and flushed, and a restart skips
(query, condition, seed) triples already present.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
)
from datasets.driving_clip_dataset import _read_image, _resize_if_needed  # noqa: E402
from scripts.evaluate_same_image_utility import build_segmenter  # noqa: E402
from scripts.run_direction_transfer_study import (  # noqa: E402
    directional_delta,
    normalised_embedding,
    release_at_mse,
)
from eval.sanitizers import SANITIZERS  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="MSLS manifest; its query_path entries are the frames "
                         "the retrieval attack is evaluated on.")
    ap.add_argument("--root", required=True, help="Root for query_path.")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"])
    ap.add_argument("--target_mse", type=float, default=15.68)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--random_start", type=float, default=1.0)
    ap.add_argument("--eot_sanitizers", nargs="*",
                    default=["jpeg75", "jpeg50", "blur", "denoise"])
    ap.add_argument("--eot_samples", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output_dir", required=True)
    return ap.parse_args()


@torch.no_grad()
def segment(model, preprocess, frame_255: torch.Tensor,
            device: torch.device) -> np.ndarray:
    """Class index per pixel for one frame given in 0..255, shape (1,3,H,W)."""
    out = model(preprocess(frame_255 / 255.0).to(device))["out"]
    return out.argmax(1)[0].to(torch.int16).cpu().numpy()


def agreement(pred: np.ndarray, reference: np.ndarray) -> Dict[str, List[int]]:
    """Per-class intersection and union between two label maps.

    Classes are those either map uses, so a class the release invents counts
    against the score exactly as one it loses does.
    """
    out: Dict[str, List[int]] = {}
    for cls in np.union1d(np.unique(reference), np.unique(pred)):
        p, r = pred == cls, reference == cls
        inter = int(np.logical_and(p, r).sum())
        union = int(np.logical_or(p, r).sum())
        if union:
            out[str(int(cls))] = [inter, union]
    return out


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[joint] device={device}", flush=True)

    records = []
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                records.append({"query_id": r["query_id"],
                                "path": os.path.join(args.root, r["query_path"])})
    if args.limit:
        records = records[: args.limit]
    print(f"[joint] {len(records)} query frames", flush=True)

    for name in args.eot_sanitizers:
        if name not in SANITIZERS:
            raise SystemExit(f"unknown sanitizer for EOT: {name!r}")
    eot_ops = [SANITIZERS[n] for n in args.eot_sanitizers]

    segmenter, seg_preprocess = build_segmenter(device)
    embedders, sizes = {}, {}
    for b in args.surrogates:
        cfg = RetrievalConfig(backbone=b,
                              input_size=default_input_size_for_backbone(b))
        embedders[b] = make_default_embedder(cfg).eval().to(device)
        sizes[b] = cfg.input_size
        print(f"[joint] surrogate ready: {b}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "per_image.jsonl"
    done = set()
    if cache.is_file():
        with cache.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    done.add((row["image_id"], row["condition"],
                              str(row["seed"])))
        print(f"[joint] resuming, {len(done)} rows present", flush=True)
    sink = cache.open("a", encoding="utf-8")

    conditions = ["clean", "isotropic", "direction"]
    if eot_ops:
        conditions.append("hardened_direction")
    resize_hw = (int(args.height), int(args.width))

    for idx, record in enumerate(records, 1):
        if all((record["query_id"], c, str(s)) in done
               for c in conditions for s in args.seeds):
            continue
        frame = _resize_if_needed(_read_image(record["path"]),
                                  resize_hw).unsqueeze(0).to(device)
        reference = segment(segmenter, seg_preprocess, frame, device)
        for seed in args.seeds:
            for cond in conditions:
                key = (record["query_id"], cond, str(seed))
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
                        zlib.crc32(f"{record['query_id']}|{seed}".encode())
                        & 0x7FFFFFFF)
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
                            f"Zero perturbation for {record['query_id']}")
                    released = release_at_mse(frame, delta, args.target_mse)
                mse = float((released - frame).square().mean())
                pred = segment(segmenter, seg_preprocess, released, device)
                row = {
                    "image_id": record["query_id"],
                    "condition": cond,
                    "seed": seed,
                    "target_mse": args.target_mse,
                    "effective_mse": round(mse, 6),
                    "psnr": round(float(10.0 * np.log10(
                        255.0 ** 2 / max(mse, 1e-9))), 4),
                    "seg_iu": agreement(pred, reference),
                }
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                done.add(key)
        if idx % 25 == 0:
            print(f"[joint] {idx}/{len(records)} frames", flush=True)

    sink.close()
    summary = {
        "manifest": args.manifest,
        "n_frames": len(records),
        "target_mse": args.target_mse,
        "surrogates": list(args.surrogates),
        "seeds": list(args.seeds),
        "steps": args.steps,
        "linf": args.linf,
        "random_start": args.random_start,
        "eot_sanitizers": list(args.eot_sanitizers),
        "reference": "segmenter prediction on the clean frame",
        "metric": "dataset-level mIoU of the released prediction against the "
                  "clean prediction; agreement, not accuracy",
    }
    (out_dir / "joint_utility_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {cache}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
