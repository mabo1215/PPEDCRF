"""Does either axis reach a model that emits a coordinate instead of a neighbour?

The fourteenth review's fourth finding: every attacker in this paper retrieves
from a gallery, while the literature it addresses increasingly targets
image-to-GPS models. The manuscript states that as a scope condition and gives
the reason it will not close by analogy -- our account of why the axes differ
is stated for a correct-versus-hardest-negative *ranking* margin, and a model
that regresses a coordinate has no gallery, no hardest negative and no rank to
flip, so neither half of the argument transfers. That makes it an open
empirical question, which is what this measures.

The attacker is GeoCLIP, which encodes a frame with CLIP ViT-L/14 and scores it
against a 100k-point GPS gallery. Two things make it the right first
geolocator here. Its backbone is one of the five retrieval attackers the paper
already reports, so a difference between the two settings is a difference in
the head and the task rather than in the features. And it is public, so the
measurement is reproducible without us training anything.

Success is not Top-k. The threat model for a geolocator is distance, so every
condition is scored by the great-circle error between the predicted coordinate
and the query's own, and summarised at the thresholds this literature uses:
1 km (street), 25 km (city), 200 km (region).

Conditions are the released frames of the retrieval study, rebuilt from the
same seeds at the same delivered MSE, so nothing about the defense changes --
only who reads the frame. Rows are flushed and fsynced per completed
(query, condition, seed) and completed keys are skipped at startup.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig, default_input_size_for_backbone, make_default_embedder)
from eval.sanitizers import SANITIZERS  # noqa: E402
from scripts.run_direction_transfer_study import (  # noqa: E402
    directional_delta, normalised_embedding, release_at_mse)
from scripts.run_geotagged_vpr_benchmark import load_image, load_manifest  # noqa: E402
from scripts.run_placement_rule_study import (  # noqa: E402
    center_map, edge_map, renormalise_to_energy, saliency_map)

EARTH_KM = 6371.0088
THRESHOLDS_KM = (1.0, 25.0, 200.0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def placement_weights(frame: torch.Tensor, rule: str,
                      generator: torch.Generator) -> torch.Tensor:
    """Per-pixel weights, later renormalised to the uniform arm's energy."""
    if rule == "uniform":
        return torch.ones_like(frame[:, :1])
    if rule == "edge":
        return edge_map(frame)
    if rule == "saliency":
        return saliency_map(frame)
    if rule == "center":
        return center_map(frame)
    if rule == "random_fixed":
        g = torch.Generator(device="cpu").manual_seed(20260912)
        return torch.rand(frame[:, :1].shape, generator=g).to(frame.device)
    raise SystemExit(f"unknown placement rule {rule!r}")


def noise_release(frame: torch.Tensor, rule: str, target_mse: float,
                  generator: torch.Generator) -> torch.Tensor:
    """A weighted Gaussian release at the prescribed delivered MSE.

    The weights are renormalised to the uniform map's sum of squares before the
    bisection, so every placement spends the same energy before the clamp and
    the same delivered distortion after it -- the study's own energy gate.
    """
    w = placement_weights(frame, rule, generator)
    uni = torch.ones_like(w)
    w = renormalise_to_energy(w, uni.square().sum())
    eps = torch.randn(frame.shape, generator=generator).to(frame.device)
    return release_at_mse(frame, w * eps, target_mse)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"])
    ap.add_argument("--target_mse", type=float, default=15.68)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--random_start", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conditions", nargs="+",
                    default=["clean", "isotropic", "edge", "saliency",
                             "direction", "hardened"])
    ap.add_argument("--eot_sanitizers", nargs="*",
                    default=["jpeg75", "jpeg50", "blur", "denoise"])
    ap.add_argument("--eot_samples", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[geo] device={device}", flush=True)

    records, _gallery = load_manifest(args.manifest, args.root)
    queries = records if not args.limit else records[: args.limit]
    resize_hw = (args.height, args.width)
    print(f"[geo] {len(queries)} queries", flush=True)

    # Surrogates only: the geolocator is never in the optimiser, which is what
    # makes this a transfer measurement rather than a white-box bound.
    embedders, sizes = {}, {}
    for b in args.surrogates:
        cfg = RetrievalConfig(backbone=b,
                              input_size=default_input_size_for_backbone(b))
        embedders[b] = make_default_embedder(cfg).eval().to(device)
        sizes[b] = cfg.input_size
    eot_ops = [SANITIZERS[n] for n in args.eot_sanitizers]

    from geoclip import GeoCLIP
    geo = GeoCLIP().eval().to(device)
    gps_gallery = geo.gps_gallery.to(device)
    print(f"[geo] GeoCLIP loaded, {gps_gallery.shape[0]} gallery points",
          flush=True)

    @torch.no_grad()
    def predict(frame: torch.Tensor):
        """Top-1 coordinate for a released frame, without a file round trip."""
        from PIL import Image
        arr = frame.squeeze(0).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        x = geo.image_encoder.preprocess_image(Image.fromarray(arr)).to(device)
        logits = geo.forward(x, gps_gallery)
        idx = int(torch.topk(logits.softmax(dim=-1), 1, dim=1).indices[0, 0])
        pt = gps_gallery[idx]
        return float(pt[0]), float(pt[1])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["query_id"], r["condition"], r["seed"]))
        print(f"[geo] resuming, {len(done)} rows already present", flush=True)

    fields = ["query_id", "place_id", "city", "condition", "seed",
              "pred_lat", "pred_lon", "true_lat", "true_lon",
              "error_km", "effective_mse"]
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new:
            writer.writeheader()
        for qi, rec in enumerate(queries):
            frame = load_image(rec["query_path"], resize_hw).unsqueeze(0).to(device)
            lat, lon = float(rec["latitude"]), float(rec["longitude"])
            targets = None
            for seed in args.seeds:
                for cond in args.conditions:
                    key = (rec["query_id"], cond, str(seed))
                    if key in done:
                        continue
                    # Seed per (query, condition, seed) rather than per
                    # (query, seed). A generator advanced through the
                    # condition list is only reproducible when every
                    # condition is drawn: the resume path skips completed
                    # ones without advancing it, so a resumed process drew a
                    # different field for every later condition than a fresh
                    # one. That is invisible until two writers race the same
                    # file and disagree, which is how it was found.
                    gen = torch.Generator().manual_seed(
                        seed
                        + (zlib.crc32(rec["query_id"].encode()) % 100000)
                        + (zlib.crc32(cond.encode()) % 100000))
                    if cond == "clean":
                        rel = frame
                    elif cond in ("isotropic", "uniform"):
                        rel = noise_release(frame, "uniform", args.target_mse, gen)
                    elif cond in ("edge", "saliency", "center", "random_fixed"):
                        rel = noise_release(frame, cond, args.target_mse, gen)
                    elif cond in ("direction", "hardened"):
                        if targets is None:
                            targets = [normalised_embedding(
                                embedders[b], frame, sizes[b])
                                for b in args.surrogates]
                        delta = directional_delta(
                            frame, targets,
                            [embedders[b] for b in args.surrogates],
                            [sizes[b] for b in args.surrogates],
                            steps=args.steps, step_size=args.step_size,
                            linf=args.linf, random_start=args.random_start,
                            generator=gen,
                            eot_ops=eot_ops if cond == "hardened" else (),
                            eot_samples=args.eot_samples)
                        rel = release_at_mse(frame, delta, args.target_mse)
                    else:
                        raise SystemExit(f"unknown condition {cond!r}")
                    plat, plon = predict(rel)
                    writer.writerow({
                        "query_id": rec["query_id"],
                        "place_id": rec.get("place_id", ""),
                        "city": rec.get("city", ""),
                        "condition": cond, "seed": seed,
                        "pred_lat": f"{plat:.6f}", "pred_lon": f"{plon:.6f}",
                        "true_lat": f"{lat:.6f}", "true_lon": f"{lon:.6f}",
                        "error_km": f"{haversine_km(lat, lon, plat, plon):.4f}",
                        "effective_mse": f"{float((rel - frame).square().mean()):.4f}",
                    })
                    fh.flush()
                    os.fsync(fh.fileno())
            if (qi + 1) % 25 == 0:
                print(f"[geo] {qi + 1}/{len(queries)} queries", flush=True)
    print(f"[geo] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
