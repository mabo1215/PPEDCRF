"""Which part of an optimised perturbation carries the privacy effect?

The transfer study compares an optimised direction against weighted Gaussian
noise at matched delivered distortion and attributes the difference to
direction. That comparison changes three things at once: the sign pattern, the
per-pixel magnitude pattern, and how much energy the clamp removes. This
script separates them by deriving every condition from the *same* optimised
perturbation, so nothing but the named factor differs:

  direction          the optimised perturbation itself
  sign_shuffle       its per-pixel magnitudes, with signs redrawn at random
  magnitude_uniform  its signs, with every magnitude set to one value
  isotropic          neither: Gaussian noise at the same budget

and crosses each with a placement map that reweights the perturbation before
it is rescaled, which is the cell the placement and operator studies never
ran: placement applied to an optimised direction rather than to noise.

Every row records the realised energy before clipping as well as the delivered
MSE after it, the maximum absolute perturbation, and the clipped fraction.
That is deliberate: matching the sum of squared weights matches expected
energy, not the energy a particular draw delivers, and the difference is an
empirical question rather than an argument.

Top-5 and Top-10 hits are exported beside the rank, because a Top-1 reduction
is not by itself a statement about location privacy.

Writes one row per query per condition per placement, flushed immediately, and
skips completed rows on restart.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

import torch

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
)
from eval.sanitizers import SANITIZERS  # noqa: E402
from scripts.run_direction_transfer_study import (  # noqa: E402
    directional_delta,
    embed_gallery_batched,
    normalised_embedding,
)
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)
from scripts.run_placement_rule_study import (  # noqa: E402
    center_map,
    edge_map,
    saliency_map,
)

CONDITIONS = ("direction", "sign_shuffle", "magnitude_uniform", "isotropic")
PLACEMENTS = ("uniform", "edge", "center", "saliency")


def placement_weight(name: str, frame: torch.Tensor) -> torch.Tensor:
    """A placement map normalised so its mean square is one.

    Normalising this way keeps the map from changing the budget on its own:
    the rescaling to the target MSE happens afterwards, and a map whose mean
    square differed from one would only move the starting point of that
    search.
    """
    if name == "uniform":
        raw = torch.ones_like(frame[:, :1])
    elif name == "edge":
        raw = edge_map(frame)
    elif name == "center":
        raw = center_map(frame)
    elif name == "saliency":
        raw = saliency_map(frame)
    else:
        raise SystemExit(f"unknown placement {name!r}; "
                         f"available: {sorted(PLACEMENTS)}")
    raw = raw.clamp_min(0.0)
    rms = raw.square().mean().sqrt().clamp_min(1e-12)
    return raw / rms


def derive_delta(condition: str, base: torch.Tensor,
                 generator: torch.Generator) -> torch.Tensor:
    """Build a condition's perturbation from the optimised one.

    `sign_shuffle` keeps |delta| exactly and redraws every sign, so the
    magnitude allocation survives and the alignment does not. `magnitude_
    uniform` does the reverse: it keeps sign(delta) and flattens the
    magnitudes to a constant, so the alignment survives and the allocation
    does not. Any difference between them at the same delivered distortion is
    the part of the effect that alignment carries.
    """
    if condition == "direction":
        return base.clone()
    if condition == "sign_shuffle":
        signs = torch.randint(0, 2, base.shape, generator=generator,
                              dtype=torch.float32) * 2.0 - 1.0
        return base.abs() * signs.to(base.device)
    if condition == "magnitude_uniform":
        return torch.sign(base)
    raise SystemExit(f"unknown condition {condition!r}")


def release_with_stats(frame: torch.Tensor, delta: torch.Tensor,
                       target_mse: float, iters: int = 40) -> Dict[str, float]:
    """Scale a perturbation to a delivered MSE and report what the clamp cost.

    `release_at_mse` in the transfer study returns only the released frame.
    Here the pre-clamp energy is needed too: matching the delivered MSE says
    nothing on its own about how much of the perturbation the clamp removed,
    and the manuscript currently argues about that quantity without measuring
    it.
    """
    def released_at(gain: float):
        scaled = gain * delta
        out = (frame + scaled).clamp(0.0, 255.0)
        return out, scaled, float((out - frame).square().mean())

    _, _, m1 = released_at(1.0)
    if m1 <= 1e-12:
        return {"frame": frame.clone(), "effective_mse": 0.0,
                "pre_clip_mse": 0.0, "max_abs_delta": 0.0,
                "clipped_fraction": 0.0, "gain": 0.0}
    lo, hi = 0.0, 1.0
    _, _, m_hi = released_at(hi)
    while m_hi < target_mse and hi < 1e4:
        hi *= 2.0
        _, _, m_hi = released_at(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, _, m = released_at(mid)
        if m < target_mse:
            lo = mid
        else:
            hi = mid
    gain = 0.5 * (lo + hi)
    out, scaled, mse = released_at(gain)
    applied = out - frame
    # Tolerance in pixel levels, not floating-point epsilon: (frame + scaled)
    # - frame does not return scaled exactly in float32, so a 1e-6 threshold
    # counts rounding noise as clipping and reports a clipped fraction of
    # around 0.6 on frames where nothing is clipped at all.
    clipped = (scaled - applied).abs() > 1e-2
    return {
        "frame": out,
        "effective_mse": mse,
        "pre_clip_mse": float(scaled.square().mean()),
        "max_abs_delta": float(applied.abs().max()),
        "clipped_fraction": float(clipped.float().mean()),
        "gain": gain,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval_backbone", default="resnet18",
                    help="the attacker; the optimiser never sees it")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"])
    ap.add_argument("--target_mse", type=float, default=15.68,
                    help="delivered MSE every condition is rescaled to")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--placements", nargs="+", default=["uniform", "edge"],
                    choices=list(PLACEMENTS))
    ap.add_argument("--sanitizer", default="none", choices=sorted(SANITIZERS),
                    help="attacker-side transform applied to the release")
    ap.add_argument("--eot_sanitizers", nargs="*", default=[],
                    help="optimise in expectation over these transforms")
    ap.add_argument("--eot_samples", type=int, default=2)
    ap.add_argument("--random_start", type=float, default=1.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    for name in list(args.eot_sanitizers) + [args.sanitizer]:
        if name not in SANITIZERS:
            raise SystemExit(f"unknown sanitizer {name!r}; "
                             f"available: {sorted(SANITIZERS)}")
    eot_ops = [SANITIZERS[n] for n in args.eot_sanitizers]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[factorial] device={device} conditions={args.conditions} "
          f"placements={args.placements} eot={args.eot_sanitizers or 'off'}",
          flush=True)

    records, gallery = load_manifest(args.manifest, args.root)
    queries = records if not args.limit else records[: args.limit]
    gallery_ids = sorted(gallery)
    resize_hw = (args.height, args.width)
    gallery_tensor = torch.stack(
        [load_image(gallery[g]["path"], resize_hw) for g in gallery_ids])
    place_of = {g: gallery[g]["place_id"] for g in gallery_ids}
    print(f"[factorial] {len(queries)} queries, {len(gallery_ids)} gallery",
          flush=True)

    names = [args.eval_backbone] + [b for b in args.surrogates
                                    if b != args.eval_backbone]
    embedders: Dict[str, object] = {}
    sizes: Dict[str, object] = {}
    gal_emb: Dict[str, torch.Tensor] = {}
    for b in names:
        cfg = RetrievalConfig(backbone=b,
                              input_size=default_input_size_for_backbone(b))
        e = make_default_embedder(cfg).eval().to(device)
        gal_emb[b] = embed_gallery_batched(cfg, e, gallery_tensor)
        embedders[b] = e
        sizes[b] = cfg.input_size
        torch.cuda.empty_cache()
        print(f"[factorial] gallery embedded with {b}", flush=True)

    surrogates = [b for b in names if b != args.eval_backbone]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["query_id"], r["condition"], r["placement"],
                          r["seed"]))
        print(f"[factorial] resuming, {len(done)} rows already present",
              flush=True)

    fields = ["query_id", "condition", "placement", "seed", "sanitizer",
              "correct_rank", "top5_hit", "top10_hit", "top1_place",
              "correct_place", "pre_clip_mse", "effective_mse", "psnr",
              "max_abs_delta", "clipped_fraction", "n_surrogates"]
    new = not out.is_file()
    fh = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new:
        writer.writeheader()

    ev = args.eval_backbone
    ev_gal = (gal_emb[ev] / gal_emb[ev].norm(dim=-1, keepdim=True)
              .clamp_min(1e-12)).to(device)

    for qi, rec in enumerate(queries, 1):
        qid = rec["query_id"]
        want = rec["place_id"]
        if not any(place_of[g] == want for g in gallery_ids):
            continue
        frame = load_image(rec["query_path"], resize_hw).unsqueeze(0).to(device)
        maps = {p: placement_weight(p, frame) for p in args.placements}
        for seed in args.seeds:
            todo = [(c, p) for c in args.conditions for p in args.placements
                    if (qid, c, p, str(seed)) not in done]
            if not todo:
                continue
            gen = torch.Generator(device="cpu").manual_seed(
                zlib.crc32(f"{qid}|{seed}".encode()) & 0x7FFFFFFF)
            base = None
            if any(c != "isotropic" for c, _ in todo):
                with torch.no_grad():
                    targets = [normalised_embedding(
                        embedders[b], frame, sizes[b]).detach()
                        for b in surrogates]
                with torch.enable_grad():
                    base = directional_delta(
                        frame, targets, [embedders[b] for b in surrogates],
                        [sizes[b] for b in surrogates], args.steps,
                        args.step_size, args.linf,
                        random_start=args.random_start, generator=gen,
                        eot_ops=eot_ops, eot_samples=args.eot_samples)
                if float(base.abs().max()) == 0.0:
                    print(f"[factorial] WARNING zero perturbation for "
                          f"{qid}/seed{seed}", flush=True)
            for condition, placement in todo:
                if condition == "isotropic":
                    delta = torch.randn(frame.shape, generator=gen).to(device)
                else:
                    delta = derive_delta(condition, base, gen)
                delta = delta * maps[placement]
                stats = release_with_stats(frame, delta, args.target_mse)
                released = SANITIZERS[args.sanitizer](stats["frame"])
                with torch.no_grad():
                    qe = normalised_embedding(embedders[ev], released,
                                              sizes[ev])
                    sims = ev_gal @ qe.flatten()
                    order = torch.argsort(sims, descending=True).tolist()
                    rank = next(i + 1 for i, j in enumerate(order)
                                if place_of[gallery_ids[j]] == want)
                    top1_place = place_of[gallery_ids[order[0]]]
                mse = stats["effective_mse"]
                writer.writerow({
                    "query_id": qid, "condition": condition,
                    "placement": placement, "seed": seed,
                    "sanitizer": args.sanitizer,
                    "correct_rank": rank,
                    "top5_hit": int(rank <= 5), "top10_hit": int(rank <= 10),
                    "top1_place": top1_place, "correct_place": want,
                    "pre_clip_mse": f"{stats['pre_clip_mse']:.6f}",
                    "effective_mse": f"{mse:.6f}",
                    "psnr": f"{10.0 * torch.log10(torch.tensor(255.0 ** 2 / max(mse, 1e-9))):.4f}",
                    "max_abs_delta": f"{stats['max_abs_delta']:.4f}",
                    "clipped_fraction": f"{stats['clipped_fraction']:.6f}",
                    "n_surrogates": len(surrogates),
                })
                fh.flush()
                os.fsync(fh.fileno())
        if qi % 25 == 0:
            print(f"[factorial] {qi}/{len(queries)} queries", flush=True)

    fh.close()
    print(f"[factorial] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
