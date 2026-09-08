"""The closest published approach, run under this paper's protocol.

Le et al., "Rethinking Adversarial Examples for Location Privacy Protection",
combine a spatial mask with multi-model projected-gradient perturbation and
evaluate transfer to an unseen recogniser. That is the method this manuscript
has to be compared against rather than argued past: it already couples
allocation with adversarial direction, which is exactly the pairing the
manuscript treats as unexplored.

Their task is scene/landmark recognition and ours is gallery ranking, so this
is an adaptation, not a reimplementation. What is held identical to our own
direction arm is everything the comparison depends on: the surrogate ensemble,
the step count and step size, the projection radius, the random start, the
delivered distortion after clipping, and the held-out attacker. What differs
is the one thing under test -- whether the perturbation is confined to a mask.

Three arms:

  maskguided_pgd  perturbation confined to a mask covering the most
                  informative fraction of the frame
  fullframe_pgd   the same optimiser with no mask (this paper's direction)
  isotropic       the budget with no direction at all

The mask source is selectable. `cam` is the closest available analogue of the
class-activation masks of the original work: the surrogate ensemble's own
gradient magnitude, thresholded at a coverage fraction. `edge` and `saliency`
are the priors already used in the placement study, kept so the baseline can
be run without assuming the CAM analogue is the faithful choice.

Writes one row per query per arm, flushed immediately, resumable.
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
from scripts.run_direction_factorial import release_with_stats  # noqa: E402
from scripts.run_direction_transfer_study import (  # noqa: E402
    embed_gallery_batched,
    normalised_embedding,
)
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)
from scripts.run_placement_rule_study import edge_map, saliency_map  # noqa: E402

ARMS = ("maskguided_pgd", "fullframe_pgd", "isotropic")
MASK_SOURCES = ("cam", "edge", "saliency")


def gradient_saliency(frame: torch.Tensor, embedders: Sequence[object],
                      targets: Sequence[torch.Tensor],
                      input_sizes: Sequence[object]) -> torch.Tensor:
    """Per-pixel gradient magnitude of the surrogate ensemble's own score.

    This stands in for the class-activation mask of the original work: both
    answer "which pixels does the recogniser rely on", one through activations
    and one through gradients. The substitution is recorded in the paper
    rather than presented as the same construction.
    """
    x = frame.detach().clone().requires_grad_(True)
    score = torch.zeros((), device=frame.device)
    for emb, tgt, isz in zip(embedders, targets, input_sizes):
        q = normalised_embedding(emb, x, isz)
        t = tgt / tgt.norm().clamp_min(1e-12)
        score = score + (q.flatten() * t.flatten()).sum()
    grad, = torch.autograd.grad(score, x)
    return grad.abs().sum(dim=1, keepdim=True)


def coverage_mask(raw: torch.Tensor, coverage: float) -> torch.Tensor:
    """Binary mask over the highest-valued `coverage` fraction of pixels."""
    flat = raw.flatten()
    k = max(1, int(round(coverage * flat.numel())))
    threshold = flat.sort(descending=True).values[k - 1]
    return (raw >= threshold).float()


def masked_directional_delta(
    frame: torch.Tensor,
    targets: Sequence[torch.Tensor],
    embedders: Sequence[object],
    input_sizes: Sequence[object],
    mask: torch.Tensor | None,
    steps: int,
    step_size: float,
    linf: float,
    random_start: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sign-gradient descent confined to a mask.

    Identical to the unmasked optimiser except that every update and the
    final perturbation are multiplied by the mask, so the two arms differ in
    where the budget may go and in nothing else.
    """
    original = frame.detach().float()
    m = torch.ones_like(original[:, :1]) if mask is None else mask
    if random_start > 0:
        noise = torch.empty_like(original.cpu()).uniform_(
            -random_start, random_start, generator=generator)
        candidate = (original + noise.to(original.device) * m).clamp(0.0, 255.0)
    else:
        candidate = original.clone()
    for _ in range(max(1, steps)):
        candidate.requires_grad_(True)
        loss = torch.zeros((), device=frame.device)
        for emb, tgt, isz in zip(embedders, targets, input_sizes):
            q = normalised_embedding(emb, candidate, isz)
            t = tgt / tgt.norm().clamp_min(1e-12)
            loss = loss + (q.flatten() * t.flatten()).sum()
        grad, = torch.autograd.grad(loss, candidate)
        candidate = candidate - step_size * grad.sign() * m
        delta = (candidate - original).clamp(-linf, linf) * m
        candidate = (original + delta).clamp(0.0, 255.0).detach()
    return candidate - original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval_backbone", default="resnet18")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--mask_source", default="cam", choices=list(MASK_SOURCES))
    ap.add_argument("--mask_coverage", type=float, default=0.25,
                    help="fraction of pixels the mask admits")
    ap.add_argument("--target_mse", type=float, default=15.68)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--random_start", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sanitizer", default="none", choices=sorted(SANITIZERS))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not 0.0 < args.mask_coverage <= 1.0:
        raise SystemExit("--mask_coverage must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[maskpgd] device={device} arms={args.arms} "
          f"mask={args.mask_source}@{args.mask_coverage}", flush=True)

    records, gallery = load_manifest(args.manifest, args.root)
    queries = records if not args.limit else records[: args.limit]
    gallery_ids = sorted(gallery)
    resize_hw = (args.height, args.width)
    gallery_tensor = torch.stack(
        [load_image(gallery[g]["path"], resize_hw) for g in gallery_ids])
    place_of = {g: gallery[g]["place_id"] for g in gallery_ids}
    print(f"[maskpgd] {len(queries)} queries, {len(gallery_ids)} gallery",
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
        print(f"[maskpgd] gallery embedded with {b}", flush=True)

    surrogates = [b for b in names if b != args.eval_backbone]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["query_id"], r["arm"], r["seed"]))
        print(f"[maskpgd] resuming, {len(done)} rows already present",
              flush=True)

    fields = ["query_id", "arm", "seed", "mask_source", "mask_coverage",
              "sanitizer", "correct_rank", "top5_hit", "top10_hit",
              "top1_place", "correct_place", "pre_clip_mse", "effective_mse",
              "psnr", "max_abs_delta", "clipped_fraction", "n_surrogates"]
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
        for seed in args.seeds:
            todo = [a for a in args.arms if (qid, a, str(seed)) not in done]
            if not todo:
                continue
            gen = torch.Generator(device="cpu").manual_seed(
                zlib.crc32(f"{qid}|{seed}".encode()) & 0x7FFFFFFF)
            with torch.no_grad():
                targets = [normalised_embedding(
                    embedders[b], frame, sizes[b]).detach()
                    for b in surrogates]
            mask = None
            if "maskguided_pgd" in todo:
                if args.mask_source == "cam":
                    with torch.enable_grad():
                        raw = gradient_saliency(
                            frame, [embedders[b] for b in surrogates],
                            targets, [sizes[b] for b in surrogates])
                elif args.mask_source == "edge":
                    raw = edge_map(frame)
                else:
                    raw = saliency_map(frame)
                mask = coverage_mask(raw, args.mask_coverage)
            for arm in todo:
                if arm == "isotropic":
                    delta = torch.randn(frame.shape, generator=gen).to(device)
                else:
                    use_mask = mask if arm == "maskguided_pgd" else None
                    with torch.enable_grad():
                        delta = masked_directional_delta(
                            frame, targets, [embedders[b] for b in surrogates],
                            [sizes[b] for b in surrogates], use_mask,
                            args.steps, args.step_size, args.linf,
                            args.random_start, gen)
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
                    "query_id": qid, "arm": arm, "seed": seed,
                    "mask_source": args.mask_source if arm == "maskguided_pgd"
                    else "",
                    "mask_coverage": args.mask_coverage
                    if arm == "maskguided_pgd" else "",
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
            print(f"[maskpgd] {qi}/{len(queries)} queries", flush=True)

    fh.close()
    print(f"[maskpgd] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
