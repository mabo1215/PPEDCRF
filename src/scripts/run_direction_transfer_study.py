"""Does directional alignment survive without white-box access?

The placement and operator studies establish that spatial allocation of a fixed
distortion budget buys almost nothing, while a perturbation aligned with the
attacker's own gradient drives retrieval to zero at the same budget. That
result is measured with white-box access, which a deployed sanitizer does not
have, so on its own it is a diagnosis rather than a defense.

This script asks the deployable version of the question. The perturbation
direction is optimised against a *surrogate* ensemble of embedders and then
evaluated against a held-out attacker the optimiser never saw. Three regimes
are compared at matched delivered distortion:

  white_box     direction from the evaluation backbone itself (upper bound)
  transfer_N    direction from N surrogate backbones, evaluation backbone held out
  isotropic     the operating-point control: same budget, no direction

Every condition is rescaled to deliver the same mean squared error on the same
frames, so the comparison isolates direction from budget. If transfer works,
the negative results imply a usable defense; if it does not, they imply that
attacker-agnostic mechanisms in this family are confined to the regime where
nothing works. Both outcomes are informative, which is why the experiment is
worth running either way.

Writes one row per query per condition, flushed immediately, and skips
completed rows on restart.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    build_gallery_embeddings,
    default_input_size_for_backbone,
    make_default_embedder,
    preprocess_for_embed,
)
from eval.sanitizers import SANITIZERS  # noqa: E402
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)


def embed_gallery_batched(cfg, embedder, images: torch.Tensor,
                          batch: int = 128) -> torch.Tensor:
    """Embed the gallery in chunks and keep the result on CPU.

    The shared helper embeds the whole gallery in one forward pass, which is
    fine for a light backbone but exhausts an 8 GB card on VGG16 at 2,000
    images. Chunking here rather than changing the shared helper keeps every
    previously published number reproducible from the same code path.
    """
    out = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch):
            out.append(build_gallery_embeddings(cfg, embedder,
                                                images[i:i + batch]).cpu())
    return torch.cat(out, dim=0)


def normalised_embedding(embedder, frame: torch.Tensor, input_size) -> torch.Tensor:
    """Unit-norm embedding of a [0,255] frame.

    Must use the same preprocessing as the gallery: the shared helper applies
    ImageNet normalisation after resizing, and embedding queries without it
    puts query and gallery in different colour spaces, which silently degrades
    every condition and misdirects the gradient the aligned perturbation
    follows.
    """
    x = preprocess_for_embed(frame, input_size)
    e = embedder(x)
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def directional_delta(
    frame: torch.Tensor,
    targets: Sequence[torch.Tensor],
    embedders: Sequence[object],
    input_sizes: Sequence[object],
    steps: int,
    step_size: float,
    linf: float,
) -> torch.Tensor:
    """Sign-gradient descent on summed similarity across a surrogate ensemble.

    Averaging the objective over several embedders is what turns a white-box
    perturbation into a transferable one: a direction that lowers similarity
    for every surrogate is more likely to lower it for an unseen attacker than
    one tuned to a single network's idiosyncrasies.
    """
    original = frame.detach().float()
    candidate = original.clone()
    for _ in range(max(1, steps)):
        candidate.requires_grad_(True)
        loss = torch.zeros((), device=frame.device)
        for emb, tgt, isz in zip(embedders, targets, input_sizes):
            q = normalised_embedding(emb, candidate, isz)
            t = tgt / tgt.norm().clamp_min(1e-12)
            loss = loss + (q.flatten() * t.flatten()).sum()
        grad, = torch.autograd.grad(loss, candidate)
        candidate = candidate - step_size * grad.sign()
        delta = (candidate - original).clamp(-linf, linf)
        candidate = (original + delta).clamp(0.0, 255.0).detach()
    return candidate - original


def release_at_mse(frame: torch.Tensor, delta: torch.Tensor, target_mse: float,
                   iters: int = 40) -> torch.Tensor:
    """Scale a perturbation so the released frame hits a prescribed MSE."""
    def at(g: float):
        out = (frame + g * delta).clamp(0.0, 255.0)
        return out, float((out - frame).square().mean())

    _, m1 = at(1.0)
    if m1 <= 1e-12:
        return frame.clone()
    lo, hi = 0.0, 1.0
    _, mhi = at(hi)
    while mhi < target_mse and hi < 1e4:
        hi *= 2.0
        _, mhi = at(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, m = at(mid)
        if m < target_mse:
            lo = mid
        else:
            hi = mid
    out, _ = at(0.5 * (lo + hi))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval_backbone", default="resnet18",
                    help="the attacker; never seen by the optimiser except in "
                         "the white_box condition")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"])
    ap.add_argument("--target_mse", type=float, default=15.68,
                    help="delivered MSE every condition is matched to; the "
                         "default is the mechanism's operating point")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sanitizer", default="none", choices=sorted(SANITIZERS),
                    help="G2 tier-1 adaptive adversary: preprocessing the "
                         "attacker applies to the received frame before "
                         "embedding it, e.g. jpeg75/jpeg50/blur/denoise. "
                         "'none' reproduces the original (non-adaptive) study.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transfer] device={device}", flush=True)

    records, gallery = load_manifest(args.manifest, args.root)
    queries = records if not args.limit else records[: args.limit]
    gallery_ids = sorted(gallery)
    resize_hw = (args.height, args.width)
    gallery_tensor = torch.stack(
        [load_image(gallery[g]["path"], resize_hw) for g in gallery_ids])
    place_of = {g: gallery[g]["place_id"] for g in gallery_ids}
    print(f"[transfer] {len(queries)} queries, {len(gallery_ids)} gallery",
          flush=True)

    names = [args.eval_backbone] + [b for b in args.surrogates
                                    if b != args.eval_backbone]
    embedders, sizes, gal_emb = {}, {}, {}
    for b in names:
        cfg = RetrievalConfig(backbone=b,
                              input_size=default_input_size_for_backbone(b))
        e = make_default_embedder(cfg).eval().to(device)
        embedders[b] = e
        sizes[b] = cfg.input_size
        gal_emb[b] = embed_gallery_batched(cfg, e, gallery_tensor)
        torch.cuda.empty_cache()
        print(f"[transfer] gallery embedded with {b}", flush=True)

    conditions = ["isotropic", "white_box"]
    surr = [b for b in names if b != args.eval_backbone]
    for k in range(1, len(surr) + 1):
        conditions.append(f"transfer_{k}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["query_id"], r["condition"], r["seed"]))
        print(f"[transfer] resuming, {len(done)} rows already present",
              flush=True)

    fields = ["query_id", "condition", "seed", "sanitizer", "correct_rank",
              "top1_place", "correct_place", "effective_mse", "psnr"]
    new = not out.is_file()
    fh = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    if new:
        writer.writeheader()

    ev = args.eval_backbone
    ev_gal = (gal_emb[ev] / gal_emb[ev].norm(dim=-1, keepdim=True)
              .clamp_min(1e-12)).to(device)

    for qi, rec in enumerate(queries, 1):
        qid = rec["query_id"]
        frame = load_image(rec["query_path"], resize_hw).unsqueeze(0).to(device)
        want = rec["place_id"]
        pos = [i for i, g in enumerate(gallery_ids) if place_of[g] == want]
        if not pos:
            continue
        for seed in args.seeds:
            for cond in conditions:
                if (qid, cond, str(seed)) in done:
                    continue
                if cond == "isotropic":
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    delta = torch.randn(frame.shape, generator=g).to(device)
                else:
                    if cond == "white_box":
                        use = [ev]
                    else:
                        use = surr[: int(cond.split("_")[1])]
                    tgts = [gal_emb[b][pos[0]].to(device) for b in use]
                    with torch.enable_grad():
                        delta = directional_delta(
                            frame, tgts, [embedders[b] for b in use],
                            [sizes[b] for b in use], args.steps,
                            args.step_size, args.linf)
                released = release_at_mse(frame, delta, args.target_mse)
                mse = float((released - frame).square().mean())
                sanitized = SANITIZERS[args.sanitizer](released)
                with torch.no_grad():
                    qe = normalised_embedding(embedders[ev], sanitized, sizes[ev])
                    sims = ev_gal @ qe.flatten()
                    order = torch.argsort(sims, descending=True)
                    rank = next(i + 1 for i, j in enumerate(order.tolist())
                                if place_of[gallery_ids[j]] == want)
                    top1_place = place_of[gallery_ids[int(order[0].item())]]
                writer.writerow({
                    "query_id": qid, "condition": cond, "seed": seed,
                    "sanitizer": args.sanitizer,
                    "correct_rank": rank, "top1_place": top1_place,
                    "correct_place": want, "effective_mse": f"{mse:.6f}",
                    "psnr": f"{10 * torch.log10(torch.tensor(255.0 ** 2 / max(mse, 1e-9))):.4f}",
                })
                fh.flush()
                os.fsync(fh.fileno())
        if qi % 25 == 0:
            print(f"[transfer] {qi}/{len(queries)} queries", flush=True)

    fh.close()
    print(f"[transfer] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
