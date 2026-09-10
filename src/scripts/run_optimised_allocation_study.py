"""Give the allocation axis the search budget the direction axis gets.

Every placement rule the audit tests so far is *prescribed*: the mechanism's
own map, a uniform control, spectral-residual saliency, Sobel edge magnitude,
a centre-bias prior, a fixed random field, a one-step score gradient and its
inverse, three published segmentation priors, and the margin gradient. Not one
of them is optimised against anything. The direction axis, by contrast, is the
output of twenty sign-gradient steps against a surrogate ensemble. A reader is
therefore entitled to ask whether the paper's central contrast is between the
two axes or between an optimised variable and an unoptimised one.

This script closes that gap. It solves for the placement map directly:

    minimise   sum_k < f_k( release(x, w) ), t_k >
    subject to mean(w^2) = 1,  w >= 0

where `w` is a single-channel non-negative spatial map broadcast over colour,
`release(x, w) = clamp(x + g * w (*) eps, 0, 255)` with `eps` a fixed standard
normal field drawn once per (query, seed), and `g` solved per frame by
bisection so the released frame delivers exactly the target MSE. The noise
field is shared between the optimised arm and the uniform control, so the only
thing that differs between them is where the budget sits.

Three points about the formulation, each chosen to be generous to allocation:

  * The optimiser sees the *realised* noise draw, not its distribution. A
    deployed sanitizer draws that noise itself, so this is not an unfair
    advantage -- it is the strongest allocation the mechanism could actually
    ship.
  * `opt_whitebox` optimises `w` against the evaluation backbone itself. No
    deployable mechanism has that, and it is included precisely because a null
    under it is decisive in a way a transfer null is not.
  * The optimiser is Adam on an unconstrained parameterisation, run for at
    least the direction arm's step budget and reported again at double it, so
    "you did not search hard enough" can be checked rather than argued.

The objective's value is recorded before and after optimisation for every row.
That is the control that matters: if the surrogate similarity falls sharply
while retrieval does not move, the null is about what allocation can buy and
not about whether the optimiser worked.

Queries are optimised in batches so the card is used rather than idled by a
per-query Python loop. Rows are written one per completed (query, condition,
seed), flushed and fsynced, and completed rows are skipped on restart.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
)
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)
from scripts.run_direction_transfer_study import (  # noqa: E402
    embed_gallery_batched,
    normalised_embedding,
)


def normalise_weights(w: torch.Tensor) -> torch.Tensor:
    """Scale each map in the batch to unit mean square, the study's gate.

    The energy-conservation gate the placement family uses fixes sum_i w_i^2.
    Expressed per pixel that is mean(w^2) = 1, which is also the scale the
    operator study's weight maps carry, so an optimised map released through
    the same path is directly comparable with the prescribed ones.
    """
    flat = w.flatten(1)
    rms = flat.square().mean(dim=1).clamp_min(1e-12).sqrt()
    return w / rms.view(-1, *([1] * (w.dim() - 1)))


def batched_gain_for_mse(frames: torch.Tensor, deltas: torch.Tensor,
                         target_mse: float, iters: int = 40) -> torch.Tensor:
    """Per-frame gain g such that clamp(x + g*delta) delivers `target_mse`.

    Vectorised over the batch. The clamp is inside the measurement, so
    delivered MSE is sublinear in g and the square-law scaling that matching a
    nominal sigma assumes does not hold -- which is the whole reason the study
    matches delivered rather than nominal distortion.
    """
    b = frames.size(0)
    dev = frames.device

    def mse_at(g: torch.Tensor) -> torch.Tensor:
        out = (frames + g.view(b, 1, 1, 1) * deltas).clamp(0.0, 255.0)
        return (out - frames).square().flatten(1).mean(dim=1)

    lo = torch.zeros(b, device=dev)
    hi = torch.ones(b, device=dev)
    for _ in range(24):
        need = mse_at(hi) < target_mse
        if not bool(need.any()):
            break
        hi = torch.where(need, hi * 2.0, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = mse_at(mid) < target_mse
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def release_with_weights(frames: torch.Tensor, weights: torch.Tensor,
                         eps: torch.Tensor, target_mse: float,
                         gain: torch.Tensor | None = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Released frames at matched delivered MSE, plus the gain used.

    When `gain` is supplied the release is differentiable in `weights` with the
    gain held constant; that is what the optimiser steps through. When it is
    not, the gain is solved for the weights as given, which is what the
    evaluation path uses.
    """
    delta = weights * eps
    if gain is None:
        gain = batched_gain_for_mse(frames, delta.detach(), target_mse)
    out = (frames + gain.view(-1, 1, 1, 1) * delta).clamp(0.0, 255.0)
    return out, gain


def surrogate_similarity(released: torch.Tensor,
                         embedders: Sequence[torch.nn.Module],
                         sizes: Sequence[Tuple[int, int]],
                         targets: Sequence[torch.Tensor]) -> torch.Tensor:
    """Per-sample sum over the ensemble of cosine similarity to its target.

    This is the quantity the direction arm minimises, so optimising a
    placement against it is the like-for-like comparison: same objective, same
    surrogates, same budget, different variable.
    """
    total = None
    for emb, size, tgt in zip(embedders, sizes, targets):
        q = normalised_embedding(emb, released, size)
        sim = (q * tgt).sum(dim=-1)
        total = sim if total is None else total + sim
    return total


def optimise_weights(frames: torch.Tensor, eps: torch.Tensor,
                     embedders: Sequence[torch.nn.Module],
                     sizes: Sequence[Tuple[int, int]],
                     targets: Sequence[torch.Tensor],
                     target_mse: float, steps: int, lr: float,
                     checkpoints: Sequence[int]
                     ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Adam on softplus(theta), projected to the energy gate after every step.

    Returns the normalised weight map and the objective value at each requested
    step count, so a run reports the matched-budget answer and the
    double-budget answer from one optimisation rather than two.
    """
    b = frames.size(0)
    # softplus(theta) is positive by construction, so the non-negativity
    # constraint never needs a projection that could stall the optimiser at
    # the boundary; the energy gate is applied explicitly after each step.
    theta = torch.zeros(b, 1, frames.size(2), frames.size(3),
                        device=frames.device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    want = sorted(set(int(c) for c in checkpoints))
    maps: Dict[int, torch.Tensor] = {}
    losses: Dict[int, torch.Tensor] = {}

    def current() -> torch.Tensor:
        return normalise_weights(torch.nn.functional.softplus(theta))

    with torch.no_grad():
        w0 = current()
        rel0, _ = release_with_weights(frames, w0, eps, target_mse)
        losses[0] = surrogate_similarity(rel0, embedders, sizes, targets).detach()
        if 0 in want:
            maps[0] = w0.detach()

    for step in range(1, max(want) + 1):
        w = current()
        with torch.no_grad():
            gain = batched_gain_for_mse(frames, (w * eps).detach(), target_mse)
        released, _ = release_with_weights(frames, w, eps, target_mse, gain=gain)
        sim = surrogate_similarity(released, embedders, sizes, targets)
        loss = sim.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in want:
            with torch.no_grad():
                wk = current().detach()
                relk, _ = release_with_weights(frames, wk, eps, target_mse)
                maps[step] = wk
                losses[step] = surrogate_similarity(
                    relk, embedders, sizes, targets).detach()
    return maps, losses


def top_decile_share(w: torch.Tensor) -> torch.Tensor:
    """Squared weight carried by the largest tenth of the pixels.

    The manuscript reports this for every prescribed placement (0.100 for
    uniform, 0.839 for edge), so an optimised map has to be reported on the
    same scale to be comparable with them.
    """
    flat = w.flatten(1).square()
    k = max(1, int(0.10 * flat.size(1)))
    top = flat.topk(k, dim=1).values.sum(dim=1)
    return top / flat.sum(dim=1).clamp_min(1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval_backbone", default="resnet18")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"])
    ap.add_argument("--conditions", nargs="+",
                    default=["uniform", "opt_transfer", "opt_whitebox"],
                    help="uniform is the energy-matched control every "
                         "optimised arm is paired against on the same noise "
                         "draw; opt_transfer optimises the map against the "
                         "surrogate ensemble; opt_whitebox optimises it "
                         "against the evaluation backbone itself.")
    ap.add_argument("--target_mse", type=float, default=15.68)
    ap.add_argument("--steps", type=int, default=20,
                    help="the direction arm's budget, and the headline number")
    ap.add_argument("--double_steps", type=int, default=40,
                    help="reported alongside so 'the search was too short' is "
                         "checkable; 0 disables the second checkpoint")
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 5678, 9012])
    ap.add_argument("--batch", type=int, default=8,
                    help="queries optimised together; the card, not the "
                         "Python loop, should be the constraint")
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gallery_batch", type=int, default=128)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[alloc] device={device} steps={args.steps} "
          f"double={args.double_steps} lr={args.lr} batch={args.batch}",
          flush=True)

    records, gallery = load_manifest(args.manifest, args.root)
    queries = records if not args.limit else records[: args.limit]
    gallery_ids = sorted(gallery)
    resize_hw = (args.height, args.width)
    gallery_tensor = torch.stack(
        [load_image(gallery[g]["path"], resize_hw) for g in gallery_ids])
    place_of = {g: gallery[g]["place_id"] for g in gallery_ids}
    print(f"[alloc] {len(queries)} queries, {len(gallery_ids)} gallery",
          flush=True)

    names = [args.eval_backbone] + [b for b in args.surrogates
                                    if b != args.eval_backbone]
    embedders, sizes = {}, {}
    ev_gal = None
    for b in names:
        cfg = RetrievalConfig(backbone=b,
                              input_size=default_input_size_for_backbone(b))
        e = make_default_embedder(cfg).eval().to(device)
        for p in e.parameters():
            p.requires_grad_(False)
        if b == args.eval_backbone:
            g = embed_gallery_batched(cfg, e, gallery_tensor,
                                      batch=args.gallery_batch)
            ev_gal = (g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(device)
        embedders[b] = e
        sizes[b] = cfg.input_size
        torch.cuda.empty_cache()
        print(f"[alloc] embedder ready: {b}", flush=True)

    surr = [b for b in names if b != args.eval_backbone]
    checkpoints = [args.steps] + ([args.double_steps]
                                  if args.double_steps > args.steps else [])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["query_id"], r["condition"], r["seed"]))
        print(f"[alloc] resuming, {len(done)} rows already present", flush=True)

    fields = ["query_id", "condition", "seed", "opt_steps", "optimised_against",
              "correct_rank", "top1_place", "correct_place", "effective_mse",
              "psnr", "weight_top10pct_share", "weight_mean_square",
              "surrogate_sim_start", "surrogate_sim_end"]
    new = not out.is_file()
    fh = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new:
        writer.writeheader()

    def condition_rows(cond: str) -> List[str]:
        """Row labels a condition writes: one per optimisation checkpoint."""
        if cond == "uniform":
            return ["uniform"]
        return [cond if s == args.steps else f"{cond}_x2" for s in checkpoints]

    # A query with no gallery item of its own place has no rank to report, so
    # it is dropped once here rather than checked inside the batch loop.
    gallery_places = {place_of[g] for g in gallery_ids}
    dropped = [r for r in queries if r["place_id"] not in gallery_places]
    if dropped:
        print(f"[alloc] dropping {len(dropped)} queries with no gallery "
              f"positive", flush=True)
    queries = [r for r in queries if r["place_id"] in gallery_places]

    batches: List[List[dict]] = [queries[i:i + args.batch]
                                 for i in range(0, len(queries), args.batch)]
    for bi, group in enumerate(batches, 1):
        keep = group
        if not keep:
            continue
        frames = torch.stack([load_image(r["query_path"], resize_hw)
                              for r in keep]).to(device)
        for seed in args.seeds:
            eps = torch.stack([
                torch.randn(frames.shape[1:], generator=torch.Generator(
                    device="cpu").manual_seed(
                        zlib.crc32(f"{r['query_id']}|{seed}".encode())
                        & 0x7FFFFFFF))
                for r in keep]).to(device)
            for cond in args.conditions:
                labels = condition_rows(cond)
                pending = [lab for lab in labels
                           if any((r["query_id"], lab, str(seed)) not in done
                                  for r in keep)]
                if not pending:
                    continue
                if cond == "uniform":
                    w = normalise_weights(torch.ones_like(frames[:, :1]))
                    maps = {args.steps: w}
                    losses = {}
                    with torch.no_grad():
                        rel, _ = release_with_weights(frames, w, eps,
                                                      args.target_mse)
                        s = surrogate_similarity(
                            rel, [embedders[b] for b in surr],
                            [sizes[b] for b in surr],
                            [normalised_embedding(embedders[b], frames,
                                                  sizes[b]).detach()
                             for b in surr])
                    losses = {0: s, args.steps: s}
                    against = "none"
                    step_of = {"uniform": args.steps}
                else:
                    use = surr if cond == "opt_transfer" else [args.eval_backbone]
                    against = "+".join(use)
                    with torch.no_grad():
                        targets = [normalised_embedding(embedders[b], frames,
                                                        sizes[b]).detach()
                                   for b in use]
                    maps, losses = optimise_weights(
                        frames, eps, [embedders[b] for b in use],
                        [sizes[b] for b in use], targets, args.target_mse,
                        args.steps, args.lr, checkpoints)
                    step_of = {lab: (args.steps if not lab.endswith("_x2")
                                     else args.double_steps)
                               for lab in labels}

                for lab in labels:
                    nsteps = step_of[lab]
                    w = maps[nsteps]
                    with torch.no_grad():
                        released, _ = release_with_weights(frames, w, eps,
                                                           args.target_mse)
                        mse = (released - frames).square().flatten(1).mean(dim=1)
                        share = top_decile_share(w)
                        wms = w.flatten(1).square().mean(dim=1)
                        qe = normalised_embedding(embedders[args.eval_backbone],
                                                  released,
                                                  sizes[args.eval_backbone])
                        sims = qe @ ev_gal.t()
                        order = torch.argsort(sims, dim=1, descending=True)
                    for j, rec in enumerate(keep):
                        key = (rec["query_id"], lab, str(seed))
                        if key in done:
                            continue
                        want = rec["place_id"]
                        ranked = order[j].tolist()
                        rank = next((i + 1 for i, gi in enumerate(ranked)
                                     if place_of[gallery_ids[gi]] == want),
                                    None)
                        if rank is None:
                            continue
                        m = float(mse[j])
                        writer.writerow({
                            "query_id": rec["query_id"], "condition": lab,
                            "seed": seed, "opt_steps": nsteps,
                            "optimised_against": against,
                            "correct_rank": rank,
                            "top1_place": place_of[gallery_ids[ranked[0]]],
                            "correct_place": want,
                            "effective_mse": f"{m:.6f}",
                            "psnr": f"{10 * torch.log10(torch.tensor(255.0 ** 2 / max(m, 1e-9))):.4f}",
                            "weight_top10pct_share": f"{float(share[j]):.6f}",
                            "weight_mean_square": f"{float(wms[j]):.6f}",
                            "surrogate_sim_start": f"{float(losses[0][j]):.6f}",
                            "surrogate_sim_end": f"{float(losses[nsteps][j]):.6f}",
                        })
                        done.add(key)
                    fh.flush()
                    os.fsync(fh.fileno())
        if bi % 5 == 0:
            print(f"[alloc] {bi}/{len(batches)} batches", flush=True)

    fh.close()
    print(f"[alloc] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
