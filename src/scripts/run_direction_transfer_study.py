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

Two optimisation objectives are available, and the difference between them is
what the defender is assumed to know:

  positive  push the released frame away from the embedding of its own correct
            gallery entry. This is the stronger direction, but it presumes the
            defender can identify the reference image the frame matches --
            which is the very fact the attacker is trying to recover.
  self      push the released frame away from *its own* clean embedding. This
            needs nothing but the frame in hand, so it is what a deployed
            sanitizer can actually compute.

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
import zlib
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
    random_start: float = 0.0,
    generator: torch.Generator | None = None,
    eot_ops: Sequence = (),
    eot_samples: int = 2,
) -> torch.Tensor:
    """Sign-gradient descent on summed similarity across a surrogate ensemble.

    Averaging the objective over several embedders is what turns a white-box
    perturbation into a transferable one: a direction that lowers similarity
    for every surrogate is more likely to lower it for an unseen attacker than
    one tuned to a single network's idiosyncrasies.

    `random_start` displaces the first iterate by a uniform perturbation of
    that magnitude in pixel units. It is mandatory for the gallery-free "self"
    objective and pointless for the gallery-targeted one. The reason is that
    the self objective steers the frame away from *its own* clean embedding,
    so the starting point is the objective's exact maximum, where the gradient
    is zero: without a random start the sign of that gradient is decided by
    floating-point noise, and for a sizeable fraction of frames it is exactly
    zero, leaving the frame unperturbed and silently entering the study as a
    "direction" condition that delivered no distortion at all. A random start
    is the standard remedy and makes the first step well defined.

    Passing `eot_ops` optimises the direction in expectation over those
    attacker-side transforms instead of against a frame that arrives
    untouched. Those transforms are not differentiable, so the backward pass
    treats each as the identity (the standard BPDA substitution): the forward
    value is the transformed frame, the gradient is taken with respect to the
    frame that produced it. Ops are sampled per step rather than all applied
    every step, the cheap unbiased estimator of the same expectation.
    """
    original = frame.detach().float()
    if random_start > 0:
        noise = torch.empty_like(original.cpu()).uniform_(
            -random_start, random_start, generator=generator)
        candidate = (original + noise.to(original.device)).clamp(0.0, 255.0)
    else:
        candidate = original.clone()
    for _ in range(max(1, steps)):
        candidate.requires_grad_(True)
        loss = torch.zeros((), device=frame.device)
        if eot_ops:
            # The transformed views depend only on the current iterate, so
            # they are built once per step and shared across the ensemble.
            # Computing them inside the embedder loop instead re-ran the same
            # OpenCV round-trip once per surrogate -- three to four times the
            # CPU work per step, which dominated the runtime of this path.
            picks = []
            for _ in range(max(1, eot_samples)):
                op = eot_ops[int(torch.randint(len(eot_ops), (1,),
                                               generator=generator).item())]
                # BPDA: forward through the real (non-differentiable)
                # transform, backward as if it were the identity.
                picks.append(candidate + (op(candidate) - candidate).detach())
        else:
            picks = [candidate]
        for emb, tgt, isz in zip(embedders, targets, input_sizes):
            for view in picks:
                q = normalised_embedding(emb, view, isz)
                t = tgt / tgt.norm().clamp_min(1e-12)
                loss = loss + (q.flatten() * t.flatten()).sum() / len(picks)
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
    ap.add_argument("--eval_sanitizers", nargs="*", default=[],
                    help="evaluate every released frame under each of these "
                         "attacker-side transforms, writing one row per "
                         "transform, instead of the single --sanitizer. The "
                         "perturbation is optimised and released once per "
                         "(query, condition, seed) and only the attacker's "
                         "embedding step is repeated, so a held-out-transform "
                         "study costs one optimisation pass plus one forward "
                         "pass per transform.")
    ap.add_argument("--eval_checkpoint", default="",
                    help="G2 tier-2 adaptive adversary: state_dict checkpoint "
                         "for the eval_backbone, e.g. from "
                         "finetune_adaptive_attacker.py. Loaded onto the "
                         "eval_backbone embedder before it embeds the gallery "
                         "or any query, so the white_box condition is "
                         "automatically re-optimised against the adapted "
                         "model. Surrogates are unaffected.")
    ap.add_argument("--query_id_file", default="",
                    help="optional JSON file with a list of query_ids to "
                         "restrict evaluation to, e.g. the held-out test "
                         "split written by finetune_adaptive_attacker.py so "
                         "fine-tuning and evaluation never share queries.")
    ap.add_argument("--objective", default="positive",
                    choices=("positive", "self"),
                    help="What the perturbation is optimised away from. "
                         "'positive' targets the query's correct gallery "
                         "embedding and therefore assumes the defender knows "
                         "which reference image the frame matches; 'self' "
                         "targets the frame's own clean embedding and needs "
                         "no gallery knowledge at all.")
    ap.add_argument("--random_start", type=float, default=None,
                    help="uniform random displacement (pixel units) applied "
                         "before the first sign-gradient step. Defaults to "
                         "0.0 for the 'positive' objective, which reproduces "
                         "every published run, and to 1.0 for 'self', where "
                         "the unperturbed frame is a stationary point of the "
                         "objective and a zero start yields no perturbation "
                         "at all on some frames.")
    ap.add_argument("--eot_sanitizers", nargs="*", default=[],
                    help="optimise the direction in expectation over these "
                         "attacker-side transforms (names from "
                         "eval/sanitizers.py, e.g. jpeg75 blur denoise). "
                         "Empty means the published behaviour: the direction "
                         "assumes the frame arrives untouched.")
    ap.add_argument("--eot_samples", type=int, default=2,
                    help="transforms sampled per optimisation step when "
                         "--eot_sanitizers is given.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.random_start is None:
        args.random_start = 1.0 if args.objective == "self" else 0.0
    for name in args.eot_sanitizers:
        if name not in SANITIZERS:
            raise SystemExit(f"unknown sanitizer for EOT: {name!r}; "
                             f"available: {sorted(SANITIZERS)}")
    eot_ops = [SANITIZERS[n] for n in args.eot_sanitizers]
    eval_sanitizers = list(args.eval_sanitizers) or [args.sanitizer]
    for name in eval_sanitizers:
        if name not in SANITIZERS:
            raise SystemExit(f"unknown evaluation sanitizer: {name!r}; "
                             f"available: {sorted(SANITIZERS)}")
    print(f"[transfer] objective={args.objective} "
          f"random_start={args.random_start} "
          f"eot={args.eot_sanitizers or 'off'} "
          f"eval_sanitizers={eval_sanitizers}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[transfer] device={device}", flush=True)

    records, gallery = load_manifest(args.manifest, args.root)
    queries = records if not args.limit else records[: args.limit]
    if args.query_id_file:
        with open(args.query_id_file, encoding="utf-8") as fh:
            keep = set(json.load(fh))
        queries = [r for r in queries if r["query_id"] in keep]
        print(f"[transfer] restricted to {len(queries)} queries from "
              f"{args.query_id_file}", flush=True)
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
        if b == args.eval_backbone and args.eval_checkpoint:
            # The gallery stays indexed with the attacker's original (stock)
            # model: re-embedding a whole reference database every time a
            # query encoder is adapted is not something a real deployed
            # system does, and it is what finetune_adaptive_attacker.py
            # trained against (fixed pretrained-model targets). Only the
            # query-side encoder -- used below for every condition's ranking
            # and as the white_box gradient target -- is replaced.
            gal_emb[b] = embed_gallery_batched(cfg, e, gallery_tensor)
            adapted = make_default_embedder(cfg).eval().to(device)
            state = torch.load(args.eval_checkpoint, map_location=device)
            adapted.load_state_dict(state, strict=True)
            adapted.eval()
            e = adapted
            print(f"[transfer] loaded adapted checkpoint for {b} from "
                  f"{args.eval_checkpoint} (gallery stays stock-indexed)",
                  flush=True)
        else:
            gal_emb[b] = embed_gallery_batched(cfg, e, gallery_tensor)
        embedders[b] = e
        sizes[b] = cfg.input_size
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
                done.add((r["query_id"], r["condition"], r["seed"],
                          r.get("sanitizer", "none")))
        print(f"[transfer] resuming, {len(done)} rows already present",
              flush=True)

    fields = ["query_id", "condition", "seed", "sanitizer", "objective",
              "correct_rank", "top1_place", "correct_place", "effective_mse",
              "psnr"]
    new = not out.is_file()
    if not new:
        # An export written before the objective column existed must keep its
        # own header, or appended rows would be shifted by one column against
        # it. Resuming such a file is still safe; it just stays on the old
        # schema, and its objective is implicitly "positive".
        with open(out, newline="", encoding="utf-8") as probe:
            existing = next(csv.reader(probe), None)
        if existing:
            fields = existing
    fh = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new:
        writer.writeheader()

    ev = args.eval_backbone
    ev_gal = (gal_emb[ev] / gal_emb[ev].norm(dim=-1, keepdim=True)
              .clamp_min(1e-12)).to(device)
    zero_delta = 0

    for qi, rec in enumerate(queries, 1):
        qid = rec["query_id"]
        frame = load_image(rec["query_path"], resize_hw).unsqueeze(0).to(device)
        want = rec["place_id"]
        pos = [i for i, g in enumerate(gallery_ids) if place_of[g] == want]
        if not pos:
            continue
        for seed in args.seeds:
            for cond in conditions:
                pending = [name for name in eval_sanitizers
                           if (qid, cond, str(seed), name) not in done]
                if not pending:
                    continue
                if cond == "isotropic":
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    delta = torch.randn(frame.shape, generator=g).to(device)
                else:
                    if cond == "white_box":
                        use = [ev]
                    else:
                        use = surr[: int(cond.split("_")[1])]
                    if args.objective == "self":
                        # Gallery-free: the only thing the perturbation is
                        # steered away from is the frame's own clean
                        # embedding under each surrogate, so nothing about
                        # the reference database is required.
                        with torch.no_grad():
                            tgts = [normalised_embedding(
                                embedders[b], frame, sizes[b]).detach()
                                for b in use]
                    else:
                        tgts = [gal_emb[b][pos[0]].to(device) for b in use]
                    # zlib.crc32, not hash(): Python randomises string
                    # hashing per process, which would make the random start
                    # -- and therefore the whole run -- irreproducible.
                    gstart = torch.Generator(device="cpu").manual_seed(
                        zlib.crc32(f"{qid}|{cond}|{seed}".encode()) & 0x7FFFFFFF)
                    with torch.enable_grad():
                        delta = directional_delta(
                            frame, tgts, [embedders[b] for b in use],
                            [sizes[b] for b in use], args.steps,
                            args.step_size, args.linf,
                            random_start=args.random_start,
                            generator=gstart, eot_ops=eot_ops,
                            eot_samples=args.eot_samples)
                    if float(delta.abs().max()) == 0.0:
                        zero_delta += 1
                        print(f"[transfer] WARNING zero perturbation for "
                              f"{qid}/{cond}/seed{seed}", flush=True)
                released = release_at_mse(frame, delta, args.target_mse)
                mse = float((released - frame).square().mean())
                # One released frame, several attackers' preprocessing: the
                # perturbation above is the expensive part and is shared.
                for name in pending:
                    sanitized = SANITIZERS[name](released)
                    with torch.no_grad():
                        qe = normalised_embedding(embedders[ev], sanitized,
                                                  sizes[ev])
                        sims = ev_gal @ qe.flatten()
                        order = torch.argsort(sims, descending=True)
                        rank = next(i + 1 for i, j in enumerate(order.tolist())
                                    if place_of[gallery_ids[j]] == want)
                        top1_place = place_of[gallery_ids[int(order[0].item())]]
                    writer.writerow({
                        "query_id": qid, "condition": cond, "seed": seed,
                        "sanitizer": name,
                        "objective": args.objective,
                        "correct_rank": rank, "top1_place": top1_place,
                        "correct_place": want, "effective_mse": f"{mse:.6f}",
                        "psnr": f"{10 * torch.log10(torch.tensor(255.0 ** 2 / max(mse, 1e-9))):.4f}",
                    })
                    fh.flush()
                    os.fsync(fh.fileno())
        if qi % 25 == 0:
            print(f"[transfer] {qi}/{len(queries)} queries", flush=True)

    fh.close()
    if zero_delta:
        print(f"[transfer] WARNING {zero_delta} conditions produced a zero "
              f"perturbation and delivered no distortion", flush=True)
    print(f"[transfer] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
