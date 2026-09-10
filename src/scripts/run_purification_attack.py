"""An attacker that removes the perturbation instead of adapting to it.

The preprocessing study tests four fixed, cheap transforms an attacker might
apply for reasons unrelated to this defense, and the EOT arm hardens against
them. Neither bounds the attack a reader of this paper would try next:
purification. An attacker who can collect examples of what the mechanism
releases, alongside the frames they came from, can train an inverse -- a
denoiser mapping released to clean -- and embed its output. That is stronger
than any fixed transform for the same reason the adaptive attacker is stronger
than a generic one: the learned inverse sees the release distribution, where a
blur sees a generic corruption.

This script measures it. The attacker's training data comes from the same
place-disjoint split the adaptive-attacker study uses, so the 200 evaluation
queries are ones the purifier never saw and share no place with anything it
did. Three exposures are purified -- the operating-point isotropic control, the
directional release, and the EOT-hardened release actually shipped -- each with
a purifier trained on that exposure, which is the attacker that knows what it
is facing. One cross-exposure cell asks whether it has to: the purifier trained
on the unhardened direction, applied to the hardened release.

The purifier is a residual CNN of the DnCNN family trained to reconstruct the
clean frame under mean absolute error. Reconstruction is a proxy for the
attacker's real objective, which is retrieval; it is the standard purification
baseline and it is what an attacker without the retrieval labels can optimise.

Releases are cached one file per (query, condition, draw) and the run resumes
from whatever is on disk, so an interrupted job continues rather than
regenerating.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
)
from eval.sanitizers import SANITIZERS  # noqa: E402
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)
from scripts.run_direction_transfer_study import (  # noqa: E402
    directional_delta,
    embed_gallery_batched,
    normalised_embedding,
    release_at_mse,
)
from scripts.finetune_adaptive_attacker import split_queries  # noqa: E402

CONDITIONS = ("isotropic", "direction", "hardened")


class Purifier(nn.Module):
    """Residual denoiser: predicts what to subtract, not what to output.

    The residual form is what makes a small network competitive here -- the
    released frame is the clean frame plus a bounded perturbation, so the
    quantity to learn is the perturbation and the identity is free.
    """

    def __init__(self, channels: int = 64, depth: int = 12):
        super().__init__()
        layers: List[nn.Module] = [nn.Conv2d(3, channels, 3, padding=1),
                                   nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                       nn.BatchNorm2d(channels), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.net(x)).clamp(0.0, 1.0)


def cache_path(root: Path, condition: str, query_id: str, draw: int) -> Path:
    return root / condition / f"{query_id}.d{draw}.pt"


def release(frame: torch.Tensor, condition: str, qid: str, draw: int,
            args, embedders, sizes, surrogates, eot_ops) -> torch.Tensor:
    """One released realisation of one frame under one condition."""
    if condition == "isotropic":
        gen = torch.Generator(device="cpu").manual_seed(
            zlib.crc32(f"{qid}|{draw}".encode()) & 0x7FFFFFFF)
        delta = torch.randn(frame.shape, generator=gen).to(frame.device)
    else:
        with torch.no_grad():
            targets = [normalised_embedding(embedders[b], frame,
                                            sizes[b]).detach()
                       for b in surrogates]
        gen = torch.Generator(device="cpu").manual_seed(
            zlib.crc32(f"{qid}|{condition}|{draw}".encode()) & 0x7FFFFFFF)
        with torch.enable_grad():
            delta = directional_delta(
                frame, targets, [embedders[b] for b in surrogates],
                [sizes[b] for b in surrogates], args.steps, args.step_size,
                args.linf, random_start=1.0, generator=gen,
                eot_ops=eot_ops if condition == "hardened" else (),
                eot_samples=args.eot_samples)
    return release_at_mse(frame, delta, args.target_mse)


def build_releases(records, conditions, draws, cache_root: Path, args,
                   embedders, sizes, surrogates, eot_ops, resize_hw, device):
    """Cache every release this run needs, skipping what is already there."""
    for condition in conditions:
        (cache_root / condition).mkdir(parents=True, exist_ok=True)
        todo = [(r, d) for r in records for d in range(draws)
                if not cache_path(cache_root, condition, r["query_id"],
                                  d).is_file()]
        if not todo:
            print(f"[purify] {condition}: {len(records)}x{draws} releases "
                  f"already cached", flush=True)
            continue
        print(f"[purify] {condition}: building {len(todo)} releases",
              flush=True)
        for i, (rec, draw) in enumerate(todo, 1):
            frame = load_image(rec["query_path"], resize_hw).unsqueeze(0).to(device)
            out = release(frame, condition, rec["query_id"], draw, args,
                          embedders, sizes, surrogates, eot_ops)
            torch.save(out.squeeze(0).cpu(), cache_path(
                cache_root, condition, rec["query_id"], draw))
            if i % 50 == 0:
                print(f"[purify]   {i}/{len(todo)}", flush=True)


def train_purifier(records, condition: str, draws: int, cache_root: Path,
                   resize_hw, device, args) -> Purifier:
    """Train the attacker's inverse on its own place-disjoint frames."""
    pairs = []
    for rec in records:
        clean = load_image(rec["query_path"], resize_hw) / 255.0
        for draw in range(draws):
            path = cache_path(cache_root, condition, rec["query_id"], draw)
            if path.is_file():
                pairs.append((torch.load(path, map_location="cpu") / 255.0,
                              clean))
    if not pairs:
        raise SystemExit(f"no cached releases for {condition}")
    print(f"[purify] training the {condition} purifier on {len(pairs)} pairs",
          flush=True)
    model = Purifier(args.channels, args.depth).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.purifier_lr)
    gen = torch.Generator().manual_seed(args.seed)
    crop, batch = args.crop, args.batch_size
    for step in range(1, args.purifier_steps + 1):
        idx = torch.randint(len(pairs), (batch,), generator=gen)
        xs, ys = [], []
        for j in idx.tolist():
            rel, cln = pairs[j]
            top = int(torch.randint(rel.shape[1] - crop + 1, (1,),
                                    generator=gen).item())
            left = int(torch.randint(rel.shape[2] - crop + 1, (1,),
                                     generator=gen).item())
            xs.append(rel[:, top:top + crop, left:left + crop])
            ys.append(cln[:, top:top + crop, left:left + crop])
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        loss = F.l1_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"[purify]   step {step}/{args.purifier_steps} "
                  f"L1 {loss.item():.5f}", flush=True)
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval_backbone", default="resnet18")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"])
    ap.add_argument("--target_mse", type=float, default=15.68)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--eot_sanitizers", nargs="*",
                    default=["jpeg75", "jpeg50", "blur", "denoise"])
    ap.add_argument("--eot_samples", type=int, default=2)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--n_val", type=int, default=80)
    ap.add_argument("--train_draws", type=int, default=2,
                    help="Releases of each attacker-side frame; the attacker "
                         "sees the release distribution, not one realisation.")
    ap.add_argument("--eval_draws", type=int, default=3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--purifier_lr", type=float, default=1e-3)
    ap.add_argument("--purifier_steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--gallery_batch", type=int, default=64)
    ap.add_argument("--cache", default="src/outputs/purification/cache")
    ap.add_argument("--build_only", action="store_true",
                    help="Cache releases and exit. Several workers can share "
                         "one cache directory: each skips what is already on "
                         "disk, so building parallelises even though training "
                         "and evaluation do not.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[purify] device={device}", flush=True)
    for name in args.eot_sanitizers:
        if name not in SANITIZERS:
            raise SystemExit(f"unknown sanitizer {name!r}")
    eot_ops = [SANITIZERS[n] for n in args.eot_sanitizers]

    records, gallery = load_manifest(args.manifest, args.root)
    train, val, test = split_queries(records, args.n_test, args.n_val,
                                     args.seed)
    attacker_records = train + val
    places = {r["place_id"] for r in attacker_records}
    leak = [r for r in test if r["place_id"] in places]
    if leak:
        raise SystemExit(f"{len(leak)} evaluation queries share a place with "
                         f"the attacker's training data")
    print(f"[purify] attacker trains on {len(attacker_records)} queries, "
          f"evaluates on {len(test)}, no shared place", flush=True)

    resize_hw = (args.height, args.width)
    gallery_ids = sorted(gallery)
    place_of = {g: gallery[g]["place_id"] for g in gallery_ids}

    names = [args.eval_backbone] + [b for b in args.surrogates
                                    if b != args.eval_backbone]
    # Building releases needs the surrogates and nothing else. Embedding a
    # 2,000-image gallery in every cache-building worker is the difference
    # between a job that shares one card comfortably and one that thrashes it.
    if args.build_only:
        names = [b for b in names if b != args.eval_backbone]
    embedders, sizes, ev_gal = {}, {}, None
    for b in names:
        cfg = RetrievalConfig(backbone=b,
                              input_size=default_input_size_for_backbone(b))
        e = make_default_embedder(cfg).eval().to(device)
        if b == args.eval_backbone:
            gallery_tensor = torch.stack(
                [load_image(gallery[g]["path"], resize_hw)
                 for g in gallery_ids])
            g = embed_gallery_batched(cfg, e, gallery_tensor,
                                      batch=args.gallery_batch)
            ev_gal = (g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(device)
            del gallery_tensor
        embedders[b], sizes[b] = e, cfg.input_size
        torch.cuda.empty_cache()
        print(f"[purify] {b} ready", flush=True)
    surrogates = [b for b in args.surrogates if b != args.eval_backbone]

    cache_root = Path(args.cache)
    mine = (lambda rs: [r for i, r in enumerate(rs)
                        if i % args.num_shards == args.shard])
    build_releases(mine(attacker_records), CONDITIONS, args.train_draws,
                   cache_root, args, embedders, sizes, surrogates, eot_ops,
                   resize_hw, device)
    build_releases(mine(test), CONDITIONS, args.eval_draws, cache_root, args,
                   embedders, sizes, surrogates, eot_ops, resize_hw, device)
    if args.build_only:
        print("[purify] cache build finished for this shard", flush=True)
        return 0
    if not args.output:
        raise SystemExit("--output is required unless --build_only")

    # The surrogates are only needed to build releases; free them before the
    # purifier trains, so the run fits on a small card.
    for b in surrogates:
        embedders.pop(b)
    torch.cuda.empty_cache()

    purifiers = {c: train_purifier(attacker_records, c, args.train_draws,
                                   cache_root, resize_hw, device, args)
                 for c in CONDITIONS}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_id", "condition", "purifier", "draw", "correct_rank",
              "top1_place", "correct_place", "psnr_to_clean"]
    fh = open(out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    ev = args.eval_backbone

    def rank_of(image: torch.Tensor, want: str):
        with torch.no_grad():
            q = normalised_embedding(embedders[ev], image, sizes[ev])
            order = torch.argsort(ev_gal @ q.flatten(), descending=True)
            rank = next(i + 1 for i, j in enumerate(order.tolist())
                        if place_of[gallery_ids[j]] == want)
            return rank, place_of[gallery_ids[int(order[0].item())]]

    for qi, rec in enumerate(test, 1):
        qid, want = rec["query_id"], rec["place_id"]
        if not any(place_of[g] == want for g in gallery_ids):
            continue
        clean = load_image(rec["query_path"], resize_hw).unsqueeze(0).to(device)
        rank, top1 = rank_of(clean, want)
        writer.writerow({"query_id": qid, "condition": "clean",
                         "purifier": "none", "draw": 0, "correct_rank": rank,
                         "top1_place": top1, "correct_place": want,
                         "psnr_to_clean": ""})
        for condition in CONDITIONS:
            for draw in range(args.eval_draws):
                path = cache_path(cache_root, condition, qid, draw)
                if not path.is_file():
                    continue
                released = torch.load(path, map_location=device).unsqueeze(0)
                arms = {"none": released}
                with torch.no_grad():
                    # Matched: the attacker knows which release it faces.
                    arms[condition] = purifiers[condition](
                        released / 255.0) * 255.0
                    if condition == "hardened":
                        # Cross-exposure: does it have to know?
                        arms["direction"] = purifiers["direction"](
                            released / 255.0) * 255.0
                for name, image in arms.items():
                    rank, top1 = rank_of(image, want)
                    mse = float((image - clean).square().mean())
                    writer.writerow({
                        "query_id": qid, "condition": condition,
                        "purifier": name, "draw": draw, "correct_rank": rank,
                        "top1_place": top1, "correct_place": want,
                        "psnr_to_clean":
                            f"{10 * torch.log10(torch.tensor(255.0 ** 2 / max(mse, 1e-9))):.4f}",
                    })
        fh.flush()
        os.fsync(fh.fileno())
        if qi % 25 == 0:
            print(f"[purify] evaluated {qi}/{len(test)} queries", flush=True)

    fh.close()
    print(f"[purify] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
