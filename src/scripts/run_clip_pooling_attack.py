"""Does the direction survive an attacker that holds the whole clip?

Every retrieval result in this work releases one frame and lets the attacker
embed it once. The scenario the work is motivated by --- dashcam, bodycam,
drone and wearable footage uploaded for auditing --- hands the attacker the
video. That matters specifically for the directional release rather than for
allocation: the deployable direction is re-derived per frame from that frame's
own clean embedding, so the displacement it applies points a different way in
each frame of a clip, while the place the frames depict is common to all of
them. An attacker who averages the embeddings of k released frames averages k
differently oriented displacements against one coherent signal, and to first
order the perturbation's contribution shrinks like 1/sqrt(k) relative to it.

This script measures what that costs. For each query it releases the whole
clip --- the query frame and its neighbours in the same MSLS sequence, from
the clip manifest --- under the same condition, optimiser, seed and delivered
MSE as the single-frame study, and then ranks the gallery from a pooled query
embedding for every clip length and pooling strategy. The isotropic control is
drawn per frame in the same way, so the comparison isolates what pooling does
to a *direction*, not what it does to more frames.

Poolings follow the definitions already used for the sequence study: ``first``
uses the query frame alone and reproduces the single-frame result, ``mean``
and ``max`` pool the frame embeddings before scoring, and ``best_frame``
scores each gallery item by its largest similarity over the frames, which
upper-bounds any attacker that selects one frame to query with.

Writes one row per (query, condition, seed, clip length, pooling), flushed
immediately, and skips completed work on restart.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from pathlib import Path

import torch
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

POOLINGS = ("first", "mean", "max", "best_frame")


def pooled_scores(frame_embeddings: torch.Tensor, gallery: torch.Tensor,
                  pooling: str) -> torch.Tensor:
    """Similarity of one clip against every gallery item under one pooling."""
    if pooling == "best_frame":
        # Per gallery item, the best any single frame of the clip achieves.
        return (gallery @ frame_embeddings.t()).max(dim=1).values
    if pooling == "first":
        value = frame_embeddings[0]
    elif pooling == "mean":
        value = frame_embeddings.mean(dim=0)
    elif pooling == "max":
        value = frame_embeddings.max(dim=0).values
    else:
        raise ValueError(f"unknown pooling {pooling!r}")
    return gallery @ F.normalize(value.unsqueeze(0), dim=1).squeeze(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips", required=True,
                    help="Clip manifest from build_msls_clip_manifest.py.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval_backbone", default="resnet18")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"])
    ap.add_argument("--conditions", nargs="+",
                    default=["isotropic", "direction"],
                    choices=["isotropic", "direction", "hardened", "white_box"])
    ap.add_argument("--clip_lens", type=int, nargs="+", default=[1, 2, 4, 7])
    ap.add_argument("--poolings", nargs="+", default=list(POOLINGS))
    ap.add_argument("--target_mse", type=float, default=15.68)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0,
                    help="This worker's index, for splitting the query list "
                         "across concurrent processes.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="How many workers the query list is split across. "
                         "Queries are dealt round-robin so every shard sees "
                         "all eight cities and finishes in comparable time.")
    ap.add_argument("--gallery_batch", type=int, default=128)
    ap.add_argument("--eot_sanitizers", nargs="*",
                    default=["jpeg75", "jpeg50", "blur", "denoise"],
                    help="Transforms the hardened condition is optimised over.")
    ap.add_argument("--eot_samples", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[clip] device={device}", flush=True)
    for name in args.eot_sanitizers:
        if name not in SANITIZERS:
            raise SystemExit(f"unknown sanitizer {name!r}; "
                             f"available: {sorted(SANITIZERS)}")
    eot_ops = [SANITIZERS[n] for n in args.eot_sanitizers]

    records, gallery = load_manifest(args.manifest, args.root)
    clips = json.load(open(args.clips, encoding="utf-8"))["clips"]
    queries = [r for r in records if r["query_id"] in clips]
    if args.limit:
        queries = queries[: args.limit]
    if args.num_shards > 1:
        queries = [q for i, q in enumerate(queries)
                   if i % args.num_shards == args.shard]
    missing = len(records) - len([r for r in records if r["query_id"] in clips])
    resize_hw = (args.height, args.width)
    gallery_ids = sorted(gallery)
    place_of = {g: gallery[g]["place_id"] for g in gallery_ids}
    gallery_tensor = torch.stack(
        [load_image(gallery[g]["path"], resize_hw) for g in gallery_ids])
    print(f"[clip] shard {args.shard + 1}/{args.num_shards}: "
          f"{len(queries)} queries ({missing} without a clip), "
          f"{len(gallery_ids)} gallery images", flush=True)

    names = [args.eval_backbone] + [b for b in args.surrogates
                                    if b != args.eval_backbone]
    embedders, sizes = {}, {}
    ev_gal = None
    for b in names:
        cfg = RetrievalConfig(backbone=b,
                              input_size=default_input_size_for_backbone(b))
        e = make_default_embedder(cfg).eval().to(device)
        if b == args.eval_backbone:
            g = embed_gallery_batched(cfg, e, gallery_tensor,
                                      batch=args.gallery_batch)
            ev_gal = (g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(device)
        embedders[b], sizes[b] = e, cfg.input_size
        torch.cuda.empty_cache()
        print(f"[clip] {b} ready", flush=True)
    surr = [b for b in names if b != args.eval_backbone]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["query_id"], r["condition"], r["seed"]))
        print(f"[clip] resuming, {len(done)} completed (query, condition, "
              f"seed) groups on file", flush=True)

    fields = ["query_id", "condition", "seed", "clip_len", "pooling",
              "correct_rank", "top1_place", "correct_place", "frames_used",
              "mean_mse", "psnr", "max_frame_offset"]
    new = not out.is_file()
    fh = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new:
        writer.writeheader()

    ev = args.eval_backbone
    zero_delta = 0
    for qi, rec in enumerate(queries, 1):
        qid = rec["query_id"]
        clip = clips[qid]
        want = rec["place_id"]
        if not any(place_of[g] == want for g in gallery_ids):
            continue
        paths = clip["frames"][: max(args.clip_lens)]
        frames = [load_image(str(Path(args.root) / p), resize_hw)
                  .unsqueeze(0).to(device) for p in paths]
        offsets = clip["frame_offsets"][: len(frames)]
        for seed in args.seeds:
            for cond in args.conditions:
                if (qid, cond, str(seed)) in done:
                    continue
                released, mses = [], []
                for fi, frame in enumerate(frames):
                    if cond == "isotropic":
                        g = torch.Generator(device="cpu").manual_seed(
                            zlib.crc32(f"{qid}|{seed}|{fi}".encode())
                            & 0x7FFFFFFF)
                        delta = torch.randn(frame.shape, generator=g).to(device)
                    else:
                        use = [ev] if cond == "white_box" else surr
                        with torch.no_grad():
                            tgts = [normalised_embedding(
                                embedders[b], frame, sizes[b]).detach()
                                for b in use]
                        gstart = torch.Generator(device="cpu").manual_seed(
                            zlib.crc32(f"{qid}|{cond}|{seed}|{fi}".encode())
                            & 0x7FFFFFFF)
                        with torch.enable_grad():
                            delta = directional_delta(
                                frame, tgts, [embedders[b] for b in use],
                                [sizes[b] for b in use], args.steps,
                                args.step_size, args.linf,
                                random_start=1.0, generator=gstart,
                                eot_ops=eot_ops if cond == "hardened" else (),
                                eot_samples=args.eot_samples)
                        if float(delta.abs().max()) == 0.0:
                            zero_delta += 1
                    rel = release_at_mse(frame, delta, args.target_mse)
                    released.append(rel)
                    mses.append(float((rel - frame).square().mean()))
                with torch.no_grad():
                    embs = torch.stack([
                        normalised_embedding(embedders[ev], r, sizes[ev])
                        .flatten() for r in released])
                for k in args.clip_lens:
                    n = min(k, len(released))
                    if n < k:
                        # A clip that cannot reach k frames is not padded: a
                        # shorter clip evaluated as if it were length k would
                        # report the attacker's k-frame accuracy from k-1
                        # frames.
                        continue
                    mse = sum(mses[:n]) / n
                    for pooling in args.poolings:
                        scores = pooled_scores(embs[:n], ev_gal, pooling)
                        order = torch.argsort(scores, descending=True)
                        rank = next(i + 1 for i, j in enumerate(order.tolist())
                                    if place_of[gallery_ids[j]] == want)
                        writer.writerow({
                            "query_id": qid, "condition": cond, "seed": seed,
                            "clip_len": n, "pooling": pooling,
                            "correct_rank": rank,
                            "top1_place": place_of[gallery_ids[int(order[0])]],
                            "correct_place": want, "frames_used": n,
                            "mean_mse": f"{mse:.6f}",
                            "psnr": f"{10 * torch.log10(torch.tensor(255.0 ** 2 / max(mse, 1e-9))):.4f}",
                            "max_frame_offset": max(abs(o) for o in offsets[:n]),
                        })
                fh.flush()
                os.fsync(fh.fileno())
        if qi % 10 == 0:
            print(f"[clip] {qi}/{len(queries)} queries", flush=True)

    fh.close()
    if zero_delta:
        print(f"[clip] WARNING {zero_delta} frames received a zero "
              f"perturbation", flush=True)
    print(f"[clip] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
