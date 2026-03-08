"""
Compute average privacy-to-privacy nearest-neighbor distance before (d0) and after (d1) NCP
for the sentence in main.tex: "... the average privacy-to-privacy nearest-neighbor
distance increases from <d0> to <d1> ..."

Uses the same embedding as retrieval attack: query frames (original vs protected)
in embedding space; d0 = mean nearest-neighbor distance over original embeddings,
d1 = mean over protected embeddings. So d1 > d0 indicates that after protection
samples are more spread out (harder to retrieve).

Usage:
  python scripts/compute_privacy_neighbor_distances.py
  python scripts/compute_privacy_neighbor_distances.py --update-tex
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from utils.config import load_yaml
from main import load_sensnet_checkpoint, protect_clip
from eval.retrieval_attack import (
    RetrievalConfig,
    make_default_embedder,
    preprocess_for_embed,
    build_gallery_embeddings,
)


@torch.no_grad()
def mean_nearest_neighbor_distance(emb: torch.Tensor, normalize: bool = True) -> float:
    """
    emb: (N, D). Returns mean over i of min_{j != i} distance(i, j).
    If normalize=True we use L2 distance on unit vectors (so dist^2 = 2 - 2*cos).
    """
    if emb.size(0) < 2:
        return 0.0
    if normalize:
        emb = F.normalize(emb, dim=1)
    # (N, N) pairwise L2^2 for normalized: 2 - 2*cos = 2 - 2*(emb @ emb.t())
    sim = emb @ emb.t()
    dist_sq = (2.0 - 2.0 * sim).clamp(min=0.0)
    # exclude self
    dist_sq.fill_diagonal_(float("inf"))
    nn_dist_sq = dist_sq.min(dim=1).values
    nn_dist = nn_dist_sq.clamp(min=0.0).sqrt()
    return nn_dist.mean().item()


def main():
    p = argparse.ArgumentParser(description="Compute d0, d1 for privacy-to-privacy NN distance")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="outputs/sensnet_final.pt")
    p.add_argument("--max_clips", type=int, default=50)
    p.add_argument("--split", type=str, default="test", help="Dataset split: test, val, or train (default: test)")
    p.add_argument("--update-tex", action="store_true", help="Replace \\texttt{<d0>} and \\texttt{<d1>} (and optionally $N_{\\max}$, $N_{\\min}$) in doc/main.tex")
    p.add_argument("--N_max", type=int, default=None, help="Largest class sample count for main.tex $N_{\\max}$ (use with --update-tex)")
    p.add_argument("--N_min", type=int, default=None, help="Smallest class sample count for main.tex $N_{\\min}$ (use with --update-tex)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    data_root = args.data_root or cfg["data"]["root"]
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_cfg = cfg.get("attack", {})
    rcfg = RetrievalConfig(
        backbone=str(attack_cfg.get("backbone", "resnet18")),
        device=str(device),
        normalize=bool(attack_cfg.get("normalize", True)),
        input_size=int(attack_cfg.get("input_size", 224)),
        topk=(1, 5, 10),
    )
    embedder = make_default_embedder(rcfg).eval().to(device)

    from datasets.driving_clip_dataset import DrivingClipDataset
    resize_hw = tuple(cfg["data"]["resize_hw"])
    clip_len = int(cfg["data"]["clip_len"])
    split = args.split or attack_cfg.get("query_split", "test")
    # Use first split that has at least 1 clip; we will take multiple frames per clip if needed
    candidates = [split] + [s for s in ("test", "val", "train") if s != split]
    ds = None
    for candidate in candidates:
        if not os.path.isdir(os.path.join(data_root, candidate)):
            continue
        ds = DrivingClipDataset(
            data_root,
            split=candidate,
            clip_len=clip_len,
            sample_mode="uniform",
            resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
            max_clips=args.max_clips,
        )
        if len(ds) >= 1:
            split = candidate
            break
    if ds is None or len(ds) < 1:
        print(f"No clips found in {data_root} (tried {candidates})")
        sys.exit(1)

    # Need at least 2 samples (frames) for NN distance. Take multiple frames per clip when there are few clips.
    min_samples = 2
    max_samples_per_clip = 50
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)

    orig_frames = []
    prot_frames = []
    for s in ds:
        frames = s.frames  # (T, C, H, W)
        n_f = frames.size(0)
        n_take = min(max_samples_per_clip, max(min_samples, n_f))
        indices = torch.linspace(0, n_f - 1, n_take).long().clamp(0, n_f - 1)
        for i in indices:
            orig_frames.append(frames[i])
        protected = protect_clip(frames, sensnet, cfg, device)
        for i in indices:
            prot_frames.append(protected[i])
    orig_frames = torch.stack(orig_frames, dim=0).to(device)
    prot_frames = torch.stack(prot_frames, dim=0).to(device)
    if orig_frames.size(0) < min_samples:
        print(f"Need at least {min_samples} frames; got {orig_frames.size(0)} from {len(ds)} clip(s)")
        sys.exit(1)
    print(f"Using split={split}, {len(ds)} clip(s), {orig_frames.size(0)} frames for d0/d1")

    # Embed: (N,3,H,W) in [0,255] -> preprocess -> embed
    def embed_images(imgs: torch.Tensor) -> torch.Tensor:
        x = preprocess_for_embed(imgs, rcfg.input_size)
        emb = embedder(x)
        if rcfg.normalize:
            emb = F.normalize(emb, dim=1)
        return emb

    E_orig = embed_images(orig_frames)
    E_prot = embed_images(prot_frames)

    d0 = mean_nearest_neighbor_distance(E_orig, normalize=False)  # already normalized in embed_images
    d1 = mean_nearest_neighbor_distance(E_prot, normalize=False)

    print("Average privacy-to-privacy nearest-neighbor distance (embedding space):")
    print(f"  Before NCP (d0): {d0:.4f}")
    print(f"  After NCP (d1):  {d1:.4f}")
    print(f"  Increase: d1/d0 = {d1/d0:.4f}" if d0 > 0 else "  (d0=0)")
    print("For main.tex: replace \\texttt{<d0>} with", f"{d0:.3f}", "and \\texttt{<d1>} with", f"{d1:.3f}")

    if args.update_tex:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_tex = os.path.join(root, "doc", "main.tex")
        if not os.path.isfile(main_tex):
            print(f"Warning: {main_tex} not found")
        else:
            with open(main_tex, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = content.replace(r"\texttt{<d0>}", f"{d0:.3f}")
            new_content = new_content.replace(r"\texttt{<d1>}", f"{d1:.3f}")
            if args.N_max is not None:
                new_content = new_content.replace(r"$N_{\max}$", f"{args.N_max}")
            if args.N_min is not None:
                new_content = new_content.replace(r"$N_{\min}$", f"{args.N_min}")
            if new_content != content:
                with open(main_tex, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated {main_tex}: <d0> -> {d0:.3f}, <d1> -> {d1:.3f}", end="")
                if args.N_max is not None:
                    print(f", $N_{{\\max}}$ -> {args.N_max}", end="")
                if args.N_min is not None:
                    print(f", $N_{{\\min}}$ -> {args.N_min}", end="")
                print()
            else:
                print("No \\texttt{<d0>} or \\texttt{<d1>} found in main.tex")
    return d0, d1


if __name__ == "__main__":
    main()
