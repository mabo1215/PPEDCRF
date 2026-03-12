"""
Run retrieval attack with PPEDCRF protection under multiple noise seeds and report mean ± std
for Top-k metrics (R@1, R@5, R@10) over 3 seeds, for reproducibility (paper: "we report mean ± std").

Usage:
  python scripts/run_attack_multiseed.py [--config config/config.yaml] [--seeds 1234 1235 1236] [--update-tex]
  python scripts/run_attack_multiseed.py --backbone resnet18   # if default backbone needs cv2/ultralytics

Output: mean ± std for R@1, R@5, R@10. With --update-tex, updates paper/main.tex placeholders
  \\texttt{<R1>}, \\texttt{<R5>}, \\texttt{<R10>} and \\texttt{<R1std>}, \\texttt{<R5std>}, \\texttt{<R10std>}.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.config import load_yaml
from main import load_sensnet_checkpoint, protect_clip
from datasets.driving_clip_dataset import DrivingClipDataset
from eval.retrieval_attack import (
    RetrievalConfig,
    make_default_embedder,
    build_gallery_embeddings,
    query_topk,
)


def run_one_seed(
    cfg: dict,
    device: torch.device,
    sensnet,
    embedder,
    rcfg,
    gallery_emb: torch.Tensor,
    gallery_ids: list,
    ds_q: DrivingClipDataset,
    query_ids: list,
    seed: int,
) -> dict:
    """Protect query set with given seed, run retrieval, return R@1, R@5, R@10."""
    cfg_seed = copy.deepcopy(cfg)
    cfg_seed["ppedcrf"]["noise"] = dict(cfg_seed["ppedcrf"]["noise"])
    cfg_seed["ppedcrf"]["noise"]["seed"] = seed

    query_imgs = []
    for idx, s in enumerate(ds_q):
        frames = s.frames.to(device)
        protected = protect_clip(frames, sensnet, cfg_seed, device)
        query_imgs.append(protected[0].cpu())
    query_imgs = torch.stack(query_imgs, dim=0)

    res = query_topk(rcfg, query_imgs, query_ids, gallery_emb, gallery_ids, embedder)
    return res


def main():
    p = argparse.ArgumentParser(description="Run retrieval attack with 3 seeds, report mean ± std")
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default="outputs/sensnet_final.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    p.add_argument("--max_gallery", type=int, default=None)
    p.add_argument("--max_query", type=int, default=None)
    p.add_argument("--update-tex", dest="update_tex", action="store_true", help="Update paper/main.tex placeholders")
    p.add_argument("--tex", type=str, default=None, help="Path to main.tex (default: paper/main.tex)")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    data_root = cfg["data"]["root"]
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_cfg = cfg.get("attack", {})
    backbone = args.backbone or str(attack_cfg.get("backbone", "resnet18"))
    clip_len = int(cfg["data"]["clip_len"])
    resize_hw = tuple(cfg["data"]["resize_hw"])
    max_gallery = args.max_gallery or int(attack_cfg.get("max_gallery", 200))
    max_query = args.max_query or int(attack_cfg.get("max_query", 50))

    rcfg = RetrievalConfig(
        backbone=backbone,
        device=str(device),
        normalize=bool(attack_cfg.get("normalize", True)),
        input_size=int(attack_cfg.get("input_size", 224)),
        topk=tuple(attack_cfg.get("topk", [1, 5, 10])),
    )

    print("[multiseed] building gallery...")
    ds_g = DrivingClipDataset(
        data_root,
        split=attack_cfg.get("gallery_split", "train"),
        clip_len=clip_len,
        sample_mode="uniform",
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        max_clips=max_gallery,
    )
    gallery_imgs = torch.stack([s.frames[0] for s in ds_g], dim=0)
    gallery_ids = [s.clip_id for s in ds_g]

    print("[multiseed] building query dataset...")
    ds_q = DrivingClipDataset(
        data_root,
        split=attack_cfg.get("query_split", "val"),
        clip_len=clip_len,
        sample_mode="uniform",
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        max_clips=max_query,
    )
    query_ids = [s.clip_id for s in ds_q]
    if len(ds_q) == 0:
        print("No query clips.")
        sys.exit(1)

    embedder = make_default_embedder(rcfg).eval().to(device)
    gallery_emb = build_gallery_embeddings(rcfg, embedder, gallery_imgs)

    sensnet = load_sensnet_checkpoint(args.checkpoint, device)

    runs = []
    for seed in args.seeds:
        print(f"[multiseed] seed={seed} ...")
        res = run_one_seed(
            cfg, device, sensnet, embedder, rcfg,
            gallery_emb, gallery_ids, ds_q, query_ids, seed,
        )
        runs.append(res)

    # Aggregate
    keys = [f"R@{k}" for k in rcfg.topk]
    means = {}
    stds = {}
    for k in keys:
        vals = [r[k] for r in runs]
        means[k] = sum(vals) / len(vals)
        if len(vals) >= 2:
            variance = sum((x - means[k]) ** 2 for x in vals) / (len(vals) - 1)
            stds[k] = variance ** 0.5
        else:
            stds[k] = 0.0

    print("Top-k retrieval accuracy (protected queries, mean ± std over {} seeds):".format(len(args.seeds)))
    for k in keys:
        print(f"  {k}: {means[k]:.4f} ± {stds[k]:.4f}")

    if args.update_tex:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_tex = args.tex
        if main_tex is None:
            main_tex = os.path.join(root, "paper", "main.tex")
        if not os.path.isabs(main_tex):
            main_tex = os.path.join(root, main_tex)
        if not os.path.isfile(main_tex):
            print(f"Warning: {main_tex} not found")
        else:
            with open(main_tex, "r", encoding="utf-8") as f:
                content = f.read()
            replacements = [
                (r"\texttt{<R1>}", f"{means.get('R@1', 0):.3f}"),
                (r"\texttt{<R5>}", f"{means.get('R@5', 0):.3f}"),
                (r"\texttt{<R10>}", f"{means.get('R@10', 0):.3f}"),
                (r"\texttt{<R1std>}", f"{stds.get('R@1', 0):.3f}"),
                (r"\texttt{<R5std>}", f"{stds.get('R@5', 0):.3f}"),
                (r"\texttt{<R10std>}", f"{stds.get('R@10', 0):.3f}"),
            ]
            for old, new in replacements:
                content = content.replace(old, new)
            with open(main_tex, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {main_tex} with Top-k mean ± std placeholders.")

    return means, stds


if __name__ == "__main__":
    main()
