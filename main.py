# main.py
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import torch

from datasets.driving_clip_dataset import DrivingClipDataset
from eval.retrieval_attack import (
    RetrievalConfig, make_default_embedder, build_gallery_embeddings, query_topk
)
from eval.metrics import psnr_torch
from models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from privacy.NCP import NCPAllocator, NCPConfig
from privacy.noise_injector import NoiseInjector, NoiseConfig
from utils.config import load_yaml, maybe_override


def load_sensnet_checkpoint(ckpt_path: str, device: torch.device):
    from run_train import SensitiveRegionNet
    model = SensitiveRegionNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def protect_clip(frames: torch.Tensor, sensnet, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    """
    frames: (T,3,H,W) in [0,255]
    returns: (T,3,H,W) protected
    """
    dcfg = cfg["ppedcrf"]["dynamic_crf"]
    ncfg = cfg["ppedcrf"]["ncp"]
    pcfg = cfg["ppedcrf"]["noise"]

    crf = DynamicCRF(DynamicCRFConfig(
        n_iters=int(dcfg["n_iters"]),
        spatial_weight=float(dcfg["spatial_weight"]),
        temporal_weight=float(dcfg["temporal_weight"]),
        smooth_kernel=int(dcfg["smooth_kernel"]),
    ))

    ncp = NCPAllocator(NCPConfig(alpha=float(ncfg.get("alpha", 1.0))), class_sensitivity=ncfg.get("class_sensitivity", {}))

    injector = NoiseInjector(NoiseConfig(
        mode=str(pcfg["mode"]),
        sigma=float(pcfg["sigma"]),
        clamp_min=float(pcfg["clamp_min"]),
        clamp_max=float(pcfg["clamp_max"]),
        seed=pcfg.get("seed", None),
    ))

    prev_prob = None
    out = []
    for t in range(frames.size(0)):
        fr = frames[t:t+1].to(device)
        unary = sensnet(fr)  # logits
        refined_prob, prev_prob = crf.refine(unary, prev_prob=prev_prob, flow=None)

        strength = ncp.allocate(refined_prob)
        protected = injector.apply(fr, refined_prob, strength, foreground_mask=None, t_index=t)
        out.append(protected.squeeze(0).cpu())

    return torch.stack(out, dim=0)


def cmd_train(cfg: Dict[str, Any]) -> None:
    from run_train import train_from_cfg_dict
    train_from_cfg_dict(cfg)


def cmd_attack(cfg: Dict[str, Any]) -> None:
    device_str = str(cfg["project"].get("device", "cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    data_root = cfg["data"]["root"]
    clip_len = int(cfg["data"]["clip_len"])
    resize_hw = tuple(cfg["data"]["resize_hw"])
    attack_cfg = cfg["attack"]

    ds_g = DrivingClipDataset(
        data_root, split=attack_cfg.get("gallery_split", "train"),
        clip_len=clip_len, sample_mode="uniform",
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        max_clips=int(attack_cfg.get("max_gallery", 200)),
    )
    ds_q = DrivingClipDataset(
        data_root, split=attack_cfg.get("query_split", "val"),
        clip_len=clip_len, sample_mode="uniform",
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        max_clips=int(attack_cfg.get("max_query", 50)),
    )

    gallery_imgs, gallery_ids = [], []
    for s in ds_g:
        gallery_imgs.append(s.frames[0])
        gallery_ids.append(s.clip_id)
    gallery_imgs = torch.stack(gallery_imgs, dim=0)

    query_imgs, query_ids = [], []
    for s in ds_q:
        query_imgs.append(s.frames[0])
        query_ids.append(s.clip_id)
    query_imgs = torch.stack(query_imgs, dim=0)

    rcfg = RetrievalConfig(
        backbone=str(attack_cfg.get("backbone", "resnet18")),
        device=str(device),
        normalize=bool(attack_cfg.get("normalize", True)),
        input_size=int(attack_cfg.get("input_size", 224)),
        topk=tuple(attack_cfg.get("topk", [1, 5, 10])),
    )

    embedder = make_default_embedder(rcfg)
    gallery_emb = build_gallery_embeddings(rcfg, embedder, gallery_imgs)
    res = query_topk(rcfg, query_imgs, query_ids, gallery_emb, gallery_ids, embedder)

    print("Retrieval results:")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")


def cmd_protect(cfg: Dict[str, Any]) -> None:
    device_str = str(cfg["project"].get("device", "cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    data_root = cfg["data"]["root"]
    clip_len = int(cfg["data"]["clip_len"])
    resize_hw = tuple(cfg["data"]["resize_hw"])

    protect_cfg = cfg["protect"]
    split = str(protect_cfg.get("split", "val"))
    ckpt = str(protect_cfg["checkpoint"])
    max_clips = int(protect_cfg.get("max_clips", 20))

    ds = DrivingClipDataset(
        data_root, split=split, clip_len=clip_len, sample_mode="uniform",
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        max_clips=max_clips,
    )

    sensnet = load_sensnet_checkpoint(ckpt, device)

    psnr_vals = []
    for s in ds:
        frames = s.frames
        protected = protect_clip(frames, sensnet, cfg, device)
        p = psnr_torch(frames[0], protected[0])
        psnr_vals.append(p)
        print(f"[{s.clip_id}] PSNR(first frame) = {p:.2f} dB")

    if psnr_vals:
        mean_psnr = sum(psnr_vals) / len(psnr_vals)
        print(f"Mean PSNR(first frame): {mean_psnr:.2f} dB")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("PPEDCRF entrypoint")
    p.add_argument("--config", type=str, default="config/config.yaml", help="Path to YAML config")

    sub = p.add_subparsers(dest="cmd", required=True)

    # Train overrides
    t = sub.add_parser("train", help="Train sensitive region network")
    t.add_argument("--device", type=str, default=None)
    t.add_argument("--data_root", type=str, default=None)
    t.add_argument("--out_dir", type=str, default=None)
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch_size", type=int, default=None)
    t.add_argument("--num_workers", type=int, default=None)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--mask_root", type=str, default=None)

    # Attack overrides
    a = sub.add_parser("attack", help="Run retrieval attack")
    a.add_argument("--device", type=str, default=None)
    a.add_argument("--data_root", type=str, default=None)
    a.add_argument("--backbone", type=str, default=None)
    a.add_argument("--max_gallery", type=int, default=None)
    a.add_argument("--max_query", type=int, default=None)

    # Protect overrides
    pr = sub.add_parser("protect", help="Protect clips and report metrics")
    pr.add_argument("--device", type=str, default=None)
    pr.add_argument("--data_root", type=str, default=None)
    pr.add_argument("--checkpoint", type=str, default=None)
    pr.add_argument("--split", type=str, default=None)
    pr.add_argument("--max_clips", type=int, default=None)

    return p


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    # Shared
    maybe_override(cfg, "project.device", getattr(args, "device", None))
    maybe_override(cfg, "data.root", getattr(args, "data_root", None))

    if args.cmd == "train":
        maybe_override(cfg, "train.out_dir", args.out_dir)
        maybe_override(cfg, "train.epochs", args.epochs)
        maybe_override(cfg, "train.batch_size", args.batch_size)
        maybe_override(cfg, "train.num_workers", args.num_workers)
        maybe_override(cfg, "train.lr", args.lr)
        # mask_root: allow explicit "null-like" values via empty string -> None
        if args.mask_root == "":
            cfg["train"]["mask_root"] = None
        else:
            maybe_override(cfg, "train.mask_root", args.mask_root)

    elif args.cmd == "attack":
        maybe_override(cfg, "attack.backbone", args.backbone)
        maybe_override(cfg, "attack.max_gallery", args.max_gallery)
        maybe_override(cfg, "attack.max_query", args.max_query)

    elif args.cmd == "protect":
        maybe_override(cfg, "protect.checkpoint", args.checkpoint)
        maybe_override(cfg, "protect.split", args.split)
        maybe_override(cfg, "protect.max_clips", args.max_clips)

    return cfg


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.cmd == "train":
        cmd_train(cfg)
    elif args.cmd == "attack":
        cmd_attack(cfg)
    elif args.cmd == "protect":
        cmd_protect(cfg)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()