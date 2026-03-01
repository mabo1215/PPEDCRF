# main.py
from __future__ import annotations

import argparse
import os
from typing import List

import torch

from ppedcrf.datasets.driving_clip_dataset import DrivingClipDataset
from ppedcrf.eval.retrieval_attack import RetrievalConfig, make_default_embedder, build_gallery_embeddings, query_topk
from ppedcrf.eval.metrics import psnr_torch
from ppedcrf.models.dynamic_crf import DynamicCRF, DynamicCRFConfig
from ppedcrf.privacy.ncp import NCPAllocator, NCPConfig
from ppedcrf.privacy.noise_injector import NoiseInjector, NoiseConfig


def load_sensnet_checkpoint(ckpt_path: str, device: torch.device):
    from run_train import SensitiveRegionNet
    model = SensitiveRegionNet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def protect_clip(frames: torch.Tensor, sensnet, device: torch.device) -> torch.Tensor:
    """
    frames: (T,3,H,W) in [0,255]
    returns: protected frames (T,3,H,W)
    """
    crf = DynamicCRF(DynamicCRFConfig(n_iters=5, spatial_weight=2.0, temporal_weight=2.0))
    ncp = NCPAllocator(NCPConfig(alpha=1.0))
    injector = NoiseInjector(NoiseConfig(mode="wiener", sigma=8.0, seed=1234))

    prev_prob = None
    out = []
    for t in range(frames.size(0)):
        fr = frames[t:t+1].to(device)  # (1,3,H,W)
        unary = sensnet(fr)            # (1,1,H,W) logits
        refined_prob, prev_prob = crf.refine(unary, prev_prob=prev_prob, flow=None)

        strength = ncp.allocate(refined_prob)
        protected = injector.apply(fr, refined_prob, strength, foreground_mask=None, t_index=t)
        out.append(protected.squeeze(0).cpu())
    return torch.stack(out, dim=0)


def cmd_train(args):
    from run_train import TrainConfig, train
    cfg = TrainConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        clip_len=args.clip_len,
        resize_hw=(args.resize_h, args.resize_w),
        mask_root=args.mask_root,
    )
    train(cfg)


def cmd_attack(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build gallery and query sets from dataset
    ds_g = DrivingClipDataset(args.data_root, split=args.gallery_split, clip_len=args.clip_len, sample_mode="uniform",
                             resize_hw=(args.resize_h, args.resize_w), max_clips=args.max_gallery)
    ds_q = DrivingClipDataset(args.data_root, split=args.query_split, clip_len=args.clip_len, sample_mode="uniform",
                             resize_hw=(args.resize_h, args.resize_w), max_clips=args.max_query)

    # Use the first frame of each clip as representative (simple baseline).
    gallery_imgs = []
    gallery_ids: List[str] = []
    for s in ds_g:
        gallery_imgs.append(s.frames[0])
        gallery_ids.append(s.clip_id)
    gallery_imgs = torch.stack(gallery_imgs, dim=0)  # (N,3,H,W)

    query_imgs = []
    query_ids: List[str] = []
    for s in ds_q:
        query_imgs.append(s.frames[0])
        query_ids.append(s.clip_id)
    query_imgs = torch.stack(query_imgs, dim=0)  # (M,3,H,W)

    cfg = RetrievalConfig(backbone=args.backbone, device=str(device), topk=tuple(args.topk))
    embedder = make_default_embedder(cfg)

    gallery_emb = build_gallery_embeddings(cfg, embedder, gallery_imgs)
    res = query_topk(cfg, query_imgs, query_ids, gallery_emb, gallery_ids, embedder)

    print("Retrieval results:")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")


def cmd_protect(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds = DrivingClipDataset(args.data_root, split=args.split, clip_len=args.clip_len, sample_mode="uniform",
                            resize_hw=(args.resize_h, args.resize_w), max_clips=args.max_clips)

    sensnet = load_sensnet_checkpoint(args.ckpt, device)

    psnr_vals = []
    for s in ds:
        frames = s.frames  # (T,3,H,W)
        protected = protect_clip(frames, sensnet, device)

        # compute PSNR on first frame as a quick sanity metric
        p = psnr_torch(frames[0], protected[0])
        psnr_vals.append(p)
        print(f"[{s.clip_id}] PSNR(first frame) = {p:.2f} dB")

    if len(psnr_vals) > 0:
        mean_psnr = sum(psnr_vals) / len(psnr_vals)
        print(f"Mean PSNR(first frame): {mean_psnr:.2f} dB")


def build_parser():
    p = argparse.ArgumentParser("PPEDCRF entrypoint")
    sub = p.add_subparsers(dest="cmd", required=True)

    # shared args helper
    def add_data_args(sp):
        sp.add_argument("--data_root", type=str, required=True)
        sp.add_argument("--device", type=str, default="cuda")
        sp.add_argument("--clip_len", type=int, default=8)
        sp.add_argument("--resize_h", type=int, default=384)
        sp.add_argument("--resize_w", type=int, default=640)

    # train
    t = sub.add_parser("train", help="Train sensitive region network")
    add_data_args(t)
    t.add_argument("--out_dir", type=str, default="outputs")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch_size", type=int, default=2)
    t.add_argument("--num_workers", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--mask_root", type=str, default=None)
    t.set_defaults(fn=cmd_train)

    # attack
    a = sub.add_parser("attack", help="Run top-k retrieval attack evaluation")
    add_data_args(a)
    a.add_argument("--backbone", type=str, default="resnet18")
    a.add_argument("--gallery_split", type=str, default="train")
    a.add_argument("--query_split", type=str, default="val")
    a.add_argument("--max_gallery", type=int, default=200)
    a.add_argument("--max_query", type=int, default=50)
    a.add_argument("--topk", type=int, nargs="+", default=[1, 5, 10])
    a.set_defaults(fn=cmd_attack)

    # protect
    pr = sub.add_parser("protect", help="Protect clips with PPEDCRF and report basic metrics")
    add_data_args(pr)
    pr.add_argument("--split", type=str, default="val")
    pr.add_argument("--ckpt", type=str, required=True)
    pr.add_argument("--max_clips", type=int, default=20)
    pr.set_defaults(fn=cmd_protect)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()