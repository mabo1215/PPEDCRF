# main.py
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import numpy as np
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
    # Support Hugging Face Hub repo id (e.g. "mabo1215/ppedcrf-sensnet"); requires pip install huggingface_hub
    is_hf_repo = (
        "/" in ckpt_path
        and "\\" not in ckpt_path
        and not ckpt_path.lower().endswith((".pt", ".pth", ".bin", ".ckpt"))
        and not os.path.exists(ckpt_path)
        and hasattr(SensitiveRegionNet, "from_pretrained")
    )
    if is_hf_repo:
        model = SensitiveRegionNet.from_pretrained(ckpt_path, map_location=device)
    else:
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


@torch.no_grad()
def compute_refined_sensitivity(
    frames: torch.Tensor,
    sensnet,
    cfg: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the refined sensitivity map and NCP strength map for each frame."""
    dcfg = cfg["ppedcrf"]["dynamic_crf"]
    ncfg = cfg["ppedcrf"]["ncp"]

    crf = DynamicCRF(DynamicCRFConfig(
        n_iters=int(dcfg["n_iters"]),
        spatial_weight=float(dcfg["spatial_weight"]),
        temporal_weight=float(dcfg["temporal_weight"]),
        smooth_kernel=int(dcfg["smooth_kernel"]),
    ))

    ncp = NCPAllocator(NCPConfig(alpha=float(ncfg.get("alpha", 1.0))), class_sensitivity=ncfg.get("class_sensitivity", {}))

    prev_prob = None
    refined_probs = []
    strength_maps = []

    for t in range(frames.size(0)):
        fr = frames[t:t+1].to(device)
        unary = sensnet(fr)
        refined_prob, prev_prob = crf.refine(unary, prev_prob=prev_prob, flow=None)

        strength = ncp.allocate(refined_prob)
        refined_probs.append(refined_prob.cpu())
        strength_maps.append(strength.cpu())

    return torch.cat(refined_probs, dim=0), torch.cat(strength_maps, dim=0)


@torch.no_grad()
def apply_mask_guided_blur(
    frame: torch.Tensor,
    mask: torch.Tensor,
    kernel_size: int = 15,
) -> torch.Tensor:
    """Blur only the sensitive regions defined by mask."""
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    pad = kernel_size // 2
    blurred = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(frame, (pad, pad, pad, pad), mode="reflect"),
        kernel_size=kernel_size,
        stride=1,
    )

    mask = mask.clamp(0.0, 1.0)
    out = frame * (1.0 - mask) + blurred * mask
    return out.clamp(0.0, 255.0)


def _load_maskrcnn_model(device: torch.device):
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    try:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
    except Exception:
        model = maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def overlay_detection_boxes(
    frame: torch.Tensor,
    device: torch.device,
    threshold: float = 0.7,
    line_width: int = 3,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.0,
    categories: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
) -> torch.Tensor:
    """Draw detection boxes for sensitive objects using a pretrained Mask R-CNN."""
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)

    # Convert to [0,1] float for torchvision detector
    inp = frame[0].clone()
    if inp.max() > 1.0:
        inp = inp / 255.0
    import torchvision.transforms.functional as TF
    inp = TF.convert_image_dtype(inp, torch.float32)

    model = _load_maskrcnn_model(device)
    preds = model([inp])[0]

    boxes = preds["boxes"]
    labels = preds["labels"]
    scores = preds["scores"]
    keep = scores >= float(threshold)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if categories is not None:
        keep_cat = [(int(lbl.item()) in categories) for lbl in labels]
        if len(keep_cat) > 0:
            keep_cat = torch.tensor(keep_cat, device=device, dtype=torch.bool)
            boxes = boxes[keep_cat]
            labels = labels[keep_cat]
            scores = scores[keep_cat]

    out = frame.clone()
    color_t = torch.tensor(color, device=device, dtype=frame.dtype).view(1, 3, 1, 1)

    for i in range(boxes.size(0)):
        x0, y0, x1, y1 = boxes[i].round().to(torch.int64).tolist()
        x0 = max(0, min(x0, frame.size(3) - 1))
        x1 = max(0, min(x1, frame.size(3) - 1))
        y0 = max(0, min(y0, frame.size(2) - 1))
        y1 = max(0, min(y1, frame.size(2) - 1))
        if alpha > 0:
            out[:, :, y0 : y1 + 1, x0 : x1 + 1] = (
                out[:, :, y0 : y1 + 1, x0 : x1 + 1] * (1.0 - alpha)
                + color_t * alpha
            )
        out[:, :, y0 : y0 + line_width, x0 : x1 + 1] = color_t
        out[:, :, y1 - line_width + 1 : y1 + 1, x0 : x1 + 1] = color_t
        out[:, :, y0 : y1 + 1, x0 : x0 + line_width] = color_t
        out[:, :, y0 : y1 + 1, x1 - line_width + 1 : x1 + 1] = color_t

    return out.clamp(0.0, 255.0)


@torch.no_grad()
def overlay_detection_and_sensitive_region_boxes(
    frame: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    det_threshold: float = 0.7,
    mask_threshold: float = 0.85,
    line_width: int = 3,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.0,
    min_area: int = 50,
) -> torch.Tensor:
    """Overlay both detection boxes and sensitive-region boxes."""
    detected = overlay_detection_boxes(
        frame, device=device, threshold=det_threshold,
        line_width=line_width, color=color, alpha=alpha,
    )
    combined = overlay_sensitive_boxes(
        detected, mask, threshold=mask_threshold,
        line_width=line_width, color=color, alpha=alpha,
        min_area=min_area,
    )
    return combined


@torch.no_grad()
def overlay_sensitive_boxes(
    frame: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.85,
    line_width: int = 3,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.0,
    min_area: int = 50,
) -> torch.Tensor:
    """Overlay small bounding boxes around sensitive regions."""
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    frame = frame.clone()
    device = frame.device
    color_t = torch.tensor(color, device=device, dtype=frame.dtype).view(1, 3, 1, 1)

    sens = mask.squeeze(1)
    sens_min = float(sens.min().item())
    sens_max = float(sens.max().item())
    if sens_max > sens_min:
        threshold_value = sens_min + threshold * (sens_max - sens_min)
    else:
        threshold_value = threshold

    bin_mask = (sens > threshold_value).cpu().numpy().astype(np.uint8)
    h, w = bin_mask.shape[1:]
    visited = np.zeros((h, w), dtype=np.bool_)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    bboxes = []

    for y in range(h):
        for x in range(w):
            if bin_mask[0, y, x] == 1 and not visited[y, x]:
                stack = [(y, x)]
                visited[y, x] = True
                coords = []
                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))
                    for dy, dx in dirs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and bin_mask[0, ny, nx] == 1:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                if len(coords) >= min_area:
                    ys = [p[0] for p in coords]
                    xs = [p[1] for p in coords]
                    y0, y1 = min(ys), max(ys)
                    x0, x1 = min(xs), max(xs)
                    bboxes.append((y0, y1, x0, x1))

    if not bboxes:
        return frame.clamp(0.0, 255.0)

    for (y0, y1, x0, x1) in bboxes:
        y1 = min(y1, frame.size(2) - 1)
        x1 = min(x1, frame.size(3) - 1)
        if alpha > 0:
            frame[:, :, y0 : y1 + 1, x0 : x1 + 1] = (
                frame[:, :, y0 : y1 + 1, x0 : x1 + 1] * (1.0 - alpha)
                + color_t * alpha
            )
        frame[:, :, y0 : y0 + line_width, x0 : x1 + 1] = color_t
        frame[:, :, y1 - line_width + 1 : y1 + 1, x0 : x1 + 1] = color_t
        frame[:, :, y0 : y1 + 1, x0 : x0 + line_width] = color_t
        frame[:, :, y0 : y1 + 1, x1 - line_width + 1 : x1 + 1] = color_t

    return frame.clamp(0.0, 255.0)


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

    print(f"[attack] device={device}, data_root={data_root}")

    print("[attack] building gallery dataset...")
    ds_g = DrivingClipDataset(
        data_root, split=attack_cfg.get("gallery_split", "train"),
        clip_len=clip_len, sample_mode="uniform",
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        max_clips=int(attack_cfg.get("max_gallery", 200)),
    )
    print(f"[attack] gallery dataset size={len(ds_g)}")

    print("[attack] building query dataset...")
    ds_q = DrivingClipDataset(
        data_root, split=attack_cfg.get("query_split", "val"),
        clip_len=clip_len, sample_mode="uniform",
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        max_clips=int(attack_cfg.get("max_query", 50)),
    )
    print(f"[attack] query dataset size={len(ds_q)}")

    gallery_imgs, gallery_ids = [], []
    print("[attack] sampling gallery images...")
    for idx, s in enumerate(ds_g):
        gallery_imgs.append(s.frames[0])
        gallery_ids.append(s.clip_id)
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[attack]  gallery clip {idx+1}/{len(ds_g)} id={s.clip_id}")
    gallery_imgs = torch.stack(gallery_imgs, dim=0)
    print(f"[attack] gallery tensor shape={tuple(gallery_imgs.shape)}")

    query_imgs, query_ids = [], []
    print("[attack] sampling query images...")
    for idx, s in enumerate(ds_q):
        query_imgs.append(s.frames[0])
        query_ids.append(s.clip_id)
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[attack]  query clip {idx+1}/{len(ds_q)} id={s.clip_id}")
    query_imgs = torch.stack(query_imgs, dim=0)
    print(f"[attack] query tensor shape={tuple(query_imgs.shape)}")

    rcfg = RetrievalConfig(
        backbone=str(attack_cfg.get("backbone", "resnet18")),
        device=str(device),
        normalize=bool(attack_cfg.get("normalize", True)),
        input_size=int(attack_cfg.get("input_size", 224)),
        topk=tuple(attack_cfg.get("topk", [1, 5, 10])),
    )

    print(f"[attack] building embedder backbone={rcfg.backbone}")
    embedder = make_default_embedder(rcfg)

    print("[attack] computing gallery embeddings...")
    gallery_emb = build_gallery_embeddings(rcfg, embedder, gallery_imgs)
    print(f"[attack] gallery embedding shape={tuple(gallery_emb.shape)}")

    print("[attack] running retrieval on queries...")
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
    p.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to YAML config")

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
