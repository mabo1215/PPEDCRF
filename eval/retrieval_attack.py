# ppedcrf/eval/retrieval_attack.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


@dataclass
class RetrievalConfig:
    backbone: str = "resnet18"   # "resnet18" | "resnet50"
    device: str = "cuda"
    normalize: bool = True       # L2 normalize embeddings
    input_size: int = 224        # embedding network input resolution
    topk: Tuple[int, ...] = (1, 5, 10)


class ImageEmbedder(nn.Module):
    """
    Simple embedding model based on torchvision backbones.
    """

    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet50":
            m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            dim = 2048
        else:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            dim = 512

        # remove classifier
        self.features = nn.Sequential(*list(m.children())[:-1])  # -> (B,dim,1,1)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1] recommended
        f = self.features(x).flatten(1)  # (B,dim)
        return f


def preprocess_for_embed(x: torch.Tensor, input_size: int = 224) -> torch.Tensor:
    """
    x: (B,3,H,W) float32, either [0,255] or [0,1]
    returns: (B,3,input_size,input_size) normalized for imagenet
    """
    if x.max() > 1.5:
        x = x / 255.0

    x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


@torch.no_grad()
def build_gallery_embeddings(
    cfg: RetrievalConfig,
    embedder: nn.Module,
    gallery_images: torch.Tensor,     # (N,3,H,W)
) -> torch.Tensor:
    embedder.eval()
    dev = torch.device(cfg.device)
    embedder = embedder.to(dev)

    imgs = gallery_images.to(dev)
    imgs = preprocess_for_embed(imgs, cfg.input_size)
    emb = embedder(imgs)  # (N,D)
    if cfg.normalize:
        emb = F.normalize(emb, dim=1)
    return emb


@torch.no_grad()
def query_topk(
    cfg: RetrievalConfig,
    query_images: torch.Tensor,       # (M,3,H,W)
    query_ids: List[str],             # length M
    gallery_emb: torch.Tensor,        # (N,D)
    gallery_ids: List[str],           # length N
    embedder: nn.Module,
) -> Dict[str, float]:
    """
    Evaluate top-k retrieval accuracy:
      correct if the true id appears in top-k results.

    Returns:
      dict like {"R@1":..., "R@5":..., ...}
    """
    assert len(query_ids) == query_images.size(0)
    assert len(gallery_ids) == gallery_emb.size(0)

    dev = torch.device(cfg.device)
    embedder.eval().to(dev)

    q = query_images.to(dev)
    q = preprocess_for_embed(q, cfg.input_size)
    q_emb = embedder(q)  # (M,D)
    if cfg.normalize:
        q_emb = F.normalize(q_emb, dim=1)

    # similarity (cosine if normalized)
    sim = q_emb @ gallery_emb.t()  # (M,N)

    results = {}
    for k in cfg.topk:
        topk_idx = sim.topk(k=min(k, sim.size(1)), dim=1).indices  # (M,k)
        correct = 0
        for i in range(sim.size(0)):
            true_id = query_ids[i]
            retrieved_ids = [gallery_ids[j] for j in topk_idx[i].tolist()]
            if true_id in retrieved_ids:
                correct += 1
        results[f"R@{k}"] = correct / max(1, sim.size(0))

    return results


def make_default_embedder(cfg: RetrievalConfig) -> nn.Module:
    return ImageEmbedder(cfg.backbone)