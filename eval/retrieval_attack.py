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
    backbone: str = "resnet18"   # "resnet18" | "resnet50" | "yolox_s" | "yolov11n" | "yolov11s" ...
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


class YOLOXEmbedder(nn.Module):
    """
    Use YOLOX backbone (e.g. yolox_s) as an image feature extractor.
    The output is a global pooled concatenation of FPN feature maps.
    """

    def __init__(self, name: str = "yolox_s"):
        super().__init__()
        try:
            from yolox.exp import get_exp  # type: ignore
        except ImportError as e:
            raise ImportError(
                "YOLOX is not installed. Please run `pip install yolox` to use YOLOX backbone."
            ) from e

        self.exp_name = name
        exp = get_exp(None, name)
        model = exp.get_model()
        model.eval()
        self.backbone = model.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,255] or [0,1]
        if x.max() > 1.5:
            x = x / 255.0

        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        pooled = [F.adaptive_avg_pool2d(f, (1, 1)).flatten(1) for f in feats]
        emb = torch.cat(pooled, dim=1)
        return emb


class YOLOv11Embedder(nn.Module):
    """
    Use Ultralytics YOLOv11 as an image feature extractor.
    Loads yolo11n-cls.pt / yolo11s-cls.pt etc. (224 input) and uses features before the classifier.
    Only the backbone is kept as a submodule so that .eval() does not trigger Ultralytics trainer/dataset.
    """

    def __init__(self, name: str = "yolov11n"):
        super().__init__()
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Ultralytics is not installed. Please run `pip install ultralytics` to use YOLOv11 backbone."
            ) from e

        # Map config name to Ultralytics classification weight: yolov11n -> yolo11n-cls.pt
        suffix = name.lower().replace("yolo11", "").replace("yolov11", "").strip("_") or "n"
        if suffix not in ("n", "s", "m", "l", "x"):
            suffix = "n"
        weight = f"yolo11{suffix}-cls.pt"
        yolo = YOLO(weight)
        model = yolo.model

        # Get backbone (all but classification head). Structure varies by ultralytics version.
        backbone = None
        if hasattr(model, "model"):
            inner = getattr(model, "model")
            if isinstance(inner, nn.Sequential):
                layers = list(inner.children())
                if layers:
                    backbone = nn.Sequential(*layers[:-1])
            elif hasattr(inner, "children"):
                layers = list(inner.children())
                if len(layers) > 1:
                    backbone = nn.Sequential(*layers[:-1])
        if backbone is None and hasattr(model, "children"):
            layers = list(model.children())
            if len(layers) > 1:
                backbone = nn.Sequential(*layers[:-1])
        if backbone is None:
            backbone = nn.Sequential(*list(model.children())[:-1]) if len(list(model.children())) > 1 else model

        backbone.eval()
        with torch.no_grad():
            out = backbone(torch.zeros(1, 3, 224, 224))
            out_dim = out.flatten(1).size(1)

        # Register only backbone so embedder.eval() does not call Ultralytics model.eval() (which starts trainer/dataset)
        self.backbone = backbone
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.5:
            x = x / 255.0
        return self.backbone(x).flatten(1)


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
    b = cfg.backbone.lower()
    if b.startswith("yolov11") or (b.startswith("yolo11") and "cls" not in b):
        return YOLOv11Embedder(cfg.backbone)
    if b.startswith("yolox"):
        return YOLOXEmbedder(cfg.backbone)
    return ImageEmbedder(cfg.backbone)