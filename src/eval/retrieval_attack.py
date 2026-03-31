# ppedcrf/eval/retrieval_attack.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


@dataclass
class RetrievalConfig:
    backbone: str = "resnet18"   # "resnet18" | "resnet50" | "vgg16" | "vit_b_16" | "clip_vitl14" | "dinov2_vitb14" | "yolox_s" | "yolov11n" | ...
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
            self.features = nn.Sequential(*list(m.children())[:-1])  # -> (B,dim,1,1)
            self.out_dim = dim
            self._pool_flatten = True
            return
        if backbone == "vgg16":
            m = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
            # Take penultimate classifier features before logits (4096-d)
            self.features = nn.Sequential(m.features, m.avgpool)
            self.proj = nn.Sequential(*list(m.classifier.children())[:-1])
            self.out_dim = 4096
            self._pool_flatten = False
            return
        if backbone == "vit_b_16":
            m = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
            # Keep encoder trunk and use class token output before classifier head.
            self.vit = m
            self.out_dim = 768
            self._is_vit = True
            return
        if backbone.startswith("clip_"):
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from transformers import CLIPModel
            tag = backbone.replace("clip_", "")
            model_map = {
                "vitl14": "openai/clip-vit-large-patch14",
                "vitb32": "openai/clip-vit-base-patch32",
                "vitb16": "openai/clip-vit-base-patch16",
            }
            if tag not in model_map:
                raise ValueError(f"Unknown CLIP variant '{tag}'. Supported: {list(model_map)}")
            hf_id = model_map[tag]
            clip_model = CLIPModel.from_pretrained(hf_id)
            clip_model.eval()
            self.clip_vision = clip_model.vision_model
            self.clip_proj = clip_model.visual_projection
            self.out_dim = clip_model.config.projection_dim
            self._is_clip = True
            return
        if backbone.startswith("dinov2_"):
            # e.g. dinov2_vitb14, dinov2_vitl14
            tag = backbone  # torch.hub name is used directly
            self.dino = torch.hub.load("facebookresearch/dinov2", tag)
            self.dino.eval()
            self.out_dim = self.dino.embed_dim
            self._is_dino = True
            return
        else:
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            dim = 512
            self.features = nn.Sequential(*list(m.children())[:-1])  # -> (B,dim,1,1)
            self.out_dim = dim
            self._pool_flatten = True
            return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1] recommended
        if getattr(self, "_is_clip", False):
            vis_out = self.clip_vision(pixel_values=x)
            return self.clip_proj(vis_out.pooler_output)

        if getattr(self, "_is_dino", False):
            return self.dino(x)

        if getattr(self, "_is_vit", False):
            # Follow torchvision ViT forward path up to pre-logits features.
            n = x.shape[0]
            x = self.vit._process_input(x)
            cls_token = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = self.vit.encoder(x)
            return x[:, 0]

        if getattr(self, "_pool_flatten", False):
            return self.features(x).flatten(1)

        # VGG16 branch
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


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
        weight_name = f"yolo11{suffix}-cls.pt"
        local_weight = Path(__file__).resolve().parents[1] / weight_name
        weight = str(local_weight) if local_weight.exists() else weight_name
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
