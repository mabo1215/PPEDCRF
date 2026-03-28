# run_train.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.driving_clip_dataset import DrivingClipDataset

try:
    from huggingface_hub import PyTorchModelHubMixin
    _HF_MIXIN = (PyTorchModelHubMixin,)
except ImportError:
    _HF_MIXIN = ()


def _identity_collate(batch):
    return batch


@dataclass
class TrainConfig:
    data_root: str
    out_dir: str = "outputs"
    device: str = "cuda"
    epochs: int = 5
    batch_size: int = 2
    num_workers: int = 2
    lr: float = 1e-4

    clip_len: int = 8
    resize_hw: Tuple[int, int] = (384, 640)

    mask_root: Optional[str] = None


class SensitiveRegionNet(nn.Module, *_HF_MIXIN):
    """
    Sensitive-region predictor (unary logits for PPEDCRF).
    When huggingface_hub is installed, supports save_pretrained / from_pretrained / push_to_hub.
    """
    def _save_pretrained(self, save_directory):
        import pathlib
        p = pathlib.Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "pytorch_model.bin")

    @classmethod
    def _from_pretrained(cls, model_id, revision=None, force_download=False, token=None, cache_dir=None, local_files_only=False, **model_kwargs):
        from huggingface_hub import hf_hub_download
        map_location = model_kwargs.pop("map_location", "cpu")
        path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin", revision=revision, force_download=force_download, token=token, cache_dir=cache_dir, local_files_only=local_files_only)
        model = cls()
        model.load_state_dict(torch.load(path, map_location=map_location, weights_only=True))
        return model

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() > 1.5:
            x = x / 255.0
        f = self.enc(x)
        y = self.dec(f)
        return y  # (B,1,H,W) logits


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_frame_masks(mask_root: str, split: str, clip_id: str, indices: list, device: torch.device) -> torch.Tensor:
    import cv2
    masks = []
    for idx in indices:
        mp = os.path.join(mask_root, split, clip_id, f"{idx:06d}.png")
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {mp}")
        t = torch.from_numpy(m).unsqueeze(0).float() / 255.0
        masks.append(t)
    return torch.stack(masks, dim=0).to(device)  # (T,1,H,W)


def train(cfg: TrainConfig) -> str:
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _ensure_dir(cfg.out_dir)

    ds = DrivingClipDataset(
        root=cfg.data_root,
        split="train",
        clip_len=cfg.clip_len,
        sample_mode="random",
        resize_hw=cfg.resize_hw,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_identity_collate,
    )

    model = SensitiveRegionNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()

    global_step = 0
    model.train()

    for epoch in range(cfg.epochs):
        for batch in dl:
            frames_list = []
            masks_list = []

            for sample in batch:
                frames = sample.frames.to(dev)  # (T,3,H,W)
                T = frames.size(0)
                frames_list.append(frames)

                if cfg.mask_root is None:
                    masks = torch.zeros((T, 1, frames.size(2), frames.size(3)), device=dev)
                else:
                    masks = _load_frame_masks(cfg.mask_root, "train", sample.clip_id, sample.indices, dev)
                    if masks.shape[-2:] != frames.shape[-2:]:
                        masks = F.interpolate(masks, size=frames.shape[-2:], mode="nearest")
                masks_list.append(masks)

            frames_b = torch.cat(frames_list, dim=0)
            masks_b = torch.cat(masks_list, dim=0)

            logits = model(frames_b)
            loss = bce(logits, masks_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_step += 1
            if global_step % 50 == 0:
                print(f"[epoch {epoch+1}/{cfg.epochs}] step={global_step} loss={loss.item():.6f}")

        ckpt_path = os.path.join(cfg.out_dir, f"sensnet_epoch_{epoch+1}.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(cfg.out_dir, "sensnet_final.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, final_path)
    print(f"Saved final: {final_path}")
    return final_path


def train_from_cfg_dict(cfg_dict: Dict[str, Any]) -> str:
    """
    Build TrainConfig from the unified YAML config dict.
    Expected keys:
      data.root, data.resize_hw, data.clip_len
      train.out_dir, train.epochs, train.batch_size, train.num_workers, train.lr, train.mask_root
      project.device
    """
    data_root = cfg_dict["data"]["root"]
    resize_hw = tuple(cfg_dict["data"]["resize_hw"])
    clip_len = int(cfg_dict["data"]["clip_len"])

    tc = TrainConfig(
        data_root=data_root,
        out_dir=str(cfg_dict["train"].get("out_dir", "outputs")),
        device=str(cfg_dict["project"].get("device", "cuda")),
        epochs=int(cfg_dict["train"].get("epochs", 5)),
        batch_size=int(cfg_dict["train"].get("batch_size", 2)),
        num_workers=int(cfg_dict["train"].get("num_workers", 2)),
        lr=float(cfg_dict["train"].get("lr", 1e-4)),
        clip_len=clip_len,
        resize_hw=(int(resize_hw[0]), int(resize_hw[1])),
        mask_root=cfg_dict["train"].get("mask_root", None),
    )
    return train(tc)


if __name__ == "__main__":
    # Optional: local direct run using YAML (no CLI).
    from utils.config import load_yaml
    cfg = load_yaml("src/config/config.yaml")
    train_from_cfg_dict(cfg)
