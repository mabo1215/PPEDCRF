# ppedcrf/datasets/driving_clip_dataset.py
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

try:
    import cv2  # optional but recommended
except Exception:
    cv2 = None


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass
class ClipSample:
    frames: torch.Tensor  # (T,3,H,W) float32 in [0,255]
    clip_id: str
    path: str
    indices: List[int]
    extra: Dict[str, Union[str, int, float]]


def _list_frame_files(frame_dir: str) -> List[str]:
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(frame_dir, f"*{ext}")))
    # natural sort by filename
    files = sorted(files)
    return files


def _read_image_cv2(path: str) -> torch.Tensor:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read images/videos. Please install opencv-python.")
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,H,W) uint8
    return t.float()


def _read_video_cv2(path: str, indices: List[int]) -> List[torch.Tensor]:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read images/videos. Please install opencv-python.")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            # If we cannot read, break and let caller pad
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float()
        frames.append(t)

    cap.release()
    return frames


def _resize_if_needed(fr: torch.Tensor, size_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
    if size_hw is None:
        return fr
    h, w = size_hw
    # TF.resize expects (C,H,W) float tensor OK
    return TF.resize(fr, [h, w], antialias=True)


class DrivingClipDataset(Dataset):
    """
    Driving clip dataset loader.

    Directory layout (recommended):
      root/
        train/
          clip_0001/000001.jpg ...
          clip_0002.mp4
        val/
        test/

    Each item returns:
      ClipSample(frames=(T,3,H,W) float32 [0,255], clip_id, path, indices, extra)
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 8,
        stride: int = 1,
        sample_mode: str = "uniform",  # "uniform" or "random"
        resize_hw: Optional[Tuple[int, int]] = (384, 640),
        max_clips: Optional[int] = None,
    ):
        self.root = root
        self.split = split
        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.sample_mode = str(sample_mode)
        self.resize_hw = resize_hw

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Collect clip sources: frame folders OR video files
        sources = []
        for p in sorted(os.listdir(split_dir)):
            full = os.path.join(split_dir, p)
            if os.path.isdir(full):
                # frame folder
                frame_files = _list_frame_files(full)
                if len(frame_files) >= 1:
                    sources.append(("frames", full))
            else:
                # maybe video file
                ext = os.path.splitext(full)[1].lower()
                if ext in VID_EXTS:
                    sources.append(("video", full))

        # Also支持你这种“train 下面全是 PNG”的情况：
        # 如果没找到任何子文件夹 / 视频，但 split_dir 里直接有图片，就把整个 split_dir 当成 1 个 clip。
        if len(sources) == 0:
            flat_frames = _list_frame_files(split_dir)
            if len(flat_frames) >= 1:
                sources.append(("frames", split_dir))

        if max_clips is not None:
            sources = sources[: int(max_clips)]
        self.sources = sources

        if len(self.sources) == 0:
            raise RuntimeError(f"No clips found under: {split_dir}")

    def __len__(self) -> int:
        return len(self.sources)

    def _sample_indices(self, total_frames: int) -> List[int]:
        # We sample clip_len frames with stride.
        span = (self.clip_len - 1) * self.stride + 1
        if total_frames <= 0:
            return [0] * self.clip_len

        if total_frames >= span:
            if self.sample_mode == "random":
                start = torch.randint(low=0, high=(total_frames - span + 1), size=(1,)).item()
            else:
                start = (total_frames - span) // 2
            idxs = [start + i * self.stride for i in range(self.clip_len)]
        else:
            # not enough frames -> pad by repeating last
            idxs = [min(i * self.stride, total_frames - 1) for i in range(self.clip_len)]

        return idxs

    def __getitem__(self, i: int) -> ClipSample:
        kind, path = self.sources[i]
        clip_id = os.path.splitext(os.path.basename(path))[0]

        if kind == "frames":
            frame_files = _list_frame_files(path)
            total = len(frame_files)
            indices = self._sample_indices(total)

            frames = []
            for idx in indices:
                img_path = frame_files[min(idx, total - 1)]
                fr = _read_image_cv2(img_path)
                fr = _resize_if_needed(fr, self.resize_hw)
                frames.append(fr)

        else:
            # video
            if cv2 is None:
                raise RuntimeError("OpenCV (cv2) is required for video reading.")
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

            indices = self._sample_indices(total)
            frames = _read_video_cv2(path, indices)

            # pad if short
            if len(frames) == 0:
                # fallback: return zeros
                h, w = self.resize_hw if self.resize_hw else (384, 640)
                frames = [torch.zeros((3, h, w), dtype=torch.float32) for _ in range(self.clip_len)]
            else:
                last = frames[-1]
                while len(frames) < self.clip_len:
                    frames.append(last.clone())

            frames = [_resize_if_needed(fr, self.resize_hw) for fr in frames]

        frames_t = torch.stack(frames, dim=0)  # (T,3,H,W) float32 in [0,255]
        extra = {"split": self.split, "kind": kind, "total_frames_est": int(total)}
        return ClipSample(frames=frames_t, clip_id=clip_id, path=path, indices=indices, extra=extra)