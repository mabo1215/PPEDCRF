from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from datasets.driving_clip_dataset import ClipSample, _read_image, _resize_if_needed


FRAME_RE = re.compile(r"(?P<clip_id>.+)_frame(?P<frame_idx>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def index_monitoring_sequences(
    root: str,
    min_frames: int = 6,
) -> Dict[str, List[Path]]:
    """
    Group monitoring frames by sequence id.

    Expected filenames:
      <clip_id>_frame<number>.jpg
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Monitoring root not found: {root}")

    grouped: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for path in sorted(root_path.iterdir()):
        if not path.is_file():
            continue
        match = FRAME_RE.match(path.name)
        if match is None:
            continue
        grouped[match.group("clip_id")].append((int(match.group("frame_idx")), path))

    indexed: Dict[str, List[Path]] = {}
    for clip_id, items in grouped.items():
        if len(items) < int(min_frames):
            continue
        indexed[clip_id] = [path for _, path in sorted(items, key=lambda item: item[0])]

    if not indexed:
        raise RuntimeError(f"No eligible monitoring sequences were found in: {root}")
    return indexed


class MonitoringClipDataset(Dataset):
    """
    A lightweight sequence dataset for controlled retrieval/privacy experiments.

    Each sequence is split into an early gallery view and a later query view so
    the retrieval task matches the same scene under temporal variation.
    """

    def __init__(
        self,
        root: str,
        clip_ids: Optional[Sequence[str]] = None,
        view: str = "gallery",  # "gallery" | "query" | "full"
        clip_len: int = 4,
        resize_hw: Optional[Tuple[int, int]] = (256, 384),
        min_frames: int = 6,
        sample_mode: str = "uniform",
    ):
        self.root = root
        self.view = view
        self.clip_len = int(clip_len)
        self.resize_hw = resize_hw
        self.sample_mode = sample_mode

        indexed = index_monitoring_sequences(root=root, min_frames=min_frames)
        available_ids = sorted(indexed.keys())

        if clip_ids is None:
            self.clip_ids = available_ids
        else:
            clip_id_set = set(clip_ids)
            self.clip_ids = [clip_id for clip_id in available_ids if clip_id in clip_id_set]
        if not self.clip_ids:
            raise RuntimeError("No matching monitoring sequences were selected.")

        self.sequence_paths = {clip_id: self._select_view(indexed[clip_id]) for clip_id in self.clip_ids}

    def __len__(self) -> int:
        return len(self.clip_ids)

    def _select_view(self, paths: List[Path]) -> List[Path]:
        if self.view == "full":
            return paths

        split = max(3, len(paths) // 2)
        if self.view == "gallery":
            selected = paths[:split]
        elif self.view == "query":
            selected = paths[-split:]
        else:
            raise ValueError(f"Unsupported monitoring view: {self.view}")

        if not selected:
            raise RuntimeError(f"Sequence split for view={self.view} is empty.")
        return selected

    def _sample_indices(self, total_frames: int) -> List[int]:
        if total_frames <= 0:
            return [0] * self.clip_len

        if total_frames >= self.clip_len:
            if self.sample_mode == "random":
                start_max = max(0, total_frames - self.clip_len)
                start = torch.randint(low=0, high=start_max + 1, size=(1,)).item()
                return [start + i for i in range(self.clip_len)]

            if self.clip_len == 1:
                return [total_frames // 2]

            return torch.linspace(0, total_frames - 1, steps=self.clip_len).round().long().tolist()

        return [min(i, total_frames - 1) for i in range(self.clip_len)]

    def __getitem__(self, index: int) -> ClipSample:
        clip_id = self.clip_ids[index]
        frame_paths = self.sequence_paths[clip_id]
        indices = self._sample_indices(len(frame_paths))

        frames = []
        for idx in indices:
            frame = _read_image(str(frame_paths[idx]))
            frames.append(_resize_if_needed(frame, self.resize_hw))

        frames_t = torch.stack(frames, dim=0)
        extra = {
            "root": self.root,
            "view": self.view,
            "total_frames_est": len(frame_paths),
        }
        return ClipSample(
            frames=frames_t,
            clip_id=clip_id,
            path=str(frame_paths[0]),
            indices=indices,
            extra=extra,
        )


def split_monitoring_ids(
    root: str,
    num_queries: int,
    max_gallery: int,
    min_frames: int = 6,
) -> Tuple[List[str], List[str]]:
    """
    Select deterministic query ids and extra distractor ids for a benchmark.
    """
    indexed = index_monitoring_sequences(root=root, min_frames=min_frames)
    ordered = sorted(indexed.keys())
    if len(ordered) < num_queries:
        raise RuntimeError(
            f"Requested {num_queries} query sequences but only {len(ordered)} eligible sequences were found."
        )

    query_ids = ordered[:num_queries]
    remaining = ordered[num_queries:]
    num_distractors = max(0, max_gallery - num_queries)
    distractor_ids = remaining[:num_distractors]
    return query_ids, distractor_ids


def get_clip_ids(dataset: MonitoringClipDataset) -> List[str]:
    return list(dataset.clip_ids)


def iter_clip_ids(root: str, min_frames: int = 6) -> Iterable[str]:
    return sorted(index_monitoring_sequences(root=root, min_frames=min_frames).keys())
