"""
从 train 中分出 20% 作为 val：在数据根目录下创建 val/，并将 train 中 20% 的 clip 移入 val。
与 DrivingClipDataset 一致：train/ 下每项为子目录（帧序列）或视频文件，视为一个 clip。

Usage:
  python src/scripts/split_train_val.py
  python src/scripts/split_train_val.py --data_root C:/work/dataset/driving --ratio 0.2 --dry_run
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_yaml


def main():
    p = argparse.ArgumentParser(description="Split 20% of train into val")
    p.add_argument("--config", type=str, default="src/config/config.yaml")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--ratio", type=float, default=0.2, help="Fraction of train to use as val (default 0.2 = 20%%)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true", help="Only print what would be moved, do not move")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    data_root = args.data_root or cfg["data"]["root"]
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        print(f"Train directory not found: {train_dir}")
        sys.exit(1)

    entries = sorted(os.listdir(train_dir))
    clip_ext = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    img_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    items = []
    for p in entries:
        full = os.path.join(train_dir, p)
        if os.path.isdir(full):
            items.append(p)
        elif os.path.isfile(full) and p.lower().endswith(clip_ext):
            items.append(p)
    # 若没有子目录/视频，则按「直接放在 train 下的图片」处理：20% 的图片移到 val
    if not items:
        for p in entries:
            full = os.path.join(train_dir, p)
            if os.path.isfile(full) and p.lower().endswith(img_ext):
                items.append(p)
    if not items:
        print(f"No clip subdirs, video files, or image files found under {train_dir}")
        sys.exit(1)

    random.seed(args.seed)
    shuffled = items.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.ratio))
    to_val = shuffled[-n_val:]

    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")
    print(f"Total items in train: {len(items)}, moving {n_val} ({100*args.ratio:.0f}%) to val")
    if args.dry_run:
        print("Dry run - would move first", min(10, n_val), "of", n_val, "items:", to_val[:10], "..." if n_val > 10 else "")
        return

    os.makedirs(val_dir, exist_ok=True)
    for name in to_val:
        src = os.path.join(train_dir, name)
        dst = os.path.join(val_dir, name)
        if os.path.exists(dst):
            print(f"Skip (already exists): {name}")
            continue
        shutil.move(src, dst)
        print(f"Moved: {name}")
    print("Done. Val now has", n_val, "items; train has", len(items) - n_val, "items.")


if __name__ == "__main__":
    main()
