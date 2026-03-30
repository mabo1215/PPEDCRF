"""Generate synthetic monitoring and driving data for pipeline testing.

This creates structured images (not random noise) so that:
- Different "locations" have visually distinct patterns
- Multiple frames per location share spatial structure
- The retrieval embedder can produce meaningful (non-random) features
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_ROOT)


def make_scene_image(
    h: int, w: int, scene_id: int, frame_idx: int, rng: np.random.RandomState
) -> np.ndarray:
    """Create a structured synthetic scene image.

    Each scene_id produces a distinctive spatial pattern (color gradients,
    block structures) so that an image embedder can differentiate scenes.
    Frame index adds small temporal variation within a scene.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Base hue rotation per scene
    base_hue = (scene_id * 37) % 360  # golden-angle-ish spacing

    # Create block structure unique to this scene
    block_h = max(1, h // (3 + scene_id % 4))
    block_w = max(1, w // (3 + (scene_id * 7) % 5))

    for by in range(0, h, block_h):
        for bx in range(0, w, block_w):
            block_idx = (by // block_h) * (w // block_w + 1) + (bx // block_w)
            seed_val = scene_id * 1000 + block_idx
            local_rng = np.random.RandomState(seed_val)
            r = local_rng.randint(30, 255)
            g = local_rng.randint(30, 255)
            b = local_rng.randint(30, 255)
            img[by : by + block_h, bx : bx + block_w] = [r, g, b]

    # Add gradient overlay based on scene_id
    gy = np.linspace(0, 1, h).reshape(h, 1, 1)
    gx = np.linspace(0, 1, w).reshape(1, w, 1)
    angle = base_hue / 360.0 * 2 * np.pi
    grad = (np.sin(gy * np.pi * (2 + scene_id % 3) + angle) * 0.3 +
            np.cos(gx * np.pi * (2 + (scene_id * 3) % 4) + angle) * 0.3)
    grad = (grad * 40).astype(np.int16)
    img = np.clip(img.astype(np.int16) + grad, 0, 255).astype(np.uint8)

    # Small per-frame temporal variation (simulates camera movement)
    noise_scale = 3 + (frame_idx % 5)
    noise = rng.randint(-noise_scale, noise_scale + 1, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_monitoring_data(root: str, n_scenes: int, frames_per_scene: int,
                              h: int, w: int) -> None:
    """Generate synthetic monitoring frames: <clip_id>_frame<number>.jpg"""
    os.makedirs(root, exist_ok=True)
    rng = np.random.RandomState(42)

    try:
        import cv2
        use_cv2 = True
    except ImportError:
        from PIL import Image
        use_cv2 = False

    total = 0
    for sid in range(n_scenes):
        clip_id = f"scene_{sid:04d}"
        for fid in range(frames_per_scene):
            img = make_scene_image(h, w, sid, fid, rng)
            fname = f"{clip_id}_frame{fid:05d}.jpg"
            fpath = os.path.join(root, fname)
            if use_cv2:
                cv2.imwrite(fpath, img[:, :, ::-1])  # RGB to BGR
            else:
                Image.fromarray(img).save(fpath, quality=90)
            total += 1

    print(f"Generated {total} monitoring frames in {root} "
          f"({n_scenes} scenes x {frames_per_scene} frames)")


def generate_driving_data(root: str, n_clips: int, frames_per_clip: int,
                           h: int, w: int) -> None:
    """Generate synthetic driving clips as frame folders."""
    rng = np.random.RandomState(123)

    try:
        import cv2
        use_cv2 = True
    except ImportError:
        from PIL import Image
        use_cv2 = False

    for split in ["train", "val"]:
        split_dir = os.path.join(root, split)
        os.makedirs(split_dir, exist_ok=True)
        total = 0
        offset = 0 if split == "train" else n_clips
        for cid in range(n_clips):
            clip_dir = os.path.join(split_dir, f"clip_{offset + cid:04d}")
            os.makedirs(clip_dir, exist_ok=True)
            for fid in range(frames_per_clip):
                img = make_scene_image(h, w, offset + cid, fid, rng)
                fname = f"{fid + 1:06d}.jpg"
                fpath = os.path.join(clip_dir, fname)
                if use_cv2:
                    cv2.imwrite(fpath, img[:, :, ::-1])
                else:
                    Image.fromarray(img).save(fpath, quality=90)
                total += 1
        print(f"Generated {total} driving frames in {split_dir} "
              f"({n_clips} clips x {frames_per_clip} frames)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for PPEDCRF experiments")
    parser.add_argument("--monitoring_root", type=str,
                        default=r"F:\work\datasets\monitoring\images",
                        help="Output directory for monitoring frames")
    parser.add_argument("--driving_root", type=str,
                        default="src/data/driving",
                        help="Output directory for driving clips")
    parser.add_argument("--n_monitoring_scenes", type=int, default=60,
                        help="Number of distinct monitoring scenes")
    parser.add_argument("--monitoring_frames", type=int, default=12,
                        help="Frames per monitoring scene")
    parser.add_argument("--n_driving_clips", type=int, default=10,
                        help="Number of driving clips per split")
    parser.add_argument("--driving_frames", type=int, default=10,
                        help="Frames per driving clip")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=384)
    args = parser.parse_args()

    print("=== Generating Monitoring Data ===")
    generate_monitoring_data(
        args.monitoring_root,
        args.n_monitoring_scenes,
        args.monitoring_frames,
        args.height,
        args.width,
    )

    print("\n=== Generating Driving Data ===")
    generate_driving_data(
        args.driving_root,
        args.n_driving_clips,
        args.driving_frames,
        args.height,
        args.width,
    )


if __name__ == "__main__":
    main()
