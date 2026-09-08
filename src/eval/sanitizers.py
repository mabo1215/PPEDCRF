"""Attacker-side input sanitization for the adaptive-adversary study (G2, tier 1).

The direction-transfer study asks whether a perturbation aligned with a
surrogate ensemble's gradient still suppresses retrieval on a held-out
attacker. That attacker is otherwise passive: it embeds whatever frame it
receives. This module gives it one more degree of freedom -- a cheap,
untrained preprocessing step applied before embedding -- to test whether the
directional advantage survives an attacker who suspects the frame has been
adversarially perturbed and tries to scrub it first. None of these operators
require training or attacker-side access to the defense.

Every function takes and returns a (1,3,H,W) float tensor in [0,255], RGB,
matching the frame representation used throughout eval/retrieval_attack.py.
"""
from __future__ import annotations

import zlib
from typing import Callable, Dict

import cv2
import numpy as np
import torch


def _to_uint8_hwc(frame: torch.Tensor) -> np.ndarray:
    arr = frame.detach().to("cpu").clamp(0, 255).round().byte().numpy()
    return np.transpose(arr[0], (1, 2, 0))  # HWC, RGB


def _from_uint8_hwc(arr: np.ndarray, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    chw = np.ascontiguousarray(np.transpose(arr, (2, 0, 1))).astype(np.float32)
    return torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=dtype)


def identity(frame: torch.Tensor) -> torch.Tensor:
    return frame


def jpeg_recompress(frame: torch.Tensor, quality: int) -> torch.Tensor:
    """Re-encode through JPEG at the given quality and decode back."""
    rgb = _to_uint8_hwc(frame)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    rgb_out = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return _from_uint8_hwc(rgb_out, frame.device, frame.dtype)


def gaussian_blur(frame: torch.Tensor, sigma: float) -> torch.Tensor:
    rgb = _to_uint8_hwc(frame)
    k = max(3, int(2 * round(3 * sigma) + 1))
    out = cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma)
    return _from_uint8_hwc(out, frame.device, frame.dtype)


def light_denoise(frame: torch.Tensor) -> torch.Tensor:
    """A classical (non-learned) denoiser -- no attacker retraining needed."""
    rgb = _to_uint8_hwc(frame)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    out = cv2.fastNlMeansDenoisingColored(bgr, None, h=7, hColor=7,
                                          templateWindowSize=7,
                                          searchWindowSize=21)
    rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return _from_uint8_hwc(rgb_out, frame.device, frame.dtype)


def median_filter(frame: torch.Tensor, ksize: int) -> torch.Tensor:
    rgb = _to_uint8_hwc(frame)
    out = cv2.medianBlur(rgb, int(ksize))
    return _from_uint8_hwc(out, frame.device, frame.dtype)


def resize_round_trip(frame: torch.Tensor, factor: float) -> torch.Tensor:
    """Bilinear downscale by `factor` and back to the original size."""
    rgb = _to_uint8_hwc(frame)
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, int(round(w * factor))),
                             max(1, int(round(h * factor)))),
                       interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return _from_uint8_hwc(out, frame.device, frame.dtype)


def bit_depth_reduce(frame: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantise each channel to `bits` bits and rescale to [0, 255]."""
    levels = float(2 ** int(bits) - 1)
    rgb = _to_uint8_hwc(frame).astype(np.float32) / 255.0
    out = np.round(rgb * levels) / levels * 255.0
    return _from_uint8_hwc(out.round().astype(np.uint8), frame.device,
                           frame.dtype)


# The four transforms the EOT-hardened direction is optimised over. Everything
# after them in HELD_OUT is an attacker-side operator the optimiser never saw,
# which is what a fair test of hardening has to include.
TRAINED = ("jpeg75", "jpeg50", "blur", "denoise")

SANITIZERS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "none": identity,
    "jpeg75": lambda f: jpeg_recompress(f, 75),
    "jpeg50": lambda f: jpeg_recompress(f, 50),
    "blur": lambda f: gaussian_blur(f, 1.0),
    "denoise": light_denoise,
    # in-family parameter not trained on: interpolation, not extrapolation
    "jpeg60": lambda f: jpeg_recompress(f, 60),
    # held out: outside the trained family or beyond its parameter range
    "jpeg30": lambda f: jpeg_recompress(f, 30),
    "median3": lambda f: median_filter(f, 3),
    "resize_half": lambda f: resize_round_trip(f, 0.5),
    "blur2": lambda f: gaussian_blur(f, 2.0),
    "bitdepth4": lambda f: bit_depth_reduce(f, 4),
}

HELD_OUT = ("jpeg60", "jpeg30", "median3", "resize_half", "blur2", "bitdepth4")


# The pool `random_one` draws from. Fixed at definition time and holding only
# the simple operators: drawing from HELD_OUT instead would include
# `random_one` itself once the composites below are appended to it, and a
# frame whose checksum selected that slot would recurse without end.
RANDOM_ONE_POOL = TRAINED + HELD_OUT


def random_one(frame: torch.Tensor) -> torch.Tensor:
    """Apply one operator chosen uniformly from the trained and held-out sets.

    The choice is seeded by the frame's own content, so the same released
    frame always meets the same operator and the study stays reproducible
    without threading a generator through the evaluation loop.
    """
    key = zlib.crc32(_to_uint8_hwc(frame).tobytes()) & 0x7FFFFFFF
    return SANITIZERS[RANDOM_ONE_POOL[key % len(RANDOM_ONE_POOL)]](frame)


def jpeg50_then_blur(frame: torch.Tensor) -> torch.Tensor:
    """Two trained operators stacked -- a composition the optimiser never saw."""
    return gaussian_blur(jpeg_recompress(frame, 50), 1.0)


SANITIZERS["random_one"] = random_one
SANITIZERS["jpeg50_blur"] = jpeg50_then_blur
HELD_OUT = HELD_OUT + ("random_one", "jpeg50_blur")
