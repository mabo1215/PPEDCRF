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


SANITIZERS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "none": identity,
    "jpeg75": lambda f: jpeg_recompress(f, 75),
    "jpeg50": lambda f: jpeg_recompress(f, 50),
    "blur": lambda f: gaussian_blur(f, 1.0),
    "denoise": light_denoise,
}
