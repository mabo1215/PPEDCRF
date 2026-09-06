"""Placement-rule study: does *where* a fixed perturbation budget lands matter?

This extends the energy-matched intervention benchmark from a single learned
support map to a family of placement rules, and adds an attacker-gradient
"oracle" placement that deliberately targets the pixels the attacker embedding
is most sensitive to.

Every placement is renormalised to carry exactly the same sum of squared
weights as the learned map, so any difference in retrieval outcome is
attributable to location rather than magnitude. Two questions follow:

  1. Does any unsupervised placement rule (saliency, edge, centre bias, fixed
     random) beat the learned map or the uniform control at matched energy?
  2. Does an *oracle* placement, built from the attacker's own input gradients,
     beat them? This separates "placement cannot matter at this budget" from
     "placement matters but no practical map finds the right pixels", which are
     very different scientific conclusions.

The script also exports the per-pixel embedding-sensitivity statistics needed
to explain the result: how spatially homogeneous the attacker's sensitivity is,
and how well the learned map correlates with it.

Outputs are written incrementally (one row per completed query/placement,
flushed and fsynced) and completed keys are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    build_gallery_embeddings,
    default_input_size_for_backbone,
    make_default_embedder,
)
from scripts.run_icme2027_mask_intervention import build_modules  # noqa: E402
from scripts.run_controlled_retrieval_benchmark import build_gallery_tensor  # noqa: E402
from scripts.run_tomm_review_proxy import (  # noqa: E402
    _build_proxy_data,
    detailed_retrieval,
)
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)
from eval.metrics import psnr_torch  # noqa: E402
from utils.config import load_yaml  # noqa: E402


PLACEMENT_LABELS = {
    "learned": "learned support (DCRF)",
    "uniform": "uniform same-energy",
    "saliency": "spectral-residual saliency",
    "edge": "gradient-magnitude (edge)",
    "center": "centre-bias prior",
    "random_fixed": "fixed random field",
    "oracle_grad": "attacker-gradient oracle",
    "anti_oracle_grad": "attacker-gradient anti-oracle",
    "segmentation": "semantic background (DeepLabV3)",
    "segmentation_fcn": "semantic background (FCN-ResNet50)",
    "segmentation_ade": "scene structure (SegFormer/ADE20K)",
    "margin_oracle": "attacker margin gradient (positive minus best rival)",
    "anti_margin_oracle": "inverse of the attacker margin gradient",
}
PLACEMENTS = tuple(PLACEMENT_LABELS)


# --------------------------------------------------------------------------
# placement maps (all returned unnormalised; renormalisation happens later)
# --------------------------------------------------------------------------

def _gray(frame: torch.Tensor) -> torch.Tensor:
    """frame: (1,3,H,W) in [0,255] -> (1,1,H,W)."""
    if frame.size(1) == 3:
        return (0.299 * frame[:, 0:1] + 0.587 * frame[:, 1:2] + 0.114 * frame[:, 2:3])
    return frame[:, 0:1]


def saliency_map(frame: torch.Tensor) -> torch.Tensor:
    """Spectral-residual saliency (Hou & Zhang), a classic unsupervised prior."""
    gray = _gray(frame)
    fft = torch.fft.fft2(gray)
    log_amp = torch.log(fft.abs().clamp_min(1e-8))
    phase = torch.angle(fft)
    kernel = torch.ones(1, 1, 3, 3, device=frame.device) / 9.0
    smoothed = F.conv2d(log_amp, kernel, padding=1)
    residual = log_amp - smoothed
    recon = torch.fft.ifft2(torch.exp(residual + 1j * phase))
    sal = recon.abs().square()
    blur = torch.ones(1, 1, 5, 5, device=frame.device) / 25.0
    return F.conv2d(sal, blur, padding=2).clamp_min(0.0)


def edge_map(frame: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude: perturb structurally informative pixels."""
    gray = _gray(frame)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=frame.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return (gx.square() + gy.square()).sqrt().clamp_min(0.0)


def center_map(frame: torch.Tensor, sigma_frac: float = 0.35) -> torch.Tensor:
    """Centre-bias prior: photographers frame the subject centrally."""
    _, _, height, width = frame.shape
    ys = torch.linspace(-1.0, 1.0, height, device=frame.device).view(-1, 1)
    xs = torch.linspace(-1.0, 1.0, width, device=frame.device).view(1, -1)
    dist2 = ys.square() + xs.square()
    return torch.exp(-dist2 / (2.0 * sigma_frac ** 2)).view(1, 1, height, width)


def random_fixed_map(frame: torch.Tensor, seed: int) -> torch.Tensor:
    """A fixed smooth random field, identical for every frame at a given seed."""
    _, _, height, width = frame.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7717)
    coarse = torch.rand(1, 1, max(2, height // 16), max(2, width // 16),
                        generator=generator)
    field = F.interpolate(coarse, size=(height, width), mode="bilinear",
                          align_corners=False)
    return field.to(frame.device).clamp_min(0.0)


_SEG_MODEL = {}
_SEG_CACHE = {}


# ADE20K classes that make up the built scene: these are the pixels a
# scene-privacy method is actually trying to protect, and unlike VOC's single
# "background" label they are named explicitly by the taxonomy.
_ADE_SCENE_CLASSES = ("building", "sky", "road", "tree", "sidewalk", "wall",
                      "house", "skyscraper", "grass", "plant", "earth", "path",
                      "fence")


def segmentation_map_ade(frame: torch.Tensor, key: str) -> torch.Tensor:
    """Scene-structure probability from SegFormer trained on ADE20K.

    This is the closest available instantiation of the published strategy: a
    pretrained model with a taxonomy that names the built environment
    directly, rather than inferring it from the complement of VOC objects.
    """
    ck = "ade::" + key
    if ck in _SEG_CACHE:
        return _SEG_CACHE[ck]
    if "ade" not in _SEG_MODEL:
        from transformers import SegformerForSemanticSegmentation
        mid = "nvidia/segformer-b0-finetuned-ade-512-512"
        m = SegformerForSemanticSegmentation.from_pretrained(mid)
        _SEG_MODEL["ade"] = m.eval().to(frame.device)
        lbl = m.config.id2label
        _SEG_MODEL["ade_ids"] = [i for i, v in lbl.items()
                                 if v.split(",")[0].strip() in _ADE_SCENE_CLASSES]
        mean = torch.tensor([0.485, 0.456, 0.406], device=frame.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=frame.device).view(1, 3, 1, 1)
        _SEG_MODEL["ade_norm"] = (mean, std)
    mean, std = _SEG_MODEL["ade_norm"]
    with torch.no_grad():
        x = F.interpolate(frame / 255.0, size=(512, 512), mode="bilinear",
                          align_corners=False)
        x = (x - mean) / std
        logits = _SEG_MODEL["ade"](pixel_values=x).logits
        prob = torch.softmax(logits, dim=1)
        scene = prob[:, _SEG_MODEL["ade_ids"]].sum(dim=1, keepdim=True)
        scene = F.interpolate(scene, size=frame.shape[-2:], mode="bilinear",
                              align_corners=False)
    out = scene.clamp_min(0.0).detach()
    if len(_SEG_CACHE) < 8192:
        _SEG_CACHE[ck] = out
    return out


def segmentation_map_fcn(frame: torch.Tensor, key: str) -> torch.Tensor:
    """Same strategy, a second published architecture.

    Running two independent segmentation models answers whether the
    segmentation result is a property of the strategy or of one particular
    network.
    """
    ck = "fcn::" + key
    if ck in _SEG_CACHE:
        return _SEG_CACHE[ck]
    if "fcn" not in _SEG_MODEL:
        from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
        w = FCN_ResNet50_Weights.DEFAULT
        _SEG_MODEL["fcn"] = fcn_resnet50(weights=w).eval().to(frame.device)
        _SEG_MODEL["fcn_t"] = w.transforms()
    with torch.no_grad():
        logits = _SEG_MODEL["fcn"](_SEG_MODEL["fcn_t"](frame / 255.0))["out"]
        prob_bg = torch.softmax(logits, dim=1)[:, 0:1]
        prob_bg = F.interpolate(prob_bg, size=frame.shape[-2:], mode="bilinear",
                                align_corners=False)
    out = prob_bg.clamp_min(0.0).detach()
    if len(_SEG_CACHE) < 8192:
        _SEG_CACHE[ck] = out
    return out


def segmentation_map(frame: torch.Tensor, key: str) -> torch.Tensor:
    """Background probability from an off-the-shelf semantic segmentation model.

    This is the placement a practitioner implementing the stated strategy --
    perturb the background scene, preserve foreground objects -- would actually
    build, using a pretrained model rather than a hand-designed prior. The
    torchvision DeepLabV3 head is trained on COCO with VOC labels, whose
    ``background`` class is precisely the scene structure (buildings, road,
    sky, vegetation) that carries the location signal.
    """
    if key in _SEG_CACHE:
        return _SEG_CACHE[key]
    if "m" not in _SEG_MODEL:
        import torchvision
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
        _SEG_MODEL["m"] = model.eval().to(frame.device)
        _SEG_MODEL["t"] = weights.transforms()
    with torch.no_grad():
        x = _SEG_MODEL["t"](frame / 255.0)
        logits = _SEG_MODEL["m"](x)["out"]
        prob_bg = torch.softmax(logits, dim=1)[:, 0:1]
        prob_bg = F.interpolate(prob_bg, size=frame.shape[-2:], mode="bilinear",
                                align_corners=False)
    out = prob_bg.clamp_min(0.0).detach()
    if len(_SEG_CACHE) < 4096:
        _SEG_CACHE[key] = out
    return out


def attacker_gradient_map(
    frame: torch.Tensor,
    embedder: torch.nn.Module,
    target_embedding: torch.Tensor,
    input_size: Tuple[int, int],
) -> torch.Tensor:
    """Per-pixel |d cos(f(x), g+) / dx|, the attacker's own sensitivity.

    High values mark pixels where a perturbation most changes similarity to the
    correct gallery item -- i.e. exactly where a placement rule *should* spend
    its budget if placement matters at all.
    """
    probe = frame.clone().detach().requires_grad_(True)
    resized = F.interpolate(probe / 255.0, size=input_size, mode="bilinear",
                            align_corners=False)
    emb = embedder(resized)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    target = target_embedding / target_embedding.norm().clamp_min(1e-12)
    similarity = (emb.flatten() * target.flatten()).sum()
    grad, = torch.autograd.grad(similarity, probe)
    return grad.abs().sum(dim=1, keepdim=True).detach()


def margin_gradient_map(
    frame: torch.Tensor,
    embedder: torch.nn.Module,
    positive: torch.Tensor,
    negative: torch.Tensor,
    input_size: Tuple[int, int],
) -> torch.Tensor:
    """Per-pixel sensitivity of the quantity that actually decides Top-1.

    The existing oracle targets the similarity to the correct gallery item.
    Retrieval, however, is decided by the *margin* between that similarity and
    the best competitor: a perturbation that lowers both equally changes no
    ranking. Placing the budget by the margin gradient is therefore the
    strictly correct oracle, and if even that fails to beat uniform, the
    remaining objection that we simply built the wrong oracle is closed.
    """
    probe = frame.clone().detach().requires_grad_(True)
    resized = F.interpolate(probe / 255.0, size=input_size, mode="bilinear",
                            align_corners=False)
    emb = embedder(resized)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pos = positive / positive.norm().clamp_min(1e-12)
    neg = negative / negative.norm().clamp_min(1e-12)
    margin = ((emb.flatten() * pos.flatten()).sum()
              - (emb.flatten() * neg.flatten()).sum())
    grad, = torch.autograd.grad(margin, probe)
    return grad.abs().sum(dim=1, keepdim=True).detach()


def renormalise_to_energy(raw: torch.Tensor, target_sq_sum: torch.Tensor) -> torch.Tensor:
    """Scale a non-negative map so its sum of squares matches the target exactly."""
    raw = raw.clamp_min(0.0)
    current = raw.square().sum()
    if float(current) <= 1e-12:
        n = raw.numel()
        return torch.full_like(raw, float((target_sq_sum / max(n, 1)).sqrt()))
    return raw * torch.sqrt(target_sq_sum / current)


# --------------------------------------------------------------------------
# sensitivity statistics (the mechanistic half)
# --------------------------------------------------------------------------

def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().float().cpu().numpy()
    y = b.flatten().float().cpu().numpy()
    if x.size < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def sensitivity_stats(grad_map: torch.Tensor, learned: torch.Tensor) -> Dict[str, float]:
    g = grad_map.flatten().float()
    mean = float(g.mean())
    std = float(g.std(unbiased=False))
    k = max(1, int(0.10 * g.numel()))
    top_idx = torch.topk(g, k).indices
    captured = float(g[top_idx].square().sum() / g.square().sum().clamp_min(1e-12))
    learned_flat = learned.flatten().float()
    learned_top = torch.topk(learned_flat, k).indices
    captured_by_learned = float(
        g[learned_top].square().sum() / g.square().sum().clamp_min(1e-12)
    )
    return {
        "grad_mean": mean,
        "grad_std": std,
        # coefficient of variation: how far the attacker's sensitivity is from
        # spatially homogeneous. Near 0 predicts that placement cannot matter.
        "grad_cv": float(std / mean) if mean > 0 else float("nan"),
        "grad_energy_top10pct_oracle": captured,
        "grad_energy_top10pct_learned": captured_by_learned,
        "spearman_learned_vs_grad": spearman(learned, grad_map),
    }


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# perturbation operators
#
# The placement study asks where a fixed budget should go. It holds the
# operator fixed at additive isotropic Gaussian noise, which is what the
# selective-privacy literature overwhelmingly uses. The first-order account
# says that for such an operator placement can only set how far the embedding
# moves, never in which direction, because an isotropic draw has no preferred
# direction to begin with. That predicts something testable: give the operator
# spatial structure, and placement should regain leverage it does not have
# here. These operators exist to test that, at matched delivered distortion.
# --------------------------------------------------------------------------

OPERATORS = {
    "gaussian": "additive isotropic Gaussian (the literature's operator)",
    "correlated": "additive spatially correlated Gaussian",
    "blur": "selective low-pass, placed by the weight map",
    "mosaic": "selective block quantisation, placed by the weight map",
    "sign_grad": "deterministic sign along the attacker gradient",
}


def _box_blur(x: torch.Tensor, k: int) -> torch.Tensor:
    """Separable box blur, reflect-padded, applied per channel."""
    if k <= 1:
        return x
    pad = k // 2
    c = x.shape[1]
    ker = torch.ones(c, 1, 1, k, device=x.device, dtype=x.dtype) / k
    y = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(y, ker, groups=c)
    ker = ker.transpose(-1, -2).contiguous()
    y = F.pad(y, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(y, ker, groups=c)


def _block_average(x: torch.Tensor, block: int) -> torch.Tensor:
    """Mosaic: average within non-overlapping blocks, then expand back."""
    if block <= 1:
        return x
    h, w = x.shape[-2:]
    ph, pw = (-h) % block, (-w) % block
    y = F.pad(x, (0, pw, 0, ph), mode="replicate")
    y = F.avg_pool2d(y, block)
    y = F.interpolate(y, scale_factor=block, mode="nearest")
    return y[..., :h, :w]


def operator_delta(
    frame: torch.Tensor,
    weight: torch.Tensor,
    operator: str,
    sigma: float,
    generator: torch.Generator,
    grad_dir: Optional[torch.Tensor] = None,
    blur_kernel: int = 9,
    mosaic_block: int = 8,
    corr_kernel: int = 5,
) -> torch.Tensor:
    """The unscaled perturbation this operator would apply under ``weight``.

    Returned before distortion matching: the caller rescales it so that every
    operator delivers the same measured MSE, which is what makes a comparison
    across operators a comparison of structure rather than of budget.
    """
    if operator == "gaussian":
        eps = torch.randn(frame.shape, generator=generator,
                          dtype=frame.dtype, device="cpu").to(frame.device)
        return weight * eps * sigma
    if operator == "correlated":
        eps = torch.randn(frame.shape, generator=generator,
                          dtype=frame.dtype, device="cpu").to(frame.device)
        eps = _box_blur(eps, corr_kernel)
        std = eps.std().clamp_min(1e-8)
        return weight * (eps / std) * sigma
    if operator == "blur":
        return weight * (_box_blur(frame, blur_kernel) - frame)
    if operator == "mosaic":
        return weight * (_block_average(frame, mosaic_block) - frame)
    if operator == "sign_grad":
        if grad_dir is None:
            raise ValueError("sign_grad requires an attacker gradient direction")
        return weight * torch.sign(grad_dir) * sigma
    raise ValueError(f"unknown operator {operator}")


def apply_at_matched_mse(
    frame: torch.Tensor,
    raw_delta: torch.Tensor,
    target_mse: float,
    clamp_min: float,
    clamp_max: float,
    iters: int = 40,
) -> Tuple[torch.Tensor, float]:
    """Scale a perturbation so the released frame has a prescribed MSE.

    Matching nominal weight energy is not the same as matching what reaches
    the image: concentrated placements lose part of their budget at the pixel
    clamp, a confound this study previously reported rather than removed.
    Solving for the gain that hits a target measured MSE removes it, and is
    the only defensible way to compare operators whose distortion profiles
    differ by construction.
    """
    def mse_at(gain: float) -> Tuple[torch.Tensor, float]:
        out = (frame + gain * raw_delta).clamp(clamp_min, clamp_max)
        return out, float((out - frame).square().mean())

    _, m1 = mse_at(1.0)
    if m1 <= 1e-12:
        return frame.clone(), 0.0
    lo, hi = 0.0, 1.0
    _, mhi = mse_at(hi)
    # Clipping makes MSE sublinear in the gain, so grow the bracket instead of
    # assuming the unclipped square-law scaling holds.
    while mhi < target_mse and hi < 1e4:
        hi *= 2.0
        _, mhi = mse_at(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        _, m = mse_at(mid)
        if m < target_mse:
            lo = mid
        else:
            hi = mid
    out, achieved = mse_at(0.5 * (lo + hi))
    return out, achieved


@torch.no_grad()
def _protect_with_operator(
    frames: torch.Tensor,
    weight_maps: Sequence[torch.Tensor],
    operator: str,
    sigma: float,
    clamp_min: float,
    clamp_max: float,
    seed: int,
    device: torch.device,
    reference_mse: Optional[List[float]] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """Release frames through one operator at a prescribed per-frame MSE.

    When ``reference_mse`` is None the frames are released at the operator's
    own natural scale and the achieved MSE is returned, which is how the
    reference condition establishes the target every other condition is then
    matched to.
    """
    out: List[torch.Tensor] = []
    achieved: List[float] = []
    for t_index in range(frames.size(0)):
        frame = frames[t_index : t_index + 1].to(device).float()
        w = weight_maps[t_index].to(device)
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + int(t_index))
        raw = operator_delta(frame, w, operator, sigma, g)
        if reference_mse is None:
            released = (frame + raw).clamp(clamp_min, clamp_max)
            mse = float((released - frame).square().mean())
        else:
            released, mse = apply_at_matched_mse(
                frame, raw, reference_mse[t_index], clamp_min, clamp_max
            )
        out.append(released.squeeze(0).cpu())
        achieved.append(mse)
    return torch.stack(out, dim=0), achieved


@torch.no_grad()
def _protect_with_weight(
    frames: torch.Tensor,
    weight_maps: Sequence[torch.Tensor],
    cfg: Mapping[str, object],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    _, _, injector = build_modules(cfg, device, seed)
    out: List[torch.Tensor] = []
    for t_index in range(frames.size(0)):
        frame = frames[t_index : t_index + 1].to(device)
        w = weight_maps[t_index]
        protected = injector.apply(frame, torch.ones_like(w), w, t_index=t_index)
        out.append(protected.squeeze(0).cpu())
    return torch.stack(out, dim=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Placement-rule study (energy-matched).")
    p.add_argument("--config", default="src/config/config.yaml")
    p.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    p.add_argument("--monitoring_root", default="",
                   help="mined paired-scene corpus (proxy mode)")
    p.add_argument("--manifest", default="",
                   help="real place-labelled MSLS manifest (geotagged mode); "
                        "mutually exclusive with --monitoring_root")
    p.add_argument("--root", default="",
                   help="image root the manifest paths resolve against")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--backbones", nargs="+", default=["resnet18"])
    p.add_argument("--placements", nargs="+", default=list(PLACEMENTS))
    p.add_argument("--operator", default="legacy",
                   choices=["legacy"] + list(OPERATORS),
                   help="perturbation operator; 'legacy' keeps the original "
                        "energy-matched additive-Gaussian release path, any "
                        "other value releases every condition at the MSE the "
                        "uniform Gaussian reference delivers")
    p.add_argument("--num_queries", type=int, default=12)
    p.add_argument("--pair_pool_size", type=int, default=240)
    p.add_argument("--max_gallery", type=int, default=48)
    p.add_argument("--gallery_sizes", type=int, nargs="+", default=[48])
    p.add_argument("--max_external_distractors", type=int, default=0)
    p.add_argument("--coco_root", default="")
    p.add_argument("--digica_root", default="")
    p.add_argument("--clip_len", type=int, default=4)
    p.add_argument("--min_frames", type=int, default=6)
    p.add_argument("--resize_h", type=int, default=192)
    p.add_argument("--resize_w", type=int, default=320)
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    p.add_argument("--energy_tolerance", type=float, default=1e-4)
    p.add_argument("--sigma", type=float, default=None,
                   help="override the perturbation budget sigma_0")
    return p.parse_args()



def embed_gallery_batched(cfg, embedder, images: torch.Tensor,
                          batch: int = 128) -> torch.Tensor:
    """Embed the gallery in chunks, keeping the result on the compute device.

    The shared helper embeds the whole gallery in one forward pass, which is
    fine for a light backbone but exhausts an 8 GB card on a VPR backbone at
    2,000 images. Chunking here rather than changing the shared helper keeps
    every previously published number reproducible from the same code path;
    under eval/no_grad the chunked result is the same embedding matrix.
    """
    out = []
    with torch.no_grad():
        for i in range(0, images.size(0), batch):
            out.append(build_gallery_embeddings(cfg, embedder,
                                                images[i:i + batch]))
    return torch.cat(out, dim=0)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(args.config)
    if args.sigma is not None:
        import copy as _copy
        cfg = _copy.deepcopy(cfg)
        cfg["ppedcrf"]["noise"] = dict(cfg["ppedcrf"]["noise"])
        cfg["ppedcrf"]["noise"]["sigma"] = float(args.sigma)
        print(f"[placement] sigma override -> {args.sigma}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[placement] device={device}")

    geotagged = bool(args.manifest)
    if geotagged:
        # Real place-labelled evaluation: each query is a single frame and
        # correctness is decided by the official place id, not by a mined pair.
        records, gallery_records = load_manifest(args.manifest, args.root)
        resize_hw = (int(args.resize_h), int(args.resize_w))
        geo_gallery_ids = sorted(gallery_records)
        geo_gallery_tensor = torch.stack(
            [load_image(gallery_records[g]["path"], resize_hw) for g in geo_gallery_ids])
        query_ids = [r["query_id"] for r in records]
        # one-frame "clips" so the placement machinery below is unchanged
        query_clips = {r["query_id"]: load_image(r["query_path"], resize_hw).unsqueeze(0)
                       for r in records}
        positive_place = {r["query_id"]: r["place_id"] for r in records}
        gallery_place = {g: rec["place_id"] for g, rec in gallery_records.items()}
        print(f"[placement] geotagged mode: {len(query_ids)} queries, "
              f"{len(geo_gallery_ids)} gallery, "
              f"{len(set(positive_place.values()))} place ids")
    else:
        if not args.monitoring_root:
            raise SystemExit("either --manifest or --monitoring_root is required")
        data = _build_proxy_data(args, device)
        pairs, _hard, query_ids, query_clips, _raw, gallery_by_id, distractors = data

    from main import load_sensnet_checkpoint  # noqa: E402
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)

    rows_path = out_dir / "per_query.csv"
    sens_path = out_dir / "sensitivity_stats.jsonl"
    done: set = set()
    if rows_path.is_file():
        import csv as _csv
        with rows_path.open(newline="", encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                done.add((r.get("query_id"), r.get("placement"),
                          r.get("backbone"), r.get("seed"), r.get("gallery_size")))
        print(f"[placement] resuming, {len(done)} rows already complete")

    import csv
    fieldnames: List[str] | None = None
    fh = rows_path.open("a", newline="", encoding="utf-8")
    writer = None
    sens_fh = sens_path.open("a", encoding="utf-8")

    for backbone in args.backbones:
        rcfg = RetrievalConfig(backbone=backbone, device=str(device), normalize=True,
                               input_size=default_input_size_for_backbone(backbone),
                               topk=(1, 5, 10))
        embedder = make_default_embedder(rcfg).eval().to(device)
        for gallery_size in sorted(args.gallery_sizes):
            if geotagged:
                gallery_tensor, gallery_ids = geo_gallery_tensor, geo_gallery_ids
            else:
                gallery_tensor, gallery_ids = build_gallery_tensor(
                    gallery_frame_by_id=gallery_by_id, query_ids=query_ids,
                    distractor_ids=distractors, gallery_size=int(gallery_size))
            gallery_emb = embed_gallery_batched(rcfg, embedder, gallery_tensor)

            for seed in args.seeds:
                # --- build every placement's weight maps, energy-matched ---
                per_placement_clips: Dict[str, List[torch.Tensor]] = {
                    k: [] for k in args.placements}
                for q_index, query_id in enumerate(query_ids):
                    frames = query_clips[query_id]
                    crf, ncp, _ = build_modules(cfg, device, seed)
                    prev = None
                    learned_maps: List[torch.Tensor] = []
                    for t in range(frames.size(0)):
                        frame = frames[t : t + 1].to(device)
                        unary = sensnet(frame)
                        refined, prev = crf.refine(unary, prev_prob=prev, flow=None)
                        strength = ncp.allocate(refined)
                        learned_maps.append((refined * strength).clamp_min(0.0))

                    if geotagged:
                        want = positive_place[query_id]
                        pos_idx = [i for i, g in enumerate(gallery_ids)
                                   if gallery_place.get(g) == want]
                    else:
                        pos_idx = [i for i, g in enumerate(gallery_ids) if g == query_id]
                    target_emb = gallery_emb[pos_idx[0]] if pos_idx else gallery_emb[0]
                    # The best competitor from a different place: the item the
                    # margin is actually measured against.
                    rival_emb = None
                    if any(p.startswith(("margin_oracle", "anti_margin"))
                           for p in args.placements):
                        with torch.no_grad():
                            qe = embedder(F.interpolate(
                                frames[frames.size(0) // 2: frames.size(0) // 2 + 1]
                                .to(device) / 255.0,
                                size=rcfg.input_size, mode="bilinear",
                                align_corners=False))
                            qe = qe / qe.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                            gnorm = gallery_emb / gallery_emb.norm(
                                dim=-1, keepdim=True).clamp_min(1e-12)
                            sims = (gnorm @ qe.flatten()).clone()
                            for i in pos_idx:
                                sims[i] = -2.0
                            rival_emb = gallery_emb[int(torch.argmax(sims).item())]

                    grad_map_ref = None
                    for placement in args.placements:
                        maps: List[torch.Tensor] = []
                        for t in range(frames.size(0)):
                            frame = frames[t : t + 1].to(device)
                            learned = learned_maps[t]
                            target_sq = learned.square().sum()
                            if placement == "learned":
                                raw = learned
                            elif placement == "uniform":
                                raw = torch.ones_like(learned)
                            elif placement == "saliency":
                                raw = saliency_map(frame)
                            elif placement == "edge":
                                raw = edge_map(frame)
                            elif placement == "center":
                                raw = center_map(frame)
                            elif placement == "random_fixed":
                                raw = random_fixed_map(frame, seed)
                            elif placement == "segmentation":
                                raw = segmentation_map(frame, f"{query_id}::{t}")
                            elif placement == "segmentation_fcn":
                                raw = segmentation_map_fcn(frame, f"{query_id}::{t}")
                            elif placement == "segmentation_ade":
                                raw = segmentation_map_ade(frame, f"{query_id}::{t}")
                            elif placement in ("margin_oracle", "anti_margin_oracle"):
                                with torch.enable_grad():
                                    gm = margin_gradient_map(
                                        frame, embedder, target_emb, rival_emb,
                                        rcfg.input_size)
                                if placement == "margin_oracle":
                                    raw = gm
                                else:
                                    raw = 1.0 / (gm + gm.mean().clamp_min(1e-8))
                            elif placement in ("oracle_grad", "anti_oracle_grad"):
                                with torch.enable_grad():
                                    g = attacker_gradient_map(
                                        frame, embedder, target_emb, rcfg.input_size)
                                if t == 0 and grad_map_ref is None:
                                    grad_map_ref = g
                                if placement == "oracle_grad":
                                    raw = g
                                else:
                                    raw = (g.max() - g).clamp_min(0.0)
                            else:
                                raise ValueError(f"unknown placement {placement}")
                            maps.append(renormalise_to_energy(raw, target_sq))
                        per_placement_clips[placement].append((query_id, maps))

                    if grad_map_ref is not None:
                        stats = sensitivity_stats(grad_map_ref, learned_maps[0])
                        stats.update({"query_id": query_id, "backbone": backbone,
                                      "seed": int(seed),
                                      "gallery_size": int(gallery_size)})
                        sens_fh.write(json.dumps(stats) + "\n")
                        sens_fh.flush(); os.fsync(sens_fh.fileno())

                # --- protect, evaluate, write incrementally ---
                rmin_cfg = float(cfg["ppedcrf"]["noise"]["clamp_min"])
                rmax_cfg = float(cfg["ppedcrf"]["noise"]["clamp_max"])
                sigma_cfg = float(cfg["ppedcrf"]["noise"]["sigma"])
                # When operators are compared, every condition is released at
                # the MSE the reference condition (uniform placement, additive
                # Gaussian) delivers on the same frames. Matching on measured
                # distortion rather than on nominal weight energy is what makes
                # the comparison one of structure instead of budget.
                ref_mse_by_query: Dict[str, List[float]] = {}
                if args.operator != "legacy":
                    ref_maps = dict(per_placement_clips[args.placements[0]])
                    for q_id, maps in per_placement_clips[args.placements[0]]:
                        _, mses = _protect_with_operator(
                            query_clips[q_id], maps, "gaussian", sigma_cfg,
                            rmin_cfg, rmax_cfg, seed, device, reference_mse=None)
                        ref_mse_by_query[q_id] = mses

                for placement in args.placements:
                    protected_frames: List[torch.Tensor] = []
                    qual: Dict[str, Dict[str, float]] = {}
                    for q_id, maps in per_placement_clips[placement]:
                        frames = query_clips[q_id]
                        if args.operator == "legacy":
                            clip = _protect_with_weight(frames, maps, cfg, device, seed)
                        else:
                            clip, _ = _protect_with_operator(
                                frames, maps, args.operator, sigma_cfg,
                                rmin_cfg, rmax_cfg, seed, device,
                                reference_mse=ref_mse_by_query[q_id])
                        mid = clip[clip.size(0) // 2]
                        protected_frames.append(mid)
                        orig = frames.detach().float().cpu()
                        eff = torch.stack([m.squeeze(0).cpu() for m in maps]).float()
                        # Weight-space energy is matched by construction, but what
                        # actually reaches the image can differ: concentrating a
                        # fixed budget raises per-pixel amplitude, and the excess
                        # is lost when the result clips against the pixel range.
                        # Measuring both separates "placement" from "delivered
                        # distortion", which are not the same control variable.
                        rmin = float(cfg["ppedcrf"]["noise"]["clamp_min"])
                        rmax = float(cfg["ppedcrf"]["noise"]["clamp_max"])
                        clipped = ((clip <= rmin + 1e-6) | (clip >= rmax - 1e-6)).float()
                        w = eff.flatten()
                        k = max(1, int(0.10 * w.numel()))
                        top_share = float(
                            torch.topk(w, k).values.square().sum()
                            / w.square().sum().clamp_min(1e-12))
                        qual[q_id] = {
                            "psnr_mean": float(np.mean([
                                psnr_torch(orig[i], clip[i]) for i in range(orig.size(0))])),
                            "effective_weight_energy": float(eff.square().mean().item()),
                            "effective_mse": float(
                                (clip.float() - orig).square().mean().item()),
                            "clipping_fraction": float(clipped.mean().item()),
                            "weight_top10pct_share": top_share,
                        }
                    extra = {}
                    if geotagged:
                        extra = {"positive_place_by_query": positive_place,
                                 "gallery_place_by_id": gallery_place}
                    qrows = detailed_retrieval(
                        torch.stack(protected_frames), query_ids, gallery_tensor,
                        gallery_ids, embedder, device, input_size=rcfg.input_size,
                        quality_by_query=qual, **extra)
                    for qrow in qrows:
                        qrow.update({"placement": placement,
                                     "operator": args.operator,
                                     "label": PLACEMENT_LABELS[placement],
                                     "backbone": backbone, "seed": int(seed),
                                     "gallery_size": int(gallery_size)})
                        key = (str(qrow["query_id"]), placement, backbone,
                               str(int(seed)), str(int(gallery_size)))
                        if key in done:
                            continue
                        if writer is None:
                            fieldnames = list(qrow.keys())
                            writer = csv.DictWriter(fh, fieldnames=fieldnames)
                            if rows_path.stat().st_size == 0:
                                writer.writeheader()
                        writer.writerow({k: qrow.get(k) for k in fieldnames})
                        fh.flush(); os.fsync(fh.fileno())
                    print(f"[placement] {backbone} g{gallery_size} seed{seed} "
                          f"{placement}: {len(qrows)} rows")

    fh.close(); sens_fh.close()
    print(f"[placement] done -> {rows_path}")


if __name__ == "__main__":
    main()
