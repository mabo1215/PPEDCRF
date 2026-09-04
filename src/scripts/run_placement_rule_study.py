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
from typing import Dict, List, Mapping, Sequence, Tuple

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
    p.add_argument("--monitoring_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--backbones", nargs="+", default=["resnet18"])
    p.add_argument("--placements", nargs="+", default=list(PLACEMENTS))
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[placement] device={device}")

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
            gallery_tensor, gallery_ids = build_gallery_tensor(
                gallery_frame_by_id=gallery_by_id, query_ids=query_ids,
                distractor_ids=distractors, gallery_size=int(gallery_size))
            gallery_emb = build_gallery_embeddings(rcfg, embedder, gallery_tensor)

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

                    pos_idx = [i for i, g in enumerate(gallery_ids) if g == query_id]
                    target_emb = gallery_emb[pos_idx[0]] if pos_idx else gallery_emb[0]

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
                for placement in args.placements:
                    protected_frames: List[torch.Tensor] = []
                    qual: Dict[str, Dict[str, float]] = {}
                    for q_id, maps in per_placement_clips[placement]:
                        frames = query_clips[q_id]
                        clip = _protect_with_weight(frames, maps, cfg, device, seed)
                        mid = clip[clip.size(0) // 2]
                        protected_frames.append(mid)
                        orig = frames.detach().float().cpu()
                        eff = torch.stack([m.squeeze(0).cpu() for m in maps]).float()
                        qual[q_id] = {
                            "psnr_mean": float(np.mean([
                                psnr_torch(orig[i], clip[i]) for i in range(orig.size(0))])),
                            "effective_weight_energy": float(eff.square().mean().item()),
                            "effective_mse": float(
                                (clip.float() - orig).square().mean().item()),
                        }
                    qrows = detailed_retrieval(
                        torch.stack(protected_frames), query_ids, gallery_tensor,
                        gallery_ids, embedder, device, input_size=rcfg.input_size,
                        quality_by_query=qual)
                    for qrow in qrows:
                        qrow.update({"placement": placement,
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
