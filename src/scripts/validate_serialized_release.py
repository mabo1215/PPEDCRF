"""What is actually released, as opposed to what the optimiser produced?

Two gaps separate the manuscript's evaluated object from a deployed one, and
finding R10 asks for both to be measured.

The first is the amplitude bound. `directional_delta` projects the perturbation
to ||delta||_inf <= linf on every step, so the optimiser's output respects that
bound. `release_at_mse` then scales the whole perturbation by a bisection-found
gain g chosen to hit a target delivered MSE, and nothing constrains g <= 1. The
scaling follows the projection, so the released amplitude is g * linf. The
manuscript reports max|delta| = 76 against a projection of 16, which is exactly
this effect; stating the projection alone would misdescribe the released
object.

The second is serialisation. The experiment releases a float tensor. A
deployment transmits a file: pixel values quantised to 8-bit integers, then
encoded, then decoded by whoever receives them. Quantisation and codec loss
both act on the perturbation, and a perturbation optimised against an embedding
has no reason to survive them. This script saves each released frame as PNG
(lossless after quantisation) and as JPEG at stated qualities, decodes each
back, re-measures delivered MSE, maximum amplitude and clipped fraction, and
re-runs retrieval against the held-out attacker on the decoded pixels.

The output supports either conclusion. If the effect survives the codec, the
privacy claim can be stated for a transmitted file rather than for a tensor. If
JPEG erases it, that is a bound on the deployment claim and is reported as one.

Writes one row per query per condition per serialisation, flushed immediately,
and skips completed rows on restart.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    build_gallery_embeddings,
    default_input_size_for_backbone,
    make_default_embedder,
    preprocess_for_embed,
)
from eval.sanitizers import SANITIZERS  # noqa: E402
from scripts.run_direction_transfer_study import (  # noqa: E402
    directional_delta,
    release_at_mse,
)
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)

FIELDS = [
    "query_id", "condition", "serialisation", "seed", "eval_backbone",
    "linf_projection", "release_gain",
    "max_abs_delta_prescale", "mse_prescale",
    "max_abs_delta_float", "mse_float", "clipped_frac_float",
    "max_abs_delta_decoded", "mse_decoded", "bytes",
    "top1", "top5", "top10", "rank_of_positive",
]


def quantise(frame: torch.Tensor) -> torch.Tensor:
    """Round a [0,255] float frame to the integer grid a file can hold.

    Applied before every codec, including the lossless one, so that the
    float-to-integer step is measured separately from the codec's own loss.
    """
    return frame.clamp(0.0, 255.0).round()


def roundtrip(frame: torch.Tensor, fmt: str, quality: int
              ) -> Tuple[torch.Tensor, int]:
    """Encode a frame in memory, decode it back, and report the encoded size.

    Kept in memory rather than on disk: the question is what the decoder
    returns and how large the payload is, and a temporary file would answer the
    same question more slowly. Returns the decoded frame in [0,255] float and
    the byte count of the encoded payload.
    """
    from PIL import Image
    import numpy as np

    arr = quantise(frame)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    buf = io.BytesIO()
    img = Image.fromarray(arr)
    if fmt == "png":
        img.save(buf, format="PNG", optimize=True)
    elif fmt == "jpeg":
        img.save(buf, format="JPEG", quality=quality, subsampling=0)
    else:
        raise ValueError(f"Unknown format: {fmt}")
    payload = buf.getvalue()
    decoded = Image.open(io.BytesIO(payload)).convert("RGB")
    out = torch.from_numpy(np.asarray(decoded).astype("float32")
                           ).permute(2, 0, 1).unsqueeze(0)
    return out.to(frame.device), len(payload)


def retrieval_row(frame: torch.Tensor, embedder, input_size, gallery_emb,
                  gallery_ids, gallery_place, positive_place
                  ) -> Dict[str, float]:
    """Top-k outcome of one released frame against the fixed gallery.

    Correctness follows the manuscript's convention: a hit is any retrieved
    entry whose place label matches the query's, since several gallery images
    can depict the same place.
    """
    with torch.no_grad():
        emb = embedder(preprocess_for_embed(frame, input_size))
        emb = (emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)).cpu()
    sims = (gallery_emb @ emb.flatten()).flatten()
    order = torch.argsort(sims, descending=True).tolist()
    labels = [gallery_place[gallery_ids[i]] == positive_place for i in order]
    rank = next((i + 1 for i, hit in enumerate(labels) if hit), -1)
    return {
        "top1": float(any(labels[:1])),
        "top5": float(any(labels[:5])),
        "top10": float(any(labels[:10])),
        "rank_of_positive": rank,
    }


def synthetic_manifest(tmp: Path, n_queries: int, gallery_per: int,
                       h: int, w: int) -> Tuple[str, str]:
    """Deterministic images and manifest for the no-dataset smoke test.

    Exercises the full path -- manifest, gallery embedding, optimisation,
    rescaling, both codecs, retrieval, CSV -- on a machine with no MSLS. No
    number produced from these frames describes MSLS or may be cited.
    """
    from PIL import Image
    import numpy as np

    root = tmp / "images"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260909)
    lines = []
    for q in range(n_queries):
        base = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        qp = root / f"q{q:03d}.png"
        Image.fromarray(base).save(qp)
        gal = []
        for g in range(gallery_per):
            if g == 0:
                arr = np.clip(base.astype(np.int16)
                              + rng.integers(-12, 12, base.shape), 0, 255
                              ).astype(np.uint8)
                place = f"p{q:03d}"
            else:
                arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
                place = f"p{q:03d}_neg{g}"
            gp = root / f"g{q:03d}_{g:02d}.png"
            Image.fromarray(arr).save(gp)
            gal.append({"gallery_id": f"g{q:03d}_{g:02d}",
                        "path": str(gp), "place_id": place})
        lines.append(json.dumps({
            "query_id": f"q{q:03d}", "query_path": str(qp),
            "place_id": f"p{q:03d}", "gallery": gal}))
    mpath = tmp / "manifest.jsonl"
    mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(mpath), str(root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--eval_backbone", default="resnet18",
                    help="Held-out attacker; never a surrogate.")
    ap.add_argument("--surrogates", nargs="+",
                    default=["resnet50", "vgg16", "vit_b_16"])
    ap.add_argument("--conditions", nargs="+",
                    default=["direction", "direction_eot", "isotropic"])
    ap.add_argument("--target_mse", type=float, default=15.68)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--linf", type=float, default=16.0)
    ap.add_argument("--random_start", type=float, default=8.0)
    ap.add_argument("--eot_sanitizers", nargs="*",
                    default=["jpeg", "blur", "resize", "noise"])
    ap.add_argument("--eot_samples", type=int, default=2)
    ap.add_argument("--jpeg_qualities", type=int, nargs="+", default=[95, 75])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="Smoke-test mode; produces no citable number.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synthetic:
        manifest, root = synthetic_manifest(out / "synthetic", args.synthetic,
                                            4, args.height, args.width)
        print(f"[smoke] synthetic manifest with {args.synthetic} queries")
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required unless --synthetic is set")
        manifest, root = args.manifest, args.root

    records, gallery_by_id = load_manifest(manifest, root)
    if args.limit:
        records = records[:args.limit]

    eval_cfg = RetrievalConfig(backbone=args.eval_backbone, device=str(device))
    eval_size = default_input_size_for_backbone(args.eval_backbone)
    eval_embedder = make_default_embedder(eval_cfg).to(device).eval()
    for p in eval_embedder.parameters():
        p.requires_grad_(False)

    surrogates, sur_sizes = [], []
    for name in args.surrogates:
        cfg = RetrievalConfig(backbone=name, device=str(device))
        m = make_default_embedder(cfg).to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        surrogates.append(m)
        sur_sizes.append(default_input_size_for_backbone(name))

    resize_hw = (args.height, args.width)
    gallery_ids = sorted(gallery_by_id)
    gallery_images = torch.stack(
        [load_image(gallery_by_id[g]["path"], resize_hw) for g in gallery_ids])
    with torch.no_grad():
        gallery_emb = build_gallery_embeddings(
            eval_cfg, eval_embedder, gallery_images.to(device)).cpu()
    gallery_emb = gallery_emb / gallery_emb.norm(
        dim=-1, keepdim=True).clamp_min(1e-12)
    gallery_place = {g: gallery_by_id[g]["place_id"] for g in gallery_ids}

    eot_ops = [SANITIZERS[n] for n in args.eot_sanitizers
               if n in SANITIZERS] if args.eot_sanitizers else []

    serialisations: List[Tuple[str, str, int]] = [("float", "", 0),
                                                  ("png", "png", 0)]
    for q in args.jpeg_qualities:
        serialisations.append((f"jpeg{q}", "jpeg", q))

    csv_path = out / "serialized_release.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["query_id"], row["condition"],
                          row["serialisation"], row["seed"]))
    handle = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    if not done:
        writer.writeheader()
        handle.flush()

    for record in records:
        qid = record["query_id"]
        frame = load_image(record["query_path"], resize_hw
                           ).unsqueeze(0).to(device)
        for seed in args.seeds:
            for condition in args.conditions:
                if all((qid, condition, s[0], str(seed)) in done
                       for s in serialisations):
                    continue
                generator = torch.Generator().manual_seed(
                    seed + abs(hash(qid)) % 100000)

                if condition == "isotropic":
                    # The operating-point control: same delivered budget, no
                    # direction, and no projection to speak of.
                    raw = torch.randn(frame.shape, generator=generator
                                      ).to(device)
                else:
                    # The self objective: push the frame away from its own
                    # clean embedding, which is all a deployed sanitiser knows.
                    targets = []
                    with torch.no_grad():
                        for emb, isz in zip(surrogates, sur_sizes):
                            e = emb(preprocess_for_embed(frame, isz))
                            targets.append(
                                (e / e.norm(dim=-1, keepdim=True
                                            ).clamp_min(1e-12)).detach())
                    raw = directional_delta(
                        frame, targets, surrogates, sur_sizes,
                        steps=args.steps, step_size=args.step_size,
                        linf=args.linf, random_start=args.random_start,
                        generator=generator,
                        eot_ops=eot_ops if condition == "direction_eot" else (),
                        eot_samples=args.eot_samples)

                pre_max = float(raw.abs().max())
                pre_mse = float(((frame + raw).clamp(0.0, 255.0) - frame
                                 ).square().mean())

                released = release_at_mse(frame, raw, args.target_mse)
                eff = released - frame
                # The gain the bisection settled on, recovered from the
                # perturbation it produced. This is the number that decides
                # whether the released amplitude respects the projection.
                denom = float(raw.abs().max())
                gain = (float(eff.abs().max()) / denom) if denom > 1e-12 else 0.0
                float_max = float(eff.abs().max())
                float_mse = float(eff.square().mean())
                clipped = float((((frame + gain * raw) < 0.0)
                                 | ((frame + gain * raw) > 255.0)
                                 ).float().mean())

                for tag, fmt, quality in serialisations:
                    if (qid, condition, tag, str(seed)) in done:
                        continue
                    if tag == "float":
                        decoded, nbytes = released, 0
                    else:
                        decoded, nbytes = roundtrip(released, fmt, quality)
                    d_eff = decoded - frame
                    stats = retrieval_row(decoded, eval_embedder, eval_size,
                                          gallery_emb, gallery_ids,
                                          gallery_place, record["place_id"])
                    writer.writerow({
                        "query_id": qid, "condition": condition,
                        "serialisation": tag, "seed": seed,
                        "eval_backbone": args.eval_backbone,
                        # The isotropic control is drawn, not optimised, so no
                        # projection was ever applied to it; reporting the
                        # optimiser's linf on that row would invent a bound the
                        # condition does not have.
                        "linf_projection": ("" if condition == "isotropic"
                                            else args.linf),
                        "release_gain": gain,
                        "max_abs_delta_prescale": pre_max,
                        "mse_prescale": pre_mse,
                        "max_abs_delta_float": float_max,
                        "mse_float": float_mse,
                        "clipped_frac_float": clipped,
                        "max_abs_delta_decoded": float(d_eff.abs().max()),
                        "mse_decoded": float(d_eff.square().mean()),
                        "bytes": nbytes,
                        **stats,
                    })
                    handle.flush()
                bound = ("no projection" if condition == "isotropic"
                         else f"projection {args.linf:g}")
                print(f"[ok] {qid} {condition} seed={seed} "
                      f"gain={gain:.3f} max|d|={float_max:.2f} ({bound})")

    handle.close()
    (out / "run_config.json").write_text(json.dumps({
        "eval_backbone": args.eval_backbone, "surrogates": args.surrogates,
        "conditions": args.conditions, "target_mse": args.target_mse,
        "steps": args.steps, "step_size": args.step_size, "linf": args.linf,
        "random_start": args.random_start,
        "eot_sanitizers": args.eot_sanitizers, "eot_samples": args.eot_samples,
        "jpeg_qualities": args.jpeg_qualities, "seeds": args.seeds,
        "height": args.height, "width": args.width,
        "queries": len(records), "synthetic": bool(args.synthetic),
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
