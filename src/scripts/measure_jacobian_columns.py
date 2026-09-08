"""What do the sensitivity maps actually measure?

Section III-H of the manuscript reports a top-decile energy concentration and
identifies it with the Jacobian column norms of the embedding map. The map that
produced that number is `attacker_gradient_map` in
`run_placement_rule_study.py`, which computes

    s_i = sum_c |d cos(f(x), g+) / d x_{c,i}|

-- the gradient of one scalar score, summed over colour channels. The Jacobian
column norm of the embedding is

    ||J_i||_2 = sqrt( sum_c sum_j (d f_j / d x_{c,i})^2 ).

These are different quantities: the first is a directional derivative of a
projection of f, the second is the total sensitivity of the whole embedding.
Naming them apart in the prose is necessary but not sufficient; this script
measures how far apart they are, which is what the review asks for.

The column norms are estimated by random projection. For v ~ N(0, I_d),

    E[(J^T v)_i^2] = ||J_i||_2^2,

so averaging K such terms is unbiased, and each term costs a single
vector-Jacobian product -- one backward pass rather than the d passes a full
Jacobian would need. Because the estimate is itself noisy, the script reports a
split-half correlation between two independent K/2 estimates: that is the
ceiling any agreement with another map can reach, and comparing a cross-map
correlation against it is the only way to tell disagreement from estimator
noise.

The second half addresses R2 rather than R3. Under local linearisation the
margin m(x) between the positive similarity and the best competitor has
gradient a, and a weighted Gaussian perturbation delta = D_w eps has predicted
margin variance

    v(w) = sigma^2 * sum_i a_i^2 w_i^2,

which depends on the weight map. The script draws realisations, records the
measured margin change against that prediction, counts rank flips, and reports
pre-clamp against post-clamp energy so the clamp's share of any shortfall is
measured instead of assumed. The two-coordinate counterexample from the review
is included as a closed-form control: if the estimator cannot reproduce a
separation that is true by construction, it is not measuring what it claims.

Writes one row per query per map, flushed immediately, and skips completed rows
on restart.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
    preprocess_for_embed,
)
from scripts.run_direction_transfer_study import (  # noqa: E402
    embed_gallery_batched,
)
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)

FIELDS = [
    "query_id", "backbone", "map", "n_probes",
    "spearman_vs_jacobian", "topdecile_jaccard_vs_jacobian",
    "topdecile_concentration", "gini",
    "pred_margin_sd", "obs_margin_sd", "pred_over_obs",
    "rank_flip_rate", "clean_margin",
    "energy_preclamp", "energy_postclamp", "clamp_loss_frac",
    "jacobian_split_half_spearman",
]


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation between two flattened maps.

    Ties are handled by average ranking. The maps here are dense float maps in
    which exact ties are rare, but the uniform control map is entirely ties:
    average ranking collapses it to a constant, its rank variance is zero, and
    the correlation is undefined. That row therefore reports NaN, which is the
    honest answer -- without average ranking it would instead report whatever
    order `argsort` happened to impose, which is an artefact.
    """
    def rank(x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        order = torch.argsort(x)
        ranks = torch.empty(n, dtype=torch.float64, device=x.device)
        ranks[order] = torch.arange(n, dtype=torch.float64, device=x.device)
        # Average the ranks inside each run of equal values.
        sorted_x = x[order]
        start = 0
        for i in range(1, n + 1):
            if i == n or sorted_x[i] != sorted_x[start]:
                if i - start > 1:
                    ranks[order[start:i]] = ranks[order[start:i]].mean()
                start = i
        return ranks

    ra, rb = rank(a.flatten().double()), rank(b.flatten().double())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) <= 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def topdecile_jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Overlap of the top-10% pixel sets of two maps."""
    n = a.numel()
    k = max(1, int(round(0.1 * n)))
    ia = set(torch.topk(a.flatten(), k).indices.tolist())
    ib = set(torch.topk(b.flatten(), k).indices.tolist())
    union = len(ia | ib)
    return len(ia & ib) / union if union else float("nan")


def topdecile_concentration(a: torch.Tensor) -> float:
    """Share of the map's total energy held by its top 10% of pixels.

    This is the statistic the manuscript prints as 67.7%. Energy means the
    squared map value, matching the constraint sum_i w_i^2 = E under which the
    placement rules are compared.
    """
    flat = a.flatten().double().square()
    total = float(flat.sum())
    if total <= 0:
        return float("nan")
    k = max(1, int(round(0.1 * flat.numel())))
    return float(torch.topk(flat, k).values.sum() / total)


def gini(a: torch.Tensor) -> float:
    """Concentration of the map, independent of the arbitrary decile cut."""
    x = a.flatten().double().abs().sort().values
    n = x.numel()
    total = float(x.sum())
    if total <= 0:
        return float("nan")
    idx = torch.arange(1, n + 1, dtype=torch.float64, device=x.device)
    return float((2.0 * (idx * x).sum()) / (n * total) - (n + 1.0) / n)


def jacobian_column_norms(
    frame: torch.Tensor,
    embedder: torch.nn.Module,
    input_size: int,
    n_probes: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, float]:
    """Random-projection estimate of per-pixel ||J_i||_2, plus a split-half check.

    Each probe is one vector-Jacobian product: with v drawn from a standard
    normal in embedding space, autograd.grad of <f(x), v> with respect to x
    returns J^T v, whose squared entries are unbiased for the squared column
    norms. The embedding is L2-normalised first, because that is the quantity
    retrieval actually compares; the Jacobian of the unnormalised backbone
    output would describe a map the attacker never uses.

    Returns the per-pixel norm map summed in quadrature over colour channels,
    and the Spearman correlation between two independent half-estimates, which
    bounds how well any other map could possibly agree with this one.
    """
    halves: List[torch.Tensor] = []
    for half in range(2):
        acc = torch.zeros_like(frame)
        probes = max(1, n_probes // 2)
        for _ in range(probes):
            probe = frame.clone().detach().requires_grad_(True)
            emb = embedder(preprocess_for_embed(probe, input_size))
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            v = torch.randn(emb.shape, generator=generator).to(emb.device)
            grad, = torch.autograd.grad((emb * v).sum(), probe)
            acc = acc + grad.detach().square()
        halves.append(acc / probes)

    full = 0.5 * (halves[0] + halves[1])
    # Sum the channel contributions before the square root: the column norm of
    # pixel i collects all three of its colour coordinates.
    full_map = full.sum(dim=1, keepdim=True).sqrt()
    h0 = halves[0].sum(dim=1, keepdim=True).sqrt()
    h1 = halves[1].sum(dim=1, keepdim=True).sqrt()
    return full_map.detach(), spearman(h0, h1)


def score_gradient_map(frame, embedder, target, input_size) -> torch.Tensor:
    """The map the published concentration statistic was computed from.

    Reproduced here rather than imported so this script does not depend on the
    placement study's argument plumbing; the arithmetic is identical.
    """
    probe = frame.clone().detach().requires_grad_(True)
    emb = embedder(preprocess_for_embed(probe, input_size))
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    tgt = target / target.norm().clamp_min(1e-12)
    grad, = torch.autograd.grad((emb.flatten() * tgt.flatten()).sum(), probe)
    return grad.abs().sum(dim=1, keepdim=True).detach()


def margin_gradient_map(frame, embedder, positive, negative,
                        input_size) -> torch.Tensor:
    """Sensitivity of the positive-minus-best-competitor margin.

    This is closer to what decides Top-1 than the positive score alone, since a
    perturbation lowering both similarities equally changes no ranking. It
    remains a heuristic spatial weighting rather than an optimum.
    """
    probe = frame.clone().detach().requires_grad_(True)
    emb = embedder(preprocess_for_embed(probe, input_size))
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    pos = positive / positive.norm().clamp_min(1e-12)
    neg = negative / negative.norm().clamp_min(1e-12)
    margin = ((emb.flatten() * pos.flatten()).sum()
              - (emb.flatten() * neg.flatten()).sum())
    grad, = torch.autograd.grad(margin, probe)
    return grad.abs().sum(dim=1, keepdim=True).detach()


def margin_of(frame, embedder, positive, negative, input_size) -> float:
    """Scalar margin at a frame, with no gradient tracking."""
    with torch.no_grad():
        emb = embedder(preprocess_for_embed(frame, input_size))
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        pos = positive / positive.norm().clamp_min(1e-12)
        neg = negative / negative.norm().clamp_min(1e-12)
        return float((emb.flatten() * pos.flatten()).sum()
                     - (emb.flatten() * neg.flatten()).sum())


def normalise_weights(w: torch.Tensor, energy: float) -> torch.Tensor:
    """Rescale a non-negative map so that sum_i w_i^2 equals `energy`.

    This is the constraint under which every placement rule in the paper is
    compared, so a map only becomes a placement after passing through here.
    """
    w = w.clamp_min(0.0)
    s = float(w.square().sum())
    if s <= 1e-20:
        return torch.full_like(w, math.sqrt(energy / w.numel()))
    return w * math.sqrt(energy / s)


def margin_variance_probe(
    frame: torch.Tensor,
    weights: torch.Tensor,
    grad_map_signed: torch.Tensor,
    embedder,
    positive,
    negative,
    input_size: int,
    sigma: float,
    draws: int,
    generator: torch.Generator,
) -> Dict[str, float]:
    """Predicted against realised margin variation under a weight map.

    The prediction is the first-order quantity v(w) = sigma^2 sum_i a_i^2 w_i^2
    from the review's R2, where a is the (signed, per-channel) margin gradient.
    The realisation draws Gaussian noise, applies the weight map, clamps the
    released frame to the valid pixel range, and measures the margin actually
    obtained. Reporting both is the point: the gap between them is the
    linearisation error the manuscript currently assumes away.

    Energy is recorded before and after the clamp so that a delivered-distortion
    shortfall can be attributed to the clamp by measurement instead of by the
    assumption that there is nowhere else for it to go.
    """
    w3 = weights.expand_as(frame)
    pred_var = float((sigma ** 2) * (grad_map_signed.square() * w3.square()).sum())
    clean = margin_of(frame, embedder, positive, negative, input_size)

    obs: List[float] = []
    pre_e: List[float] = []
    post_e: List[float] = []
    flips = 0
    for _ in range(draws):
        eps = torch.randn(frame.shape, generator=generator).to(frame.device)
        delta = w3 * eps * sigma
        released = (frame + delta).clamp(0.0, 255.0)
        effective = released - frame
        pre_e.append(float(delta.square().sum()))
        post_e.append(float(effective.square().sum()))
        m = margin_of(released, embedder, positive, negative, input_size)
        obs.append(m - clean)
        if (clean > 0) != (m > 0):
            flips += 1

    t = torch.tensor(obs, dtype=torch.float64)
    obs_sd = float(t.std(unbiased=True)) if t.numel() > 1 else float("nan")
    pre = sum(pre_e) / max(1, len(pre_e))
    post = sum(post_e) / max(1, len(post_e))
    pred_sd = math.sqrt(max(pred_var, 0.0))
    return {
        "pred_margin_sd": pred_sd,
        "obs_margin_sd": obs_sd,
        "pred_over_obs": (pred_sd / obs_sd) if obs_sd and obs_sd > 1e-12 else float("nan"),
        "rank_flip_rate": flips / max(1, draws),
        "clean_margin": clean,
        "energy_preclamp": pre,
        "energy_postclamp": post,
        "clamp_loss_frac": (pre - post) / pre if pre > 1e-12 else 0.0,
    }


def two_coordinate_control() -> Dict[str, float]:
    """The review's closed-form counterexample, computed exactly.

    With a = (1, 0) and the constraint w_1^2 + w_2^2 = E, putting all energy on
    coordinate 1 gives margin variance sigma^2 E while putting it on coordinate
    2 gives exactly zero. Both allocations satisfy the same energy budget, so
    allocation demonstrably can change a ranking probability under the paper's
    own first-order assumptions. This row is a self-check on the estimator: if
    the measured ratio is not the analytic one, the instrument is wrong before
    any real image is considered.
    """
    sigma, energy = 1.0, 4.0
    on_signal = (sigma ** 2) * energy * 1.0
    off_signal = (sigma ** 2) * energy * 0.0
    return {
        "variance_on_signal": on_signal,
        "variance_off_signal": off_signal,
        "flip_prob_on_signal": 0.5 * math.erfc(1.0 / math.sqrt(2.0 * on_signal)),
        "flip_prob_off_signal": 0.0,
        "note": ("Analytic; clean margin fixed at 1.0. Equal energy, "
                 "different allocation, different flip probability."),
    }


def synthetic_manifest(tmp: Path, n_queries: int, gallery_per: int,
                       h: int, w: int) -> Tuple[Path, Path]:
    """Build a deterministic image set and manifest for the smoke test.

    The point is to exercise the real code path -- manifest parsing, gallery
    embedding, both gradient maps, the probe estimator -- on a machine with no
    dataset. Nothing measured on these frames is a result about MSLS, and no
    number produced from them may enter the paper.
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
            # The first gallery entry is a mild perturbation of the query, so
            # it is a plausible positive; the rest are unrelated.
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
    return mpath, root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="")
    ap.add_argument("--root", default="")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--n_probes", type=int, nargs="+", default=[128],
                    help="Random probes for the column-norm estimate; several "
                         "values sweep the estimator's own convergence.")
    ap.add_argument("--draws", type=int, default=64,
                    help="Gaussian realisations per query per map for the "
                         "margin-variance half.")
    ap.add_argument("--sigma", type=float, default=4.0,
                    help="Noise scale in pixel units for the realisations.")
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gallery_batch", type=int, default=64,
                    help="Gallery images embedded per forward pass. Lower it "
                         "on a card shared with other jobs.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="Smoke-test mode: generate N synthetic queries "
                         "instead of reading a manifest. Produces no result "
                         "that may be cited.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)

    if args.synthetic:
        manifest, root = synthetic_manifest(out / "synthetic", args.synthetic,
                                            4, args.height, args.width)
        manifest, root = str(manifest), str(root)
        print(f"[smoke] synthetic manifest with {args.synthetic} queries "
              f"at {manifest}")
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required unless --synthetic is set")
        manifest, root = args.manifest, args.root

    records, gallery_by_id = load_manifest(manifest, root)
    if args.limit:
        records = records[:args.limit]

    cfg = RetrievalConfig(backbone=args.backbone, device=str(device))
    input_size = default_input_size_for_backbone(args.backbone)
    embedder = make_default_embedder(cfg).to(device).eval()
    for p in embedder.parameters():
        p.requires_grad_(False)

    resize_hw = (args.height, args.width)
    gallery_ids = sorted(gallery_by_id)
    gallery_images = torch.stack(
        [load_image(gallery_by_id[g]["path"], resize_hw) for g in gallery_ids])
    # Chunked, not one forward pass: the shared helper embeds the whole gallery
    # at once, which at 2,000 images is several GB of activations and fails on
    # a card shared with other tenants.
    gallery_emb = embed_gallery_batched(
        cfg, embedder, gallery_images.to(device), batch=args.gallery_batch)
    gallery_emb = gallery_emb / gallery_emb.norm(
        dim=-1, keepdim=True).clamp_min(1e-12)
    gallery_place = {g: gallery_by_id[g]["place_id"] for g in gallery_ids}
    index_of = {g: i for i, g in enumerate(gallery_ids)}

    csv_path = out / "jacobian_columns.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["query_id"], row["map"], row["n_probes"]))
    handle = csv_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDS)
    if not done:
        writer.writeheader()
        handle.flush()

    for record in records:
        qid = record["query_id"]
        frame = load_image(record["query_path"], resize_hw
                           ).unsqueeze(0).to(device)

        # The positive is the gallery entry sharing this query's place; the
        # negative is the highest-scoring entry that does not. That pair is what
        # the margin is defined over, so it must be recomputed per query rather
        # than fixed in advance.
        with torch.no_grad():
            q_emb = embedder(preprocess_for_embed(frame, input_size))
            q_emb = (q_emb / q_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                     ).cpu()
        sims = (gallery_emb @ q_emb.flatten()).flatten()
        pos_ids = [g for g in gallery_ids
                   if gallery_place[g] == record["place_id"]]
        if not pos_ids:
            print(f"[skip] {qid}: no gallery entry shares its place")
            continue
        pos_id = max(pos_ids, key=lambda g: float(sims[index_of[g]]))
        neg_ids = [g for g in gallery_ids
                   if gallery_place[g] != record["place_id"]]
        if not neg_ids:
            print(f"[skip] {qid}: no negative gallery entry")
            continue
        neg_id = max(neg_ids, key=lambda g: float(sims[index_of[g]]))
        positive = gallery_emb[index_of[pos_id]].to(device)
        negative = gallery_emb[index_of[neg_id]].to(device)

        signed_margin_grad = None
        for probes in args.n_probes:
            key_probe = str(probes)
            jac, split_half = jacobian_column_norms(
                frame, embedder, input_size, probes, generator)

            score_map = score_gradient_map(frame, embedder, positive,
                                           input_size)
            margin_map = margin_gradient_map(frame, embedder, positive,
                                             negative, input_size)
            if signed_margin_grad is None:
                # The signed per-channel gradient is what R2's variance formula
                # needs; the |.|-summed map above is a spatial weighting only.
                probe = frame.clone().detach().requires_grad_(True)
                emb = embedder(preprocess_for_embed(probe, input_size))
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                pos_n = positive / positive.norm().clamp_min(1e-12)
                neg_n = negative / negative.norm().clamp_min(1e-12)
                m = ((emb.flatten() * pos_n.flatten()).sum()
                     - (emb.flatten() * neg_n.flatten()).sum())
                signed_margin_grad, = torch.autograd.grad(m, probe)
                signed_margin_grad = signed_margin_grad.detach()

            uniform = torch.ones_like(jac)
            maps = {"jacobian_colnorm": jac,
                    "score_gradient": score_map,
                    "margin_gradient": margin_map,
                    "uniform": uniform}
            energy = float(uniform.square().sum())

            for name, m_map in maps.items():
                if (qid, name, key_probe) in done:
                    continue
                weights = normalise_weights(m_map, energy)
                stats = margin_variance_probe(
                    frame, weights, signed_margin_grad, embedder, positive,
                    negative, input_size, args.sigma, args.draws, generator)
                writer.writerow({
                    "query_id": qid,
                    "backbone": args.backbone,
                    "map": name,
                    "n_probes": key_probe,
                    "spearman_vs_jacobian": spearman(m_map, jac),
                    "topdecile_jaccard_vs_jacobian": topdecile_jaccard(m_map, jac),
                    "topdecile_concentration": topdecile_concentration(m_map),
                    "gini": gini(m_map),
                    "jacobian_split_half_spearman": split_half,
                    **stats,
                })
                handle.flush()
            print(f"[ok] {qid} probes={probes} split-half={split_half:.4f}")

    handle.close()
    (out / "two_coordinate_control.json").write_text(
        json.dumps(two_coordinate_control(), indent=2), encoding="utf-8")
    (out / "run_config.json").write_text(json.dumps({
        "backbone": args.backbone, "n_probes": args.n_probes,
        "draws": args.draws, "sigma": args.sigma,
        "height": args.height, "width": args.width,
        "seed": args.seed, "queries": len(records),
        "synthetic": bool(args.synthetic),
    }, indent=2), encoding="utf-8")
    print(f"[done] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
