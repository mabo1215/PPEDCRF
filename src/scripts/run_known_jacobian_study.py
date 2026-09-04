"""A controlled retrieval task with a known Jacobian.

The measured null -- no placement of a fixed noise budget beats spreading it
uniformly -- is explained in the paper by a distinction between the
*magnitude* of the embedding displacement, which placement does control, and
its *direction*, which decides retrieval. On real data that account is an
interpretation, because the attacker's Jacobian is only estimated. Here the
Jacobian is exact and every quantity in the argument is computable, so the
account can be checked rather than believed.

Setup. The attacker embedding is linear, $f(x) = Ax$ with $A \\in
\\mathbb{R}^{d \\times n}$ known. Its per-pixel sensitivity is exactly the
column norm $\\lVert A_i \\rVert$, which we set to a chosen profile -- by
default a heavy-tailed one matched to the concentration measured on the real
attacker (top decile of pixels carrying about 68% of gradient energy). A
gallery of $m$ items is drawn, the query is perturbed by $\\delta_i = w_i
\\varepsilon_i$ with $\\sum_i w_i^2 = E$ held fixed, and Top-1 retrieval is
scored under each placement rule.

What it tests. First-order theory says
$\\mathbb{E}\\lVert \\Delta f \\rVert^2 = \\sigma^2 \\sum_i w_i^2 \\lVert A_i
\\rVert^2$, maximised at fixed energy by putting all weight on the
largest-norm columns -- the oracle placement. The script verifies that this
identity holds exactly (it must; $A$ is known), and then asks whether the
displacement it maximises translates into retrieval error.

The alignment condition. Retrieval is decided by how much of $\\Delta f$ falls
along the directions separating the positive from its competitors. The script
sweeps an alignment parameter controlling how much of that discriminative
subspace lies in the span of the high-sensitivity columns. This isolates the
prediction the paper's account makes: placement should pay only when
sensitivity and discriminative directions coincide, and the oracle's advantage
should vanish as they decouple -- which is the regime real attacker
embeddings occupy.

CPU-only and self-contained: no dataset, no checkpoint, no GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import torch


def make_sensitivity_profile(n: int, top_decile_share: float,
                             generator: torch.Generator) -> torch.Tensor:
    """Column norms whose squared energy has a target top-decile share.

    A log-normal profile is used and its dispersion is solved for by bisection
    so the top 10% of pixels carry the requested fraction of total squared
    sensitivity. This lets the synthetic attacker be matched to the
    concentration actually measured on the real one instead of being chosen
    for convenience.
    """
    base = torch.randn(n, generator=generator)

    def share_for(sigma: float) -> tuple[float, torch.Tensor]:
        norms = torch.exp(sigma * base)
        energy = norms.square()
        k = max(1, int(round(0.10 * n)))
        return float(energy.topk(k).values.sum() / energy.sum()), norms

    lo, hi = 0.0, 6.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        share, _ = share_for(mid)
        if share < top_decile_share:
            lo = mid
        else:
            hi = mid
    _, norms = share_for(0.5 * (lo + hi))
    return norms / norms.mean()


def build_embedding(n: int, d: int, norms: torch.Tensor, alignment: float,
                    generator: torch.Generator) -> torch.Tensor:
    """A known Jacobian whose column norms follow ``norms``.

    ``alignment`` controls whether high-sensitivity pixels write into the same
    embedding directions as everything else (alignment 0, so sensitivity and
    discriminative structure are decoupled) or into a distinguished subspace
    that dominates the gallery geometry (alignment 1, so they coincide).
    """
    directions = torch.randn(d, n, generator=generator)
    if alignment > 0.0:
        # A privileged low-dimensional subspace, into which the highest
        # sensitivity columns are progressively rotated.
        k = max(1, d // 8)
        rank = torch.argsort(norms, descending=True)
        weight = torch.zeros(n)
        weight[rank[: max(1, n // 10)]] = 1.0
        privileged = torch.zeros(d, n)
        privileged[:k] = torch.randn(k, n, generator=generator)
        mix = alignment * weight.unsqueeze(0)
        directions = (1.0 - mix) * directions + mix * privileged
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return directions * norms.unsqueeze(0)


class Encoder:
    """Either an exactly linear map or a smooth nonlinear one.

    The linear case is the setting in which the first-order argument is not an
    approximation at all, so any failure of its prediction cannot be blamed on
    linearisation error. The nonlinear case adds the property every real image
    encoder has: the Jacobian is exact only in a neighbourhood of the point
    where it was taken, so a placement that concentrates the budget on few
    pixels pushes those pixels out of the regime in which its own sensitivity
    estimate is valid.
    """

    def __init__(self, A: torch.Tensor, nonlinear: bool, scale: float):
        self.A = A
        self.nonlinear = nonlinear
        self.scale = scale
        if nonlinear:
            d = A.shape[0]
            g = torch.Generator().manual_seed(int(A.sum().abs().item() * 1e3) % 2**31)
            self.W2 = torch.randn(d, d, generator=g) / math.sqrt(d)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.A.T
        if not self.nonlinear:
            return h
        return torch.tanh(h / self.scale) @ self.W2.T

    def pixel_sensitivity(self, x: torch.Tensor) -> torch.Tensor:
        """Exact per-pixel Jacobian column norms at ``x``."""
        if not self.nonlinear:
            return self.A.norm(dim=0)
        h = x @ self.A.T
        g = (1.0 - torch.tanh(h / self.scale) ** 2) / self.scale
        J = self.W2 @ (g.unsqueeze(1) * self.A)
        return J.norm(dim=0)


def placements(norms: torch.Tensor, energy: float,
               generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Energy-matched weight vectors, all with sum of squares equal to energy."""
    n = norms.numel()
    raw = {
        "uniform": torch.ones(n),
        "oracle": norms.clone(),
        "anti_oracle": 1.0 / norms.clamp_min(1e-6),
        "random": torch.rand(n, generator=generator) + 1e-6,
    }
    out = {}
    for name, r in raw.items():
        r = r.clamp_min(0.0)
        out[name] = r * math.sqrt(energy / float(r.square().sum()))
    return out


def evaluate(enc: "Encoder", norms: torch.Tensor, weights: torch.Tensor,
             sigma: float, gallery: int, trials: int,
             generator: torch.Generator, clip: float | None = None,
             nuisance: float = 0.0) -> dict[str, float]:
    """Top-1 accuracy and mean squared embedding displacement under a placement.

    ``nuisance`` is the view-to-view variation between two images of the same
    place; it sets how hard the clean retrieval problem is, and therefore how
    large a margin any perturbation has to overcome.

    ``clip`` bounds the released signal to $[-c, c]$, the synthetic counterpart
    of an image being stored in a finite range. It is the one constraint the
    first-order argument omits, and switching it on or off is what separates
    the regime where placement pays from the regime where it does not.
    """
    n = enc.A.shape[1]
    hits = 0
    clean_hits = 0
    disp = 0.0
    delivered = 0.0
    clean = None
    for _ in range(trials):
        # Place recognition matches two *different* views of the same place, so
        # the query is never the gallery item it must retrieve. Modelling it as
        # its own positive would give the pair a similarity of exactly one and
        # an unrealistically large margin over every negative, which is the
        # regime in which any displacement at all decides the outcome.
        places = torch.randn(gallery, n, generator=generator)
        views = torch.randn(gallery, n, generator=generator) * nuisance
        xs = places + views
        emb = enc(xs)
        emb = emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)
        q = places[0] + torch.randn(n, generator=generator) * nuisance
        clean = enc(q.unsqueeze(0))[0]
        eps = torch.randn(n, generator=generator) * sigma
        released = q + weights * eps
        if clip is not None:
            released = released.clamp(-clip, clip)
        delta = released - q
        delivered += float(delta.square().sum())
        pert = enc(released.unsqueeze(0))[0]
        disp += float((pert - clean).square().sum())
        qe = pert / pert.norm().clamp_min(1e-12)
        sims = emb @ qe
        hits += int(torch.argmax(sims).item() == 0)
        # Unperturbed accuracy on the same draw, so the perturbation's effect
        # is read against the difficulty of the clean task rather than in
        # absolute terms.
        cq = clean / clean.norm().clamp_min(1e-12)
        clean_hits += int(torch.argmax(emb @ cq).item() == 0)
    return {
        "top1": hits / trials,
        "clean_top1": clean_hits / trials,
        "mean_sq_displacement": disp / trials,
        "delivered_input_energy": delivered / trials,
        "predicted_sq_displacement": float(
            (sigma ** 2) * (weights.square() * norms.square()).sum()
        ),
        "top_decile_weight_share": float(
            weights.square().topk(max(1, weights.numel() // 10)).values.sum()
            / weights.square().sum()
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pixels", type=int, default=1024)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--gallery", type=int, default=200)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--top_decile_share", type=float, default=0.677,
                    help="target concentration, matched to the real attacker")
    ap.add_argument("--alignments", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--clips", type=float, nargs="+", default=[-1.0],
                    help="bound on the released signal; -1 means unbounded")
    ap.add_argument("--nonlinear", action="store_true",
                    help="use a smooth nonlinear encoder, so the Jacobian is "
                         "exact only locally, as in a real image encoder")
    ap.add_argument("--nuisance", type=float, default=1.0,
                    help="view-to-view variation between two images of the "
                         "same place; 0 makes the query its own positive")
    ap.add_argument("--tanh_scale", type=float, default=8.0,
                    help="width of the nonlinear encoder's linear regime")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    for seed in args.seeds:
        g = torch.Generator().manual_seed(seed)
        norms = make_sensitivity_profile(args.n_pixels, args.top_decile_share, g)
        achieved = float(
            norms.square().topk(max(1, args.n_pixels // 10)).values.sum()
            / norms.square().sum()
        )
        for alignment in args.alignments:
            A = build_embedding(args.n_pixels, args.embed_dim, norms, alignment, g)
            enc = Encoder(A, args.nonlinear, args.tanh_scale)
            # The oracle targets the encoder's true per-pixel sensitivity,
            # which for the nonlinear encoder is measured at a clean sample.
            sens = enc.pixel_sensitivity(torch.randn(args.n_pixels, generator=g))
            energy = float(args.n_pixels)
            for clip in args.clips:
                c = None if clip < 0 else clip
                for name, w in placements(sens, energy, g).items():
                    r = evaluate(enc, sens, w, args.sigma, args.gallery,
                                 args.trials, g, clip=c,
                                 nuisance=args.nuisance)
                    rows.append({
                        "seed": seed,
                        "alignment": alignment,
                        "clip": clip,
                        "sigma": args.sigma,
                        "placement": name,
                        "nonlinear": bool(args.nonlinear),
                        "nuisance": args.nuisance,
                        "top_decile_share": achieved,
                        **r,
                    })
                    print(
                        f"seed={seed} align={alignment:.2f} "
                        f"clip={'none' if c is None else f'{c:g}'} {name:>12}: "
                        f"top1={r['top1']:.4f} (clean {r['clean_top1']:.4f}) "
                        f"E||df||^2={r['mean_sq_displacement']:.1f} "
                        f"(1st-order {r['predicted_sq_displacement']:.1f}) "
                        f"delivered={r['delivered_input_energy']:.1f}",
                        flush=True,
                    )

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    # Summary: the displacement identity, then the retrieval consequence.
    print("\n--- first-order identity, unbounded case (measured / predicted) ---")
    ratios = [r["mean_sq_displacement"] / r["predicted_sq_displacement"]
              for r in rows if r["predicted_sq_displacement"] > 0 and r["clip"] < 0]
    if ratios:
        print(f"  ratio over {len(ratios)} cells: "
              f"min={min(ratios):.4f} max={max(ratios):.4f}")

    order = ("uniform", "oracle", "anti_oracle", "random")
    for clip in args.clips:
        tag = "unbounded" if clip < 0 else f"clipped at +/-{clip:g}"
        print(f"\n--- Top-1 by alignment, {tag} (lower = better privacy) ---")
        print(f"  {'align':>6} " + "".join(f"{p:>13}" for p in order)
              + f"{'oracle-unif':>14}")
        for alignment in args.alignments:
            cells = {}
            for p in order:
                v = [r["top1"] for r in rows if r["alignment"] == alignment
                     and r["placement"] == p and r["clip"] == clip]
                cells[p] = sum(v) / len(v) if v else float("nan")
            print(f"  {alignment:>6.2f} "
                  + "".join(f"{cells[p]:>13.4f}" for p in order)
                  + f"{cells['oracle'] - cells['uniform']:>+14.4f}")

        eng = {}
        for p in order:
            v = [r["delivered_input_energy"] for r in rows if r["placement"] == p
                 and r["clip"] == clip]
            eng[p] = sum(v) / len(v) if v else float("nan")
        print("  delivered input energy (nominal budget "
              f"{args.n_pixels * args.sigma ** 2:.0f}): "
              + "  ".join(f"{p}={eng[p]:.0f}" for p in order))

    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
