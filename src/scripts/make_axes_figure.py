"""The paper's central contrast, on one axis.

A referee reading this paper has to assemble its main claim from a table of
placement deltas, a second table of direction contrasts and three paragraphs.
The claim is simpler than that: at one delivered distortion, every way of
allocating the budget lands on zero and pointing it somewhere does not. This
draws both families against the same Top-1 axis, with the intervals that decide
each verdict and the equivalence margin the protocol declared.

Left: every energy-matched placement against the uniform control, on both
attackers, with the place-clustered interval and the $\\pm0.01$ margin drawn.
Right: the directional contrasts against the isotropic control at the same
delivered distortion, on four held-out attackers.

Everything is recomputed from the released per-query rows; nothing is
transcribed.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]


def place_map(d6_dir: Path) -> dict:
    out = {}
    for path in sorted(glob.glob(str(d6_dir / "*.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if "correct_place" not in (reader.fieldnames or []):
                continue
            for row in reader:
                out.setdefault(row["query_id"], row["correct_place"])
    return out


def per_query(path: str, key: str, backbone: str = "",
              sanitizer: str = "") -> dict:
    """key -> query -> Top-1, seeds averaged within a query.

    The transfer exports carry every attacker-side transform in the same file,
    so a contrast that does not filter on the sanitizer silently averages the
    untouched release together with twelve preprocessed ones and reports a
    third of the effect. That is what the first draft of this figure did.
    """
    arms = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if backbone and r.get("backbone", backbone) != backbone:
                continue
            if sanitizer and r.get("sanitizer", sanitizer) != sanitizer:
                continue
            arms[r[key]][r["query_id"]].append(int(r["correct_rank"]) == 1)
    return {k: {q: float(np.mean(v)) for q, v in d.items()}
            for k, d in arms.items()}


def contrast(arm: dict, ref: dict, places: dict, seed: int, n_boot: int):
    qs = sorted(set(arm) & set(ref) & set(places))
    d = np.array([arm[q] - ref[q] for q in qs])
    ids = np.array([places[q] for q in qs])
    uniq, inv = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        draws[b] = d[np.concatenate([groups[i] for i in pick])].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


PLACEMENT_LABEL = {"learned": "learned support", "oracle_grad": "score-gradient",
                   "anti_oracle_grad": "anti-score-grad.", "saliency": "saliency",
                   "center": "centre bias", "random_fixed": "fixed random",
                   "edge": "edge magnitude"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports", default=str(REPO / "src" / "exports"))
    ap.add_argument("--out", default=str(REPO / "paper" / "figs" / "fig_axes.pdf"))
    ap.add_argument("--n_boot", type=int, default=4000)
    args = ap.parse_args()
    ex = Path(args.exports)
    places = place_map(ex / "tifs_d6")

    rows = []
    for label, path, backbone in [
            ("ResNet18", str(ex / "icme2027_placement_msls/final/per_query.csv"),
             "resnet18"),
            ("MixVPR", str(ex / "placement_mixvpr_rows/per_query.csv"), "")]:
        arms = per_query(path, "placement", backbone)
        ref = arms["uniform"]
        for name in ["learned", "anti_oracle_grad", "oracle_grad", "saliency",
                     "center", "random_fixed", "edge"]:
            if name not in arms:
                continue
            d, lo, hi = contrast(arms[name], ref, places,
                                 abs(hash(name)) % 9999, args.n_boot)
            rows.append((f"{PLACEMENT_LABEL[name]} ({label})", d, lo, hi))

    direction = []
    for label, path, cond in [
            ("ResNet18, 3 surrogates", ex / "tifs_d6" / "d6_r18_plain.csv",
             "transfer_3"),
            ("Patch-NetVLAD, 3", ex / "tifs_a7b" / "a7b_pnv_plain.csv",
             "transfer_3"),
            ("MixVPR, 4 surrogates", ex / "tifs_d6" / "d6_mix_plain.csv",
             "transfer_4"),
            ("ViT-B/16, 3 (no shared trunk)", ex / "tifs6_vit" / "vit_plain.csv",
             "transfer_3")]:
        if not Path(path).is_file():
            print(f"[skip ] {label}: {path} absent")
            continue
        arms = per_query(str(path), "condition", sanitizer="none")
        if cond not in arms or "isotropic" not in arms:
            print(f"[skip ] {label}: conditions {sorted(arms)}")
            continue
        d, lo, hi = contrast(arms[cond], arms["isotropic"], places,
                             len(label), args.n_boot)
        direction.append((label, d, lo, hi))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.axvspan(-0.01, 0.01, color="0.88", zorder=0)
    ax.axvline(0, color="0.4", lw=0.8, zorder=1)
    ys = np.arange(len(rows))
    for y, (_, d, lo, hi) in zip(ys, rows):
        colour = "#b03030" if lo > 0 else ("#2a6099" if hi < 0 else "0.25")
        ax.plot([lo, hi], [y, y], color=colour, lw=1.2, zorder=2)
        ax.plot([d], [y], "o", ms=3.4, color=colour, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ Top-1 vs the uniform control", fontsize=7.5)
    ax.set_title("Allocation: where the budget goes", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.06, 0.06)

    ax = axes[1]
    ax.axvspan(-0.01, 0.01, color="0.88", zorder=0)
    ax.axvline(0, color="0.4", lw=0.8, zorder=1)
    ys = np.arange(len(direction))
    for y, (_, d, lo, hi) in zip(ys, direction):
        ax.plot([lo, hi], [y, y], color="#2a6099", lw=1.4, zorder=2)
        ax.plot([d], [y], "o", ms=4.0, color="#2a6099", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in direction], fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ Top-1 vs the isotropic control", fontsize=7.5)
    ax.set_title("Direction: where it points", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-0.25, 0.045)
    ax.set_xticks([-0.2, -0.1, 0.0])

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # The two panels differ in scale by a factor of four; the caption says so,
    # and the shared margin band is drawn on both so the reader can see it.
    fig.tight_layout(pad=0.5, w_pad=1.6)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[figure] {out}")
    for name, d, lo, hi in rows + direction:
        print(f"[data ] {name:34s} {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
