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
import zlib
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


# A solved map is still a placement, so it is labelled as one; what separates
# the two entries is only what the optimiser was allowed to see.
OPTIMISED_LABEL = {
    "opt_transfer": "solved, surrogates",
    "opt_whitebox": "solved, attacker",
}


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
                                 zlib.crc32(name.encode()) % 9999,
                                 args.n_boot)
            rows.append((f"{PLACEMENT_LABEL[name]} ({label})", d, lo, hi))

    # The solved maps go on the same panel as the prescribed ones, because the
    # whole point of the comparison is that they are the same axis given a
    # different amount of search. Absent trees are skipped so the figure can be
    # regenerated on a checkout that does not carry the run.
    alloc_dir = ex / "optimised_allocation"
    for label, stem, cond in [
            ("ResNet18", "r1_r18_exp", "opt_transfer"),
            ("ResNet18", "r1_r18_exp", "opt_whitebox"),
            ("MixVPR", "r1_mix_exp", "opt_transfer"),
            ("MixVPR", "r1_mix_exp", "opt_whitebox")]:
        paths = sorted(alloc_dir.glob(f"{stem}*.csv"))
        if not paths:
            continue
        arms = {}
        for path in paths:
            for key, val in per_query(str(path), "condition").items():
                arms.setdefault(key, {}).update(val)
        if cond not in arms or "uniform" not in arms:
            continue
        d, lo, hi = contrast(arms[cond], arms["uniform"], places,
                             zlib.crc32(f"{stem}{cond}".encode()) % 9999,
                             args.n_boot)
        rows.append((f"{OPTIMISED_LABEL[cond]} ({label})", d, lo, hi))

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

    # One column of an IEEE two-column page. Each placement rule carries both
    # attackers on one row, so the seven rules stay legible where fourteen
    # separate rows would not.
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.78),
                             gridspec_kw={"width_ratios": [1.32, 1]})
    ax = axes[0]
    ax.axvspan(-0.01, 0.01, color="0.88", zorder=0)
    ax.axvline(0, color="0.4", lw=0.7, zorder=1)
    names = [PLACEMENT_LABEL[k] for k in
             ["learned", "anti_oracle_grad", "oracle_grad", "saliency",
              "center", "random_fixed", "edge"]]
    by_rule = {n: [] for n in names}
    for label, d, lo, hi in rows:
        rule, attacker = label.rsplit(" (", 1)
        by_rule[rule].append((attacker.rstrip(")"), d, lo, hi))
    for y, rule in enumerate(names):
        for (attacker, d, lo, hi), off, marker in zip(
                by_rule[rule], (-0.17, 0.17), ("o", "s")):
            colour = "#2a6099" if attacker == "ResNet18" else "#b03030"
            ax.plot([lo, hi], [y + off, y + off], color=colour, lw=0.9, zorder=2)
            ax.plot([d], [y + off], marker, ms=2.6, color=colour, zorder=3)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=5.2)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlabel(r"$\Delta$ Top-1 vs uniform", fontsize=6.4)
    ax.set_title("Allocation", fontsize=7)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_xlim(-0.045, 0.045)
    ax.set_xticks([-0.04, 0.0, 0.04])
    # No legend: it collides with the widest interval whichever corner it goes
    # in, and the caption can carry two words.

    ax = axes[1]
    ax.axvspan(-0.01, 0.01, color="0.88", zorder=0)
    ax.axvline(0, color="0.4", lw=0.7, zorder=1)
    short = {"ResNet18, 3 surrogates": "ResNet18",
             "Patch-NetVLAD, 3": "Patch-NetVLAD",
             "MixVPR, 4 surrogates": "MixVPR",
             "ViT-B/16, 3 (no shared trunk)": "ViT-B/16$^{*}$"}
    ys = np.arange(len(direction))
    for y, (label, d, lo, hi) in zip(ys, direction):
        ax.plot([lo, hi], [y, y], color="#2a6099", lw=1.0, zorder=2)
        ax.plot([d], [y], "o", ms=3.0, color="#2a6099", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([short.get(r[0], r[0]) for r in direction], fontsize=5.6)
    ax.set_ylim(len(direction) - 0.5, -0.5)
    ax.set_xlabel(r"$\Delta$ Top-1 vs isotropic", fontsize=6.4)
    ax.set_title("Direction", fontsize=7)
    ax.tick_params(axis="x", labelsize=5.8)
    ax.set_xlim(-0.26, 0.05)
    ax.set_xticks([-0.2, -0.1, 0.0])

    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # The two panels differ in scale by a factor of four; the caption says so,
    # and the shared margin band is drawn on both so the reader can see it.
    fig.tight_layout(pad=0.35, w_pad=0.9)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[figure] {out}")

    # A figure's numbers should be checkable like a table's. This sidecar
    # carries every plotted point and interval in the form the claim auditor
    # locates, so what the figure shows is verified against the released rows
    # rather than trusted because it was drawn by a script.
    side = out.with_suffix("").with_name("fig_axes_values") .with_suffix(".tex")
    side = Path(str(REPO / "paper" / "generated" / "fig_axes_values.tex"))
    lines = ["% Generated by src/scripts/make_axes_figure.py. Not typeset:",
             "% the values plotted in the two-axis figure, so the claim",
             "% auditor can verify a figure the way it verifies a table.",
             "\\begin{comment}"]
    for name, d, lo, hi in rows + direction:
        lines.append(f"% {name}: ${d:+.4f}$ $[{lo:+.3f},{hi:+.3f}]$")
    lines.append("\\end{comment}")
    side.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[values] {side}")
    for name, d, lo, hi in rows + direction:
        print(f"[data ] {name:34s} {d:+.4f} [{lo:+.4f},{hi:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
