"""Paper figures for the TIFS revision (review R5), from published numbers.

F2  placement delta versus uniform across the budget sweep, both checkpoints
    (supplementary Table `tab:placement_budget`, plus the 50-pair significance
    confirmation). Emphasis form: the three rules the text argues about carry
    colour, the rest are context in gray.
F3  what the attacker's preprocessing costs the deployable direction and how
    much hardening recovers (main-text Table `tab:sanitize`). Grouped bars,
    two series, one panel per attacker.

Every number is transcribed from the .tex tables so the figure and the table
cannot disagree; the transcription is asserted against a few anchor values.
Output: paper/figs/fig_budget_dependence.pdf, paper/figs/fig_preprocessing_eot.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Reference categorical palette (light mode), fixed slot order; the muted ink
# is the axis/label token. Slots 1-3 are documented as validating all pairs.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, INK, GRID = "#898781", "#0b0b0b", "#e6e5e0"
CONTEXT = "#c3c2b7"

COLUMN_IN = 3.5  # IEEE single-column width in inches


def _style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    })


# ----------------------------------------------------------------------- F2
SIGMAS = [4, 16, 32, 50]
BUDGET = {  # rule: (constant-map deltas, selective-map deltas), sigma_0 = 4/16/32/50
    "learned":      ([0.000, 0.000, 0.000, 0.000],   [-0.028, 0.000, 0.111, 0.028]),
    "anti-oracle":  ([0.000, -0.028, 0.000, 0.000],  [0.000, 0.000, 0.028, 0.000]),
    "oracle":       ([-0.028, 0.028, 0.111, 0.083],  [-0.028, 0.111, 0.139, 0.056]),
    "saliency":     ([0.000, 0.056, 0.056, 0.056],   [0.000, 0.083, 0.111, 0.056]),
    "centre bias":  ([0.028, 0.139, 0.167, 0.139],   [0.028, 0.167, 0.222, 0.222]),
    "fixed random": ([0.028, 0.083, 0.111, 0.056],   [0.028, 0.111, 0.111, 0.028]),
    "edge":         ([0.028, 0.083, -0.111, -0.167], [0.028, 0.083, -0.111, -0.139]),
}
# 50-pair confirmation: (panel, sigma) cells where edge beats uniform at the
# cluster-robust level.
EDGE_SIGNIFICANT = {(0, 50), (1, 32), (1, 50)}
OPERATING_POINT = 8

assert BUDGET["edge"][0][3] == -0.167 and BUDGET["centre bias"][1][2] == 0.222


def fig_budget(out: Path) -> None:
    emphasis = {"edge": BLUE, "learned": ORANGE, "oracle": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.9), sharey=True)
    titles = ["constant map", "selective map"]
    for k, (ax, title) in enumerate(zip(axes, titles)):
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.axvline(OPERATING_POINT, color=GRID, lw=0.8, ls=(0, (3, 2)), zorder=0)
        for rule, series in BUDGET.items():
            y = series[k]
            if rule in emphasis:
                ax.plot(SIGMAS, y, color=emphasis[rule], lw=1.4, marker="o",
                        ms=3.2, zorder=3)
            else:
                ax.plot(SIGMAS, y, color=CONTEXT, lw=0.9, zorder=2)
        for sig in SIGMAS:
            if (k, sig) in EDGE_SIGNIFICANT:
                yv = BUDGET["edge"][k][SIGMAS.index(sig)]
                ax.plot([sig], [yv], marker="*", ms=7, color=BLUE, zorder=4,
                        markeredgecolor="white", markeredgewidth=0.5)
        ax.set_title(title, color=INK, pad=3)
        ax.set_xticks(SIGMAS)
        ax.set_xticklabels([str(v) for v in SIGMAS])
        ax.set_xlabel(r"noise scale $\sigma_0$")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.text(OPERATING_POINT + 1.5, -0.195, "operating point", ha="left",
                va="bottom", fontsize=5.5, color=MUTED, rotation=90)
        # Direct labels at the right end for the emphasised rules, offset in
        # points so close endpoints do not collide.
        for rule, col in emphasis.items():
            yv = BUDGET[rule][k][-1]
            dy = {"edge": 0, "learned": -4, "oracle": 4}[rule]
            ax.annotate(rule, (SIGMAS[-1], yv), xytext=(3, dy),
                        textcoords="offset points", fontsize=6, color=col,
                        va="center", ha="left")
    axes[0].set_ylabel(r"$\Delta$ Top-1 vs uniform")
    axes[0].set_ylim(-0.2, 0.3)
    axes[0].set_xlim(2, 64)
    axes[1].set_xlim(2, 64)
    handles = [Line2D([], [], color=CONTEXT, lw=0.9, label="other rules"),
               Line2D([], [], color=BLUE, marker="*", ms=6, lw=0,
                      label="significant at 50 pairs")]
    axes[0].legend(handles=handles, loc="upper left", frameon=False,
                   handlelength=1.4, borderaxespad=0.1, labelspacing=0.2)
    fig.text(0.5, -0.03, "positive is worse privacy",
             ha="center", fontsize=6, color=MUTED)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------------------------------------------------------- F3
TRANSFORMS = ["none", "JPEG-75", "JPEG-50", "blur", "denoise"]
SANITIZE = {  # attacker: (unhardened delta, hardened delta), per transform
    "ResNet18": ([-0.1608, -0.0883, -0.0492, -0.0292, -0.0350],
                 [-0.1800, -0.1792, -0.1525, -0.1175, -0.0925]),
    "MixVPR":   ([-0.0458, -0.0242, -0.0108, -0.0317, -0.0050],
                 [-0.0842, -0.0833, -0.0592, -0.1058, -0.1042]),
}
NOT_SIGNIFICANT = {("MixVPR", "JPEG-50"), ("MixVPR", "denoise")}

assert SANITIZE["ResNet18"][1][1] == -0.1792 and SANITIZE["MixVPR"][0][4] == -0.0050


def fig_sanitize(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.9))
    width = 0.36
    for ax, (attacker, (unh, hard)) in zip(axes, SANITIZE.items()):
        x = list(range(len(TRANSFORMS)))
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.bar([i - width / 2 - 0.02 for i in x], unh, width, color=CONTEXT,
               edgecolor="none", zorder=2, label="unhardened")
        ax.bar([i + width / 2 + 0.02 for i in x], hard, width, color=BLUE,
               edgecolor="none", zorder=2, label="EOT-hardened")
        for i, (u, h) in enumerate(zip(unh, hard)):
            ax.text(i + width / 2 + 0.02, h - 0.004, f"{h:.2f}".replace("-0.", "−."),
                    ha="center", va="top", fontsize=5.5, color=INK)
            if (attacker, TRANSFORMS[i]) in NOT_SIGNIFICANT:
                ax.text(i - width / 2 - 0.02, 0.003, "n.s.", ha="center",
                        va="bottom", fontsize=5.5, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels(TRANSFORMS, rotation=30, ha="right")
        ax.set_title(attacker, color=INK, pad=3)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.215, 0.012)
    axes[0].set_ylabel(r"$\Delta$ Top-1 vs isotropic control")
    axes[1].set_yticklabels([])
    axes[1].legend(loc="lower left", frameon=False, handlelength=1.0,
                   borderaxespad=0.2)
    fig.text(0.5, -0.05, "negative is better privacy",
             ha="center", fontsize=6, color=MUTED)
    fig.tight_layout(w_pad=0.4)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "paper" / "figs")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _style()
    fig_budget(args.out_dir / "fig_budget_dependence.pdf")
    fig_sanitize(args.out_dir / "fig_preprocessing_eot.pdf")
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
