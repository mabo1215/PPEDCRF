"""Regenerate frontier and robustness figures with actual experimental data."""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("paper/figs", exist_ok=True)

VARIANT_LABELS = {"ppedcrf": "PPEDCRF", "full_frame": "Global Gaussian noise"}
colors = {"ppedcrf": "#1f77b4", "full_frame": "#d62728"}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


frontier_rows = load_csv("src/outputs/controlled_retrieval/frontier_summary.csv")
robustness_rows = load_csv("src/outputs/controlled_retrieval/robustness_summary.csv")

# ----- Privacy-utility frontier -----
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
metric_pairs = [("R@1_mean", "R@1"), ("R@5_mean", "R@5")]

for ax, (metric_key, metric_label) in zip(axes, metric_pairs):
    for variant in ("ppedcrf", "full_frame"):
        rows = sorted(
            [r for r in frontier_rows if r["variant"] == variant],
            key=lambda r: float(r["sigma"]),
        )
        xs = [float(r["psnr_mean"]) for r in rows]
        ys = [float(r[metric_key]) for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=2,
                label=VARIANT_LABELS[variant], color=colors[variant])
        for row, x, y in zip(rows, xs, ys):
            sigma_val = int(float(row["sigma"]))
            ax.annotate(
                r"$\sigma=" + str(sigma_val) + r"$",
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("PSNR (dB)", fontsize=11)
    ax.set_ylabel(f"{metric_label} retrieval accuracy", fontsize=11)
    ax.set_title(f"Privacy-utility frontier ({metric_label})", fontsize=12)
    ax.grid(alpha=0.25)

axes[1].legend(frameon=False, loc="best")
plt.savefig("paper/figs/privacy_utility_tradeoff.jpg", dpi=300, bbox_inches="tight")
plt.close()
print("Saved privacy_utility_tradeoff.jpg")

# ----- Robustness plot -----
backbones = sorted({r["backbone"] for r in robustness_rows})
fig, axes = plt.subplots(1, len(backbones), figsize=(5.2 * len(backbones), 4.2), constrained_layout=True)
if len(backbones) == 1:
    axes = [axes]

rcolors = {"raw": "#7f7f7f", "ppedcrf": "#1f77b4"}
rlabels = {"raw": "Raw query", "ppedcrf": "PPEDCRF"}

for ax, backbone in zip(axes, backbones):
    for variant in ("raw", "ppedcrf"):
        rows = sorted(
            [r for r in robustness_rows if r["backbone"] == backbone and r["variant"] == variant],
            key=lambda r: int(r["gallery_size"]),
        )
        xs = [int(r["gallery_size"]) for r in rows]
        ys = [float(r["R@1_mean"]) for r in rows]
        yerrs = [float(r["R@1_std"]) for r in rows]
        ax.errorbar(xs, ys, yerr=yerrs, marker="o", linewidth=2, capsize=3,
                    label=rlabels[variant], color=rcolors[variant])
    ax.set_title(f"Retrieval robustness ({backbone})", fontsize=12)
    ax.set_xlabel("Gallery size", fontsize=11)
    ax.set_ylabel("R@1 retrieval accuracy", fontsize=11)
    ax.set_xticks([12, 24, 48])
    ax.grid(alpha=0.25)

axes[-1].legend(frameon=False, loc="best")
plt.savefig("paper/figs/retrieval_robustness_topk.jpg", dpi=300, bbox_inches="tight")
plt.close()
print("Saved retrieval_robustness_topk.jpg")
