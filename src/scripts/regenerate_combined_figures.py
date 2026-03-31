"""Regenerate paper figures from the current unified benchmark outputs."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figs")
UNIFIED_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "controlled_retrieval_unified8_large")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_robustness():
    """Load robustness_summary from the unified eight-backbone run."""
    return read_csv(os.path.join(UNIFIED_DIR, "robustness_summary.csv"))


def plot_robustness(rows, outpath, backbones):
    """Plot Top-1 vs gallery size for selected backbones, raw vs PPEDCRF."""
    # Display-friendly backbone labels
    label_map = {
        "resnet18": "ResNet18",
        "resnet50": "ResNet50",
        "vgg16": "VGG16",
        "clip_vitb32": "CLIP ViT-B/32",
        "clip_vitl14": "CLIP ViT-L/14",
        "cosplace": "CosPlace",
        "mixvpr": "MixVPR",
        "patchnetvlad": "Patch-NetVLAD",
    }

    fig, axes = plt.subplots(1, len(backbones), figsize=(3.0 * len(backbones), 3.0),
                              sharey=True)
    if len(backbones) == 1:
        axes = [axes]

    for ax, bb in zip(axes, backbones):
        bb_rows = [r for r in rows if r["backbone"] == bb]
        for variant, label, marker, color in [
            ("raw", "Raw", "s", "#1f77b4"),
            ("ppedcrf", "PPEDCRF", "o", "#d62728"),
        ]:
            vrows = sorted(
                [r for r in bb_rows if r["variant"] == variant],
                key=lambda r: int(r["gallery_size"]),
            )
            gs = [int(r["gallery_size"]) for r in vrows]
            t1 = [float(r["R@1_mean"]) for r in vrows]
            ax.plot(gs, t1, marker=marker, color=color, label=label, linewidth=1.5)
        ax.set_title(label_map.get(bb, bb), fontsize=10)
        ax.set_xlabel("Gallery size")
        ax.set_xticks([12, 24, 48])
        ax.set_ylim(-0.02, 0.75)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Top-1 Retrieval Accuracy")
    axes[0].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot_frontier(outpath):
    """Plot privacy-utility frontier from current base frontier_summary."""
    rows = read_csv(os.path.join(UNIFIED_DIR, "frontier_summary.csv"))
    fig, ax = plt.subplots(figsize=(5, 3.5))

    for variant, label, marker, color in [
        ("ppedcrf", "PPEDCRF", "o", "#d62728"),
        ("full_frame", "Global Gaussian", "^", "#1f77b4"),
    ]:
        vrows = sorted(
            [r for r in rows if r["variant"] == variant],
            key=lambda r: float(r["psnr_mean"]),
        )
        psnr = [float(r["psnr_mean"]) for r in vrows]
        t1 = [float(r["R@1_mean"]) for r in vrows]
        ax.plot(psnr, t1, marker=marker, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("Top-1 Retrieval Accuracy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    rob_rows = load_robustness()
    top_backbones = ["resnet18", "resnet50", "vgg16", "clip_vitb32"]
    bottom_backbones = ["clip_vitl14", "cosplace", "mixvpr", "patchnetvlad"]

    rob_top = os.path.join(PAPER_DIR, "retrieval_robustness_topk_top.jpg")
    plot_robustness(rob_rows, rob_top, top_backbones)

    rob_bottom = os.path.join(PAPER_DIR, "retrieval_robustness_topk_bottom.jpg")
    plot_robustness(rob_rows, rob_bottom, bottom_backbones)

    frontier_out = os.path.join(PAPER_DIR, "privacy_utility_tradeoff.jpg")
    plot_frontier(frontier_out)
