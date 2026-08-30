"""Regenerate retrieval figures from the proxy12 paired-scene benchmark."""
from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT = "src/outputs/tomm_review_proxy12/per_query.csv"
OUTPUT_DIR = "paper/figs"
LABELS = {
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "vgg16": "VGG16",
    "clip_vitb32": "CLIP ViT-B/32",
    "clip_vitl14": "CLIP ViT-L/14",
    "cosplace": "CosPlace",
    "mixvpr": "MixVPR",
    "patchnetvlad": "Patch-NetVLAD",
}


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def seed_mean(rows: list[dict[str, str]], value_fn) -> tuple[float, float]:
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(value_fn(row))
    means = [sum(values) / len(values) for values in by_seed.values()]
    return sum(means) / len(means), (
        statistics.stdev(means) if len(means) > 1 else 0.0
    )


def subset(
    rows: list[dict[str, str]],
    *,
    variant: str,
    backbone: str | None = None,
    gallery_size: int | None = None,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["variant"] == variant
        and (backbone is None or row["backbone"] == backbone)
        and (gallery_size is None or int(row["gallery_size"]) == gallery_size)
    ]


def top1(row: dict[str, str]) -> float:
    return float(int(row["correct_rank"]) == 1)


def top5(row: dict[str, str]) -> float:
    return float(row["top5_hit"])


def quality(rows: list[dict[str, str]], key: str) -> tuple[float, float]:
    valid = [row for row in rows if row[key]]
    return seed_mean(valid, lambda row: float(row[key]))


def plot_fixed_budget(rows: list[dict[str, str]]) -> None:
    variants = ["full", "global_noise", "masked_blur", "masked_mosaic"]
    labels = {
        "full": "PPEDCRF",
        "global_noise": "Global Gaussian",
        "masked_blur": "Mask-guided blur",
        "masked_mosaic": "Mask-guided mosaic",
    }
    colors = {
        "full": "#1f77b4",
        "global_noise": "#d62728",
        "masked_blur": "#2ca02c",
        "masked_mosaic": "#9467bd",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, metric_fn, metric_label in (
        (axes[0], top1, "Top-1 retrieval accuracy"),
        (axes[1], top5, "Top-5 retrieval accuracy"),
    ):
        for variant in variants:
            current = subset(
                rows, variant=variant, backbone="resnet18", gallery_size=48
            )
            retrieval, retrieval_std = seed_mean(current, metric_fn)
            psnr, _ = quality(current, "psnr_mean")
            ax.errorbar(
                psnr,
                retrieval,
                xerr=None,
                yerr=retrieval_std,
                marker="o",
                capsize=3,
                linewidth=2,
                label=labels[variant],
                color=colors[variant],
            )
            ax.annotate(labels[variant], (psnr, retrieval), xytext=(5, 5),
                        textcoords="offset points", fontsize=8,
                        color=colors[variant])
        ax.set_xlabel("PSNR (dB)", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title("Fixed-budget proxy12 comparison", fontsize=12)
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8, loc="best")
    plt.savefig(
        os.path.join(OUTPUT_DIR, "privacy_utility_tradeoff.jpg"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_robustness_group(
    rows: list[dict[str, str]], backbones: list[str], output_name: str
) -> None:
    fig, axes = plt.subplots(
        1, len(backbones), figsize=(3.0 * len(backbones), 3.0), sharey=True
    )
    axes = [axes] if len(backbones) == 1 else list(axes)
    for ax, backbone in zip(axes, backbones):
        for variant, color, label in (
            ("raw", "#7f7f7f", "Raw query"),
            ("full", "#1f77b4", "PPEDCRF"),
        ):
            values = []
            errors = []
            galleries = [12, 24, 48]
            for gallery in galleries:
                current = subset(
                    rows,
                    variant=variant,
                    backbone=backbone,
                    gallery_size=gallery,
                )
                mean, std = seed_mean(current, top1)
                values.append(mean)
                errors.append(std)
            ax.errorbar(
                galleries,
                values,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                label=label,
                color=color,
            )
        ax.set_title(LABELS.get(backbone, backbone), fontsize=10)
        ax.set_xlabel("Gallery size", fontsize=10)
        ax.set_xticks([12, 24, 48])
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Top-1 retrieval accuracy", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, output_name), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = read_rows(INPUT)
    plot_fixed_budget(rows)
    plot_robustness_group(
        rows,
        ["resnet18", "resnet50", "vgg16", "clip_vitb32"],
        "retrieval_robustness_topk_top.jpg",
    )
    plot_robustness_group(
        rows,
        ["clip_vitl14", "cosplace", "mixvpr", "patchnetvlad"],
        "retrieval_robustness_topk_bottom.jpg",
    )
    print("Regenerated proxy12 retrieval figures from", INPUT)


if __name__ == "__main__":
    main()
