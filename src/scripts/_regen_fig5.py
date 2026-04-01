"""Regenerate Fig 5 (baseline_param_sweep) from seed-averaged CSV."""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs",
                        "controlled_retrieval_seed_avg", "baseline_sweep.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figs",
                        "baseline_param_sweep.jpg")

rows = []
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        rows.append(row)

fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.2), constrained_layout=True)
style = {
    "masked_blur": {"label": "Mask-guided blur", "color": "#2ca02c", "marker": "s"},
    "masked_mosaic": {"label": "Mask-guided mosaic", "color": "#ff7f0e", "marker": "^"},
}

for variant in ("masked_blur", "masked_mosaic"):
    vrows = [r for r in rows if r["variant"] == variant]
    vrows = sorted(vrows, key=lambda item: float(item["param"]))
    if not vrows:
        continue
    xs = [float(r["psnr_mean"]) for r in vrows]
    ys = [float(r["R@1_mean"]) for r in vrows]
    ax.plot(xs, ys, marker=style[variant]["marker"], linewidth=2,
            markersize=10, label=style[variant]["label"], color=style[variant]["color"])
    for r, x, y in zip(vrows, xs, ys):
        pn = "k" if variant == "masked_blur" else "b"
        param_val = int(float(r["param"]))
        ax.annotate(f"{pn}={param_val}", (x, y),
                    xytext=(5, 6), textcoords="offset points", fontsize=9)

ax.set_xlabel("PSNR (dB)", fontsize=11)
ax.set_ylabel("R@1 retrieval accuracy", fontsize=11)
ax.set_title("Support-aware baseline parameter sweep", fontsize=12)
ax.grid(alpha=0.25)
ax.legend(frameon=False, loc="upper left", fontsize=10)
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure with {len(rows)} data points to {OUT_PATH}")
