"""Generate the manuscript's direction tables and Fig. 3 from the D6 summaries.

Every cell the paper quotes comes from `analyze_tifs_d6.py --json`, so the
figure, the LaTeX tables and the prose cannot drift apart -- which they have
twice in this repository's history.

Writes:
  paper/figs/fig_preprocessing_eot.pdf   trained and held-out transforms,
                                         unhardened against hardened
  paper/generated/tab_transfer.tex       the transfer ladder (manuscript Table II)
  paper/generated/tab_sanitize.tex       the full preprocessing table (supplement)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, CONTEXT = "#2a78d6", "#c3c2b7"
MUTED, INK, GRID = "#898781", "#0b0b0b", "#e6e5e0"
COLUMN_IN = 3.5

TRAINED = ["none", "jpeg75", "jpeg50", "blur", "denoise"]
HELD_OUT = ["jpeg60", "jpeg30", "median3", "resize_half", "blur2",
            "bitdepth4", "random_one", "jpeg50_blur"]
PRETTY = {"none": "none", "jpeg75": "JPEG-75", "jpeg50": "JPEG-50",
          "blur": "blur", "denoise": "denoise", "jpeg60": "JPEG-60",
          "jpeg30": "JPEG-30", "median3": "median", "resize_half": "resize",
          "blur2": r"blur $\sigma$2", "bitdepth4": "4-bit",
          "random_one": "random", "jpeg50_blur": "JPEG+blur"}
ALPHA = 0.05


def sig(cell):
    return cell["wilcoxon_p"] < ALPHA


def fmt_p(p):
    if p >= 0.01:
        return "%.2f" % p
    exp = 0
    while p < 1 and p != 0:
        p *= 10
        exp += 1
    return "$%.0f{\\times}10^{-%d}$" % (p, exp)


def figure(summaries, out):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.labelsize": 7.5,
        "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6,
        "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    order = TRAINED + HELD_OUT
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.3), sharex=True)
    width = 0.38
    for ax, (name, s) in zip(axes, summaries):
        x = list(range(len(order)))
        unh = [s["unhardened"][k]["delta"] for k in order]
        har = [s["hardened"][k]["delta"] for k in order]
        floor = min(min(unh), min(har)) * 1.30
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.axvline(len(TRAINED) - 0.5, color=GRID, lw=1.0, zorder=0)
        ax.bar([i - width / 2 - 0.02 for i in x], unh, width, color=CONTEXT,
               edgecolor="none", zorder=2, label="unhardened")
        ax.bar([i + width / 2 + 0.02 for i in x], har, width, color=BLUE,
               edgecolor="none", zorder=2, label="EOT-hardened")
        for i, k in enumerate(order):
            if not sig(s["unhardened"][k]):
                ax.text(i - width / 2 - 0.02, unh[i] - 0.004, "n.s.",
                        ha="center", va="top", fontsize=5, color=MUTED)
        # The panel name goes inside, at the floor, where no bar reaches.
        ax.text(0.5, 0.06, name, transform=ax.transAxes, ha="left",
                va="bottom", fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(floor, 0.008)
        ax.set_ylabel(r"$\Delta$ Top-1")

    # Region labels above the top panel, clear of everything else.
    axes[0].text((len(TRAINED) - 1) / 2.0, 1.06, "hardened against these",
                 transform=axes[0].get_xaxis_transform(), ha="center",
                 va="bottom", fontsize=5.5, color=MUTED)
    axes[0].text(len(TRAINED) + (len(HELD_OUT) - 1) / 2.0, 1.06,
                 "never seen by the optimiser",
                 transform=axes[0].get_xaxis_transform(), ha="center",
                 va="bottom", fontsize=5.5, color=MUTED)
    axes[1].set_xticks(list(range(len(order))))
    axes[1].set_xticklabels([PRETTY[k] for k in order], rotation=40, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2,
               handlelength=1.0, bbox_to_anchor=(0.99, 1.10))
    fig.text(0.5, -0.02, "negative is better privacy; n.s. marks a cell whose "
             "95% interval spans zero", ha="center", fontsize=5.5, color=MUTED)
    fig.tight_layout(h_pad=0.5)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("figure -> %s" % out)


def transfer_table(summaries, out):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Direction transfers with no attacker access of any kind, and",
        r"how much of the benefit is reachable shrinks as the attacker gets",
        r"stronger. $\Delta$ is paired against the isotropic control, with a",
        r"query-level bootstrap interval and a Wilcoxon signed-rank test over the",
        r"400 per-query differences, seeds averaged within a query. Place-clustered",
        r"intervals for the headline rows are given in \S\ref{sec:direction}.}",
        r"\label{tab:transfer}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccc}", r"\hline",
        r"Direction from & Top-1 $\downarrow$ & $\Delta$ (95\% CI) & $p$ \\",
        r"\hline",
    ]
    titles = {"resnet18": r"\textit{Weak attacker: ResNet18, clean Top-1 0.21}",
              "mixvpr": r"\textit{Strong attacker: MixVPR, clean Top-1 0.79}"}
    names = {"transfer_1": "1 surrogate", "transfer_2": "2 surrogates",
             "transfer_3": "3 surrogates", "transfer_4": "4 surrogates",
             "white_box": "white box"}
    for key, s in summaries:
        tag = "resnet18" if "ResNet18" in key else "mixvpr"
        lines.append(r"\multicolumn{4}{l}{%s} \\" % titles[tag])
        ctrl = list(s["transfer"].values())[0]["control"]
        lines.append(r"none (isotropic control) & %.4f & --- & --- \\" % ctrl)
        for cond in ("transfer_1", "transfer_2", "transfer_3", "transfer_4"):
            if cond not in s["transfer"]:
                continue
            c = s["transfer"][cond]
            lines.append(r"%s & %.4f & $%+.4f$ [%+.3f, %+.3f] & %s \\"
                         % (names[cond], c["top1"], c["delta"], c["ci_low"],
                            c["ci_high"], fmt_p(c["wilcoxon_p"])))
        c = s["transfer"]["white_box"]
        lines.append(r"white box & %.4f & $%+.4f$ [%+.3f, %+.3f] & %s \\"
                     % (c["top1"], c["delta"], c["ci_low"], c["ci_high"],
                        fmt_p(c["wilcoxon_p"])))
        lines.append(r"\hline")
    lines += [r"\end{tabular}%", r"}", r"\end{table}"]
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("transfer table -> %s" % out)


def sanitize_table(summaries, out):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Every attacker-side transform, unhardened against",
        r"EOT-hardened, on both attackers. The first five are the transforms",
        r"the hardening was optimised over; the last eight it never saw.",
        r"$\Delta$ is against that row's own isotropic control, with a",
        r"query-cluster bootstrap interval; $\dagger$ marks a cell whose",
        r"interval spans zero. ``W.b.'' is the white-box bound under the same",
        r"transform.}",
        r"\label{tab:sanitize}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccccc}", r"\hline",
        r" & & \multicolumn{2}{c}{Unhardened} & \multicolumn{2}{c}{EOT-hardened} & \\",
        r"Attacker & Transform & Top-1 $\downarrow$ & $\Delta$ & Top-1 $\downarrow$ & $\Delta$ & W.b. \\",
        r"\hline",
    ]
    order = TRAINED + HELD_OUT
    for key, s in summaries:
        name = "ResNet18" if "ResNet18" in key else "MixVPR"
        lines.append(r"\multirow{13}{*}{%s}" % name)
        for k in order:
            u, h = s["unhardened"][k], s["hardened"][k]
            wb = s["white_box"].get(k, {}).get("hardened")
            mark = "" if sig(u) else r"$^\dagger$"
            sep = r"\cline{2-7}" if k == "denoise" else ""
            lines.append(r" & %s & %.4f & $%+.4f$%s & \textbf{%.4f} & "
                         r"$\mathbf{%+.4f}$ & %s \\ %s"
                         % (PRETTY[k], u["top1"], u["delta"], mark, h["top1"],
                            h["delta"], "%.4f" % wb if wb is not None else "--",
                            sep))
        lines.append(r"\hline")
    lines += [r"\end{tabular}%", r"}", r"\end{table}"]
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("sanitize table -> %s" % out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", default="src/outputs/tifs_d6/summary")
    ap.add_argument("--paper", default="paper")
    args = ap.parse_args()

    summaries = []
    for tag, label in (("resnet18", "ResNet18"), ("mixvpr", "MixVPR")):
        with open(os.path.join(args.summary_dir, tag + ".json"),
                  encoding="utf-8") as fh:
            summaries.append((label, json.load(fh)))

    gen = os.path.join(args.paper, "generated")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(os.path.join(args.paper, "figs"), exist_ok=True)
    figure(summaries, os.path.join(args.paper, "figs",
                                   "fig_preprocessing_eot.pdf"))
    transfer_table(summaries, os.path.join(gen, "tab_transfer.tex"))
    sanitize_table(summaries, os.path.join(gen, "tab_sanitize.tex"))

    # A one-line summary of the finding the held-out block exists to test.
    for label, s in summaries:
        u_fail = [k for k in HELD_OUT if not sig(s["unhardened"][k])]
        h_fail = [k for k in HELD_OUT if not sig(s["hardened"][k])]
        print("%-9s held-out: unhardened fails %d/8 %s; hardened fails %d/8 %s"
              % (label, len(u_fail), u_fail or "", len(h_fail), h_fail or ""))


if __name__ == "__main__":
    main()
