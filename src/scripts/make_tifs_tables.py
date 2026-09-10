"""Generate the manuscript's direction tables and Fig. 3 from the D6 summaries.

Every cell the paper quotes comes from `analyze_tifs_d6.py --json`, so the
figure, the LaTeX tables and the prose cannot drift apart -- which they have
twice in this repository's history.

The summaries live under src/outputs/, which is gitignored, so a fresh
checkout does not have them. The per-query rows they are built from do
travel, under src/exports/tifs_d6/, and one command turns them back into the
summaries:

    python src/scripts/rebuild_tifs_d6_summaries.py

Writes:
  paper/figs/fig_preprocessing_eot.pdf   trained and held-out transforms,
                                         unhardened against hardened
  paper/generated/tab_transfer.tex       the transfer ladder (manuscript Table II)
  paper/generated/tab_sanitize.tex       the full preprocessing table (supplement)

`make_tifs_review_figures.py` writes a *different* figure to that same
filename -- two panels over the trained transforms only, no held-out block --
and that is the one currently in the supplement. Until the two are reconciled,
pass --no_figure to regenerate the tables without disturbing it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pvalue import fmt_p_exponent as fmt_p  # noqa: E402

BLUE, CONTEXT = "#2a78d6", "#c3c2b7"
MUTED, INK, GRID = "#898781", "#0b0b0b", "#e6e5e0"
COLUMN_IN = 3.5

# Three groups, kept apart because a list called TRAINED that contains the
# untransformed control is how the supplement's caption came to count five
# transforms where the EOT expectation runs over four. The row order the
# tables and the figure use is CONTROL + TRAINED + HELD_OUT; nothing else may
# assume that the first element of TRAINED is the control.
CONTROL = ["none"]
TRAINED = ["jpeg75", "jpeg50", "blur", "denoise"]
HELD_OUT = ["jpeg60", "jpeg30", "median3", "resize_half", "blur2",
            "bitdepth4", "random_one", "jpeg50_blur"]
ROW_ORDER = CONTROL + TRAINED + HELD_OUT
# Everything left of the figure's divider and above the tables' first rule.
SEEN_AT_OPTIMISATION = len(CONTROL) + len(TRAINED)
PRETTY = {"none": "none", "jpeg75": "JPEG-75", "jpeg50": "JPEG-50",
          "blur": "blur", "denoise": "denoise", "jpeg60": "JPEG-60",
          "jpeg30": "JPEG-30", "median3": "median", "resize_half": "resize",
          "blur2": r"blur $\sigma$2", "bitdepth4": "4-bit",
          "random_one": "random", "jpeg50_blur": "JPEG+blur"}
ALPHA = 0.05


def sig(cell):
    return cell["wilcoxon_p"] < ALPHA


def wb_cell(value):
    """A white-box Top-1 cell, or a dash where the summary has no value."""
    return "--" if value is None else "%.4f" % value


def held_out_summary(csv_paths):
    """Transfer statistics for an attacker that has its own export tree.

    The two held-out attackers (a VGG16/NetVLAD retriever and a ViT-B/16 one)
    were run separately from the D6 sweep, so their rows cannot come from the
    D6 summaries. They must still be the *same* statistic as the rest of the
    table, or the table mixes units of inference: the query-level bootstrap and
    Wilcoxon signed-rank of ``analyze_tifs_d6`` are therefore imported and
    applied to those exports rather than reimplemented here. The place-clustered
    intervals the text quotes for these attackers are a different unit and stay
    in the text.
    """
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from analyze_tifs_d6 import compare, load

    hits, _ = load(csv_paths)
    out = {}
    for cond in ("transfer_3", "white_box"):
        st = compare(hits, "none", cond)
        if st is not None:
            out[cond] = st
    return out


def figure(summaries, out):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 7, "axes.labelsize": 7.5,
        "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 6,
        "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    order = ROW_ORDER
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_IN, 3.3), sharex=True)
    width = 0.38
    for ax, (name, s) in zip(axes, summaries):
        x = list(range(len(order)))
        unh = [s["unhardened"][k]["delta"] for k in order]
        har = [s["hardened"][k]["delta"] for k in order]
        floor = min(min(unh), min(har)) * 1.30
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.axvline(SEEN_AT_OPTIMISATION - 0.5, color=GRID, lw=1.0, zorder=0)
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
    axes[0].text((SEEN_AT_OPTIMISATION - 1) / 2.0, 1.06,
                 "hardened against these",
                 transform=axes[0].get_xaxis_transform(), ha="center",
                 va="bottom", fontsize=5.5, color=MUTED)
    axes[0].text(SEEN_AT_OPTIMISATION + (len(HELD_OUT) - 1) / 2.0, 1.06,
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


def transfer_table(summaries, held_out, out):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Direction transfers with no attacker access of any kind, and",
        r"how much of the benefit is reachable shrinks as the attacker gets",
        r"stronger. $\Delta$ is paired against the isotropic control, with a",
        r"query-level bootstrap interval and a Wilcoxon signed-rank test over the",
        r"per-query differences, seeds averaged within a query. All four",
        r"attackers are held out from the optimiser; the lower two are held out",
        r"from the surrogate ensemble's trunks as well, and the trunk of the last",
        r"appears in no surrogate at all. Place-clustered",
        r"intervals for the headline rows are given in \S\ref{sec:direction}.}",
        r"\label{tab:transfer}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lccc}", r"\hline",
        r"Direction from & Top-1 $\downarrow$ & $\Delta$ (95\% CI) & $p$ \\",
        r"\hline",
    ]
    titles = {"resnet18": r"\textit{Weak attacker: ResNet18, clean Top-1 0.21}",
              "mixvpr": r"\textit{Strong attacker: MixVPR, clean Top-1 0.79}"}
    names = {"transfer_1": "1 surrogate", "transfer_2": "2 surrogates",
             "transfer_3": "3 surrogates", "transfer_4": "4 surrogates",
             "white_box": "white box"}

    def row(label, cell):
        return (r"%s & %.4f & $%+.4f$ [%+.3f, %+.3f] & %s \\"
                % (label, cell["top1"], cell["delta"], cell["ci_low"],
                   cell["ci_high"], fmt_p(cell["wilcoxon_p"])))

    for key, s in summaries:
        tag = "resnet18" if "ResNet18" in key else "mixvpr"
        lines.append(r"\multicolumn{4}{l}{%s} \\" % titles[tag])
        ctrl = list(s["transfer"].values())[0]["control"]
        lines.append(r"none (isotropic control) & %.4f & --- & --- \\" % ctrl)
        for cond in ("transfer_1", "transfer_2", "transfer_3", "transfer_4",
                     "white_box"):
            if cond in s["transfer"]:
                lines.append(row(names[cond], s["transfer"][cond]))
        lines.append(r"\hline")

    # The two attackers whose trunks the surrogate ensemble does not share, on
    # the same manifest, budget and seeds. Reported here rather than only in
    # prose so that every attacker the section discusses is in the table.
    for title, stats in held_out:
        if not stats:
            continue
        lines.append(r"\multicolumn{4}{l}{%s} \\" % title)
        ctrl = list(stats.values())[0]["control"]
        lines.append(r"none (isotropic control) & %.4f & --- & --- \\" % ctrl)
        for cond in ("transfer_3", "white_box"):
            if cond in stats:
                lines.append(row(names[cond], stats[cond]))
        lines.append(r"\hline")

    lines += [r"\end{tabular}", r"\end{table}"]
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("transfer table -> %s" % out)


def sanitize_table(summaries, out):
    """The supplement's full preprocessing table.

    Two things the caption has to get right, because a referee checks them
    against the prose. First the counts: the first row is the untransformed
    control, and the EOT expectation runs over the four transforms after it,
    not five. Second the white-box column: the summary carries the bound under
    both releases, and printing only the hardened one under an unqualified
    heading contradicts the manuscript, which quotes the unhardened range. Both
    are printed, each inside its own release's block.
    """
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Every attacker-side transform, unhardened against",
        r"EOT-hardened, on both attackers. The first row is the untransformed",
        r"control; the next four are the transforms the hardening was optimised",
        r"over; the last eight it never saw.",
        r"$\Delta$ is against that row's own isotropic control, with a",
        r"query-cluster bootstrap interval; $\dagger$ marks a cell whose",
        r"interval spans zero. The two ``W.b.'' columns are the white-box bound",
        r"under the same transform, one for the unhardened release and one for",
        r"the EOT-hardened release.}",
        r"\label{tab:sanitize}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{lcccccc}", r"\hline",
        r" & \multicolumn{3}{c}{Unhardened} & \multicolumn{3}{c}{EOT-hardened} \\",
        r"Transform & Top-1 $\downarrow$ & $\Delta$ & W.b. & Top-1 $\downarrow$ "
        r"& $\Delta$ & W.b. \\",
        r"\hline",
    ]
    titles = {"ResNet18": r"\textit{Weak attacker: ResNet18}",
              "MixVPR": r"\textit{Strong attacker: MixVPR}"}
    for key, s in summaries:
        name = "ResNet18" if "ResNet18" in key else "MixVPR"
        lines.append(r"\multicolumn{7}{l}{%s} \\" % titles[name])
        for k in ROW_ORDER:
            u, h = s["unhardened"][k], s["hardened"][k]
            wb = s["white_box"].get(k, {})
            mark = "" if sig(u) else r"$^\dagger$"
            lines.append(r"%s & %.4f & $%+.4f$%s & %s & \textbf{%.4f} & "
                         r"$\mathbf{%+.4f}$ & %s \\"
                         % (PRETTY[k], u["top1"], u["delta"], mark,
                            wb_cell(wb.get("unhardened")), h["top1"], h["delta"],
                            wb_cell(wb.get("hardened"))))
            # The rules that make the caption's three groups checkable.
            if k in (CONTROL[-1], TRAINED[-1]):
                lines.append(r"\cline{1-7}")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("sanitize table -> %s" % out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", default="src/outputs/tifs_d6/summary")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--exports", default="src/exports",
                    help="Where the held-out attackers' own export trees live.")
    ap.add_argument("--no_figure", action="store_true",
                    help="Write only the tables. Another script writes a "
                         "different figure to the same filename, so the "
                         "tables can be regenerated without touching it.")
    args = ap.parse_args()

    summaries = []
    for tag, label in (("resnet18", "ResNet18"), ("mixvpr", "MixVPR")):
        path = os.path.join(args.summary_dir, tag + ".json")
        if not os.path.exists(path):
            raise SystemExit(
                "%s is missing. src/outputs/ is gitignored, so a fresh "
                "checkout has the per-query rows but not the summaries built "
                "from them. Rebuild them with:\n"
                "    python src/scripts/rebuild_tifs_d6_summaries.py" % path)
        with open(path, encoding="utf-8") as fh:
            summaries.append((label, json.load(fh)))

    # Two attackers with no surrogate trunk in common with the ensemble. Absent
    # exports leave their blocks out rather than failing the whole build, so the
    # tables can still be regenerated on a checkout without them.
    held_out = []
    for title, tree, stem in (
            (r"\textit{Third attacker: Patch-NetVLAD, a VGG16 trunk under "
             r"NetVLAD}", "tifs_a7b", "a7b_pnv_plain"),
            (r"\textit{Fourth attacker: ViT-B/16, a trunk no surrogate has}",
             "tifs6_vit", "vit_plain")):
        path = os.path.join(args.exports, tree, stem + ".csv")
        held_out.append((title, held_out_summary([path])
                         if os.path.exists(path) else {}))
        if not os.path.exists(path):
            print("[warn] %s missing; its block is omitted" % path)

    gen = os.path.join(args.paper, "generated")
    os.makedirs(gen, exist_ok=True)
    os.makedirs(os.path.join(args.paper, "figs"), exist_ok=True)
    if not args.no_figure:
        figure(summaries, os.path.join(args.paper, "figs",
                                       "fig_preprocessing_eot.pdf"))
    transfer_table(summaries, held_out, os.path.join(gen, "tab_transfer.tex"))
    sanitize_table(summaries, os.path.join(gen, "tab_sanitize.tex"))

    # A one-line summary of the finding the held-out block exists to test.
    for label, s in summaries:
        u_fail = [k for k in HELD_OUT if not sig(s["unhardened"][k])]
        h_fail = [k for k in HELD_OUT if not sig(s["hardened"][k])]
        print("%-9s held-out: unhardened fails %d/8 %s; hardened fails %d/8 %s"
              % (label, len(u_fail), u_fail or "", len(h_fail), h_fail or ""))


if __name__ == "__main__":
    main()
