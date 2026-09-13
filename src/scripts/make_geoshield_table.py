"""The published-mechanism audit, tabulated.

R3 asks whether the paper's null survives contact with a mechanism someone
actually released, rather than only with the perturbation families we
construct ourselves. This tabulates that arm: a public geo-privacy release
run end to end against the same attacker, the same gallery and the same
delivered distortion as every other arm, next to two controls it must beat
to mean anything -- no perturbation at all, and an isotropic perturbation
carrying the identical delivered MSE.

The isotropic row is the one that matters. A mechanism that lowers retrieval
relative to clean imagery has shown only that it added energy; the question
is whether *where and how* it spends that energy beats spending it at random,
which is the same question the placement and operator arms ask. So the
contrast, the interval and the equivalence test are all taken against the
isotropic control, and the clean row is reported only to locate the attacker.

Two properties of the release are carried into the caption from the run's own
metadata rather than retyped here: the geo-semantic term is degenerate in the
public code -- the vision-language stub returns one constant caption for
every image -- and the attack is solved at the release's native resolution
with only the resulting perturbation resampled to the protocol geometry, so
the shape comes from the mechanism and the energy from the protocol.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "scripts"))
from _pvalue import fmt_p, fmt_p_exponent  # noqa: E402
from analyze_placement_equivalence import tost  # noqa: E402
from make_msls_placement_table import boot  # noqa: E402

# Printed in the order a reader should read them: the attacker's unimpeded
# score, then the control that fixes the energy, then the mechanism.
CONDITIONS = [
    ("clean", "No perturbation"),
    ("isotropic", "Isotropic, matched MSE"),
    ("geoshield_published_fullframe", "Public release, as published"),
]
REFERENCE = "isotropic"


def per_query(rows: Sequence[dict], condition: str, field: str) -> Dict[str, float]:
    """Seeds averaged within a query, so a query is one unit of inference."""
    acc = collections.defaultdict(list)
    for r in rows:
        if r["condition"] != condition:
            continue
        if field == "top1":
            acc[r["query_id"]].append(1.0 if int(r["correct_rank"]) == 1 else 0.0)
        else:
            acc[r["query_id"]].append(float(r[field]))
    return {q: float(np.mean(v)) for q, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--metadata",
                    default=str(REPO / "src/outputs/r17_geoshield/run_metadata.json"))
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"))
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--boots", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out",
                    default=str(REPO / "paper/generated/tab_geoshield.tex"))
    args = ap.parse_args()

    rows: List[dict] = []
    for path in args.rows:
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        raise SystemExit("no rows")

    meta = json.loads(Path(args.metadata).read_text(encoding="utf-8"))

    place_of = {}
    with open(args.places, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            place_of[r["query_id"]] = r["correct_place"]

    # The energy gate, re-derived from the delivered rows rather than trusted.
    # A table that cannot show its own operating point is not checkable, and
    # this arm has already once produced tidy numbers at the wrong energy.
    delivered = {}
    for cond, _ in CONDITIONS:
        mse = [float(r["effective_mse"]) for r in rows if r["condition"] == cond]
        if not mse:
            raise SystemExit(f"condition {cond!r} missing from the rows")
        delivered[cond] = (min(mse), max(mse))
    for cond, _ in CONDITIONS:
        if cond == "clean":
            continue
        lo, hi = delivered[cond]
        if abs(hi - meta["target_mse"]) > 0.05 or abs(lo - meta["target_mse"]) > 0.05:
            raise SystemExit(
                f"{cond} delivered MSE [{lo:.4f},{hi:.4f}] is off the "
                f"{meta['target_mse']} target; this is not a matched comparison")

    # The hit columns must agree with the rank they are derived from. This
    # caught nothing on the pilot -- Top-5 was identical across all three
    # conditions, which looked like a stuck column and turned out to be real
    # (the mechanism reorders within the shortlist without evicting from it).
    # Keeping the check means the next reader of an identical column does not
    # have to redo that investigation by hand.
    for r in rows:
        rank = int(r["correct_rank"])
        for k, thr in (("top5_hit", 5), ("top10_hit", 10)):
            if int(r[k]) != int(rank <= thr):
                raise SystemExit(
                    f"{r['query_id']}/{r['condition']}: {k}={r[k]} "
                    f"disagrees with correct_rank={rank}")

    ref1 = per_query(rows, REFERENCE, "top1")
    n_places = len({place_of.get(q, q) for q in ref1})

    lines = []
    for cond, label in CONDITIONS:
        cur1 = per_query(rows, cond, "top1")
        cur5 = per_query(rows, cond, "top5_hit")
        top1, top5 = np.mean(list(cur1.values())), np.mean(list(cur5.values()))
        if cond == REFERENCE:
            lines.append(f"{label} & {top1:.4f} & {top5:.4f} & "
                         r"\multicolumn{4}{c}{reference} \\")
            continue
        qs = sorted(set(cur1) & set(ref1))
        d = np.array([cur1[q] - ref1[q] for q in qs])
        g = [place_of.get(q, q) for q in qs]
        nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if nz.size else 1.0
        ci = boot(d, g, n=args.boots, seed=args.seed)
        pt = tost(d, g, args.margin, args.boots, args.seed)
        pf = fmt_p_exponent(p) if p < 0.01 else fmt_p(p)
        verdict = ("separates" if ci[0] * ci[1] > 0
                   else ("equivalent" if pt < 0.05 else "none det."))
        lines.append(f"{label} & {top1:.4f} & {top5:.4f} & ${d.mean():+.4f}$ & "
                     f"[{ci[0]:+.3f},{ci[1]:+.3f}] & {pf}\\,({nz.size}) & "
                     f"{pt:.3f} & {verdict} \\\\")

    # The seed count is stated in the caption, derived from the rows rather
    # than remembered. This arm was run at a single seed while the placement
    # and operator arms beside it use three; a reader comparing tables would
    # otherwise reasonably assume the same design, and the difference is
    # exactly the kind of thing that is obvious to whoever ran it and
    # invisible to everyone else.
    seeds = sorted({r["seed"] for r in rows})
    if len(seeds) == 1:
        seed_note = (r"a single seed (not the three used by the "
                     r"placement and operator arms in this supplement, so the "
                     r"contrast below is a bounded result at one seed)")
    else:
        seed_note = f"{len(seeds)} seeds averaged within a query before pairing"

    ens = ", ".join(meta["clip_ensemble"])
    caption = (
        r"The published-mechanism arm: a public geo-privacy release run end to "
        r"end under this paper's protocol. Every row is read by the same "
        r"attacker over the same gallery at the same delivered distortion "
        f"(MSE {meta['target_mse']}, solved per frame by bisection through the "
        r"pixel clamp), on the primary place-labelled manifest --- "
        f"{len(ref1)} query clusters over {n_places} places, "
        f"{seed_note}. $\\Delta$ is taken "
        r"against the isotropic control rather than against clean "
        r"imagery, because lowering retrieval below clean only shows that "
        r"energy was added; the question this paper asks is whether spending "
        r"that energy as the mechanism directs beats spending it at random. "
        r"The interval is bootstrapped over places, the figure beside $p$ "
        r"counts the discordant pairs the signed-rank test runs on, and "
        f"$p_{{\\mathrm{{TOST}}}}$ is the equivalence test at the declared "
        f"$\\pm{args.margin:g}$ margin. Two properties of the release are "
        r"material to reading this row and are reported rather than assumed: "
        r"its geo-semantic term is degenerate in the public code, whose "
        r"vision-language component returns the single constant caption "
        f"``{meta['released_caption']}'' for every image, so what is measured "
        r"is the released artefact and not the method as described; and the "
        r"attack is solved at the release's own input resolution "
        f"({meta['attack_resolution']}px) with only the resulting perturbation "
        r"resampled to the protocol geometry, so its shape is the mechanism's "
        f"and its energy is the protocol's. Surrogate ensemble: {ens}."
    )

    out = [
        "% Generated by src/scripts/make_geoshield_table.py from",
        "% src/outputs/r17_geoshield/. Do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:geoshield}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Condition & Top-1 & Top-5 & $\Delta$ & place 95\% CI & $p$ & "
        r"$p_{\mathrm{TOST}}$ & Verdict \\",
        r"\hline",
        *lines,
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    Path(args.out).write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    for ln in lines:
        print("  ", ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
