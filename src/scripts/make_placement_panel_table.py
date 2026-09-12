"""The prescribed placement rules, on the three attackers that never saw them.

The fourteenth review's first finding was that this paper's two axes are read
by different attacker panels: every prescribed rule was reported against
ResNet18 and MixVPR, while the direction arm and the solved maps were reported
against five -- and the two attackers a solved map separates on, Patch-NetVLAD
and CLIP ViT-L/14, were exactly the two no prescribed rule had ever faced. The
manuscript states that as a limitation. This tabulates the measurement that
removes it.

Rows come from the r15 run: the same ten placements tab_placement_msls prints,
the same manifest, gallery, seeds, delivered distortion and energy gate, with
only the attacker that reads the released frame changed. Seeds are averaged
within a query before pairing, exactly as the protocol prescribes everywhere
else, and intervals are place-clustered.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "src" / "scripts"))
from _pvalue import fmt_p  # noqa: E402
from analyze_placement_equivalence import tost, holm, permutation_p  # noqa: E402

LABEL = {
    "learned": "learned support",
    "oracle_grad": "score-gradient",
    "anti_oracle_grad": "anti-score-grad.",
    "saliency": "saliency",
    "center": "centre bias",
    "random_fixed": "fixed random",
    "edge": "edge magnitude",
    "margin_oracle": "margin gradient",
    "anti_margin_oracle": "anti-margin grad.",
}
ORDER = list(LABEL)
ATTACKERS = [("pnv", "Patch-NetVLAD"), ("vit", "ViT-B/16"),
             ("clip", "CLIP ViT-L/14")]


def per_query(rows: Sequence[dict], placement: str) -> Dict[str, float]:
    """Top-1 indicator per query, averaged over whatever seeds are present."""
    acc = collections.defaultdict(list)
    for r in rows:
        if r["placement"] == placement:
            acc[r["query_id"]].append(1.0 if int(r["correct_rank"]) == 1 else 0.0)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def load_attacker(root: Path, tag: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(str(root / f"r15_{tag}_s*" / "per_query.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def cluster_ci(d: np.ndarray, groups, n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = collections.defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    keys = sorted(idx)
    out = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        out[b] = d[np.concatenate([idx[keys[j]] for j in pick])].mean()
    return tuple(float(x) for x in np.percentile(out, [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(REPO / "src/exports/placement_panel"))
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"))
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=10000)
    ap.add_argument("--out",
                    default=str(REPO / "paper/generated/tab_placement_panel.tex"))
    ap.add_argument("--summary",
                    default=str(REPO / "src/exports/placement_panel/summary.json"))
    args = ap.parse_args()

    place_of = {}
    with open(args.places, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            place_of[r["query_id"]] = r["correct_place"]

    body: List[str] = []
    summary = {"margin": args.margin, "attackers": []}
    for si, (tag, label) in enumerate(ATTACKERS):
        rows = load_attacker(Path(args.root), tag)
        if not rows:
            print(f"[skip] no rows for {tag}")
            continue
        ref = per_query(rows, "uniform")
        if not ref:
            print(f"[skip] {tag} has no uniform arm")
            continue
        ref_top1 = float(np.mean(list(ref.values())))
        seeds = sorted({r["seed"] for r in rows})
        body.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{label}}} "
                    rf"(uniform reference {ref_top1:.4f}, "
                    rf"{len(seeds)} seeds)}} \\")
        cells, raw_p = [], []
        for name in ORDER:
            cur = per_query(rows, name)
            if not cur:
                continue
            qs = sorted(set(cur) & set(ref))
            d = np.array([cur[q] - ref[q] for q in qs])
            g = [place_of.get(q, q) for q in qs]
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if nz.size else 1.0
            lo, hi = cluster_ci(d, g, args.n_boot, 1234 + si)
            pt = tost(d, g, args.margin, args.n_boot, 1234 + si)
            pp = permutation_p(d, g, args.n_boot, 1234 + si)
            raw_p.append(p)
            cells.append({"placement": name, "label": LABEL[name],
                          "top1": float(np.mean([cur[q] for q in qs])),
                          "delta": float(d.mean()), "ci": [lo, hi],
                          "p": p, "p_tost": pt, "p_perm": pp,
                          "n_discordant": int(nz.size)})
        for c, ph in zip(cells, holm(raw_p)):
            c["p_holm"] = ph
            # "separates" means the place-clustered interval clears zero after
            # the family correction this paper applies to the placement arm.
            c["separates"] = bool(c["ci"][1] < 0 or c["ci"][0] > 0) and ph < 0.05
            verdict = ("separates" if c["separates"]
                       else ("negligible" if max(abs(c["ci"][0]), abs(c["ci"][1]))
                             <= args.margin else "none det."))
            body.append(
                rf"\quad {c['label']} & {c['top1']:.4f} & ${c['delta']:+.4f}$ & "
                rf"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] & "
                rf"{fmt_p(c['p_holm'])} & {c['p_tost']:.3f} & {verdict} \\")
        summary["attackers"].append(
            {"tag": tag, "label": label, "uniform_top1": ref_top1,
             "seeds": seeds, "cells": cells,
             "n_separating": sum(1 for c in cells if c["separates"])})

    if not body:
        raise SystemExit("no rows found; is the run finished?")

    caption = (
        r"The prescribed placement rules against the three attackers they had "
        r"never been run against, on the primary place-labelled manifest. Same "
        r"ten placements, manifest, gallery, seeds, delivered distortion and "
        r"energy gate as the weak- and strong-attacker tables; only the "
        r"attacker that reads the released frame changes, so this is a "
        r"re-embedding and not a new search. $\Delta$ is paired against that "
        r"attacker's own uniform reference with the query as the unit and "
        r"three seeds averaged within a query before pairing; intervals are "
        r"place-clustered. $p$ is Holm-corrected across the nine contrasts of "
        r"its own block, $p_{\mathrm{TOST}}$ the place-clustered equivalence "
        r"test at the $\pm0.01$ margin. ``Separates'' means the interval "
        r"clears zero after correction, which is what the two attackers a "
        r"solved map reaches would have to show for the null over prescribed "
        r"rules to be panel-dependent.")
    lines = [
        "% Generated by src/scripts/make_placement_panel_table.py from",
        "% src/exports/placement_panel. Do not edit by hand.",
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:placement_panel}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\begin{tabular}{lcccccc}", r"\hline",
        r"Placement & Top-1 & $\Delta$ & place 95\% CI & $p$ & "
        r"$p_{\mathrm{TOST}}$ & Verdict \\",
        r"\hline", *body, r"\hline", r"\end{tabular}", r"\end{table}",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dest = Path(args.summary)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[done] wrote {out}")
    for a in summary["attackers"]:
        print(f"  {a['label']:16s} uniform {a['uniform_top1']:.4f}  "
              f"{a['n_separating']} of {len(a['cells'])} rules separate")
        for c in a["cells"]:
            flag = "  <-- SEPARATES" if c["separates"] else ""
            print(f"    {c['label']:18s} {c['delta']:+.4f} "
                  f"[{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
                  f"holm={c['p_holm']:.3f} tost={c['p_tost']:.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
