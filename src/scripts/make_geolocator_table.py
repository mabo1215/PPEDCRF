"""The geolocator arm, read the way the rest of the paper reads its contrasts.

Success against a model that emits a coordinate is distance, not rank, so
every condition is scored by whether the predicted point lands inside a
threshold of the query's own coordinate. That indicator is then treated
exactly as Top-1 is treated elsewhere: three seeds averaged within a query
before pairing, the query as the unit of inference, place-clustered bootstrap
intervals, Holm correction within the family, and the same two-one-sided-test
equivalence check at a margin declared in advance.

Two subsets are reported and the difference between them matters. On a query
the clean attacker already fails, a perturbation that moves the prediction
earns credit it did not earn, and one that happens to move it closer looks
like harm; neither is evidence about a defense. The localisable subset --
queries the clean attacker places inside the primary threshold -- is where
location privacy is actually at stake, so it carries the claim, and the
all-query column is printed beside it rather than instead of it.
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
from analyze_placement_equivalence import holm, tost  # noqa: E402

LABEL = {
    "clean": "unperturbed",
    "isotropic": "isotropic control",
    "edge": "edge magnitude",
    "saliency": "saliency",
    "direction": "direction (surrogates)",
    "hardened": "direction, hardened",
}
ORDER = ["clean", "isotropic", "edge", "saliency", "direction", "hardened"]


def load(root: Path) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(str(root / "*.csv"))):
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def hit_rate(rows: Sequence[dict], cond: str, km: float) -> Dict[str, float]:
    """Per query, the share of seeds whose prediction lands inside km."""
    acc = collections.defaultdict(list)
    for r in rows:
        if r["condition"] == cond:
            acc[r["query_id"]].append(1.0 if float(r["error_km"]) <= km else 0.0)
    return {q: float(np.mean(v)) for q, v in acc.items()}


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
    ap.add_argument("--root", default=str(REPO / "src/exports/geolocator"))
    ap.add_argument("--primary-km", dest="primary_km", type=float, default=25.0)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 25.0, 200.0])
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--n-boot", dest="n_boot", type=int, default=10000)
    ap.add_argument("--out", default=str(REPO / "paper/generated/tab_geolocator.tex"))
    ap.add_argument("--summary", default=str(
        REPO / "src/exports/geolocator/summary.json"))
    args = ap.parse_args()

    rows = load(Path(args.root))
    if not rows:
        raise SystemExit("no rows; is the run finished?")
    place_of = {r["query_id"]: r["place_id"] for r in rows}
    conds = [c for c in ORDER if any(r["condition"] == c for r in rows)]
    print(f"[geo] {len(rows)} rows, conditions {conds}")

    # The localisable subset: queries the unperturbed attacker already places
    # inside the primary threshold. Everything else cannot speak to a defense.
    clean = hit_rate(rows, "clean", args.primary_km)
    localisable = {q for q, v in clean.items() if v >= 0.5}
    print(f"[geo] {len(localisable)} of {len(clean)} queries localisable "
          f"within {args.primary_km:g} km")

    summary = {"primary_km": args.primary_km, "margin": args.margin,
               "n_queries": len(clean), "n_localisable": len(localisable),
               "thresholds": args.thresholds, "cells": []}
    body: List[str] = []
    for subset_name, keep in (("all queries", None),
                              ("localisable", localisable)):
        body.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{subset_name}}}"
                    rf"{'' if keep is None else f' ({len(keep)} queries)'}}} \\")
        ref = hit_rate(rows, "isotropic", args.primary_km)
        cells, raw_p = [], []
        for cond in conds:
            cur = hit_rate(rows, cond, args.primary_km)
            qs = sorted(set(cur) & set(ref) if keep is None
                        else (set(cur) & set(ref) & keep))
            if not qs:
                continue
            acc = float(np.mean([cur[q] for q in qs]))
            others = {t: float(np.mean(list(
                {q: v for q, v in hit_rate(rows, cond, t).items()
                 if q in qs}.values()))) for t in args.thresholds}
            if cond == "isotropic":
                cells.append({"condition": cond, "subset": subset_name,
                              "acc": acc, "by_threshold": others,
                              "delta": None})
                raw_p.append(1.0)
                continue
            d = np.array([cur[q] - ref[q] for q in qs])
            g = [place_of[q] for q in qs]
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if nz.size else 1.0
            lo, hi = cluster_ci(d, g, args.n_boot, 4242)
            pt = tost(d, g, args.margin, args.n_boot, 4242)
            raw_p.append(p)
            cells.append({"condition": cond, "subset": subset_name,
                          "acc": acc, "by_threshold": others,
                          "delta": float(d.mean()), "ci": [lo, hi],
                          "p": p, "p_tost": pt, "n": len(qs)})
        for c, ph in zip(cells, holm(raw_p)):
            c["p_holm"] = ph
            if c["delta"] is None:
                body.append(rf"\quad {LABEL[c['condition']]} & {c['acc']:.3f} & "
                            rf"--- & --- & --- & reference \\")
                continue
            sep = (c["ci"][1] < 0 or c["ci"][0] > 0) and ph < 0.05
            c["separates"] = bool(sep)
            verdict = ("separates" if sep else
                       ("negligible" if max(abs(c["ci"][0]), abs(c["ci"][1]))
                        <= args.margin else "none det."))
            body.append(
                rf"\quad {LABEL[c['condition']]} & {c['acc']:.3f} & "
                rf"${c['delta']:+.3f}$ & [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] & "
                rf"{fmt_p(ph)} & {verdict} \\")
        summary["cells"].extend(cells)

    caption = (
        rf"Both axes against a geolocator. GeoCLIP predicts a coordinate "
        rf"directly, so success is distance: each cell is the share of queries "
        rf"placed within {args.primary_km:g}\,km of the query's own position, "
        rf"and $\Delta$ is paired against the isotropic control at the same "
        rf"delivered distortion, three seeds averaged within a query before "
        rf"pairing, with place-clustered intervals and Holm correction within "
        rf"each block. GeoCLIP is never in the optimiser; the perturbations are "
        rf"the retrieval study's, rebuilt from the same seeds. The lower block "
        rf"restricts to queries the unperturbed attacker already localises, "
        rf"which is the only subset where a defense can be credited: on a "
        rf"query the attacker already fails, moving the prediction earns "
        rf"nothing and moving it closer is not harm.")
    lines = [
        "% Generated by src/scripts/make_geolocator_table.py from",
        "% src/exports/geolocator. Do not edit by hand.",
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:geolocator}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lccccc}", r"\hline",
        rf"Condition & within {args.primary_km:g}\,km & $\Delta$ & "
        r"95\% CI & $p$ & Verdict \\",
        r"\hline", *body, r"\hline", r"\end{tabular}", r"\end{table}",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dest = Path(args.summary)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[done] wrote {out}")
    for c in summary["cells"]:
        bt = " ".join(f"{t:g}km={v:.3f}" for t, v in c["by_threshold"].items())
        extra = ("" if c["delta"] is None else
                 f"  d={c['delta']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}]"
                 f" holm={c['p_holm']:.3f}"
                 f"{'  <-- SEPARATES' if c.get('separates') else ''}")
        print(f"  {c['subset']:12s} {c['condition']:10s} {bt}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
