"""The VLM geolocator arm, read the way the GeoCLIP arm is read.

N2 asks whether the direction result survives a stronger image-to-GPS
attacker. The model the review names, PIGEON, has no public weights, so the
substitute is a frontier vision-language model prompted for a coordinate --
a different architecture, different training data, and no gallery at all,
which also tests whether the direction result is a fact about the
perturbation or a fact about CLIP.

Two conventions are inherited deliberately from the GeoCLIP arm so the two
are comparable rather than merely adjacent.

First, the claim is carried by the localisable subset: the queries the clean
attacker already places inside the primary threshold. On a query the clean
attacker already fails, a perturbation that moves the prediction earns credit
it never earned, and one that happens to move it closer looks like harm.
Neither is evidence about a defense. The all-query column is printed beside
it, not instead of it.

Second, the capability gate is reported whether or not it passes. A
geolocator weaker than GeoCLIP cannot answer "does this survive a stronger
attacker", so an arm that quietly skipped the gate could report a reassuring
null that means only that the substitute was bad.

Refusals are counted, never scored. A refusal turned into a coordinate would
land far from the truth and read as a defense working spectacularly.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "scripts"))
from _pvalue import fmt_p  # noqa: E402
from make_msls_placement_table import boot  # noqa: E402

THRESH = [(1.0, "1 km"), (25.0, "25 km"), (200.0, "200 km")]
PRIMARY = 25.0
ORDER = [("clean", "No perturbation"),
         ("isotropic", "Isotropic, matched MSE"),
         ("direction", "Direction, matched MSE")]


def load(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def errs(rows: Sequence[dict], cond: str) -> Dict[str, float]:
    """Per-query great-circle error, refusals excluded rather than imputed."""
    out = {}
    for r in rows:
        if r["condition"] != cond:
            continue
        if str(r.get("refused", "0")) == "1" or not r.get("error_km"):
            continue
        out[r["query_id"]] = float(r["error_km"])
    return out


def share(d: Dict[str, float], km: float, keys=None) -> float:
    ks = list(keys) if keys is not None else list(d)
    ks = [k for k in ks if k in d]
    return float(np.mean([1.0 if d[k] <= km else 0.0 for k in ks])) if ks else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vlm", required=True, nargs="+")
    ap.add_argument("--geoclip", required=True)
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"))
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--out",
                    default=str(REPO / "paper/generated/tab_vlm_geolocator.tex"))
    args = ap.parse_args()

    vrows: List[dict] = []
    for p in args.vlm:
        vrows += load(Path(p))
    grows = load(Path(args.geoclip))

    place_of = {}
    if Path(args.places).is_file():
        for r in load(Path(args.places)):
            place_of[r["query_id"]] = r["correct_place"]

    v = {c: errs(vrows, c) for c, _ in ORDER}
    g = {c: errs(grows, c) for c, _ in ORDER}
    refused = sum(1 for r in vrows if str(r.get("refused", "0")) == "1")

    # The gate, on clean frames, against the attacker it must beat to matter.
    gate_v = share(v["clean"], PRIMARY)
    gate_g = share(g["clean"], PRIMARY)
    med_v = float(np.median(list(v["clean"].values()))) if v["clean"] else float("nan")
    med_g = float(np.median(list(g["clean"].values()))) if g["clean"] else float("nan")
    passed = gate_v > gate_g

    print(f"gate: VLM within {PRIMARY:g} km {gate_v:.3f} (median {med_v:.1f} km) "
          f"vs GeoCLIP {gate_g:.3f} (median {med_g:.1f} km) -> "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"refusals: {refused}")

    # Localisable subset: what the clean VLM already places inside PRIMARY.
    loc = {q for q, d in v["clean"].items() if d <= PRIMARY}
    print(f"localisable subset: {len(loc)} of {len(v['clean'])} queries\n")

    lines = []
    for cond, label in ORDER:
        cur = v[cond]
        cells = [f"{share(cur, km):.3f}" for km, _ in THRESH]
        loc_share = share(cur, PRIMARY, loc)
        if cond == "clean":
            lines.append(f"{label} & " + " & ".join(cells)
                         + f" & {loc_share:.3f} & \\multicolumn{{3}}{{c}}{{reference}} \\\\")
            continue
        qs = sorted(set(cur) & set(v["clean"]) & loc)
        d = np.array([(1.0 if cur[q] <= PRIMARY else 0.0)
                      - (1.0 if v["clean"][q] <= PRIMARY else 0.0) for q in qs])
        nz = d[d != 0]
        p = float(wilcoxon(nz).pvalue) if nz.size else 1.0
        ci = boot(d, [place_of.get(q, q) for q in qs])
        lines.append(f"{label} & " + " & ".join(cells)
                     + f" & {loc_share:.3f} & ${d.mean():+.3f}$ & "
                     f"[{ci[0]:+.3f},{ci[1]:+.3f}] & {fmt_p(p)}\\,({nz.size}) \\\\")
        print(f"  {label:24s} localisable {loc_share:.3f}  "
              f"delta {d.mean():+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] p={fmt_p(p)} ({nz.size})")

    gate_sentence = (
        f"The substitute clears the gate: on clean frames it places "
        f"{gate_v*100:.1f}\\% of queries inside {PRIMARY:g}~km at a median error of "
        f"{med_v:.1f}~km, against {gate_g*100:.1f}\\% and {med_g:.1f}~km for the "
        f"gallery-based geolocator on the identical released bytes."
        if passed else
        f"The substitute does \\emph{{not}} clear the gate: on clean frames it "
        f"places {gate_v*100:.1f}\\% of queries inside {PRIMARY:g}~km at a median of "
        f"{med_v:.1f}~km, against {gate_g*100:.1f}\\% and {med_g:.1f}~km for the "
        f"gallery-based geolocator. This arm is therefore a bound on what this "
        f"substitute recovers, not a test against a stronger attacker.")

    caption = (
        r"A geolocator that is not a gallery retriever: a vision--language "
        r"model prompted for a coordinate --- different architecture, "
        r"different training data, no gallery --- read on the same released "
        r"bytes the gallery-based geolocator read, so the two differ in the "
        r"reader and nothing else. " + gate_sentence +
        r" Columns give the share of queries inside each threshold. The claim "
        r"is carried by the ``localisable'' column, the queries the clean "
        r"attacker already places inside " + f"{PRIMARY:g}~km" +
        r": where it already fails, a perturbation that moves the prediction "
        r"earns credit it never earned. $\Delta$ is against that subset's "
        r"clean rate, interval bootstrapped over places, the figure beside "
        f"$p$ counting discordant pairs. Refusals ({refused}) are counted and "
        r"excluded, never scored as a coordinate. Attacker: "
        f"\\texttt{{{args.model}}}.")

    out = [
        "% Generated by src/scripts/make_vlm_geolocator_table.py. Do not edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{tab:vlm_geolocator}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Release & 1 km & 25 km & 200 km & localisable & $\Delta$ & "
        r"place 95\% CI & $p$ \\",
        r"\hline",
        *lines,
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    Path(args.out).write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
