"""What a purifying attacker recovers, and what it still cannot reach.

Reads the purification run and answers three questions in order.

1. Does purification help the attacker at all? Each release is scored with and
   without the purifier trained on it, paired on the same query.
2. Does the directional release still beat the operating-point control once
   both are purified? That is the comparison the paper's claim rests on; a
   purifier that lifts both equally leaves the claim standing.
3. Does the attacker have to know which release it faces? The purifier trained
   on the unhardened direction is applied to the hardened one.

The clean row is the same 200 queries with no perturbation at all, so a reader
can see how much of the gap a purifier closes rather than only whether it moves
anything. Intervals resample places, which is the unit this protocol
prescribes; the 200 evaluation queries occupy far fewer places than queries.
"""
from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pvalue import fmt_p  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
LABEL = {("clean", "none"): "clean (unperturbed)",
         ("isotropic", "none"): "isotropic control",
         ("isotropic", "isotropic"): "\\quad purified",
         ("direction", "none"): "direction",
         ("direction", "direction"): "\\quad purified",
         ("hardened", "none"): "hardened direction",
         ("hardened", "hardened"): "\\quad purified",
         ("hardened", "direction"): "\\quad purified, wrong release"}
ORDER = [("clean", "none"),
         ("isotropic", "none"), ("isotropic", "isotropic"),
         ("direction", "none"), ("direction", "direction"),
         ("hardened", "none"), ("hardened", "hardened"),
         ("hardened", "direction")]


def boot(diff: np.ndarray, ids: np.ndarray, n: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(ids, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    draws = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        draws[b] = diff[np.concatenate([groups[i] for i in pick])].mean()
    return tuple(np.percentile(draws, [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default=str(REPO / "src" / "outputs" /
                                          "purification" / "per_query.csv"))
    ap.add_argument("--export", default=str(REPO / "src" / "exports" /
                                            "purification" / "per_query.csv"))
    ap.add_argument("--out", default=str(REPO / "paper" / "generated" /
                                         "tab_purification.tex"))
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--attacker", default="",
                    help="Named in the caption and appended to the label; the "
                         "table is per attacker because the release is "
                         "optimised against that attacker's surrogates.")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(args.rows)):
        with open(path, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    if not rows:
        print(f"[FAIL] no rows matched {args.rows}")
        return 1
    cells: Dict[tuple, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    psnr: Dict[tuple, list] = defaultdict(list)
    place: Dict[str, str] = {}
    for r in rows:
        cells[(r["condition"], r["purifier"])][r["query_id"]].append(
            float(int(r["correct_rank"]) == 1))
        if r.get("psnr_to_clean"):
            psnr[(r["condition"], r["purifier"])].append(
                float(r["psnr_to_clean"]))
        place[r["query_id"]] = r["correct_place"]
    per = {k: {q: float(np.mean(v)) for q, v in d.items()}
           for k, d in cells.items()}

    ran = set(per)
    missing = sorted(ran - set(ORDER))
    if missing:
        print(f"[FAIL] run but not in the table: {missing}")
        return 1

    Path(args.export).parent.mkdir(parents=True, exist_ok=True)
    keep = ["query_id", "condition", "purifier", "draw", "correct_rank",
            "correct_place", "psnr_to_clean"]
    with open(args.export, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[export] {len(rows)} rows -> {args.export}")

    def contrast(a: tuple, b: tuple, seed: int):
        """Paired Top-1 difference a - b, on the place as the unit."""
        arm, ref = per[a], per[b]
        qs = sorted(set(arm) & set(ref))
        d = np.array([arm[q] - ref[q] for q in qs])
        lo, hi = boot(d, np.array([place[q] for q in qs]), args.n_boot, seed)
        try:
            pv = float(wilcoxon(d).pvalue)
        except ValueError:
            pv = 1.0
        return float(d.mean()), lo, hi, pv

    lines, stats = [], []
    for i, key in enumerate(ORDER):
        if key not in per:
            continue
        top1 = float(np.mean(list(per[key].values())))
        if key[1] == "none" or key[0] == "clean":
            lines.append(rf"{LABEL[key]:<28} & {top1:.4f} & --- & --- & --- \\")
            stats.append({"cell": key, "top1": top1})
            continue
        # Purified rows are paired against the same release unpurified: the
        # question is what the purifier bought, not what the release cost.
        delta, lo, hi, pv = contrast(key, (key[0], "none"), i)
        lines.append(rf"{LABEL[key]:<28} & {top1:.4f} & ${delta:+.4f}$ & "
                     rf"$[{lo:+.3f},{hi:+.3f}]$ & {fmt_p(pv)} \\")
        stats.append({"cell": key, "top1": top1, "delta": delta,
                      "ci": [lo, hi], "p": pv})

    n_q = len(per[("clean", "none")])
    n_places = len({place[q] for q in per[("clean", "none")]})
    head = [
        "% Generated by src/scripts/make_purification_table.py from",
        "% src/exports/purification. Do not edit by hand.",
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{An attacker that removes the perturbation. A residual",
        r"denoiser is trained on released/clean pairs from a place-disjoint",
        rf"split and applied to the release before {args.attacker or 'the'}",
        rf"attacker embeds it ({n_q} held-out queries over {n_places} places,",
        r"3 draws). $\Delta$ is the purified",
        r"row against the same release unpurified, so it is what the purifier",
        r"bought; the interval resamples places. The last row applies the",
        r"purifier trained on the unhardened direction to the hardened",
        r"release, which is the attacker that does not know what it faces.}",
        rf"\label{{tab:purification{'_' + args.attacker.lower() if args.attacker else ''}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        r"Condition & Top-1 $\downarrow$ & $\Delta$ & 95\% CI & $p$ \\",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head + lines + [r"\hline", r"\end{tabular}",
                                             r"\end{table}"]) + "\n",
                   encoding="utf-8")
    print(f"[table] {out}")
    # Whether the purifier is any good at its own job. A purifier that lowers
    # PSNR is damaging the frame rather than cleaning it, and a null from one
    # of those says nothing about purification.
    print("[psnr] distance to the clean frame, held-out queries:")
    for key in ORDER:
        if key in psnr and psnr[key]:
            print(f"[psnr]   {key[0]:10s}/{key[1]:10s} "
                  f"{float(np.mean(psnr[key])):6.2f} dB")
    for st in stats:
        extra = (f" delta {st['delta']:+.4f} "
                 f"[{st['ci'][0]:+.3f},{st['ci'][1]:+.3f}] p={st['p']:.2g}"
                 if "delta" in st else "")
        print(f"[done] {st['cell'][0]:10s}/{st['cell'][1]:10s} "
              f"top1 {st['top1']:.4f}{extra}")

    # The comparison the paper's claim rests on: does the direction still beat
    # the operating-point control when both are purified?
    for a, b, name in [
            (("direction", "direction"), ("isotropic", "isotropic"),
             "direction vs control, both purified"),
            (("hardened", "hardened"), ("isotropic", "isotropic"),
             "hardened vs control, both purified"),
            (("direction", "none"), ("isotropic", "none"),
             "direction vs control, neither purified")]:
        if a in per and b in per:
            d, lo, hi, pv = contrast(a, b, 99)
            print(f"[key ] {name}: {d:+.4f} [{lo:+.3f},{hi:+.3f}] p={pv:.2g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
