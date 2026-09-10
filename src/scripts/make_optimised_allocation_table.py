"""Table for the optimised-allocation arm.

The arm exists because a referee can read the paper's central contrast as
optimised-versus-unoptimised rather than direction-versus-allocation: every
placement the audit tests is prescribed, while the direction is the output of
twenty gradient steps. So the placement map is solved for instead, against the
same objective, the same surrogates and the same delivered-distortion gate.

The table has to carry four things at once, and the column order follows them:

  Top-1 and Delta      does an optimised placement beat uniform at all
  Delta on a fresh draw whether what it found is a spatial preference or a
                       selection of signs -- a map optimised against the draw
                       it will be released with can pick the signs it likes,
                       and that is the direction axis, not this one
  objective            whether the optimiser moved its own loss, so that a
                       null in Top-1 is about what allocation can buy rather
                       than about a search that never got going
  top-decile share     how concentrated the solved map is, on the scale the
                       manuscript already reports for every prescribed rule

Generated from the released per-query rows. Do not edit the output by hand.
"""
from __future__ import annotations

import argparse
import csv
import collections
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "src" / "scripts"))
from _pvalue import fmt_p  # noqa: E402

LABEL = {
    "opt_transfer": r"optimised, surrogates",
    "opt_whitebox": r"optimised, attacker",
    "opt_transfer_x2": r"\quad at twice the budget",
    "opt_whitebox_x2": r"\quad at twice the budget",
}


def load(paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        if not p.is_file():
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def per_query(rows, condition: str) -> Dict[str, float]:
    acc = collections.defaultdict(list)
    for r in rows:
        if r["condition"] == condition:
            acc[r["query_id"]].append(1.0 if int(r["correct_rank"]) == 1 else 0.0)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def boot(diffs: np.ndarray, groups, n=10000, seed=1234):
    rng = np.random.default_rng(seed)
    keys = sorted(set(groups))
    idx = collections.defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    means = np.empty(n)
    k = len(keys)
    for b in range(n):
        pick = rng.integers(0, k, k)
        means[b] = diffs[np.concatenate([idx[keys[j]] for j in pick])].mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def stats(rows, condition: str, reference: str, place_of):
    cur, ref = per_query(rows, condition), per_query(rows, reference)
    qs = sorted(set(cur) & set(ref))
    if not qs:
        return None
    d = np.array([cur[q] - ref[q] for q in qs])
    p = float(wilcoxon(d, zero_method="wilcox", method="approx").pvalue) \
        if np.any(d != 0) else 1.0
    sub = [r for r in rows if r["condition"] == condition]
    return {
        "n": len(qs),
        "top1": float(np.mean([cur[q] for q in qs])),
        "delta": float(d.mean()),
        "ci": boot(d, [place_of.get(q, q) for q in qs]),
        "p": p,
        "share": float(np.mean([float(r["weight_top10pct_share"]) for r in sub])),
        "obj0": float(np.mean([float(r["surrogate_sim_start"]) for r in sub])),
        "obj1": float(np.mean([float(r["surrogate_sim_end"]) for r in sub])),
    }


def block(rows, place_of, title: str, conditions: Sequence[str],
          out: List[str]) -> None:
    ref = per_query(rows, "uniform")
    if not ref:
        return
    out.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}} \\")
    out.append(rf"uniform (reference) & {np.mean(list(ref.values())):.4f} & --- "
               rf"& --- & --- & --- & 0.100 \\")
    for cond in conditions:
        st = stats(rows, cond, "uniform", place_of)
        if st is None:
            continue
        cross = stats(rows, cond + "_crossdraw", "uniform", place_of)
        cs = f"${cross['delta']:+.4f}$" if cross else "---"
        out.append(
            rf"{LABEL.get(cond, cond)} & {st['top1']:.4f} & ${st['delta']:+.4f}$ "
            rf"& [{st['ci'][0]:+.3f},{st['ci'][1]:+.3f}] & {fmt_p(st['p'])} "
            rf"& {cs} & {st['share']:.3f} \\")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows_dir", default="src/exports/optimised_allocation")
    ap.add_argument("--places", default="src/exports/tifs_d6/d6_r18_plain.csv")
    ap.add_argument("--out", default="paper/generated/tab_optimised_allocation.tex")
    ap.add_argument("--mode", default="expectation",
                    choices=("expectation", "realised"))
    args = ap.parse_args()

    place_of = {}
    with open(REPO / args.places, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            place_of[r["query_id"]] = r["correct_place"]

    root = REPO / args.rows_dir
    tag = "exp" if args.mode == "expectation" else "real"
    body: List[str] = []
    for attacker, title in (("r18", "Weak attacker: ResNet18"),
                            ("mix", "Strong attacker: MixVPR")):
        rows = load(sorted(root.glob(f"r1_{attacker}_{tag}*.csv")))
        if not rows:
            print(f"[warn] no rows for {attacker}/{args.mode}")
            continue
        block(rows, place_of, title, ["opt_transfer", "opt_whitebox"], body)
    if not body:
        raise SystemExit("no rows found; nothing written")

    caption = (
        r"Solving for the placement map instead of prescribing one. The map is "
        r"optimised against the same objective, the same surrogate ensemble and "
        r"the same delivered distortion as the directional arm, under the "
        r"study's own energy gate, and for the same number of steps. "
        r"``optimised, surrogates'' never sees the attacker; ``optimised, "
        r"attacker'' does, and is an upper bound no deployable mechanism has. "
        r"$\Delta$ is paired against the uniform control on identical queries "
        r"and noise draws, with a place-clustered interval. The fifth column "
        r"scores the same map on a noise draw the optimiser never saw: a "
        r"spatial preference survives it, a selection of signs does not. "
        r"``top-decile'' is the squared weight in the largest tenth of the "
        r"map, $0.100$ for uniform and $0.839$ for the edge rule.")
    lines = [
        "% Generated by src/scripts/make_optimised_allocation_table.py.",
        "% Do not edit by hand.",
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:optimised_allocation}}",
        r"\footnotesize", r"\setlength{\tabcolsep}{1.6pt}",
        r"\begin{tabular}{lcccccc}", r"\hline",
        r"Condition & Top-1 $\downarrow$ & $\Delta$ & 95\% CI & $p$ "
        r"& $\Delta$ fresh & top-dec. \\",
        r"\hline",
        *body,
        r"\hline", r"\end{tabular}", r"\end{table}",
    ]
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
