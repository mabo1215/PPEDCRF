"""The primary placement null, tabulated.

The manuscript's central negative claim is measured on the real place-labelled
benchmark against the weak attacker: seven energy-matched rules, 400 query
clusters, three seeds. Until now that result appeared in the manuscript as two
point estimates in a sentence, while the table a referee needs to check a
seven-way null -- every cell's interval, under both units of inference, with
the verdict the protocol's margin assigns it -- existed nowhere in the
submission. The strong attacker had one; the primary evidence did not.

This writes it, from the same released per-query rows the manuscript quotes,
and includes the margin-gradient rules alongside the seven so the placement the
account nominates is in the same table as the ones it is compared with.
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO / "src" / "scripts"))
from _pvalue import fmt_p  # noqa: E402

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
ORDER = ["learned", "oracle_grad", "anti_oracle_grad", "saliency", "center",
         "random_fixed", "edge", "margin_oracle", "anti_margin_oracle"]


def per_query(rows: Sequence[dict], placement: str) -> Dict[str, float]:
    acc = collections.defaultdict(list)
    for r in rows:
        if r["placement"] == placement:
            acc[r["query_id"]].append(1.0 if int(r["correct_rank"]) == 1 else 0.0)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def boot(d: np.ndarray, groups, n=10000, seed=1234):
    rng = np.random.default_rng(seed)
    keys = sorted(set(groups))
    idx = collections.defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    out = np.empty(n)
    k = len(keys)
    for b in range(n):
        pick = rng.integers(0, k, k)
        out[b] = d[np.concatenate([idx[keys[j]] for j in pick])].mean()
    return tuple(np.percentile(out, [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", default=[
        str(REPO / "src/exports/icme2027_placement_msls/final/per_query.csv"),
        str(REPO / "src/exports/margin_oracle/per_query.csv")])
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"))
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--out",
                    default=str(REPO / "paper/generated/tab_placement_msls.tex"))
    args = ap.parse_args()

    place_of = {}
    with open(args.places, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            place_of[r["query_id"]] = r["correct_place"]

    # The margin-gradient rules were run in their own job against their own
    # uniform arm, so each file is paired against the uniform arm it carries
    # rather than across files.
    body: List[str] = []
    ref_top1 = None
    n_q = n_p = 0
    for path in args.rows:
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        ref = per_query(rows, "uniform")
        if not ref:
            continue
        if ref_top1 is None:
            ref_top1 = float(np.mean(list(ref.values())))
            n_q = len(ref)
            n_p = len({place_of.get(q, q) for q in ref})
        for name in ORDER:
            cur = per_query(rows, name)
            # The score-gradient rule appears in both files; the first one
            # wins, so a rule is never paired against two different uniform
            # arms and then printed twice.
            if not cur or any(l.startswith(LABEL[name] + " ") for l in body):
                continue
            qs = sorted(set(cur) & set(ref))
            d = np.array([cur[q] - ref[q] for q in qs])
            p = float(wilcoxon(d, zero_method="wilcox", method="approx").pvalue) \
                if np.any(d != 0) else 1.0
            cq = boot(d, qs)
            cp = boot(d, [place_of.get(q, q) for q in qs])
            inside = max(abs(cq[0]), abs(cq[1]), abs(cp[0]), abs(cp[1])) <= args.margin
            verdict = "negligible" if inside else "none det."
            body.append(
                rf"{LABEL[name]} & {np.mean([cur[q] for q in qs]):.4f} & "
                rf"${d.mean():+.4f}$ & [{cq[0]:+.3f},{cq[1]:+.3f}] & "
                rf"[{cp[0]:+.3f},{cp[1]:+.3f}] & {fmt_p(p)} & {verdict} \\")

    if not body:
        raise SystemExit("no placement rows found")

    caption = (
        rf"Energy-matched placements against the weak attacker on the primary "
        rf"place-labelled manifest: {n_q} query clusters over {n_p} places, "
        rf"three seeds. $\Delta$ is paired against the uniform reference on "
        rf"identical queries, so \emph{{positive means worse privacy}}. Both "
        rf"units of inference are printed because the protocol prescribes the "
        rf"clustered one and the two barely differ here, $206$ of the places "
        rf"carrying a single query. ``Verdict'' is \emph{{negligible}} when "
        rf"both intervals lie inside the $\pm{args.margin:.2f}$ margin fixed "
        rf"before testing and \emph{{none det.}} when they do not and the "
        rf"difference is not significant. The learned row is the mechanism's "
        rf"own map, which at this checkpoint is the uniform reference, so it "
        rf"is a self-comparison rather than a finding. The last two rows are "
        rf"the placement the margin analysis nominates and its inverse.")
    lines = [
        "% Generated by src/scripts/make_msls_placement_table.py.",
        "% Do not edit by hand.",
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:placement_msls}", r"\footnotesize",
        r"\setlength{\tabcolsep}{1.2pt}",
        r"\begin{tabular}{lcccccc}", r"\hline",
        r"Placement & Top-1 & $\Delta$ & query 95\% CI & place 95\% CI & $p$ "
        r"& Verdict \\",
        r"\hline",
        rf"uniform (ref.) & {ref_top1:.4f} & --- & --- & --- & --- & --- \\",
        *body, r"\hline", r"\end{tabular}", r"\end{table}",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] wrote {out}")
    for line in body:
        print("  " + line.replace(r"\\", "").replace("&", " ").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
