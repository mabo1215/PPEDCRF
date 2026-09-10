"""Turn the D6 exports into the tables the manuscript needs.

The D6 runs regenerate, at full scale and on one consistent dataset, both of
the paper's direction tables:

  transfer      isotropic / transfer_k / white_box under no preprocessing,
                which is the manuscript's Table II
  preprocessing every attacker-side transform, unhardened against
                EOT-hardened, which is Fig. 3 and the supplement's full table,
                extended with eight transforms the hardening never saw

Both are reported with the query as the unit of inference (review R1): each
query is collapsed to its seed-averaged hit rate, the interval is a
query-cluster bootstrap, and the test is a Wilcoxon signed-rank over the
per-query differences. The exact-McNemar column over (query, seed) pairs is
kept alongside it so the change from the published numbers is visible rather
than silent.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

CONTROL = "isotropic"
TRAINED = ("jpeg75", "jpeg50", "blur", "denoise")
HELD_OUT = ("jpeg60", "jpeg30", "median3", "resize_half", "blur2",
            "bitdepth4", "random_one", "jpeg50_blur")


def load(paths):
    """rows keyed by (sanitizer, condition, query, seed) -> hit.

    A run still in progress has a final line that is half written, so rows
    missing a field are skipped -- and counted, because losing more than the
    last row of a file means a truncation worth looking into.
    """
    hits, mses = {}, defaultdict(list)
    skipped = 0
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    key = (r["sanitizer"], r["condition"], r["query_id"],
                           r["seed"])
                    hit = int(r["correct_rank"]) == 1
                    mse = float(r["effective_mse"])
                except (TypeError, ValueError, KeyError):
                    skipped += 1
                    continue
                hits[key] = hit
                mses[r["sanitizer"]].append(mse)
    if skipped:
        print("[warn] skipped %d incomplete row(s) across %d file(s)"
              % (skipped, len(paths)))
    return hits, mses


def compare(hits, sanitizer, condition, n_boot=10000, seed=0):
    """Query-level and pair-level comparison against the isotropic control."""
    per_query = defaultdict(lambda: [[], []])   # query -> [control, condition]
    pairs = []
    for (san, cond, q, sd), hit in hits.items():
        if san != sanitizer:
            continue
        if cond == CONTROL:
            per_query[q][0].append(hit)
        elif cond == condition:
            per_query[q][1].append(hit)
    for (san, cond, q, sd), hit in hits.items():
        if san == sanitizer and cond == condition:
            ctrl = hits.get((san, CONTROL, q, sd))
            if ctrl is not None:
                pairs.append((ctrl, hit))
    qs = [q for q, (a, b) in per_query.items() if a and b]
    if not qs:
        return None
    ctrl = np.array([np.mean(per_query[q][0]) for q in qs])
    cond = np.array([np.mean(per_query[q][1]) for q in qs])
    diff = cond - ctrl

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qs), size=(n_boot, len(qs)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    nz = diff[diff != 0]
    wp = float(wilcoxon(nz).pvalue) if len(nz) else 1.0

    b01 = sum(1 for a, b in pairs if a and not b)
    b10 = sum(1 for a, b in pairs if b and not a)
    mp = float(binomtest(b10, b01 + b10, 0.5).pvalue) if (b01 + b10) else 1.0
    return {
        "n_queries": len(qs), "n_pairs": len(pairs),
        "control": float(ctrl.mean()), "top1": float(cond.mean()),
        "delta": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
        "wilcoxon_p": wp, "mcnemar_p": mp,
    }


def report(title, hits, sanitizers, condition, into=None):
    print("\n== %s ==" % title)
    print("%-12s %6s %6s %8s %8s %8s %18s %10s %10s"
          % ("transform", "nq", "npair", "iso", "top1", "delta",
             "95% CI (query)", "wilcox p", "McNemar p"))
    for san in sanitizers:
        st = compare(hits, san, condition)
        if st is None:
            print("%-12s  --" % san)
            continue
        print("%-12s %6d %6d %8.4f %8.4f %+8.4f  [%+7.4f,%+7.4f] %10.2e %10.2e"
              % (san, st["n_queries"], st["n_pairs"], st["control"],
                 st["top1"], st["delta"], st["ci_low"], st["ci_high"],
                 st["wilcoxon_p"], st["mcnemar_p"]))
        if into is not None:
            into[san] = st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="src/outputs/tifs_d6")
    ap.add_argument("--backbone",
                    choices=("resnet18", "mixvpr", "patchnetvlad", "vit_b_16"),
                    required=True)
    ap.add_argument("--tag", default="",
                    help="filename stem the exports use, if it is not this "
                         "backbone's default. The two attackers added in the "
                         "eleventh cycle were run under their own job names "
                         "rather than the d6 naming, and re-exporting them "
                         "under a second name would put two copies of the "
                         "same rows in the repository.")
    ap.add_argument("--deployable", default="",
                    help="the transfer condition to report as the deployable "
                         "one; defaults to the surrogate count each attacker "
                         "was actually run with.")
    ap.add_argument("--json", default="",
                    help="also write every cell to this path, so the figure "
                         "and the LaTeX tables are generated from the same "
                         "numbers the text quotes")
    args = ap.parse_args()
    collected = {"backbone": args.backbone, "transfer": {},
                 "unhardened": {}, "hardened": {}, "white_box": {}}

    # MixVPR is the one attacker run with four surrogates; the rest have
    # three, because ResNet18 is a surrogate for MixVPR and an attacker
    # everywhere else.
    default_tag = {"resnet18": "r18", "mixvpr": "mix",
                   "patchnetvlad": "pnv", "vit_b_16": "vit"}[args.backbone]
    tag = args.tag or default_tag
    deployable = args.deployable or (
        "transfer_4" if args.backbone == "mixvpr" else "transfer_3")
    def find(kind: str):
        # Accept both the d6 naming and the eleventh cycle's job naming, so
        # neither set of exports has to be duplicated under the other's name.
        pats = ["d6_%s_%s*.csv" % (tag, kind), "r5_pre_%s_%s*.csv" % (tag, kind)]
        out = []
        for pat in pats:
            out.extend(glob.glob(os.path.join(args.out_dir, pat)))
        return sorted(set(out))
    plain = find("plain")
    eot = find("eot")
    abl = find("ablation")
    print("%s: %d plain, %d eot, %d ablation file(s)"
          % (args.backbone, len(plain), len(eot), len(abl)))

    hits_plain, mse_plain = load(plain)
    hits_eot, _ = load(eot)

    # Energy gate: every condition must have been released at the same MSE.
    for name, mses in (("plain", mse_plain),):
        allv = [v for vals in mses.values() for v in vals]
        if allv:
            print("[gate] delivered MSE in [%.4f, %.4f] spread=%.2e %s"
                  % (min(allv), max(allv), max(allv) - min(allv),
                     "OK" if max(allv) - min(allv) <= 1e-3 else "FAIL"))

    # Table II: the transfer ladder under no preprocessing.
    hits_all = dict(hits_plain)
    hits_all.update(load(abl)[0])
    conds = sorted({c for (_s, c, _q, _sd) in hits_all if c != CONTROL},
                   key=lambda c: (c != "white_box", c))
    print("\n== transfer ladder, no preprocessing (manuscript Table II) ==")
    print("%-12s %6s %8s %8s %8s %18s %10s %10s"
          % ("condition", "nq", "iso", "top1", "delta", "95% CI (query)",
             "wilcox p", "McNemar p"))
    for cond in conds:
        st = compare(hits_all, "none", cond)
        if st is None:
            continue
        collected["transfer"][cond] = st
        print("%-12s %6d %8.4f %8.4f %+8.4f  [%+7.4f,%+7.4f] %10.2e %10.2e"
              % (cond, st["n_queries"], st["control"], st["top1"],
                 st["delta"], st["ci_low"], st["ci_high"],
                 st["wilcoxon_p"], st["mcnemar_p"]))

    report("unhardened, trained transforms", hits_plain,
           ("none",) + TRAINED, deployable, collected["unhardened"])
    report("unhardened, HELD-OUT transforms", hits_plain, HELD_OUT, deployable,
           collected["unhardened"])
    report("EOT-hardened, trained transforms", hits_eot,
           ("none",) + TRAINED, deployable, collected["hardened"])
    report("EOT-hardened, HELD-OUT transforms", hits_eot, HELD_OUT, deployable,
           collected["hardened"])

    # White-box bound under each transform, both objectives.
    print("\n== white-box bound under each transform ==")
    print("%-12s %10s %10s" % ("transform", "unhardened", "hardened"))
    for san in ("none",) + TRAINED + HELD_OUT:
        wb = [h for (s, c, _q, _sd), h in hits_plain.items()
              if s == san and c == "white_box"]
        wbe = [h for (s, c, _q, _sd), h in hits_eot.items()
               if s == san and c == "white_box"]
        if wb or wbe:
            collected["white_box"][san] = {
                "unhardened": float(np.mean(wb)) if wb else None,
                "hardened": float(np.mean(wbe)) if wbe else None}
            print("%-12s %10s %10s"
                  % (san,
                     "%.4f" % np.mean(wb) if wb else "--",
                     "%.4f" % np.mean(wbe) if wbe else "--"))


    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(collected, fh, indent=1, sort_keys=True)
        print("\n[json] %s" % args.json)


if __name__ == "__main__":
    main()
