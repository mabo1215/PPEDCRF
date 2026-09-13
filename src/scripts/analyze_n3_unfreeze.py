"""Does unfreezing the whole encoder change what the adaptive attacker buys?

N3 asks for an adaptive attacker at a larger budget than the published arm,
which fine-tuned only the last residual block. This reads the two budgets run
side by side under one protocol: k=1 (layer4 only, the published setting) and
k=5 (every residual stage plus the stem, the only fully-unfrozen setting).

Running both here rather than quoting the published k=1 numbers is the point.
Those came from a different run with its own split and seed, and a control
that has to be compared across runs is not a control.

Each adapted model is scored on the held-out split its own checkpoint was
trained against, and differenced against an unadapted baseline on that same
split -- never against a baseline from another split, which would confound
adaptation with which queries happened to land in the test set.

Reported per exposure and gallery mode: Top-1 under the isotropic release and
under the three-surrogate direction release, the change from the unadapted
baseline, and the k=5 minus k=1 difference that answers N3 directly.
"""
from __future__ import annotations

import argparse
import collections
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "scripts"))
from make_msls_placement_table import boot  # noqa: E402

CONDS = ["isotropic", "transfer_3"]


def per_query(path: Path, condition: str) -> Dict[str, float]:
    acc = collections.defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["condition"] == condition:
                acc[r["query_id"]].append(1.0 if int(r["correct_rank"]) == 1 else 0.0)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def parse_name(stem: str) -> Optional[dict]:
    m = re.match(r"n3_k(\d)_(?:s(\d+)_)?(isotropic|direction|hardened_direction)_(stock|rebuilt)$",
                 stem)
    if not m:
        return None
    return {"k": int(m.group(1)), "seed": m.group(2) or "1234",
            "exposure": m.group(3), "mode": m.group(4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(REPO / "src/outputs/n3_eval"))
    ap.add_argument("--places",
                    default=str(REPO / "src/exports/tifs_d6/d6_r18_plain.csv"))
    args = ap.parse_args()

    root = Path(args.root)
    place_of = {}
    if Path(args.places).is_file():
        with open(args.places, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                place_of[r["query_id"]] = r["correct_place"]

    # Baselines, keyed by seed. The 1234 baseline was written under the name
    # the launcher derived from its ids file rather than the seed.
    base: Dict[str, Path] = {}
    for p in root.glob("baseline_*.csv"):
        base["1235" if "s1235" in p.stem else "1234"] = p
    if not base:
        raise SystemExit(f"no baseline_*.csv under {root}")

    runs = []
    for p in sorted(root.glob("n3_*.csv")):
        meta = parse_name(p.stem)
        if meta:
            meta["path"] = p
            runs.append(meta)
    if not runs:
        raise SystemExit(f"no n3_*.csv under {root}")
    print(f"{len(runs)} adapted runs, {len(base)} baselines\n")

    rows = {}
    for cond in CONDS:
        print(f"=== release: {cond} ===")
        print(f"{'exposure':22s} {'mode':8s} {'seed':5s} "
              f"{'k=1':>7s} {'k=5':>7s} {'k5-k1':>8s} {'base':>7s}")
        for exposure in ["isotropic", "direction", "hardened_direction"]:
            for mode in ["stock", "rebuilt"]:
                for seed in sorted({r["seed"] for r in runs}):
                    got = {}
                    for k in (1, 5):
                        hit = [r for r in runs if r["k"] == k and r["seed"] == seed
                               and r["exposure"] == exposure and r["mode"] == mode]
                        if hit:
                            got[k] = per_query(hit[0]["path"], cond)
                    if len(got) != 2 or seed not in base:
                        continue
                    b = per_query(base[seed], cond)
                    qs = sorted(set(got[1]) & set(got[5]) & set(b))
                    if not qs:
                        continue
                    t1 = float(np.mean([got[1][q] for q in qs]))
                    t5 = float(np.mean([got[5][q] for q in qs]))
                    bb = float(np.mean([b[q] for q in qs]))
                    d = np.array([got[5][q] - got[1][q] for q in qs])
                    ci = boot(d, [place_of.get(q, q) for q in qs])
                    flag = "" if ci[0] * ci[1] <= 0 else "  <-- differs"
                    print(f"{exposure:22s} {mode:8s} {seed:5s} "
                          f"{t1:7.4f} {t5:7.4f} {d.mean():+8.4f} {bb:7.4f}"
                          f"  [{ci[0]:+.3f},{ci[1]:+.3f}]{flag}")
                    rows[(cond, exposure, mode, seed)] = (t1, t5, bb, d, ci)
        print()

    # The N3 answer in one line: pooled over every configuration, does the
    # fully-unfrozen budget beat the published one?
    print("=== pooled: k=5 minus k=1 over all configurations ===")
    for cond in CONDS:
        ds = [r[3] for key, r in rows.items() if key[0] == cond]
        if not ds:
            continue
        allq = np.concatenate(ds)
        print(f"  {cond:12s} mean {allq.mean():+.4f} over {len(ds)} configs, "
              f"{allq.size} paired queries")
    print("\nPositive means the fully-unfrozen attacker did better than the "
          "published last-block one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
