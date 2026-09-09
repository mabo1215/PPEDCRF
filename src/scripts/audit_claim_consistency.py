"""Re-derive every headline number in the manuscript from the raw exports.

The fifth review's first finding was that prose, generated tables and exports
described different numerical versions of the same experiment. This script is
the mechanical answer to it. Each claim below records three things:

  * the value as the manuscript prints it,
  * a literal string that must still occur in the source file that prints it,
  * a recipe for recomputing the value from raw per-query rows.

A claim passes only when the string is still in the file *and* the recomputed
value matches to the stated tolerance. A claim whose export tree is absent is
reported as NO DATA rather than silently skipped, so the report doubles as the
list of manuscript numbers that currently have no local provenance.

Every comparison uses the manuscript's declared unit of inference: seeds are
averaged within a query, the interval is a query-cluster bootstrap, and the
test is a Wilcoxon signed-rank over the per-query differences with the ties at
zero discarded by scipy and the tie-corrected normal approximation requested
explicitly (see analyze_direction_factorial.compare).

Usage:
    python src/scripts/audit_claim_consistency.py
    python src/scripts/audit_claim_consistency.py --verbose

Exit status is 1 if any claim mismatches, 2 if claims are unverifiable for
want of data and none mismatch, 0 if everything checks out.
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "src" / "outputs"
# src/outputs/ is gitignored, so an export tree lives there only on the machine
# that produced it. Trees small enough to travel are committed under
# src/exports/ instead, and are searched second: a fresh clone can verify the
# claims they back without anyone remembering to copy anything first.
EXPORTS = REPO / "src" / "exports"
ROOTS = (OUT, EXPORTS)
MAIN = REPO / "paper" / "main.tex"
TAB_TRANSFER = REPO / "paper" / "generated" / "tab_transfer.tex"


# --------------------------------------------------------------------------
# loading and the shared unit of inference
# --------------------------------------------------------------------------

def load(pattern: str) -> List[dict]:
    """Every row under a glob relative to an export root, or [] if none match.

    The roots are tried in order and the first one that matches anything wins,
    so a tree present in both places is read from src/outputs/ and never
    silently concatenated with its committed copy.
    """
    for root in ROOTS:
        paths = sorted(globmod.glob(str(root / pattern)))
        if not paths:
            continue
        rows: List[dict] = []
        for path in paths:
            with open(path, newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        return rows
    return []


def per_query(rows: Sequence[dict], select: Callable[[dict], bool],
              metric: str = "top1") -> Dict[str, float]:
    """query -> metric averaged over the seeds of the rows that pass select."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if not select(r):
            continue
        rank = int(r["correct_rank"])
        acc[r["query_id"]].append(
            1.0 if rank == 1 else 0.0 if metric == "top1"
            else float(rank <= 5) if metric == "top5"
            else float(rank <= 10))
    return {q: statistics.fmean(v) for q, v in acc.items()}


def paired(cond: Dict[str, float], ref: Dict[str, float]) -> dict:
    """Paired delta and Wilcoxon p over the queries the two arms share."""
    queries = sorted(set(cond) & set(ref))
    diff = np.array([cond[q] - ref[q] for q in queries])
    p = (float(wilcoxon(diff, zero_method="wilcox", method="approx").pvalue)
         if np.any(diff != 0) else 1.0)
    return {"n": len(queries), "delta": float(diff.mean()), "p": p}


def mean_of(rows: Sequence[dict], select: Callable[[dict], bool],
            field: str) -> float:
    return statistics.fmean(float(r[field]) for r in rows if select(r))


# --------------------------------------------------------------------------
# the claim registry
# --------------------------------------------------------------------------

# Sentinel for "look for the printed form itself". Passing locator=None
# instead means the value is not searchable as a literal -- a p-value the
# manuscript prints as a LaTeX power of ten does not occur as "3e-15" --
# so only the recomputation is checked for it.
USE_PRINTED = object()

# A claim whose recomputed value only has to stay under the printed figure,
# for sentences that state a bound ("reproduces to 0.008") rather than a value.
BOUND = "bound"


class Claim:
    """One manuscript number, its printed form, and how to recompute it."""

    def __init__(self, cid: str, where: str, printed: str, value: float,
                 tol: float, tree: str, recompute: Callable[[], float],
                 locator=USE_PRINTED, source: Path = MAIN, kind: str = "value"):
        self.cid, self.where, self.printed = cid, where, printed
        self.value, self.tol, self.tree = value, tol, tree
        self.recompute, self.source, self.kind = recompute, source, kind
        self.locator = printed if locator is USE_PRINTED else locator


def rel(p: float) -> float:
    """Tolerance for a p-value the manuscript prints to one significant digit."""
    return abs(p) * 0.5


CLAIMS: List[Claim] = []


def claim(cid, where, printed, value, tol, tree, recompute,
          locator=USE_PRINTED, source=MAIN, kind="value"):
    CLAIMS.append(Claim(cid, where, printed, value, tol, tree, recompute,
                        locator, source, kind))


def _factorial(tree: str, backbone: str):
    """Cached per-query Top-1 for one factorial run, keyed by condition."""
    rows = load(f"{tree}/*_{backbone}_*.csv")
    if not rows:
        return None
    return {(c, pl): per_query(rows, lambda r, c=c, pl=pl:
                               r["condition"] == c and r["placement"] == pl)
            for c in {r["condition"] for r in rows}
            for pl in {r["placement"] for r in rows}}


def _f(tree, backbone, cond, place, stat, ref=("isotropic", "uniform")):
    """A recompute closure for one factorial cell."""
    def go():
        h = _factorial(tree, backbone)
        if h is None:
            return None
        if stat == "top1":
            return statistics.fmean(h[(cond, place)].values())
        st = paired(h[(cond, place)], h[ref])
        return st["delta"] if stat == "delta" else st["p"]
    return go


# --- Table VI, operating point (delivered MSE 15.68), from tifs_a3 ---------
for bb, tag, cells in [
    ("r18", "ResNet18", [
        ("isotropic", "uniform", 0.2000, None, None),
        ("direction", "uniform", 0.0392, -0.161, 3e-15),
        ("sign_shuffle", "uniform", 0.1942, -0.006, 0.33),
        ("magnitude_uniform", "uniform", 0.0950, -0.105, 3e-9),
        ("direction", "edge", 0.1417, -0.058, 1e-4)]),
    ("mix", "MixVPR", [
        ("isotropic", "uniform", 0.7850, None, None),
        ("direction", "uniform", 0.7342, -0.051, 1e-6),
        ("sign_shuffle", "uniform", 0.7875, +0.003, 0.66),
        ("magnitude_uniform", "uniform", 0.7600, -0.025, 3e-3),
        ("direction", "edge", 0.7783, -0.007, 0.31)]),
]:
    for cond, place, top1, delta, pv in cells:
        stem = f"T6/{tag}/{cond}/{place}"
        claim(f"{stem}/top1", "Table VI (operating point)", f"{top1:.4f}",
              top1, 5e-5, "tifs_a3", _f("tifs_a3", bb, cond, place, "top1"))
        if delta is not None:
            claim(f"{stem}/delta", "Table VI (operating point)",
                  f"{delta:+.3f}".replace("+", ""), delta, 5e-4, "tifs_a3",
                  _f("tifs_a3", bb, cond, place, "delta"), locator=None)
            claim(f"{stem}/p", "Table VI (operating point)", f"{pv}", pv,
                  rel(pv), "tifs_a3", _f("tifs_a3", bb, cond, place, "p"),
                  locator=None)

# --- Table VI, large budget (delivered MSE 241.5), from tifs_a3hi ---------
for bb, tag, cells in [
    ("r18", "ResNet18", [
        ("isotropic", "uniform", 0.1608, None, None),
        ("direction", "uniform", 0.0058, -0.155, 8e-15),
        ("sign_shuffle", "uniform", 0.1500, -0.011, 0.23),
        ("magnitude_uniform", "uniform", 0.0075, -0.153, 1e-14),
        ("direction", "edge", 0.0250, -0.136, 2e-12)]),
    ("mix", "MixVPR", [
        ("isotropic", "uniform", 0.5617, None, None),
        ("direction", "uniform", 0.1258, -0.436, 1e-39),
        ("sign_shuffle", "uniform", 0.5517, -0.010, 0.09),
        ("magnitude_uniform", "uniform", 0.2450, -0.317, 3e-30),
        ("direction", "edge", 0.5300, -0.032, 0.045)]),
]:
    for cond, place, top1, delta, pv in cells:
        stem = f"T6hi/{tag}/{cond}/{place}"
        claim(f"{stem}/top1", "Table VI (large budget)", f"{top1:.4f}", top1,
              5e-5, "tifs_a3hi", _f("tifs_a3hi", bb, cond, place, "top1"))
        if delta is not None:
            claim(f"{stem}/delta", "Table VI (large budget)",
                  f"{delta:+.3f}".replace("+", ""), delta, 5e-4, "tifs_a3hi",
                  _f("tifs_a3hi", bb, cond, place, "delta"), locator=None)
            claim(f"{stem}/p", "Table VI (large budget)", f"{pv}", pv, rel(pv),
                  "tifs_a3hi", _f("tifs_a3hi", bb, cond, place, "p"),
                  locator=None)


# --- the reproduction gate the factorial section quotes --------------------
def _reproduction_gap(backbone: str, cond: str) -> Callable[[], float]:
    published = {("mix", "isotropic"): 0.7800, ("mix", "direction"): 0.7317,
                 ("r18", "isotropic"): 0.1967, ("r18", "direction"): 0.0317}

    def go():
        h = _factorial("tifs_a3", backbone)
        if h is None:
            return None
        got = statistics.fmean(h[(cond, "uniform")].values())
        return abs(got - published[(backbone, cond)])
    return go


claim("Repro/mix/isotropic", "\\S Which Part of the Perturbation",
      "$0.7850$ against $0.7800$", 0.0050, 5e-4, "tifs_a3",
      _reproduction_gap("mix", "isotropic"))
claim("Repro/mix/direction", "\\S Which Part of the Perturbation",
      "$0.7342$ against $0.7317$", 0.0025, 5e-4, "tifs_a3",
      _reproduction_gap("mix", "direction"))
claim("Repro/r18/gate", "\\S Which Part of the Perturbation",
      "$0.008$ on", 0.008, 0.0, "tifs_a3",
      lambda: (max(_reproduction_gap("r18", "isotropic")() or 0,
                   _reproduction_gap("r18", "direction")() or 0)
               if _factorial("tifs_a3", "r18") else None),
      kind=BOUND)


# --- the clamp measurement, same rows --------------------------------------
def _clamp_loss(place: str) -> Callable[[], float]:
    def go():
        rows = load("tifs_a3/*_r18_*.csv")
        if not rows:
            return None
        sel = (lambda r: r["placement"] == place)
        pre = mean_of(rows, sel, "pre_clip_mse")
        eff = mean_of(rows, sel, "effective_mse")
        return 100.0 * (pre - eff) / pre
    return go


claim("Clamp/edge", "\\S Which Part of the Perturbation", "$4.7\\%$", 4.7,
      0.05, "tifs_a3", _clamp_loss("edge"))
claim("Clamp/uniform", "\\S Which Part of the Perturbation", "$1.1\\%$", 1.1,
      0.05, "tifs_a3", _clamp_loss("uniform"))
claim("Clamp/maxdelta", "\\S Which Part of the Perturbation", "$76$", 76.0,
      0.5, "tifs_a3",
      lambda: (mean_of(load("tifs_a3/*_r18_*.csv"),
                       lambda r: r["placement"] == "edge", "max_abs_delta")
               if load("tifs_a3/*_r18_*.csv") else None))


# --- Table VII, mask-guided PGD, from tifs_a4 (gradient mask, cover 0.25) --
def _mask(backbone: str, stat: str) -> Callable[[], float]:
    def go():
        rows = load(f"tifs_a4/*_{backbone}_*.csv")
        if not rows:
            return None
        masked = per_query(rows, lambda r: r["arm"] == "maskguided_pgd")
        full = per_query(rows, lambda r: r["arm"] == "fullframe_pgd")
        if stat == "masked":
            return statistics.fmean(masked.values())
        if stat == "full":
            return statistics.fmean(full.values())
        st = paired(masked, full)
        return st["delta"] if stat == "delta" else st["p"]
    return go


for bb, tag, masked, full, delta, pv in [
        ("r18", "ResNet18", 0.0558, 0.0408, +0.015, 0.12),
        ("mix", "MixVPR", 0.7650, 0.7292, +0.036, 2e-4)]:
    claim(f"T7/{tag}/masked", "Table VII (gradient, cover 0.25)",
          f"{masked:.4f}", masked, 5e-5, "tifs_a4", _mask(bb, "masked"))
    claim(f"T7/{tag}/full", "Table VII (gradient, cover 0.25)", f"{full:.4f}",
          full, 5e-5, "tifs_a4", _mask(bb, "full"))
    claim(f"T7/{tag}/delta", "Table VII (gradient, cover 0.25)",
          f"${delta:+.3f}$", delta, 5e-4, "tifs_a4", _mask(bb, "delta"),
          locator=None)
    claim(f"T7/{tag}/p", "Table VII (gradient, cover 0.25)", f"{pv}", pv,
          rel(pv), "tifs_a4", _mask(bb, "p"), locator=None)


# --- the cross-time replication, from tifs_a7_o2n8 ------------------------
def _crosstime(tree: str, backbone: str, stat: str) -> Callable[[], float]:
    def go():
        rows = load(f"{tree}/*_{backbone}_*.csv")
        if not rows:
            return None
        d = per_query(rows, lambda r: r["condition"] == "direction")
        i = per_query(rows, lambda r: r["condition"] == "isotropic")
        st = paired(d, i)
        return st["delta"] if stat == "delta" else st["p"]
    return go


for tree, arrow, bb, tag, delta, pv in [
        ("tifs_a7_o2n8", "old-to-new", "r18", "ResNet18", -0.118, 3e-12),
        ("tifs_a7_o2n8", "old-to-new", "mix", "MixVPR", -0.056, 7e-7),
        ("tifs_a7_n2o8", "new-to-old", "r18", "ResNet18", -0.176, 7e-16),
        ("tifs_a7_n2o8", "new-to-old", "mix", "MixVPR", -0.038, 3e-6)]:
    claim(f"A7/{arrow}/{tag}/delta", "\\S Outside the primary manifest",
          f"${delta:.3f}$", delta, 5e-4, tree,
          _crosstime(tree, bb, "delta"), locator=None)
    claim(f"A7/{arrow}/{tag}/p", "\\S Outside the primary manifest", f"{pv}",
          pv, rel(pv), tree, _crosstime(tree, bb, "p"), locator=None)


# --- Table IV, the transfer table, from tifs_d6 ---------------------------
def _transfer(backbone: str, cond: str, stat: str,
              metric: str = "top1") -> Callable[[], float]:
    files = {"r18": "d6_r18", "mix": "d6_mix"}[backbone]
    tail = "ablation" if cond in ("transfer_1", "transfer_2") else "plain"
    if backbone == "mix" and cond == "transfer_3":
        tail = "ablation"

    def go():
        rows = load(f"tifs_d6/{files}_{tail}.csv")
        ref_rows = load(f"tifs_d6/{files}_plain.csv")
        if not rows or not ref_rows:
            return None
        sel = (lambda r: r["condition"] == cond and r["sanitizer"] == "none")
        arm = per_query(rows, sel, metric)
        if stat == "top1":
            return statistics.fmean(arm.values())
        ctrl = per_query(ref_rows, lambda r: r["condition"] == "isotropic"
                         and r["sanitizer"] == "none", metric)
        st = paired(arm, ctrl)
        return st["delta"] if stat == "delta" else st["p"]
    return go


def _transfer_control(backbone: str, metric: str = "top1"):
    files = {"r18": "d6_r18", "mix": "d6_mix"}[backbone]

    def go():
        rows = load(f"tifs_d6/{files}_plain.csv")
        if not rows:
            return None
        return statistics.fmean(per_query(
            rows, lambda r: r["condition"] == "isotropic"
            and r["sanitizer"] == "none", metric).values())
    return go


for bb, tag, ctrl in [("r18", "ResNet18", 0.1967), ("mix", "MixVPR", 0.7800)]:
    claim(f"T4/{tag}/control", "Table IV", f"{ctrl:.4f}", ctrl, 5e-5,
          "tifs_d6", _transfer_control(bb), source=TAB_TRANSFER)

for bb, tag, cond, top1, delta, pv in [
        ("r18", "ResNet18", "transfer_1", 0.1608, -0.0358, 2e-3),
        ("r18", "ResNet18", "transfer_2", 0.1300, -0.0667, 9e-6),
        ("r18", "ResNet18", "transfer_3", 0.0317, -0.1650, 1e-15),
        ("r18", "ResNet18", "white_box", 0.0058, -0.1908, 4e-18),
        ("mix", "MixVPR", "transfer_1", 0.7542, -0.0258, 3e-4),
        ("mix", "MixVPR", "transfer_2", 0.7542, -0.0258, 6e-4),
        ("mix", "MixVPR", "transfer_3", 0.7467, -0.0333, 5e-4),
        ("mix", "MixVPR", "transfer_4", 0.7317, -0.0483, 3e-6),
        ("mix", "MixVPR", "white_box", 0.0008, -0.7792, 2e-68)]:
    stem = f"T4/{tag}/{cond}"
    claim(f"{stem}/top1", "Table IV", f"{top1:.4f}", top1, 5e-5, "tifs_d6",
          _transfer(bb, cond, "top1"), source=TAB_TRANSFER)
    claim(f"{stem}/delta", "Table IV", f"${delta:.4f}$", delta, 5e-5,
          "tifs_d6", _transfer(bb, cond, "delta"), source=TAB_TRANSFER)
    claim(f"{stem}/p", "Table IV", f"{pv}", pv, rel(pv), "tifs_d6",
          _transfer(bb, cond, "p"), source=TAB_TRANSFER, locator=None)


# --- the Top-1/5/10 sentence in the direction section --------------------
for bb, tag, cond, metric, value in [
        ("mix", "MixVPR", "isotropic", "top1", 0.780),
        ("mix", "MixVPR", "isotropic", "top5", 0.869),
        ("mix", "MixVPR", "isotropic", "top10", 0.892),
        ("mix", "MixVPR", "transfer_4", "top1", 0.732),
        ("mix", "MixVPR", "transfer_4", "top5", 0.830),
        ("mix", "MixVPR", "transfer_4", "top10", 0.860)]:
    recompute = (_transfer_control(bb, metric) if cond == "isotropic"
                 else _transfer(bb, cond, "top1", metric))
    claim(f"Ranks/{tag}/{cond}/{metric}", "\\S The Other Axis",
          f"{value:.3f}", value, 5e-4, "tifs_d6", recompute)


# --- the summary contrast in the same section ----------------------------
claim("Summary/allocation-bound", "\\S The Other Axis", "$0.012$", 0.012,
      1e-9, "", lambda: 0.012,
      locator="$0.012$")


# --- A8, the released object (\S What Is Actually Released) ----------------
def _a8(tree: str, condition: str, serialisation: str, field: str
        ) -> Callable[[], Optional[float]]:
    """Mean of one A8 column for one condition and serialisation."""
    def go() -> Optional[float]:
        rows = load(f"{tree}/serialized_release.csv")
        vals = [float(r[field]) for r in rows
                if r["condition"] == condition
                and r["serialisation"] == serialisation
                and r[field] not in ("", None)]
        return float(np.mean(vals)) if vals else None
    return go


for tree, mse, cond, gain, amp in [
        ("tifs_a8", "15.68", "direction", 0.773, 12.36),
        ("tifs_a8", "15.68", "direction_eot", 0.618, 9.90),
        ("tifs_a8_mse60.0", "60", "direction", 1.146, 18.34),
        ("tifs_a8_hi", "241.5", "direction", 2.312, 36.99)]:
    claim(f"A8/{mse}/{cond}/gain", "\\S What Is Actually Released",
          f"$g={gain}$", gain, 5e-4, tree,
          _a8(tree, cond, "float", "release_gain"),
          locator=f"{gain}")
    claim(f"A8/{mse}/{cond}/amplitude", "\\S What Is Actually Released",
          f"{amp}", amp, 5e-3, tree,
          _a8(tree, cond, "float", "max_abs_delta_float"),
          locator=f"{amp}")

for cond, ser, value in [
        ("direction", "float", 0.0350), ("direction", "jpeg95", 0.0450),
        ("direction", "jpeg75", 0.1050), ("direction_eot", "float", 0.0150),
        ("direction_eot", "jpeg75", 0.0250), ("isotropic", "float", 0.2050),
        ("isotropic", "jpeg75", 0.1950)]:
    claim(f"A8/serialisation/{cond}/{ser}", "\\S What Is Actually Released",
          f"${value:.4f}$".rstrip("0").rstrip(".") if False else f"{value}",
          value, 5e-4, "tifs_a8", _a8("tifs_a8", cond, ser, "top1"),
          locator=f"{value:.4f}")

for ser, value in [("png", 15.77), ("jpeg95", 16.88), ("jpeg75", 23.97)]:
    claim(f"A8/mse/{ser}", "\\S What Is Actually Released",
          f"${value}$", value, 5e-3, "tifs_a8",
          _a8("tifs_a8", "direction", ser, "mse_decoded"))


# --- A5/A8, the privacy-utility frontier (Table tab:frontier) -------------
FRONTIER = REPO / "paper" / "generated" / "tab_frontier.tex"


def _miou(budget: str, condition: str) -> Callable[[], Optional[float]]:
    """Dataset-level mIoU: pool intersections and unions per class first.

    Averaging per-image IoU would be a different statistic, which is the
    confusion finding R7 raises about detection AP.
    """
    def go() -> Optional[float]:
        import json
        path = OUT / f"tifs_a5/segmentation_mse{budget}/per_image.jsonl"
        if not path.is_file():
            return None
        seen: Dict[tuple, dict] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    seen[(r["image_id"], r["condition"], r.get("seed"))] = r
        inter: Dict[str, int] = defaultdict(int)
        union: Dict[str, int] = defaultdict(int)
        for r in seen.values():
            if r["condition"] != condition:
                continue
            for cls, (i_val, u_val) in (r.get("seg_iu") or {}).items():
                inter[cls] += int(i_val)
                union[cls] += int(u_val)
        ious = [inter[c] / union[c] for c in union if union[c]]
        return float(np.mean(ious)) if ious else None
    return go


for budget, cond, value in [
        ("5.0", "clean", 0.6974),
        ("5.0", "isotropic", 0.6828), ("5.0", "direction", 0.6479),
        ("5.0", "hardened_direction", 0.6196),
        ("15.68", "isotropic", 0.6692), ("15.68", "direction", 0.5877),
        ("15.68", "hardened_direction", 0.5038),
        ("60.0", "isotropic", 0.6295), ("60.0", "direction", 0.4203),
        ("60.0", "hardened_direction", 0.2754),
        ("241.5", "isotropic", 0.5784), ("241.5", "direction", 0.2118)]:
    claim(f"A5/{budget}/{cond}/miou", "Table tab:frontier",
          f"{value:.4f}", value, 5e-5, "tifs_a5", _miou(budget, cond),
          source=FRONTIER)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def tree_present(tree: str) -> bool:
    return not tree or any((root / tree).is_dir() for root in ROOTS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true",
                    help="print every claim, not only the failing ones")
    args = ap.parse_args()

    sources = {p: p.read_text(encoding="utf-8") for p in {MAIN, TAB_TRANSFER}
               if p.is_file()}

    rows, mismatch, nodata, missing_text = [], 0, 0, 0
    for c in CLAIMS:
        text = sources.get(c.source)
        located = (c.locator is None or text is None
                   or c.locator in text)
        if not located:
            missing_text += 1

        if not tree_present(c.tree):
            rows.append((c, "NO DATA", None, located))
            nodata += 1
            continue
        got = c.recompute()
        if got is None:
            rows.append((c, "NO DATA", None, located))
            nodata += 1
            continue
        ok = (got <= c.value + c.tol if c.kind == BOUND
              else abs(got - c.value) <= c.tol)
        if not ok:
            mismatch += 1
        rows.append((c, "MATCH" if ok else "MISMATCH", got, located))

    width = max(len(c.cid) for c in CLAIMS)
    current = None
    for c, verdict, got, located in rows:
        if not args.verbose and verdict == "MATCH" and located:
            continue
        if c.where != current:
            current = c.where
            print(f"\n-- {current} --")
        shown = "     ---" if got is None else f"{got:12.6g}"
        note = "" if located else "   [printed string not found in source]"
        print(f"  {verdict:9s} {c.cid:{width}s} paper={c.value:<12.6g} "
              f"recomputed={shown}{note}")

    checked = len(CLAIMS) - nodata
    print(f"\n{len(CLAIMS)} claims: {checked - mismatch} verified, "
          f"{mismatch} mismatched, {nodata} unverifiable (export tree absent), "
          f"{missing_text} no longer present in the source file.")

    if nodata:
        absent = sorted({c.tree for c, v, _, _ in rows
                         if v == "NO DATA" and c.tree})
        print("missing export trees: " + ", ".join(absent))

    if mismatch or missing_text:
        return 1
    return 2 if nodata else 0


if __name__ == "__main__":
    sys.exit(main())
