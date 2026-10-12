"""Every prescribed placement rule against all five attackers, in one table.

The supplement is capped at six double-column pages, and the placement study
used to occupy three tables there: the weak attacker (tab_placement_msls), the
strong one (tab_placement_mixvpr) and the three-attacker panel
(tab_placement_panel). Between them they print the same nine rules five times
over, so this folds them into one row per rule and one column per attacker,
keeping what a reader needs to check the manuscript's attacker-dependent claim:
each paired difference, its interval and the verdict the declared margin
assigns it. The full tables, with both units of inference, the signed-rank
and equivalence p-values and the discordant-pair counts, move to the extended
evidence report unchanged.

This is a view of those three generated tables rather than a fourth
computation from the exports: it parses them, so it can never disagree with
them. Regenerate them first, then this.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "paper" / "generated"

# Row order and the one spelling the compact table prints. The three source
# tables abbreviate the inverse score-gradient rule differently.
RULES = [
    ("learned support", ("learned support",)),
    ("score-gradient", ("score-gradient",)),
    ("anti-score-grad.", ("anti-score-grad.", "anti-score-gradient")),
    ("saliency", ("saliency",)),
    ("centre bias", ("centre bias",)),
    ("fixed random", ("fixed random",)),
    ("edge magnitude", ("edge magnitude",)),
    ("margin gradient", ("margin gradient",)),
    ("anti-margin grad.", ("anti-margin grad.",)),
]
ATTACKERS = ["ResNet18", "MixVPR", "Patch-NetVLAD", "ViT-B/16", "CLIP ViT-L/14"]

NUM = r"[+-]?\d+\.\d+"
CI = re.compile(rf"\[\s*({NUM})\s*,\s*({NUM})\s*\]")


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.rstrip().rstrip("\\").split("&")]


def _delta(cell: str) -> float:
    return float(cell.strip("$ "))


def _ci(cell: str) -> Tuple[float, float]:
    m = CI.search(cell.replace("$", ""))
    if not m:
        raise SystemExit(f"no interval in cell {cell!r}")
    return float(m.group(1)), float(m.group(2))


Cell = Dict[str, object]


def parse_single(path: Path, ci_column: int) -> Tuple[float, Dict[str, Cell]]:
    """One attacker's table: the uniform reference and one cell per rule."""
    ref, cells = None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "&" not in line or not line.rstrip().endswith("\\\\"):
            continue
        c = _cells(line)
        name = c[0].replace(r"\quad", "").strip()
        if name.startswith("uniform"):
            ref = float(c[1])
            continue
        if name == "Placement":
            continue
        cells[name] = {"delta": _delta(c[2]), "ci": _ci(c[ci_column]),
                       "verdict": c[-1]}
    if ref is None:
        raise SystemExit(f"no uniform reference in {path}")
    return ref, cells


def parse_panel(path: Path) -> Dict[str, Tuple[float, Dict[str, Cell]]]:
    """The three-attacker panel: one block per attacker."""
    head = re.compile(r"\\textit\{([^}]*)\}\s*\(uniform reference\s*(" + NUM + r")")
    out: Dict[str, Tuple[float, Dict[str, Cell]]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = head.search(line)
        if m:
            current = m.group(1)
            out[current] = (float(m.group(2)), {})
            continue
        if current is None or "&" not in line or line.lstrip().startswith("Placement"):
            continue
        c = _cells(line)
        name = c[0].replace(r"\quad", "").strip()
        out[current][1][name] = {"delta": _delta(c[2]), "ci": _ci(c[3]),
                                 "verdict": c[-1]}
    return out


def fmt(cell: Cell) -> str:
    lo, hi = cell["ci"]
    # The interval is set in math mode so its signs print as minus signs
    # rather than hyphens.
    body = (rf"${cell['delta']:+.4f}$ "
            rf"{{\tiny$[{lo:+.3f},{hi:+.3f}]$}}")
    if cell["verdict"] == "separates":
        return rf"{{\bfseries\boldmath {body}}}"
    if cell["verdict"] == "negligible":
        return body + r"$^{\dagger}$"
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(GEN / "tab_placement_compact.tex"))
    args = ap.parse_args()

    # The weak-attacker table prints both units; the place-clustered interval
    # (fifth column) is the one the panel prints, so that is the one kept.
    r18 = parse_single(GEN / "tab_placement_msls.tex", ci_column=4)
    mix = parse_single(GEN / "tab_placement_mixvpr.tex", ci_column=3)
    panel = parse_panel(GEN / "tab_placement_panel.tex")
    source = {"ResNet18": r18, "MixVPR": mix, **panel}
    missing = [a for a in ATTACKERS if a not in source]
    if missing:
        raise SystemExit(f"attackers missing from the source tables: {missing}")

    ref_row = "uniform (Top-1)" + "".join(
        f" & {source[a][0]:.4f}" for a in ATTACKERS) + r" \\"
    body = []
    for label, aliases in RULES:
        row = [label]
        for a in ATTACKERS:
            cells = source[a][1]
            hit = next((cells[n] for n in aliases if n in cells), None)
            row.append(fmt(hit) if hit else "not run")
        body.append(" & ".join(row) + r" \\")

    caption = (
        r"Every prescribed placement rule against all five attackers on the "
        r"place-labelled manifest: 400 query clusters, three seeds averaged "
        r"within a query, identical delivered distortion. Each cell is "
        r"$\Delta$ Top-1 against that attacker's uniform reference (positive "
        r"is worse privacy) with its 95\% interval, place-clustered except "
        r"MixVPR's, which is query-level. \textbf{Bold}: the interval clears "
        r"zero after Holm correction within the attacker; $\dagger$: the whole "
        r"interval lies inside the $\pm0.01$ margin declared in advance. The "
        r"learned row is the released map, at this checkpoint the uniform "
        r"control itself; the margin-gradient pair was not run against MixVPR. "
        r"Signed-rank and equivalence $p$-values and both units of "
        r"inference are in the extended report.")
    lines = [
        "% Generated by src/scripts/make_placement_compact_table.py from the",
        "% three generated placement tables. Do not edit by hand.",
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:placement_compact}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(ATTACKERS) + "}", r"\hline",
        "Placement & " + " & ".join(ATTACKERS) + r" \\",
        r"\hline", ref_row, r"\hline", *body, r"\hline",
        r"\end{tabular}%", r"}", r"\end{table*}",
    ]
    out = Path(args.out)
    # newline="\n": the platform separator would otherwise rewrite every line
    # of a committed LF table when regenerated on Windows. open() rather than
    # write_text(), whose newline argument needs Python 3.10.
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
