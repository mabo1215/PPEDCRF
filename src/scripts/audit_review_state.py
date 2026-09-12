"""One-off audit of the submission package against the current review cycle.

Reports bibliography hygiene, abstract length and math usage, generated-table
inclusion, and deferred-evidence pointers, so each finding can be checked
without re-reading both documents by hand.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"

COMMENT = re.compile(r"\\begin\{comment\}.*?\\end\{comment\}", re.S)


def live_text(path):
    return COMMENT.sub("", path.read_text(encoding="utf8"))


def bibliography():
    bib = (PAPER / "ref.bib").read_text(encoding="utf8")
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    cited = set()
    for name in ("main.tex", "supplementary.tex"):
        for m in re.finditer(r"\\cite[tp]?\{([^}]*)\}", live_text(PAPER / name)):
            cited.update(k.strip() for k in m.group(1).split(","))
    print(f"bib entries: {len(keys)}   distinct cited: {len(cited)}")
    print("  uncited:", sorted(k for k in keys if k not in cited) or "none")
    print("  cited but missing:", sorted(k for k in cited if k not in keys) or "none")


def abstract():
    body = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        (PAPER / "main.tex").read_text(encoding="utf8"),
        re.S,
    ).group(1)
    plain = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    plain = re.sub(r"[{}~\\]", " ", plain)
    print(f"abstract: {len(plain.split())} words, {body.count('$')} dollar signs")


def tables():
    main = live_text(PAPER / "main.tex")
    supp = live_text(PAPER / "supplementary.tex")
    print("generated tables:")
    for src in sorted((PAPER / "generated").glob("*.tex")):
        stem = src.stem
        where = []
        if f"generated/{stem}" in main:
            where.append("MAIN")
        if f"generated/{stem}" in supp:
            where.append("SUPP")
        print(f"  {stem:34s} {' '.join(where) or 'NOT INCLUDED'}")


def pointers():
    for name in ("main.tex", "supplementary.tex"):
        text = live_text(PAPER / name)
        hits = len(re.findall(r"extended report", text))
        print(f"{name}: 'extended report' x{hits}")


def floats():
    main = (PAPER / "main.tex").read_text(encoding="utf8")
    live = live_text(PAPER / "main.tex")
    for env in ("figure", "figure*", "table", "table*"):
        pat = r"\\begin\{" + env.replace("*", r"\*") + r"\}"
        print(
            f"main.tex {env:8s}: {len(re.findall(pat, main))} total, "
            f"{len(re.findall(pat, live))} live"
        )


if __name__ == "__main__":
    for section in (bibliography, abstract, tables, pointers, floats):
        print(f"--- {section.__name__} ---")
        section()
        print()
