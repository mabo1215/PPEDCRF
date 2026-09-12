"""Page counts and build warnings for the submitted documents.

The three things a TIFS submission can silently fail on are page count,
undefined references and boxes that overflow the column, and none of them
stops a build. This reads them back out of the compiled PDFs and the logs so
a page regression is caught in the same pass that made it.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
BUILD = PAPER / "build"

DOCS = [("main", PAPER / "main.pdf", 13),
        ("supplementary", PAPER / "supplementary.pdf", None),
        ("titlepage", PAPER / "titlepage.pdf", None)]

PAGE = re.compile(rb"/Type\s*/Page[^s]")


def pages(pdf: Path) -> int:
    return len(PAGE.findall(pdf.read_bytes()))


def main() -> int:
    bad = 0
    for name, pdf, cap in DOCS:
        if not pdf.is_file():
            print(f"{name}: MISSING")
            bad += 1
            continue
        n = pages(pdf)
        note = ""
        if cap is not None and n > cap:
            note = f"  OVER the {cap}-page limit"
            bad += 1
        print(f"{name:14s} {n:3d} pages{note}")

    for name, _, _ in DOCS:
        log = BUILD / f"{name}.log"
        if not log.is_file():
            continue
        text = log.read_text(encoding="utf8", errors="replace")
        undef = len(re.findall(r"LaTeX Warning: (Reference|Citation) .* undefined",
                               text))
        over = len(re.findall(r"Overfull \\hbox", text))
        fonts = len(re.findall(r"LaTeX Font Warning: Font shape", text))
        flag = "  <-- fix" if (undef or over or fonts) else ""
        print(f"{name:14s} undefined={undef} overfull={over} "
              f"fontshape={fonts}{flag}")
        bad += undef + over + fonts
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
