"""Check the manuscript's references into the Supplementary Material.

Why this exists. LaTeX resolves every reference inside one document and reports
the ones it cannot. It has nothing to say about a reference from the manuscript
into the supplement, because the two are compiled separately: such a reference
has to be written as a literal ``Table~S6'', and a literal cannot go stale
loudly. One did. Merging two supplementary tables shifted the numbering by one,
and the manuscript went on pointing at ``Table~S7'' for the budget sweep while
S7 had become the utility table. Every automatic check in this repository
passed while that was true -- the build reported zero undefined references, and
the claim auditor verifies values rather than pointers.

What it checks.

1. Every literal ``Table~S<n>'' or ``Fig.~S<n>'' in the manuscript resolves to
   a float that actually exists in the supplement, and the caption of that
   float is printed so a human can confirm it is the intended one.
2. Every phrase that promises material to the reader -- ``Supplementary
   Material'', ``extended evidence report'' -- is listed with its surrounding
   sentence, so the promises can be read against what the documents contain.
   This part cannot be decided mechanically; it is a checklist, not a verdict.

Floats are counted the way LaTeX numbers them: in source order, tables and
figures separately, following \\input into the generated table files.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]

FLOAT = re.compile(r"\\begin\{(table|figure)\}")
CAPTION = re.compile(r"\\caption\{")
INPUT = re.compile(r"\\input\{([^}]*)\}")
COMMENT_BLOCK = re.compile(r"\\begin\{comment\}.*?\\end\{comment\}", re.S)
REF = re.compile(r"(Table|Fig\.|Figure)~?\\?s?\s*S(\d+)")
PROMISE = re.compile(r"[^.]*\b(Supplementary Material|extended evidence report|"
                     r"extended report)\b[^.]*\.", re.S)


def expand(path: Path, seen=None) -> str:
    """Read a source with \\input files spliced in, comments removed."""
    seen = seen or set()
    if path in seen or not path.is_file():
        return ""
    seen.add(path)
    text = COMMENT_BLOCK.sub("", path.read_text(encoding="utf-8"))
    text = re.sub(r"(?m)^%.*$", "", text)

    def splice(match: re.Match) -> str:
        target = path.parent / (match.group(1) + ".tex")
        return expand(target, seen)

    return INPUT.sub(splice, text)


def floats_of(text: str) -> Dict[str, List[Tuple[int, str]]]:
    """Number the floats as LaTeX would: source order, per environment."""
    out: Dict[str, List[Tuple[int, str]]] = {"table": [], "figure": []}
    for match in FLOAT.finditer(text):
        kind = match.group(1)
        caption = CAPTION.search(text, match.end())
        title = ""
        if caption:
            depth, i = 1, caption.end()
            while i < len(text) and depth:
                depth += (text[i] == "{") - (text[i] == "}")
                i += 1
            title = " ".join(text[caption.end():i - 1].split())[:72]
        out[kind].append((len(out[kind]) + 1, title))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--main", default=str(REPO / "paper" / "main.tex"))
    ap.add_argument("--supplement",
                    default=str(REPO / "paper" / "supplementary.tex"))
    ap.add_argument("--list-promises", action="store_true",
                    help="also print every sentence that promises material, "
                         "for the manual half of the check")
    args = ap.parse_args()

    main_text = expand(Path(args.main))
    supp = floats_of(expand(Path(args.supplement)))
    print(f"[supplement] {len(supp['table'])} tables, {len(supp['figure'])} figures")

    failures = 0
    refs = list(REF.finditer(main_text))
    if not refs:
        print("[refs] no literal S-numbered reference in the manuscript; "
              "nothing can go stale")
    for match in refs:
        kind = "figure" if match.group(1).startswith("Fig") else "table"
        number = int(match.group(2))
        available = supp[kind]
        if number > len(available):
            print(f"[FAIL] {match.group(0)}: the supplement has only "
                  f"{len(available)} {kind}s")
            failures += 1
            continue
        print(f"[ref ] {match.group(0)} -> {available[number - 1][1]}")

    if args.list_promises:
        print("\n[promises] each of these names material a reader will look "
              "for; confirm the document named is the one that has it:")
        for match in PROMISE.finditer(main_text):
            print("   " + " ".join(match.group(0).split())[:160])

    print(f"\n[done] {failures} broken cross-document reference(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
