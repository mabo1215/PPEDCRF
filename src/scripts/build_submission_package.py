"""Assemble the TIFS upload in submit/ from the built paper and the code.

The first package was put together by hand and nothing recorded how, so when
the editorial office returned the submission there was no way to rebuild it
except by reading commit messages. This script is that record. It writes:

  01_manuscript.pdf, 02_supplementary.pdf, 03_titlepage.pdf
      the built PDFs from paper/, copied unchanged
  05_prior_review_disclosure.pdf
      the response to the earlier ACM TOMM reviews with the original paper
      appended (paper/response.pdf). This PDF is the upload the office asked
      for, as a Supporting Document; the folder and zip of the same name add
      its LaTeX source for completeness only
  main/, supplementary/, 05_prior_review_disclosure/, 07_code/
      one folder per upload, each with a .zip of the same name beside it
  figs/
      every image the manuscript and the supplement include, flat, with
      figs.zip beside it for the system's figure slot
  cover_letter.txt
      docs/cover_letter.txt, for pasting into the submission form

Membership of each LaTeX folder is read from the document itself -- every
\\input, \\includegraphics and \\bibliography it reaches outside comments --
rather than listed by hand, so a table added to the manuscript is carried
without anyone remembering to add it. Each folder that reads another
document's numbers through xr also carries that document's .aux: without
main.aux the supplement compiles with "??" for its section references and
restarts its table numbering at I.

The code folder is the git-tracked code only: no exports, no outputs, no
caches, no weights, no manuscript text. It is screened for credentials before
it is zipped, and the build fails if anything matches.

Run paper/build.bat first; the script refuses to package a PDF older than its
source.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Iterable, List, Set

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper"
BUILD = PAPER / "build"
SUBMIT = REPO / "submit"

PDFS = [("main.pdf", "01_manuscript.pdf"),
        ("supplementary.pdf", "02_supplementary.pdf"),
        ("titlepage.pdf", "03_titlepage.pdf"),
        ("response.pdf", "05_prior_review_disclosure.pdf")]

# Code directories shipped whole (tracked files only), and single files.
CODE_DIRS = ["src/scripts", "src/eval", "src/models", "src/privacy",
             "src/datasets", "src/utils", "src/tests", "src/Docker",
             "src/config"]
CODE_FILES = ["src/main.py", "src/run_train.py", "src/demo_pipeline_smoke.py",
              "src/requirements.txt", "src/requirements-verify.txt"]

COMMENT_ENV = re.compile(r"\\begin\{comment\}.*?\\end\{comment\}", re.S)
LINE_COMMENT = re.compile(r"(?<!\\)%.*")
INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
BIB = re.compile(r"\\bibliography\{([^}]+)\}")
INCLUDEPDF = re.compile(r"\\includepdf(?:\[[^\]]*\])?\{([^}]+)\}")
EXTERNAL = re.compile(r"\\externaldocument(?:\[[^\]]*\])?\{([^}]+)\}")

# Patterns that would mean a secret is in the code upload. Deliberately broad:
# a false positive costs a look, a miss costs a revoked key. The remote-run
# helpers pass sshpass a shell variable read at run time from a file outside
# the repository; only a literal after -p is a leak.
SECRET = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}"
    r"|ghp_[A-Za-z0-9]{20,}|sshpass\s+-p\s*(?![\"']?\$)\S+"
    r"|password\s*[:=]\s*['\"][^'\"$]+['\"])",
    re.I)


def strip_comments(text: str) -> str:
    return LINE_COMMENT.sub("", COMMENT_ENV.sub("", text))


def tex_closure(root: Path) -> Set[Path]:
    """Every file a document reaches, as paths relative to paper/."""
    seen: Set[Path] = set()
    todo = [root]
    while todo:
        path = todo.pop()
        rel = path.relative_to(PAPER)
        if rel in seen or not path.is_file():
            continue
        seen.add(rel)
        if path.suffix != ".tex":
            continue
        text = strip_comments(path.read_text(encoding="utf-8"))
        for name in INPUT.findall(text):
            target = PAPER / name
            if target.suffix != ".tex":
                target = target.with_suffix(".tex")
            todo.append(target)
        for name in GRAPHIC.findall(text) + INCLUDEPDF.findall(text):
            target = PAPER / name
            if not target.suffix:
                hits = sorted(PAPER.glob(name + ".*"))
                target = hits[0] if hits else target
            todo.append(target)
        for names in BIB.findall(text):
            for name in names.split(","):
                todo.append(PAPER / (name.strip() + ".bib"))
    missing = [p for p in seen if not (PAPER / p).is_file()]
    if missing:
        raise SystemExit(f"{root.name} reaches files that do not exist: {missing}")
    return seen


def externals(root: Path) -> List[str]:
    text = strip_comments(root.read_text(encoding="utf-8"))
    return [name.strip() for name in EXTERNAL.findall(text)]


def check_fresh(pdf: Path, source: Path) -> None:
    if not pdf.is_file():
        raise SystemExit(f"{pdf} is missing: run paper/build.bat first")
    if pdf.stat().st_mtime < source.stat().st_mtime:
        raise SystemExit(f"{pdf.name} is older than {source.name}: "
                         "run paper/build.bat first")


def tracked(paths: Iterable[str]) -> List[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--", *paths],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def make_zip(folder: Path) -> Path:
    """Zip a folder with forward-slash entry names, in sorted order.

    zipfile writes POSIX separators whatever the platform, which matters: an
    archive with backslash entry names extracts on a Linux host as one flat
    directory of files with backslashes in their names.
    """
    archive = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in folder.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(folder).as_posix())
    return archive


def fresh_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def latex_folder(name: str, doc: str, extra: Iterable[str] = ()) -> Path:
    folder = fresh_dir(SUBMIT / name)
    files = tex_closure(PAPER / f"{doc}.tex")
    for rel in sorted(files):
        dest = folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER / rel, dest)
    # xr reads another document's numbers out of its .aux, which is a build
    # product, so it is taken from build/ and placed where xr looks for it.
    for other in list(externals(PAPER / f"{doc}.tex")) + list(extra):
        aux = BUILD / f"{other}.aux"
        if not aux.is_file():
            raise SystemExit(f"{aux} is missing: run paper/build.bat first")
        shutil.copy2(aux, folder / aux.name)
    return folder


IMAGE = {".png", ".jpg", ".jpeg", ".pdf", ".eps"}


def figs_folder() -> Path:
    """The images the two reviewed documents include, and nothing else.

    The first submission's figs.zip carried four images, three of which the
    documents had stopped using; reading the closure keeps this to what is
    actually printed. Commented-out figures are not included.
    """
    folder = fresh_dir(SUBMIT / "figs")
    images = sorted({rel for doc in ("main", "supplementary")
                     for rel in tex_closure(PAPER / f"{doc}.tex")
                     if rel.suffix.lower() in IMAGE})
    names = [rel.name for rel in images]
    if len(names) != len(set(names)):
        raise SystemExit(f"two figures share a file name: {names}")
    for rel in images:
        shutil.copy2(PAPER / rel, folder / rel.name)
    return folder


def code_folder() -> Path:
    folder = fresh_dir(SUBMIT / "07_code")
    files = tracked(CODE_DIRS) + [f for f in CODE_FILES if (REPO / f).is_file()]
    hits = []
    for rel in files:
        src = REPO / rel
        dest = folder / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        if src.suffix.lower() in {".py", ".sh", ".txt", ".md", ".yaml",
                                  ".yml", ".json", ".cfg", ".toml", ""}:
            for n, line in enumerate(src.read_text(encoding="utf-8",
                                                   errors="replace").splitlines(), 1):
                if SECRET.search(line):
                    hits.append(f"{rel}:{n}")
    if hits:
        raise SystemExit("possible credentials in the code upload:\n  "
                         + "\n  ".join(hits))
    shutil.copy2(SUBMIT / "07_code_README.md", folder / "README.md")
    exports = folder / "src" / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    (exports / "README.md").write_text(
        "Empty on purpose. Unpack the per-query exports deposited at IEEE "
        "DataPort (https://doi.org/10.21227/jnr0-jm15) here; see the README "
        "at the top of this package.\n", encoding="utf-8")
    leaked = [p for p in folder.rglob("*") if p.suffix in {".tex", ".pdf"}]
    if leaked:
        raise SystemExit(f"manuscript files in the code upload: {leaked}")
    return folder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()

    for built, _ in PDFS:
        check_fresh(PAPER / built, PAPER / built.replace(".pdf", ".tex"))
    SUBMIT.mkdir(exist_ok=True)

    for built, name in PDFS:
        shutil.copy2(PAPER / built, SUBMIT / name)
    shutil.copy2(REPO / "docs" / "cover_letter.txt", SUBMIT / "cover_letter.txt")

    folders = [latex_folder("main", "main"),
               latex_folder("supplementary", "supplementary")]
    disclosure = latex_folder("05_prior_review_disclosure", "response")
    shutil.copy2(PAPER / "response.pdf",
                 disclosure / "05_prior_review_disclosure.pdf")
    folders += [disclosure, code_folder(), figs_folder()]

    print(f"submit/ at {SUBMIT}")
    for _, name in PDFS:
        print(f"  {name:36s} {(SUBMIT / name).stat().st_size:>10,d} bytes")
    print(f"  {'cover_letter.txt':36s} "
          f"{(SUBMIT / 'cover_letter.txt').stat().st_size:>10,d} bytes")
    for folder in folders:
        archive = make_zip(folder)
        count = sum(1 for p in folder.rglob("*") if p.is_file())
        print(f"  {archive.name:36s} {archive.stat().st_size:>10,d} bytes, "
              f"{count} files (folder {folder.name}/ beside it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
