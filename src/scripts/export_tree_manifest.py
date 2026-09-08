"""SHA-256 manifest of an export tree, for the F3 provenance record.

Prints one `sha256  relative/path` line per file, sorted, followed by a tree
digest: the SHA-256 of that listing. Two copies of a tree agree if and only if
their digests agree, and when they do not the listings say which file moved.

Run it with the same arguments on both machines and compare.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the per-tree digest lines")
    args = ap.parse_args()

    for root in args.roots:
        if not os.path.isdir(root):
            print("MISSING\t%s" % root)
            continue
        entries = []
        nbytes = 0
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                entries.append("%s  %s" % (file_sha256(full), rel))
                nbytes += os.path.getsize(full)
        entries.sort()
        if not args.quiet:
            for line in entries:
                print(line)
        digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
        print("TREE\t%s\t%s\t%d files\t%d bytes"
              % (digest, os.path.basename(root.rstrip("/")), len(entries),
                 nbytes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
