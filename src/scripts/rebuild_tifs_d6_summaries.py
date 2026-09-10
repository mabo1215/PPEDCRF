"""Rebuild the two D6 summary JSONs that make_tifs_tables.py reads.

The summaries live under src/outputs/, which is gitignored, so a fresh
checkout has the per-query rows but not the summaries, and
`make_tifs_tables.py` cannot run. The rows themselves are small enough to
travel and are committed under src/exports/tifs_d6/: eight files, an
unhardened run and three EOT-hardened seeds per attacker. This script turns
them back into the summaries, so the step is one command rather than a piece
of local knowledge.

    python src/scripts/rebuild_tifs_d6_summaries.py
    python src/scripts/make_tifs_tables.py

Nothing is recomputed here. `analyze_tifs_d6.py --json` is invoked once per
attacker, unchanged, so the summaries are the same artifact the tables were
originally generated from and the regenerated tables reproduce the committed
ones cell for cell.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
BACKBONES = ("resnet18", "mixvpr")
# One unhardened run and three hardened seeds per attacker.
EXPECTED_FILES = 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports",
                    default=str(REPO / "src" / "exports" / "tifs_d6"),
                    help="The committed per-query rows.")
    ap.add_argument("--summary_dir",
                    default=str(REPO / "src" / "outputs" / "tifs_d6"
                                / "summary"),
                    help="Where make_tifs_tables.py looks for the summaries.")
    args = ap.parse_args()

    exports = Path(args.exports)
    rows = sorted(exports.glob("d6_*.csv"))
    if len(rows) < EXPECTED_FILES:
        print(f"[FAIL] {exports} holds {len(rows)} d6_*.csv file(s), expected "
              f"at least {EXPECTED_FILES}; the summaries would be built from "
              f"an incomplete sweep")
        return 1

    out = Path(args.summary_dir)
    out.mkdir(parents=True, exist_ok=True)
    for backbone in BACKBONES:
        target = out / f"{backbone}.json"
        cmd = [sys.executable, str(SCRIPTS / "analyze_tifs_d6.py"),
               "--out_dir", str(exports), "--backbone", backbone,
               "--json", str(target)]
        done = subprocess.run(cmd, cwd=str(REPO))
        if done.returncode != 0:
            print(f"[FAIL] {backbone}: analyze_tifs_d6.py exited "
                  f"{done.returncode}")
            return done.returncode
        print(f"[done] {backbone} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
