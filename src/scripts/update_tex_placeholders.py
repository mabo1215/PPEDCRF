"""
Update placeholders in paper/main.tex: \\texttt{<d0>}, \\texttt{<d1>}, $N_{\\max}$, $N_{\\min}$.

Usage:
  # Update only N_max, N_min from command line (e.g. from MOT16 class distribution):
  python src/scripts/update_tex_placeholders.py --N_max 1200 --N_min 50

  # Update N_max, N_min from a per-class count file (one count per line, or CSV with a count column):
  python src/scripts/update_tex_placeholders.py --counts_file path/to/counts.csv --update-tex

  # Update d0, d1 from command line (e.g. after computing elsewhere):
  python src/scripts/update_tex_placeholders.py --d0 0.45 --d1 0.62 --update-tex
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

def main():
    p = argparse.ArgumentParser(description="Update placeholders in paper/main.tex")
    p.add_argument("--d0", type=float, default=None)
    p.add_argument("--d1", type=float, default=None)
    p.add_argument("--N_max", type=int, default=None)
    p.add_argument("--N_min", type=int, default=None)
    p.add_argument("--counts_file", type=str, default=None,
                   help="CSV or text file with per-class counts (one per line or column 'count' / 'n')")
    p.add_argument("--update-tex", action="store_true", help="Write changes to paper/main.tex")
    args = p.parse_args()

    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(src_root)
    main_tex = os.path.join(project_root, "paper", "main.tex")
    if not os.path.isfile(main_tex):
        print(f"Not found: {main_tex}")
        sys.exit(1)

    n_max, n_min = args.N_max, args.N_min
    if args.counts_file and os.path.isfile(args.counts_file):
        counts = []
        with open(args.counts_file, "r", encoding="utf-8") as f:
            try:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
                reader = csv.DictReader(f, dialect=dialect)
                if reader.fieldnames:
                    # Prefer column named 'count' or 'n' or first numeric column
                    for row in reader:
                        for k, v in row.items():
                            try:
                                counts.append(int(float(v)))
                                break
                            except (ValueError, TypeError):
                                continue
                else:
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            try:
                                counts.append(int(float(line.split(",")[0].strip())))
                            except (ValueError, IndexError):
                                pass
            except csv.Error:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            counts.append(int(float(line.split()[0])))
                        except (ValueError, IndexError):
                            pass
        if counts:
            n_max = max(counts) if n_max is None else n_max
            n_min = min(counts) if n_min is None else n_min
            print(f"From {args.counts_file}: {len(counts)} classes, N_max={n_max}, N_min={n_min}")

    with open(main_tex, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = content
    if args.d0 is not None:
        new_content = new_content.replace(r"\texttt{<d0>}", f"{args.d0:.3f}")
    if args.d1 is not None:
        new_content = new_content.replace(r"\texttt{<d1>}", f"{args.d1:.3f}")
    if n_max is not None:
        new_content = new_content.replace(r"$N_{\max}$", str(n_max))
    if n_min is not None:
        new_content = new_content.replace(r"$N_{\min}$", str(n_min))

    if new_content == content:
        print("No placeholders to update (provide --d0, --d1, --N_max, --N_min and/or --counts_file)")
        return
    if not args.update_tex:
        print("Dry run. Add --update-tex to write to main.tex")
        return
    with open(main_tex, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Updated {main_tex}")


if __name__ == "__main__":
    main()
