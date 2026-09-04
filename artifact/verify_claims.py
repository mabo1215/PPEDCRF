"""Raw-output-to-table verifier for the PPEDCRF submission.

Recomputes every headline number in the paper's tables directly from the
released per-query CSV exports and compares each against the value printed in
the manuscript. No GPU and no model weights are required: this operates purely
on the exported raw retrieval outcomes, so a reviewer can confirm that the
tables were derived from the data rather than transcribed by hand.

Usage:
    python verify_claims.py --results_root <dir> [--tolerance 0.0005]

Exit code 0 means every claim matched; 1 means at least one mismatch, which is
printed with the expected and recomputed values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def top1(df: pd.DataFrame, variant: str) -> float:
    sub = df[df.variant == variant]
    if sub.empty:
        raise ValueError(f"variant '{variant}' absent")
    return float((sub["correct_rank"] == 1).mean())


def load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# (label, relative csv path, variant, value printed in the paper)
GEOTAGGED_CLAIMS = [
    # Table: wider 8-city primary manifest (all8)
    ("all8/ResNet18 raw", "session3/n1_expanded_msls/resnet18/geotagged_vpr_per_query.csv", "raw", 0.2100),
    ("all8/ResNet18 full", "session3/n1_expanded_msls/resnet18/geotagged_vpr_per_query.csv", "full", 0.1958),
    ("all8/ResNet50 raw", "session3/n1_expanded_msls/resnet50/geotagged_vpr_per_query.csv", "raw", 0.2675),
    ("all8/ResNet50 full", "session3/n1_expanded_msls/resnet50/geotagged_vpr_per_query.csv", "full", 0.2283),
    ("all8/VGG16 raw", "session3/n1_expanded_msls/vgg16/geotagged_vpr_per_query.csv", "raw", 0.1775),
    ("all8/VGG16 full", "session3/n1_expanded_msls/vgg16/geotagged_vpr_per_query.csv", "full", 0.1600),
    ("all8/CosPlace raw", "session3/n1_expanded_msls/cosplace/geotagged_vpr_per_query.csv", "raw", 0.4725),
    ("all8/CosPlace full", "session3/n1_expanded_msls/cosplace/geotagged_vpr_per_query.csv", "full", 0.4683),
    ("all8/MixVPR raw", "session3/n1_expanded_msls/mixvpr/geotagged_vpr_per_query.csv", "raw", 0.7925),
    ("all8/MixVPR full", "session3/n1_expanded_msls/mixvpr/geotagged_vpr_per_query.csv", "full", 0.7800),
    ("all8/Patch-NetVLAD raw", "session3/n1_expanded_msls/patchnetvlad/geotagged_vpr_per_query.csv", "raw", 0.5125),
    ("all8/Patch-NetVLAD full", "session3/n1_expanded_msls/patchnetvlad/geotagged_vpr_per_query.csv", "full", 0.4883),
    # Table: 8-city cross-time subtasks (o2n8 / n2o8)
    ("o2n8/ResNet18 raw", "session4/n6_crosstime8/o2n8/resnet18/geotagged_vpr_per_query.csv", "raw", 0.1550),
    ("o2n8/ResNet18 full", "session4/n6_crosstime8/o2n8/resnet18/geotagged_vpr_per_query.csv", "full", 0.1608),
    ("o2n8/ResNet50 raw", "session4/n6_crosstime8/o2n8/resnet50/geotagged_vpr_per_query.csv", "raw", 0.2375),
    ("o2n8/ResNet50 full", "session4/n6_crosstime8/o2n8/resnet50/geotagged_vpr_per_query.csv", "full", 0.1792),
    ("o2n8/VGG16 raw", "session4/n6_crosstime8/o2n8/vgg16/geotagged_vpr_per_query.csv", "raw", 0.1450),
    ("o2n8/VGG16 full", "session4/n6_crosstime8/o2n8/vgg16/geotagged_vpr_per_query.csv", "full", 0.1233),
    ("o2n8/CosPlace raw", "session4/n6_crosstime8/o2n8/cosplace/geotagged_vpr_per_query.csv", "raw", 0.4950),
    ("o2n8/CosPlace full", "session4/n6_crosstime8/o2n8/cosplace/geotagged_vpr_per_query.csv", "full", 0.4950),
    ("o2n8/MixVPR raw", "session4/n6_crosstime8/o2n8/mixvpr/geotagged_vpr_per_query.csv", "raw", 0.8200),
    ("o2n8/MixVPR full", "session4/n6_crosstime8/o2n8/mixvpr/geotagged_vpr_per_query.csv", "full", 0.7925),
    ("o2n8/Patch-NetVLAD raw", "session4/n6_crosstime8/o2n8/patchnetvlad/geotagged_vpr_per_query.csv", "raw", 0.4950),
    ("o2n8/Patch-NetVLAD full", "session4/n6_crosstime8/o2n8/patchnetvlad/geotagged_vpr_per_query.csv", "full", 0.4383),
    ("n2o8/ResNet18 raw", "session4/n6_crosstime8/n2o8/resnet18/geotagged_vpr_per_query.csv", "raw", 0.2550),
    ("n2o8/ResNet18 full", "session4/n6_crosstime8/n2o8/resnet18/geotagged_vpr_per_query.csv", "full", 0.2658),
    ("n2o8/ResNet50 raw", "session4/n6_crosstime8/n2o8/resnet50/geotagged_vpr_per_query.csv", "raw", 0.3400),
    ("n2o8/ResNet50 full", "session4/n6_crosstime8/n2o8/resnet50/geotagged_vpr_per_query.csv", "full", 0.2800),
    ("n2o8/VGG16 raw", "session4/n6_crosstime8/n2o8/vgg16/geotagged_vpr_per_query.csv", "raw", 0.2300),
    ("n2o8/VGG16 full", "session4/n6_crosstime8/n2o8/vgg16/geotagged_vpr_per_query.csv", "full", 0.2183),
    ("n2o8/CosPlace raw", "session4/n6_crosstime8/n2o8/cosplace/geotagged_vpr_per_query.csv", "raw", 0.5750),
    ("n2o8/CosPlace full", "session4/n6_crosstime8/n2o8/cosplace/geotagged_vpr_per_query.csv", "full", 0.5733),
    ("n2o8/MixVPR raw", "session4/n6_crosstime8/n2o8/mixvpr/geotagged_vpr_per_query.csv", "raw", 0.8450),
    ("n2o8/MixVPR full", "session4/n6_crosstime8/n2o8/mixvpr/geotagged_vpr_per_query.csv", "full", 0.8233),
    ("n2o8/Patch-NetVLAD raw", "session4/n6_crosstime8/n2o8/patchnetvlad/geotagged_vpr_per_query.csv", "raw", 0.6125),
    ("n2o8/Patch-NetVLAD full", "session4/n6_crosstime8/n2o8/patchnetvlad/geotagged_vpr_per_query.csv", "full", 0.5867),
    # White-box attacker (separate threat model, reported separately in the paper)
    ("all8 white-box aware", "session3/n1_whitebox/geotagged_vpr_per_query.csv", "attacker_aware", 0.0000),
    ("o2n8 white-box aware", "session4/n6_whitebox/o2n8/geotagged_vpr_per_query.csv", "attacker_aware", 0.0025),
    ("n2o8 white-box aware", "session4/n6_whitebox/n2o8/geotagged_vpr_per_query.csv", "attacker_aware", 0.0125),
]

# Deterministic-baseline matched-PSNR table: (label, sweep dir, point, variant, psnr, top1)
DETERMINISTIC_CLAIMS = [
    ("blur kernel 11", "session4/n7_merged/blur/k_11", "masked_blur", 30.61, 0.5833),
    ("blur kernel 5", "session4/n7_merged/blur/k_5", "masked_blur", 34.40, 0.9167),
    ("blur kernel 3", "session4/n7_merged/blur/k_3", "masked_blur", 37.95, 0.8333),
    ("mosaic block 4", "session4/n7_merged/mosaic/b_4", "masked_mosaic", 31.74, 0.7500),
    ("mosaic block 3", "session4/n7_merged/mosaic/b_3", "masked_mosaic", 32.34, 0.5833),
    ("mosaic block 2", "session4/n7_merged/mosaic/b_2", "masked_mosaic", 36.64, 0.8333),
]

# Matched-PSNR significance: every comparison must have zero discordant pairs.
SIGNIFICANCE_FILES = [
    ("gallery 12", "session3/n2_gallery_sweep/g12/matched_psnr_significance.csv"),
    ("gallery 100", "session3/n2_gallery_sweep/g100/matched_psnr_significance.csv"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="directory containing session3/ and session4/ result trees")
    ap.add_argument("--tolerance", type=float, default=0.0005)
    args = ap.parse_args()
    root = Path(args.results_root)

    failures, checked = [], 0

    print("== Retrieval Top-1 claims (recomputed from per-query CSVs) ==")
    for label, rel, variant, expected in GEOTAGGED_CLAIMS:
        try:
            got = top1(load(root / rel), variant)
        except (FileNotFoundError, ValueError) as exc:
            failures.append(f"{label}: {exc}")
            print(f"  MISSING  {label}: {exc}")
            continue
        checked += 1
        ok = abs(got - expected) <= args.tolerance
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:32s} paper={expected:.4f} recomputed={got:.4f}")
        if not ok:
            failures.append(f"{label}: paper={expected:.4f} recomputed={got:.4f}")

    print("\n== Deterministic-baseline matched-PSNR claims ==")
    for label, rel, variant, exp_psnr, exp_top1 in DETERMINISTIC_CLAIMS:
        path = root / rel / "summary.csv"
        try:
            df = load(path)
        except FileNotFoundError as exc:
            failures.append(f"{label}: {exc}")
            print(f"  MISSING  {label}: {exc}")
            continue
        sub = df[(df.variant == variant) & (df.backbone == "resnet18") & (df.gallery_size == 48)]
        if sub.empty:
            failures.append(f"{label}: no matching summary row")
            print(f"  FAIL  {label}: no matching summary row")
            continue
        checked += 2
        got_psnr = float(sub["psnr_mean_mean"].iloc[0])
        got_top1 = float(sub["top1"].iloc[0])
        ok = abs(got_psnr - exp_psnr) <= 0.01 and abs(got_top1 - exp_top1) <= args.tolerance
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:16s} paper=({exp_psnr:.2f} dB, {exp_top1:.4f}) "
              f"recomputed=({got_psnr:.2f} dB, {got_top1:.4f})")
        if not ok:
            failures.append(f"{label}: PSNR/Top-1 mismatch")

    print("\n== Matched-PSNR significance: zero discordant pairs claimed ==")
    for label, rel in SIGNIFICANCE_FILES:
        try:
            df = load(root / rel)
        except FileNotFoundError as exc:
            failures.append(f"{label}: {exc}")
            print(f"  MISSING  {label}: {exc}")
            continue
        checked += 1
        total = int(df["n_discordant"].sum())
        any_sig = bool(df["significant_at_alpha"].any())
        ok = total == 0 and not any_sig
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:12s} comparisons={len(df)} "
              f"discordant_total={total} any_significant={any_sig}")
        if not ok:
            failures.append(f"{label}: discordant={total} significant={any_sig}")

    print(f"\n{checked} claims checked, {len(failures)} mismatch(es).")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All paper claims reproduce from the raw exports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
