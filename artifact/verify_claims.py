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
import json
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


# Placement-rule study: pooled Top-1 per placement, per checkpoint.
# (label, results subdir, placement, value printed in the paper)
PLACEMENT_CLAIMS = [
    ("constant/uniform",      "placement_study", "uniform", 0.6475),
    ("constant/learned",      "placement_study", "learned", 0.6475),
    ("constant/anti-oracle",  "placement_study", "anti_oracle_grad", 0.6448),
    ("constant/oracle",       "placement_study", "oracle_grad", 0.6585),
    ("constant/saliency",     "placement_study", "saliency", 0.6776),
    ("constant/centre",       "placement_study", "center", 0.6639),
    ("constant/random",       "placement_study", "random_fixed", 0.6885),
    ("constant/edge",         "placement_study", "edge", 0.7022),
    ("selective/uniform",     "placement_study_maskbacked", "uniform", 0.6421),
    ("selective/learned",     "placement_study_maskbacked", "learned", 0.6749),
    ("selective/anti-oracle", "placement_study_maskbacked", "anti_oracle_grad", 0.6421),
    ("selective/oracle",      "placement_study_maskbacked", "oracle_grad", 0.6393),
    ("selective/saliency",    "placement_study_maskbacked", "saliency", 0.6803),
    ("selective/centre",      "placement_study_maskbacked", "center", 0.6585),
    ("selective/random",      "placement_study_maskbacked", "random_fixed", 0.6749),
    ("selective/edge",        "placement_study_maskbacked", "edge", 0.6885),
]


# Real place-labelled MSLS replication of the placement null.
# (label, results subdir, placement, Top-1 printed in the paper)
MSLS_PLACEMENT_CLAIMS = [
    ("msls/uniform",      "placement_msls/final", "uniform", 0.1950),
    ("msls/oracle",       "placement_msls/final", "oracle_grad", 0.1942),
    ("msls/learned",      "placement_msls/final", "learned", 0.1958),
    ("msls/random",       "placement_msls/final", "random_fixed", 0.1958),
    ("msls/saliency",     "placement_msls/final", "saliency", 0.1958),
    ("msls/anti-oracle",  "placement_msls/final", "anti_oracle_grad", 0.1975),
    ("msls/edge",         "placement_msls/final", "edge", 0.2000),
    ("msls/centre",       "placement_msls/final", "center", 0.2042),
    ("msls/segmentation", "placement_msls/segmentation", "segmentation", 0.1942),
    # Three placements driven by published segmentation models, each paired
    # against a uniform control inside its own run.
    ("msls/segfcn uniform",   "placement_msls/segfcn", "uniform", 0.1950),
    ("msls/deeplabv3",        "placement_msls/segfcn", "segmentation", 0.1942),
    ("msls/fcn-resnet50",     "placement_msls/segfcn", "segmentation_fcn", 0.1958),
    ("msls/segade uniform",   "placement_msls/segade", "uniform", 0.1950),
    ("msls/segformer-ade20k", "placement_msls/segade", "segmentation_ade", 0.1950),
]

# How much spatial selectivity each placement rule expresses, as the share of
# squared weight carried by the top decile of pixels (uniform = 0.100 exactly).
CONCENTRATION_CLAIMS = [
    ("edge",         "edge",         0.839),
    ("centre",       "center",       0.649),
    ("saliency",     "saliency",     0.353),
    ("fixed random", "random_fixed", 0.242),
    ("uniform",      "uniform",      0.100),
]

# Controlled retrieval task with an exactly known Jacobian. The identity check
# is a bound on measured/predicted displacement; the advantage claims are the
# oracle's Top-1 difference against uniform at a given clean-task difficulty.
KNOWN_JACOBIAN_IDENTITY = (0.95, 1.02)
KNOWN_JACOBIAN_CLAIMS = [
    ("oracle advantage, clean 0.86", "linear", 1.0, -1.0, -0.423),
    ("oracle advantage, clean 0.34", "linear", 1.5, -1.0, -0.149),
    ("oracle advantage, clean 0.12", "linear", 2.0, -1.0, -0.042),
    ("oracle advantage, clean 0.03", "linear", 3.0, -1.0, +0.001),
    ("clipping halves it, clean 0.86", "linear", 1.0, 2.0, -0.329),
    ("nonlinear encoder, clean 0.70", "nonlinear", 1.0, -1.0, -0.373),
]

# Sensitivity statistics replicated on real imagery.
MSLS_SENSITIVITY_CLAIMS = [
    ("msls gradient CV",                 "grad_cv", 1.012, 0.05),
    ("msls top-decile (oracle)",         "grad_energy_top10pct_oracle", 0.647, 0.02),
    ("msls top-decile (learned)",        "grad_energy_top10pct_learned", 0.094, 0.02),
    ("msls Spearman learned vs grad",    "spearman_learned_vs_grad", 0.020, 0.02),
]

# The three cluster-robust significant high-budget results, all favouring edge
# placement. (label, subdir, expected delta, expected CI low, expected CI high)
HIGH_SIGMA_CLAIMS = [
    ("edge sigma50 constant",  "placement_highsigma_50pair/final/sigma_50",      -0.1400, -0.2467, -0.0400),
    ("edge sigma32 selective", "placement_highsigma_50pair/maskbacked/sigma_32", -0.0867, -0.1733, -0.0067),
    ("edge sigma50 selective", "placement_highsigma_50pair/maskbacked/sigma_50", -0.1333, -0.2533, -0.0133),
]

# Attacker-sensitivity statistics underpinning the mechanistic account.
SENSITIVITY_CLAIMS = [
    ("gradient CV",                 "grad_cv", 1.115, 0.05),
    ("top-decile energy (oracle)",  "grad_energy_top10pct_oracle", 0.677, 0.02),
    ("top-decile energy (learned)", "grad_energy_top10pct_learned", 0.099, 0.02),
    ("Spearman learned vs grad",    "spearman_learned_vs_grad", 0.009, 0.02),
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


    print("\n== Placement-rule study: pooled Top-1 per placement ==")
    for label, subdir, placement, expected in PLACEMENT_CLAIMS:
        paths = sorted((root / subdir).rglob("per_query.csv"))
        if not paths:
            failures.append(f"{label}: no per_query.csv under {subdir}")
            print(f"  MISSING  {label}")
            continue
        df = pd.concat([pd.read_csv(x) for x in paths], ignore_index=True)
        sub = df[df.placement == placement]
        if sub.empty:
            failures.append(f"{label}: placement absent")
            print(f"  FAIL  {label}: placement absent")
            continue
        checked += 1
        got = float((sub["correct_rank"] == 1).mean())
        ok = abs(got - expected) <= args.tolerance
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:24s} paper={expected:.4f} recomputed={got:.4f}")
        if not ok:
            failures.append(f"{label}: paper={expected:.4f} recomputed={got:.4f}")

    print("\n== High-budget reversal: edge placement beats uniform ==")
    for label, subdir, exp_d, exp_lo, exp_hi in HIGH_SIGMA_CLAIMS:
        path = root / subdir / "significance.csv"
        try:
            r = load(path)
        except FileNotFoundError as exc:
            failures.append(f"{label}: {exc}")
            print(f"  MISSING  {label}")
            continue
        row = r[r.placement == "edge"]
        if row.empty:
            failures.append(f"{label}: no edge row")
            print(f"  FAIL  {label}: no edge row")
            continue
        row = row.iloc[0]
        checked += 1
        ok = (abs(float(row.top1_diff) - exp_d) <= 0.002
              and abs(float(row.bootstrap_ci_low) - exp_lo) <= 0.01
              and abs(float(row.bootstrap_ci_high) - exp_hi) <= 0.01
              and bool(row.cluster_robust_significant))
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:24s} paper=({exp_d:+.4f}, [{exp_lo:.3f},{exp_hi:.3f}]) "
              f"recomputed=({float(row.top1_diff):+.4f}, [{float(row.bootstrap_ci_low):.3f},"
              f"{float(row.bootstrap_ci_high):.3f}]) sig={bool(row.cluster_robust_significant)}")
        if not ok:
            failures.append(f"{label}: mismatch or not cluster-robust significant")

    print("\n== Attacker-sensitivity statistics ==")
    sens_paths = sorted((root / "placement_study").rglob("sensitivity_stats.jsonl"))
    if not sens_paths:
        failures.append("sensitivity stats: no sensitivity_stats.jsonl found")
        print("  MISSING  sensitivity_stats.jsonl")
    else:
        recs = [json.loads(l) for p in sens_paths for l in p.open(encoding="utf-8") if l.strip()]
        sdf = pd.DataFrame(recs)
        for label, col, expected, tol in SENSITIVITY_CLAIMS:
            if col not in sdf:
                failures.append(f"{label}: column {col} absent")
                print(f"  FAIL  {label}: column absent")
                continue
            checked += 1
            got = float(sdf[col].mean())
            ok = abs(got - expected) <= tol
            print(f"  {'OK  ' if ok else 'FAIL'}  {label:28s} paper={expected:.3f} recomputed={got:.3f}")
            if not ok:
                failures.append(f"{label}: paper={expected:.3f} recomputed={got:.3f}")


    print("\n== Placement-rule concentration (top-decile energy share) ==")
    cpath = root / "placement_msls" / "concentration_cheap.jsonl"
    if not cpath.is_file():
        failures.append("concentration: concentration_cheap.jsonl absent")
        print("  MISSING  concentration_cheap.jsonl")
    else:
        conc = {r["placement"]: r for r in
                (json.loads(l) for l in cpath.open(encoding="utf-8") if l.strip())}
        for label, key, expected in CONCENTRATION_CLAIMS:
            if key not in conc:
                failures.append(f"concentration/{label}: absent")
                print(f"  FAIL  concentration/{label}: absent")
                continue
            checked += 1
            got = float(conc[key]["top_decile_energy_share_mean"])
            ok = abs(got - expected) <= 0.005
            print(f"  {'OK  ' if ok else 'FAIL'}  {label:16s} "
                  f"paper={expected:.3f} recomputed={got:.3f}")
            if not ok:
                failures.append(f"concentration/{label}: "
                                f"paper={expected:.3f} recomputed={got:.3f}")

    print("\n== Controlled study with a known Jacobian ==")
    kj_dir = root / "known_jacobian"
    kj = [json.loads(l) for f in sorted(kj_dir.glob("*.jsonl"))
          for l in f.open(encoding="utf-8") if l.strip()] if kj_dir.is_dir() else []
    if not kj:
        failures.append("known Jacobian: no exports found")
        print("  MISSING  known_jacobian/*.jsonl")
    else:
        ratios = [r["mean_sq_displacement"] / r["predicted_sq_displacement"]
                  for r in kj if r["clip"] < 0 and r.get("nonlinear") is not True
                  and r["predicted_sq_displacement"] > 0]
        checked += 1
        lo, hi = KNOWN_JACOBIAN_IDENTITY
        ok = bool(ratios) and min(ratios) >= lo and max(ratios) <= hi
        print(f"  {'OK  ' if ok else 'FAIL'}  first-order identity   "
              f"paper=[{lo:.2f},{hi:.2f}] recomputed="
              f"[{min(ratios):.3f},{max(ratios):.3f}] over {len(ratios)} cells")
        if not ok:
            failures.append("known Jacobian: first-order identity outside bound")
        for label, kind, nuis, clip, expected in KNOWN_JACOBIAN_CLAIMS:
            want_nl = (kind == "nonlinear")
            # The budget sweep reuses nuisance 1.5 at several sigmas, so the
            # cell must be pinned on sigma as well as on nuisance and clip.
            sel = [r for r in kj if bool(r.get("nonlinear")) == want_nl
                   and abs(r["nuisance"] - nuis) < 1e-9
                   and abs(r["clip"] - clip) < 1e-9
                   and abs(r["sigma"] - 0.35) < 1e-9]
            u = [r["top1"] for r in sel if r["placement"] == "uniform"]
            o = [r["top1"] for r in sel if r["placement"] == "oracle"]
            if not u or not o:
                failures.append(f"known Jacobian/{label}: cell absent")
                print(f"  FAIL  {label}: cell absent")
                continue
            checked += 1
            got = sum(o) / len(o) - sum(u) / len(u)
            ok = abs(got - expected) <= 0.02
            print(f"  {'OK  ' if ok else 'FAIL'}  {label:32s} "
                  f"paper={expected:+.3f} recomputed={got:+.3f}")
            if not ok:
                failures.append(f"known Jacobian/{label}: "
                                f"paper={expected:+.3f} recomputed={got:+.3f}")

    print("\n== Real place-labelled MSLS: placement null ==")
    for label, subdir, placement, expected in MSLS_PLACEMENT_CLAIMS:
        paths = sorted((root / subdir).rglob("per_query.csv"))
        if not paths:
            failures.append(f"{label}: no per_query.csv under {subdir}")
            print(f"  MISSING  {label}")
            continue
        df = pd.concat([pd.read_csv(x) for x in paths], ignore_index=True)
        sub = df[df.placement == placement]
        if sub.empty:
            failures.append(f"{label}: placement absent")
            print(f"  FAIL  {label}: placement absent")
            continue
        checked += 1
        got = float((sub["correct_rank"] == 1).mean())
        ok = abs(got - expected) <= args.tolerance
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:22s} paper={expected:.4f} recomputed={got:.4f}")
        if not ok:
            failures.append(f"{label}: paper={expected:.4f} recomputed={got:.4f}")

    print("\n== Real MSLS: attacker-sensitivity replication ==")
    msls_sens = sorted((root / "placement_msls/final").rglob("sensitivity_stats.jsonl"))
    if not msls_sens:
        failures.append("msls sensitivity: file not found")
        print("  MISSING  msls sensitivity_stats.jsonl")
    else:
        recs = [json.loads(l) for p in msls_sens for l in p.open(encoding="utf-8") if l.strip()]
        sdf = pd.DataFrame(recs)
        for label, col, expected, tol in MSLS_SENSITIVITY_CLAIMS:
            if col not in sdf:
                failures.append(f"{label}: column absent")
                print(f"  FAIL  {label}: column absent")
                continue
            checked += 1
            got = float(sdf[col].mean())
            ok = abs(got - expected) <= tol
            print(f"  {'OK  ' if ok else 'FAIL'}  {label:32s} paper={expected:.3f} recomputed={got:.3f}")
            if not ok:
                failures.append(f"{label}: paper={expected:.3f} recomputed={got:.3f}")

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
