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


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
          - inter)
    return inter / ua if ua > 0 else 0.0


def per_image_ap(pred, target, iou_threshold: float = 0.5) -> float:
    """Class-averaged AP@50 for a single image.

    Inlined rather than imported so the artifact stays standalone: it needs
    only the exported predictions and the exported box targets, no dataset,
    no model and no repository code. This mirrors the evaluation used to
    produce the numbers, over one image at a time.
    """
    classes = sorted(set(int(x) for x in target.get("labels", []))
                     | set(int(x) for x in pred.get("labels", [])))
    if not classes:
        return 0.0
    aps = []
    for cid in classes:
        gt = [b for b, l in zip(target.get("boxes", []),
                                target.get("labels", [])) if int(l) == cid]
        scored = sorted(
            ((float(sc), b) for b, l, sc in zip(pred.get("boxes", []),
                                                pred.get("labels", []),
                                                pred.get("scores", []))
             if int(l) == cid), key=lambda t: t[0], reverse=True)
        if not gt:
            continue
        matched, tp, fp = set(), [], []
        for _, box in scored:
            best, best_j = 0.0, -1
            for j, g in enumerate(gt):
                if j in matched:
                    continue
                v = _iou(box, g)
                if v > best:
                    best, best_j = v, j
            if best >= iou_threshold and best_j >= 0:
                matched.add(best_j)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        c_tp = c_fp = 0.0
        prec, rec = [], []
        for t, f in zip(tp, fp):
            c_tp += t
            c_fp += f
            prec.append(c_tp / max(c_tp + c_fp, 1e-12))
            rec.append(c_tp / float(len(gt)))
        ap = 0.0
        for k in range(101):
            r = k / 100.0
            vals = [p for p, rr in zip(prec, rec) if rr >= r]
            ap += max(vals) if vals else 0.0
        aps.append(ap / 101.0)
    return sum(aps) / len(aps) if aps else 0.0


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

# Direction, optimised against surrogates and evaluated on a held-out
# attacker at the operating point's delivered MSE. (label, condition, Top-1)
# Superseded by the gallery-free objective below; kept so the earlier
# single-seed export still verifies against the number it produced.
TRANSFER_CLAIMS = [
    ("isotropic control", "isotropic",  0.1900),
    ("1 surrogate",       "transfer_1", 0.1450),
    ("2 surrogates",      "transfer_2", 0.1125),
    ("3 surrogates",      "transfer_3", 0.0625),
    ("white box",         "white_box",  0.0000),
]

# The paper's headline transfer table: the direction is optimised away from the
# frame's own clean embedding, so no reference database is assumed. Three seeds,
# n=1200 per condition. (file, label, condition, Top-1 printed in the paper)
GALLERY_FREE_CLAIMS = [
    ("resnet18.csv", "r18/isotropic",   "isotropic",  0.1967),
    ("resnet18.csv", "r18/1 surrogate", "transfer_1", 0.1508),
    ("resnet18.csv", "r18/2 surrogates","transfer_2", 0.1325),
    ("resnet18.csv", "r18/3 surrogates","transfer_3", 0.0358),
    ("resnet18.csv", "r18/white box",   "white_box",  0.0033),
    ("resnet18_reference_targeted.csv", "r18/ref-targeted 3 surr", "transfer_3", 0.0517),
    ("resnet18_reference_targeted.csv", "r18/ref-targeted white box", "white_box", 0.0000),
    ("mixvpr.csv", "mix/isotropic",    "isotropic",  0.7800),
    ("mixvpr.csv", "mix/1 surrogate",  "transfer_1", 0.7550),
    ("mixvpr.csv", "mix/2 surrogates", "transfer_2", 0.7517),
    ("mixvpr.csv", "mix/3 surrogates", "transfer_3", 0.7483),
    ("mixvpr.csv", "mix/4 surrogates", "transfer_4", 0.7342),
    ("mixvpr.csv", "mix/white box",    "white_box",  0.0008),
]

# Non-adaptive preprocessing, three seeds per cell (n=1200).
# (backbone, sanitizer, transfer condition, isotropic, white box, transfer)
SANITIZE_CLAIMS = [
    ("resnet18", "jpeg75",  "transfer_3", 0.2050, 0.0083, 0.1217),
    ("resnet18", "jpeg50",  "transfer_3", 0.1992, 0.0333, 0.1400),
    ("resnet18", "blur",    "transfer_3", 0.1725, 0.0175, 0.1283),
    ("resnet18", "denoise", "transfer_3", 0.1417, 0.0600, 0.1125),
    ("mixvpr",   "jpeg75",  "transfer_4", 0.7883, 0.4717, 0.7350),
    ("mixvpr",   "jpeg50",  "transfer_4", 0.7667, 0.5850, 0.7308),
    ("mixvpr",   "blur",    "transfer_4", 0.7642, 0.4817, 0.7033),
    ("mixvpr",   "denoise", "transfer_4", 0.7300, 0.5392, 0.7017),
]

# Preprocessing under the gallery-free objective, unhardened. Three seeds.
# (backbone, sanitizer, transfer condition, isotropic, white box, transfer)
SANFREE_CLAIMS = [
    ("resnet18", "jpeg75",  "transfer_3", 0.2050, 0.0133, 0.1167),
    ("resnet18", "jpeg50",  "transfer_3", 0.1992, 0.0533, 0.1500),
    ("resnet18", "blur",    "transfer_3", 0.1725, 0.0217, 0.1433),
    ("resnet18", "denoise", "transfer_3", 0.1417, 0.0617, 0.1067),
    ("mixvpr",   "jpeg75",  "transfer_4", 0.7875, 0.5733, 0.7633),
    ("mixvpr",   "jpeg50",  "transfer_4", 0.7667, 0.7042, 0.7558),
    ("mixvpr",   "blur",    "transfer_4", 0.7642, 0.5800, 0.7325),
    ("mixvpr",   "denoise", "transfer_4", 0.7300, 0.5625, 0.7250),
]

# The same sweep with the direction optimised in expectation over those
# transforms. (backbone, sanitizer, transfer condition, transfer, white box)
EOT_CLAIMS = [
    ("resnet18", "none",    "transfer_3", 0.0167, 0.0050),
    ("resnet18", "jpeg75",  "transfer_3", 0.0258, 0.0042),
    ("resnet18", "jpeg50",  "transfer_3", 0.0467, 0.0058),
    ("resnet18", "blur",    "transfer_3", 0.0550, 0.0042),
    ("resnet18", "denoise", "transfer_3", 0.0492, 0.0117),
    ("mixvpr",   "none",    "transfer_4", 0.6958, 0.0083),
    ("mixvpr",   "jpeg75",  "transfer_4", 0.7042, 0.0250),
    ("mixvpr",   "jpeg50",  "transfer_4", 0.7075, 0.1233),
    ("mixvpr",   "blur",    "transfer_4", 0.6583, 0.0192),
    ("mixvpr",   "denoise", "transfer_4", 0.6258, 0.0975),
]

# Downstream utility of the direction perturbation, per-image means over the
# repeated runs. (task, condition, value printed in the paper)
UTILITY_CLAIMS = [
    ("detection",    "clean",     0.6949),
    ("detection",    "isotropic", 0.6741),
    ("detection",    "direction", 0.6161),
    ("segmentation", "clean",     0.7387),
    ("segmentation", "isotropic", 0.7318),
    ("segmentation", "direction", 0.6835),
]

# Controlled retrieval task with an exactly known Jacobian. The identity check
# is a bound on measured/predicted displacement; the advantage claims are the
# oracle's Top-1 difference against uniform at a given clean-task difficulty.
KNOWN_JACOBIAN_IDENTITY = (0.95, 1.02)

# Operators at matched delivered MSE on the real benchmark.
# (label, subdir, uniform Top-1, edge minus uniform)
OPERATOR_CLAIMS = [
    ("op sigma8 gaussian",    "operator_study/sigma8_gaussian",    0.1950, +0.0025),
    ("op sigma8 correlated",  "operator_study/sigma8_correlated",  0.1892, -0.0008),
    ("op sigma8 blur",        "operator_study/sigma8_blur",        0.1975, +0.0100),
    ("op sigma8 mosaic",      "operator_study/sigma8_mosaic",      0.2000, -0.0075),
    ("op sigma32 gaussian",   "operator_study/sigma32_gaussian",   0.1592, -0.0433),
    ("op sigma32 correlated", "operator_study/sigma32_correlated", 0.0650, +0.0167),
    ("op sigma32 blur",       "operator_study/sigma32_blur",       0.1025, +0.0500),
    ("op sigma32 mosaic",     "operator_study/sigma32_mosaic",     0.0525, +0.1150),
]

# The strictly correct oracle, placing by the margin gradient.
MARGIN_CLAIMS = [
    ("margin/uniform",       "uniform",            0.1950),
    ("margin/margin oracle", "margin_oracle",      0.1867),
    ("margin/similarity",    "oracle_grad",        0.1942),
    ("margin/anti-margin",   "anti_margin_oracle", 0.1942),
]

# Operators in the controlled model, at matched delivered energy and the
# easiest task setting. (label, operator, uniform Top-1)
CONTROLLED_OPERATOR_CLAIMS = [
    ("controlled isotropic",    "isotropic",    0.832),
    ("controlled correlated",   "correlated",   0.843),
    ("controlled sign-random",  "sign_random",  0.845),
    ("controlled sign-aligned", "sign_aligned", 0.001),
]
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

    print("\n== Direction transfer to a held-out attacker ==")
    tf = root / "direction_transfer" / "per_query.csv"
    if not tf.is_file():
        failures.append("direction transfer: per_query.csv absent")
        print("  MISSING  direction_transfer/per_query.csv")
    else:
        tr = pd.read_csv(tf)
        hit = (tr.correct_rank == 1).astype(float)
        mses = tr.effective_mse
        checked += 1
        matched = float(mses.max() - mses.min()) <= 1e-3
        print(f"  {'OK  ' if matched else 'FAIL'}  delivered MSE matched across "
              f"conditions: [{float(mses.min()):.4f}, {float(mses.max()):.4f}]")
        if not matched:
            failures.append("direction transfer: delivered MSE not matched")
        for label, cond, expected in TRANSFER_CLAIMS:
            sel = hit[tr.condition == cond]
            if sel.empty:
                failures.append(f"transfer/{label}: no rows")
                print(f"  FAIL  transfer/{label}: no rows")
                continue
            checked += 1
            got = float(sel.mean())
            ok = abs(got - expected) <= 0.002
            print(f"  {'OK  ' if ok else 'FAIL'}  {label:20s} "
                  f"paper={expected:.4f} recomputed={got:.4f} (n={len(sel)})")
            if not ok:
                failures.append(f"transfer/{label}: paper={expected:.4f} "
                                f"recomputed={got:.4f}")

    print("\n== Gallery-free direction transfer (headline table) ==")
    gf_root = root / "direction_transfer_galleryfree"
    gf_cache = {}
    for fname, label, cond, expected in GALLERY_FREE_CLAIMS:
        f = gf_root / fname
        if fname not in gf_cache:
            if not f.is_file():
                failures.append(f"gallery-free: {fname} absent")
                print(f"  MISSING  {fname}")
                gf_cache[fname] = None
            else:
                gf_cache[fname] = pd.read_csv(f)
        df = gf_cache[fname]
        if df is None:
            continue
        sel = (df.correct_rank == 1).astype(float)[df.condition == cond]
        if sel.empty:
            failures.append(f"gallery-free/{label}: no rows")
            print(f"  FAIL  gallery-free/{label}: no rows")
            continue
        checked += 1
        got = float(sel.mean())
        ok = abs(got - expected) <= 0.002
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:28s} "
              f"paper={expected:.4f} recomputed={got:.4f} (n={len(sel)})")
        if not ok:
            failures.append(f"gallery-free/{label}: paper={expected:.4f} "
                            f"recomputed={got:.4f}")
    for fname in sorted({c[0] for c in GALLERY_FREE_CLAIMS}):
        df = gf_cache.get(fname)
        if df is None:
            continue
        checked += 1
        spread = float(df.effective_mse.max() - df.effective_mse.min())
        ok = spread <= 1e-3
        print(f"  {'OK  ' if ok else 'FAIL'}  {fname}: delivered MSE spread "
              f"{spread:.2e}")
        if not ok:
            failures.append(f"gallery-free/{fname}: delivered MSE not matched")

    print("\n== Non-adaptive preprocessing, three seeds per cell ==")
    for backbone, san, cond, exp_iso, exp_wb, exp_tr in SANITIZE_CLAIMS:
        f = root / "sanitize_3seed" / backbone / f"{san}.csv"
        if not f.is_file():
            failures.append(f"sanitize/{backbone}/{san}: absent")
            print(f"  MISSING  {backbone}/{san}")
            continue
        df = pd.read_csv(f)
        hit = (df.correct_rank == 1).astype(float)
        for name, want, c in (("isotropic", exp_iso, "isotropic"),
                              ("white box", exp_wb, "white_box"),
                              ("transfer", exp_tr, cond)):
            sel = hit[df.condition == c]
            if sel.empty:
                failures.append(f"sanitize/{backbone}/{san}/{name}: no rows")
                continue
            checked += 1
            got = float(sel.mean())
            ok = abs(got - want) <= 0.002
            print(f"  {'OK  ' if ok else 'FAIL'}  {backbone}/{san}/{name:9s} "
                  f"paper={want:.4f} recomputed={got:.4f} (n={len(sel)})")
            if not ok:
                failures.append(f"sanitize/{backbone}/{san}/{name}: "
                                f"paper={want:.4f} recomputed={got:.4f}")

    print("\n== Preprocessing, gallery-free objective (unhardened) ==")
    for backbone, san, cond, exp_iso, exp_wb, exp_tr in SANFREE_CLAIMS:
        f = root / "sanitize_galleryfree" / backbone / f"{san}.csv"
        if not f.is_file():
            failures.append(f"sanfree/{backbone}/{san}: absent")
            print(f"  MISSING  {backbone}/{san}")
            continue
        df = pd.read_csv(f)
        hit = (df.correct_rank == 1).astype(float)
        for name, want, c in (("isotropic", exp_iso, "isotropic"),
                              ("white box", exp_wb, "white_box"),
                              ("transfer", exp_tr, cond)):
            sel = hit[df.condition == c]
            if sel.empty:
                failures.append(f"sanfree/{backbone}/{san}/{name}: no rows")
                continue
            checked += 1
            got = float(sel.mean())
            ok = abs(got - want) <= 0.002
            print(f"  {'OK  ' if ok else 'FAIL'}  {backbone}/{san}/{name:9s} "
                  f"paper={want:.4f} recomputed={got:.4f} (n={len(sel)})")
            if not ok:
                failures.append(f"sanfree/{backbone}/{san}/{name}: "
                                f"paper={want:.4f} recomputed={got:.4f}")

    print("\n== The same directions hardened over those transforms (EOT) ==")
    for backbone, san, cond, exp_tr, exp_wb in EOT_CLAIMS:
        f = root / "direction_eot" / backbone / f"{san}.csv"
        if not f.is_file():
            failures.append(f"eot/{backbone}/{san}: absent")
            print(f"  MISSING  {backbone}/{san}")
            continue
        df = pd.read_csv(f)
        hit = (df.correct_rank == 1).astype(float)
        checked += 1
        spread = float(df.effective_mse.max() - df.effective_mse.min())
        if spread > 1e-3:
            failures.append(f"eot/{backbone}/{san}: delivered MSE not matched")
            print(f"  FAIL  {backbone}/{san}: MSE spread {spread:.2e}")
        for name, want, c in (("transfer", exp_tr, cond),
                              ("white box", exp_wb, "white_box")):
            sel = hit[df.condition == c]
            if sel.empty:
                failures.append(f"eot/{backbone}/{san}/{name}: no rows")
                continue
            checked += 1
            got = float(sel.mean())
            ok = abs(got - want) <= 0.002
            print(f"  {'OK  ' if ok else 'FAIL'}  {backbone}/{san}/{name:9s} "
                  f"paper={want:.4f} recomputed={got:.4f} (n={len(sel)})")
            if not ok:
                failures.append(f"eot/{backbone}/{san}/{name}: "
                                f"paper={want:.4f} recomputed={got:.4f}")

    print("\n== Downstream utility of the direction perturbation ==")
    for task, cond, expected in UTILITY_CLAIMS:
        f = root / "direction_utility" / f"{task}.jsonl"
        if not f.is_file():
            failures.append(f"utility/{task}: absent")
            print(f"  MISSING  utility/{task}")
            continue
        det_targets = {}
        if task == "detection":
            tf = root / "direction_utility" / "targets_detection.json"
            if not tf.is_file():
                failures.append("utility/detection: targets_detection.json absent")
                print("  MISSING  utility/detection targets")
                continue
            det_targets = json.loads(tf.read_text(encoding="utf-8"))
        per_image = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["condition"] != cond:
                    continue
                if "det" in row:
                    score = per_image_ap(row["det"],
                                         det_targets[row["image_id"]])
                else:
                    ious = [i / u for i, u in row["seg_iu"].values() if u]
                    score = sum(ious) / len(ious) if ious else 0.0
                per_image.setdefault(row["image_id"], []).append(score)
        if not per_image:
            failures.append(f"utility/{task}/{cond}: no rows")
            continue
        means = [sum(v) / len(v) for v in per_image.values()]
        got = sum(means) / len(means)
        checked += 1
        ok = abs(got - expected) <= 0.002
        print(f"  {'OK  ' if ok else 'FAIL'}  {task}/{cond:10s} "
              f"paper={expected:.4f} recomputed={got:.4f} "
              f"(n={len(means)} images)")
        if not ok:
            failures.append(f"utility/{task}/{cond}: paper={expected:.4f} "
                            f"recomputed={got:.4f}")

    print("\n== Operators at matched delivered MSE (real benchmark) ==")
    for label, subdir, exp_u, exp_d in OPERATOR_CLAIMS:
        f = root / subdir / "per_query.csv"
        if not f.is_file():
            failures.append(f"{label}: {subdir}/per_query.csv absent")
            print(f"  MISSING  {label}")
            continue
        r = pd.read_csv(f)
        hit = (r.correct_rank == 1).astype(float)
        u = float(hit[r.placement == "uniform"].mean())
        e = float(hit[r.placement == "edge"].mean())
        checked += 1
        ok = abs(u - exp_u) <= 0.002 and abs((e - u) - exp_d) <= 0.002
        print(f"  {'OK  ' if ok else 'FAIL'}  {label:22s} "
              f"paper=({exp_u:.4f}, {exp_d:+.4f}) recomputed=({u:.4f}, {e - u:+.4f})")
        if not ok:
            failures.append(f"{label}: paper=({exp_u:.4f},{exp_d:+.4f}) "
                            f"recomputed=({u:.4f},{e - u:+.4f})")

    print("\n== Placement by the margin gradient ==")
    mf = root / "margin_oracle" / "per_query.csv"
    if not mf.is_file():
        failures.append("margin oracle: per_query.csv absent")
        print("  MISSING  margin_oracle/per_query.csv")
    else:
        mr = pd.read_csv(mf)
        mhit = (mr.correct_rank == 1).astype(float)
        for label, placement, expected in MARGIN_CLAIMS:
            sel = mhit[mr.placement == placement]
            if sel.empty:
                failures.append(f"{label}: no rows")
                print(f"  FAIL  {label}: no rows")
                continue
            checked += 1
            got = float(sel.mean())
            ok = abs(got - expected) <= 0.002
            print(f"  {'OK  ' if ok else 'FAIL'}  {label:22s} "
                  f"paper={expected:.4f} recomputed={got:.4f}")
            if not ok:
                failures.append(f"{label}: paper={expected:.4f} recomputed={got:.4f}")

    print("\n== Operators in the controlled model (matched energy) ==")
    co_dir = root / "known_jacobian_operators"
    co = [json.loads(l) for f in sorted(co_dir.glob("*.jsonl"))
          for l in f.open(encoding="utf-8") if l.strip()] if co_dir.is_dir() else []
    if not co:
        failures.append("controlled operators: no exports found")
        print("  MISSING  known_jacobian_operators/*.jsonl")
    else:
        easiest = min(r["nuisance"] for r in co)
        for label, operator, expected in CONTROLLED_OPERATOR_CLAIMS:
            sel = [r["top1"] for r in co if r["operator"] == operator
                   and r["placement"] == "uniform" and r["clip"] < 0
                   and abs(r["nuisance"] - easiest) < 1e-9]
            if not sel:
                failures.append(f"{label}: cell absent")
                print(f"  FAIL  {label}: cell absent")
                continue
            checked += 1
            got = sum(sel) / len(sel)
            ok = abs(got - expected) <= 0.01
            print(f"  {'OK  ' if ok else 'FAIL'}  {label:24s} "
                  f"paper={expected:.3f} recomputed={got:.3f}")
            if not ok:
                failures.append(f"{label}: paper={expected:.3f} recomputed={got:.3f}")

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
