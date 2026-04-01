"""Format seed-averaged benchmark results for paper updates."""
import csv
import os

d = r"C:\source\PPEDCRF\src\outputs\controlled_retrieval_seed_avg"

# Robustness summary
with open(os.path.join(d, "robustness_summary.csv")) as f:
    rows = list(csv.DictReader(f))

pp = [r for r in rows if r["variant"] == "ppedcrf"]
print("=== ROBUSTNESS TABLE ===")
for r in pp:
    raw_r = [
        x
        for x in rows
        if x["variant"] == "raw"
        and x["backbone"] == r["backbone"]
        and x["gallery_size"] == r["gallery_size"]
    ][0]
    delta = float(r["R@1_mean"]) - float(raw_r["R@1_mean"])
    std = float(r["R@1_std"])
    print(
        f"{r['backbone']:15s} g{r['gallery_size']:>2s}: "
        f"raw={float(raw_r['R@1_mean']):.3f}  ppedcrf={float(r['R@1_mean']):.3f}+/-{std:.3f}  D={delta:+.3f}"
    )

neg = sum(
    1
    for r in pp
    if float(r["R@1_mean"])
    < float(
        [
            x
            for x in rows
            if x["variant"] == "raw"
            and x["backbone"] == r["backbone"]
            and x["gallery_size"] == r["gallery_size"]
        ][0]["R@1_mean"]
    )
)
pos = sum(
    1
    for r in pp
    if float(r["R@1_mean"])
    > float(
        [
            x
            for x in rows
            if x["variant"] == "raw"
            and x["backbone"] == r["backbone"]
            and x["gallery_size"] == r["gallery_size"]
        ][0]["R@1_mean"]
    )
)
zero = len(pp) - neg - pos
print(f"\nDelta stats: {neg} negative, {zero} zero, {pos} positive")

# Ablation table
print("\n=== ABLATION TABLE ===")
with open(os.path.join(d, "ablation_summary.csv")) as f:
    arows = list(csv.DictReader(f))
for r in arows:
    v = r["variant"]
    r1m = float(r["R@1_mean"])
    r1s = float(r.get("R@1_std", "0") or "0")
    r5m = float(r["R@5_mean"])
    r5s = float(r.get("R@5_std", "0") or "0")
    psnr = r.get("psnr_mean", "")
    ssim = r.get("ssim_mean", "")
    miou = r.get("mask_iou_mean", "")
    psnr_s = r.get("psnr_std", "")
    ssim_s = r.get("ssim_std", "")
    print(
        f"  {v:20s}: Top1={r1m:.3f}+/-{r1s:.3f}  Top5={r5m:.3f}+/-{r5s:.3f}  "
        f"PSNR={psnr[:6] if psnr else '--':6s}  SSIM={ssim[:5] if ssim else '--':5s}  IoU={miou[:5] if miou else '--'}"
    )

# Matched OP
print("\n=== MATCHED OPERATING POINT ===")
with open(os.path.join(d, "matched_operating_point.csv")) as f:
    mrows = list(csv.DictReader(f))
for r in mrows:
    print(
        f"  {float(r['target_psnr']):.0f}dB {r['variant']:15s}: "
        f"sigma={float(r['actual_sigma']):.0f}  psnr={float(r['actual_psnr']):.2f}  "
        f"R@1={float(r['R@1_mean']):.3f}+/-{float(r['R@1_std']):.3f}"
    )
