"""Synthesize a tiny sigma-sweep directory tree matching
run_tomm_review_proxy.py's per_query.csv/summary.csv schema, so
significance_test_matched_psnr.py and check_run_determinism.py can be
schema/logic smoke-tested without GPU access or the real monitoring corpus
(matching the synthetic-tensor smoke-test gate in docs/Design.md's
"Protocol freeze and local smoke test" step). This produces synthetic
retrieval outcomes only; it must never be used as a source of paper-facing
numbers.

Usage: python make_synthetic_sigma_sweep.py <out_dir> [tie]
The optional "tie" argument makes every variant's outcome identical, for
exercising the near-zero-difference code path (the pattern the real
matched-PSNR export actually shows)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIGMAS = [8, 16]
VARIANTS = ["full", "no_temporal", "no_ncp", "unary_only", "no_dcrf", "global_noise"]
SEEDS = [1234, 1235, 1236]
N_QUERIES = 12
BACKBONE = "resnet18"
GALLERY_SIZE = 48


def build(out_dir: Path, tie_all: bool) -> None:
    rng = np.random.default_rng(42)
    for sigma in SIGMAS:
        sigma_dir = out_dir / f"sigma_{sigma}"
        sigma_dir.mkdir(parents=True, exist_ok=True)
        per_query_rows = []
        summary_rows = []
        for variant in VARIANTS:
            psnr = 36.0 - sigma * 0.3 + (0.0 if variant == "full" else -0.02)
            for seed in SEEDS:
                for q in range(N_QUERIES):
                    if tie_all:
                        hit = 1 if (q + seed) % 3 != 0 else 0
                    else:
                        hit = 1 if rng.random() > (0.3 if variant == "full" else 0.5) else 0
                    per_query_rows.append({
                        "query_id": f"q{q:02d}",
                        "correct_rank": 1 if hit else 2,
                        "top5_hit": 1,
                        "top10_hit": 1,
                        "retrieval_margin": rng.normal(),
                        "psnr_mean": psnr + rng.normal(scale=0.01),
                        "ssim_mean": 0.9,
                        "variant": variant,
                        "seed": seed,
                        "backbone": BACKBONE,
                        "gallery_size": GALLERY_SIZE,
                    })
            top1 = np.mean([r["correct_rank"] == 1 for r in per_query_rows
                             if r["variant"] == variant])
            summary_rows.append({
                "variant": variant,
                "backbone": BACKBONE,
                "gallery_size": GALLERY_SIZE,
                "top1": top1,
                "psnr_mean_mean": psnr,
            })
        pd.DataFrame(per_query_rows).to_csv(sigma_dir / "per_query.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(sigma_dir / "summary.csv", index=False)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    build(target, tie_all=(len(sys.argv) > 2 and sys.argv[2] == "tie"))
    print(f"wrote synthetic sweep to {target}")
