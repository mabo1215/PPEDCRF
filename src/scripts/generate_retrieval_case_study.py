"""Generate a retrieval-evidence case study for the revised qualitative figure.

The script consumes the per-query CSV from ``run_tomm_review_proxy.py`` and
renders the same query, its correct gallery image, and its strongest negative
alongside the original/sanitized ranks, similarities, and margins. It is
intended for a completed run only; smoke outputs must not be copied into
``paper/figs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SRC_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from datasets.monitoring_clip_dataset import MonitoringClipDataset  # noqa: E402
from main import load_sensnet_checkpoint  # noqa: E402
from run_tomm_review_proxy import protect_review_clip, select_eval_frame  # noqa: E402
from utils.config import load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a retrieval case study figure.")
    parser.add_argument("--per_query_csv", required=True)
    parser.add_argument("--selection_json", required=True)
    parser.add_argument("--monitoring_root", default=r"F:\work\datasets\monitoring\images")
    parser.add_argument("--config", default="src/config/config.yaml")
    parser.add_argument("--checkpoint", default="src/outputs/sensnet_final.pt")
    parser.add_argument("--output", default="paper/figs/retrieval_case_study.jpg")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gallery_size", type=int, default=48)
    return parser.parse_args()


def read_rows(path: str) -> List[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_frame_map(root: str, clip_ids: Sequence[str]) -> Dict[str, torch.Tensor]:
    dataset = MonitoringClipDataset(
        root=root,
        clip_ids=list(clip_ids),
        view="query",
        clip_len=4,
        resize_hw=(192, 320),
        min_frames=6,
    )
    return {sample.clip_id: select_eval_frame(sample.frames) for sample in dataset}


def to_image(frame: torch.Tensor):
    return frame.clamp(0, 255).byte().numpy().transpose(1, 2, 0)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.per_query_csv)
    selection = json.loads(Path(args.selection_json).read_text(encoding="utf-8"))
    filtered = [
        row
        for row in rows
        if row.get("backbone") == args.backbone
        and row.get("gallery_size") == str(args.gallery_size)
        and row.get("variant") in {"raw", args.variant}
        and (row.get("seed") == str(args.seed) or row.get("seed") == "raw")
    ]
    if not filtered:
        raise ValueError("No matching rows found for the requested case-study settings.")
    by_query = {row["query_id"]: row for row in filtered}
    candidates = []
    for query_id, row in by_query.items():
        if row.get("variant") == args.variant:
            candidates.append((float(row["retrieval_margin"]), query_id))
    if not candidates:
        raise ValueError("No sanitized rows found for the requested variant and seed.")
    _, query_id = min(candidates)
    sanitized_row = next(row for row in filtered if row["query_id"] == query_id and row.get("variant") == args.variant)
    raw_row = next(row for row in filtered if row["query_id"] == query_id and row.get("variant") == "raw")
    pair = next(item for item in selection["pairs"] if item["location_id"] == query_id)
    query_clip_id = pair["query_clip_id"]
    correct_clip_id = pair["gallery_clip_id"]
    negative_id = sanitized_row["hardest_negative_id"]
    clip_ids = [query_clip_id, correct_clip_id]
    if not negative_id.startswith("ext_"):
        clip_ids.append(negative_id)
    frame_map = load_frame_map(args.monitoring_root, clip_ids)
    query = frame_map[query_clip_id]
    correct = frame_map[correct_clip_id]
    negative = frame_map.get(negative_id, correct)
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sensnet = load_sensnet_checkpoint(args.checkpoint, device)
    sanitized_clip, _ = protect_review_clip(query.unsqueeze(0), sensnet, cfg, device, args.variant, args.seed)
    sanitized = sanitized_clip[0]

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    panels = [
        (query, "(a) Original query"),
        (sanitized, "(b) Sanitized query"),
        (correct, "(c) Correct gallery image"),
        (negative, "(d) Strongest competing image"),
    ]
    for axis, (frame, title) in zip(axes.flat, panels):
        axis.imshow(to_image(frame))
        axis.set_title(title, fontsize=10, weight="bold")
        axis.axis("off")
    figure.suptitle(
        "{} | query={} | raw rank={} sim={:.3f} margin={:.3f}; "
        "sanitized rank={} sim={:.3f} margin={:.3f}".format(
            args.backbone,
            query_id,
            raw_row["correct_rank"],
            float(raw_row["correct_similarity"]),
            float(raw_row["retrieval_margin"]),
            sanitized_row["correct_rank"],
            float(sanitized_row["correct_similarity"]),
            float(sanitized_row["retrieval_margin"]),
        ),
        fontsize=10,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    metadata = output.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "query_id": query_id,
                "query_clip_id": query_clip_id,
                "correct_gallery_id": sanitized_row["correct_gallery_id"],
                "hardest_negative_id": negative_id,
                "backbone": args.backbone,
                "variant": args.variant,
                "seed": args.seed,
                "gallery_size": args.gallery_size,
                "raw": raw_row,
                "sanitized": sanitized_row,
                "scientific_evidence": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[case-study] wrote {output} and {metadata}")


if __name__ == "__main__":
    main()
