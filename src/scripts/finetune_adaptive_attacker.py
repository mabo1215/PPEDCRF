"""G2 tier 2: an attacker that fine-tunes on this defense's typical output.

Tier 1 (see sanitizers.py / run_direction_transfer_study.py --sanitizer)
tested an attacker that mechanically preprocesses received frames without
knowing anything about the defense. This script tests a stronger adversary:
one that has collected examples of frames sanitized by this mechanism (the
same isotropic, delivered-MSE-matched noise used as the operating-point
control throughout this study) and fine-tunes its own embedder's last block
on them, so it recognizes places despite that noise. Everything before the
last block stays frozen -- an attacker who retrains only the head of an
off-the-shelf backbone is far more realistic than one who retrains from
scratch.

The gallery side of the objective -- both the fixed positive/negative
targets used to compute the loss, and the reference index used to rank
against -- is deliberately kept at the pretrained model's embeddings
throughout and after fine-tuning: re-embedding a whole reference gallery
every time a query encoder is adapted is not something a real deployed
system does. run_direction_transfer_study.py --eval_checkpoint mirrors this:
it embeds the gallery with the stock model and only the query side (and the
white_box perturbation's gradient target) with the fine-tuned one.

Queries are split by place into a training set, a validation set, and a
held-out test set before any training happens. Negatives are hard-mined each
step (the most confusable of K random wrong-place candidates, scored against
the current, evolving anchor embedding -- free, since gallery embeddings are
already precomputed and fixed) rather than drawn uniformly at random, since
an early version of this script trained against easy random negatives for
many epochs and overfit: it drove the training loss to near zero while
*losing* held-out isotropic Top-1 accuracy relative to the un-fine-tuned
model. The validation set (also place-disjoint from training and test) is
used for simple early stopping -- the checkpoint saved is whichever epoch
had the best validation Top-1 retrieval accuracy under the same noise the
model trains on, not necessarily the last epoch -- to guard against the same
failure mode recurring.

The held-out test query_ids are written alongside the checkpoint so
run_direction_transfer_study.py --query_id_file can evaluate on exactly the
queries the fine-tuned model never saw during training or model selection.

--train_perturbation selects what the attacker is assumed to have collected.
`isotropic` is the original behaviour and reproduces the published sweep.
`direction` instead trains on frames carrying the *direction* perturbation
this paper actually proposes, which is the exposure an adversary who
anticipates this defense would have. The distinction matters: an attacker
adapted to isotropic noise has adapted to the operating-point control, not to
the defense, so the two experiments answer different questions and both are
reported. Direction-perturbed training frames are deterministic given the
frame and the surrogate ensemble, so they are computed once into a cache
directory and reused across epochs and across configurations; that also means
this mode trains without the fresh-noise augmentation the isotropic mode gets
for free, which is a real difference rather than an implementation detail.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import zlib
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
    preprocess_for_embed,
)
from scripts.run_direction_transfer_study import (  # noqa: E402
    directional_delta,
    embed_gallery_batched,
    normalised_embedding,
    release_at_mse,
)
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)


def trainable_submodule(embedder: torch.nn.Module, backbone: str,
                        unfreeze_blocks: int = 1) -> List[torch.nn.Module]:
    """Freeze everything except the last `unfreeze_blocks` blocks.

    resnet18/resnet50: `features` is [conv1,bn1,relu,maxpool,layer1..4,avgpool]
    (see eval/retrieval_attack.py ImageEmbedder); layer4 is index 7, layer3 is
    index 6, so unfreeze_blocks=2 unfreezes layer3+layer4 together.
    mixvpr: `aggregator` sits on top of a frozen `backbone`; unfreeze_blocks
    is not meaningful there and must be 1.
    """
    for p in embedder.parameters():
        p.requires_grad_(False)
    b = backbone.lower()
    if b in ("resnet18", "resnet50"):
        if not 1 <= unfreeze_blocks <= 4:
            raise ValueError("unfreeze_blocks must be between 1 and 4 for resnet")
        blocks = [embedder.features[7 - i] for i in range(unfreeze_blocks)]
    elif b.startswith("mixvpr"):
        if unfreeze_blocks != 1:
            raise ValueError("unfreeze_blocks>1 not implemented for mixvpr")
        blocks = [embedder.aggregator]
    else:
        raise ValueError(f"Fine-tuning not implemented for backbone {backbone!r}")
    for block in blocks:
        for p in block.parameters():
            p.requires_grad_(True)
    return blocks


def split_queries(records: List[dict], n_test: int, n_val: int, seed: int):
    """Deterministic, mutually place-disjoint train/val/test split."""
    ordered = sorted(records, key=lambda r: r["query_id"])
    test_records = ordered[:n_test]
    test_places = {r["place_id"] for r in test_records}
    remaining = [r for r in ordered[n_test:] if r["place_id"] not in test_places]

    val_records = remaining[:n_val]
    val_places = {r["place_id"] for r in val_records}
    train_records = [r for r in remaining[n_val:] if r["place_id"] not in val_places]

    rng = random.Random(seed)
    rng.shuffle(train_records)
    return train_records, val_records, test_records


def isotropic_perturb(frame: torch.Tensor, target_mse: float, seed: int) -> torch.Tensor:
    gnoise = torch.Generator(device="cpu").manual_seed(seed)
    delta = torch.randn(frame.shape, generator=gnoise)
    return release_at_mse(frame.unsqueeze(0), delta.unsqueeze(0), target_mse).squeeze(0)


def build_direction_cache(records, cache_dir: Path, args, gallery, gallery_ids,
                          place_of, gallery_tensor, resize_hw, device) -> None:
    """Cache the direction-perturbed version of every training/validation frame.

    The perturbation is deterministic given the frame and the surrogate
    ensemble, so it is computed once and reused across epochs and across
    hyper-parameter configurations. Files are written one per query and the
    build skips whatever is already on disk, so an interrupted run resumes
    where it stopped instead of regenerating everything.

    The surrogate embedders live only inside this function so they are freed
    before training starts; on a shared GPU, holding four extra backbones for
    the whole run is what turns a comfortable job into an OOM.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in records
            if not (cache_dir / f"{r['query_id']}.pt").is_file()]
    if not todo:
        print(f"[finetune] direction cache complete ({len(records)} frames)",
              flush=True)
        return
    print(f"[finetune] building direction cache: {len(todo)} of "
          f"{len(records)} frames missing", flush=True)

    embedders, sizes, surr_gal = {}, {}, {}
    for b in args.direction_surrogates:
        scfg = RetrievalConfig(backbone=b,
                               input_size=default_input_size_for_backbone(b))
        e = make_default_embedder(scfg).eval().to(device)
        embedders[b] = e
        sizes[b] = scfg.input_size
        if args.direction_objective == "positive":
            surr_gal[b] = embed_gallery_batched(scfg, e, gallery_tensor)
        torch.cuda.empty_cache()
        print(f"[finetune] surrogate ready: {b}", flush=True)

    for i, rec in enumerate(todo, 1):
        frame = load_image(rec["query_path"], resize_hw).unsqueeze(0).to(device)
        if args.direction_objective == "positive":
            pos = next((j for j, g in enumerate(gallery_ids)
                        if place_of[g] == rec["place_id"]), None)
            if pos is None:
                continue
            tgts = [surr_gal[b][pos].to(device)
                    for b in args.direction_surrogates]
        else:
            with torch.no_grad():
                tgts = [normalised_embedding(embedders[b], frame,
                                             sizes[b]).detach()
                        for b in args.direction_surrogates]
        gstart = torch.Generator(device="cpu").manual_seed(
            zlib.crc32(f"{rec['query_id']}|{args.seed}".encode()) & 0x7FFFFFFF)
        with torch.enable_grad():
            delta = directional_delta(
                frame, tgts, [embedders[b] for b in args.direction_surrogates],
                [sizes[b] for b in args.direction_surrogates],
                args.direction_steps, args.direction_step_size,
                args.direction_linf,
                random_start=args.direction_random_start,
                generator=gstart)
        if float(delta.abs().max()) == 0.0:
            raise RuntimeError(
                f"Zero perturbation for {rec['query_id']}: the random start "
                f"is not doing its job.")
        released = release_at_mse(frame, delta, args.target_mse)
        torch.save(released.squeeze(0).cpu(),
                   cache_dir / f"{rec['query_id']}.pt")
        if i % 25 == 0:
            print(f"[finetune] direction cache {i}/{len(todo)}", flush=True)

    for b in list(embedders):
        embedders.pop(b)
    surr_gal.clear()
    torch.cuda.empty_cache()
    print("[finetune] direction cache complete", flush=True)


def cached_direction_perturb(rec, cache_dir: Path) -> torch.Tensor:
    path = cache_dir / f"{rec['query_id']}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Direction-perturbed frame missing for {rec['query_id']}; the "
            f"cache under {cache_dir} is incomplete.")
    return torch.load(path, map_location="cpu")


def validation_top1(embedder, cfg, gal_emb, gallery_ids, place_of, val_records,
                    resize_hw, target_mse, device, perturb=None):
    """Top-1 retrieval accuracy on perturbed validation queries.

    The perturbation matches whatever the model trains on, so the
    early-stopping signal measures the thing being adapted to rather than a
    different distribution. In the default isotropic mode the noise is fixed
    per query (seeded by index, not resampled), so the metric is comparable
    across epochs -- otherwise re-randomizing the noise every call would add
    its own variance to the early-stopping signal. The direction mode is
    deterministic by construction.
    """
    correct = 0
    ranks: List[int] = []
    with torch.no_grad():
        for i, rec in enumerate(val_records):
            frame = load_image(rec["query_path"], resize_hw)
            if perturb is None:
                released = isotropic_perturb(frame, target_mse, seed=i)
            else:
                released = perturb(rec, frame, i)
            perturbed = released.unsqueeze(0).to(device)
            x = preprocess_for_embed(perturbed, cfg.input_size)
            q = F.normalize(embedder(x), dim=1).flatten()
            sims = gal_emb @ q
            top1 = int(torch.argmax(sims).item())
            if place_of[gallery_ids[top1]] == rec["place_id"]:
                correct += 1
            # Rank of the query's own place. Top-1 alone is a very coarse
            # early-stopping signal when the perturbation has already driven
            # it near zero: on the direction arm the baseline is 2 correct out
            # of 50, so any gain smaller than one query is invisible and "the
            # attacker did not improve" would be unfalsifiable. The rank of
            # the correct place moves continuously and is reported alongside,
            # without being used for selection -- selection stays on Top-1 so
            # the direction arm remains comparable to the published isotropic
            # sweep.
            order = torch.argsort(sims, descending=True)
            for pos, j in enumerate(order.tolist(), start=1):
                if place_of[gallery_ids[j]] == rec["place_id"]:
                    ranks.append(pos)
                    break
    if ranks:
        sorted_ranks = sorted(ranks)
        median_rank = float(sorted_ranks[len(sorted_ranks) // 2])
        mrr = sum(1.0 / r for r in ranks) / len(ranks)
    else:
        median_rank, mrr = float("nan"), float("nan")
    return correct / max(len(val_records), 1), median_rank, mrr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50", "mixvpr"])
    ap.add_argument("--target_mse", type=float, default=15.68,
                    help="matches the operating point used throughout this "
                         "study, and the isotropic control's own MSE")
    ap.add_argument("--n_test", type=int, default=100,
                    help="held-out, place-disjoint queries never used for "
                         "fine-tuning or model selection")
    ap.add_argument("--n_val", type=int, default=50,
                    help="place-disjoint from both train and test; used only "
                         "for early-stopping checkpoint selection")
    ap.add_argument("--neg_k", type=int, default=8,
                    help="hard-negative candidates sampled per anchor; the "
                         "most confusable one (vs the current anchor "
                         "embedding) is used, at no extra forward-pass cost "
                         "since gallery embeddings are precomputed")
    ap.add_argument("--unfreeze_blocks", type=int, default=1,
                    help="how many of the last resnet blocks to fine-tune "
                         "(1 = layer4 only, 2 = layer3+layer4, ...); "
                         "must be 1 for mixvpr")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--train_perturbation", default="isotropic",
                    choices=("isotropic", "direction"),
                    help="what the attacker collected and adapts to: the "
                         "operating-point isotropic control (published "
                         "behaviour) or this paper's direction perturbation "
                         "(an adversary that anticipates the defense).")
    ap.add_argument("--direction_surrogates", nargs="+",
                    default=["resnet50", "vgg16", "cosplace"],
                    help="surrogate ensemble the collected direction "
                         "perturbation was optimised against.")
    ap.add_argument("--direction_objective", default="self",
                    choices=("self", "positive"),
                    help="'self' is the gallery-free deployable direction; "
                         "'positive' targets the query's correct gallery "
                         "entry and matches the originally published "
                         "transfer conditions.")
    ap.add_argument("--direction_steps", type=int, default=20)
    ap.add_argument("--direction_step_size", type=float, default=1.0)
    ap.add_argument("--direction_linf", type=float, default=16.0)
    ap.add_argument("--direction_random_start", type=float, default=None,
                    help="uniform displacement (pixel units) before the first "
                         "sign-gradient step; defaults to 1.0 for the 'self' "
                         "objective, whose clean frame is a stationary point, "
                         "and 0.0 for 'positive'.")
    ap.add_argument("--direction_cache", default="",
                    help="directory of cached direction-perturbed frames; "
                         "defaults to <output>.dircache. Reusable across "
                         "configurations, since the perturbation is "
                         "deterministic given frame and surrogates.")
    ap.add_argument("--output", required=True, help="checkpoint path (state_dict)")
    ap.add_argument("--test_ids_output", default="",
                    help="defaults to <output>.test_query_ids.json")
    args = ap.parse_args()
    if args.direction_random_start is None:
        args.direction_random_start = \
            1.0 if args.direction_objective == "self" else 0.0

    test_ids_path = Path(args.test_ids_output) if args.test_ids_output else \
        Path(str(args.output) + ".test_query_ids.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] device={device}", flush=True)

    records, gallery = load_manifest(args.manifest, args.root)
    gallery_ids = sorted(gallery)
    resize_hw = (args.height, args.width)
    gallery_tensor = torch.stack(
        [load_image(gallery[g]["path"], resize_hw) for g in gallery_ids])
    place_of = {g: gallery[g]["place_id"] for g in gallery_ids}

    train_records, val_records, test_records = split_queries(
        records, args.n_test, args.n_val, args.seed)
    print(f"[finetune] {len(train_records)} train / {len(val_records)} "
          f"validation / {len(test_records)} held-out test queries "
          f"(all mutually place-disjoint)", flush=True)

    cache_dir = Path(args.direction_cache) if args.direction_cache else \
        Path(str(args.output) + ".dircache")
    if args.train_perturbation == "direction":
        build_direction_cache(train_records + val_records, cache_dir, args,
                              gallery, gallery_ids, place_of, gallery_tensor,
                              resize_hw, device)

        def train_perturb(rec, frame):
            return cached_direction_perturb(rec, cache_dir)

        def val_perturb(rec, frame, index):
            return cached_direction_perturb(rec, cache_dir)
    else:
        def train_perturb(rec, frame):
            return isotropic_perturb(frame, args.target_mse,
                                     seed=rng.randrange(0, 2 ** 31))

        val_perturb = None

    place_to_indices: Dict[str, List[int]] = {}
    for i, g in enumerate(gallery_ids):
        place_to_indices.setdefault(place_of[g], []).append(i)
    all_places = list(place_to_indices)

    cfg = RetrievalConfig(backbone=args.backbone,
                          input_size=default_input_size_for_backbone(args.backbone))
    embedder = make_default_embedder(cfg).eval().to(device)

    with torch.no_grad():
        gal_emb = embed_gallery_batched(cfg, embedder, gallery_tensor).to(device)
    print(f"[finetune] fixed (pretrained-model) gallery embeddings ready",
          flush=True)

    blocks = trainable_submodule(embedder, args.backbone, args.unfreeze_blocks)
    trainable_params = [p for block in blocks for p in block.parameters()]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"[finetune] training {n_trainable} params in the last "
          f"{args.unfreeze_blocks} block(s)", flush=True)
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    init_val_acc, init_rank, init_mrr = validation_top1(
        embedder, cfg, gal_emb, gallery_ids, place_of,
        val_records, resize_hw, args.target_mse, device, val_perturb)
    print(f"[finetune] epoch 0 (pretrained, no fine-tuning) "
          f"val_top1={init_val_acc:.4f} val_median_rank={init_rank:.1f} "
          f"val_mrr={init_mrr:.5f}", flush=True)
    best_val_acc = init_val_acc
    best_epoch = 0
    best_state = copy.deepcopy(embedder.state_dict())

    rng = random.Random(args.seed)
    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(train_records)
        epoch_loss, epoch_n = 0.0, 0
        for i in range(0, len(train_records), args.batch_size):
            batch = train_records[i:i + args.batch_size]
            frames, pos_idx, neg_cand_idx = [], [], []
            for rec in batch:
                frame = load_image(rec["query_path"], resize_hw)
                frames.append(train_perturb(rec, frame))

                pos_place = rec["place_id"]
                pos_idx.append(rng.choice(place_to_indices[pos_place]))

                cands = []
                for _ in range(args.neg_k):
                    neg_place = rng.choice(all_places)
                    while neg_place == pos_place:
                        neg_place = rng.choice(all_places)
                    cands.append(rng.choice(place_to_indices[neg_place]))
                neg_cand_idx.append(cands)

            x = preprocess_for_embed(torch.stack(frames).to(device), cfg.input_size)
            anchor = F.normalize(embedder(x), dim=1)
            pos = gal_emb[torch.tensor(pos_idx, device=device)]

            cand_emb = gal_emb[torch.tensor(neg_cand_idx, device=device)]  # (B,K,D)
            with torch.no_grad():
                # Hardest of K random wrong-place candidates against the
                # *current* anchor embedding -- free, since gal_emb is
                # precomputed and fixed; forces the model to actually
                # discriminate confusable places instead of coasting past
                # easy random negatives (see module docstring).
                sims = torch.einsum("bd,bkd->bk", anchor.detach(), cand_emb)
                hard = sims.argmax(dim=1)
            neg = cand_emb[torch.arange(cand_emb.size(0), device=device), hard]

            pos_sim = (anchor * pos).sum(dim=1)
            neg_sim = (anchor * neg).sum(dim=1)
            loss = F.relu(args.margin - pos_sim + neg_sim).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(batch)
            epoch_n += len(batch)
            step += 1

        val_acc, val_rank, val_mrr = validation_top1(
            embedder, cfg, gal_emb, gallery_ids, place_of,
            val_records, resize_hw, args.target_mse, device, val_perturb)
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(embedder.state_dict())
            marker = " (best so far)"
        print(f"[finetune] epoch {epoch + 1}/{args.epochs} "
              f"mean_triplet_loss={epoch_loss / max(epoch_n, 1):.4f} "
              f"val_top1={val_acc:.4f} val_median_rank={val_rank:.1f} "
              f"val_mrr={val_mrr:.5f}{marker} (step {step})", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    with open(test_ids_path, "w", encoding="utf-8") as fh:
        json.dump([r["query_id"] for r in test_records], fh)
    print(f"[finetune] best epoch {best_epoch} (val_top1={best_val_acc:.4f}, "
          f"vs {init_val_acc:.4f} before fine-tuning) -> {args.output}",
          flush=True)
    print(f"[finetune] held-out test query_ids -> {test_ids_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
