"""Run the public GeoShield release through this audit's protocol.

The manuscript's negative claim is about the rules it enumerates, and the
review asks for at least one *published* mechanism carried end to end. This
runs GeoShield (AAAI 2026, `thinwayliu/Geoshield`) on the same place-labelled
manifest, at the same delivered distortion, against the same attacker panel.

Three properties of the public release decide what this can measure, and all
three were established by reading the released source rather than inferred:

1. `describe_image_placeholder()` returns the constant string
   ``"A scenic outdoor photograph."`` for every image. It does not raise and
   it is not guarded, so the released code runs to completion with the same
   text embedding for every frame, and the geo-semantic term its objective
   subtracts carries no per-image signal. `--caption_source published` runs
   that release as published; `--caption_source claude` fills the stub the
   way the release's own docstring instructs, which is a different objective
   and is labelled as such.
2. `bbox_json_path` defaults to empty, and with no detections the code
   appends a full-frame box and perturbs the whole image, so the region
   module is off by default. Unlike the text term it is publicly
   recoverable: the README documents GroundingDINO with released weights.
   Auditing a mechanism with a module it documents how to enable would be a
   straw man, so `--bbox_json` turns it on.
3. In the untargeted mode the target images are load-bearing --- the attack
   descends similarity to a masked crop of the *target* --- while the README
   says target images are "for M-Attack only". The untargeted target set is
   therefore undefined by the release, and `--target_dir` makes the choice
   explicit rather than silent.

Nothing under `src/third_party/Geoshield` is edited. This module imports the
released attack and adapts only at the boundary: it feeds frames in and
rescales what comes out to the protocol's delivered MSE, solved per frame by
bisection through the pixel clamp exactly as every other arm is.

Rows are written one per completed (query, condition, seed), flushed and
fsynced, and a restart skips what is already on disk.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
REPO_SRC = REPO / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from eval.retrieval_attack import (  # noqa: E402
    RetrievalConfig,
    default_input_size_for_backbone,
    make_default_embedder,
)
from scripts.run_geotagged_vpr_benchmark import (  # noqa: E402
    load_image,
    load_manifest,
)
from scripts.run_direction_transfer_study import (  # noqa: E402
    embed_gallery_batched,
    normalised_embedding,
    release_at_mse,
)

GEOSHIELD = REPO / "src" / "third_party" / "Geoshield"

# The caption the released stub returns for every image. Asserted at startup:
# if a future checkout returns something else, this is no longer "the release
# as published" and the label has to change with it.
PUBLISHED_CAPTION = "A scenic outdoor photograph."

CAPTION_PROMPT = (
    "Describe the visual content of this photograph in one sentence: the "
    "objects, scene type and activities. Do not name or guess the location, "
    "the city, the country or any landmark by name."
)


# --------------------------------------------------------------------------
# the released code, imported rather than copied
# --------------------------------------------------------------------------
def import_geoshield():
    """Import the released module without editing it.

    One packaging defect has to be stepped around to get that far.
    `config_schema.MainConfig` annotates `data`, `optim`, `model` and `wandb`
    with non-Optional types and gives each a default of `None`; importing
    `geoshield` runs `cs.store(node=MainConfig)`, and current OmegaConf
    rejects that with "field 'data' is not Optional" before any attack code
    is reached. `__post_init__` would have filled them in, but store()
    validates the declared defaults first. The README points at a
    `requirements.txt` for pinned versions and the repository does not ship
    one, so there is no version to fall back to.

    ConfigStore registration is only used by Hydra's command-line resolution.
    This harness builds the config itself and never resolves one through
    Hydra, so the registration is disabled for the duration of the import and
    restored immediately. Nothing in the attack path is touched, and the
    objective is bit-for-bit the released one.
    """
    if not GEOSHIELD.is_dir():
        raise SystemExit(
            f"GeoShield checkout not found at {GEOSHIELD}. Clone it there:\n"
            f"  git clone https://github.com/thinwayliu/Geoshield.git "
            f"{GEOSHIELD}")
    if str(GEOSHIELD) not in sys.path:
        sys.path.insert(0, str(GEOSHIELD))

    from hydra.core.config_store import ConfigStore
    original = ConfigStore.store
    ConfigStore.store = lambda self, *a, **k: None

    # GeoShield imports a top-level `utils`, and so does this repository. By
    # the time we get here ours is already in sys.modules, so the released
    # `from utils import hash_training_config` would resolve to ours and fail.
    # Shadow the colliding names for the duration of the import and put them
    # back, so neither side sees the other's module.
    shadowed = {name: sys.modules.pop(name)
                for name in ("utils", "surrogates", "config_schema")
                if name in sys.modules}
    saved_path = list(sys.path)
    sys.path.insert(0, str(GEOSHIELD))
    try:
        import geoshield as gs  # noqa: E402
    finally:
        ConfigStore.store = original
        sys.path[:] = saved_path
        for name in ("utils", "surrogates", "config_schema"):
            sys.modules.pop(name, None)
        sys.modules.update(shadowed)
    return gs


def assert_published_stub(gs) -> None:
    caption = gs.describe_image_placeholder("")
    if caption != PUBLISHED_CAPTION:
        raise SystemExit(
            "The released stub no longer returns the published constant "
            f"caption ({caption!r} against {PUBLISHED_CAPTION!r}). A run "
            "against this checkout is no longer of the release as published; "
            "update the label before continuing.")


def third_party_commit() -> Optional[str]:
    head = GEOSHIELD / ".git" / "HEAD"
    if not head.is_file():
        return None
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref: "):
        target = GEOSHIELD / ".git" / ref[5:]
        return (target.read_text(encoding="utf-8").strip()
                if target.is_file() else None)
    return ref


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------
def read_api_key(env_path: Path, label: str) -> str:
    """The value on the line after `label:` in the credentials file.

    Read transiently and never logged: the key is passed to the API and not
    written to any export, metadata file or progress note.
    """
    lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().rstrip(":") == label.lower():
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    return nxt.strip()
    raise SystemExit(f"no value found under {label!r} in {env_path}")


class CaptionSource:
    """Per-image captions, cached on disk so a resume pays no API cost.

    `openai` is the default non-published source because the release's own
    stub docstring names "OpenAI GPT-4V API" first among the models a user is
    told to plug in, and `gpt-4o` is that model's successor -- supplying the
    component the authors pointed at is a narrower substitution than picking
    a different vendor's model.
    """

    def __init__(self, mode: str, cache: Path, api_key: Optional[str] = None,
                 model: str = "gpt-4o"):
        self.mode = mode
        self.cache_path = cache
        self.api_key = api_key
        self.model = model
        self._sdk = None
        self.cache: Dict[str, str] = {}
        if cache.is_file():
            self.cache = json.loads(cache.read_text(encoding="utf-8"))

    def __call__(self, image: torch.Tensor, key: str) -> str:
        if self.mode == "published":
            return PUBLISHED_CAPTION
        if key in self.cache:
            return self.cache[key]
        caption = self._describe(image)
        self.cache[key] = caption
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1) + "\n",
                                   encoding="utf-8", newline="\n")
        return caption

    @staticmethod
    def _png_b64(image: torch.Tensor) -> str:
        from PIL import Image
        arr = (image.detach().clamp(0, 255).byte().cpu()
               .squeeze(0).permute(1, 2, 0).numpy())
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _describe(self, image: torch.Tensor) -> str:
        caption = (self._describe_openai(image) if self.mode == "openai"
                   else self._describe_claude(image))
        if not caption:
            raise SystemExit(
                "the caption model returned nothing; aborting rather than "
                "falling back to the released constant caption, which would "
                "silently mislabel this arm as the published one")
        return caption

    def _describe_openai(self, image: torch.Tensor) -> str:
        if self._sdk is None:
            from openai import OpenAI
            self._sdk = OpenAI(api_key=self.api_key)
        result = self._sdk.chat.completions.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{self._png_b64(image)}"}},
            ]}],
        )
        return (result.choices[0].message.content or "").strip()

    def _describe_claude(self, image: torch.Tensor) -> str:
        """Kept as an alternative source; `max_tokens` covers thinking too.

        Thinking is on by default on this model family and shares the
        `max_tokens` ceiling with the response text, so a caption-sized budget
        returns an empty or truncated text block rather than an error.
        """
        if self._sdk is None:
            import anthropic
            self._sdk = anthropic.Anthropic(api_key=self.api_key)
        message = self._sdk.messages.create(
            model=self.model,
            max_tokens=1000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": self._png_b64(image)}},
                {"type": "text", "text": CAPTION_PROMPT},
            ]}],
        )
        if message.stop_reason == "refusal":
            raise SystemExit(
                "the caption model declined a frame; aborting rather than "
                "falling back to the released constant caption")
        return "".join(b.text for b in message.content
                       if b.type == "text").strip()


# --------------------------------------------------------------------------
# the attack, called as released
# --------------------------------------------------------------------------
def attack_one(gs, args, frame: torch.Tensor, target: torch.Tensor,
               caption: str, bbox_dict: Dict, img_name: str, img_index: int,
               ensemble_extractor, ensemble_loss, source_crop, target_crop,
               device) -> torch.Tensor:
    """One frame through the released `fgsm_attack_masked`, unmodified.

    The box list and the area-weighted `probs` are built exactly as the
    released `attack_imgpair` builds them, including its append of a
    full-frame box, so a run with no detections reproduces the shipped
    default where the whole frame is the region.
    """
    if img_name in bbox_dict:
        img_size = bbox_dict[img_name]["size"]
        boxes = list(bbox_dict[img_name]["boxes"])
    else:
        img_size = (frame.shape[2], frame.shape[3])
        boxes = []
    boxes.append([0, 0, img_size[1], img_size[0]])

    areas = torch.stack([
        torch.sum(gs.bbox_to_mask(box, img_size, args.input_res,
                                  str(device))).float()
        for box in boxes])
    total = areas.sum() + 1e-8
    probs = [(a / total) for a in areas]

    with torch.enable_grad():
        adv = gs.fgsm_attack_masked(
            cfg=args.cfg, ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss, source_crop=source_crop,
            target_crop=target_crop, img_index=img_index, image_org=frame,
            image_tgt=target, boxes=boxes, image_size=img_size, probs=probs,
            description=caption)
    adv = adv.detach()
    # The released attack returns the frame divided by 255 and clamped to
    # [0,1] on its final line; everything downstream here is in pixel units.
    return adv * 255.0 if float(adv.max()) <= 1.0 else adv


def score(embedder, size, released, frame, gallery_emb, gallery_ids, place_of,
          want) -> Dict[str, object]:
    with torch.no_grad():
        qe = normalised_embedding(embedder, released, size)
        sims = gallery_emb @ qe.flatten()
        order = torch.argsort(sims, descending=True)
        ranked = [place_of[gallery_ids[j]] for j in order.tolist()]
        rank = next(i + 1 for i, p in enumerate(ranked) if p == want)
    mse = float((released - frame).square().mean())
    delta = released - frame
    return {
        "correct_rank": rank,
        "top5_hit": int(rank <= 5),
        "top10_hit": int(rank <= 10),
        "effective_mse": f"{mse:.6f}",
        "psnr": f"{10 * np.log10(255.0 ** 2 / max(mse, 1e-9)):.4f}",
        "max_abs_delta": f"{float(delta.abs().max()):.4f}",
        "clipped_fraction": f"{float(((released <= 0) | (released >= 255)).float().mean()):.6f}",
    }


def completed_keys(path: Path) -> set:
    if not path.is_file():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["query_id"], r["condition"], r["seed"])
                for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--eval_backbone", default="resnet18")
    ap.add_argument("--target_mse", type=float, default=15.68)
    ap.add_argument("--bbox_json", default="")
    ap.add_argument("--caption_source", default="published",
                    choices=["published", "openai", "claude"])
    ap.add_argument("--caption_model", default="",
                    help="defaults to gpt-4o for openai, claude-opus-5 for "
                         "claude; gpt-4o is the successor to the GPT-4V the "
                         "release's own stub names first")
    ap.add_argument("--env", default="C:/source/.env")
    ap.add_argument("--env_label", default="",
                    help="defaults to chatgpt-api / claude-code-api to match "
                         "--caption_source")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--epsilon", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--backbone", nargs="+", default=["B16", "B32", "Laion"])
    ap.add_argument("--input_res", type=int, default=640)
    ap.add_argument("--crop_scale", type=float, nargs=2, default=[0.5, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234])
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gallery_batch", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="import, assert the stub and exit; needs no GPU")
    args = ap.parse_args()

    gs = import_geoshield()
    assert_published_stub(gs)
    print(f"[geoshield] released stub returns {PUBLISHED_CAPTION!r} for every "
          f"frame", flush=True)
    if args.smoke:
        print(f"[geoshield] smoke: import and stub assertion passed; "
              f"commit={third_party_commit()}", flush=True)
        return 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records, gallery = load_manifest(args.manifest, args.root)
    if args.limit:
        records = records[:args.limit]

    bbox_dict = (gs.load_bboxes(args.bbox_json)
                 if args.bbox_json and os.path.exists(args.bbox_json) else {})
    region = "grounded" if bbox_dict else "fullframe"
    condition = f"geoshield_{args.caption_source}_{region}"
    print(f"[geoshield] {len(records)} queries, region={region}, "
          f"captions={args.caption_source}", flush=True)

    default_model = {"openai": "gpt-4o", "claude": "claude-opus-5"}
    default_label = {"openai": "chatgpt-api", "claude": "claude-code-api"}
    caption_model = args.caption_model or default_model.get(
        args.caption_source, "")
    env_label = args.env_label or default_label.get(args.caption_source, "")
    key = (read_api_key(Path(args.env), env_label)
           if args.caption_source != "published" else None)
    captions = CaptionSource(
        args.caption_source,
        Path(args.output).parent / f"captions_{args.caption_source}.json",
        key, caption_model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(out_path)
    if done:
        print(f"[geoshield] resuming; {len(done)} rows on disk", flush=True)

    fields = ["query_id", "condition", "seed", "correct_rank", "top5_hit",
              "top10_hit", "effective_mse", "psnr", "max_abs_delta",
              "clipped_fraction", "region_mode", "caption_source", "steps",
              "epsilon"]
    fresh = not out_path.is_file()
    stream = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    if fresh:
        writer.writeheader()
        stream.flush()
        os.fsync(stream.fileno())

    # Two different sizes, which the first cut conflated: `resize_hw` is the
    # frame geometry every arm loads at, while `cfg.input_size` is the square
    # resolution the embedder resamples to internally.
    cfg = RetrievalConfig(backbone=args.eval_backbone)
    cfg.input_size = default_input_size_for_backbone(args.eval_backbone)
    resize_hw = (args.height, args.width)
    embed_size = cfg.input_size
    embedder = make_default_embedder(cfg).eval().to(device)
    gallery_ids = sorted(gallery)
    place_of = {g: gallery[g]["place_id"] for g in gallery_ids}
    gal = embed_gallery_batched(
        cfg, embedder,
        torch.stack([load_image(gallery[g]["path"], resize_hw)
                     for g in gallery_ids]),
        args.gallery_batch)
    gal = (gal / gal.norm(dim=-1, keepdim=True).clamp_min(1e-12)).to(device)
    print("[geoshield] gallery embedded", flush=True)

    args.cfg = gs.OmegaConf.create({
        "data": {"batch_size": 1},
        "optim": {"alpha": args.alpha, "epsilon": args.epsilon,
                  "steps": args.steps},
        "model": {"input_res": args.input_res, "device": str(device),
                  "ensemble": True, "backbone": list(args.backbone),
                  "use_source_crop": True, "use_target_crop": True,
                  "crop_scale": list(args.crop_scale)},
    })
    extractor, models = gs.get_models(args.cfg)
    loss = gs.get_ensemble_loss(args.cfg, models)
    import torchvision.transforms as T
    source_crop = T.RandomResizedCrop(args.input_res, scale=args.crop_scale)
    target_crop = T.RandomResizedCrop(args.input_res, scale=args.crop_scale)

    (out_path.parent / "run_metadata.json").write_text(json.dumps({
        "released_caption": PUBLISHED_CAPTION,
        "caption_source": args.caption_source,
        "caption_model": caption_model or None,
        "caption_is_constant_across_corpus": args.caption_source == "published",
        "region_mode": region,
        "steps": args.steps, "epsilon": args.epsilon,
        "clip_ensemble": list(args.backbone),
        "target_mse": args.target_mse,
        "third_party_commit": third_party_commit(),
        "label": ("public release as published; geo-semantic term degenerate"
                  if args.caption_source == "published"
                  else f"GeoShield with {caption_model} filling the released "
                       f"VLM stub; not the released implementation"),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    written = 0
    for qi, rec in enumerate(records, 1):
        qid = rec["query_id"]
        want = rec["place_id"]
        if not any(place_of[g] == want for g in gallery_ids):
            continue
        frame = load_image(rec["query_path"], resize_hw).unsqueeze(0).to(device) * 255.0
        for seed in args.seeds:
            if (qid, condition, str(seed)) in done:
                continue
            # crc32, not hash(): Python randomises string hashing per
            # process, which would make the target choice irreproducible.
            pick = zlib.crc32(f"{qid}|{seed}".encode()) % len(gallery_ids)
            # The untargeted mode's target set is undefined by the release,
            # so it is fixed here to a gallery frame from a different place
            # and recorded, rather than left to directory order.
            while place_of[gallery_ids[pick]] == want:
                pick = (pick + 1) % len(gallery_ids)
            target = load_image(gallery[gallery_ids[pick]]["path"],
                                resize_hw).unsqueeze(0).to(device) * 255.0
            torch.manual_seed(seed)
            np.random.seed(seed)
            # The release captions `image_tgt`, not the source frame, and
            # the target is a deterministic function of (query, seed) -- so
            # the cache key is the target's own gallery id, which collapses
            # repeats across queries that draw the same target.
            caption = captions(target, gallery_ids[pick])
            adv = attack_one(gs, args, frame, target, caption, bbox_dict,
                             os.path.basename(str(rec["query_path"])), qi,
                             extractor, loss, source_crop, target_crop, device)
            released = release_at_mse(frame, adv - frame, args.target_mse)
            row = score(embedder, embed_size, released, frame, gal, gallery_ids,
                        place_of, want)
            row.update({"query_id": qid, "condition": condition, "seed": seed,
                        "region_mode": region,
                        "caption_source": args.caption_source,
                        "steps": args.steps, "epsilon": args.epsilon})
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
            written += 1
        if qi % 25 == 0:
            print(f"[geoshield] {qi}/{len(records)} queries, {written} rows",
                  flush=True)
    stream.close()
    print(f"[geoshield] done; {written} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
