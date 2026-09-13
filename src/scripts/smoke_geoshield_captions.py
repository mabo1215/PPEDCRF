"""Exercise the caption path on real frames before any GPU time is bought.

The second GeoShield arm replaces the released constant-caption stub with a
vision--language model, and that arm is only worth launching if the captions
are per-image, location-free and cached. All three are checkable on a laptop
against a handful of real MSLS frames, which is cheaper than discovering a
malformed request after the card is running.

Checks, in order: the key resolves; the API answers; two different frames get
two different captions (the failure this arm exists to avoid is silently
reproducing the constant the release ships); the cache round-trips so a resume
pays nothing; and no caption names a city or country, which the prompt forbids
because a location-naming caption would change what the objective subtracts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from scripts.run_geoshield_audit import (  # noqa: E402
    PUBLISHED_CAPTION,
    CaptionSource,
    read_api_key,
)

# Cities the manifest draws from; a caption naming one would mean the prompt
# failed to hold the model off the location, which is the one thing the
# geo-semantic term must not be handed for free.
FORBIDDEN = ["amman", "jordan", "manila", "philippines", "toronto", "canada",
             "boston", "usa", "united states", "melbourne", "australia",
             "budapest", "hungary", "zurich", "switzerland", "paris", "france",
             "tokyo", "japan", "london", "england", "moscow", "russia"]


def load_frame(path: Path) -> torch.Tensor:
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert("RGB").resize((320, 192))
    arr = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()
    return arr.unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", nargs="+", required=True)
    ap.add_argument("--env", default="C:/source/.env")
    ap.add_argument("--env_label", default="chatgpt-api")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--cache", default=str(REPO / "src" / "tmp" /
                                           "geoshield_caption_smoke.json"))
    args = ap.parse_args()

    key = read_api_key(Path(args.env), args.env_label)
    print(f"[key] resolved under {args.env_label!r}: len={len(key)} "
          f"prefix={key[:7]!r} (value withheld)")

    cache = Path(args.cache)
    if cache.is_file():
        cache.unlink()
    src = CaptionSource("openai", cache, key, args.model)

    captions = {}
    for path in args.frames:
        p = Path(path)
        frame = load_frame(p)
        caption = src(frame, p.stem)
        captions[p.stem] = caption
        print(f"\n[{p.parent.parent.parent.name}/{p.stem[:16]}]\n  {caption}")

    failures = []
    if len(set(captions.values())) < len(captions):
        failures.append("captions are not distinct across frames -- this arm "
                        "would reproduce the released constant")
    for name, caption in captions.items():
        if caption.strip() == PUBLISHED_CAPTION:
            failures.append(f"{name}: returned the released constant caption")
        hit = [w for w in FORBIDDEN if w in caption.lower()]
        if hit:
            failures.append(f"{name}: names a location ({', '.join(hit)})")
        if len(caption.split()) < 5:
            failures.append(f"{name}: caption is too short to carry content")

    reloaded = CaptionSource("openai", cache, key, args.model)
    if reloaded.cache != src.cache:
        failures.append("cache did not round-trip; a resume would re-pay")
    else:
        print(f"\n[cache] {len(reloaded.cache)} captions round-tripped from "
              f"{cache}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print(f"PASS  {len(captions)} distinct, location-free captions; cache "
          f"round-trips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
