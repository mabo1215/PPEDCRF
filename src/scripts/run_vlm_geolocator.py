"""Does the direction result survive a geolocator that is not a CLIP retriever?

The fourteenth review's second unscheduled item asks for a stronger
image-to-GPS attacker than GeoCLIP. The model it names, PIGEON, has no public
weights, so this substitutes the strongest geolocator that can actually be
obtained: a frontier vision-language model prompted for a coordinate.

That substitution buys more than strength. GeoCLIP was chosen precisely
because its backbone is one of the five retrieval attackers this paper already
reports, so a difference between the retrieval and geolocation settings was a
difference in the head, not in the features. The same property is a weakness
for this question: a defense tuned against CLIP features might fail only
against CLIP-derived geolocators. A generative model with a different
architecture, different training data and no gallery at all tests whether the
direction result is a fact about the perturbation or a fact about CLIP.

The scoring is unchanged from the GeoCLIP arm -- great-circle error to the
query's own coordinate, summarised at 1 km, 25 km and 200 km -- so the two
arms are directly comparable.

Two safeguards. The frames are read from disk as released, byte-identical to
what GeoCLIP scored, rather than re-rendered here. And the model must clear a
capability gate on clean frames before any conditioned result is reported: a
geolocator weaker than GeoCLIP cannot answer "does this survive a stronger
attacker", and an arm that skipped that check could report a reassuring null
that means only that the attacker was bad.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]

PROMPT = (
    "You are given a single photograph. Estimate where on Earth it was taken. "
    "Reply with ONLY a JSON object of the form "
    '{"lat": <decimal degrees>, "lon": <decimal degrees>, "confidence": <0-1>}. '
    "Give your single best point estimate even if you are unsure. Do not "
    "refuse, do not explain, do not add any text outside the JSON."
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))


def load_key(env_path: str, label: str) -> str:
    """Read the key from the operator's env file without echoing it."""
    text = Path(env_path).read_text(encoding="utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines()]
    for i, l in enumerate(lines):
        if l.lower().startswith(label.lower()) and i + 1 < len(lines):
            for cand in lines[i + 1:]:
                if cand and not cand.lower().startswith(label.lower()):
                    return cand.split()[0]
    raise SystemExit(f"no key found under label {label!r} in {env_path}")


def parse_coord(text: str) -> Optional[Tuple[float, float]]:
    """A coordinate from the model's reply, or None if it did not give one.

    Returning None rather than a default is deliberate: a refusal scored as
    (0,0) would land in the Gulf of Guinea and be counted as a ~5000 km error,
    which reads in the table as a defense that worked spectacularly. Refusals
    are counted separately instead.
    """
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            return float(o["lat"]), float(o["lon"])
        except Exception:
            pass
    m = re.search(r"(-?\d+\.?\d*)\s*[,;]\s*(-?\d+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except Exception:
            return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True,
                    help="directory of released frames from the geolocator arm")
    ap.add_argument("--truth", required=True,
                    help="the GeoCLIP arm's CSV, for true coordinates and for "
                         "the capability comparison")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--env", default="C:/source/.env")
    ap.add_argument("--env_label", default="chatgpt-api")
    ap.add_argument("--conditions", nargs="+",
                    default=["clean", "isotropic", "direction"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gate_within_km", type=float, default=25.0)
    ap.add_argument("--gate_beat", type=float, default=0.545,
                    help="GeoCLIP's share within the gate distance; the VLM "
                         "must beat this on clean frames to count as stronger")
    ap.add_argument("--gate_only", action="store_true",
                    help="score clean frames only, report the gate and stop")
    args = ap.parse_args()

    truth: Dict[str, Tuple[float, float]] = {}
    geoclip_clean: List[float] = []
    with open(args.truth, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            truth[r["query_id"]] = (float(r["true_lat"]), float(r["true_lon"]))
            if r["condition"] == "clean":
                geoclip_clean.append(float(r["error_km"]))

    frames = sorted(Path(args.frames).glob("*.png"))
    if not frames:
        raise SystemExit(f"no frames in {args.frames}")

    want = set(args.conditions)
    jobs = []
    for p in frames:
        stem = p.stem
        try:
            qid, cond, seed = stem.rsplit("__", 2)
        except ValueError:
            continue
        if cond not in want or qid not in truth:
            continue
        jobs.append((p, qid, cond, seed.lstrip("s")))
    jobs.sort(key=lambda j: (j[2], j[1]))
    if args.gate_only:
        jobs = [j for j in jobs if j[2] == "clean"]
    if args.limit:
        seen: Dict[str, int] = {}
        kept = []
        for j in jobs:
            seen[j[2]] = seen.get(j[2], 0) + 1
            if seen[j[2]] <= args.limit:
                kept.append(j)
        jobs = kept
    print(f"[vlm] {len(jobs)} frames, model={args.model}", flush=True)

    from openai import OpenAI
    client = OpenAI(api_key=load_key(args.env, args.env_label))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done.add((r["query_id"], r["condition"], r["seed"]))
        print(f"[vlm] resuming, {len(done)} rows present", flush=True)

    fields = ["query_id", "condition", "seed", "pred_lat", "pred_lon",
              "true_lat", "true_lon", "error_km", "refused", "model"]
    new = not out.exists()
    errs: Dict[str, List[float]] = {}
    refusals: Dict[str, int] = {}
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for n, (path, qid, cond, seed) in enumerate(jobs, 1):
            if (qid, cond, seed) in done:
                continue
            b64 = base64.b64encode(path.read_bytes()).decode()
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url":
                            {"url": f"data:image/png;base64,{b64}"}}]}],
                )
                text = resp.choices[0].message.content or ""
            except Exception as e:                      # noqa: BLE001
                print(f"[vlm] {qid}/{cond}: API error {e}", flush=True)
                continue
            coord = parse_coord(text)
            tlat, tlon = truth[qid]
            if coord is None:
                refusals[cond] = refusals.get(cond, 0) + 1
                row = {"query_id": qid, "condition": cond, "seed": seed,
                       "pred_lat": "", "pred_lon": "",
                       "true_lat": f"{tlat:.6f}", "true_lon": f"{tlon:.6f}",
                       "error_km": "", "refused": 1, "model": args.model}
            else:
                d = haversine_km(coord[0], coord[1], tlat, tlon)
                errs.setdefault(cond, []).append(d)
                row = {"query_id": qid, "condition": cond, "seed": seed,
                       "pred_lat": f"{coord[0]:.6f}", "pred_lon": f"{coord[1]:.6f}",
                       "true_lat": f"{tlat:.6f}", "true_lon": f"{tlon:.6f}",
                       "error_km": f"{d:.4f}", "refused": 0, "model": args.model}
            w.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
            if n % 25 == 0:
                print(f"[vlm] {n}/{len(jobs)}", flush=True)

    # The gate. Reported whether or not it passes -- a failed gate is a result
    # about the substitute attacker, not a reason to quietly drop the arm.
    #
    # Read back from the output file rather than from the in-run accumulator.
    # The accumulator holds only what this invocation scored, so on a resumed
    # run the gate silently described a subset: a 400-frame gate resumed at
    # 162 reported n=238 and never said so. It agreed on that occasion, which
    # is exactly why it needed fixing -- a subset gate can pass or fail for
    # reasons that have nothing to do with the attacker.
    final: Dict[str, List[float]] = {}
    with open(out, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("refused", "0")) == "1" or not r.get("error_km"):
                continue
            final.setdefault(r["condition"], []).append(float(r["error_km"]))
    refusals = {}
    with open(out, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("refused", "0")) == "1":
                refusals[r["condition"]] = refusals.get(r["condition"], 0) + 1
    errs = final
    clean = sorted(errs.get("clean", []))
    if clean:
        within = sum(1 for d in clean if d <= args.gate_within_km) / len(clean)
        med = clean[len(clean) // 2]
        gc = sorted(geoclip_clean)
        gc_within = (sum(1 for d in gc if d <= args.gate_within_km) / len(gc)
                     if gc else float("nan"))
        gc_med = gc[len(gc) // 2] if gc else float("nan")
        print(f"\n[gate] clean frames, n={len(clean)}"
              f" (refusals {refusals.get('clean', 0)})")
        print(f"[gate]   VLM     within {args.gate_within_km:g} km: {within:.3f}"
              f"   median {med:.1f} km")
        print(f"[gate]   GeoCLIP within {args.gate_within_km:g} km: "
              f"{gc_within:.3f}   median {gc_med:.1f} km")
        if within > args.gate_beat:
            print("[gate]   PASS -- stronger than GeoCLIP; the arm answers N2")
        else:
            print("[gate]   FAIL -- not stronger than GeoCLIP. Report this as "
                  "a bound on the substitute, not as N2 closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
