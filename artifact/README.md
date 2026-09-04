# PPEDCRF — Anonymous Reproducibility Artifact

This bundle lets a reviewer confirm, without a GPU and without model weights,
that every number in the paper's tables was derived from the released raw
experiment outputs rather than transcribed by hand.

## One command

```bash
bash run_verification.sh
```

That installs the two verification dependencies, checks the integrity of the
released exports against `MANIFEST.sha256`, and recomputes each reported value
from the per-query CSVs, printing `OK` / `FAIL` per claim. Exit code 0 means
every claim reproduced.

Runtime: well under a minute. No GPU, no network, no dataset download, and no
model checkpoint are required for this path.

## What is checked

`verify_claims.py` recomputes, straight from the raw per-query exports:

- **Wider 8-city primary manifest** (`all8`, 400 queries / 2,000-image
  gallery): raw and sanitized Top-1 for all six attacker backbones.
- **8-city cross-time subtasks** (`o2n8`, `n2o8`): the same six backbones on
  both official cross-time directions, 12 cells.
- **White-box attacker** (separate threat model, reported separately in the
  paper): attacker-aware Top-1 on all three 8-city manifests.
- **Deterministic-baseline matched-PSNR sweep**: achieved PSNR and Top-1 at
  each swept blur kernel / mosaic block size used in the matched comparison.
- **Matched-PSNR significance**: that every paired comparison at gallery sizes
  12 and 100 has zero discordant pairs and no significant result, as claimed.

Top-1 is recomputed as the fraction of queries whose correct gallery item is
ranked first, from the `correct_rank` column of the raw export — the same
definition used in the paper.

## Layout

```
run_verification.sh        one-command entry point
verify_claims.py           raw-output-to-table verifier
requirements-pinned.txt    pinned dependency versions (run environment)
MANIFEST.sha256            checksums for every released export in results/
results/
  session3/                8-city primary manifest, white-box, gallery sweep,
                           deterministic blur/mosaic sweep
  session4/                8-city cross-time manifests, white-box on those,
                           finer deterministic sweep, merged sweep analysis
```

All paths inside the bundle are relative; nothing refers to an absolute
location on the authors' machines.

## Scope and honesty notes

- This artifact verifies **table derivation**, not end-to-end retraining. The
  benchmarks themselves were executed on an RTX 3090; re-running them requires
  the pinned `torch` stack, the MSLS imagery (obtained from its official
  source under its own terms), and the released checkpoint. The datasets are
  not redistributed here.
- The exports include runs whose result is **negative or adverse** for the
  proposed method (for example, the ResNet18 cross-time cells where
  sanitization slightly increases retrieval accuracy, and the white-box
  attacker results). These are deliberately included: the verifier checks them
  on the same footing as the favourable numbers.
- `results/` contains only CSV/JSON exports and status files. Image data and
  model weights are excluded for size and licensing reasons.
