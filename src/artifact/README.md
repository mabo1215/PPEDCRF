# Reproducibility Artifact

This bundle lets a reviewer confirm, without a GPU and without model weights,
that the paper's reported values were derived from the released raw experiment
outputs rather than transcribed by hand.

Two checkers exist and they reach different amounts, so both numbers are
stated here rather than one:

- `verify_claims.py`, shipped here, recomputes **228** claims from the exports
  in `results/`. That is what `run_verification.sh` runs, and the section
  "What the one-command check does and does not cover" below names the
  families it does not reach.
- `src/scripts/audit_claim_consistency.py`, in the source repository,
  recomputes **554** -- every cell of every table in the manuscript and the
  supplement, plus the prose values registered alongside them.

Both are registries, not sweeps. A number quoted only in the running text and
never registered is not checked: of the manuscript's 207 distinct three- and
four-decimal literals, 39 have no claim, most of them interval endpoints
printed beside a point estimate that is checked. Neither checker asserts that
the registry is complete, and this bundle does not claim it is.

It also carries `extended_evidence_report.pdf`, the extended evidence report
the manuscript cites wherever a result lives only there.

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

*The allocation axis*

- **Placement-rule study**: pooled Top-1 for all eight energy-matched
  placements on both checkpoints, including the attacker-gradient oracle and
  its anti-oracle.
- **High-budget reversal**: the three cluster-robust significant results in
  which edge placement beats uniform, with their bootstrap intervals.
- **Placement-rule concentration**: the top-decile weight share each rule
  expresses, which is what makes them different placements at all.
- **Real place-labelled MSLS placement null**: all eight placements plus the
  three published-model ones, each against the uniform control in its own run.
- **Placement by the margin gradient**: the strictly correct oracle, against
  uniform and against its anti-oracle.
- **Operators at matched delivered MSE**: four operators at two budgets on the
  real benchmark, uniform Top-1 and the edge-minus-uniform difference.
- **Operators in the controlled model**: the same comparison where the
  Jacobian is known exactly.
- **Attacker-sensitivity statistics**, on the proxy and replicated on real
  MSLS: gradient coefficient of variation, top-decile energy captured by the
  oracle and by the learned map, and their rank correlation.

*The direction axis*

- **Controlled study with a known Jacobian**: the displacement identity and
  the oracle's advantage at a given clean-task difficulty.
- **Direction transfer to a held-out attacker**, and the gallery-free variant
  that the headline table reports.
- **Non-adaptive preprocessing**, three seeds per cell, unhardened and under
  the gallery-free objective, and **the same directions hardened over those
  transforms (EOT)**.
- **Downstream utility**: per-image AP@50 and IoU on the same sanitized
  frames, clean / isotropic / direction.

*Carried over from the earlier framing*

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
MANIFEST.sha256            checksums for every file below
extended_evidence_report.pdf   the report the manuscript cites for results
                           that live only there
results/
  session3/                8-city primary manifest, white-box, gallery sweep,
                           deterministic blur/mosaic sweep
  session4/                8-city cross-time manifests, white-box on those,
                           finer deterministic sweep, merged sweep analysis
  placement_study/         eight energy-matched placements, released checkpoint
  placement_study_maskbacked/  the same, mask-supervised checkpoint
  placement_sigma_sweep/   placements across sigma_0 in {4,16,32,50}
  placement_highsigma_50pair/  high-budget runs at 50-pair scale
  placement_msls/          the same placements on real place-labelled MSLS,
                           plus the three published-model placements
  margin_oracle/           placement by the margin gradient
  operator_study/          four operators at two budgets, matched delivered MSE
  known_jacobian/          controlled retrieval with an exactly known Jacobian
  known_jacobian_operators/    the operator comparison in that model
  direction_transfer/      direction transfer to a held-out attacker
  direction_transfer_galleryfree/  the gallery-free variant, three seeds
  sanitize_3seed/          preprocessing robustness, three seeds per cell
  sanitize_galleryfree/    the same under the gallery-free objective
  direction_eot/           EOT-hardened directions, thirteen transforms,
                           three seeds concatenated per condition
  direction_utility/       per-image detection and segmentation utility, with
                           the detection box targets they are scored against
```

`build_artifact.sh` regenerates `results/`, the report copy and the manifest
from the local experiment exports. It is the only supported way to rebuild
them: several trees are renamed or assembled from several per-seed runs, and
the script encodes those rules.

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
- **What the one-command check does and does not cover.** `verify_claims.py`
  recomputes 228 claims, covering the allocation families, the operator study,
  the mask-guided comparison, the cross-time replication and the earlier
  transfer runs. It does **not** yet script the families added in the most
  recent round: `transfer_table4` (Table IV), `frontier_segmentation`,
  `release_boundary_mse*`, `transfer_patchnetvlad`, `jacobian_columns` and
  `adaptive_attacker_*`. Their raw per-query rows are included here in full, so
  every number the manuscript draws from them can be recomputed by hand, but
  the automated pass does not assert them and this artifact does not claim it
  does. In the source repository those families are asserted by
  `src/scripts/audit_claim_consistency.py`, which checks 554 manuscript claims
  against the same rows; it is not run here because it resolves trees by their
  repository names rather than the reader-facing names used in `results/`.
