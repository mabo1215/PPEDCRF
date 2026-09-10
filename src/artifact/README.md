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
  recomputes **780** claims, drawn from the manuscript, the supplement, the
  extended evidence report and ten of the generated tables -- among them
  every cell of the thirteen-transform preprocessing table, which is the sole
  support for the manuscript's held-out-transform sentence.

Both are registries, not sweeps. Neither enumerates the manuscript and then
demands a claim for every number it finds; each asserts the numbers someone
registered, so a value quoted only in the running text and never registered
is not checked. That gap is measured rather than asserted. Running

```bash
python3 src/scripts/audit_claim_consistency.py --coverage
```

last printed

```
coverage over main.tex and supplementary.tex: 260 distinct three- and four-decimal literals, 231 registered, 29 not.
```

which matches each literal against every claim's locator, its printed form
and its registered value rounded to the literal's own precision, and then
lists the twenty-nine by value. They are interval endpoints, p-values quoted
inline, and running-text restatements of studies whose tables are themselves
registered. Neither checker asserts that the registry is complete, and this
bundle does not claim it is. The block above is a transcript, not a hand
count: rerun the command whenever the manuscript changes and paste what it
prints.

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
- **Non-adaptive preprocessing on the four transforms the hardening trains
  on** (JPEG-75, JPEG-50, blur, denoise), three seeds per cell, unhardened and
  under the gallery-free objective, and **the same four hardened over (EOT)**,
  there with the untouched control alongside them. The eight further
  transforms the hardening never saw are released in
  `preprocessing_13transform/`, but this checker does not recompute them.
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
  sanitize_3seed/          preprocessing robustness on the four trained
                           transforms, three seeds per cell
  sanitize_galleryfree/    the same four under the gallery-free objective
  direction_eot/           EOT-hardened directions on those four transforms
                           and the untouched control, three seeds
                           concatenated per condition
  preprocessing_13transform/   the whole thirteen-transform study behind the
                           supplement's transform table, including the eight
                           transforms the hardening never saw: one unhardened
                           run and three hardened seeds per attacker, with
                           every transform, the isotropic control and the
                           white-box bound in each file
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
  recent rounds: `transfer_table4` (Table III, whose rows are also the
  `preprocessing_13transform` release), `frontier_segmentation`,
  `release_boundary_mse*`, `transfer_patchnetvlad`, `jacobian_columns`,
  `adaptive_attacker_*`, and the two attackers added last -- `clip_pooling_*`
  (an attacker holding the whole clip) and `purification` (an attacker that
  trains a denoiser to invert the release) -- together with `transfer_vit`,
  `crossdataset_kitti360`, `segmenter_spread*` and the second-seed frontier
  trees. Their raw per-query rows are included here in full, so every number
  the manuscript draws from them can be recomputed by hand, but the automated
  pass does not assert them and this artifact does not claim it does. In the
  source repository those families are asserted by
  `src/scripts/audit_claim_consistency.py`, which checks 780 manuscript claims
  against the same rows; it is not run here because it resolves trees by their
  repository names rather than the reader-facing names used in `results/`.
