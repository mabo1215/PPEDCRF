# Committed experiment exports

`src/outputs/` is gitignored, so an export tree normally exists only on the one
machine that produced it. That is fine until the machines diverge, and on
9 September 2026 they did: the work machine held the newest families and had
concluded the older ones were lost, while the home machine held the older ones
and none of the new. Neither machine could reproduce the full manuscript, and
neither could tell that the other half existed.

This directory is the fix. Trees small enough to travel are committed here, so
a fresh clone on any host carries the evidence with it.

## What is here

| Tree | Files | Size | What it backs |
|---|---|---|---|
| `icme2027_placement_msls` | 23 | 7.0 MB | The seven MSLS placement comparisons in "The Null Holds on Real Geographic Data" |
| `operator_study` | 32 | 6.7 MB | All eight rows of the operator table, at both delivered-MSE budgets |
| `margin_oracle` | 2 | 2.2 MB | The margin-gradient placement row |
| `known_jacobian` | 36 | 272 KB | The synthetic known-Jacobian control model |
| `known_jacobian_operators` | 10 | 260 KB | The operator variant of that control |
| `d4_3seed` | 8 | 5.1 MB | A three-seed run of the direction family |
| `sanfree_3seed` | 8 | 5.3 MB | The gallery-free three-seed run |
| `tifs5_analysis` | 19 | 76 KB | Analysis-script outputs, regenerable from the rows above |
| `tifs_a5` | 8 | 740 KB | The four-budget privacy-utility frontier (dataset-level mIoU) |
| `tifs_a8_mse60.0` | 2 | 552 KB | The release-boundary measurement at delivered MSE 60 |
| `tifs_a6` | 13 | 24 KB | Adaptive-attacker training logs and the held-out query ids |
| `tifs_a6_eval` | 8 | 296 KB | The adaptive-attacker evaluation this section's conclusions rest on |
| `tifs_a7b` | 3 | 488 KB | Patch-NetVLAD, the third held-out backbone (R9's external validity) |
| `tifs_a8_mse5.0` | 2 | 544 KB | The release-boundary measurement at delivered MSE 5.0 |
| `tifs_d6` | 12 | 20 MB | Table IV: the transfer and white-box columns on both attackers |
| `tifs_d7` | 4 | 9.5 MB | Downstream detector and segmenter utility of the released frames |
| `tifs_a8` | 3 | 548 KB | The release boundary at the operating point, delivered MSE 15.68 |
| `tifs_a8_hi` | 3 | 368 KB | The release boundary at the high budget, delivered MSE 241.5 |
| `tifs_a2` | 4 | 180 KB | Jacobian column norms and margin flip rates (R2, R3) |
| `tifs_a8_vitstart8` | 2 | 552 KB | Superseded: the misconfigured operating-point run (see below) |
| `tifs_a8_hi_vitstart8` | 2 | 368 KB | Superseded: the misconfigured high-budget run (see below) |
| `tifs6_a8_mse*_s2` | 2 each | 12 MB | Release boundary, seeds 5678/9012, 400 queries, one configuration at all four budgets |
| `tifs6_a5_s2` | 8 | 4.4 MB | Utility frontier, seeds 5678/9012, 200 images |
| `tifs6_vit` | 4 | 1.6 MB | ViT-B/16 attacker: a trunk that appears in no surrogate |

Every printed number these trees back was recomputed from them on
9 September 2026 and matched the manuscript exactly: the seven placement deltas
(-0.0008, +0.0008 three times, +0.0025, +0.0050, +0.0092) and all eight
operator rows with their p values.

## How to read them

Nothing needs to be copied or extracted. Both consumers search `src/outputs/`
first and fall back to this directory:

- `src/scripts/audit_claim_consistency.py` (`ROOTS`)
- `src/artifact/build_artifact.sh` (`resolve_tree`)

The first root that matches a tree wins, so a tree present in both places is
read from `src/outputs/` and never concatenated with its committed copy.

## One trap worth knowing

The placement exports carry two different p values per comparison, and the
manuscript quotes the **query-level Wilcoxon** statistic, not the exact McNemar
p stored beside it in `significance.csv`. They differ enough to matter: the
oracle-gradient cell is 0.86 one way and 1.00 the other. Recompute that column
with `analyze_placement_query_level.py` rather than reading whichever column
comes first.

## Nothing is missing

Every family the manuscript draws on is here. The auditor verifies **153 of 153
claims, zero mismatched, nothing unverifiable** -- the first time that has been
true on any single machine.

## The two superseded runs, and why they are kept

`tifs_a8_vitstart8` and `tifs_a8_hi_vitstart8` are an earlier A8 pair run with
the wrong surrogate ensemble (`vit_b_16` in place of `cosplace`) and a random
start of 8.0 rather than 1.0. Nothing reads them and they are evidence for
nothing. They are kept because the manuscript had quoted one of them by
mistake: the high-budget release gain and amplitude read `2.312` and `36.99`,
which are this run's numbers, where the published configuration gives `3.078`
and `49.25`. Keeping both makes that correction checkable against
`run_config.json` instead of asking a reader to take it on trust.

## A second configuration trap, in the published release-boundary sweep

The four seed-1234 trees behind the original frontier are not one sweep. Their
`run_config.json` files record a random start of **8.0** at MSE 5.0 and MSE 60
and **1.0** at the operating point and MSE 241.5. The manuscript presented all
four as one budget sweep, and at MSE 60 that mattered: the gain reads 1.146
under the 8.0 start and 1.516 under 1.0.

The `tifs6_a8_mse*_s2` runs are 1.0 at every budget, and the manuscript's
amplitude figures now come from them. Where the earlier run shares the
configuration -- MSE 15.68 and MSE 241.5 -- the two agree to three decimals
(0.7727 against 0.7727; 3.078 against 3.081), which is the check that nothing
else differs between the two sets. Compare `run_config.json` before pooling any
of these trees.

## A trap in `tifs_a2`

The producer is resumable, so a restarted run appends a second row for every
query that was in flight -- 28 of the 628 rows here. Averaging the raw rows
double-weights those queries. `analyze_jacobian_columns.py` now keeps the last
row per `(query, map)` and reports how many rows a restart superseded, so the
summary does not depend on how many times the run was interrupted. Any new
consumer of this tree must do the same.

`tifs_d6` carries a `superseded/` subdirectory holding an earlier EOT run that
was replaced by the three per-seed files beside it. It is kept as provenance;
nothing reads it, because every consumer globs the tree's top level only.

## A note on `tifs_a6`

Only the logs, the run metadata and the held-out query ids are here. The
original tree on the compute host is 901 MB, almost all of it a cache of
perturbed frames (`.pt`) that is regenerable and does not belong in git. What
was kept is what the manuscript's two training-side numbers rest on: `a6.log`
records validation MRR `0.04718` at epoch 0 for the hardened exposure and
`0.05776` for the unhardened one, both runs then selecting epoch 8, which is
what the text reports as `0.047` and `0.058`.

## What does not belong here

Only text evidence: CSV, JSONL, TXT and the run logs that establish provenance.
No imagery, no model weights, no checkpoints. A tree that would push this
directory past roughly 100 MB should be summarised down to its per-query rows
first, or left in `src/outputs/` with its absence recorded in
`docs/progress.md`.

The local `.gitignore` here re-includes `*.log`, which the repository ignores
globally, because for frozen evidence the run log is part of the record.
