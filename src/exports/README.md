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

## Still missing

Five trees from the A5/A8/D6 round exist only on the work machine:
`tifs_a5`, `tifs_a8`, `tifs_a8_hi`, `tifs_a8_mse60.0`, `tifs_d6`. They back the
privacy-utility frontier, the release-boundary section and the transfer and
white-box columns of Table IV -- 65 of the auditor's 140 claims. Until they are
committed here the same way, the auditor reports 75 of 140 on any machine that
lacks them.

To add them from the work machine:

```bash
cd <repo>
for t in tifs_a5 tifs_a8 tifs_a8_hi tifs_a8_mse60.0 tifs_d6; do
  cp -r "src/outputs/$t" "src/exports/$t"
done
python3 src/scripts/audit_claim_consistency.py   # expect 140 of 140
git add src/exports && git commit
```

## What does not belong here

Only text evidence: CSV, JSONL, TXT and the run logs that establish provenance.
No imagery, no model weights, no checkpoints. A tree that would push this
directory past roughly 100 MB should be summarised down to its per-query rows
first, or left in `src/outputs/` with its absence recorded in
`docs/progress.md`.

The local `.gitignore` here re-includes `*.log`, which the repository ignores
globally, because for frozen evidence the run log is part of the record.
