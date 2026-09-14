# Code — *Allocation or Direction? A Matched-Distortion Audit of Visual Location-Privacy Mechanisms*

The audit code behind every number in the manuscript and the supplement.

The paper's claim is that its numbers can be rechecked **without a GPU and
without the imagery**. This package is the half of that claim that is code; the
other half is the per-query evidence, deposited separately at IEEE DataPort,
DOI [10.21227/jnr0-jm15](https://doi.org/10.21227/jnr0-jm15). Neither is useful
alone. The layout here is the one the code expects, so the only step between
unzipping and recomputing is dropping the exports into place. No manuscript
text travels in this package; where the code needs the documents, it is pointed
at them with `PPEDCRF_PAPER` (below).

## The one command that checks the paper

```bash
pip install -r src/requirements-verify.txt   # numpy + scipy; no torch, no GPU
python src/scripts/audit_claim_consistency.py
```

(`src/requirements.txt` is the other one — it pulls torch and the retrieval
backbones, and is needed only to re-run an experiment from imagery.)

It walks every headline number in the manuscript and recomputes it from the raw
per-query rows. Each claim carries three things: the value as the manuscript
prints it, a literal string that must still occur in the file that prints it,
and a recipe for recomputing it. A claim passes only when the string is still
there **and** the recomputed value matches to the stated tolerance. Every
comparison uses the manuscript's declared unit of inference — seeds averaged
within a query, a query-cluster bootstrap for the interval, a Wilcoxon
signed-rank over per-query differences for the test.

A claim whose export tree is absent is reported as `NO DATA`, never silently
skipped, so a partial drop of the evidence produces a list of exactly which
numbers went unchecked rather than a smaller denominator that reads like
success.

Against the complete evidence the run reports:

```
750 claims: 750 verified, 0 mismatched, 0 unverifiable (export tree absent),
0 no longer present in the source file.
Exit status 0.
```

**Every number it can reach checks.** This package carries no manuscript text —
the paper travels in the source upload, not here — and that costs part of the
registry, which is worth stating rather than leaving to be discovered. Each
claim carries two tests: recompute the number from the rows, and locate the
string the paper prints in the document that prints it. The second needs the
document, so it is inert here and the run reports `0 no longer present`. And 156
claims are *registered* by parsing the paper's generated tables, so without them
they are not registered at all.

**To run the full registry, point the code at the manuscript you already have:**

```bash
# extract 06_source.zip, then put both documents and the union of their
# generated/ directories in one folder, e.g. ./paper
PPEDCRF_PAPER=./paper python3 src/scripts/audit_claim_consistency.py
```

That reports `804 claims: 804 verified, 0 mismatched, 15 no longer present`,
and exits 1 — because the exit code also trips on those 15 locator misses,
which are numbers the paper stopped printing when it was compressed to the page
limit. Their rows are unaffected, which is why they keep verifying; what is gone
is the sentence that quoted them. **None of the 15 is a disagreement between a
printed number and its rows** — that count is `0 mismatched`, and it is the one
to read. The remaining 102 of the paper's 906 are registered from the extended
evidence report and the tables only it prints, which are not part of this
submission. Run with `--verbose` to see every claim individually.

## Two ways to reproduce, and what each needs

**A. Recompute the reported numbers — no GPU, no imagery.** Download the
deposit, unpack `exports.zip` into `src/exports/`, and run the command above,
or regenerate any individual table with the `src/scripts/make_*_table.py`
script that owns it. Every table in the paper is written by one of these from
the rows, so a table and its prose can be checked against the same source
independently. This is the path the paper's reproducibility claim rests on, and
it needs nothing beyond numpy and scipy.

**B. Re-run an experiment from imagery — GPU, and the datasets.** The
`src/scripts/run_*.py` drivers re-run the arms end to end: the placement study,
the operator arm, direction transfer, the geolocator arms, the adaptive
attacker, purification, and the published-mechanism comparison. These need a
GPU and the benchmark imagery. **The imagery is not redistributable and is not
included anywhere in this submission** — Mapillary Street-Level Sequences and
KITTI-360 must be obtained from their own sources under their own terms. The
manifests carry identifiers only, which is enough to rebuild every split
exactly.

Two of the drivers call a hosted vision–language model. They read the API key
at run time from a credentials file you supply and point at with `--env`; no
key is embedded anywhere in this package, and none is written to any export,
log or metadata file.

## Layout

| Path | What it is |
|---|---|
| `src/scripts/` | The audit. `analyze_*` recompute an arm from rows, `make_*_table.py` write the manuscript's tables, `run_*` re-run an arm from imagery, `audit_*`/`check_*` are the consistency gates |
| `src/eval/`, `src/models/`, `src/privacy/`, `src/datasets/`, `src/utils/` | The mechanism and attacker implementations the drivers call |
| `src/tests/` | Unit tests for the parts where a silent numerical error would not be visible in a plot |
| `src/Docker/`, `src/requirements.txt` | Environment |
| `src/exports/` | **Empty on purpose.** This is where the deposited evidence goes |

No manuscript file ships here in any form — no `.tex`, no figures, no tables.
`PPEDCRF_PAPER` is how the verifier is pointed at them when you want the wider
registry; it defaults to a sibling `paper/` directory, which is where they sit
in the authoring repository.

## Two things that are true and worth saying plainly

The `src/scripts/launch_*.sh` files carry the absolute paths of the rented
hosts the experiments actually ran on. They are kept as run records rather than
rewritten into something tidier that never executed; nothing else in the
package depends on them.

A Code Ocean capsule is published for this project as well, at
<https://codeocean.com/capsule/9035965/tree>, and the manuscript's first-page
footnote names it. It runs this same verifier under one click against the
deposited evidence, which is worth having. It is **not** this package, though.
It carries the mechanism and the verifier, not the drivers that recompute a
table from rows or re-run an arm from imagery. It carries no manuscript text
either, so on its own it reports the same 750 verified, 0 mismatched that this
package does. The difference is that you have the manuscript: with
`PPEDCRF_PAPER` set, this package reaches 804. Where the two disagree about what
has been checked, this package is the one to believe.
