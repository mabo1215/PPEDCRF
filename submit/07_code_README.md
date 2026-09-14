# Code — *Allocation or Direction? A Matched-Distortion Audit of Visual Location-Privacy Mechanisms*

The audit code behind every number in the manuscript and the supplement.

The paper's claim is that its numbers can be rechecked **without a GPU and
without the imagery**. This package is the half of that claim that is code; the
other half is the per-query evidence, deposited separately at IEEE DataPort,
DOI [10.21227/jnr0-jm15](https://doi.org/10.21227/jnr0-jm15). Neither is useful
alone. The layout here is the one the code expects, so the only step between
unzipping and recomputing is dropping the exports into place.

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

Against the complete evidence the current run reports:

```
906 claims: 906 verified, 0 mismatched, 0 unverifiable (export tree absent),
15 no longer present in the source file.
Exit status 1.
```

**Every number checks. The exit status is still 1, and that is expected here** —
it also trips when a claim's locator string is no longer found, which is a
statement about the manuscript's wording rather than about any number. The 15
are numbers the paper stopped printing when it was compressed to the page
limit. Their claims keep verifying because the rows behind them are unaffected;
what is gone is the sentence that quoted them. They are left registered rather
than deleted, so the report doubles as the list of results the compression cost
the paper, which is a thing worth being able to see.

None of the 15 is a disagreement between a printed number and the rows behind
it. That count is the `0 mismatched`, and it is the one to read. Run with
`--verbose` to see all 906 individually.

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
| `paper/` | `main.tex`, the supplement, and the generated tables — the verifier parses these to find the printed strings it checks against |
| `src/exports/` | **Empty on purpose.** This is where the deposited evidence goes |

`paper/generated/` carries a few tables that neither document `\input` any more,
left behind when the manuscript was compressed. They are not dead: the verifier
still reads them, because the claims they support moved to the extended
evidence report rather than disappearing.

## Two things that are true and worth saying plainly

The `src/scripts/launch_*.sh` files carry the absolute paths of the rented
hosts the experiments actually ran on. They are kept as run records rather than
rewritten into something tidier that never executed; nothing else in the
package depends on them.

A Code Ocean capsule is published for this project as well, at
<https://codeocean.com/capsule/9035965/tree>, and the manuscript's first-page
footnote names it. It runs this same verifier under one click against the
deposited evidence, which is worth having. It is **not** this package, though:
it carries the perturbation mechanism and the verifier, not the drivers that
recompute a table from rows or re-run an arm from imagery, and as published its
registry resolves 783 of these claims rather than 906 — it does not ship the
generated tables the registry parses to enumerate the rest, and its run states
that coverage rather than implying more. Where the two disagree about what has
been checked, this package is the one to believe.
