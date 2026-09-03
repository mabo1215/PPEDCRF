# F1/F2 Handoff: Matched-PSNR Significance Test and MixVPR Determinism Check

Written 2026-09-03 by a background Claude Code session that had **no access**
to the real monitoring corpus, MSLS/KITTI-360 data, a CUDA-capable PyTorch
build, or a live vGPU 3090 connection (no `.env`, no D:/F:/G: drives — a
different sandbox from the machine(s) referenced throughout
`docs/progress.md`). This doc exists so the next session with real access can
run F1/F2 immediately without re-deriving anything. See
`docs/Revision_suggestions.tex` (Findings F1/F2) and `docs/Design.md`
("Fresh independent review follow-up") for why these are needed.

**Completion update (2026-09-03):** The handoff plan was executed on the
reachable vGPU 3090. F1 completed the real 12-point sigma sweep and produced
15 paired comparisons; F2 completed two real MixVPR proxy50 runs and passed
the determinism checker for all 3,750 rows and 20 columns. The resulting
paper updates are complete. The remaining provenance work is tracked in
`docs/experiment_provenance.md`; E1 MSLS and E5 KITTI-360 exports were not
present on the reachable machine.

Both items are **cheap** relative to the earlier E1-E7 revision cycle:
neither needs new datasets, and F1 in particular may need zero new GPU time
if its input sweep still exists on disk.

## Step 0: prefer the original local RTX 3070 machine over renting vGPU 3090

Everything below was previously run on a machine with a local RTX 3070,
Python env at `D:\source\.venv`, and datasets under `F:\work\datasets` /
`G:\work\datasets` (per `docs/progress.md` items 84-94). If that machine is
available, run F1/F2 there first — it already has everything needed and
costs nothing. Only fall back to renting a fresh vGPU 3090 (AutoDL/SeetaCloud
style, per `docs/archived/vgpu3090_experiment_handoff.md`) if that machine is
unavailable or its `src/outputs/` has been cleaned since. Renting vGPU 3090
requires the user to spin up a fresh instance and share the new SSH
host/port — the previous instance was intentionally shut down (billing
stopped) and its address is not reusable.

## F1: matched-PSNR significance test

1. Check whether `src/outputs/tomm_review_e2_sigma/` (the sigma-sweep
   directory that already backs `paper/appendix.tex` Table `tab:matched_psnr`)
   still exists with its `sigma_*/per_query.csv` files intact. If so, **skip
   straight to step 3** — no GPU needed.
2. If it does not exist, regenerate it (this is the only part that needs
   GPU/data access):
   ```bash
   for sigma in 4 6 8 10 12 16 20 24 28 32 40 50; do
     python src/scripts/run_tomm_review_proxy.py --mode proxy --sigma "$sigma" \
       --backbones resnet18 --gallery_sizes 48 --seeds 1234 1235 1236 \
       --output_dir "src/outputs/tomm_review_e2_sigma/sigma_${sigma}"
   done
   ```
   (Matches the command already used to build the existing table; each point
   took under 2 minutes on the RTX 3070 per `docs/progress.md` item 88.)
3. Run the new significance test (pure post-processing, no GPU):
   ```bash
   python src/scripts/significance_test_matched_psnr.py \
     --sweep_dir src/outputs/tomm_review_e2_sigma \
     --backbone resnet18 --gallery_size 48 \
     --targets 30 33 36 --compare_against global_noise \
     --output src/outputs/tomm_review_e2_sigma/matched_psnr_significance.csv
   ```
4. Read the printed table. For each `target_psnr` / `variant_a` row, note
   `mcnemar_exact_p`, `bootstrap_ci_low`/`_high`, and
   `min_significant_discordant_asymmetry_at_this_n`.
5. Write these into `paper/appendix.tex` Section 5 (Matched-PSNR/Effective-MSE
   Comparison) as a new column or a follow-up paragraph, and adjust the
   "statistically indistinguishable" sentence in `paper/main.tex` Section 4.4
   and the Conclusion to state the tested claim (e.g. "no significant
   difference was detected by an exact paired test... though the sample only
   has power to detect roughly a 1-in-6 discordant-pair asymmetry").

## F2: MixVPR determinism check

1. Re-run the MixVPR cell of the proxy50 benchmark **twice**, with identical
   arguments, into two separate output directories:
   ```bash
   python src/scripts/run_tomm_review_proxy.py --mode proxy \
     --backbones mixvpr --gallery_sizes 50 75 100 --seeds 1234 1235 1236 \
     --pair_pool_size 600 --max_gallery 100 \
     --output_dir src/outputs/f2_determinism_run_a

   python src/scripts/run_tomm_review_proxy.py --mode proxy \
     --backbones mixvpr --gallery_sizes 50 75 100 --seeds 1234 1235 1236 \
     --pair_pool_size 600 --max_gallery 100 \
     --output_dir src/outputs/f2_determinism_run_b
   ```
   (Flags mirror the existing proxy50 protocol; adjust `--pair_pool_size`/
   `--max_gallery` if the real invocation used different values — check
   `src/outputs/tomm_review_proxy50/run_metadata.json` if it still exists.)
2. Diff them:
   ```bash
   python src/scripts/check_run_determinism.py \
     src/outputs/f2_determinism_run_a/per_query.csv \
     src/outputs/f2_determinism_run_b/per_query.csv
   ```
3. If it prints `[determinism] OK: ...`, add one sentence to the appendix's
   "MixVPR Margin Diagnosis and Mitigation Status" subsection: re-running the
   current pipeline twice under identical seeds reproduces every value
   exactly, so the earlier discrepancy reflects a prior code/checkpoint
   change, not run-to-run nondeterminism. If it reports a mismatch, name the
   differing columns in the appendix instead and treat it as a new,
   higher-priority finding (check first: unseeded RNG in the MixVPR loader,
   non-deterministic cuDNN kernel selection, or thread-order-dependent
   floating-point reduction) — do not paper over it.

## Known environment gotchas from prior sessions (still likely to apply)

- `pandas` was missing from `requirements.txt` despite being a de facto
  dependency of `matched_psnr_from_sweep.py`; it has been added in this
  session, but if working from an older checkout, `pip install pandas`.
- `run_controlled_retrieval_benchmark.py`/the VPR backbones need
  `matplotlib`, `faiss-cpu`, `scikit-learn`, `scipy` beyond
  `requirements.txt` — install them if a fresh venv is used (see
  `docs/archived/vgpu3090_experiment_handoff.md` for the exact history of
  this).
- If renting a fresh vGPU and pulling CLIP/VPR weights via Hugging Face
  through a proxy (`network_turbo` or similar), the default `hf-xet`
  transfer backend has repeatedly stalled at 0 bytes in this project's prior
  sessions — prefer plain `curl -C -` against the resolve URL, or
  `hf-mirror.com`, per the detailed notes in
  `docs/archived/vgpu3090_experiment_handoff.md`.

## Do not

- Do not write any F1/F2 number into `paper/` from a smoke-test or synthetic
  run. The synthetic-data smoke tests already run in this session
  (`make_synthetic_sigma_sweep.py` + both new scripts) only validate script
  logic; they produced no paper-facing evidence and none should be inserted.
- Do not spin up a new vGPU rental preemptively "just in case" — check the
  original local machine first (Step 0), and if renting is genuinely needed,
  let the user do the actual power-on, consistent with their stated
  preference in earlier sessions (`docs/progress.md`, item 88 context).
