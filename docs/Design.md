# PPEDCRF ACM TOMM Revision-Cycle Experiment Plan (2026-08-31)

This section is the active experiment design for the review letter in
`docs/TOMM_Response_Letter.md`. It takes precedence over the historical
CertiGround material below for this PPEDCRF revision cycle. No new number may
be written into `paper/` until the corresponding run passes its completion
gate and its manifest records the code revision, data provenance, seed set,
and output checksum.

## Review-to-experiment mapping

| ID | Reviewer request | Evidence needed | Planned output |
|---|---|---|---|
| E1 | Use a real place-recognition/geolocalization benchmark with true place identities or GPS, larger galleries, and cross-condition variation (R2-1, R3-5) | A manifest-driven benchmark with query/gallery place IDs or GPS, viewpoint/illumination/season/weather metadata, and fixed attacker backbones | `geotagged_vpr_summary.csv`, per-query manifest, and a paper appendix table; if no compliant dataset is available, record a blocked limitation rather than re-labeling the proxy benchmark |
| E2 | Compare at matched PSNR, MSE, or perturbation energy; add unary-only/no-DCRF ablations (R2-2, R3-3) | Full, no-temporal, no-NCP, unary-only, and no-DCRF variants evaluated on the same pairs, seeds, gallery, PSNR target, MSE, and effective perturbation energy | `review_ablation.csv`, `review_margin.csv`, and a matched-operating-point table |
| E3 | Add stronger mask-based and attacker-aware baselines (R2-2) | Existing mask-guided blur/mosaic baselines retained; a fixed-backbone gradient feature-suppression baseline constrained by the same image-space budget and reported only when its optimization completes | Baseline comparison CSV and an explicit scope statement if attacker-aware optimization is not comparable or fails its gate |
| E4 | Report detection mAP and segmentation mIoU on the same sanitized images (R2-2) | Label-backed images processed by the same release-side sanitizer, with frozen pretrained detector/segmenter, identical original/sanitized inputs, and standard evaluation metrics | `utility_same_images.json`, detector/segmenter summaries, and an appendix table; COCO/VOC may be used only as utility datasets, not as geographic ground truth |
| E5 | Explain the unary predictor and independently assess its sensitivity maps (R3-1) | Exact architecture, parameter count, training target/source, loss, checkpoint provenance, and a held-out or weakly supervised attribution-consistency diagnostic | Method/appendix text plus `unary_diagnostic.csv`; attribution agreement is labelled diagnostic evidence, not ground-truth map accuracy |
| E6 | Define Eq. (2), smoothing operator, gallery construction, and the retrieval evidence for Fig. 6 (R3-5, R3-6, R3-7) | Reproducible constants, source counts, pair/distractor IDs, per-query correct rank, strongest negative, similarities, and pre/post margins | Text revision, `retrieval_case_study.csv`, and an updated qualitative figure if the case-study gate passes |
| E7 | Analyze MixVPR adverse transfer and qualify robustness (R3-8) | Per-query and margin-level results for all gallery sizes and seeds, including the raw rank, sanitized rank, correct similarity, hardest-negative similarity, and failure category | `mixvpr_per_query.csv`, failure summary, and revised robustness wording |

## Public dataset selection for E1 and E5 (2026-08-31)

The public-data search uses the official dataset pages and repositories as the
source of record. Kaggle mirrors may be used only as a transport convenience
after their files and metadata are byte-level checked against the official
release; a Kaggle-only mirror is not sufficient evidence because it may omit
GPS, sequence, condition, or license metadata.

| Dataset | Role | Why it satisfies the gate | Acquisition and license gate |
|---|---|---|---|
| Mapillary Street-Level Sequences (MSLS) | E1 primary | 1.6M street-level images with raw longitude/latitude, capture time, sequence IDs, view direction, unique place clusters, and official cross-condition subtasks (`day2night`, `night2day`, `summer2winter`, `old2new`, and related tasks). | Download from the official Mapillary release, retain the Meta terms-of-use record, and do not redistribute images in the repository. |
| Oxford RobotCar | E1 fallback / cross-check | More than 100 repetitions of a fixed Oxford route over a year with GPS/INS ground truth and weather, traffic, lighting, construction, and seasonal change. | Registration is required; use only the registered non-commercial academic release and store paths/metadata, not raw images, in the repository. |
| KITTI-360 | E5 primary, optional E1 cross-check | Over 320k images with accurate geolocation, continuous vehicle poses, synchronized OXTS GPS/IMU measurements, and 2D semantic/instance masks. | Registration and intended-use declaration are required; the official CC BY-NC-SA 3.0 terms must be recorded before download or paper use. |

The primary E1 run will use MSLS because its official metadata already exposes
place clusters and standardized condition subtasks. The runner will construct
query/gallery records from the official query/database split, preserve the
provided `unique_cluster` as `place_id`, and report city/condition strata. A
positive gallery item must share the official place cluster while query and
gallery files remain disjoint. The initial remote target is at least 200
queries, 1,000 gallery candidates, two cities, and two condition subtasks; the
final size may be reduced only with a recorded resource limitation. Oxford
RobotCar remains a fallback if MSLS registration or download access cannot be
completed.

For E5, KITTI-360 semantic and instance labels are not treated as sensitivity
ground truth. Instead, the held-out diagnostic will build an independent,
weakly supervised reference from a frozen VPR embedder: tiled query occlusion
will measure the drop in the correct-place-versus-hardest-negative margin.
Dynamic semantic instances will be excluded from the background analysis, and
the unary/refined map will be compared with the reference using Spearman
correlation, top-10-percent overlap, attribution energy concentration, and
static-background versus dynamic-object stratification. The diagnostic will
split by sequence, so no frame from a training sequence may be used in the
held-out report. It will be labelled attribution-consistency evidence rather
than pixel-level ground-truth accuracy.

Official sources recorded for reproducibility:

* MSLS release and metadata: `https://github.com/mapillary/mapillary_sls` and
  `https://www.mapillary.com/dataset/places`.
* Oxford RobotCar: `https://robotcar-dataset.robots.ox.ac.uk/`.
* KITTI-360 overview, labels, poses, and terms:
  `https://www.cvlibs.net/datasets/kitti-360/`.

The targeted Kaggle audit found an Oxford RobotCar place-recognition mirror
that may be useful after file-list, checksum, metadata, and license
verification. A KITTI-360 3D-semantics mirror reports an unknown license and
does not demonstrate the RGB--pose--2D-mask topology required by E5; it is
therefore rejected as a paper-facing source. Mapillary Vistas mirrors are
segmentation-only and are not an E1 MSLS replacement.

## Execution order and gates

1. **Protocol freeze and local smoke test.** Freeze the current code revision,
   output schema, seeds `{1234, 1235, 1236}`, default frame size, and the
   current 12-pair proxy selection. Run the new scripts on synthetic tensors
   and on at most two locally available monitoring pairs. This validates model
   loading, variant dispatch, energy accounting, rank/margin computation, and
   JSON/CSV schemas without creating paper evidence.
2. **E2/E6/E7 controlled rerun.** Reuse the repository's monitoring proxy only
   for ablation, margin, and MixVPR diagnosis. The default run must include
   the existing variants plus `unary_only` and `no_dcrf`; it must export
   per-query retrieval ranks and margins. Each variant is compared both at its
   native operating point and after nearest-neighbour matching in PSNR and
   effective perturbation MSE. A positive or null MixVPR result is retained as
   evidence and is not filtered out.
3. **E3 attacker-aware baseline.** Optimize against one fixed attacker only
   (ResNet18 for the proxy and the declared attacker for E1), using projected
   image-space updates with an explicit $ℓ_\infty$ bound and an early-stop
   criterion. The baseline is not promoted to a headline comparison unless
   it has the same gallery, same utility target, same seed protocol, and a
   finite, reproducible result. Otherwise it remains an engineering diagnostic.
4. **E4 same-image utility.** Use COCO/VOC only where annotations and frozen
   model weights are available. Apply the sanitizer to the exact images used
   by the detector/segmenter, report original versus each sanitized variant,
   and keep detection mAP and segmentation mIoU in a separate utility track.
   If pretrained weights or annotations are unavailable on 4c, mark E4
   blocked and do not substitute legacy training curves.
5. **E1 geotagged benchmark.** Use the official MSLS metadata first and
    preserve its `unique_cluster` as the place label. The new loader accepts a
    CSV/JSONL manifest rather than assuming a particular public VPR directory
    layout. Required fields are `query_id`, `query_path`, `gallery_id`,
    `gallery_path`, `place_id` (or latitude/longitude), and optional
    `viewpoint`, `illumination`, `season`, and `weather`. A run is scientific
    evidence only if the manifest passes uniqueness, file-exists, place-label,
    condition-coverage, and query/gallery disjointness gates. If the primary
    MSLS download is not available, use the registered Oxford RobotCar release
    under the same gate; never relabel the current monitoring proxy as GPS data.
6. **E5 independent unary diagnostic.** Build a KITTI-360 manifest from
    sequence-held-out images, 2D semantic/instance labels, and georegistered
    poses. Use frozen VPR occlusion attribution as a weak reference, report the
    metrics defined above, and fail the gate when the checkpoint is missing,
    maps are constant, or fewer than the declared held-out samples have valid
    labels and gallery positives.

## Fair-comparison definitions

- **Retrieval privacy:** report Top-1/Top-5/Top-10 and per-query rank. The
  correct-versus-hardest-negative margin is
  $m_i = s(q_i,g_i^+) - \max_{j\in\mathcal{G}_i^-}s(q_i,g_j^-)$.
  Report the mean, median, and bootstrap interval before and after sanitization.
- **Image utility:** report PSNR, SSIM, and full-frame MSE. Also report
  effective perturbation MSE, `mean((I'-I)^2)` over the full frame, and support
  coverage. “Matched energy” means nearest-neighbour matching on effective
  perturbation MSE, not nominal `sigma_0`.
- **Variants:** `full` uses unary → DCRF → NCP; `no_temporal` disables only
  temporal reuse; `no_ncp` keeps DCRF support with fixed unit strength;
  `unary_only` uses sigmoid unary logits with NCP and no DCRF refinement;
  `no_dcrf` uses the sigmoid unary support with fixed unit strength. The
  labels must remain distinct in every output file.
- **Attacker-aware baseline:** use a frozen embedder and gallery, optimize only
  the released query image, constrain pixel updates, and record iteration
  count, final PSNR/MSE, and whether the correct gallery item was removed from
  Top-1. This baseline tests an adaptive attacker/defense boundary; it is not
  evidence of attacker-agnostic protection.

## Resource and handoff policy

- Local GPU smoke uses the available RTX 3070 and must fit within 2 GB VRAM.
- Long runs on 4c use a fresh root
  `/data1/PPEDCRF/tomm_revision_20260830`, one manifest per run, and a
  separate worktree. Workers must pass a free-VRAM gate before model loading;
  an unexplained process disappearance stops the queue and preserves partial
  JSONL/log files.
- The current 4c SSH alias (`h800cs2`) is blocked by a changed host key. Do
  not disable host-key verification or overwrite `known_hosts` without an
  administrator/owner-confirmed fingerprint. Until resolved, the remote
  status is `blocked`, while local code and smoke tests may proceed.
- After the code is pushed, `docs/archived/4c_experiment_handoff.md` is the handoff
  contract for Claude Code. It must contain the commit SHA, exact launch
  command, run root, completion gates, polling cadence, and explicit rules
  against paper writeback from incomplete or failed runs.

## Paper writeback rules

E1 results can support a real-geolocation claim only after its manifest and
ground-truth gate pass. E2/E6/E7 may update the main paper's ablation,
robustness, and limitation text after the output checks pass. E3 remains
appendix/diagnostic unless fairness gates pass. E4 can update utility claims
only with same-image labelled evaluation. If an item is blocked by missing
data, unavailable weights, or remote access, the paper must explicitly state
the limitation and the next step; no placeholder number or proxy relabeling is
allowed.

## Local validation status (2026-09-01)

The new review-cycle scripts compile successfully and the CUDA smoke suite
passes on the local RTX 3070. The original `src/outputs/sensnet_final.pt`
checkpoint stores `mask_root=null` and produces an almost spatially constant
unary map on monitoring clips (mean spatial standard deviation about
`2.6e-4`). Its provenance diagnostic therefore reports no ground-truth
sensitivity-map accuracy. A separate mask-backed checkpoint is used for the
registered KITTI-360 diagnostic described below; it passes the operational
non-constant-map gate but does not establish sensitivity-map accuracy.

The geotagged benchmark loader and same-image utility evaluator pass smoke
tests. Official-source MSLS manifests and the KITTI-360 diagnostic manifest
are now built and paper-facing results have been written back. MSLS all/o2n/n2o
runs pass the real-place gates, but the realized subset is limited to two
cities, day/Forward-view samples, and one attacker backbone. The original
unary checkpoint fails the non-constant-map gate; a sequence-0000
mask-backed checkpoint passes that operational gate on held-out sequence 0002
but has near-zero attribution agreement, so no sensitivity-map accuracy claim
is made. The 4c/vGPU launch remains unnecessary for the completed E1/E5
diagnostics; see `docs/archived/4c_experiment_handoff.md` for the remote
contract if a future run requires it.

## Fresh independent review follow-up (2026-09-03): F1-F3

All items from `docs/TOMM_Response_Letter.md` (E1-E7 above) are confirmed
implemented in `paper/main.tex` and `paper/appendix.tex` as of this date,
checked directly against the manuscript text rather than against prior
review documents. Per the repository's fresh-review protocol, a new
independent review of the current manuscript (not the old reviewer letter)
was run and is recorded in full in `docs/Revision_suggestions.tex`. It found
three follow-up items, none requiring new data collection, all requiring
either a rerun of an existing benchmark configuration or a pure
post-processing analysis over its output:

| ID | Gap identified by the fresh review | Evidence needed | Planned output |
|---|---|---|---|
| F1 | The appendix's matched-PSNR "statistically indistinguishable" claim (`tab:matched_psnr`) is asserted from rounded point estimates on a 36-paired-outcome sample (12 pairs x 3 seeds, one backbone, one gallery size), with no formal test and no statement of test power. | Exact McNemar test and a query-level cluster bootstrap CI on the Top-1 difference between each sigma-tunable variant and `global_noise` at each matched-PSNR target, plus the minimum discordant-pair asymmetry that would reach significance at this sample size. | `matched_psnr_significance.csv` from `src/scripts/significance_test_matched_psnr.py`, integrated into Appendix Section 5 and the main-text/Conclusion wording softened to match the tested claim. |
| F2 | The appendix states the current pipeline no longer reproduces an earlier reviewer-cited adverse MixVPR result, with no root cause identified, but never checks whether the *current* pipeline is even deterministic run-to-run under fixed seeds. | Two independent executions of the same benchmark invocation (same seeds, checkpoint, code revision), at minimum for the MixVPR proxy50 cell, diffed for exact numerical agreement. | A determinism verdict (pass, or a named mismatched field) from `src/scripts/check_run_determinism.py`, stated as one sentence in the appendix MixVPR subsection. |
| F3 | Paper-facing CSV/JSON exports under `src/outputs/` are git-ignored; this session found none of the large `tomm_review_*` run directories present in its own working environment, with no lightweight versioned record of which run (by checksum) backs which paper table. | A provenance table recording, per export cited by a paper table/figure, its path, producing git commit SHA, and SHA-256 checksum. | New provenance table in `docs/experiment_progress.tex` or a new `docs/experiment_provenance.md`; documentation only, no experiment. |

F1 and F2 required a machine with the real monitoring corpus, the
`sensnet_final.pt` checkpoint, and CUDA. That execution requirement was met on
the reachable vGPU 3090 on 2026-09-03. The analysis code itself
(`significance_test_matched_psnr.py`, `check_run_determinism.py`) remains
GPU-free and was also schema-smoke-tested against synthetic sigma-sweep data
(`src/scripts/make_synthetic_sigma_sweep.py`). F3 is documented in
`docs/experiment_provenance.md`; it remains partial because the E1 MSLS and E5
KITTI-360 export trees were not present on the reachable machine.

### Execution record for F1/F2 (completed 2026-09-03)

1. **F1.** Re-use the existing `src/outputs/tomm_review_e2_sigma/` sigma-sweep
   directory (or regenerate it if it no longer exists on the target machine
   via `run_tomm_review_proxy.py --mode proxy --sigma <S>` for
   `S in {4,6,8,10,12,16,20,24,28,32,40,50}`, ResNet18, gallery 48, three
   seeds — this is the exact command already used to build
   `tab:matched_psnr`). Then run:
   `python src/scripts/significance_test_matched_psnr.py --sweep_dir <dir> --backbone resnet18 --gallery_size 48 --targets 30 33 36 --compare_against global_noise --output src/outputs/tomm_review_e2_sigma/matched_psnr_significance.csv`.
   No new GPU inference is required if the sweep directory still exists;
   only a rerun of the sweep itself (already-completed work) would need GPU.
2. **F2.** Re-run the MixVPR proxy50 cell twice with identical arguments into
   two separate output directories, then run:
   `python src/scripts/check_run_determinism.py <run_a>/per_query.csv <run_b>/per_query.csv`.
   This does require fresh GPU execution (the point is to test the live
   pipeline, not archived outputs).
3. Write the results back into `paper/appendix.tex` (Section 5 for F1, the
   MixVPR subsection for F2) and adjust the corresponding main-text/Conclusion
   sentences once real numbers are in hand; do not write placeholder numbers
   before the runs complete.

The plan was executed on the vGPU 3090. F1 completed all 12 sigma points and
produced 15 paired comparisons, each with 36 pairs, zero discordant pairs,
exact McNemar $p=1.0$, and a query-cluster bootstrap 95% CI of `[0.000,0.000]`;
the minimum significant asymmetry was six pairs. F2 compared two independent
MixVPR proxy50 runs and found agreement within `rtol=1e-6` and `atol=1e-9`
for all 3,750 rows and 20 columns. The results are integrated into the paper
and tracked in `docs/progress.md`.
