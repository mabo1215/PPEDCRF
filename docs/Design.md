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
| F3 | Paper-facing CSV/JSON exports under `src/outputs/` are git-ignored; this session found none of the large `tomm_review_*` run directories present in its own working environment, with no lightweight versioned record of which run (by checksum) backs which paper table. | A provenance table recording, per export cited by a paper table/figure, its path, producing git commit SHA, and SHA-256 checksum. | New provenance table in `docs/experiment_progress.tex` or a new `docs/progress.md` (the experiment provenance section); documentation only, no experiment. |

F1 and F2 required a machine with the real monitoring corpus, the
`sensnet_final.pt` checkpoint, and CUDA. That execution requirement was met on
the reachable vGPU 3090 on 2026-09-03. The analysis code itself
(`significance_test_matched_psnr.py`, `check_run_determinism.py`) remains
GPU-free and was also schema-smoke-tested against synthetic sigma-sweep data
(`src/scripts/make_synthetic_sigma_sweep.py`). F3 is documented in
`docs/progress.md` (the experiment provenance section); it remains partial because the E1 MSLS and E5
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

## ICME 2027 revision-cycle plan (2026-09-03)

This is the active plan for the independent ICME 2027 review recorded in
`docs/RevisionSuggestions.tex`. The prior TOMM plans remain historical
evidence. No result from a smoke run, partial run, failed gate, or proxy-only
run may be written into the paper as real-place or sensitivity validation.

### Review-to-work mapping

| ID | Review finding | Action | Evidence and completion gate |
|---|---|---|---|
| ICME-M1 | The default artifact is ACM/TOMM, 17 pages plus an 11-page appendix, rather than an IEEE conference submission. | Convert main, appendix, titlepage, and build defaults to anonymous IEEE conference mode; remove ACM/TOMM metadata and reduce the main paper to the final ICME limit when the 2027 kit is released. | Both sources compile with the IEEE template; PDF page count, fonts, letter size, metadata, anonymity, and final page limit are checked. |
| ICME-M2 | The unary map is near-constant or has negative attribution agreement, and current ablations do not identify DCRF/NCP gains. | Add an energy-preserving spatial intervention benchmark comparing the learned support with uniform, deterministic shifted, and seed-controlled spatially permuted support. | Manifest-driven per-query CSV with support statistics, effective MSE, Top-k, margins, and a gate requiring finite values and equalized perturbation energy. No map-validity claim is allowed unless a held-out attack-derived reference improves. |
| ICME-M3 | Synthetic pairs have no geographic ground truth; MSLS coverage is limited to two cities, one attacker, and narrow conditions. | Re-run the existing geotagged runner on the largest available official MSLS manifests, with multiple attacker backbones and condition/city strata. Add a manifest-coverage audit before inference. | Manifest passes place-label, path-disjointness, city/condition coverage, and positive-gallery gates; report cluster-aware per-place intervals. Missing data remains a stated limitation. |
| ICME-M4 | The paper claims video-sequence protection while retrieval evaluates only one sanitized middle frame. | Reframe current paper claims as frame-level release-side sanitization; prepare a sequence-level benchmark runner for clip lengths 1, 2, 4, and 8 as the next remote experiment. | Current paper must not claim sequence-level retrieval protection. Sequence runner smoke must verify pooled frame embeddings, best-frame attacker, temporal metrics, and output schema before vGPU use. |
| ICME-M5/M6 | Fixed-budget baselines are not matched, and epsilon/sigma is only a heuristic. | Keep current matched-PSNR result as a non-rejection; add realized perturbation statistics to the intervention output and revise text to call calibration an index, not DP. | No new numerical DP guarantee. Every output records full-frame MSE, support MSE, support coverage, clipping rate, and effective standard-deviation summaries. |
| ICME-M7 | McNemar currently mixes query clusters and seeds. | Add cluster-aware post-processing for intervention and future geotagged results; use query/place as the independent unit and seeds only as within-cluster variation. | Output states cluster count, seed count, bootstrap/permutation procedure, confidence level, and multiple-comparison policy. |
| ICME-M8/M9 | Paper-facing outputs are ignored, paths are machine-local, and TOMM identifiers remain in scripts. | Make new scripts venue-neutral, write relative-path manifests and SHA-256 records, and update the paper only from complete exports. | Python compile, local CUDA smoke, deterministic synthetic test, and a remote handoff manifest all pass. |

### Experiment ICME-M2: energy-preserving spatial intervention

The intervention isolates whether spatial placement, rather than only total
noise energy, explains retrieval changes. For each query frame, the runner
computes the released effective weight w = p times s, then creates:

1. the current learned support;
2. a uniform map with the same mean effective weight;
3. a deterministic spatial roll of the effective map with the same histogram;
4. a deterministic flattened permutation of the effective map with identical
   total energy.

All conditions use the same source frame, gallery, attacker, seeds, and
noise-draw convention. The primary output is a per-query table containing
Top-1/5/10, correct rank, correct-versus-hardest-negative margin, full-frame
MSE, effective support MSE, support coverage, clipping fraction, and map
mean/std. The intervention gate passes only when all rows are finite and the
energy-preserving controls are numerically verified within a relative
tolerance of 10^-6. These controls test spatial causality; they do not
validate the unary predictor against ground-truth sensitivity.

### Experiment ICME-M3: expanded real-place MSLS audit

Use the already registered official MSLS manifests and the existing
manifest-driven geotagged evaluator. Before GPU inference, run a coverage
audit that reports query count, gallery count, positive count per query,
city counts, subtask counts, viewpoint, illumination, season, weather, and
query/gallery path overlap. Run ResNet18 plus available dedicated VPR
backbones (CosPlace, MixVPR, and Patch-NetVLAD) at the same seeds and variant.
Aggregate uncertainty by place cluster, not by seed-expanded rows. The run is
paper-eligible only if the manifest gate and all requested backbone cells pass.

### Experiment ICME-M4: sequence-level retrieval preparation

The new runner will evaluate sanitized query clips at lengths 1, 2, 4, and 8.
For each clip it will report first-frame, mean-pooling, max-pooling, and
best-frame retrieval, plus temporal flicker and perturbation-stability
metrics. The current release-side method remains frame-level in the paper
until this experiment completes. The smoke mode uses synthetic clips and a
download-free embedder; the real mode uses the monitoring proxy and records
that it is still a proxy rather than geographic ground truth.

### Code and smoke-test sequence

1. Add the ICME-M2 intervention runner and the ICME-M4 sequence runner under
   `src/scripts/`, with English-only code, deterministic seeds, and explicit
   `scientific_evidence=false` in smoke output.
2. Patch the noise-mode documentation and runner metadata so the existing
   indexed per-frame Gaussian process is not called a cumulative Wiener
   process, while preserving backward compatibility for old exports.
3. Run `py_compile`, unit-level synthetic checks, and CUDA smoke on the local
   GPU. Smoke output must remain outside paper writeback.
4. Prepare a vGPU 3090 no-card handoff containing the pushed commit, exact
   command lines, output root, checkpoint/data gates, and estimated runtime.
5. After the user powers on the vGPU, execute only the complete gated runs,
   then aggregate by place/query cluster and decide whether any paper
   writeback is scientifically justified.

### Paper writeback policy for this cycle

The immediate text revision may narrow unsupported claims and change the title
to match frame-level evaluation. Existing numerical results remain unchanged
unless regenerated by a complete gate-passing run. ICME-M2 output can support
an intervention statement but cannot turn a structural mask into sensitivity
ground truth. ICME-M3 output can support broader real-place claims only after
manifest and coverage gates pass. ICME-M4 output can restore sequence-level
language only after a real sequence-level attacker result is complete.

## ICME-M3.3: white-box sign-gradient attacker on the real MSLS benchmark (2026-09-04)

`docs/RevisionSuggestions.tex` Additional Technical Comment 4 requires the
white-box sign-gradient attacker to be reported as a separate threat model
rather than compared directly with the black-box results, and
`docs/ExperimentProgress.tex`'s M3 row records this as the one still-open,
concretely GPU-blocked gap: the constrained sign-gradient optimizer
(`optimize_attacker_aware_query` in `src/scripts/run_tomm_review_proxy.py`,
already integrated into the paper for the synthetic proxy benchmark in
Appendix~F) has never been run against the real place-labeled MSLS benchmark.

**Data status (checked this session):** the full official MSLS `train_val`
metadata for all 25 cities is present locally, but only Manila and Toronto
have their per-city `images/` folders actually downloaded (other 23 cities
have only the four metadata CSVs, no images). A larger multi-city manifest
(M3 remaining gap "(i)") therefore remains genuinely data-blocked pending a
further multi-gigabyte per-city download and is not attempted in this round;
it stays a documented limitation. This experiment scopes to gap "(ii)" only,
which needs no new data acquisition.

**Design:** extend `run_geotagged_vpr_benchmark.py` with an `attacker_aware`
variant that mirrors the proxy's protocol exactly (same optimizer, same
default steps=20/step_size=1.0/linf=8.0, ResNet18 only, since the proxy
result this must stay comparable to was itself ResNet18-only). For each
query, the target embedding is the first gallery item sharing the query's
official `place_id` (in `place_id` order, deterministic). The white-box
optimizer needs the un-batched gallery embedding matrix up front to pick this
target, which is computed via the already-chunked `normalized_embeddings`
helper (32-image chunks) rather than `eval.retrieval_attack.build_gallery_embeddings`,
because the latter does one unbatched forward pass over the full gallery and
is the same OOM pattern already fixed once for the geotagged 1000-image
gallery (commit `c7dde50`); reusing the chunked path avoids reintroducing it,
even though ResNet18 alone is unlikely to trip it.

**Completion gate:** every attacker_aware row must have finite PSNR/MSE and a
finite retrieval margin; `run_metadata.json` records
`attacker_backbone`, `attacker_steps`, `attacker_step_size`, `attacker_linf`,
and keeps `scientific_evidence=true` only for this being a real MSLS run (not
smoke). This result is written into the paper as a separate white-box
diagnostic (an upper bound under adaptive knowledge of the exact release
mechanism and target gallery item), explicitly not pooled with or compared
against the black-box backbone-transfer table, per the review's instruction.

**Smoke test (local RTX 3070, synthetic + tiny real subset):** verify the new
variant dispatch, target-selection logic, and output schema on synthetic
tensors, then on the first 5 real queries of the existing `manifest_all`
MSLS manifest with `--seeds` limited to one dummy value (the variant is
deterministic and ignores seed). This is an integration check only;
`scientific_evidence=false` for the smoke output, and no number from it may
reach the paper.

**Full run (vGPU 3090, pending GPU-on):** run `--include_attacker_aware` for
`resnet18` against all three existing gate-passing manifests
(`manifest_all.jsonl`, `manifest_o2n.jsonl`, `manifest_n2o.jsonl`, 200 queries
/ 1000-gallery each, already used for the M3 six-backbone table), alongside
the existing black-box variant set so the two conditions are exported from
the same run. Estimated cost: 600 queries total, 20 PGD-style steps each,
ResNet18 forward+backward per step — the proxy-scale precedent for the
identical optimizer was a few minutes for 50 queries, so this is expected to
finish in well under 30 minutes of GPU time; it is bundled into the same
vGPU 3090 session as any other queued ICME work rather than justifying its
own power-on cycle by itself.

**Paper writeback:** only after the full run's completion gate passes. Add
one appendix subsection reporting attacker_aware Top-1/margin against the
raw and `full` (PPEDCRF) rows on the same manifests, with the same
diagnostic-not-comparable framing already used for the proxy version, and one
sentence in the main-text M3 discussion cross-referencing it. No change to
the black-box six-backbone headline table.

## Fourth independent TIFS review (2026-09-08): D6 and D7

Source: `docs/RevisionSuggestions.tex`, findings R3 and R4. Both experiments
re-use the direction-transfer pipeline unchanged except where stated; neither
adds a new benchmark, backbone or objective. No number from a smoke test may
reach the paper.

### D6 -- held-out attacker-side transforms for the EOT-hardened direction (R3)

**Question.** The hardened direction is optimised in expectation over
{JPEG-75, JPEG-50, blur sigma=1, NLM denoise} and, so far, evaluated only
against those four. Does the repair generalise to transforms the optimiser
never saw, or is it fitted to the training set?

**Code (done 2026-09-08, `src/eval/sanitizers.py`,
`src/scripts/run_direction_transfer_study.py`).**
- New attacker-side operators, registered in `SANITIZERS` and listed in
  `HELD_OUT`: `jpeg60` (in-family parameter, interpolation), `jpeg30`
  (beyond the trained range), `median3`, `resize_half` (bilinear down and
  up), `blur2` (sigma=2), `bitdepth4`, `random_one` (one operator chosen
  uniformly from trained+held-out, seeded by frame content) and
  `jpeg50_blur` (two trained operators stacked). `TRAINED` names the four
  the hardening saw.
- `--eval_sanitizers a b c ...` evaluates every released frame under each
  listed transform and writes one row per transform, keyed
  `(query_id, condition, seed, sanitizer)` for resume. The perturbation is
  optimised and released once per `(query, condition, seed)`; only the
  attacker's embedding step repeats, so the held-out study costs one
  optimisation pass per (backbone, seed) plus one forward pass per transform.
  Without the flag the driver behaves exactly as before.
- Tests: `src/tests/test_sanitizers.py` (shape/range/dtype for every
  operator, trained/held-out disjointness, determinism of `random_one`, and
  the optimise-once/evaluate-many path on a toy embedder at MSE 15.68).

**Runs (remote GPU; MSLS `manifest_all8.jsonl`, 400 queries, 2,000 gallery,
seeds 1234 1235 1236, `--target_mse 15.68`, `--objective self`).**
For each backbone, two invocations sharing everything but the optimiser:

```
# unhardened gallery-free direction, evaluated under every transform
python src/scripts/run_direction_transfer_study.py --manifest <all8> --root <msls> \
  --eval_backbone resnet18 --surrogates resnet50 vgg16 cosplace \
  --objective self --seeds 1234 1235 1236 \
  --eval_sanitizers none jpeg75 jpeg50 blur denoise jpeg60 jpeg30 median3 \
                    resize_half blur2 bitdepth4 random_one jpeg50_blur \
  --output src/outputs/tifs_d6/resnet18_self_multisan.csv
# hardened (EOT over the four trained transforms), same evaluation list
python src/scripts/run_direction_transfer_study.py ... --eot_sanitizers jpeg75 jpeg50 blur denoise \
  --output src/outputs/tifs_d6/resnet18_self_eot_multisan.csv
```
MixVPR: `--eval_backbone mixvpr --surrogates resnet18 resnet50 vgg16 cosplace`.
The `none` and four trained columns must reproduce the published Table 4
cells to within rounding; that is the completion gate for the run itself.

**Analysis.** `analyze_eot_sweep.py` per sanitizer, with the isotropic control
under the same transform as the paired reference, and -- per review R1 -- the
query as the unit of inference (query-cluster bootstrap CI over 400 queries
carrying all three seeds), not `(query, seed)`.

**Paper write-back.** A second block of the preprocessing table (or a compact
new table) in `paper/main.tex` for the eight held-out transforms, unhardened
against hardened, both backbones; one paragraph stating whether the repair
generalises. Either outcome is reported.

**Cost.** One optimisation pass per (backbone, seed): 400 queries x 6-7
conditions x 20 steps x 3-4 surrogates, plus 13 cheap forward passes per
released frame. Comparable to the D5 EOT run per backbone; expected to fit
one GPU session per backbone. Two hardened conditions (isotropic and white_box
are included in each run) are shared with the published table.

**Smoke test (local RTX 4050, 6 GB).** `pytest src/tests/test_sanitizers.py`
(CPU, toy embedder) plus `--limit 3 --seeds 1234` on any three MSLS queries
if the dataset is mounted; otherwise the pytest path is the gate for pushing.

### D7 -- downstream utility of the EOT-hardened direction (R4)

**Question.** The utility cost (per-image AP@50, IoU) was measured for the
unhardened surrogate direction. The paper now leads with the hardened
variant; its utility cost is unmeasured.

**Code.** `evaluate_direction_utility.py` gains the same `--eot_sanitizers`
option as the transfer driver (pass-through to `directional_delta`), and a
`hardened_direction` condition alongside `clean`, `isotropic` and
`direction`. No other change.

**Run.** Same 200-image VOC and 200-image COCO manifests, same frozen
DeepLabV3-ResNet50 and Faster R-CNN ResNet50-FPN, delivered MSE 15.68, three
independent runs, both the unhardened and hardened direction in the same run
so the paired comparison is within-run.

**Paper write-back.** One row in the supplementary utility table; one
sentence in `paper/main.tex` section The Other Axis.

**Cost.** 400 images x 4 conditions x 3 runs; minutes on any GPU. Fits the
local card.

## Fifth independent TIFS review (2026-09-08): A1-A8

Source: `docs/RevisionSuggestions.tex`, the 8 September independent review
(verdict: major revision, findings R1-R13, recommended experiments E1-E8).
The review's E-labels collide with the older TOMM E1-E7 in this file, so its
experiments are renamed A1-A8 here; the mapping to the review is stated in
each row. Nothing below has been executed at the time of writing.

### Which findings need experiments and which do not

| Finding | Nature | Route |
|---|---|---|
| R1 numerical/version drift between prose and tables | text + audit | A1, then edit `paper/` |
| R2 the allocation-cannot-decide-ranking claim is invalid as stated | text + theory | rewrite, no experiment required; A2 supports the replacement |
| R3 "oracle"/"bound" language exceeds what is measured | text + measurement | rename now; A2 measures the quantity actually claimed |
| R4 missing directly related prior work | text + baseline | cite now; A4 supplies the comparison |
| R5 inference unit, clustering, multiplicity, equivalence | analysis | A1 (place-clustered re-analysis of existing exports) |
| R6 three design factors are not isolated | experiment | A3 |
| R7 matched distortion is not a utility frontier | experiment | A5 |
| R8 adaptation claim rests on a restricted attacker | experiment | A6 |
| R9 benchmark and attacker generalisation | experiment | A7 |
| R10 Top-1 is not privacy; release boundary incomplete | analysis + experiment | A1 for Top-5/10, A8 for the serialised release |
| R11 artifact is not a complete reviewer package | packaging | extend the existing artifact builder |
| R12 method/optimiser specification | text | write the objective and operators out in the paper |
| R13 checkable interpretation and display errors | text | edit `paper/` |

### The export problem that gates A1

The D6 per-query exports the current transfer table was generated from
(`src/outputs/tifs_d6/`) are **not on this machine**: no file under
`src/outputs/` mentions any held-out transform, and the summary JSONs
`make_tifs_tables.py` reads are absent. `paper/generated/tab_transfer.tex`
and `tab_sanitize.tex` are therefore the only surviving record of those
numbers. This is the same failure mode as the `icme2027_placement_msls`
episode and it has the same two routes: recover the tree, or regenerate it.
Until one of them happens, A1 can reconcile the prose against the generated
tables (which is what R1's table of mismatches actually compares), but it
cannot recompute Top-5/Top-10 or re-cluster by place, because those need the
raw rows.

### A1 -- freeze one export per family and reconcile every number (review E1, R1/R5/R10)

**Question.** Does every number printed in the manuscript come from one
identified run, condition and inference method?

**Code.** `src/scripts/audit_claim_consistency.py` (CPU, no GPU, no dataset).
It holds a table of every headline claim in `paper/main.tex` and
`paper/supplementary.tex` -- the value, the source it must agree with, and the
sentence it appears in -- and checks each against the frozen generated tables
and, where the raw export is present, against a recomputation from it. It
prints one line per claim and exits non-zero on any mismatch, so it can run in
the build loop. The point is the same as `verify_claims.py` but one level up:
`verify_claims.py` checks table against export, this checks prose against
table.

**Analysis to add once the raw D6 rows are back.** Place-clustered bootstrap
(277 places, queries carried whole) beside the query-level interval, as the
review itself computed; Top-5 and Top-10 beside Top-1 for every direction
condition; a stated equivalence margin for each "no benefit" claim.

**Gate.** The audit script reports zero mismatches, and every negative claim
in the paper is either backed by an interval inside a stated equivalence
margin or reworded to "no benefit detected".

### A2 -- what the sensitivity maps actually measure (review E2, R2/R3)

**Question.** The paper identifies its top-decile concentration statistic with
Jacobian column norms. `attacker_gradient_map` computes the gradient of a
scalar cosine score and sums absolute RGB gradients, which is not
$\lVert J_i \rVert_2$. How far apart are they, and does the margin-variance
prediction of R2 track measured rank flips?

**Code.** `src/scripts/measure_jacobian_columns.py` (local GPU is enough).
Estimates per-pixel Jacobian column norms by random projection --
$\lVert J_i \rVert_2^2 \approx \frac{1}{K}\sum_k (J^\top v_k)_i^2$ for
$v_k$ standard normal, each term one vector-Jacobian product -- and compares
that map against the positive-score gradient and the margin gradient by
Spearman correlation, top-decile overlap and concentration. It then measures,
per query, the predicted margin variance $v(w)=\sigma^2\sum_i a_i^2 w_i^2$
against the realised margin change under the same weight map, and reports
pre-clamp and post-clamp energy.

**Gate.** Every sensitivity statistic in the paper names the quantity actually
measured. If the three maps agree closely, the existing text stands with a
renamed statistic; if they disagree, the mechanistic paragraph is rewritten
around what was measured.

### A3 -- isolate magnitude, sign and placement (review E3, R6) [primary GPU run]

**Question.** The paper compares an optimised direction against weighted
Gaussian noise and attributes the difference to direction. That comparison
changes sign pattern, magnitude pattern and clipping behaviour at once. Which
of them carries the effect?

**Design.** One optimisation per (query, seed) produces the gallery-free
direction $\delta$. From it, controls that hold delivered MSE fixed after
clipping:

| Condition | Keeps | Destroys |
|---|---|---|
| `direction` | sign and magnitude | --- |
| `sign_shuffle` | per-pixel magnitude $\lvert\delta\rvert$ | sign pattern (seeded random signs) |
| `magnitude_uniform` | sign pattern $\operatorname{sign}(\delta)$ | magnitude allocation (constant amplitude) |
| `isotropic` | budget only | both |

and the placement cross: $\delta$ reweighted by a placement map $w$
(`uniform`, `edge`, `learned`) before rescaling, which is the missing cell of
the operator/placement factorial. Every row records pre-clamp realised energy,
post-clamp delivered MSE, maximum absolute perturbation and clipped fraction,
so R6's objection that matched $\sum_i w_i^2$ is not matched realised energy
is answered with a measurement rather than an argument.

**Code.** `src/scripts/run_direction_factorial.py`, reusing
`directional_delta`, `release_at_mse` and the placement maps already in the
repository. Exports `top5_hit`/`top10_hit` alongside `correct_rank` so R10 is
covered for this family from the start.

**Run.** MSLS `manifest_all8.jsonl`, 400 queries, 2,000 gallery, seeds
1234/1235/1236, delivered MSE 15.68, both attackers (ResNet18 with three
surrogates, MixVPR with four). Six conditions x three placements is 18 cells
per (query, seed), but only four optimisations are needed per (query, seed)
because the placement and sign/magnitude variants are derived from the same
$\delta$.

**Gate.** The `direction` and `isotropic` cells reproduce the published
transfer-table values to rounding; every row's delivered MSE is within
$10^{-3}$ of target; no condition has a zero perturbation.

**Cost.** Comparable to one D6 backbone pass: the optimiser dominates and runs
once per (query, seed, surrogate set). Estimated 4-6 h per backbone on the
vGPU 3090.

### A4 -- the closest prior method, run under this protocol (review E4, R4)

**Question.** Le et al. combine a spatial mask with multi-model PGD for
location privacy. The paper's novelty claim needs that comparison rather than
an argument that no such method exists.

**Design.** Three arms at matched delivered distortion, same surrogate access,
same restart budget, same query/gallery protocol: mask-guided multi-model PGD
(perturbation confined to a mask, gradient averaged over the surrogate
ensemble), full-frame multi-model PGD (this paper's direction), and the
isotropic control. The mask is the one this repository can compute without
new dependencies -- the segmentation background map already used as a
published-model placement -- and the deviation from Le et al.'s CAM mask is
recorded as an adaptation, not hidden.

**Code.** `src/scripts/run_maskguided_pgd_baseline.py`.

**Gate.** All three arms deliver the same MSE to $10^{-3}$; the report states
the task adaptation (gallery ranking rather than scene classification)
explicitly. Either outcome is written up.

**Cost.** Same order as A3 for one backbone; 2-3 h on the vGPU 3090.

### A5 -- privacy against measured utility, over budgets (review E5, R7)

Sweep delivered MSE over four points spanning the claimed operating regime and
report retrieval privacy against detector/segmenter metrics for isotropic,
edge placement, gallery-free direction and the EOT-hardened direction, with a
declared utility tolerance. `evaluate_direction_utility.py` gains
`--target_mse` as a sweep and the hardened condition it already supports.
Cross-dataset transfer (MSLS retrieval, VOC/COCO utility) is labelled as such.
Local GPU is sufficient for the utility half; the retrieval half rides on A3.

### A6 -- adaptation against the final hardened release (review E6, R8)

Fine-tune the attacker on **hardened** releases with fresh randomness, add a
gallery-rebuilding configuration, select on a larger and more informative
validation set, and evaluate on a place-disjoint test split, reporting
Top-1/5/10 and rank statistics. Extends `finetune_adaptive_attacker.py`.
vGPU 3090, one session.

### A7 -- external validity (review E7, R9)

Replicate the decisive contrast on a geographic holdout (city-disjoint split
of the existing 8-city manifest, which needs no new download) and on one more
held-out strong VPR backbone, and audit cluster-based positives against
GPS/heading positives from the MSLS metadata already on disk. vGPU 3090.

### A8 -- what is actually released (review E8, R10)

Save each released frame as PNG and as JPEG at a stated quality, decode it
back, and re-measure delivered MSE, maximum absolute perturbation, clipped
fraction and retrieval outcome. This closes the gap between the float tensor
the optimiser produces and the file a deployment would transmit, and it checks
whether `release_at_mse`'s post-optimisation scaling can push the perturbation
past the $\ell_\infty$ projection that preceded it.
`src/scripts/validate_serialized_release.py`; local GPU is sufficient.

### Execution order

1. A1 (CPU) and the R1/R13/R2/R3/R4/R12 text edits -- no GPU, done first,
   because R1 says not to launch new benchmarks until the claims are stable.
2. A8 and A2 on the local RTX 3070 -- both are small.
3. A3 then A4 on the vGPU 3090, in that order: A3 is what R6 asks for and
   also regenerates the direction cells that A1 needs at Top-5/10.
4. A5, A6, A7 as a second GPU session, scoped by what A3/A4 return.

### Writeback rules for this cycle

No number enters `paper/` before its run passes the gate stated above. The
published transfer table is not overwritten by A3: A3's `direction` and
`isotropic` cells are a reproduction check, and if they disagree with the
published values beyond rounding, that disagreement is itself reported rather
than quietly replacing the table. Negative outcomes from A3-A8 are written up
in the same detail as positive ones.
