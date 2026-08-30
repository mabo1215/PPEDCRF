# PPEDCRF ACM TOMM Revision-Cycle Experiment Plan (2026-08-30)

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
| E6 | Define Eq. (2), smoothing operator, gallery construction, and the retrieval evidence for Fig. 6 (R3-2, R3-5, R3-7) | Reproducible constants, source counts, pair/distractor IDs, per-query correct rank, strongest negative, similarities, and pre/post margins | Text revision, `retrieval_case_study.csv`, and an updated qualitative figure if the case-study gate passes |
| E7 | Analyze MixVPR adverse transfer and qualify robustness (R3-8) | Per-query and margin-level results for all gallery sizes and seeds, including the raw rank, sanitized rank, correct similarity, hardest-negative similarity, and failure category | `mixvpr_per_query.csv`, failure summary, and revised robustness wording |

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
5. **E1 geotagged benchmark.** Prefer a standard VPR/geolocalization release
   with explicit place IDs or GPS and multiple acquisition conditions. The new
   loader accepts a CSV/JSONL manifest rather than assuming a particular
   directory layout. Required fields are `query_id`, `query_path`,
   `gallery_id`, `gallery_path`, `place_id` (or latitude/longitude), and
   optional `viewpoint`, `illumination`, `season`, and `weather`. A run is
   scientific evidence only if the manifest passes uniqueness, file-exists,
   place-label, and query/gallery disjointness gates.

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
- After the code is pushed, `docs/4c_experiment_handoff.md` is the handoff
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
