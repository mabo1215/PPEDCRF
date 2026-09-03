# ICME 2027 M3.3 vGPU 3090 Handoff: White-Box Attacker on Real MSLS

This handoff covers exactly one new experiment: running the constrained
white-box sign-gradient attacker (`attacker_aware` variant, already
integrated into the paper for the synthetic proxy in Appendix~F) against the
real place-labeled MSLS benchmark. This is the concrete GPU-blocked gap
recorded in `docs/ExperimentProgress.tex`'s M3 row and
`docs/Design.md`'s "ICME-M3.3" section. It is a launch contract, not evidence
that the remote run has started. vGPU 3090 must stay off until the user
explicitly powers it on.

## Why this needs GPU time at all (and why it is small)

The optimizer itself is cheap (ResNet18 only, 20 sign-gradient steps per
query, no training). Local RTX 3070 integration smoke test on real MSLS
images (5 queries, 3 steps, `--attacker_linf 8.0 --attacker_step_size 1.0`)
took 2m45s wall time, but that run was dominated by one-time fixed cost
(model load + chunked 1000-image gallery embedding), not the per-query
optimizer. A second, 20-query/20-step timing calibration was attempted
locally but stalled in disk I/O (Linux `D` state, negligible CPU-time growth
over 5+ minutes) reading MSLS images through the WSL 9p mount to the Windows
data drive, and was killed rather than trusted as a GPU-time estimate — that
bottleneck is specific to this local sandbox's filesystem mount and is not
expected to reproduce on the remote host's local disk. Treat the per-query
optimizer cost as small (a few forward+backward passes through ResNet18 at
192x320) and expect wall time on vGPU 3090 to be dominated by ordinary image
I/O and gallery embedding, similar in profile to the already-completed M3
six-backbone benchmark on the same manifests, not by the attacker step count.

Even so, this run is queued for vGPU 3090 rather than finished locally in
this session because the user's explicit instruction for this cycle was:
local GPU for code-correctness smoke tests only, real experiments on vGPU
3090 after an explicit power-on. There is no scientific reason this couldn't
run on the local RTX 3070 instead if the user prefers that over paying for
vGPU 3090 time for a run this light (local disk I/O permitting) — flagged as
an option, not decided unilaterally.

## Prepared repository state

- Target branch: `main`
- New code: `attacker_aware` variant support added to
  `src/scripts/run_geotagged_vpr_benchmark.py` (see `docs/Design.md`
  "ICME-M3.3" section for the design). Committed and pushed as `8f623e0`
  before this handoff was written.
- Remote project root (from the last real vGPU 3090 run, 2026-09-04):
  `/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF`
- Remote MSLS manifests (already transferred and sha256-verified in the
  2026-09-04 session, should still be present unless the instance's disk was
  reset): `/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF/data/msls/manifest_all.jsonl`,
  `manifest_o2n.jsonl`, `manifest_n2o.jsonl`, with `--root` = the MSLS
  extraction root used for that transfer (check `data/msls/` for an
  `extracted/train_val/...` layout, or re-derive from
  `data/msls/manifest_all.metadata.json`).
- Checkpoint: `src/outputs/sensnet_final.pt` (sha256
  `576055d5bb173e45d29aab384abf8a0ac06e02d3f21fe3c75f66511a69d710a4` per prior
  verification).
- SSH: `ssh -p 22766 root@connect.westd.seetacloud.com` (credentials in
  `C:\source\.env`, per `.claude/rules/host-login.md`). Re-verify reachability
  before assuming this is still the live instance — vGPU rental instances can
  be recycled between sessions.

## Launch gate after the user powers on the GPU

```bash
ssh -p 22766 root@connect.westd.seetacloud.com "
cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
git fetch origin main
git checkout main
git pull --ff-only origin main
source .venv_ppedcrf/bin/activate
python -m py_compile src/scripts/run_geotagged_vpr_benchmark.py
test -f data/msls/manifest_all.jsonl
test -f data/msls/manifest_o2n.jsonl
test -f data/msls/manifest_n2o.jsonl
test -f src/outputs/sensnet_final.pt
nvidia-smi --query-gpu=memory.free --format=csv,noheader
"
```

If any file is missing (the instance may have been recycled since
2026-09-04), the MSLS manifests + extracted images and the checkpoint need to
be re-transferred from local before proceeding — do not regenerate the
manifest from a different MSLS extraction pass, since query/gallery IDs must
match the manifest already audited and used for the black-box M3 table this
result will sit alongside.

## Experiment command

Run all three manifests for ResNet18 with the existing black-box `full`
variant plus the new `attacker_aware` variant in the same invocation, so both
conditions come from one run:

```bash
cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
for SPLIT in all o2n n2o; do
  python src/scripts/run_geotagged_vpr_benchmark.py \
    --mode geotagged \
    --manifest data/msls/manifest_${SPLIT}.jsonl \
    --root data/msls \
    --config src/config/config.yaml \
    --checkpoint src/outputs/sensnet_final.pt \
    --backbones resnet18 \
    --variants full \
    --seeds 1234 1235 1236 \
    --include_attacker_aware \
    --attacker_backbone resnet18 \
    --attacker_steps 20 \
    --attacker_step_size 1.0 \
    --attacker_linf 8.0 \
    --output_dir icme2027_m33_whitebox_msls/${SPLIT} \
    > icme2027_m33_whitebox_msls/${SPLIT}.log 2>&1
  echo "EXIT_CODE=$? for ${SPLIT}"
done
```

Note the `--root data/msls` above assumes the manifest's relative paths
resolve under that directory (matching how it was audited before); if the
transferred MSLS images live under a different root, adjust `--root`
accordingly and re-run `audit_geotagged_manifest.py` first to confirm paths
resolve before spending GPU time.

## Completion gate

- Each of the three runs must exit 0 and produce `geotagged_vpr_per_query.csv`
  with `raw`, `full`, and `attacker_aware` rows for all 200 queries.
- Every `attacker_aware` row must have finite `correct_rank`,
  `retrieval_margin`, `psnr_mean`, and `effective_mse` (matches the existing
  smoke-test assertion pattern used elsewhere in this codebase).
- `run_metadata.json` must record `attacker_aware_requested: true`,
  `attacker_backbone: "resnet18"`, `attacker_steps: 20`, `attacker_linf: 8.0`.
- Generate a SHA-256 manifest for the three output directories (same pattern
  as `CHECKSUMS.sha256` in `src/outputs/icme2027_revision_20260904/`) before
  pulling results back and shutting the instance down.

## Estimated timing

No clean GPU-side throughput number was obtained locally (see above — the
local calibration attempt was I/O-bound on this sandbox's drive mount, not
GPU-bound). Based on the per-query optimizer being computationally light
(ResNet18, 20 steps, no training) and the already-completed M3 six-backbone
benchmark on the same three manifests finishing within a normal working
session on vGPU 3090, this addition (one more backbone-restricted variant on
manifests already staged and audited) is expected to take on the order of
30-90 minutes of vGPU 3090 time for all three splits combined. This is an
engineering estimate, not a measurement; record the actual wall time when
the run completes. It is small enough that it should be bundled with the
same vGPU 3090 power-on used for any other queued ICME work rather than
justifying its own session.

## Paper writeback rule

Only write results into the paper after all three splits pass the completion
gate above. Add one appendix subsection (near the existing M3
six-backbone table) with `attacker_aware` Top-1 and margin for each split,
explicitly framed as a white-box diagnostic upper bound under adaptive
knowledge of the release mechanism and the exact target gallery item — not
pooled with, or presented as comparable to, the black-box backbone-transfer
numbers, per `docs/RevisionSuggestions.tex` Additional Technical Comment 4.
One cross-reference sentence in the main-text M3 discussion. No change to the
existing six-backbone black-box headline table.
