# ICME 2027 vGPU 3090 No-Card Handoff

This handoff prepares the experiments registered in
`docs/Design.md` for the ICME 2027 revision cycle. It is a launch contract,
not evidence that the remote experiments have started. The vGPU 3090 must
remain in no-card mode until the user explicitly powers it on.

## Prepared repository state

- Target branch: `main`
- Commit SHA: `a315880`
- Remote output root: `/data1/PPEDCRF/icme2027_revision_20260903`
- Project root: `/data1/PPEDCRF`
- Checkpoint gate: `/data1/PPEDCRF/src/outputs/sensnet_final.pt`
- Monitoring-data gate: `/data1/PPEDCRF/data/monitoring`
- Scientific-evidence policy: M2 and M4 monitoring outputs remain proxy
  evidence; M3 is paper-facing only when the official MSLS manifest and all
  place/condition gates pass.

The checkpoint, manifests, datasets, and generated outputs are not committed.
Do not copy credentials, `.env` files, private host keys, or machine-local
paths into the repository.

## Launch gate after the user powers on the GPU

Run these commands on the remote Linux host from a clean checkout:

```bash
cd /data1/PPEDCRF
git fetch origin main
git checkout main
git pull --ff-only origin main
python -m py_compile \
  src/scripts/audit_geotagged_manifest.py \
  src/scripts/run_icme2027_mask_intervention.py \
  src/scripts/run_icme2027_sequence_retrieval.py

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
test "${FREE_MIB}" -ge 40000
test -f /data1/PPEDCRF/src/outputs/sensnet_final.pt
test -d /data1/PPEDCRF/data/monitoring
mkdir -p /data1/PPEDCRF/icme2027_revision_20260903
```

If any gate fails, stop and preserve the log. Do not disable SSH host-key
verification and do not overwrite `known_hosts`.

## Experiment commands

Audit each official MSLS manifest before loading a model. Replace the example
manifest path with the registered manifest that is present on the remote host;
the audit must report the actual path used in the run:

```bash
python src/scripts/audit_geotagged_manifest.py \
  --manifest /data1/PPEDCRF/data/msls/manifest_all.jsonl \
  --root /data1/PPEDCRF/data/msls \
  --output /data1/PPEDCRF/icme2027_revision_20260903/msls_all/manifest_gate.json
```

Run M2 with all energy-preserving controls and the three registered seeds:

```bash
python src/scripts/run_icme2027_mask_intervention.py \
  --mode proxy \
  --config src/config/config.yaml \
  --checkpoint /data1/PPEDCRF/src/outputs/sensnet_final.pt \
  --monitoring_root /data1/PPEDCRF/data/monitoring \
  --num_queries 12 \
  --pair_pool_size 240 \
  --max_gallery 48 \
  --gallery_sizes 48 \
  --clip_len 4 \
  --seeds 1234 1235 1236 \
  --backbones resnet18 mixvpr cosplace patchnetvlad \
  --output_dir /data1/PPEDCRF/icme2027_revision_20260903/mask_intervention
```

Run M4 with clip lengths, pooling strategies, and a transparent proxy label:

```bash
python src/scripts/run_icme2027_sequence_retrieval.py \
  --mode proxy \
  --config src/config/config.yaml \
  --checkpoint /data1/PPEDCRF/src/outputs/sensnet_final.pt \
  --monitoring_root /data1/PPEDCRF/data/monitoring \
  --num_queries 12 \
  --pair_pool_size 240 \
  --max_gallery 48 \
  --gallery_size 48 \
  --base_clip_len 8 \
  --clip_lengths 1 2 4 8 \
  --variants full global_noise \
  --seeds 1234 1235 1236 \
  --backbones resnet18 mixvpr \
  --output_dir /data1/PPEDCRF/icme2027_revision_20260903/sequence_retrieval
```

Run M3 only after the manifest audit passes its validity checks. Keep the
coverage-gate result visible even when the manifest is narrow; a narrow but
valid manifest is a limitation, not permission to claim broad generalization:

```bash
python src/scripts/run_geotagged_vpr_benchmark.py \
  --mode geotagged \
  --manifest /data1/PPEDCRF/data/msls/manifest_all.jsonl \
  --root /data1/PPEDCRF/data/msls \
  --config src/config/config.yaml \
  --checkpoint /data1/PPEDCRF/src/outputs/sensnet_final.pt \
  --seeds 1234 1235 1236 \
  --backbones resnet18 cosplace mixvpr patchnetvlad \
  --variants full \
  --output_dir /data1/PPEDCRF/icme2027_revision_20260903/msls_all
```

## Completion gates and timing

- Local preparation is complete as of 3 September 2026, 19:30 NZST:
  Python compilation, M2/M3/M4 smoke tests, and the IEEE paper build pass.
- After GPU-on, the M3 manifest audit should finish within 15 minutes.
- M2 and M4 should finish within approximately 4 hours each, subject to
  checkpoint loading, model weights, and monitoring-data availability.
- The expanded M3 run should finish within approximately 8 hours after
  GPU-on when all four attacker weights are already cached.
- Cluster-aware summaries, checksums, and any scientifically justified paper
  writeback are estimated within one day after GPU-on.

The remote run is complete only when every required CSV/JSON export is finite,
the output metadata records the code revision and inputs, the M2 energy gate
passes, and the M3 manifest gate is auditable. Do not write smoke output,
partial output, or proxy output into the paper as real-place evidence.
