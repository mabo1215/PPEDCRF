# vGPU 3090 handoff: experiments A3 and A4 (fifth TIFS review)

Prepared 9 September 2026, before GPU-on. Everything here is pushed; the
commands below assume the remote checkout has been fast-forwarded first.

Commit: `68bfc31` (main repo), `da55b86` (paper submodule).
Plan and rationale: `docs/Design.md`, section "Fifth independent TIFS review".

## 0. Before launching

```bash
source /etc/network_turbo            # AutoDL hosts only; skip if absent
cd /root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF
git pull                             # must reach 68bfc31
python -m pytest src/tests/ -q       # expect 30 passed
nvidia-smi                           # confirm the card and free VRAM
df -h .; df -i .                     # bytes AND inodes, per src.md
```

Data gate. Both runs need the 8-city manifest and the MSLS images:

```bash
ls src/outputs/icme2027_manifest_expanded/manifest_all8.jsonl
ls data/msls/train_val | head        # 8 cities expected
```

If the images are absent, the staged tar is on the shared volume:
`/root/autodl-fs/ppedcrf_relay_20260908/msls8.tar`
(sha256 `3f877afa1476ea0cf420fb1330ed199f274ac48512704e1b867fc404c533d3ab`,
1,596,426,240 B, 7,608 images plus `manifest_all8.jsonl`). Extract into
`data/msls`, do not copy loose files — the volume is short of inodes.

## 1. A3 — magnitude / sign / placement factorial (review R6, R10)

Four conditions crossed with two placements, all derived from one optimisation
per (query, seed). Run ResNet18 first: it is the cell the paper leads with.

```bash
mkdir -p src/outputs/tifs_a3
screen -dmS a3_resnet18 bash -c '
python src/scripts/run_direction_factorial.py \
  --manifest src/outputs/icme2027_manifest_expanded/manifest_all8.jsonl \
  --root data/msls \
  --eval_backbone resnet18 --surrogates resnet50 vgg16 cosplace \
  --seeds 1234 5678 9012 \
  --target_mse 15.68 --steps 20 --step_size 1.0 --linf 16.0 \
  --conditions direction sign_shuffle magnitude_uniform isotropic \
  --placements uniform edge \
  --output src/outputs/tifs_a3/resnet18.csv \
  > src/outputs/tifs_a3/resnet18.log 2>&1; echo EXIT=$? >> src/outputs/tifs_a3/resnet18.log'

screen -dmS a3_mixvpr bash -c '
python src/scripts/run_direction_factorial.py \
  --manifest src/outputs/icme2027_manifest_expanded/manifest_all8.jsonl \
  --root data/msls \
  --eval_backbone mixvpr --surrogates resnet18 resnet50 vgg16 cosplace \
  --seeds 1234 5678 9012 \
  --target_mse 15.68 --steps 20 --step_size 1.0 --linf 16.0 \
  --conditions direction sign_shuffle magnitude_uniform isotropic \
  --placements uniform edge \
  --output src/outputs/tifs_a3/mixvpr.csv \
  > src/outputs/tifs_a3/mixvpr.log 2>&1; echo EXIT=$? >> src/outputs/tifs_a3/mixvpr.log'
```

The objective is fixed to the gallery-free formulation (the released frame is
steered away from its own clean embedding under each surrogate) and takes no
flag, so nothing about the reference database is used.

Expected rows: 400 queries x 3 seeds x 4 conditions x 2 placements = 9,600 per
backbone. The script resumes, so a killed screen can be relaunched with the
same command.

**Completion gate.** Before any number reaches the paper:

- every row's `effective_mse` within `1e-3` of 15.68;
- no `WARNING zero perturbation` line in the log;
- the `direction`/`uniform` cell reproduces the published transfer value
  (ResNet18 0.0317, MixVPR 0.7317) to rounding, and `isotropic`/`uniform`
  reproduces 0.1967 and 0.7800. If it does not, that disagreement is the
  finding and is reported, not smoothed over.

**Cost.** One optimisation per (query, seed) — the same work as one D6
backbone pass. Estimate 4--6 h per backbone; the two screens can share the
card if VRAM allows, otherwise run ResNet18 first.

## 2. A4 — mask-guided PGD baseline (review R4)

```bash
mkdir -p src/outputs/tifs_a4
screen -dmS a4_resnet18 bash -c '
python src/scripts/run_maskguided_pgd_baseline.py \
  --manifest src/outputs/icme2027_manifest_expanded/manifest_all8.jsonl \
  --root data/msls \
  --eval_backbone resnet18 --surrogates resnet50 vgg16 cosplace \
  --arms maskguided_pgd fullframe_pgd isotropic \
  --mask_source cam --mask_coverage 0.25 \
  --seeds 1234 5678 9012 --target_mse 15.68 --steps 20 --linf 16.0 \
  --output src/outputs/tifs_a4/resnet18_cam25.csv \
  > src/outputs/tifs_a4/resnet18_cam25.log 2>&1; echo EXIT=$? >> src/outputs/tifs_a4/resnet18_cam25.log'
```

Expected rows: 400 x 3 x 3 = 3,600. Run the MixVPR variant only after the
ResNet18 arm returns, and only if A3 has finished — A4 is the second priority.

**Completion gate.** All three arms deliver the same MSE to `1e-3`; the
`fullframe_pgd` arm reproduces A3's `direction`/`uniform` cell (it is the same
optimiser without a mask); the write-up states the task adaptation
(gallery ranking, not scene classification) and the CAM substitution.

## 3. Pulling results back

```bash
tar cf a3a4.tar src/outputs/tifs_a3 src/outputs/tifs_a4
sha256sum a3a4.tar
```

Then `scp` to the local checkout, verify the digest, and run
`src/scripts/export_tree_manifest.py` on both sides before deleting anything
remote. **Add both trees to `src/artifact/build_artifact.sh` in the same
session** — three separate export trees have already been lost or nearly lost
by leaving that step for later (`icme2027_placement_msls`,
`targets_detection.json`, and the D6 tree, which is still missing).

## 4. What is deliberately not launched yet

A5 (utility over budgets), A6 (adaptation against the hardened release) and
A7 (external validity) are designed in `docs/Design.md` but not coded. They
are a second GPU session, scoped by what A3 and A4 return. A2 and A8 fit the
local RTX 3070 and do not need this host.
