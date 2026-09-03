# Experiment provenance record

This record was captured on 2026-09-03 after the real-data follow-up runs on
the reachable vGPU machine. It records the exact output paths and SHA-256
values that were available for the paper-facing proxy, matched-PSNR, F1, F2,
and E4 results. Output directories are git-ignored; this file is the
lightweight cross-machine record requested by Finding F3.

## Producer and environment

- Remote repository revision: `5ccb2ac1a1c1297535fddfa27a03e2b884c06377`
- Remote checkout: `/root/autodl-tmp/ppedcrf_tomm_20260830/PPEDCRF`
- GPU: NVIDIA GeForce RTX 3090, 49152 MiB, driver 580.82.09
- Remote checkpoint: `src/outputs/sensnet_final.pt`, SHA-256
  `576055d5bb173e45d29aab384abf8a0ac06e02d3f21fe3c75f66511a69d710a4`
- Benchmark source hashes (the remote checkout's `src/` tree is untracked,
  so the file hashes are recorded in addition to the repository revision):
  `src/scripts/run_tomm_review_proxy.py`
  `fd2ee32ca4dde7e41f34d4e6bee4573fc21b9a5132eabf577036818585d2df17`;
  `src/scripts/run_controlled_retrieval_benchmark.py`
  `c3a179aee04004b144d9687457c4375c3d3933a4a5c65b20335e5676e21f25fd`;
  `src/main.py`
  `54cef92855e20d5426934051ef88f82ef6623801031625ac8a425b06e4ef5bb6`;
  `src/privacy/noise_injector.py`
  `fa4251b6e347827dedcb1d713010f044127d5fd89d5cf31a0309f9f4276846bf`.
- The F1 and F2 directories listed below were copied to the local checkout
  under the same relative paths and re-hashed locally. Their local hashes
  matched the remote values shown below.

## Reachable paper-facing exports

The following paths are relative to the remote checkout. Each row is a file,
not a directory digest.

| Artifact | SHA-256 |
|---|---|
| `src/outputs/tomm_review_proxy12/per_query.csv` | `148a78a6716171df1acd12851cc1b00cfe1a0624902ec2f2e7752ff848b63c91` |
| `src/outputs/tomm_review_proxy12/summary.csv` | `1fe9160137f0d1fdc4d930505ebe4fc54ce23713465266f775f0b4f5f945bc05` |
| `src/outputs/tomm_review_proxy12/selection.json` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` |
| `src/outputs/tomm_review_proxy12/run_metadata.json` | `bab5d62498163dea2397b286f84a4d6fd003c57364ca87bd1aeed6b9f83c385e` |
| `src/outputs/tomm_review_proxy50_split/clip_vitb32/per_query.csv` | `d77c8d27649b9814b2a269b7b294f926cfdd359557d343d987c2ae338c4dee99` |
| `src/outputs/tomm_review_proxy50_split/clip_vitb32/summary.csv` | `ea6983051eb7774c036a1ad5fba7a43b02f17ce2d4d1d85590723b844ce3b91d` |
| `src/outputs/tomm_review_proxy50_split/clip_vitb32/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/tomm_review_proxy50_split/clip_vitb32/run_metadata.json` | `2f35f2112074914cda3f452214a1f98e34a98d810f2549572027a3b39ad7d92c` |
| `src/outputs/tomm_review_proxy50_split/clip_vitl14/per_query.csv` | `f58edb2a9f949739f14100c5e0b04c88da9a7fcbb974383cadbdfd13c49d0a7b` |
| `src/outputs/tomm_review_proxy50_split/clip_vitl14/summary.csv` | `ae34522aee1f5c98ea5cc63051bf19ea676361f52e0cda80ec993a27d934edeb` |
| `src/outputs/tomm_review_proxy50_split/clip_vitl14/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/tomm_review_proxy50_split/clip_vitl14/run_metadata.json` | `e3e6de8c8e8cf8268131c4c3af91f33aa94261cd8ea1abd6eb36d7c0f787e8c8` |
| `src/outputs/tomm_review_proxy50_split/cosplace/per_query.csv` | `af7713801298c3c76359f4a1fb64b267afc7a879f283ad55abaa9c9e14eaf2c7` |
| `src/outputs/tomm_review_proxy50_split/cosplace/summary.csv` | `1486f991ead7360e90cae6e9c6737e1789291cd44417a893621e7ddebe20d957` |
| `src/outputs/tomm_review_proxy50_split/cosplace/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/tomm_review_proxy50_split/cosplace/run_metadata.json` | `669b50f32d9ae4c88b3b91d1b329a0d3488feb3bc4ccf2012a3cff3051bf2f1c` |
| `src/outputs/tomm_review_proxy50_split/mixvpr/per_query.csv` | `8d518cd560514353ff3a0277fcffc1f9542e7a861ea3ba42a91cdf2ca1c31fec` |
| `src/outputs/tomm_review_proxy50_split/mixvpr/summary.csv` | `1de7d2baa5d3264effa64dbe365b9c76467c65c1b988697765d4c09a2edf84a5` |
| `src/outputs/tomm_review_proxy50_split/mixvpr/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/tomm_review_proxy50_split/mixvpr/run_metadata.json` | `91f4561bc10cc96aaf7813403bcea2e136105fb0a0f688bfdcb223cd0edb3f73` |
| `src/outputs/tomm_review_proxy50_split/patchnetvlad/per_query.csv` | `257adb6774f60f1a3d93cb1a0cfa7ca0808f736e8fa4a8ba1a3cd8aaed622b7a` |
| `src/outputs/tomm_review_proxy50_split/patchnetvlad/summary.csv` | `dc346660fa4114070f12c81b619bf765465e7500728ac48a5a8d8eb190c70bd4` |
| `src/outputs/tomm_review_proxy50_split/patchnetvlad/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/tomm_review_proxy50_split/patchnetvlad/run_metadata.json` | `113b9c498f6bc29d233d2a8b39fb0af3ee14fdbf1a5603a0fd7a3a4effd19636` |
| `src/outputs/tomm_review_proxy50_split/resnet18/per_query.csv` | `4290f678f5c605f2f8858e21f2e7afb0835e4f8f767438f79c0027b9812e34ee` |
| `src/outputs/tomm_review_proxy50_split/resnet18/summary.csv` | `ca4d5a38328de578bf4845853ef9b5ce458945e75426aaaf7a00fa95a667cd23` |
| `src/outputs/tomm_review_proxy50_split/resnet18/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/tomm_review_proxy50_split/resnet18/run_metadata.json` | `85baf68c5d4c5107e6bc16976308535518b9338ed0611e319e5eafd228669ed3` |
| `src/outputs/tomm_review_proxy50_split/resnet50/per_query.csv` | `1a34d14110b1163600372c682b48802c805b3a365c1827fa9c4529cdff71ee57` |
| `src/outputs/tomm_review_proxy50_split/resnet50/summary.csv` | `13e0915c405028a75256f72c515eb131b75c4e8a402c4f2a1b910bf58cfa7ce7` |
| `src/outputs/tomm_review_proxy50_split/resnet50/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/tomm_review_proxy50_split/resnet50/run_metadata.json` | `ae9e49f3e402d54b1b3ca116056dcd22fb4bffe08002f6b634416a01755df982` |
| `src/outputs/tomm_review_proxy50_split/vgg16/per_query.csv` | `4dc9ee33ed55163b92f57335972926559d0baddfe68b00be1fbff2f5380e073b` |
| `src/outputs/tomm_review_proxy50_split/vgg16/summary.csv` | `cd5bbb1b4895b5585375a2a783f0d64c0844a3002c2f9617ad4adcc7c9cb67c7` |
| `src/outputs/tomm_review_proxy50_split/vgg16/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/tomm_review_proxy50_split/vgg16/run_metadata.json` | `740584c844e4c0e75d714fa7e0f0224a0bed73519a0a2727da93d932be0b9f02` |
| `src/outputs/tomm_same_image_utility/utility_summary.json` | `86a4f15caab457b5c292c567eb95fe6c426b043946f2c20ef1432231723d586c` |
| `src/outputs/tomm_same_image_utility_seg/utility_summary.json` | `d6b50be6091b39415c90e33c4caf7d06081dda41a613907e2c1013a9a724f395` |
| `src/outputs/tomm_review_provenance/provenance.json` | `3816c7f1e0807939fbd843b9a95ddccd8e8f6a37a34f18dd2f742ac54e2def0c` |

## E2/F1 matched-PSNR exports

The E2/F1 sigma sweep used ResNet18, gallery size 48, three seeds, and the
12-point set `{4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 50}`. The hashes
below cover every CSV/JSON export in the reachable sweep and the generated
paired-analysis result.

| Sigma | `per_query.csv` | `summary.csv` | `selection.json` | `run_metadata.json` |
|---:|---|---|---|---|
| 4 | `25f2ae5ad5181ad28b2efd40950a1a771b6811dc5a39ca60a85781d210a87136` | `a0c358a95b11ed47a6663b1dc98894f95038aa9cbb049941c180e7f91a069a66` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `82a47ff944d8e5e72b7a6aa00dab937d22b1ac0fb65c6e7e0a6a0f67fa22a94c` |
| 6 | `577e3efca8fc1d0b4793b6443cc99f321a41d9f4ccdef075a1cbb9da34ced2df` | `7594e80b55cb8d95aee19f71245519c08fd6a63d7d922a757b414f3ac34c8422` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `476f80d446355aa8e22ab186772a350519ed84007e2a82a3aa98a2de800f3c49` |
| 8 | `0bcbda476335522e0fed52a40ece03d5180e73416e2f277a7ec60c843a79099b` | `ea0df51010dc0fe7d7bee09c59d154ca1de15934d54686a1500ed845e6dd7268` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `8955c3564afa3318a0689485cbb3733f77ca890b19c41eaef305ac50118856e6` |
| 10 | `408553f2293d90b46c2a28c27dc9b377cbbe9d8222424eb09265d47dba76f73b` | `7690d3cc5d72ee9f2cfdc3e970875a708308ef1e8beefceb4696608ea42389ca` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `a69a9e19879f19b0df2ee949fd0231198e9c90c246676edbfc6f91b83f54f5ca` |
| 12 | `976bd313ab0d52cc37394d3d77a778d449899bfbe8f60752199d2d8cb2cee03f` | `dc8c2fce3a744a48556667914a5c6ebe112a4c6ff1e2625d6f433a679aa9407a` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `c99602f58d15f34a26350ed2515ee0467775e7c74487e194be49c7769c814a27` |
| 16 | `6345cf2b9cab541d9f426b34ea1309abc416335782e601b2cf37ee110319cdde` | `1104746c56403fe355078187a72928793205359706fee0fbce2cc001d672911a` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `95a15e211863dafe8faddb1a66445238ffac34cb2a6c04997070c4e8d4ce887b` |
| 20 | `db095d6529596a1ee4d7bcac765112072dbd0ecdb4b2024f9770f3fd83d195fa` | `95ba0f414571d2afc057df1001d1af7d806f450480ffb357c16036c734748a01` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `1b2b649f2a988b7c4ff1bfd6d632397688cb071caeee7eb92090c70113e9d2d3` |
| 24 | `1e8d1a37cd7a30eab86bdc12e78acd06c45dd5e1bf70424e4db4b9e1816e6e84` | `efef1e0be737031d1a2425ffae5434b05012a48ed87da503ce6162bff901bdff` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `800470a8940fc4fd749b73b1e4211edddc3f779911ff5c36b0f02174496e04d5` |
| 28 | `8926b358a1c1a992af53a2c8255a12e689b8406394e199e430d2bb12f895ed60` | `e362ab22bbf87e12538f79993c0ce25cc4f3250e88619fe0d74fa5b1df4ac15b` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `00becb2bcf98c4b4bcfea0210b6b1da051fd81cd8982517dabd179fea306e2fd` |
| 32 | `46faa4f6e029e1c41f28bbc7aa666f03083cc7a2eb985d09fb21449e7b2b5473` | `34f5a8b13b6a9a0d7584be15c0bf9576b82891b917d6c9d174d595069debce1f` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `05df504296c3095943fc0edc096372b71ec95940c3bc09901031d974b826188a` |
| 40 | `c1441069cdd565456775f5e9953a05961dc7d5545bfdbb3563c437dc41e3d04d` | `d5b368862d2705119f3105bbe576642ab17f797bdda164432fbe0a13b6c19dce` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `53c3691871cb9a1c6af022374e2153c7cbed3c2c43c25f297f343682553f40dc` |
| 50 | `99d71f40186a86e4ca964e8219a699115c2500ffc7e34167d2f19838c28e210f` | `775de56309701f8a4033b5b4a23d5eda4e3be0ba33422209665e38713baeefcb` | `6eac8fb5fb2c1f6373b079a7de85eedddd41b730876d64a415550f8944579bc4` | `35e8d998b171736eccdecc38ec81549fc446464ace73fd4dabf499b9aaab3005` |
| All | `matched_psnr_significance.csv` SHA-256: `43f839528a785f51bab55beba90f30cedb68606502594ed389ebb1241c516dfe` |  |  |  |

## F2 determinism exports

Both real-GPU runs used the same MixVPR proxy50 invocation, code tree,
checkpoint, gallery construction, and seeds. The checker compared the two
`per_query.csv` files and found agreement within `rtol=1e-6` and
`atol=1e-9` for all 3,750 rows and 20 columns. The identical hashes below
also show that the pulled copies and the two remote runs agree at the file
level.

| Artifact | SHA-256 |
|---|---|
| `src/outputs/f2_determinism_run_a/per_query.csv` | `d9ba41f39045752f7b209b382ac5cd982f3e1bdedf06f832175e97fc249282e2` |
| `src/outputs/f2_determinism_run_a/summary.csv` | `f18aeb7a6505d1164127c519ad8eee515c9d51870904eca762bc5226f8772988` |
| `src/outputs/f2_determinism_run_a/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/f2_determinism_run_a/run_metadata.json` | `cd3d1d83ef6c76a6521fe3f26ea52836ef820ed17f79465ea477d984c2cc8646` |
| `src/outputs/f2_determinism_run_b/per_query.csv` | `d9ba41f39045752f7b209b382ac5cd982f3e1bdedf06f832175e97fc249282e2` |
| `src/outputs/f2_determinism_run_b/summary.csv` | `f18aeb7a6505d1164127c519ad8eee515c9d51870904eca762bc5226f8772988` |
| `src/outputs/f2_determinism_run_b/selection.json` | `66611544e8ab85b9f7341e169a5269ca5919dea03f072cb94ba589d16d724867` |
| `src/outputs/f2_determinism_run_b/run_metadata.json` | `cd3d1d83ef6c76a6521fe3f26ea52836ef820ed17f79465ea477d984c2cc8646` |

## Missing exports and remaining F3 boundary

The reachable remote output tree did not contain the paper-facing E1 MSLS
manifest/results or the E5 KITTI-360 manifest/results. In particular, no
`tomm_review_e1_msls*` or `tomm_review_e5_kitti360*` directory was available
to hash. The existing `tomm_review_provenance/provenance.json` is an earlier
architecture/probe record and is not a substitute for those missing
scientific exports. F3 therefore remains partially complete until those two
export trees are restored and their file-level checksums are appended here.
