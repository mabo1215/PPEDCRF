# ACM TOMM Review Letter

**Manuscript title:** “Dynamic-CRF-Guided Selective Perturbation for Background-Based Location Privacy in Video Sequences”  
**Associate Editor:** Professor Feifei Zhang, Associate Editor, ACM TOMM  
**Editorial decision:** The manuscript was not accepted for publication.

## Editorial Decision

> I have received the reviews on your submission to the ACM TOMM titled “Dynamic-CRF-Guided Selective Perturbation for Background-Based Location Privacy in Video Sequences”. On the basis of the reviewers’ comments, I regret that I am unable to accept this paper for publication in the journal.
>
> Thanking you for submitting your work to this journal and looking forward to a more positive response in the future.
>
> Best regards,  
> Professor Feifei Zhang  
> Associate Editor, ACM TOMM

## Reviewer 1

**Recommendation:** Accept

### Comments

> Paper is well written and explained. Paper is acceptable for publication.

### Additional Questions

- **Review’s recommendation for paper type:** Full length technical paper
- **Should this paper be considered for a best paper award?** No
- **Does this paper present innovative ideas or material?** Yes
- **In what ways does this paper advance the field?** There is novelty work found in the paper.
- **Is the information in the paper sound, factual, and accurate?** Yes
- **If not, please explain why:** —
- **What are the major contributions of the paper?** Author has new concept related to video analytics which has high impact on society.
- **Rate how well the ideas are presented:** 4/5
- **Rate the overall quality of the writing:** 5/5
- **Does this paper cite and use appropriate references?** Yes
- **If not, what important references are missing?** —
- **Should anything be deleted from or condensed in the paper?** No
- **If so, please explain:** —
- **Is the treatment of the subject complete?** Yes
- **If not, what important details, ideas, or analyses are missing?** —
- **Estimated copy-editing need:** Light
- **Is this paper of potential interest to developers and engineers?** Yes

## Reviewer 2

**Recommendation:** Reject

### Comments

1. The “location pairs” in the main benchmark are not built from GPS or physical place labels, but from visually similar video sequences matched using ResNet18 features. Visual similarity does not necessarily mean that two sequences come from the same location, so the current experiments mainly show that PPEDCRF can disrupt image-similarity retrieval rather than real place recognition. The appendix expands the benchmark to 50 pairs, but follows the same construction and uses COCO images as distractors, which is still quite different from a realistic geo-tagged gallery. The paper would be much stronger with at least one standard visual place recognition or geolocalization dataset containing true place identities or GPS annotations, larger galleries, and changes in viewpoint, illumination, season, and weather.

2. The paper emphasizes an approximately 6 dB PSNR gain over global Gaussian noise at the same σ₀, but the two methods do not inject the same total perturbation energy, since PPEDCRF spatially attenuates the noise. Table 6 provides the fairer comparison and shows that, at matched PSNR, PPEDCRF and global Gaussian noise achieve the same Top-1 accuracy at all three operating points. In addition, Tables 3 and 4 show almost no change after removing temporal consistency or NCP, suggesting that these two components are not yet supported by the ablation results. The authors should therefore compare methods at matched PSNR, MSE, or perturbation energy, include stronger mask-based and attacker-aware baselines, and report detection mAP and segmentation mIoU on the same sanitized images to verify practical utility.

### Additional Questions

- **Review’s recommendation for paper type:** Full length technical paper
- **Should this paper be considered for a best paper award?** No
- **Does this paper present innovative ideas or material?** Yes
- **In what ways does this paper advance the field?** —
- **Is the information in the paper sound, factual, and accurate?** Yes
- **If not, please explain why:** —
- **What are the major contributions of the paper?** The paper proposes PPEDCRF, a release-side selective perturbation framework for protecting background-based location privacy in videos. It first uses a unary predictor to estimate pixel-level location sensitivity, then applies a dynamic conditional random field (DCRF) to generate temporally smoothed sensitive-region masks. A normalized control penalty (NCP) adjusts perturbation strength within the masks, and spatially weighted Gaussian noise is injected using a differential-privacy-inspired calibration rule. The main evaluation uses a controlled paired-scene retrieval benchmark with 12 constructed query–gallery pairs and 36 hard distractors. The appendix extends the evaluation to 50 pairs, eight retrieval backbones, three gallery sizes, and three noise seeds. Detection and segmentation results on MOT16/17, Cityscapes, KITTI, and VOC2008 are retained as auxiliary utility evidence. The results show that PPEDCRF reduces retrieval accuracy for most attacker–gallery settings and preserves about 6 dB higher PSNR than global Gaussian noise at the same nominal noise scale. However, under matched-PSNR comparison, PPEDCRF and global Gaussian noise achieve nearly identical Top-1 privacy performance.
- **Strengths:**
  1. The paper addresses a privacy threat not covered by conventional face or license-plate anonymization: background cues such as buildings, roads, signs, and skylines can be exploited by visual place recognition systems to infer recording locations. This problem is directly relevant to surveillance, dashcams, bodycams, drones, and wearable-camera applications.
  2. The paper clearly defines the released data, the gallery-based retrieval attacker, and the downstream utility objective. It also states that Equation (4) does not provide a formal end-to-end differential privacy guarantee and acknowledges that the protection is attacker-dependent rather than universally robust. The discussion of adverse transfer on MixVPR further improves the objectivity of the analysis.
  3. Table 6 compares PPEDCRF with global Gaussian noise at matched PSNR rather than only at the same noise scale. The results show identical Top-1 accuracy at approximately 36, 33, and 30 dB PSNR. Although this weakens the claimed advantage, reporting it improves the transparency of the evaluation.
  4. The experiments include classification backbones, CLIP models, and dedicated visual place recognition models, together with global noise, random masks, blur, and mosaic baselines. The use of a unified benchmark construction and the same three noise seeds across eight attackers supports more reliable cross-table comparison. The appendix-scale 50-pair evaluation also reduces concerns that the main findings are entirely caused by the small 12-pair setting.
- **Rate how well the ideas are presented:** 3/5
- **Rate the overall quality of the writing:** 3/5
- **Does this paper cite and use appropriate references?** Yes
- **If not, what important references are missing?** —
- **Should anything be deleted from or condensed in the paper?** No
- **If so, please explain:** —
- **Is the treatment of the subject complete?** Yes
- **If not, what important details, ideas, or analyses are missing?** —
- **Estimated copy-editing need:** Moderate
- **Is this paper of potential interest to developers and engineers?** Maybe

## Reviewer 3

**Recommendation:** Needs Major Revision

### Comments

This paper proposes PPEDCRF, a selective perturbation framework intended to reduce background-based location retrieval from released video frames. The method is evaluated on a controlled, visually mined paired-scene retrieval proxy using eight attacker backbones. The problem is relevant; however, the following issues should be addressed:

1. The unary sensitivity predictor $g_\theta$ is fundamental to the entire framework, but its origin and implementation are not described. The authors should clarify whether it is newly designed or adapted from an existing model and provide its architecture, training data and targets, loss function, pretrained weights, and an independent evaluation of the predicted sensitivity maps.

2. Eq. (2) is not sufficiently specified for reproduction. The function $\sigma(\cdot)$ should be explicitly defined, while $S(\cdot)$ is only described as average-pooling-based smoothing without its kernel size, stride, or padding.

3. Tables 4 and 7 show almost no measurable benefit from either NCP or the temporal component. Moreover, combining Eq. (3) and Eq. (5) effectively produces a perturbation weight proportional to $(p_t)^2/(\max(p_t)+\epsilon)$, rather than a clearly defined “penalty.” The authors should provide matched-PSNR or matched-noise-energy ablations, including unary-only/no-DCRF variants, or substantially reduce the claimed contributions of NCP and temporal refinement.

4. Although the abstract acknowledges convergence at matched utility, the manuscript repeatedly emphasizes an approximately 6 dB privacy–utility advantage. Table 6 instead shows that PPEDCRF and global Gaussian noise achieve essentially identical Top-1 accuracy at matched PSNR. Although the abstract acknowledges convergence at matched utility, the manuscript still describes an approximately 6 dB privacy–utility advantage. Table 6 shows that PPEDCRF and global Gaussian noise achieve essentially identical Top-1 accuracy at matched PSNR. The approximately 6 dB gap occurs at the same nominal $\sigma_0$, rather than at matched retrieval accuracy, and should not be interpreted as an advantage at equal privacy.

5. The actual retrieval gallery is insufficiently disclosed. The manuscript does not identify the source of the 120 monitoring sequences, selected frames, query–gallery mappings, distractors, or GPS/same-location labels. Since the “synthetic location pairs” are created solely according to ResNet18 similarity, Top-k may measure recovery of an artificially assigned visual pair rather than recovery of the true geographic location.

6. The reported privacy effect may primarily represent candidate-ranking confusion within a deliberately constructed hard-negative gallery. The authors should explicitly delimit the claimed scope, evaluate both small- and large-margin locations, and report the correct-versus-hardest-negative similarity margin before and after sanitization.

7. Figure 6 provides no retrieval evidence for the illustrated example. The authors should show the corresponding correct gallery image and strongest competing images, together with the original and sanitized Top-k ranks, similarities, retrieval margins, and attacker backbone.

8. MixVPR is a dedicated visual place recognition model closely aligned with the stated threat, yet it shows adverse transfer across all three gallery sizes in the appendix. The authors should analyze this behavior, provide per-query or margin-level results, investigate possible mitigation, and qualify the claimed cross-backbone robustness accordingly.

### Additional Questions

- **Review’s recommendation for paper type:** Full length technical paper
- **Should this paper be considered for a best paper award?** No
- **Does this paper present innovative ideas or material?** Yes
- **In what ways does this paper advance the field?** The paper highlights background-based location leakage in released video frames and explores a DCRF-guided selective perturbation mechanism that concentrates noise on predicted location-sensitive regions. Its main advance is an empirical study of retrieval accuracy and perceptual quality across several attacker backbones. However, it does not establish general real-world geolocation protection.
- **Is the information in the paper sound, factual, and accurate?** No
- **If not, please explain why:** Several central claims cannot yet be adequately verified. The unary predictor and DCRF implementation are under-specified, the synthetic query–gallery pairs lack GPS or physical-location ground truth, and the reported Top-k reduction may represent ranking changes among visually similar artificial pairs rather than protection of the true geographic location.
- **What are the major contributions of the paper:**
  1. It studies background-based location leakage from publicly released video frames under a retrieval-based attacker.
  2. It proposes a pixel-space selective perturbation pipeline based on a unary sensitivity map and DCRF-style spatial-temporal refinement.
  3. It constructs a controlled proxy benchmark to examine retrieval accuracy and perceptual quality across multiple retrieval backbones and gallery sizes.
- **Rate how well the ideas are presented:** 3/5
- **Rate the overall quality of the writing:** 3/5
- **Does this paper cite and use appropriate references?** Yes
- **If not, what important references are missing?** —
- **Should anything be deleted from or condensed in the paper?** No
- **If so, please explain:** —
- **Is the treatment of the subject complete?** No
- **If not, what important details, ideas, or analyses are missing?** The unary predictor and DCRF implementation are insufficiently specified; the retrieval gallery and geographic ground truth are not disclosed; the contributions of NCP and temporal refinement are not supported by the ablations. The adverse MixVPR results are reported but remain insufficiently explained.
- **Estimated copy-editing need:** Light
- **Is this paper of potential interest to developers and engineers?** Yes
