# TETCI-Style Reviewer Report (Simulated)

**Manuscript:** PPEDCRF: Privacy-Preserving Enhanced Dynamic CRF for Location-Privacy Protection for Sequence Videos with Minimal Detection Degradation

**Recommendation:** Minor Revision (accept after addressing the following)

---

## Summary
The paper addresses location privacy in dashcam videos under background-based retrieval attacks and proposes PPEDCRF, a dynamic CRF with NCP and feature-space noise injection. The topic is relevant to TETCI; the method is technically sound but several clarity and presentation issues should be fixed.

---

## Major Comments

**1. Section 1.3 "Research Range"**  
The heading is non-standard. Suggest renaming to **"Scope and Motivation"** or merging with Related Work so the scope is clear and the narrative flows.

**2. Post-Contributions Paragraph (Section I)**  
The paragraph starting with "As mentioned above, Gaussian distribution-based..." and the long sentence "This means that the scaling ... cannot exceed any bounded interval. Based on this assumption, have been classified ..." is grammatically broken (missing subject in "have been classified") and mixes contribution summary with technical detail. Move the technical ($\varepsilon$, $\sigma$) content to Section II and keep only a short bridge sentence after the contribution list.

**3. Link Between Eq. (1) and Eq. (2)**  
Equation (1) defines a noise update $n_{t+1}$ in a Gaussian form; Eq. (2) uses $\tilde{z}_t = z_t + M_t \odot \epsilon_t$. The relation between $n_t$/$n_{t+1}$ and $\epsilon_t$ is not stated. Please add one sentence clarifying that $\epsilon_t$ is drawn from a distribution whose scale is derived from the calibrated $n_t$ (or NCP), so readers see how the privacy-noise formulation connects to feature-space injection.

**4. Algorithm 1 Procedure Name**  
"PPEDCRF\_FutureNoise" appears to be a typo; it should be **"PPEDCRF\_FeatureNoise"** to match the caption "Feature-space Noise Injection."

**5. Figure 2 Caption**  
"Artecture" should be **"Architecture"** in the caption (done). For consistency, consider renaming the figure file to ``architecture_of_solution.png'' and updating the \texttt{includegraphics} path; the caption has been corrected.

**6. Table 2 (PSNR/SSIM)**  
PPEDCRF ($\sigma_0=8$) and "White-noise mask" show identical PSNR (36.19) and SSIM (0.9613). If this is expected (e.g., same effective noise level), add one sentence in the caption or in the text explaining why they coincide; otherwise please verify the numbers and differentiate the methods.

**7. NCP Effect (Fig. 18 / Fig. labelMOT)**  
The text states "the average privacy-to-privacy nearest-neighbor distance **decreases** from 0.219 to 0.218." A decrease of 0.001 would mean samples are *closer* after NCP, which would *increase* retrieval success. If the intended effect is that NCP *increases* distance (reduces attack success), the sentence should say "**increases**" and the numbers should be checked (e.g., 0.218 → 0.219 or a larger gap). Please correct for consistency with the claim that NCP reduces Top-$k$ retrieval success.

**8. Reproducibility**  
You mention "standard deviation when space permits." For at least one main table (e.g., detection mAP or Top-$k$ accuracy), report mean ± std (e.g., over 3 seeds) to support reproducibility and strengthen the evaluation.

---

## Minor Comments

**9. Terminology**  
Use "video sequences" or "sequential videos" instead of "sequence videos" in the title/abstract for natural English.

**10. Definition 1**  
"which can descript" → **"which can describe"**.

**11. Lemma 1 (Transfer function)**  
"In transfer function the cumulative sum" → **"In the transfer function, the cumulative sum"** (add article and comma).

**12. Conclusion**  
"Our solution has been applied to effectively protect" → **"Our solution effectively protects"** for consistent present tense in conclusions.

**13. Consistency**  
Use **"Fig."** consistently (not "Figure") when referring to figures, per IEEE style.

**14. Optional Step 5**  
Clarify in the pipeline description whether Step 5 (location matching check) is used in all experiments or only in ablation; this avoids confusion about the evaluation protocol.

---

## Strengths
- Clear threat model and motivation for location (vs. identity) privacy.
- Combination of DCRF, NCP, and feature-space noise is novel and well motivated.
- Good range of datasets (MOT16/17, Cityscapes, KITTI) and comparison with baselines.
- Algorithm 1 and equations are mostly clear; Lemma/Theorem on masked noise are useful.

---

## Summary of Required Revisions
1. Rename or merge "Research Range" (Major).
2. Fix post-contributions paragraph: grammar and move technical content to Section II (Major).
3. Clarify link between Eq. (1) and Eq. (2) (Major).
4. Fix "Future" → "Feature" in Algorithm 1 (Major).
5. Fix "Artecture" → "Architecture" in caption/figure (Major).
6. Explain or correct Table 2 PSNR/SSIM coincidence (Major).
7. Fix "decreases" → "increases" (or correct numbers) for NCP distance (Major).
8. Report std in at least one main result table (Major).
9. Apply minor language and consistency edits (Minor).
