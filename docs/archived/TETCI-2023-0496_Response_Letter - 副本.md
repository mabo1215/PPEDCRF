# Response to Reviewers  
**Manuscript Number:** TETCI-2023-0496  
**Title:** Privacy Enhanced Dynamic Conditional Random Field For Sequence Images Segmentation and Detection (PPEDCRF)

We thank the editors and reviewers for their constructive comments. Below we summarize the revisions made in the resubmitted manuscript and indicate where each change appears.

---

## Response to Referee 1

### 1. Title and naming (put proposed method name in the title)
The title already includes the proposed method name (PPEDCRF). No change was required.

### 2. Figure clarity (Figure 1, 3, 7 — labels and subfigure size)
We have redrawn/enlarged the figures so that labels are clearer and subfigures are larger and easier to read. The updated files have been placed in the `images/` folder and are referenced in the manuscript (Figures 1, 3, and 7).

### 3. Equation numbering (number every equation)
We have added `\label{eq:...}` to all display equations that previously lacked numbers. In particular:
- **Section II (Method):** Equations for the CRF Markov property, unary/pairwise terms, temporal update, and observation model (e.g., \ref{eq:crf_markov}, \ref{eq:crf1}--\ref{eq:crf8}) are now numbered and can be referenced.
- **Section II (RKHS/Wiener):** Equations for the kernel map, covariance, and noise process (e.g., \ref{eq:kernel_map}, \ref{eq:wiener_cov}, \ref{eq:noise}, \ref{eq:feature_noise_main}) are numbered.
- **Lemmas and theorems:** The variance lemma and theorem (e.g., \ref{eq:lem_dist}, \ref{eq:thm_var}) are numbered.
- We have checked the full manuscript and the appendix for any remaining unnumbered display equations and added labels where needed.

### 4. Formatting and typos
- **Quotation marks:** We have replaced straight or inconsistent double quotes with LaTeX-style quotes (`` and '') throughout the text (e.g., “sensitivity,” “generalization or aggregation scheme,” “Faster RCNN and Privacy-preserving deep learning,” “Objectness,” “Recall,” “Precision,” “Box,” “Classification,” “IoU,” and similar terms in Sections IV–V).
- **KL divergence punctuation:** We have checked the paragraph on NCP/attack accuracy and the discussion of KL divergence; where a sentence ended with “KL divergence,” we have changed the comma to a full stop where appropriate.
- **Missing full stop (“tree” / “The”):** This was already corrected in the previous revision: the sentence now reads “…will be a new node in the tree. The tuple is generalized…” in Section II.D (Building the hierarchical tree from the utility matrix).

---

## Response to Referee 2

### 1. Section II.D — Utility matrix and hierarchical tree
We have expanded Section II.D with a dedicated paragraph **“Building the hierarchical tree from the utility matrix.”** It now explains:
- How the hierarchical tree is built from the discernibility matrix (DM) and classification matrix (CM).
- How tree nodes correspond to sensitive/non-sensitive classes and NCP allocation.
- How this structure connects to the overall PPEDCRF pipeline (CRF inference for $M_t$, NCP for $\alpha_t$, and Eq.~\ref{eq:feature_noise} for noise injection).

### 2. Empirical evaluation — More comparisons and statistical metrics
We have added a new **subsection on “Statistical quality and privacy-budget analysis (PSNR, SSIM, and $\varepsilon$)”** that:
- Reports PSNR, SSIM, and MSE between original and protected frames (Table~\ref{tab:quality}).
- Links the discussion to the privacy budget in Section I (e.g., $\varepsilon \to 0$ vs.\ $\varepsilon \geq 1$).
- Compares PPEDCRF with baseline methods (global Gaussian noise, white-noise masking, and feature-based anonymization) on these metrics.
- Explains that PPEDCRF achieves a better trade-off (higher PSNR/SSIM for comparable or better Top-$k$ privacy).  
The table is filled with experimental values obtained from our driving/clip dataset (or as described in the caption).

### 3. Proofreading
The sentence “If the predictions deviate beyond the ground, loss or cost values will increase.” has been changed to: “If the predictions deviate **significantly from the ground truth**, the loss or cost values will increase.” We have also performed a general proofread of the manuscript.

---

## Response to Referee 3

### 1. Motivation vs. encryption
We have added a short paragraph **“Motivation: why not encryption?”** in the Introduction that clarifies:
- Decrypted content would still be processed by the system; encryption alone does not prevent inference on the decrypted data.
- Our goal is to limit location inference while preserving utility for detection; therefore we use selective perturbation (PPEDCRF) rather than relying only on encryption.

### 2. Privacy preservation level and image-level comparison
We have revised the caption of the DeepLabV3 figure to explicitly describe the image-level comparison:
- First row: original frames and segmentation without PPEDCRF.
- Second row: protected frames and segmentation with PPEDCRF; background changes reduce location retrieval success while foreground remains detectable.

This clarifies the level of privacy preservation at the image level. We have also added a new figure (Figure~\ref{fig:sidebyside} in the manuscript) showing a side-by-side image-level comparison: (a)~original street-view frame with sensitive content visible, and (b)~the same scene after PPEDCRF protection, with a short caption explaining the level of privacy preservation
### 3. Typos and presentation (e.g., spacing before parentheses)
We have ensured consistent spacing before parentheses for abbreviations throughout the manuscript, e.g.:
- Conditional Random Field (CRF), Normalized Control Penalty (NCP), discernibility matrix (DM), classification matrix (CM), Privacy Cost (Budget), Dynamic Conditional Random Fields (DCRF), and the figure caption for PPEDCRF.

---

## Summary of Revisions

| Referee | Comment | Change in manuscript |
|--------|---------|------------------------|
| R1 | Title / method name | Already present (PPEDCRF). |
| R1 | Figure 1, 3, 7 clarity | Figures redrawn/enlarged; see `images/`. |
| R1 | Equation numbering | All display equations now have \label{eq:...}. |
| R1 | Quotation marks; KL divergence; tree/The | Quotes fixed; KL punctuation checked; tree. The fixed. |
| R2 | II.D hierarchical tree | New paragraph “Building the hierarchical tree from the utility matrix.” |
| R2 | PSNR/SSIM/MSE and $\varepsilon$ | New subsection + Table~\ref{tab:quality}. |
| R2 | Proofreading | “deviate significantly from the ground truth” and general proofread. |
| R3 | Motivation (why not encryption) | New paragraph “Motivation: why not encryption?” in Introduction. |
| R3 | Image-level privacy | Revised DeepLabV3 caption; added Figure~\ref{fig:sidebyside} (side-by-side original vs.\ protected). |
| R3 | Spacing (e.g. “Network (CNN)”) | Consistent abbreviation spacing throughout. |

We believe these revisions address the reviewers’ concerns and hope the manuscript is now suitable for publication in IEEE TETCI.

Sincerely,  
The Authors
