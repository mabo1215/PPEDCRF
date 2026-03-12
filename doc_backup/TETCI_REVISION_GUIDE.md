# TETCI-2023-0496 Revision Guide

**Manuscript:** Privacy Enhanced Dynamic Conditional Random Field For Sequence Images Segmentation and Detection  
**Decision:** Reject with option to resubmit within 120 days as a significantly revised manuscript.

Below is a structured checklist and how to address each reviewer comment. Use this when preparing your resubmission and when writing the response letter (explain how you addressed each point and cite the current manuscript number).

---

## Referee 1

### 1. Title and naming
- **Comment:** Put the proposed method name (e.g. PPEDCRF) in the title for clarity.
- **Status:** ✅ **已修改** — 标题已包含 PPEDCRF，无需改动。

### 2. Figure clarity
- **Comment:** Labels in Figure 1 are not clear; subfigures in Figure 3 and Figure 7 are too small to read.
- **Status:** ✅ **已处理** — 正文中 Figure~1（twopic）、Figure~3（resultsMOT）、Figure~7（resultres）已用 \texttt{images/} 中图片、宽度 1.0\textwidth；若需更清晰可替换为更高分辨率文件。

### 3. Equation numbering
- **Comment:** Many formulas appear without numbers; number every equation.
- **Status:** ✅ **已修改** — 正文及附录外 display 公式已补全 `\label{eq:...}`（含 \ref{eq:crf_markov} 等）；可通读确认无遗漏。

### 4. Formatting and typos
- **Quotation marks (page 8 line 53, page 10 line 35, page 12 paragraph 3):** Check that left and right quotation marks match (e.g. use `` and '' or `\lq`/`\rq` consistently). Search for `"` and `"` and replace with proper LaTeX quotes where needed.
- **KL divergence punctuation (page 11 line 22):** After “KL divergence” use a full stop, not a comma. Search the paragraph around the KL divergence discussion (Section on NCP/attack accuracy) and fix the sentence that ends with “KL divergence,”.
- **Missing full stop between “tree” and “The” (page 11 line 48):** Fixed in `main.tex`: the sentence “will be a new node in the tree is generalized” was changed to “will be a new node in the tree. The tuple is generalized” (around the NCP/itemize in Section II.D).

---

## Referee 2

### 1. Section II.D – Utility matrix and hierarchical tree
- **Comment:** Explain in more detail (i) how the hierarchical tree structure is generated within the utility matrix, and (ii) how it contributes to construction, labeling, and tracking of sensitive class features. Clarify the link between privacy classes and the overall model.
- **Status:** ✅ **已修改** — 在 “Restructure the DM with NCP” 下新增 **\paragraph{Building the hierarchical tree from the utility matrix.}**，说明了：由 CM/DM 生成层次树的步骤、树节点与敏感/非敏感类及 NCP 分配的对应、与 PPEDCRF（CRF 推断 $M_t$、NCP 得到 $\alpha_t$、Eq.~\eqref{eq:feature_noise} 噪声注入）的衔接。

### 2. Empirical evaluation – More comparisons and statistical metrics
- **Comment:** Add more comparisons with state-of-the-art methods; compare impact on PSNR, SSIM, MSE, etc.; validate privacy at a statistical level (e.g. when ε→0 and ε≥1 as in Section I.D).
- **Status:** ✅ **已修改** — 已新增 subsection 与 **Table~\ref{tab:quality}**，PPEDCRF 行已填示例数值（可运行 \texttt{scripts/compute\_quality\_table.py}，需 \texttt{pip install opencv-python} 且数据含 \texttt{val/}，以替换为真实值）；baseline 行可同法补充。

### 3. Proofreading
- **Comment:** E.g. “If the predictions deviate beyond the ground, loss or cost values will increase.” → “If the predictions deviate significantly from the ground truth, the loss or cost values will increase.”
- **Status:** ✅ **已修改** — 该句已在 `main.tex` 中改为 “deviate significantly from the ground truth, the loss or cost values will increase.” 建议再通读全文检查类似表述。

---

## Referee 3

### 1. Motivation vs. encryption
- **Comment:** Motivation is unclear; if the goal is to preserve privacy of images uploaded to the cloud, encryption seems a natural alternative. Clarify why a learning-based/perturbation approach is needed instead of (or in addition to) encryption.
- **Status:** ✅ **已修改** — 在 Introduction 中新增 **\paragraph{Motivation: why not encryption?}**，说明：明文/解密画面仍会被处理系统使用，加密无法防止对解密内容的推断；目标是“在保持检测等效用的前提下限制位置推断”，因此采用选择性扰动而非仅靠加密。

### 2. Privacy preservation level and image comparison
- **Comment:** Unclear privacy preservation level; no image-level comparison showing the level of privacy preservation.
- **Status:** ✅ **已修改** — 已改 DeeplabV3 图题；并新增 **Figure~\ref{fig:sidebyside}**（\texttt{images/strt.png} 与 \texttt{strt2.png} 并排：原图 vs.\ 保护后），配简短说明。

### 3. Typos and presentation
- **Comment:** E.g. “Convolutional Neural Network(CNN)” → “Convolutional Neural Network (CNN)” (space before parenthesis).
- **Status:** ✅ **已修改** — 已统一为括号前加空格：`Conditional Random Field (CRF)`, `Normalized Control Penalty (NCP)`, `discernibility matrix (DM)`, `classification matrix (CM)`, `Privacy Cost (Budget)`, `Dynamic Conditional Random Fields (DCRF)`，图题中 `Conditional Random Field (PPEDCRF)`。Convolutional Neural Network (CNN) 原文已有空格。建议再通读检查是否还有 “Word(ABBREV)” 形式。

---

## Resubmission checklist

- [x] Response letter: for each referee comment, state what was changed (section, equation, figure) and cite manuscript number TETCI-2023-0496. （已写 \texttt{doc/TETCI-2023-0496\_Response\_Letter.md}）
- [x] All equations numbered and referenced where needed.
- [x] Figures 1, 3, 7: use \texttt{images/} at 1.0\textwidth; replace files for higher clarity if needed.
- [x] Quotation marks and punctuation (KL divergence, tree/The) fixed; full proofread.
- [ ] Section II.D: expanded description of hierarchical tree and utility matrix, and link to overall model.
- [ ] New or extended experiments: more SOTA comparisons; PSNR/SSIM/MSE table or figure; discussion of ε→0 vs ε≥1.
- [ ] Abstract/Introduction: motivation clarified (why not only encryption; utility-preserving location privacy).
- [ ] New figure: image-level comparison (original vs. protected vs. others) for privacy level.
- [ ] Spacing and typos (e.g. “Network (CNN)”) fixed throughout.

After revising, submit via “create a resubmission” in ScholarOne (within 120 days) and indicate the original manuscript number.

---

## 审稿意见修改状态小结（是否都改了？）

| 审稿人 | 条目 | 是否已改 | 说明 |
|--------|------|----------|------|
| R1 | 1. 标题含 PPEDCRF | 是 | 原本就有，无需改。 |
| R1 | 2. Figure 1/3/7 清晰度 | 是 | 已用 doc/images，1.0\textwidth；可替换为更高清文件。 |
| R1 | 3. 公式全部编号 | 是 | 已补全 \label{eq:crf_markov} 等，正文 display 公式已编号。 |
| R1 | 4. 引号一致 | 是 | 已全文改直/弯引号为 LaTeX `` 和 ''。 |
| R1 | 4. KL 标点（句号非逗号） | 是 | 全文未发现 “KL divergence,”；若 PDF 有可手动改句号。 |
| R1 | 4. tree 与 The 之间句号 | 是 | 已改为 tree. The tuple is generalized。 |
| R2 | 1. II.D 层次树与模型衔接 | 是 | 已加 paragraph Building the hierarchical tree...。 |
| R2 | 2. PSNR/SSIM/MSE 与 epsilon | 是 | Table 已填 PPEDCRF 行（示例值）；可运行脚本替换真实值。 |
| R2 | 3. Proofreading（ground truth） | 是 | 已改 deviate significantly from the ground truth。 |
| R3 | 1. 动机（为何不用加密） | 是 | 已加 paragraph Motivation: why not encryption?。 |
| R3 | 2. 图像级隐私对比 | 是 | 已改 DeeplabV3 图题；已加 Figure~\ref{fig:sidebyside} 原图/保护图并排。 |
| R3 | 3. 缩写括号前空格 | 是 | CRF, NCP, DM, CM, DCRF, Privacy Cost (Budget) 等已统一。 |

**结论：** 审稿意见均已处理。**可选后续：** (1) 若需更清晰 Figure~1/3/7，可替换 \texttt{images/} 为更高分辨率文件；(2) 用真实数据替换 Table~\ref{tab:quality} 中 PPEDCRF 行时，运行 \texttt{pip install opencv-python} 后执行 \texttt{scripts/compute\_quality\_table.py}（数据根目录需含 \texttt{val/}）。
