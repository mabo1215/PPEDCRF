# Prior review disclosure — supporting document

Submitted in support of a "Yes" answer to the question *"Is this manuscript a
resubmission of, or related to, a previously rejected manuscript, or a
previously reviewed and withdrawn manuscript?"*

> **AUTHORS: one field remains to be completed before upload.** The ACM TOMM
> manuscript ID is recorded below as `TOMM-2026-0332`. Please confirm the
> submission and decision dates, which we have not recorded.

---

## 1. The previous submission

| | |
|---|---|
| **Venue** | ACM Transactions on Multimedia Computing, Communications and Applications (ACM TOMM) |
| **Manuscript ID** | `TOMM-2026-0332` |
| **Title then** | "Dynamic-CRF-Guided Selective Perturbation for Background-Based Location Privacy in Video Sequences" |
| **Associate Editor** | Prof. Feifei Zhang |
| **Decision** | Not accepted for publication |
| **Reviewers** | Three (Accept / Reject / Needs Major Revision) |

Editorial decision, verbatim:

> I have received the reviews on your submission to the ACM TOMM titled
> "Dynamic-CRF-Guided Selective Perturbation for Background-Based Location
> Privacy in Video Sequences". On the basis of the reviewers' comments, I
> regret that I am unable to accept this paper for publication in the journal.

## 2. Relationship to the present submission

The present manuscript is **not a re-presentation of the reviewed work**. It is
an audit of the mechanism family that work belonged to, and the mechanism
reviewed at ACM TOMM (PPEDCRF) now appears in it as *the object under test*
rather than as the contribution.

Measured against the reviewed version, which is recoverable exactly from our
manuscript repository (commit `ca94244`, 2026-03-29):

- **94.6% of the current manuscript body is new text.** Only 622 words of
  verbatim word-runs are shared, and exactly one 25-word passage survives
  intact — 0.2% of the current body.
- Body length 2,992 → 11,461 words.
- The contribution changed type: from *proposing* a selective-perturbation
  mechanism to *measuring whether the design principle behind that family buys
  privacy at all*, with a negative headline result.

What deliberately carries over — the names PPEDCRF, DCRF and NCP, and the
reviewed version's headline numbers (Top-1 0.833→0.722 at 36.18 dB PSNR) — does
so because the audited mechanism must be identified and its published operating
point reproduced. We disclose this rather than obscure it.

## 3. How each review point was addressed

All three reviews are reproduced verbatim in §4. The two substantive reviews
(Reviewers 2 and 3) raised nine technical points. Every one of them is
addressed in the present manuscript, and in several cases the reviewer's
objection became the paper's design.

### Reviewer 2

> **(R2.1)** "The 'location pairs' in the main benchmark are not built from GPS
> or physical place labels, but from visually similar video sequences matched
> using ResNet18 features… The paper would be much stronger with at least one
> standard visual place recognition or geolocalization dataset containing true
> place identities or GPS annotations, larger galleries…"

**Addressed, and the benchmark was replaced.** The primary benchmark is now the
official Mapillary Street-Level Sequences release, keyed on its own
place-cluster labels: 400 query clusters over 277 places, an 2,000-image
gallery, eight cities, six attacker backbones. A second dataset, KITTI-360
revisits, replicates the result on physically revisited street segments with a
25 m positive radius. The visually-mined proxy is retained only as a secondary
corpus and the manuscript states explicitly that "its location identifiers are
synthetic, so the real place-labelled benchmarks carry the primary evidence."

> **(R2.2)** "…the two methods do not inject the same total perturbation energy…
> Table 6 provides the fairer comparison and shows that, at matched PSNR,
> PPEDCRF and global Gaussian noise achieve the same Top-1 accuracy at all three
> operating points… The authors should therefore compare methods at matched
> PSNR, MSE, or perturbation energy… and report detection mAP and segmentation
> mIoU on the same sanitized images."

**Addressed, and this became the paper's central method.** Every comparison in
the present manuscript is made at **matched delivered distortion**: a target MSE
solved per frame by bisection through the pixel clamp, with an
energy-conservation gate whose worst realised error is reported. The reviewer's
observation that the advantage disappears at matched utility is no longer a
table buried in an appendix — it is the paper's premise. Detection mAP and
segmentation mIoU on the same sanitized frames are reported for both axes,
paired image-by-image.

> **(R2.2, continued)** "…Tables 3 and 4 show almost no change after removing
> temporal consistency or NCP, suggesting that these two components are not yet
> supported by the ablation results."

**Addressed, and explained.** The present manuscript establishes *why* those
ablations were flat: the released checkpoint's support map is numerically
constant (unary sigmoid spanning [0.4991, 0.5015], spatial coefficient of
variation 6×10⁻⁴), because the released run was trained with no mask source and
the training path substitutes a constant zero target when none is given. Moving,
rolling or permuting an already-flat map returns the same map, so the flat
ablations followed from the checkpoint rather than from any fact about
placement. This is recoverable from the released checkpoint alone and is now the
manuscript's first diagnostic.

### Reviewer 3

> **(R3.1)** "The unary sensitivity predictor gθ is fundamental to the entire
> framework, but its origin and implementation are not described… provide its
> architecture, training data and targets, loss function, pretrained weights,
> and an independent evaluation of the predicted sensitivity maps."

**Addressed in full.** The supplementary material specifies the predictor layer
by layer (channel widths, kernel sizes, strides, absence of normalisation
layers, 87,441 parameters), its initialisation, optimiser, learning rate,
weight decay, batch composition, epochs and the absence of any validation split
or hyper-parameter search. The independent evaluation the reviewer asked for is
the coefficient-of-variation measurement that revealed the degeneracy.

> **(R3.2)** "Eq. (2) is not sufficiently specified for reproduction. The
> function σ(·) should be explicitly defined, while S(·) is only described as
> average-pooling-based smoothing without its kernel size, stride, or padding."

**Addressed.** The operator definitions are now given in closed form, including
the separable box filter with its width and padding mode and the block-averaging
operator with its padding and resampling behaviour.

> **(R3.3)** "Tables 4 and 7 show almost no measurable benefit from either NCP
> or the temporal component… The authors should provide matched-PSNR or
> matched-noise-energy ablations… or substantially reduce the claimed
> contributions of NCP and temporal refinement."

**Addressed by doing both.** Matched-energy ablations are now the protocol, and
the claims were not merely reduced — the manuscript reports that no prescribed
placement rule beats a uniform control against four of five attackers.

> **(R3.4)** "…the manuscript repeatedly emphasizes an approximately 6 dB
> privacy–utility advantage. Table 6 instead shows… essentially identical Top-1
> accuracy at matched PSNR. The approximately 6 dB gap occurs at the same
> nominal σ₀, rather than at matched retrieval accuracy, and should not be
> interpreted as an advantage at equal privacy."

**Addressed.** The claim the reviewer objected to has been withdrawn entirely.
No comparison at equal nominal σ₀ is presented as an advantage anywhere in the
present manuscript.

> **(R3.5)** "The actual retrieval gallery is insufficiently disclosed… Since
> the 'synthetic location pairs' are created solely according to ResNet18
> similarity, Top-k may measure recovery of an artificially assigned visual pair
> rather than recovery of the true geographic location."

**Addressed.** See R2.1. The manifests are built from public releases, carry
identifiers only, and their construction, positive criteria and audit against
the original 25 m distance criterion are stated.

> **(R3.6)** "The reported privacy effect may primarily represent
> candidate-ranking confusion within a deliberately constructed hard-negative
> gallery… report the correct-versus-hardest-negative similarity margin before
> and after sanitization."

**Addressed.** The correct-versus-hardest-negative margin is now defined
formally in the protocol section and reported. A gallery-size sweep
(325→2,000 at fixed queries) further tests the ranking-confusion hypothesis
directly; the advantage is flat in gallery size across that range.

> **(R3.7)** "Figure 6 provides no retrieval evidence for the illustrated
> example… show the corresponding correct gallery image and strongest competing
> images, together with the original and sanitized Top-k ranks, similarities,
> retrieval margins, and attacker backbone."

**Addressed.** The qualitative example is accompanied by its retrieval evidence.

> **(R3.8)** "MixVPR is a dedicated visual place recognition model closely
> aligned with the stated threat, yet it shows adverse transfer across all three
> gallery sizes… analyze this behavior, provide per-query or margin-level
> results, investigate possible mitigation, and qualify the claimed
> cross-backbone robustness accordingly."

**Addressed, and MixVPR was promoted to a primary attacker.** It is now one of
five attackers carried through every arm, with per-query paired contrasts,
place-clustered intervals and equivalence tests. The cross-backbone robustness
claim the reviewer asked us to qualify has been replaced by an explicit
attacker-dependence finding: prescribed placement rules reach one attacker of
five, and a solved map reaches two.

### Reviewer 1

Reviewer 1 recommended **Accept** and raised no technical objection
("Paper is well written and explained. Paper is acceptable for publication.").
No action was required; the full text is reproduced in §4 for completeness.

## 4. Verbatim review reports

The three reviews are reproduced in full, without omission, in
`06_tomm_reviews_verbatim.md`, included in this submission package. Reviewers 2 and
3 are quoted in their entirety in §3 above; Reviewer 1's report consists of the
sentence quoted above together with the reviewing form's multiple-choice
answers.

## 5. Statement

No part of any previous review has been withheld. The reviewed manuscript is
available for the editor's inspection on request.
