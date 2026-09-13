# Submission package — IEEE TIFS

Assembled 2026-09-14 from the build at `paper/`. Every PDF here is byte-identical
to the corresponding file in `paper/` (md5-verified at assembly).

## What to upload

| File | Contents | Pages |
|------|----------|-------|
| `01_manuscript.pdf` | Main manuscript | 13 |
| `02_supplementary.pdf` | Supplementary material | 11 |
| `03_titlepage.pdf` | Title page (authors, affiliations) | 1 |
| `04_cover_letter.txt` | Cover letter, incl. the over-length request | — |
| `source/` | LaTeX sources, `ref.bib`, used figures, generated tables, build script | — |

`source/figs/` carries only the four figures the documents reference, not the
whole 26 MB `paper/figs/` tree. `source/generated/` carries the table files the
documents `\input`; they are generated from the released per-query exports and
should not be hand-edited.

## NOT part of this submission

`submit/` already contained these before the package was assembled. They are
KITTI-360 **dataset download helpers**, unrelated to the paper, and were left in
place rather than deleted:

- `README.md` (describes KITTI image downloads, *not* this submission)
- `download_2d_perspective.sh`
- `download_2d_perspective_unrectified.sh`
- `download_2d_perspective.zip`

**Do not zip this directory wholesale for upload** — move those four aside
first, or upload the five numbered items individually. Note in particular that
`README.md` would otherwise reach an editor as if it described the submission.

## Build state at assembly

- 13 / 11 / 1 pages; 0 undefined references, 0 overfull boxes, 0 font warnings
- 911 registered claims, 911 verified against the released per-query data,
  0 mismatched
- Bibliography: 39 cited references, none uncited

## EDICS

TIFS asks for one primary and, usually, up to three additional categories.
Recommendation, grounded in the declared keywords ("Location privacy, visual
place recognition, image sanitization, adversarial perturbation,
privacy--utility trade-off, evaluation protocol") and in what the paper
actually measures.

**Primary — `IF-PRV-PROT` Privacy protection.** The object under study is a
family of visual location-privacy mechanisms, and the contribution is a
protocol for deciding whether their design choice buys protection. This is the
category a reviewer of that claim should be drawn from.

**Additional, in order of fit:**

1. `IF-ML-ADVE` Adversarial machine learning. The direction axis is an
   adversarial perturbation over a surrogate ensemble with expectation over
   transformations and a straight-through backward pass, and the evaluation
   follows the adaptive-evaluation methodology line explicitly (preprocessing
   attacker, fine-tuned attacker, purifying attacker). Roughly a third of the
   evidence is adversarial-ML in method.
2. `IF-PRV-ATTC` Privacy attacks. The paper builds the attackers as much as it
   evaluates the defense: five retrieval backbones, a gallery-free geolocator,
   a vision--language geolocator, an adaptive fine-tuned attacker at two
   unfreeze budgets, and a purifying denoiser.
3. `IF-SUR-PRIV` Privacy in surveillance. The motivating setting is
   mobile-recorded video from dashcams, bodycams, drones and wearable cameras
   uploaded for safety investigation and incident auditing.

**At the top level**, if only the eight broad headings are offered: choose
**Anonymization and Data Privacy** as primary, then **Machine learning for
Information Forensics and Security** and **Surveillance**.

**Deliberately not selected.** *Biometrics* is the near miss and should be
avoided: that literature is foreground-centric (faces, gait, iris) whereas
this paper states in its related work that "the sensitive signal here lives in
the background scene", and no experiment here touches a person. Selecting it
would route the paper to reviewers whose expertise does not bear on the claim.
*Applied cryptography*, *Watermarking and data hiding*, *Multimedia forensics*
and *Cybersecurity* have no corresponding contribution: nothing here is
encrypted, embedded, tamper-detected, or system-level.

## Submission questionnaire

Four of these are technical and answered below from the manuscript. Five are
factual declarations about the authors' publication history, which the
repository cannot settle — those carry a **recommended** answer plus the
evidence behind it, and must be confirmed by the authors before submission.

### Q1. Resubmission of, or related to, a previously rejected or withdrawn manuscript?

**Recommended answer: Yes.** This requires the authors' confirmation, and it
is the highest-risk item on the form — the question states that withholding
relevant information from previous reviews *may cause immediate rejection*,
and SPS policy allows only one resubmission after a rejection.

The evidence in this repository is `docs/TOMM_Response_Letter.md`, which
records:

- **Venue:** ACM Transactions on Multimedia Computing, Communications and
  Applications (ACM TOMM)
- **Title at that time:** "Dynamic-CRF-Guided Selective Perturbation for
  Background-Based Location Privacy in Video Sequences" (differs from the
  current title)
- **Decision, verbatim:** "The manuscript was not accepted for publication."
- **Three reviewer reports**, quoted verbatim in that file

Note the question covers manuscripts "related to", not only identical
resubmissions, so the change of title and the substantial rework since do not
by themselves make the answer No.

Two things the authors must supply, which are not in the repository:

1. The **ACM TOMM manuscript ID** — no ID appears anywhere in the saved letter.
2. The **supporting document**: verbatim quotations of all relevant parts of
   all three reviews, plus how each has been addressed. `TOMM_Response_Letter.md`
   already contains the verbatim reviews and can serve as its basis, but the
   "how addressed" mapping must be written against the *current* manuscript,
   which has changed substantially (fourteen internal review cycles since).

### Q2. Extended version of a conference publication?

**Cannot be answered from the repository — authors must answer.** No
conference version of this work is cited in the manuscript, and no draft,
submission record or correspondence for one exists in the repository. If the
answer is Yes, the manuscript must cite it and explain the additional
contribution; nothing of that kind is currently in the text.

### Q3. Related to other papers by the authors, published, accepted, or under review, not cited here?

**Cannot be answered from the repository — authors must answer.** The
repository contains no record of the authors' other work. Note that the ACM
TOMM submission in Q1 is a prior version of *this* work rather than a separate
paper, so it belongs under Q1; if any genuinely separate paper shares
material, it belongs here.

### Q4. Preprints identical to this submission?

**Recommended answer: No — to be confirmed by the authors.** No arXiv,
TechRxiv or institutional-repository posting of this manuscript appears
anywhere in the repository, and the manuscript cites no preprint of itself.
Only the authors can confirm nothing was posted outside it.

### Q5. Other posted preprints that should not be considered prior art?

**Recommended answer: No — to be confirmed by the authors.** This follows
from Q4; if the answer to Q4 changes, revisit this.

### Q6. Why is this contribution within the scope of IEEE TIFS?

The manuscript is an evaluation of **visual privacy protection mechanisms**
and of the attacks that defeat them, which is squarely within *Anonymization
and data privacy* and *Privacy in surveillance*.

The privacy problem is concrete rather than nominal. Mobile-recorded video
from dashcams, bodycams, drones and wearable cameras is routinely uploaded for
safety investigation and incident auditing; stripping GPS metadata does not
remove the location, because the background of a single frame can be matched
against large-scale geo-tagged databases. The threat modelled is therefore a
location-inference adversary, and the defenses evaluated are published
image-sanitization mechanisms of the kind this journal publishes.

The machine-learning content is applied to that problem rather than studied
for its own sake, which is the condition the scope note attaches to
technique-related topics. The perturbations are adversarial examples over a
surrogate ensemble; the attackers are five retrieval backbones, two
image-to-GPS geolocators, a fine-tuned adaptive attacker and a purifying
denoiser; and the evaluation follows the adaptive-evaluation methodology this
field requires of any perturbation-based defense. Every result is a statement
about whether a privacy mechanism protects a released frame.

### Q7. Why is the contribution significant?

**It removes a design assumption the field currently spends its effort on.**
Visual location-privacy mechanisms are largely designed by choosing *where* to
spend a bounded distortion budget — learned support maps, saliency, edges,
segmentation-derived masks. Under a protocol that fixes delivered distortion
and varies one factor at a time, that choice buys nothing against four of five
attackers, across nine placement rules, three published-model placements and
four operators, with every difference inside 0.012 Top-1. Effort spent on
allocation is, for those attackers, effort spent on a variable that does not
move the outcome.

**It identifies what does work, and bounds it honestly.** The leverage is in
the perturbation's *direction*, which transfers from the released frame alone
to all five attackers and survives a change of attacker class the paper's own
mechanistic account does not predict. The paper then states the limits: a
fully unfrozen, re-indexing adaptive attacker recovers about half of it; a
trained denoiser recovers half to all; and against the strongest public
geolocator the release degrades city-level localisation rather than preventing
it. Both of those qualifications run against the authors' interest and are
reported anyway.

**It supplies a reusable evaluation protocol and a diagnostic.** The
matched-delivered-distortion design, the energy-conservation gate, the
allocation-matched control and the solved-map control are the apparatus a
future mechanism can be tested with. The protocol's first output on the
mechanism under test was that its released support map is numerically
constant — a defect recoverable from the released checkpoint, invisible to
every check this literature currently applies, and one that still produced a
plausible privacy--utility curve. That failure mode is the practical case for
the protocol.

### Q8. The three most closely related published papers

1. **X. Liu, X. Jia, Y. Xun, S. Qin and X. Cao**, "GeoShield: Safeguarding
   geolocation privacy from vision-language models via adversarial
   perturbations," in *Proc. AAAI Conf. Artif. Intell.*, vol. 40, 2026,
   pp. 35653–35661. DOI: `10.1609/aaai.v40i42.40877`
2. **T.-N. Le, T. Gu, H. H. Nguyen and I. Echizen**, "Rethinking adversarial
   examples for location privacy protection," in *Proc. IEEE Int. Workshop
   Inf. Forensics Security (WIFS)*, 2022, pp. 1–6.
   DOI: `10.1109/WIFS55849.2022.9975388`
3. **E. Radiya-Dixit, S. Hong, N. Carlini and F. Tramèr**, "Data poisoning
   won't save you from facial recognition," in *Proc. Int. Conf. Learn.
   Represent. (ICLR)*, 2022. *(No DOI recorded in our bibliography; ICLR
   proceedings are distributed via OpenReview. Fill in from the publisher
   record if the form requires one — we have not invented an identifier.)*

### Q9. What is distinctive about this paper relative to those three?

**Against GeoShield (1).** GeoShield proposes a mechanism that localises
revealing regions and perturbs them; this paper asks whether that localisation
is what buys the protection. We run GeoShield's public release end to end
under our protocol against an isotropic perturbation of identical delivered
energy, and it does not separate from that control (−0.010 Top-1,
[−0.041, +0.020]). We also report, as a reproducibility observation, that the
public release's vision-language component returns one constant caption for
every image, so the arm measures the released artefact rather than the method
as described. GeoShield contributes a mechanism; we contribute a measurement
of whether its defining choice does work.

**Against Le et al. (2).** Le et al. confine a multi-model projected-gradient
perturbation to a class-activation mask, the closest published formulation to
our direction axis. Their contribution is the formulation; ours is the
controlled comparison it was never subjected to — the mask against no mask at
matched delivered distortion and identical surrogate access. Across three mask
sources and three coverages the mask never wins, and costs up to +0.063 Top-1
at a tenth of the frame. Their work asks whether a masked adversarial
perturbation protects; ours asks whether the masking is the part that protects.

**Against Radiya-Dixit et al. (3).** They show that protective perturbations
(Fawkes, LowKey) fail once the adversary retrains after the protected images
are published — an existence result about adaptation, on faces. Ours differs in
three ways. The setting is background-scene location rather than facial
identity, so the leaking signal and the attacker class are different. The
result is a measurement rather than an existence proof: we report how much a
given adaptation budget recovers, and find the binding resource is the
attacker's *index* rather than its capacity — a fully unfrozen encoder held at
the pretrained gallery gains nothing, while the same capacity allowed to
re-embed the gallery recovers about half of what the release removed. And the
adaptation arm is one of several attacks we bound rather than the whole claim.

**Common to all three.** None of them separates *where* a bounded budget is
placed from *what direction* it points, holding delivered distortion fixed.
That control is what this paper adds, and it is what turns "this mechanism
protects" into "this component of the mechanism is, or is not, what protects".

## Still required from the authors before upload

These cannot be completed from the repository (tracked as S1, S2, S6 in
`docs/ExperimentProgress.tex`):

1. **ORCIDs, EDICS categories and submission metadata** — entered in the
   submission system, not in the manuscript.
2. **Prior-submission disclosure.** `docs/TOMM_Response_Letter.md` records an
   earlier ACM TOMM review of a differently-titled version of this work. Whether
   and how to disclose it is an author decision; it is not included here.
3. **Supplementary over-length.** The supplement is 11 pages against the 6-page
   guidance. The request and its justification are already written into
   `04_cover_letter.txt`, but it needs the Editor-in-Chief's approval. This is
   the one item that can stop the package at editorial screening.
