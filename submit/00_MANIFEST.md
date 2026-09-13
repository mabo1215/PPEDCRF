# Submission package — IEEE TIFS

Assembled 2026-09-14 from the build at `paper/`. Every PDF here is byte-identical
to the corresponding file in `paper/`, and every file in `source/` is
byte-identical to its counterpart in `paper/` (both md5-verified; last
re-verified 2026-09-14 after `source/main.tex` was refreshed and `build.bat`
added). Page counts re-counted from the PDFs themselves: 13 / 11 / 1.

## What to upload

| File | Contents | Pages |
|------|----------|-------|
| `01_manuscript.pdf` | Main manuscript | 13 |
| `02_supplementary.pdf` | Supplementary material | 11 |
| `03_titlepage.pdf` | Title page (authors, affiliations) | 1 |
| `04_cover_letter.txt` | Cover letter, incl. the over-length request | — |
| `05_prior_review_disclosure.md` | Supporting document for the prior-review question (Q1). TOMM manuscript ID filled in (`TOMM-2026-0332`); submission/decision dates still unconfirmed. | — |
| `06_tomm_reviews_verbatim.md` | The three ACM TOMM reviews, verbatim and unabridged | — |
| `source/` | LaTeX sources, `ref.bib`, used figures, generated tables, `build.bat` | — |

`source/figs/` carries only the four figures the documents reference, not the
whole 26 MB `paper/figs/` tree. `source/generated/` carries the table files the
documents `\input`; they are generated from the released per-query exports and
should not be hand-edited. All 17 `\input{generated/...}` targets and all four
`\includegraphics{figs/...}` targets resolve inside `source/`, and
`\bibliography{ref}` resolves to `source/ref.bib`, so the directory compiles
standalone. `source/generated/` also carries eight table files no longer
referenced by either document; they are harmless and were left in place.

## NOT part of this submission

Everything that does not belong in the upload has been moved into
`_not_submitted/`. Nothing was deleted. That directory now holds:

- `README.md` — a KITTI-360 **dataset download helper**, unrelated to the
  paper. It predates the package and would otherwise have reached an editor as
  if it described the submission. (The `download_2d_perspective*.sh/.zip`
  scripts it refers to are not present in `submit/`.)
- `figs/`, `figs.zip` — a duplicate of `source/figs/`, staged for an upload
  portal and superseded by `source/`.
- `supplyment/`, `supplyment.zip` — a partial copy of the supplement sources.
  Incomplete: it has no `generated/` directory, so it cannot compile.
- `source.zip` — an archive of `source/` taken before `main.tex` was refreshed
  and `build.bat` added, so it is stale. Re-zip `source/` at upload time if the
  submission system wants an archive.

`submit/` now contains exactly the seven items in the table above.

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
- **Manuscript ID:** TOMM-2026-0332

Note the question covers manuscripts "related to", not only identical
resubmissions, so the change of title and the substantial rework since do not
by themselves make the answer No.

**How much has changed, measured.** The reviewed version is recoverable exactly
(manuscript repository commit `ca94244`, 2026-03-29, whose title matches the
review letter verbatim). Comparing its body prose against the current
manuscript, tables and figures excluded:

| | ACM TOMM version | This submission |
|---|---|---|
| Title | *PPEDCRF: Dynamic-CRF-Guided Selective Perturbation for Background-Based Location Privacy in Video Sequences* | *Allocation or Direction? A Matched-Distortion Audit of Visual Location-Privacy Mechanisms* |
| Body words | 2,992 | 11,461 |
| Contribution | **Proposes** a mechanism (DCRF + NCP + selective Gaussian noise) | **Audits** that family and reports a negative result |
| Benchmark | 12 visually-mined synthetic pairs | MSLS place-labelled manifest (400 queries, 277 places) and KITTI-360 revisits |
| Attackers | retrieval backbones | five retrieval backbones, two image-to-GPS geolocators, an adaptive fine-tuned attacker, a purifying denoiser |
| PPEDCRF's role | the proposed method | **the object under audit** |

- **94.6% of the current manuscript is new text.** Only 622 words of verbatim
  word-runs are shared, and exactly one passage of 25 consecutive words
  survives — 0.2% of the current body.
- **20.8% of the old manuscript is retained** anywhere in the new one.
- 8 of 21 current section headings also appeared among the old 16.

The overlap that does remain is deliberate and visible: `PPEDCRF`, `DCRF` and
`NCP` are named throughout, and the reviewed version's headline numbers
(Top-1 0.833→0.722, 36.18 dB PSNR) still appear — because the mechanism that
was proposed there is the mechanism being audited here. That continuity is the
reason the honest answer to this question is Yes rather than No, and it is also
why answering No would be detectable: three ACM TOMM reviewers have seen this
work, and the reviewer pools of the two venues overlap.

Outstanding author actions:

1. The **ACM TOMM manuscript ID** has been supplied — `TOMM-2026-0332` — and is
   now filled into `05_prior_review_disclosure.md`. The submission and decision
   **dates** are still unrecorded and need confirming.
2. Confirmation and sign-off on the supporting document, drafted as
   `05_prior_review_disclosure.md` in this directory.

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


## Paste-ready answers (500-character fields)

Each field on the TIFS form caps at **500 characters**, counted as characters
rather than words and including spaces and punctuation. The blocks below are
verbatim paste text, each measured and confirmed to fit. Q2-Q5 are one-word
declarations that only the authors can confirm; the rest are answered from the
manuscript. The longer Q1-Q9 sections above are supporting analysis for the
authors, not form text — do not paste them into the 500-character fields.

### Q1. Resubmission of / related to a previously rejected or withdrawn manuscript?

> Yes. ACM TOMM, manuscript TOMM-2026-0332, "Dynamic-CRF-Guided Selective Perturbation for Background-Based Location Privacy in Video Sequences"; not accepted. A supporting document is uploaded with all three reviews verbatim and how each point was addressed. This manuscript continues that research line but inverts its contribution: the mechanism reviewed there (PPEDCRF) is here the object audited, not the proposal. 94.6% of the body text is new (2,992 to 11,461 words).

*472/500 characters.*

### Q2. Extended version of a conference publication?

> No.

*3/500 characters.*

### Q3. Related to other papers by the authors not cited here?

> No.

*3/500 characters.*

### Q4. Preprints identical to this submission?

> No.

*3/500 characters.*

### Q5. Other posted preprints that should not be considered prior art?

> No.

*3/500 characters.*

### Q6. Why is the contribution within the scope of IEEE TIFS?

> The paper evaluates visual privacy-protection mechanisms and the attacks defeating them. Background scene cues let an adversary infer where a released dashcam frame was recorded even after GPS metadata is stripped. We audit published image-sanitization defenses at matched delivered distortion against five retrieval attackers, two image-to-GPS geolocators, an adaptive attacker and a purifying denoiser. The machine learning is applied to that privacy problem, not studied alone.

*480/500 characters.*

### Q7. Why is the contribution significant (what impact will it have)?

> This family is designed by choosing where to spend a bounded budget. At matched delivered distortion that choice buys nothing against four of five attackers, over nine placement rules, three published-model placements and four operators, every difference within 0.012 Top-1. The leverage is the perturbation's direction, which transfers to all five. We supply the protocol, its controls, and a diagnostic that caught a released checkpoint whose map is numerically constant.

*473/500 characters.*

### Q8. The three most closely related published papers

> 1) X. Liu et al., GeoShield: Safeguarding Geolocation Privacy from Vision-Language Models via Adversarial Perturbations, AAAI 2026, 40:35653-35661, doi 10.1609/aaai.v40i42.40877. 2) T.-N. Le et al., Rethinking Adversarial Examples for Location Privacy Protection, IEEE WIFS 2022, 1-6, doi 10.1109/WIFS55849.2022.9975388. 3) E. Radiya-Dixit, S. Hong, N. Carlini, F. Tramer, Data Poisoning Won't Save You From Facial Recognition, ICLR 2022 (OpenReview, no DOI assigned).

*468/500 characters.*

### Q9. What is distinctive/new relative to those works?

> None separates where a bounded budget is placed from what direction it points at fixed delivered distortion; that control is what we add. We run GeoShield's release end to end: it does not separate from isotropic noise of equal energy (-0.010 Top-1, [-0.041,+0.020]). Le et al.'s mask never beats no mask, costing up to +0.063 Top-1. Radiya-Dixit et al. show adaptation defeats face cloaking; we measure it, finding the binding resource is the attacker's index, not its capacity.

*479/500 characters.*

## Still required from the authors before upload

These cannot be completed from the repository (tracked as S1, S2, S6 in
`docs/ExperimentProgress.tex`):

1. **ORCIDs, EDICS categories and submission metadata** — entered in the
   submission system, not in the manuscript.
2. **Prior-submission disclosure — drafted, needs sign-off.** The disclosure is
   included in this package as `05_prior_review_disclosure.md`, with all three
   reviews reproduced verbatim in `06_tomm_reviews_verbatim.md`. The manuscript
   ID (`TOMM-2026-0332`) is filled in. Still outstanding: the authors' sign-off,
   and the ACM TOMM submission and decision dates, which are not recorded in
   the repository.
3. **Supplementary over-length.** The supplement is 11 pages against the 6-page
   guidance. The request and its justification are already written into
   `04_cover_letter.txt`, but it needs the Editor-in-Chief's approval. This is
   the one item that can stop the package at editorial screening.
4. **The manuscript no longer names the code repository — must be fixed in
   `paper/main.tex` before upload.** The author footnote used to carry
   "Code and evaluation exports: `https://github.com/mabo1215/PPEDCRF`"; it was
   removed from `paper/main.tex` shortly before the submitted PDFs were built,
   so `01_manuscript.pdf` names no repository of the authors' own. The only
   GitHub URL left in it is the Mapillary SLS dataset citation. Two places now
   point at something that does not exist in the text: the manuscript itself
   ("declared in the released repository") and `04_cover_letter.txt`, which
   tells the Editor the artifacts "are in a public repository named in the
   manuscript". Either restore the URL in `paper/main.tex` and rebuild, or
   reword both the manuscript sentence and the cover letter. This was left
   unfixed here because it has to be corrected in `paper/`, not in the
   assembled package.
