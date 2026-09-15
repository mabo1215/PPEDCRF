# TIFS submission — form answers and open items

Two things live here: the text to paste into the submission form, and what
still needs a decision or a signature before upload. The package itself is in
`submit/`, the cover letter in `docs/cover_letter.txt`, and the working history
in `docs/progress.md`.

## Paste-ready answers (500-character fields)

Each field on the TIFS form caps at **500 characters**, counted as characters
rather than words and including spaces and punctuation. Every block below is
verbatim paste text, measured and confirmed to fit. Q2-Q5 are the authors'
confirmed declarations (2026-09-15); the rest are answered from the manuscript.

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

**Before the form.**

1. **ORCIDs, EDICS categories and submission metadata** — entered in the
   submission system, not in the manuscript. Recommended EDICS: primary
   `IF-PRV-PROT` (privacy protection), then `IF-ML-ADVE` (adversarial machine
   learning), `IF-PRV-ATTC` (privacy attacks), `IF-SUR-PRIV` (privacy in
   surveillance). If only the broad headings are offered: *Anonymization and
   Data Privacy*, then *Machine learning for Information Forensics and
   Security* and *Surveillance*. Avoid *Biometrics* — that literature is
   foreground-centric and no experiment here touches a person, so it would
   route the paper to reviewers whose expertise does not bear on the claim.
2. **Prior-submission disclosure — drafted, needs sign-off.** Uploaded as
   `05_prior_review_disclosure.zip`, which carries all three ACM TOMM reviews
   verbatim in its §4 and the manuscript ID (`TOMM-2026-0332`). Still
   outstanding: the authors' sign-off, and the ACM TOMM **submission and
   decision dates**, which are not recorded in this repository. This answers
   Q1, so it is required whatever the preprint answers are.
3. **Supplementary over-length.** The supplement is 11 pages against the 6-page
   guidance. The request and its justification are in `docs/cover_letter.txt`,
   but it needs the Editor-in-Chief's approval. This is the one item that can
   stop the package at editorial screening.

**Decisions that change what a reviewer sees.**

4. **Publish v1.1 of the Code Ocean capsule.** The manuscript's first-page
   footnote points at `https://codeocean.com/capsule/9035965/tree`. The refresh
   — manuscript text removed, README rewritten for this paper, Chinese data
   README translated, private submodule declaration dropped — is pushed to the
   working capsule, but a published capsule is a read-only snapshot, so the
   public page stays at v1.0 until it is released again from the Code Ocean
   side. **Until then the public v1.0 still carries `main.tex` and the
   supplement**, which is the only place a copy of this manuscript is public;
   it is also the one fact that sits behind the No to the preprint questions.
5. **Where the extended evidence report lives.** The two documents cite "the
   extended report" four times. It is no longer in any submitted archive, it is
   not in the DataPort deposit (which holds exports, artifact results and
   scripts, no documents), and it is not in the capsule. Recommended: add
   `supplementary_extended.pdf` to the deposit, which leaves the manuscript
   untouched. The alternative is removing those four references.
6. **Whether to put `main.aux` back in `submit/source/supplementary/`.** The
   supplement reads the manuscript's section numbers through
   `\externaldocument[M-]{main}`, and xr takes them from that file. Without it a
   host that compiles the supplement on its own leaves **17 references
   unresolved — printed as `??` — and the build still exits 0**, so nothing
   announces it. The uploaded `02_supplementary.pdf` is unaffected.
   Recommended: put back that one 13 KB file, which is a build input rather
   than a build artifact, and leave the build scripts and PDFs out.
7. **Whether to cite arXiv:2603.01593.** It is the authors' public posting of
   the mechanism this paper audits, and the manuscript introduces that
   mechanism as one "we implement as a testbed" without citing it. The form
   asks about papers published, accepted, or under review, so the confirmed No
   to Q3 and Q5 is correct as asked. The exposure is a reviewer who knows the
   preprint and reads that sentence as understating where the mechanism came
   from. Citing it there is a one-line change that alters no result.
8. **Two facts about the capsule this repository cannot see.** Whether the
   `ppedcrf-evidence` data asset is attached to it — without the asset its
   one-click run only exercises the synthetic self-check and verifies nothing —
   and whether the publication minted a DOI (`10.24433/CO.*`). If it did, the
   footnote should cite the DOI rather than the capsule URL; DataCite has no
   record of one yet.
