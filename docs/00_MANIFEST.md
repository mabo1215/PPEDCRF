# TIFS submission — form answers and open items

Two things live here: the text to paste into the submission form, and what
still needs a decision or a signature before upload. The package itself is in
`submit/`, the cover letter in `docs/cover_letter.txt`, and the working history
in `docs/progress.md`.

## Paste-ready answers (500-character fields)

Each field on the TIFS form caps at **500 characters**, counted as characters
rather than words and including spaces and punctuation. Every block below is
verbatim paste text, measured and confirmed to fit. Q2-Q4 are the authors'
confirmed declarations (2026-09-15); the rest are answered from the manuscript.
Q5 was a No until the two arXiv postings were identified on 2026-09-15 — see
the item on them under the next section.

### Q1. Resubmission of / related to a previously rejected or withdrawn manuscript?

> Yes. ACM TOMM, manuscript TOMM-2026-0332, "Dynamic-CRF-Guided Selective Perturbation for Background-Based Location Privacy in Video Sequences"; not accepted, and posted as arXiv:2604.17163. A supporting document is uploaded with all three reviews verbatim and the response to each. This manuscript continues that line but inverts its contribution: the mechanism reviewed there (PPEDCRF) is here the object audited, not the proposal. 94.6% of the body text is new (2,992 to 11,461 words).

*487/500 characters.*

### Q2. Extended version of a conference publication?

> No.

*3/500 characters.*

### Q3. Related to other papers by the authors not cited here?

> No.

*3/500 characters.*

### Q4. Preprints identical to this submission?

> No.

*3/500 characters.* The question asks for preprints "identical to the submitted
manuscript, except for minor differences". arXiv:2604.17163 is not one: it is
the pre-audit version, different title, 2,992 body words against 11,461, and the
opposite contribution. It is named in the prior-submission answer and listed in
the other-preprints answer, so nothing about it is withheld by this No.

### Q5. Other posted preprints that should not be considered prior art?

> Yes, two, neither identical to this submission. arXiv:2604.17163 (18 Apr 2026) is the earlier version of this work: the ACM TOMM manuscript declared elsewhere on this form as previously rejected. This submission inverts its contribution and 94.6% of its body text is new. arXiv:2603.01593 (2 Mar 2026, five authors) is the earlier paper proposing the PPEDCRF mechanism that this submission audits. Neither should be read as a preprint of the manuscript submitted here.

*468/500 characters.* The Q-numbers used in this file are our own labels for
the form's questions, so no pasted answer refers to another one by number. If this field turns out to be a Yes/No box with no room
to list, answer Yes and paste the text into the covering comments field.

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

**From the system proof (`submit/6e98b5a0-…​.pdf`, 28 pages, re-checked against
the 19:24 EST regeneration).** The conversion is clean: 4 system pages, the
manuscript on pages 5-17 (13), the supplement on 18-28 (11), no `??` anywhere,
figures and reference list identical to our own PDFs, and the author footnote
carrying the revised affiliation with both links.

Two items from the first proof are now fixed on the form: the EDICS list leads
with `IF-PRV-PROT` (privacy protection) followed by `IF-PRV-ATTC`,
`IF-ML-ADVE` and `IF-SUR-PRIV`, and the prior-submission answer carries "posted
as arXiv:2604.17163". Two remain.

A. **The manuscript's LaTeX source is still not in "Files for peer review".**
   That list has six entries — `01_manuscript.pdf`, `figs.zip`,
   `02_supplementary.pdf`, `supplementary.zip`,
   `05_prior_review_disclosure.zip`, `07_code.zip` — and none is
   `manuscript.zip` (1,615,935 bytes, ≈1.54 MB; the only 1.9 MB entry is
   `figs.zip`). Either the upload did not take, or it landed in a slot the
   peer-review list does not print. Worth checking in the submission UI: the
   supplement's source is on file and the main document's is not.
B. **`figs.zip` came from `_not_submitted/`.** It is 4 PNGs already embedded in
   both PDFs and already inside both source archives — harmless, and pure
   duplication once A is settled.

The cover letter appears nowhere in the proof — no "Dear", no
"Editor-in-Chief", no "six double-column" — which is what a system that keeps
the letter editor-only looks like, so the proof can neither confirm nor deny it
was entered. Confirm it in the submission UI rather than from this file: the
11-page supplement's approval exists only in that letter.

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
5. **Where the extended evidence report lives — now live in a real submission.**
   The supplement in the proof refers to it three times, so reviewers reading
   pages 18-28 will look for it. It is no longer in any submitted archive, it is
   not in the DataPort deposit (which holds exports, artifact results and
   scripts, no documents), and it is not in the capsule. Recommended: add
   `supplementary_extended.pdf` to the deposit, which leaves the manuscript
   untouched. The alternative is removing those four references.
6. **The supplement's cross-references to the manuscript, left as they are.**
   The supplement names the manuscript's sections through
   `\externaldocument[M-]{main}`, and the xr package reads those numbers out of
   the manuscript's `.aux`. Compiled on its own, without that file beside it,
   the supplement leaves **17 references unresolved — printed as `??` — and the
   build still exits 0**, so nothing announces it. They are 6 distinct sections
   (`causality` 7 times, `operator` and `msls_placement` 3 each, `direction` 2,
   `mechanism` and `nonadaptive_preprocessing` once).

   Recommended: leave it. `main.aux` is an input to xr rather than a source
   file, and the packages are source only by decision. The uploaded
   `02_supplementary.pdf` was built with every number resolved, and that is the
   file reviewers read; a submission system that makes its review proof from the
   uploaded PDFs never compiles the sources at all. The exposure is production
   at acceptance, when IEEE does compile — and that is a conversation with
   production, not a reason to ship a build input now. If a self-contained
   source tree is wanted later, the clean fix is to drop xr and write those six
   section numbers into the supplement's text.
7. **The two arXiv postings: what to cite, and one answer that depends on a
   fact only the authors hold.**

   - **arXiv:2604.17163** (18 Apr 2026, Ma, Yan, Wu) carries the ACM TOMM title
     verbatim: it is the reviewed prior version of this work. Now declared in Q1
     and listed in Q5. Q4 stays No because the question asks for a preprint
     *identical* to this submission and that one is the opposite contribution.
   - **arXiv:2603.01593** (2 Mar 2026, Ma, Wu, Yan, Shi, Nguyen) is the earlier
     paper proposing PPEDCRF, the mechanism this submission audits. **Q3 turns
     on whether that paper is published, accepted, or under review anywhere
     right now.** If it is, Q3 becomes Yes and it must be listed there; if it
     exists only as a posting, the confirmed No stands and Q5 already covers it.
     Only the authors can settle that.
   - **Whether to cite either in the manuscript.** The text introduces PPEDCRF
     as a mechanism "we implement as a testbed" and cites neither. The exposure
     is a reviewer who knows the postings and reads that sentence as
     understating where the mechanism came from. A citation at that sentence is
     a one-line change that alters no result, and it is the cheapest way to
     close the gap.
**Answered 2026-09-15, recorded so it is not asked again.** The
`ppedcrf-evidence` data asset is attached to the capsule, so its one-click run
recomputes claims rather than only exercising the synthetic self-check. There is
no separate Code Ocean DOI: the project's DOI is the IEEE DataPort deposit,
`https://doi.org/10.21227/jnr0-jm15`, which the manuscript's first-page footnote
already cites beside the capsule URL. Nothing in the manuscript changes.
