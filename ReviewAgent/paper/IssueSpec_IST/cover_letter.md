# Cover letter

To the Editors,
*Information and Software Technology*

Dear Editors,

We submit **Typed Issue Specifications from App Reviews: What Scales, and a Downstream
Benefit That Does Not Replicate** for consideration as a research article.

**The problem.** App developers receive review traffic at a volume no team can triage by
hand, and defect trackers cannot consume what arrives. JIRA and GitHub Issues need typed
fields, steps to reproduce, a severity, an affected component, while a review is free text
written by a frustrated user on a phone. Two decades of review-mining research stops at the
edge of that gap. Classifiers predict a label and leave the remaining fields empty. Response
generators write a reply without naming the defect. Agentic repair tools assume a well-formed
issue already exists. Knowledge-graph systems return free-form summaries rather than typed
fields.

**What we establish.** We propose a typed intermediate representation that routes each review
by issue class into one of five standards-body templates, together with a verified-anchor
procedure that makes the representation producible at corpus scale from noisy LLM labels. On
215,583 reviews from 58 Android apps, type routing reaches 0.96 substantive template-fill
against 0.69 without routing, and the verified-anchor procedure lifts Cohen's kappa against a
three-rater human gold standard from 0.163 to 0.592, and to 0.616 on the reviews the
classifier never saw during training. Along the way we find that an LLM labeller on software-engineering text can omit whole classes
rather than merely err on them, which is the failure a small human anchor repairs. We should note
that we first reported this the other way round, as noise concentrated on minority classes;
recomputing per stratum showed 1,302 of 1,307 disagreements fall in the largest class, and the
manuscript now states the corrected version.

**Why the title says what it says.** An earlier version of this work reported that supplying
the typed specification to a response generator raised human-rated reply quality by 2.35
Likert points. We no longer believe that number, and the paper explains why in detail rather
than quietly dropping it. The two conditions in that experiment were produced by different
deterministic template composers rather than by one model. One emitted lemmatised text
averaging 78 words, the other polished prose averaging 123 words with cleanup passes the first
arm never received. A rater comparing them was separating two writing styles, and the presence
of the specification was confounded with that separation.

We re-ran the comparison with the confound removed. Both arms now come from the same model and
the same candidate retrieval pool, differing only in whether the specification appears in the
prompt. Three raters scored all 200 blinded rows. The gain is 0.04 Likert points, p = 0.54, and
the raters disagree in direction, with
the lead author prefers the specification arm decisively, the two independent raters lean the
other way. We looked for a content mechanism behind that disagreement and did not find one, so
we report it as an open question rather than resolving it in our own favour.

**A methodological contribution we did not set out to make.** Comparing the two rating rounds
gave us a check on rating provenance. In the discarded round the two additional raters deviated
from the lead author on 16 and 19 of 400 rows, those deviation sets did not overlap on a single
row, and every deviation was exactly one point. Independent rating cannot produce that pattern.
In the reported round the same raters differ from the lead author on 133 and 145 of 200 rows,
the deviation sets overlap on 101 where chance predicts about 96, and the magnitudes range from
one to four. We release both rounds and the check, because annotation-heavy software-engineering
research has few tools for auditing rating provenance and we now have a worked positive and
negative example.

**Other negative results.** Three further components of our own pipeline fail their tests and
we report each rather than removing it. The knowledge-graph layer loses to count-matched flat
clustering on all three intrinsic metrics; its defensible role is prioritization, not cluster
quality. An extractive-coverage faithfulness scorer we built and release shows no rank
correlation with independent human faithfulness judgements and orders three generators in the
opposite direction to the humans, so we withdraw it as a faithfulness measure. The constrained
alignment layer engages once the policy is capable of violating its constraint, which our
proof-of-concept model was not, but never demonstrates the quality-versus-compliance trade-off
it was designed to expose.

**Why we think this belongs in IST.** The positive contribution is an empirical
software-engineering one, anchored in defect-tracking practice and evaluated with human
annotators under a documented protocol. The negative contribution is the kind that only gets
published when a venue values it: a replication of our own result under a corrected design,
reported with the artifact that shows both. We are aware this is a less quotable paper than the
one we could have submitted six months ago. We think it is the more useful one.

**Reproducibility.** Code, data, and models are released. A nineteen-segment verifier runs from
the released bundle alone and reports the numerical claims in the results section, including each
negative result above. We audited the verifier rather than trusting it, and the manuscript states
its coverage limits plainly: three segments recompute a value and assert it against the paper,
the rest read back a stored summary, and several quantities are not covered at all. We also measured and report run-to-run variance of the LLM-dependent
generation stages, so readers know which differences in the paper are larger than sampling
noise and which are not.

**Declarations.** This manuscript is original, is not under consideration elsewhere, and has
not been published previously. An earlier version was reviewed at a conference; that review
prompted several additions, and the manuscript has been substantially revised since, including
the withdrawal described above. All authors have approved the submission and declare no
competing interests. Generative AI use is disclosed in a dedicated section of the manuscript.

We would be glad to respond to any questions from the editors or reviewers.

Sincerely,

Fabiha Jalal, on behalf of all authors
Department of Computer Science, Islamic University of Technology
fabihajalal@iut-dhaka.edu
