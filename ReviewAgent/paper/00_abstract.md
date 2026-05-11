# Abstract

**Problem.** App stores receive millions of reviews daily containing buried but actionable bug reports, feature requests, and performance complaints; manual triage is infeasible.

**Gap.** Existing pipelines stop short of an actionable artifact: classifiers label and RAG generators reply, but no system produces the structured *issue specification* a developer can route into a defect tracker. LLM-based annotation, the de facto labeling method, also exhibits systematic class-collapse noise that propagates downstream uncorrected.

**Approach.** We present **ReviewAgent**, a four-stage pipeline composing (i) verified-anchor confident learning to correct LLM-label noise on a 215K-review corpus; (ii) aspect-grounded knowledge-graph hierarchical clustering; (iii) taxonomy-grounded `IssueSpec` generation using Zimmermann / ISO 25010 / Nielsen / user-story templates; (iv) spec-aware retrieval-augmented response generation.

**Findings.** Cohen's κ against an expert gold standard rises 0.16 → 0.59; an independently-trained classifier endorses 88.66% of cleanlab corrections. Under strict content-validity criteria, taxonomy-grounded specs reach 0.96 template-fill (Claude Opus 4.7) vs 0.53 for real GitHub issues; cross-LLM replication on Qwen2.5-3B (0.74) and Qwen2.5-1.5B (0.48) shows the score scales with capability, with the qualitative ranking preserved across a sensitivity sweep. On 400 blinded paired ratings, the spec-aware generator beats RAG-only by +2.36 quality points (paired Wilcoxon *p* < 0.001), structure and retrieval are *complementary*.

**Contribution.** A reproducible noise-correction recipe with third-opinion validation, an aspect-grounded clustering layer, and a structured-RAG architecture whose value over vanilla RAG is empirically established. Single-rater, single-LLM, and proof-of-concept-scale RLHF are stated as honest limitations. Artifacts: https://github.com/Fabiha-9876/ReviewAgent.
