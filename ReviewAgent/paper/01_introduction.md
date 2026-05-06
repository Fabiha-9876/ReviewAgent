# 1. Introduction

App-store reviews are among the largest, freshest, and most opinionated sources of feedback available to software development teams \cite{maalej2016, gao2019rrgen, dabrowski2022analysing}. The volume — Google Play alone receives millions of new reviews per day — makes manual triage infeasible, motivating a generation of automated approaches: classifiers that route reviews into actionable categories, clusterers that group recurring complaints, and generators that produce developer-style replies at scale.

We present **ReviewAgent**, a three-layer knowledge pipeline that addresses three coordinated research aims simultaneously: (Aim 1) a *unified clustering-to-issue translation framework* combining knowledge-graph construction, aspect-grounded hierarchical clustering, and standardized schema mapping that converts raw reviews into taxonomy-grounded issue specifications; (Aim 2) a *coupled resolution-and-response system* with a four-agent code-resolution pipeline (Planner / Navigator / Editor / Executor) that produces real source-code patches alongside resolution-aware responses that reference specific fix locations; and (Aim 3) a *dual-objective RLHF loop* that embeds human oversight at three pipeline stages and progressively trains KTO, DPO, and Constrained PPO objectives, jointly optimizing for response quality and safety constraints.

A foundational sub-component of this pipeline is the cleaning of upstream auto-annotated training data — a problem we measure directly and correct via verified-anchor confident learning \cite{northcutt2021cleanlab}.

## 1.1 The LLM Annotation Noise Problem

Direct measurement on 5,230 LLM-labeled reviews from the RRGen corpus \cite{gao2019rrgen} shows **25% of LLM-assigned `praise` labels are incorrect** when verified by a domain expert (1,305 of 5,041 corrected to a different category). Two failure modes dominate:

1. **Class collapse:** the LLM concentrates predictions in popular classes (`bug_report`: 80,058 of 215,583; `praise`: 57,940). Rare classes are systematically under-detected — `performance` receives only 184 labels (0.085%) and `compatibility` receives 8 (0.004%), despite both being abundantly represented in the corpus when measured against expert verification.
2. **Boundary confusion:** between semantically adjacent classes (`praise` ↔ `other`, `bug_report` ↔ `performance`, `bug_report` ↔ `compatibility`), the LLM systematically defaults to the more populous category.

These errors are not random; they are **structural artifacts of LLM annotation behavior** that propagate to downstream models trained on the labels.

## 1.2 Contributions

This paper makes three contributions, each empirically grounded in measurable experiments on RRGen:

**(C1) An empirical characterization of LLM annotation noise in app-review datasets.** We measure the LLM error rate directly via expert verification on 5,230 reviews and an additional 490-review stratified gold-standard sample. The original LLM labels achieve Cohen's κ = 0.16 against the expert (slight agreement on the standard scale \cite{landis1977}); per-class F1 is 0.00 on `compatibility` and 0.00 on `performance`.

**(C2) A verified-anchor confident-learning correction pipeline that improves expert κ from 0.16 to 0.59.** We train a small RoBERTa "anchor" classifier on the 5,230 expert-verified labels plus 5,008 MAALEJ human-annotated reviews \cite{maalej2016} (10,238 total), then apply Cleanlab \cite{northcutt2021cleanlab} to flag and correct likely-mislabeled reviews in the full 215,583-review corpus. The pipeline produces 44,214 corrections (20.51% of the corpus), recovering 7,460 misclassified `performance` reviews and 2,503 misclassified `usability` reviews. Critically, a separately-trained classifier (V5) independently endorses **88.66%** of the corrections — the strongest evidence that they are genuine improvements rather than artifacts of the cleanlab procedure.

**(C3) A free-tier downstream pipeline (no LLM API access required) that delivers paper-grade clustering and human-rated response generation.** Stages 2 (clustering with UMAP+HDBSCAN), 3 (taxonomy-grounded issue specifications), and 4b (RAG-augmented response generation) all run on local compute. Cluster purity is 0.66 from automatic clustering, rising to 0.81 after lead-author curation of the top 100 clusters. The full ReviewAgent response generator achieves quality 4.62/5 in a 400-rating blinded human evaluation, vs 2.26/5 for the RAG-only baseline (paired Wilcoxon p < 0.001, Δ = +2.36).

## 1.3 Why This Matters

The verified-anchor correction approach turns a small budget of expert annotation (5,000–10,000 reviews; ≈30 person-hours of effort) into a leverage point that improves a 215,000-review dataset. The result is not just a cleaner training set but **measurably better downstream models**: a classifier that recovers entire classes the LLM had effectively erased (compatibility, performance), a clustering pipeline that produces semantically coherent issue groups, and a response generator that human raters prefer by a large and statistically significant margin over RAG-only baselines. We expect the verified-anchor methodology to generalize to other LLM-labeled software-engineering datasets where expert verification is feasible at small scale.

The remainder of the paper is structured as follows. Section 2 surveys related work in app-review mining and confident-learning. Section 3 describes the four-stage ReviewAgent methodology. Section 4 reports the three core experiments: classifier validation against the expert gold standard (Experiment 1.1), Stage 3 issue-specification quality across four conditions (Experiment 1.2), and Stage 4b response generation quality across four conditions (Experiment 2). Section 5 discusses limitations and future work. All datasets, scripts, and trained model artifacts are publicly released.
