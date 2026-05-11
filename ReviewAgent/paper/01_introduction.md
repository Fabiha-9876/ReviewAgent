# 1. Introduction

## 1.1 The Problem

**App-store reviews contain actionable software-engineering signal, buried bug reports, feature requests, performance complaints, usability obstacles, but no current pipeline turns them into a developer-ready *issue specification* that a defect-tracking system can consume.** Google Play receives on the order of millions of new reviews per day \cite{maalej2016, gao2019rrgen, dabrowski2022analysing}; the actionable subset is exactly what a development team must surface, prioritize, and respond to, but manual triage is infeasible at this volume.

Concretely, the end-to-end task is: **given a stream of unstructured reviews, produce (a) a structured issue specification (issue type, severity, affected component, reproduction steps, environment) suitable for routing into a defect tracker, and (b) a user-facing response that references the specific issue rather than a generic acknowledgement.** This is a multi-stage information-extraction cascade \cite{hearst1999, sarawagi2008}, not a single classification or generation problem.

## 1.2 Why It Is Hard

Three properties of the data make the cascade difficult:

- **Linguistic noise.** Reviews are short, slang-heavy, multilingual, often contradictory within one sentence (*"Great UI but battery drain after the update"*), and rarely contain the structured fields a defect tracker expects.
- **Long-tailed, skewed class distribution.** A few classes (`praise`, `bug_report`) absorb the bulk of reviews; operationally important minority classes (`compatibility`, `performance`) are rare enough that supervised training data is hard to assemble at scale.
- **LLM bootstrap labels are biased.** Production-scale labeling now relies on LLM annotation \cite{wang2021want, gilardi2023chatgpt}, but LLM annotators introduce *systematic* (not random) biases, class collapse, boundary confusion, that downstream classifiers inherit without warning \cite{pangakis2023automated, reiss2023testing, laban2023llm}.

## 1.3 The Research Gap

Each component of the cascade has substantial prior work, but **the components do not compose into an end-to-end pipeline because each stops one step short of an actionable artifact.** Stated explicitly:

- **G1, Noise correction.** Existing LLM-annotation pipelines \cite{wang2021want, gilardi2023chatgpt} produce labels at scale but do not validate them; existing label-noise correctors \cite{northcutt2021cleanlab, patrini2017making} target *crowd-labeled* noise, not LLM-flavored class collapse, and have not been validated via **verified-anchor pipelines with independent third-opinion endorsement**.
- **G2, Issue specification.** Existing systems **classify reviews into Maalej taxonomies** \cite{maalej2016, dabrowski2022analysing} **but do not generate structured actionable IssueSpecs**, there is no developer-routable artifact at the end of the pipeline.
- **G3, Clustering.** Existing clustering approaches \cite{villarroel2016, disorbo2016surf} **are flat embedding or topic clusters, not aspect-aware or knowledge-graph-guided**, so they cannot drill from "battery complaints" to "battery → Samsung drain after v3.2".
- **G4, RAG response generation.** Existing RAG pipelines for review response \cite{gao2019rrgen, gao2020core} **lack structured developer-oriented representations**: they retrieve dev-rel phrasing but do not condition on a typed issue specification, so generated replies remain template-like.
- **G5, RLHF alignment.** Existing RLHF methods (DPO, KTO) \cite{rafailov2023, ethayarajh2024} **conflate quality and policy compliance into one reward signal**; the dual-objective constrained formulation \cite{dai2023, altman1999} has not been applied to app-review response generation.

**The missing capability this work introduces** is a single pipeline that closes all five gaps with **typed intermediate artifacts**, each independently evaluable, each pushing the data one step further along the cascade toward an actionable issue specification.

## 1.4 Hypotheses

Four testable hypotheses, one per gap-closing component:

- **H1, Verified-anchor confident learning improves label reliability over standard LLM auto-labeling.** Operationally: Cohen's κ against a held-out expert gold standard rises ≥ 2× after the correction pipeline.
- **H2, Aspect-grounded knowledge-graph clustering produces higher-quality, finer-grained issue groupings than flat clustering.** Operationally: ≥ 2× more clusters at smaller average size, at comparable curated purity.
- **H3, Taxonomy-grounded IssueSpec prompting improves structural completeness over free-form LLM and human-written GitHub issues**, under **strict content-validity criteria** (not just "field non-empty"). Operationally: substantive template-fill rate higher for the templated LLM than for either baseline.
- **H4, Structured IssueSpecs improve response generation quality over RAG-only and no-context baselines.** Operationally: paired Wilcoxon, Friedman+Nemenyi, Bradley-Terry, and McNemar all separate the spec-aware system from baselines at *p* < 0.001.

A fifth hypothesis, **dual-objective Constrained PPO Pareto-dominates single-objective RLHF on the quality–compliance frontier**, is *implemented but not yet tested end-to-end* and is reported as future work in §5.5 and §7.

## 1.5 The Unified Conceptual Framework

The five components of ReviewAgent, **confident learning, clustering, IssueSpec generation, RAG, and RLHF**, are not loosely connected modules. They are five necessary stages of one coherent **structuring cascade**, each one introducing a typed intermediate artifact that the next stage consumes. The cascade is the framework; the central thesis is what the cascade is *for*.

### 1.5.1 The Central Thesis

> **Structure is what bridges noisy app reviews and developer-actionable artifacts. Structure must be re-introduced explicitly at every stage of the pipeline rather than left implicit in an end-to-end model. Every stage produces a typed intermediate artifact whose presence closes one specific gap in the prior literature, and whose absence breaks the next stage's input contract.**

This single thesis subsumes all five components: each one is a *structure-introduction* operation at a specific level of the pipeline. The five components are not interchangeable, and removing any one of them breaks the input contract of the next.

### 1.5.2 The Cascade, All Five Components in One Diagram

```
raw reviews
   │
   ▼  ┌──────────────────────────────────────────┐
      │ (1) CONFIDENT LEARNING                    │  Structure introduced:
      │     verified anchor + cleanlab            │  reliable per-review labels
      │     output: corrected labels              │  (Gap G1, H1)
      └──────────────────────────────────────────┘
   ▼  ┌──────────────────────────────────────────┐
      │ (2) CLUSTERING                            │  Structure introduced:
      │     aspect-grounded KG hierarchical       │  aspect-grouped clusters
      │     output: prioritized aspect clusters   │  with sub-aspect drill-down
      └──────────────────────────────────────────┘  (Gap G3, H2)
   ▼  ┌──────────────────────────────────────────┐
      │ (3) ISSUESPEC GENERATION                  │  Structure introduced:
      │     taxonomy-grounded templates           │  typed `IssueSpec` ,
      │     output: structured `IssueSpec`        │  the developer-actionable artifact
      └──────────────────────────────────────────┘  (Gap G2, H3)
   ▼  ┌──────────────────────────────────────────┐
      │ (4) RAG                                   │  Structure introduced:
      │     spec-aware retrieval-augmented        │  spec-conditioned response
      │     output: resolution-aware response     │  (not retrieval-style only)
      └──────────────────────────────────────────┘  (Gap G4, H4)
   ▼  ┌──────────────────────────────────────────┐
      │ (5) RLHF                                  │  Structure introduced:
      │     dual-objective Constrained PPO        │  separated quality + compliance
      │     output: aligned response policy       │  (CMDP, not single-scalar)
      └──────────────────────────────────────────┘  (Gap G5, H5*)
```

\* H5 (dual-objective dominates single) is implemented but not yet validated end-to-end (§4.7).

### 1.5.3 What Each Component Contributes to the Unified Thesis

| Component | What structure it introduces | Why it cannot be skipped | Gap closed |
|,|,|,|,|
| **(1) Confident learning** | Reliable per-review labels (κ 0.16 → 0.59) | Without it, downstream classifiers inherit the LLM's class-collapse bias and Stages 2–4b operate on biased data | **G1** |
| **(2) Clustering** | Aspect-grouped issue clusters with sub-aspect drill-down (605 vs 194 flat) | Without it, Stage 3 has no per-cluster substrate to taxonomy-ground; the IssueSpec would have to be generated per-review, infeasible at 215K scale | **G3** |
| **(3) IssueSpec generation** | Typed, taxonomy-grounded `IssueSpec` (the developer-actionable artifact) | Without it, Stage 4b has no structured intermediate, the entire +2.36-quality-point gain over RAG-only disappears (§4.3) | **G2** |
| **(4) RAG (structured)** | Spec-conditioned response with retrieval style anchoring | Without it, Stage 5 has no policy to align, the response generator does not exist | **G4** |
| **(5) RLHF (dual-objective)** | Quality- and compliance-separated alignment under CMDP | Without it, the response generator's alignment with policy compliance is implicit and unverifiable; over-promises and leaks cannot be bounded | **G5** |

### 1.5.4 The Three Theoretical Frameworks That Bind the Five Components

The same three theoretical frameworks justify the cascade at three different levels of abstraction (full discussion §3.0):

1. **Information-Extraction Cascade Theory** \cite{hearst1999, sarawagi2008}, justifies the *existence* of typed intermediates: progressive structuring monotonically reduces entropy and is independently auditable per stage.
2. **Human-AI Complementarity** \cite{kamar2016, bansal2019}, justifies the *placement* of HITL checkpoints at the three stages where the model is most uncertain (Stage 1 anchor labels, Stage 2 cluster validation, Stage 4b response rating).
3. **Constrained MDPs** \cite{altman1999, dai2023}, justifies the *separation* of quality from compliance in Stage 5: response generation is a constrained-optimization problem, not a single-scalar maximization.

### 1.5.5 Why This Is One Framework, Not Five Modules

Three properties make the cascade a *unified framework* rather than a stitched-together pipeline:

1. **Each stage's output is the next stage's typed input contract.** Removing Stage 2 breaks Stage 3's input (no cluster centroids to ground); removing Stage 3 breaks Stage 4b's IssueSpec channel (the +2.36 quality gain disappears). The components compose because their input/output types match by design.
2. **Each stage is independently evaluable.** §4.1 evaluates Stage 1 in isolation (κ 0.16 → 0.59); §4.4 evaluates Stage 2 (cluster purity 0.81); §4.2 evaluates Stage 3 (template-fill 0.96 vs 0.53 GitHub); §4.3 evaluates Stage 4b (4.62 vs 2.26 quality); §4.7 reports Stage 5 status. The cascade is auditable per-stage, which is the IE-cascade theoretical claim made operational.
3. **The same central thesis applies at every stage.** Each stage *introduces structure* into a previously unstructured signal: labels (Stage 1), groups (Stage 2), typed specs (Stage 3), spec-conditioned responses (Stage 4b), policy-separated alignment (Stage 5). The stages are five instances of the same operation at different levels of granularity.

The cascade is therefore *one framework with five instantiations of structure-introduction*, not five modules.

## 1.6 Contributions

Four contributions, each tied to one hypothesis and one gap:

- **(C1, G1+H1) A verified-anchor confident-learning recipe.** ≈30 person-hours of expert annotation lift Cohen's κ vs an expert gold standard from 0.16 → 0.59. A separately-trained classifier (V5) independently endorses **88.66%** of the 44,214 cleanlab corrections, third-opinion validation that closes the unaddressed-validation gap of G1.
- **(C2, G2+H3) A taxonomy-grounded IssueSpec representation.** Under strict content-validity criteria, the templated LLM reaches 0.96 substantive template-fill rate vs 0.53 for real GitHub issues from three open-source Android repos, closing G2's missing-actionable-artifact gap.
- **(C3, G3+H2) An aspect-grounded KG hierarchical-clustering layer.** Produces 605 fine-grained clusters vs 194 flat, with per-aspect drill-down, closing G3's flat-clustering gap.
- **(C4, G4+H4) A spec-aware retrieval-augmented response generator.** On 400 blinded paired ratings, beats RAG-only by Δ = +2.36 quality points (paired Wilcoxon *p* < 0.001), closing G4's no-structured-representation gap. Structure and retrieval are *complementary*, not substitutable: RAG-without-spec is statistically indistinguishable from no-context.

Single-rater design, single-LLM dependence (Claude Opus 4.7), and proof-of-concept-scale RLHF (G5 partially closed) are stated honestly in §5.5 with the experiments that would close each.

The remainder of the paper: §2 surveys related work along the five gaps; §3 presents the unified pipeline; §4 reports the four experiments; §5 discusses limitations; §6 concludes; §7 lists 13 prioritized future-work items.
