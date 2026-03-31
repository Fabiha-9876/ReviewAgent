# ReviewAgent — An End-to-End Agentic System for Automated App Review Triage, Resolution, and Response with Human-in-the-Loop Refinement

## Problem Statement

Mobile app stores receive millions of user reviews daily. These reviews are a rich but highly noisy source of actionable information — containing bug reports, feature requests, performance complaints, and usability feedback buried within informal, unstructured, and often vague text (e.g., *"app keeps crashing"*, *"trash app 1 star"*, *"login broken on my samsung fix it!!!"*).

**The core problem is threefold:**

1. **Manual triage is slow and inconsistent.** Developers must manually read thousands of reviews to identify actionable issues. This process is labor-intensive, subjective, and does not scale — leading to delayed responses, missed critical bugs, and inconsistent prioritization across team members.

2. **No standardized issue representation exists.** Even when reviews are classified (e.g., as "bug report" or "feature request"), the output remains unstructured text — not a standardized, actionable issue specification that a developer or automated system can act on. There is no established method for converting noisy review clusters into GitHub-issue-quality specifications with reproduction steps, severity, affected components, and environment details.

3. **Existing works operate in silos.** Current research addresses fragments of this problem in isolation:
   - **Classification papers** (BERT-RCNN, transfer learning approaches) stop at labeling reviews but do not structure them into actionable issues.
   - **Response generation papers** (RRGen, CoRe) generate replies but ignore the actual underlying issue and its resolution status.
   - **Agentic SE papers** (SWE-Agent, RepairAgent) assume well-structured GitHub issues as input — they cannot consume noisy app reviews directly.
   - **Knowledge graph papers** summarize review content but do not produce actionable triage output.
   - **RLHF papers** (Safe RLHF, DPO) optimize language models but have not been applied to the app review response domain.

**The critical missing piece** identified across all five research areas: **review-to-issue translation** — converting noisy, vague, non-technical app store reviews into structured, actionable issue specifications that can drive both automated resolution and informed user responses.

---

## Theoretical Foundations

The ReviewAgent pipeline is grounded in three complementary theoretical frameworks that provide formal justification for the architecture and its design decisions.

### Framework 1: Information Extraction Cascade Theory

The pipeline follows the **information extraction cascade** model (Hearst, 1999; Sarawagi, 2008), which formalizes the progressive transformation of unstructured text into structured, actionable knowledge:

> Raw text → Entities → Relations → Structured knowledge → Downstream task

In ReviewAgent, this maps directly to:

> Reviews (raw text) → Aspects/Entities (extraction) → KG (relations) → Issue specs (structured knowledge) → Response generation (downstream task)

**Theoretical claim:** Each stage monotonically reduces entropy and increases actionability. We define an *actionability score* that quantifies how ready a representation is for developer action, and empirically demonstrate its increase across pipeline stages. This provides a formal basis for why the progressive structuring approach outperforms end-to-end models that skip intermediate representations.

**Key references:**
- Hearst, M. (1999). "Untangling Text Data Mining." ACL.
- Sarawagi, S. (2008). "Information Extraction." Foundations and Trends in Databases.

### Framework 2: Human-AI Complementary Decision Making

The HITL checkpoints are grounded in **human-AI complementarity theory** (Kamar, 2016; Bansal et al., 2019), which demonstrates that hybrid human-AI systems can achieve higher accuracy than either alone, provided humans are engaged at the right decision points.

**Theoretical claim:** By applying **selective prediction** (Geifman & El-Yaniv, 2017) — where the system defers to humans when its confidence falls below a calibrated threshold — the pipeline achieves complementary performance. Specifically:

- At Stage 1, confidence-based flagging routes ambiguous classifications to experts, creating a human-AI team whose classification accuracy exceeds either party alone.
- At Stage 3, rubric-based expert validation catches systematic LLM errors (e.g., implausible reproduction steps) before they propagate to downstream stages.

The theoretical guarantee is that selective prediction with calibrated confidence scores provably reduces the error rate compared to full automation, at the cost of human effort proportional to the uncertainty fraction of inputs.

**Key references:**
- Kamar, E. (2016). "Directions in Hybrid Intelligence." IJCAI.
- Bansal, G. et al. (2019). "Does the Whole Exceed Its Parts? The Effect of AI Explanations on Complementary Team Performance." AAAI.
- Geifman, Y. & El-Yaniv, R. (2017). "Selective Prediction." NeurIPS.

### Framework 3: Constrained Markov Decision Process (CMDP) Theory

The dual-objective RLHF formulation is grounded in **constrained MDP theory** (Altman, 1999), which formalizes optimization problems with multiple, potentially conflicting objectives:

> Maximize: R_quality(response) subject to: C_compliance(response) ≤ threshold

**Theoretical claim:** Single-objective RLHF conflates quality and compliance into one reward signal, collapsing two distinct optimization surfaces into one. Dual-objective formulation respects the **Pareto frontier** — finding responses that are maximally helpful *without* violating compliance constraints. Constrained PPO (Dai et al., 2023) is the solver that navigates this frontier.

This provides a formal explanation for why dual-objective outperforms single-objective: the single-objective model must implicitly learn the compliance boundary from mixed signals, while the dual-objective model has explicit constraint enforcement.

**Key references:**
- Altman, E. (1999). *Constrained Markov Decision Processes.* Chapman & Hall/CRC.
- Dai, J. et al. (2023). "Safe RLHF: Safe Reinforcement Learning from Human Feedback." ICLR.

---

## Proposed Architecture (Scoped)

Based on advisor feedback, the pipeline scope is focused on **Stages 1-3 + 4b + 5** as the core contribution. Stage 4a (agentic code resolution) is positioned as a **future work extension** since it constitutes a separate research area with existing solutions (SWE-Agent, RepairAgent).

```
┌─────────────────────────────────────────────────────┐
│                    STAGE 1: INTAKE                   │
│  Multi-label classifier (RoBERTa fine-tuned)         │
│  + Aspect-based sentiment analysis                   │
│  + Entity extraction (feature, screen, device, OS)   │
│  + HITL Checkpoint #1 (ambiguous classifications)    │
│  Input: Raw app reviews                              │
│  Output: Labeled, structured review objects           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              STAGE 2: KNOWLEDGE GRAPH                │
│  Build review KG: aspect nodes + sentiment edges     │
│  Hierarchical clustering (aspect → sub-complaint)    │
│  Standardized issue schema mapping                   │
│  Graph centrality → priority ranking                 │
│  Input: Structured review objects                    │
│  Output: Prioritized, schema-mapped issue clusters   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│         STAGE 3: REVIEW-TO-ISSUE TRANSLATION         │
│  (THE NOVEL CONTRIBUTION)                            │
│  LLM agent converts noisy review clusters into       │
│  taxonomy-grounded structured issue specs            │
│  + HITL Checkpoint #2 (expert rubric validation)     │
│  Input: Issue clusters + KG                          │
│  Output: GitHub-issue-quality specifications         │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│        STAGE 4b: RESPONSE GENERATION                 │
│  RAG (5 fixed input sources) + issue-spec-aware      │
│  drafting + self-refinement loop                     │
│  Generates resolution-aware user-facing response     │
│  Input: Issue spec + RAG corpus                      │
│  Output: Draft response                              │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│        STAGE 5: HITL CHECKPOINT #3                   │
│  Dual-objective feedback (CMDP-grounded):            │
│    Stream 1: Quality (helpfulness, specificity,      │
│              empathy, accuracy, actionability)        │
│    Stream 2: Compliance (safety, promises, tone)     │
│  Progressive RLHF: KTO → DPO → Constrained PPO      │
│  Feedback propagates BACK to Stages 1, 3, and 4b    │
└─────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │  FUTURE WORK: STAGE 4a              │
        │  Agentic Code Resolution            │
        │  (Planner → Navigator → Editor →    │
        │   Executor) consuming validated     │
        │  issue specs from Stage 3           │
        └─────────────────────────────────────┘
```

---

## Key Contributions (What Makes This Proposal New)

ReviewAgent introduces four novel components that collectively address the problem statement. No prior work combines these elements into a unified system:

### Contribution 1: Standardized Issue Schema Mapping
**What's new:** A unified framework that transforms noisy review clusters into structured, standardized issue specifications — with fixed fields for issue type, reproduction steps, severity, affected component, environment, and priority score. Each issue type follows established SE taxonomies: bug reports use the Zimmermann template, feature requests use user stories, performance complaints map to ISO/IEC 25010 NFR categories, usability issues align with Nielsen's heuristics, and compatibility issues generate device-OS matrices.
**Why it matters:** No existing work produces standardized, developer-actionable issue specs from raw reviews. This is the bridge between noisy user feedback and structured software engineering workflows.

### Contribution 2: Human-in-the-Loop Checkpoints at Critical Decision Points
**What's new:** Three strategically placed HITL checkpoints grounded in selective prediction theory (Geifman & El-Yaniv, 2017): (1) post-classification verification for ambiguous reviews, (2) expert rubric-based validation of generated issue specs before downstream processing, and (3) dual-objective feedback on final responses. Human corrections at every checkpoint propagate backward into the training loop.
**Why it matters:** Prior HITL approaches in this domain are limited to end-stage feedback. By embedding human oversight at high-uncertainty junctures — where errors would cascade — the system achieves complementary human-AI performance (Bansal et al., 2019) that exceeds either party alone.

### Contribution 3: Knowledge Graph for Review Aggregation and Prioritization
**What's new:** A three-layer KG framework (graph construction → hierarchical clustering → schema mapping) that deduplicates, clusters, and prioritizes reviews by computing graph centrality (PageRank/betweenness). Reviews are first grouped by aspect, then sub-clustered by specific complaint, and finally mapped to the standardized issue schema.
**Why it matters:** Existing KG approaches for reviews focus on summarization, not actionable triage. The graph centrality-based prioritization ensures developers see the most impactful issues first, reducing triage effort.

### Contribution 4: Issue-Spec-Aware Response Generation with Dual-Objective RLHF
**What's new:** A response generation module that is explicitly aware of the structured issue spec from Stage 3, producing responses that reference the specific issue and its status — not generic templates. Optimized via dual-objective RLHF (CMDP-grounded): maximize quality while constraining policy compliance, progressing from KTO → DPO → Constrained PPO as feedback data grows.
**Why it matters:** Existing response generators (RRGen, CoRe) operate without knowledge of the structured issue. RLHF has not been applied to app review responses, and no prior work decomposes the optimization into quality and compliance streams.

### Contribution 5: Gold-Standard Benchmark Dataset
**What's new:** The first dataset pairing app review clusters with expert-written structured issue specifications (200-300 clusters, 3+ annotators), with inter-annotator agreement measured via Krippendorff's alpha.
**Why it matters:** No such benchmark exists. This enables reproducible evaluation of review-to-issue translation systems.

### Summary Table

| Contribution | Gap It Fills | Theoretical Grounding | Papers It Builds On |
|---|---|---|---|
| **1. Standardized Issue Schema Mapping** | No standardized issue representation from reviews | IE cascade (progressive structuring) | Zimmermann (2010), Nielsen (1994), ISO 25010 |
| **2. HITL at Critical Decision Points** | HITL limited to end-stage in prior work | Human-AI complementarity + Selective prediction | Kamar (2016), Bansal (2019), Geifman (2017) |
| **3. KG for Aggregation & Prioritization** | KG papers summarize but don't produce actionable triage | Graph centrality as impact proxy | ReviewGraph, KGCPN, Villarroel (2016) |
| **4. Issue-Spec-Aware Response + Dual RLHF** | Response gen ignores issue context; RLHF not applied here | CMDP (quality reward + compliance constraint) | RRGen, CoRe, Safe RLHF, DPO |
| **5. Gold-Standard Benchmark Dataset** | No benchmark for review-to-issue translation | — | All categories |

---

## Research Questions (Focused)

1. **RQ1 (Translation Quality):** How accurately can an LLM-based agent translate noisy, unstructured app review clusters into structured, taxonomy-grounded issue specifications, and how do these compare to human-written GitHub issues in terms of completeness, accuracy, actionability, specificity, and clarity?

2. **RQ2 (Coupled Response Generation):** Does coupling knowledge-graph-based issue prioritization with issue-spec-aware response generation produce more specific, actionable, and helpful user responses compared to context-unaware baselines (e.g., RRGen, CoRe)?

3. **RQ3 (Dual-Objective RLHF):** Does dual-objective RLHF (optimizing for both response quality and policy compliance) with dimension-level expert feedback outperform single-objective preference tuning in generating safe, helpful, and accurate app review responses over iterative retraining cycles?

---

## RAG Input Materials (Fixed)

| RAG Input Source | Purpose | Availability |
|---|---|---|
| App's past review-response pairs (from RRGen 570K) | Learn tone, style, common acknowledgments | Available |
| App changelog / release notes | Reference specific versions and fixes | Scrapable from Play Store |
| App FAQ / help documentation | Provide accurate workarounds | App-specific, manual collection |
| Generated issue spec from Stage 3 | Make response issue-aware and specific | Pipeline output |
| Similar past responses (retrieved by semantic similarity) | Template-level guidance | From RRGen dataset |

---

## Experiment Design

### Experiment 1: Review-to-Issue Translation Quality (RQ1)

- **Goal:** Evaluate LLM-generated issue specs against human-written ones
- **Setup:** 100 review clusters from the KG. 3 experts write gold-standard issue specs. LLM generates issue specs for the same clusters.
- **Conditions:**
  - (a) LLM with taxonomy grounding (full system)
  - (b) LLM without taxonomy grounding (free-form generation)
  - (c) Raw review summary without structuring (lower bound)
  - (d) Human-written issue specs (upper bound)
- **Evaluation:** 3 independent raters score all specs on the 5-dimension rubric (completeness, accuracy, actionability, specificity, clarity, each 1-5). Inter-annotator agreement via Krippendorff's alpha.
- **Statistical test:** Paired Wilcoxon signed-rank test on rubric scores between conditions

### Experiment 2: Coupled vs. Uncoupled Response Generation (RQ2)

- **Goal:** Does issue-spec-aware response generation outperform context-unaware baselines?
- **Setup:** For the same 100 clusters, generate responses using:
  - (a) RRGen (no issue context, no structured input)
  - (b) CoRe (contextual but no structured issue spec)
  - (c) ReviewAgent 4b WITHOUT issue spec (ablation — RAG only)
  - (d) ReviewAgent 4b WITH issue spec from Stage 3 (full system)
- **Evaluation:** Automatic: BLEU, ROUGE-L, BERTScore. Human: 3 raters score helpfulness, specificity, empathy, accuracy (1-5 each).
- **Statistical test:** Friedman test across 4 conditions, post-hoc Nemenyi pairwise comparisons

### Experiment 3: Dual-Objective vs. Single-Objective RLHF (RQ3)

- **Goal:** Does decomposing feedback into quality + compliance outperform single-objective?
- **Setup:** Train 3 variants:
  - (a) Single-objective KTO (binary good/bad signal)
  - (b) Single-objective DPO (paired preferences on overall quality only)
  - (c) Dual-objective Constrained PPO (quality reward + compliance constraint)
- **Training data:** Start with 500 rated responses, expand to 2000 over 3 iterations
- **Evaluation:** Human preference win rate (pairwise comparisons), safety violation rate, per-dimension rubric improvement over iterations
- **Statistical test:** Bradley-Terry model for preference data; McNemar's test for safety violation rates

---

## Ablation Studies

| ID | What's Removed | What It Tests | Measured By |
|---|---|---|---|
| A1 | No KG (skip Stage 2, feed raw classified reviews to Stage 3) | Value of knowledge graph clustering | Issue spec rubric scores |
| A2 | No hierarchical clustering (flat clustering instead of aspect → sub-cluster) | Value of the two-level hierarchy | Cluster purity, NMI; downstream issue spec quality |
| A3 | No taxonomy grounding (LLM generates free-form issues) | Value of literature-grounded templates | Issue spec rubric scores (completeness, actionability) |
| A4 | No HITL at Stage 3 (skip expert validation) | Value of human checkpoint | End-to-end response quality |
| A5 | No RAG (generate responses from issue spec + LLM only) | Value of retrieval augmentation | Response BLEU, human eval |
| A6 | No issue spec in response gen (RAG but no structured issue from Stage 3) | Value of coupling between Stages 3 and 4b | Response specificity, accuracy scores |
| A7 | Single-stream feedback (merge quality and compliance into one score) | Value of dual-objective decomposition | Preference win rate, safety violation rate |

---

## Experimental Variables

### Independent Variables (Manipulated)

| Variable | Levels | Experiment |
|---|---|---|
| Issue generation method | Human / LLM with taxonomy / LLM free-form / No structuring | Exp 1 |
| Response generation context | No context / RAG only / Issue spec only / RAG + Issue spec | Exp 2 |
| RLHF objective | Single (KTO) / Single (DPO) / Dual (Constrained PPO) | Exp 3 |
| KG presence | With KG clustering / Without KG | Ablation A1 |
| HITL checkpoint | With expert validation / Without | Ablation A4 |
| Taxonomy grounding | Grounded in SE literature / Free-form | Ablation A3 |

### Dependent Variables (Measured)

| Variable | Metric | Scale |
|---|---|---|
| Issue spec quality | 5-dimension rubric scores | 1-5 per dimension |
| Inter-annotator agreement | Krippendorff's alpha | 0-1 |
| Response quality (automatic) | BLEU, ROUGE-L, BERTScore | 0-1 |
| Response helpfulness | Human rating | 1-5 |
| Response specificity | Human rating | 1-5 |
| Response empathy | Human rating | 1-5 |
| Safety violation rate | % of responses flagging compliance issues | 0-100% |
| Preference win rate | Pairwise human preference | % |

### Control Variables (Held Constant)

- Same LLM backbone across all conditions (e.g., GPT-4 or Llama 3)
- Same 100 review clusters for all experiments
- Same 3 expert raters across conditions
- Same rubric definitions and scoring guidelines
- Same RAG corpus configuration

---

## Evaluation Metrics — Finalized

### Gold Standard Evaluation (Stage 3 Output)

| Metric | What It Measures | How |
|---|---|---|
| Rubric scores (5 dimensions) | Issue spec quality | 3 experts, 1-5 scale per dimension |
| Krippendorff's alpha | Annotator agreement | Across 3+ raters |
| Completeness ratio | % of required fields filled | Automated check against schema |
| BERTScore vs. human specs | Semantic similarity to expert output | BERTScore between LLM and human specs |

### Per-Stage Metrics

| Stage | Primary Metric | Secondary Metric | Baseline |
|---|---|---|---|
| Stage 1: Classification | Macro F1, per-label F1 | Confidence calibration (ECE) | BERT-RCNN |
| Stage 2: Clustering + Schema Mapping | Cluster purity, NMI, schema mapping correctness (% of clusters with valid schema fields) | Deduplication precision/recall | Standard clustering (K-means) |
| Stage 3: Translation | Rubric scores + Krippendorff's alpha | BERTScore vs. human specs | Human-written issues |
| Stage 4b: Response | BLEU, ROUGE-L, BERTScore | Human eval (helpfulness, specificity, empathy) | CoRe, RRGen |
| Stage 5: RLHF | Preference win rate | Safety violation rate, per-dimension improvement | Single-objective tuning |
| End-to-end | % of reviews reaching validated response | Time from intake to response draft vs. manual triage time (measured in developer-hours) | Manual triage baseline (human-only process) |

---

## Dataset Strategy

| Dataset | What It Provides | Stage | Reference |
|---|---|---|---|
| **RRGen (~570K pairs)** | Google Play review-response pairs | Stages 1, 4b | Gao et al. (ASE 2019) |
| **MAALEJ (~4K labeled)** | Multi-label classified app reviews | Stage 1 | Maalej et al. (2016) |
| **GUZMAN** | Aspect-level annotations | Stage 1 | Guzman & Maalej (2014) |
| **Open-source Android apps with GitHub repos** | Ground-truth: reviews → issues → fixes | Stages 3, 4b | Manual identification |
| **Custom gold-standard (to be built)** | Expert-written issue specs for 200-300 clusters | Stage 3 evaluation | Novel contribution |
| **SWE-bench** | Agentic code resolution benchmark | Future work (Stage 4a) | Jimenez et al. (2024) |

---

## Reference Papers per Methodology Step

| Methodology Step | Reference Papers | What They Justify |
|---|---|---|
| **Step 1: Dataset collection** | Gao et al. (RRGen, ASE 2019); Maalej et al. (2016) "Automatic Classification of App Reviews" | Use of RRGen 570K; MAALEJ classification scheme |
| **Step 2: RoBERTa classification** | Kaur & Sahu (BERT-RCNN, 2023); Shah et al. (arXiv:2108.00663) | Transfer learning for app review classification |
| **Step 2: Aspect-based sentiment** | Guzman & Maalej (2014) "How Do Users Like This Feature?"; NLP survey (PeerJ CS, 2024) | Aspect extraction methodology |
| **Step 2: HITL for classification** | Geifman & El-Yaniv (2017) "Selective Prediction"; Bansal et al. (2019) | Confidence-based deferral to humans |
| **Step 3: KG construction** | ReviewGraph (arXiv:2508.13953); KGCPN (J. Cloud Computing, 2023) | KG for review representation |
| **Step 3: Hierarchical clustering** | Villarroel et al. (ICSE 2016) "CLAP: Release Planning Based on User Reviews"; Di Sorbo et al. (2016) "SURF" | Clustering app reviews for prioritization |
| **Step 3: Graph centrality for prioritization** | Keertipati et al. (2016) "App Review Prioritization" | PageRank/betweenness for issue ranking |
| **Step 4: Review-to-issue translation** | Zimmermann et al. (2010) "What Makes a Good Bug Report?"; Chaparro et al. (2017) "Detecting Missing Information in Bug Descriptions" | Bug report structure and quality criteria |
| **Step 4: Issue taxonomy grounding** | Nielsen (1994) "Usability Heuristics"; ISO/IEC 25010 (NFR quality model) | Usability and performance issue structures |
| **Step 5: RAG for response gen** | Gao et al. (RRGen); CoRe (arXiv:2010.06301); Self-Improving Response Gen (ECNLP 2024) | RAG and self-refinement for review responses |
| **Step 6: KTO methodology** | Ethayarajh et al. (2024) "KTO: Model Alignment as Prospect Theoretic Optimization" | Binary feedback RLHF |
| **Step 6: DPO methodology** | Rafailov et al. (2023) "Direct Preference Optimization" | Paired preference RLHF |
| **Step 6: Constrained PPO** | Dai et al. (2023) "Safe RLHF"; Altman (1999) Constrained MDPs | Dual-objective optimization |
| **Step 7: Expert rubric evaluation** | Zimmermann et al. (2010); Chaparro et al. (2017) | Rubric dimensions based on bug report quality |
| **Theoretical: IE cascade** | Hearst (1999); Sarawagi (2008) | Progressive structuring justification |
| **Theoretical: Human-AI complementarity** | Kamar (2016); Bansal et al. (2019) | HITL checkpoint placement |
| **Theoretical: CMDP** | Altman (1999); Dai et al. (2023) | Dual-objective RLHF formalization |

---

## Expected Outcomes

1. **Structured Issue Database:** The pipeline produces a continuously growing database of standardized, schema-mapped issue specifications from raw reviews — each with fields for issue type, reproduction steps, severity, affected component, and environment. This replaces ad-hoc manual triage with a structured, queryable knowledge base.

2. **Faster Triaging:** KG-based prioritization with graph centrality ranking significantly reduces developer triage time compared to manually reading raw review feeds. The end-to-end metric measures time from review intake to actionable response draft (in developer-hours) against the manual baseline.

3. **Better Developer Insights:** The knowledge graph and hierarchical clustering provide developers with aggregated, deduplicated views of user complaints — revealing temporal patterns (e.g., regressions after specific updates), cross-cutting issues (e.g., multiple aspects affected by the same root cause), and device/OS-specific problem distributions that are invisible when reading reviews individually.

4. **Improved User Response Quality:** Issue-spec-aware response generation produces replies that reference the specific problem and its status, replacing generic template responses. Dual-objective RLHF ensures responses are both helpful and policy-compliant, with measurable improvement over iterations.

5. **Novel Benchmark Dataset:** The first gold-standard dataset pairing 200-300 review clusters with expert-written structured issue specifications (3+ annotators), enabling reproducible evaluation of future review-to-issue translation systems.

---

## Why This Is Publishable

- **Novelty:** First end-to-end system connecting review classification, KG-based triage, issue translation, response generation, and dual-objective RLHF
- **Theoretically grounded:** IE cascade, human-AI complementarity, and CMDP theories — not just "wire-in wire-out"
- **The review-to-issue translation component** is entirely unexplored in literature
- **Dual-objective RLHF applied to app review responses** is a new application with formal CMDP justification
- **Practical impact:** Developers currently manually read thousands of reviews — this automates the triage-to-response loop
- **Multiple publication angles:**
  - Full system paper → ICSE/FSE
  - Translation component + benchmark dataset → ASE
  - RLHF application → EMNLP
  - KG-based triage → RE conference

---

## Suggested Title

> *"From Noise to Fix: An Agentic Pipeline for Automated App Review Understanding, Issue Resolution, and Response Generation with Human-in-the-Loop Alignment"*

---

## References (by Category)

### Category 1: App Review Classification & Problem Identification

1. BERT-RCNN: An Automatic Classification of App Reviews using Transfer Learning based RCNN Deep Model (ResearchGate, 2023)
2. Transfer Learning for Mining Feature Requests and Bug Reports from Tweets and App Store Reviews (arXiv:2108.00663)
3. Can GitHub Issues Help in App Review Classifications? (ACM, DOI:10.1145/3678170)
4. Mining User Reviews for Method-Level Bug Localization Using Transformers in Java-Based Applications (Springer Neural Computing and Applications, 2025)
5. Towards a Data-Driven Requirements Engineering Approach: Automatic Analysis of User Reviews (arXiv:2206.14669)
6. Natural Language Processing for Analyzing Online Customer Reviews: A Survey, Taxonomy, and Open Research Challenges (PeerJ CS, 2024)

### Category 2: Automated Response Generation

7. Automating App Review Response Generation — RRGen (ASE 2019, arXiv:2002.03552)
8. Automating App Review Response Generation Based on Contextual Knowledge — CoRe (arXiv:2010.06301)
9. Self-Improving Customer Review Response Generation (ECNLP 2024)

### Category 3: RLHF / Human-in-the-Loop

10. A Survey of Reinforcement Learning from Human Feedback (arXiv:2312.14925)
11. Reinforcement Learning from Human Feedback — Book (rlhfbook.com, 2025)
12. Safe RLHF: Safe Reinforcement Learning from Human Feedback (OpenReview, Dai et al., 2023)
13. MA-RLHF: Reinforcement Learning from Human Feedback with Macro Actions (OpenReview, 2024)
14. Preference Tuning with Human Feedback on Language — Columbia Survey (Wang et al.)

### Category 4: Graph-Based Summarization & Knowledge Graphs

15. ReviewGraph: A Knowledge Graph Embedding Based Framework for Review Rating Prediction with Sentiment Features (arXiv:2508.13953)
16. A Knowledge-Graph Based Text Summarization Scheme for Mobile Edge Computing — KGCPN (Journal of Cloud Computing, 2023)
17. Knowledge Graph-Augmented Long-Document Summarization (2024)

### Category 5: Agentic AI for Software Issue Resolution

18. Agentic Software Issue Resolution with Large Language Models: A Survey (arXiv:2512.22256, 2024)
19. RepairAgent: An Autonomous, LLM-Based Agent for Program Repair (ICSE 2025, arXiv:2403.17134)
20. HyperAgent: Generalist Software Engineering Agents to Solve Coding Tasks at Scale (OpenReview)
21. Demystifying LLM-based Software Engineering Agents (FSE 2025)
22. The Rise of Agentic AI: A Review of Definitions, Frameworks, Architectures, Applications, Evaluation Metrics, and Challenges (MDPI Future Internet, 2025)

### Category 6: Theoretical Foundations (NEW)

23. Hearst, M. (1999). "Untangling Text Data Mining." ACL.
24. Sarawagi, S. (2008). "Information Extraction." Foundations and Trends in Databases.
25. Kamar, E. (2016). "Directions in Hybrid Intelligence." IJCAI.
26. Bansal, G. et al. (2019). "Does the Whole Exceed Its Parts?" AAAI.
27. Geifman, Y. & El-Yaniv, R. (2017). "Selective Classification for Deep Neural Networks." NeurIPS.
28. Altman, E. (1999). *Constrained Markov Decision Processes.* Chapman & Hall/CRC.
29. Rafailov, R. et al. (2023). "Direct Preference Optimization." NeurIPS.
30. Ethayarajh, K. et al. (2024). "KTO: Model Alignment as Prospect Theoretic Optimization."

### Category 7: App Review Clustering & Prioritization (NEW)

31. Villarroel, L. et al. (2016). "Release Planning of Mobile Apps Based on User Reviews." ICSE.
32. Di Sorbo, A. et al. (2016). "What Would Users Change in My App? Summarizing App Reviews for Recommending Software Changes." FSE.
33. Keertipati, S. et al. (2016). "Approaches for Prioritizing Feature Improvements Extracted from App Reviews." EASE.
34. Guzman, E. & Maalej, W. (2014). "How Do Users Like This Feature? A Fine Grained Sentiment Analysis of App Reviews." RE.

### Category 8: Bug Report Quality & Issue Taxonomies (NEW)

35. Zimmermann, T. et al. (2010). "What Makes a Good Bug Report?" IEEE TSE.
36. Chaparro, O. et al. (2017). "Detecting Missing Information in Bug Descriptions." FSE.
37. Nielsen, J. (1994). "10 Usability Heuristics for User Interface Design."
38. ISO/IEC 25010:2011. Systems and Software Quality Requirements and Evaluation (SQuaRE).
