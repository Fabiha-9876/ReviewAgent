# ReviewAgent — An End-to-End Agentic System for Automated App Review Triage, Resolution, and Response with Human-in-the-Loop Refinement

## The Core Research Gap

No existing work connects these five research areas into a unified system. Each category operates in isolation:

- **Classification papers** stop at labeling reviews
- **Response generation papers** ignore the actual fix
- **Agentic SE papers** assume well-structured GitHub issues, not noisy reviews
- **KG papers** summarize but don't act
- **RLHF papers** optimize language models but aren't applied to this domain

**The critical missing piece** identified across all papers: **review-to-issue translation** — converting noisy, vague, non-technical app store reviews into structured, actionable issue specifications that an SE agent can resolve.

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────┐
│                    STAGE 1: INTAKE                   │
│  Multi-label classifier (RoBERTa fine-tuned)         │
│  + Aspect-based sentiment analysis                   │
│  + Entity extraction (feature, screen, device, OS)   │
│  Input: Raw app reviews                              │
│  Output: Labeled, structured review objects           │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              STAGE 2: KNOWLEDGE GRAPH                │
│  Build review KG: aspect nodes + sentiment edges     │
│  Deduplicate & cluster similar complaints            │
│  Graph centrality → priority ranking                 │
│  Input: Structured review objects                    │
│  Output: Prioritized issue clusters with KG context  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│         STAGE 3: REVIEW-TO-ISSUE TRANSLATION         │
│  (THE NOVEL CONTRIBUTION)                            │
│  LLM agent converts noisy review clusters into       │
│  structured issue specs:                             │
│    - Reproduction steps (inferred)                   │
│    - Affected component                              │
│    - Severity score                                  │
│    - Linked KG context                               │
│  Input: Issue clusters + KG                          │
│  Output: GitHub-issue-quality specifications         │
└──────────┬─────────────────────┬────────────────────┘
           │                     │
┌──────────▼──────────┐ ┌───────▼─────────────────────┐
│  STAGE 4a: AGENTIC  │ │  STAGE 4b: RESPONSE         │
│  ISSUE RESOLUTION   │ │  GENERATION                  │
│  Multi-agent system: │ │  RAG + app context +         │
│  Planner → Navigator │ │  self-refinement             │
│  → Editor → Executor │ │  Generates user-facing       │
│  Produces patches    │ │  response acknowledging      │
│  with test validation│ │  the issue + resolution      │
└──────────┬──────────┘ └───────┬─────────────────────┘
           │                     │
┌──────────▼─────────────────────▼────────────────────┐
│           STAGE 5: HUMAN-IN-THE-LOOP                 │
│  Dual-objective feedback (Safe RLHF):                │
│    Stream 1: Quality/helpfulness (thumbs up/down)    │
│    Stream 2: Policy compliance (safe/unsafe)         │
│  Macro-action level feedback on response sections    │
│  KTO initially → DPO → Constrained PPO at scale     │
│  Iterative retraining loop                           │
└─────────────────────────────────────────────────────┘
```

---

## What Makes This Novel (Research Contributions)

| Contribution | Gap It Fills | Papers It Builds On |
|---|---|---|
| **1. Review-to-Issue Translation Agent** | No existing work converts noisy reviews → structured issue specs | Category 1 (classification) + Category 5 (agentic SE) |
| **2. KG-guided issue prioritization** | KG papers do summarization but not actionable triage | Category 4 (KG) + Category 1 (aspect extraction) |
| **3. Coupled resolution + response** | Response papers ignore the fix; SE papers ignore the user | Category 2 (response gen) + Category 5 (repair agents) |
| **4. Dual-objective RLHF for review responses** | RLHF papers haven't been applied to app review domain | Category 3 (Safe RLHF, MA-RLHF) + Category 2 |
| **5. End-to-end evaluation benchmark** | No benchmark exists for the full pipeline | All categories |

---

## Concrete Research Questions

1. **RQ1:** How accurately can an LLM agent translate noisy app review clusters into structured issue specifications compared to human-written GitHub issues?
2. **RQ2:** Does KG-based deduplication and prioritization reduce developer triage effort compared to raw review feeds?
3. **RQ3:** Does coupling code resolution with response generation produce more specific, actionable responses than context-unaware generation?
4. **RQ4:** Does dual-objective RLHF (quality + compliance) outperform single-objective preference tuning for app review responses?
5. **RQ5:** What is the end-to-end resolution rate from raw review to validated patch?

---

## Suggested Solution Approach

Based on advisor feedback, the following directions should be incorporated into the solution design:

### 1. Integrate Human-in-the-Loop (HITL) Throughout the Pipeline

Rather than confining HITL to the final stage (Stage 5), the approach should embed human oversight at **critical decision points** across the entire pipeline — particularly at the **review-to-issue translation** stage (Stage 3). This ensures that the LLM-generated issue specifications are validated before they trigger downstream code resolution or response generation. HITL integration points include:

- **Post-classification review:** Domain experts can verify or correct multi-label classifications on ambiguous reviews before they enter the KG.
- **Post-translation validation:** Before an LLM-generated issue spec is passed to the agentic resolution system, a human reviewer can confirm that the reproduction steps, severity, and affected component are reasonable.
- **Feedback propagation:** Human corrections at any stage feed back into the system's training loop, not just at the response generation level.

### 2. Unified Framework for Clustering Reviews into Structured Issues

A **unified clustering-to-structuring framework** should be designed that seamlessly transforms raw review clusters into well-defined, structured issue representations. This framework should:

- Define a **standardized issue schema** (e.g., fields for: issue type, affected component, reproduction steps, severity, user impact, device/OS context, frequency count).
- Use a **hierarchical clustering approach** — first group reviews by aspect (from the KG), then sub-cluster by specific complaint within each aspect, and finally map each sub-cluster to a structured issue template.
- Ensure the framework is **end-to-end differentiable** where possible, so that improvements in clustering quality directly improve issue specification quality downstream.

### 3. Ground the Work in Structured Issue Taxonomies from Literature

The structured issues generated by the system should align with **established issue taxonomies** reported in the software engineering literature. Key structured issue types to consider include:

- **Bug reports** — following templates from platforms like GitHub, Bugzilla, and Jira (summary, steps to reproduce, expected vs. actual behavior, environment details).
- **Feature requests** — structured as user stories or requirement specifications (as a [user], I want [feature], so that [benefit]).
- **Performance complaints** — mapped to non-functional requirement categories (latency, memory, battery drain, network usage).
- **Usability issues** — aligned with HCI heuristic evaluation categories (Nielsen's heuristics: visibility, feedback, error prevention, etc.).
- **Compatibility issues** — structured around device-OS-version matrices.

Grounding in these taxonomies ensures the system's output is **comparable to human-authored issues** and enables rigorous evaluation against existing benchmarks.

### 4. Dataset Availability and Construction

The solution approach should clearly address dataset requirements and availability:

- **Existing datasets:** Leverage RRGen's ~570K Google Play review-response pairs, the MAALEJ dataset (multi-label app review classifications), and the GUZMAN dataset (aspect-based review annotations).
- **Ground-truth linking:** Identify open-source Android apps that have both Google Play reviews AND GitHub issue trackers, enabling ground-truth mapping from reviews → issues → code fixes.
- **Custom annotation:** For the novel review-to-issue translation task, a **gold-standard dataset** will need to be constructed — pairing clusters of app reviews with expert-written structured issue specifications. This becomes a key artifact and contribution of the research.
- **Scale considerations:** Clearly define training, validation, and test splits, and ensure sufficient volume for fine-tuning (classification) and RLHF (response generation).

### 5. Expert Review of LLM-Generated Issues Using Standard Rubrics

To rigorously evaluate the quality of LLM-generated issue specifications (Stage 3 output), **domain experts should review them using standardized rubrics**. The rubric should assess:

| Rubric Dimension | Description | Scale |
|---|---|---|
| **Completeness** | Does the issue spec contain all necessary fields (steps to reproduce, severity, affected component, environment)? | 1-5 |
| **Accuracy** | Are the inferred reproduction steps and component mappings factually correct? | 1-5 |
| **Actionability** | Could a developer act on this issue spec without needing to read the original reviews? | 1-5 |
| **Specificity** | Is the issue spec specific enough to distinguish it from related but different issues? | 1-5 |
| **Clarity** | Is the language clear, unambiguous, and well-structured? | 1-5 |

- Multiple experts should independently rate each issue spec to compute **inter-annotator agreement** (Cohen's kappa or Krippendorff's alpha).
- The rubric-based evaluation serves as the **primary metric for RQ1** and provides a human baseline for comparison.
- This rubric can also be used as the basis for HITL feedback in the iterative improvement loop — experts don't just approve/reject, they score on these dimensions, providing richer training signal.

---

## Proposed Evaluation

- **Dataset:** Scrape Google Play review-response pairs (RRGen's ~570K dataset) + link to open-source apps with GitHub repos for ground-truth issue-to-fix mapping
- **Baselines:** Each stage compared against its standalone SOTA (BERT-RCNN for classification, CoRe for response gen, SWE-Agent for resolution)
- **Metrics per stage:**
  - Stage 1: F1 (multi-label classification)
  - Stage 2: Cluster purity, deduplication precision
  - Stage 3: Issue spec quality (human-rated completeness, actionability)
  - Stage 4a: Patch correctness (SWE-bench style)
  - Stage 4b: BLEU + human evaluation (helpfulness, specificity)
  - Stage 5: Preference win rate over iterations

---

## Why This Is Publishable

- **Novelty:** First end-to-end system connecting all five research threads
- **The review-to-issue translation component** is entirely unexplored in literature
- **Dual-objective RLHF applied to app review responses** is a new application
- **Practical impact:** Developers currently manually read thousands of reviews — this automates the full loop
- **Multiple publication angles:** Full system paper (ICSE/FSE), the translation component alone (ASE), the RLHF application (EMNLP), the KG triage (RE conference)

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
