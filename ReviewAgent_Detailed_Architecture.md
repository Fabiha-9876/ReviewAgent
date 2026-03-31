# Revised Detailed Architecture — Incorporating All Advisor Feedback

Here's the full architecture, redesigned with theoretical grounding, scoped pipeline, HITL woven throughout, a unified clustering-to-issue framework, literature-grounded issue taxonomies, dataset strategy, expert rubric evaluation, experiment design, ablation studies, and finalized metrics.

---

## Theoretical Foundations

The original design was criticized as "wire-in and wire-out" — components stitched together without a unifying theory. The revised architecture is grounded in three complementary theoretical frameworks that formally justify *why* each stage exists and *why* this combination outperforms isolated approaches.

### Framework 1: Information Extraction Cascade Theory

**Source:** Hearst (1999) "Untangling Text Data Mining"; Sarawagi (2008) "Information Extraction" (Foundations and Trends in Databases).

The pipeline follows the **information extraction cascade** — the established model for progressively transforming unstructured text into structured, actionable knowledge:

> Raw text → Entities → Relations → Structured knowledge → Downstream task

In ReviewAgent:

> Reviews (raw text) → Aspects/Entities (Stage 1) → KG relations (Stage 2) → Issue specs (Stage 3) → Responses (Stage 4b)

**Why this matters:** Each stage monotonically reduces entropy and increases actionability. We define an *actionability score* — a composite measure of how ready a representation is for developer action — and empirically demonstrate its increase across pipeline stages. This provides a formal basis for why progressive structuring outperforms end-to-end models that skip intermediate representations (e.g., directly generating responses from raw reviews).

**What it justifies:** The existence of distinct stages rather than a single end-to-end model. Each stage produces a reusable intermediate representation that can be evaluated independently.

### Framework 2: Human-AI Complementary Decision Making

**Source:** Kamar (2016) "Directions in Hybrid Intelligence" (IJCAI); Bansal et al. (2019) "Does the Whole Exceed Its Parts?" (AAAI); Geifman & El-Yaniv (2017) "Selective Classification for Deep Neural Networks" (NeurIPS).

The HITL checkpoints are instances of **selective prediction** — the system defers to humans when its confidence falls below a calibrated threshold. The theoretical guarantee: selective prediction with calibrated scores provably reduces the pipeline's error rate compared to full automation, at human effort proportional to the uncertainty fraction of inputs.

**Specifically:**
- **Stage 1 checkpoint:** Confidence-based flagging routes ambiguous classifications to experts. The human-AI team achieves classification accuracy exceeding either party alone (complementary performance).
- **Stage 3 checkpoint:** Rubric-based expert validation catches systematic LLM errors (implausible reproduction steps, wrong component mapping) *before* they propagate to downstream stages, preventing error cascading.

**What it justifies:** The placement, design, and number of HITL checkpoints. Not every stage needs one — only high-uncertainty decision points where errors would cascade.

### Framework 3: Constrained Markov Decision Process (CMDP) Theory

**Source:** Altman (1999) *Constrained Markov Decision Processes* (Chapman & Hall/CRC); Dai et al. (2023) "Safe RLHF" (ICLR).

The dual-objective RLHF is formalized as a CMDP:

> **Maximize:** R_quality(response) — helpfulness, specificity, empathy, accuracy, actionability
> **Subject to:** C_compliance(response) ≤ threshold — no unauthorized promises, no information leakage, tone compliance, legal safety

**Why dual outperforms single:** Single-objective RLHF conflates quality and compliance into one reward signal, collapsing two distinct optimization surfaces. The model must implicitly learn the compliance boundary from mixed signals. Dual-objective formulation explicitly enforces the compliance constraint while maximizing quality on the remaining degrees of freedom — navigating the **Pareto frontier** rather than averaging across it.

**What it justifies:** The two-stream feedback design (quality vs. compliance), the progression from KTO → DPO → Constrained PPO, and the specific rubric dimensions for each stream.

---

## The Big Picture (Scoped)

Based on advisor feedback, the pipeline scope is **focused on Stages 1-3 + 4b + 5** as the core contribution. Stage 4a (agentic code resolution) is positioned as **future work** — it constitutes a separate research area with existing solutions (SWE-Agent, RepairAgent, HyperAgent).

```
Raw App Reviews (thousands, noisy, multilingual, vague)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: INTAKE + HITL CHECKPOINT #1                    │
│  Classification → Aspect Extraction → Entity Extraction  │
│  Human experts verify ambiguous classifications           │
│  [Theory: Selective Prediction — Geifman & El-Yaniv]     │
└────────────────────────┬────────────────────────────────┘
                         │  Structured review objects
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: UNIFIED CLUSTERING-TO-ISSUE FRAMEWORK          │
│  KG Construction → Hierarchical Clustering →             │
│  Standardized Issue Schema Mapping                       │
│  Priority ranking via graph centrality                   │
│  [Theory: IE Cascade — progressive structuring]          │
└────────────────────────┬────────────────────────────────┘
                         │  Prioritized issue clusters
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: REVIEW-TO-ISSUE TRANSLATION + HITL CHECKPOINT #2│
│  LLM generates taxonomy-grounded structured issue specs   │
│  Experts review using STANDARD RUBRICS before proceeding  │
│  [Theory: IE Cascade (final structuring) + Selective Pred]│
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4b: RESPONSE GENERATION                           │
│  RAG (5 fixed input sources) + issue-spec-aware          │
│  drafting + self-refinement loop                         │
│  [Theory: Grounded generation via structured context]    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 5: HITL CHECKPOINT #3 — DUAL-OBJECTIVE FEEDBACK   │
│  Stream 1: Quality (helpfulness, specificity, empathy,   │
│            accuracy, actionability)                       │
│  Stream 2: Compliance (safety, promises, tone, legal)    │
│  Progressive RLHF: KTO → DPO → Constrained PPO          │
│  Feedback propagates BACK to Stages 1, 3, and 4b        │
│  [Theory: CMDP — Altman (1999), Dai et al. (2023)]      │
└─────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │  FUTURE WORK: STAGE 4a              │
        │  Agentic Code Resolution            │
        │  (Planner → Navigator → Editor →    │
        │   Executor) consuming validated     │
        │  issue specs from Stage 3           │
        └─────────────────────────────────────┘
```

---

## Stage 1: Intake with HITL Checkpoint #1

**What happens:** Raw reviews come in — thousands of them, written in casual language, slang, multiple languages, sometimes just *"trash app 1 star"* with zero useful information.

The system runs three parallel NLP tasks:

- **Multi-label classification (RoBERTa fine-tuned):** Each review gets one or more labels. For example, *"The camera freezes and I wish you had filters"* gets labeled as both `bug_report` and `feature_request`. The model is trained on datasets like **MAALEJ** (which provides labeled app reviews across categories like bug, feature, rating, user experience).

- **Aspect-based sentiment analysis:** Extracts *what* the user is talking about and *how they feel* about it. *"Great UI but terrible battery drain"* → `{UI: positive, battery: negative}`. This goes beyond whole-review sentiment — it captures the nuance that a user can love one thing and hate another in the same sentence.

- **Entity extraction:** Pulls out concrete details — device names (`iPhone 15 Pro`), OS versions (`iOS 18.2`), specific screens (`checkout page`), feature names (`face recognition`). These become critical metadata for the knowledge graph.

**Where HITL comes in (Checkpoint #1) — Grounded in Selective Prediction:**

The system applies **selective prediction** (Geifman & El-Yaniv, 2017): it computes a calibrated confidence score for each classification and defers to human experts when confidence falls below a tunable threshold. This creates a human-AI complementary team (Bansal et al., 2019) whose classification accuracy exceeds either party alone.

Flagged cases include:
- A review classified as `bug_report` with only 62% confidence (below threshold)
- Reviews where multi-label predictions conflict (classified as both `praise` and `complaint` with similar scores)
- Reviews in underrepresented languages or with heavy slang where the model's confidence is low

An expert looks at these flagged cases, corrects the labels if needed, and **these corrections are logged as training data** for the next fine-tuning cycle (active learning loop).

**Reference papers justifying this stage:**
- Kaur & Sahu (2023) — BERT-RCNN for app review classification
- Shah et al. (arXiv:2108.00663) — Transfer learning for mining feature requests and bug reports
- Guzman & Maalej (2014) — Aspect-based sentiment analysis for app reviews
- Geifman & El-Yaniv (2017) — Selective prediction theory
- Bansal et al. (2019) — Human-AI complementarity

**Output:** Structured review objects — each containing labels, aspect-sentiment pairs, extracted entities, and a confidence score.

**Datasets used:**
- **MAALEJ dataset** — ~4,000 manually classified app reviews (bug, feature, rating, user experience)
- **GUZMAN dataset** — aspect-level annotations for app reviews
- **RRGen's 570K review-response pairs** — for training the classifier at scale with distant supervision

---

## Stage 2: Unified Clustering-to-Issue Framework

This stage implements the **unified framework** from advisor feedback. Instead of just building a KG and ranking, this stage has a **three-layer process** that progressively structures clusters — a direct application of the **information extraction cascade** (Hearst, 1999; Sarawagi, 2008).

### Layer 1 — Knowledge Graph Construction

Every structured review object from Stage 1 becomes nodes and edges in a graph:

- **Aspect nodes:** `login`, `camera`, `notifications`, `payment`, `battery`
- **Entity nodes:** `Samsung Galaxy S24`, `Android 15`, `iOS 18`, `checkout screen`
- **Review nodes:** Each individual review, connected to its aspects and entities
- **Sentiment edges:** Weighted edges between review nodes and aspect nodes carrying polarity and intensity (e.g., `review_142 --[negative, 0.9]--> login`)
- **Temporal edges:** When the review was posted — critical for detecting regressions (e.g., "login complaints spiked after v3.2 update")

**Reference:** ReviewGraph (arXiv:2508.13953); KGCPN (Journal of Cloud Computing, 2023)

### Layer 2 — Hierarchical Clustering

This is where deduplication happens, working in two levels:

- **Level 1 (Aspect-level grouping):** All reviews connected to the `login` aspect node get grouped together. This separates "login issues" from "camera issues" from "battery issues."

- **Level 2 (Sub-clustering within each aspect):** Within the `login` group, distinct sub-clusters emerge:
  - Cluster A: "App crashes on login" (crash-related, 200 reviews)
  - Cluster B: "Forgot password doesn't work" (feature broken, 80 reviews)
  - Cluster C: "Login is too slow" (performance, 45 reviews)

  Sub-clustering uses both **semantic similarity** (embedding-based) and **graph structure** (reviews sharing entity nodes tend to cluster together).

- **Priority ranking via graph centrality:** Using PageRank or betweenness centrality on the KG, the system ranks clusters by importance. A cluster connected to many users, multiple device types, and strong negative sentiment scores higher than a niche complaint.

**Reference:** Villarroel et al. (ICSE 2016) "CLAP"; Di Sorbo et al. (FSE 2016) "SURF"; Keertipati et al. (EASE 2016)

### Layer 3 — Standardized Issue Schema Mapping

Each cluster gets mapped to a **standardized issue schema**:

```json
{
  "issue_id": "CLU-047",
  "issue_type": "bug_report",
  "aspect": "login",
  "sub_category": "crash_on_action",
  "affected_component": "authentication_service",
  "review_count": 200,
  "sentiment_distribution": {"negative": 0.92, "neutral": 0.06, "positive": 0.02},
  "representative_reviews": ["review_142", "review_891", "review_1203"],
  "entities": {
    "devices": ["Samsung Galaxy S24", "Pixel 8"],
    "os_versions": ["Android 14", "Android 15"],
    "app_versions": ["v3.2", "v3.2.1"],
    "screens": ["login_screen"]
  },
  "temporal_pattern": "spike_after_update_v3.2",
  "priority_score": 0.94,
  "kg_subgraph_ref": "subgraph_047"
}
```

This schema is **fixed and standardized** — every cluster, regardless of its content, gets mapped into this format. This enables consistent downstream processing and systematic evaluation.

**Output:** Prioritized, schema-mapped issue clusters with full KG context.

---

## Stage 3: Review-to-Issue Translation with HITL Checkpoint #2

This remains the **core novel contribution**, now strengthened by **taxonomy grounding** and **expert rubric-based validation**.

### How the LLM Agent Works

The LLM receives the standardized cluster schema from Stage 2 and generates a **GitHub-issue-quality specification**. It's guided by **issue taxonomies from SE literature**:

- For `bug_report` type clusters → Follows the **Zimmermann template** (Zimmermann et al., 2010): summary, steps to reproduce, expected behavior, actual behavior, environment, severity. The key challenge: users never provide steps to reproduce — the LLM must **infer** them from review text, app context, and similar known issues.

- For `feature_request` type clusters → Generates a **user story** format: *"As a [user type], I want [capability], so that [benefit]."* Plus acceptance criteria derived from user requests.

- For `performance_complaint` type clusters → Maps to **non-functional requirement** categories per ISO/IEC 25010: latency, memory consumption, battery drain, network usage. Quantifies where possible.

- For `usability_issue` type clusters → Aligns with **Nielsen's usability heuristics** (1994): visibility of system status, match between system and real world, user control, consistency, error prevention, recognition over recall, flexibility, aesthetic design, error recovery, help/documentation.

- For `compatibility_issue` type clusters → Generates a **device-OS-version matrix** from the entity data.

**Reference papers justifying this stage:**
- Zimmermann et al. (2010) "What Makes a Good Bug Report?" — bug report template and quality criteria
- Chaparro et al. (2017) "Detecting Missing Information in Bug Descriptions" — completeness assessment
- Nielsen (1994) — usability heuristic categories
- ISO/IEC 25010:2011 — NFR quality model

### Example Transformation

Input cluster (200 reviews like):
> *"app keeps crashing when I try to login"*
> *"cant log in anymore since the update crashes every time"*
> *"login broken on my samsung fix it!!!"*

LLM-generated structured issue:

```markdown
## Title: Login screen crashes on authentication attempt (post-v3.2 regression)

## Type: Bug Report

## Steps to Reproduce:
1. Open the app (v3.2 or v3.2.1)
2. Enter valid credentials on the login screen
3. Tap the "Sign In" button
4. App crashes immediately (force close)

## Expected Behavior:
User should be authenticated and redirected to the home screen.

## Actual Behavior:
App crashes (force close) upon tapping Sign In. No error message displayed.

## Environment:
- Devices: Samsung Galaxy S24, Google Pixel 8 (primarily Android)
- OS: Android 14, Android 15
- App versions: v3.2, v3.2.1 (not reported on v3.1.x — likely regression)

## Severity: Critical (P0)
- Blocks core functionality (login)
- Affects ~200 users in review sample
- Regression introduced in v3.2

## Affected Component: authentication_service (inferred)

## KG Context: Related to cluster CLU-048 (slow login performance) —
   may share root cause in authentication refactor shipped in v3.2.
```

### Where HITL Comes In (Checkpoint #2) — Expert Rubric Review

Before this issue spec proceeds to response generation (Stage 4b), **domain experts review it using a standardized rubric** — another instance of **selective prediction** applied at the structured output level:

| Dimension | Question the Expert Asks | Score (1-5) |
|---|---|---|
| **Completeness** | Are all fields filled? Are steps to reproduce present? Is environment info included? | ___ |
| **Accuracy** | Are the inferred reproduction steps plausible? Is the component mapping correct? Is the severity justified? | ___ |
| **Actionability** | Could a developer pick this up and start debugging without reading the original 200 reviews? | ___ |
| **Specificity** | Is this issue clearly distinct from CLU-048 (slow login)? Would someone confuse the two? | ___ |
| **Clarity** | Is the language precise and unambiguous? No jargon confusion? | ___ |

**Multiple experts** (at least 2-3) independently score each issue spec. Their agreement is measured via **Cohen's kappa** (for 2 raters) or **Krippendorff's alpha** (for 3+ raters). This provides:

1. A **quality score** for each generated issue spec
2. A **human baseline** — how do expert-written issue specs score on the same rubric?
3. **Training signal** — low-scoring dimensions tell the system exactly *what* to improve

Issues scoring below a threshold get sent back to the LLM with the expert's dimension-level feedback for **regeneration** (active learning at Stage 3).

**Output:** Validated, taxonomy-grounded, structured issue specifications.

---

## Stage 4b: Response Generation (Issue-Spec-Aware)

This stage generates user-facing responses that are **aware of the structured issue spec** from Stage 3 — the key coupling that makes responses specific rather than generic.

### RAG Input Materials (Fixed — 5 Sources)

| RAG Input Source | Purpose | Availability |
|---|---|---|
| App's past review-response pairs (from RRGen 570K) | Learn tone, style, common acknowledgments | Available |
| App changelog / release notes | Reference specific versions and fixes | Scrapable from Play Store |
| App FAQ / help documentation | Provide accurate workarounds | App-specific, manual collection |
| Generated issue spec from Stage 3 | Make response issue-aware and specific | Pipeline output |
| Similar past responses (retrieved by semantic similarity) | Template-level guidance | From RRGen dataset |

### Context-Aware Drafting

The response references the *specific* issue from the structured spec:
- *"We've identified a crash affecting login on Android devices running v3.2..."*
- *"Our team is actively investigating this issue, which appears to be related to the v3.2 update. We'll update you once resolved."*

### Self-Refinement Loop

The model generates a response, then critiques it across three dimensions:
1. Is it too vague? (Must reference the specific issue, not be a generic template)
2. Does it make unauthorized promises? (Compliance check)
3. Is it empathetic enough? (Tone check)

It iterates 2-3 times before producing a final draft.

**Reference papers justifying this stage:**
- Gao et al. (RRGen, ASE 2019) — Review response generation baseline
- CoRe (arXiv:2010.06301) — Contextual review response generation
- Self-Improving Response Gen (ECNLP 2024) — Self-refinement methodology

**Output:** A draft response ready for human review.

---

## Stage 5: HITL Checkpoint #3 — Dual-Objective Feedback (CMDP-Grounded)

The final stage implements the **constrained MDP formulation** (Altman, 1999): maximize quality reward subject to compliance constraints.

### Stream 1 — Quality (The Reward to Maximize)

Experts score the response on:
- **Helpfulness:** Does it address the user's actual complaint?
- **Specificity:** Does it reference the specific issue, or is it a generic template?
- **Empathy:** Does it acknowledge the user's frustration appropriately?
- **Accuracy:** Does it correctly describe the issue status?
- **Actionability:** Does it tell the user what to do next (update the app, try a workaround, etc.)?

### Stream 2 — Policy Compliance (The Constraint)

- Does it make promises the team can't keep?
- Does it leak internal information (code details, team names)?
- Does it follow the company's tone and communication guidelines?
- Is it legally safe (no admission of liability, no guarantees)?

### Progressive RLHF Strategy

| Phase | Method | Data Requirement | When |
|---|---|---|---|
| Phase 1 | **KTO** (Ethayarajh et al., 2024) | Binary good/bad signals (~500 responses) | Early, small data |
| Phase 2 | **DPO** (Rafailov et al., 2023) | Paired preferences ("A is better than B") (~1000 pairs) | Mid, growing data |
| Phase 3 | **Constrained PPO** (Dai et al., 2023) | Dimension-level scores + compliance labels (~2000 responses) | Scale, rich feedback |

### Feedback Propagation — The Critical Loop

Feedback doesn't just improve Stage 5 — it **flows backward**:

- **Back to Stage 1:** If experts consistently reclassify certain review types, the classification model gets retrained on these corrections.
- **Back to Stage 3:** Rubric scores on issue specs identify systematic weaknesses (e.g., "reproduction steps are always too vague for performance issues") → the LLM prompt/fine-tuning is adjusted.
- **Back to Stage 4b:** Response quality scores drive the RLHF training loop through the KTO → DPO → Constrained PPO progression.

**Reference papers justifying this stage:**
- Dai et al. (2023) "Safe RLHF" — Dual-objective RLHF with constrained optimization
- Altman (1999) — CMDP theory
- Ethayarajh et al. (2024) — KTO methodology
- Rafailov et al. (2023) — DPO methodology
- MA-RLHF (2024) — Macro-action level feedback

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

## Ablation Studies (7 Studies)

| ID | What's Removed | What It Tests | Measured By |
|---|---|---|---|
| **A1** | No KG (skip Stage 2, feed raw classified reviews to Stage 3) | Value of knowledge graph clustering | Issue spec rubric scores |
| **A2** | No hierarchical clustering (flat clustering instead of aspect → sub-cluster) | Value of the two-level hierarchy | Cluster purity, NMI; downstream issue spec quality |
| **A3** | No taxonomy grounding (LLM generates free-form issues) | Value of literature-grounded templates | Issue spec rubric scores (completeness, actionability) |
| **A4** | No HITL at Stage 3 (skip expert validation) | Value of human checkpoint | End-to-end response quality |
| **A5** | No RAG (generate responses from issue spec + LLM only) | Value of retrieval augmentation | Response BLEU, human eval |
| **A6** | No issue spec in response gen (RAG but no structured issue from Stage 3) | Value of coupling Stages 3 and 4b | Response specificity, accuracy scores |
| **A7** | Single-stream feedback (merge quality and compliance into one score) | Value of dual-objective decomposition | Preference win rate, safety violation rate |

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
- Same RAG corpus configuration (5 fixed sources)

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
| Stage 2: Clustering | Cluster purity, NMI | Deduplication precision/recall | Standard clustering (K-means) |
| Stage 3: Translation | Rubric scores + Krippendorff's alpha | BERTScore vs. human specs | Human-written issues |
| Stage 4b: Response | BLEU, ROUGE-L, BERTScore | Human eval (helpfulness, specificity, empathy) | CoRe, RRGen |
| Stage 5: RLHF | Preference win rate | Safety violation rate, per-dimension improvement | Single-objective tuning |
| End-to-end | % of reviews reaching validated response | Time from intake to response draft | Manual triage baseline |

---

## Dataset Strategy

| Dataset | What It Provides | Stage | Reference |
|---|---|---|---|
| **RRGen (~570K pairs)** | Google Play review-response pairs | Stages 1, 4b | Gao et al. (ASE 2019) |
| **MAALEJ (~4K labeled)** | Multi-label classified app reviews | Stage 1 | Maalej et al. (2016) |
| **GUZMAN** | Aspect-level annotations | Stage 1 | Guzman & Maalej (2014) |
| **Open-source Android apps with GitHub repos** | Ground-truth: reviews → issues → fixes | Stage 3 | Manual identification |
| **Custom gold-standard (to be built)** | Expert-written issue specs for 200-300 clusters | Stage 3 eval | Novel contribution |
| **SWE-bench** | Agentic code resolution benchmark | Future work (4a) | Jimenez et al. (2024) |

The **custom gold-standard dataset** is itself a contribution — no such dataset exists. Building it involves:
1. Selecting 200-300 review clusters from the KG
2. Having 3+ experts independently write structured issue specs for each cluster
3. Measuring inter-annotator agreement
4. Using this as the evaluation benchmark for RQ1

---

## Reference Papers per Methodology Step

| Methodology Step | Reference Papers | What They Justify |
|---|---|---|
| **Step 1: Dataset collection** | Gao et al. (RRGen, ASE 2019); Maalej et al. (2016) | Use of RRGen 570K; MAALEJ classification scheme |
| **Step 2: RoBERTa classification** | Kaur & Sahu (BERT-RCNN, 2023); Shah et al. (arXiv:2108.00663) | Transfer learning for app review classification |
| **Step 2: Aspect-based sentiment** | Guzman & Maalej (2014); NLP survey (PeerJ CS, 2024) | Aspect extraction methodology |
| **Step 2: HITL for classification** | Geifman & El-Yaniv (2017); Bansal et al. (2019) | Confidence-based deferral to humans |
| **Step 3: KG construction** | ReviewGraph (arXiv:2508.13953); KGCPN (J. Cloud Computing, 2023) | KG for review representation |
| **Step 3: Hierarchical clustering** | Villarroel et al. (ICSE 2016) "CLAP"; Di Sorbo et al. (FSE 2016) "SURF" | Clustering reviews for prioritization |
| **Step 3: Graph centrality** | Keertipati et al. (EASE 2016) | PageRank/betweenness for issue ranking |
| **Step 4: Review-to-issue translation** | Zimmermann et al. (2010); Chaparro et al. (2017) | Bug report structure and quality criteria |
| **Step 4: Issue taxonomy grounding** | Nielsen (1994); ISO/IEC 25010:2011 | Usability and performance issue structures |
| **Step 5: RAG for response gen** | Gao et al. (RRGen); CoRe (arXiv:2010.06301); Self-Improving (ECNLP 2024) | RAG and self-refinement |
| **Step 6: KTO methodology** | Ethayarajh et al. (2024) | Binary feedback RLHF |
| **Step 6: DPO methodology** | Rafailov et al. (2023) | Paired preference RLHF |
| **Step 6: Constrained PPO** | Dai et al. (2023); Altman (1999) | Dual-objective optimization |
| **Step 7: Expert rubric evaluation** | Zimmermann et al. (2010); Chaparro et al. (2017) | Rubric dimensions |
| **Theoretical: IE cascade** | Hearst (1999); Sarawagi (2008) | Progressive structuring justification |
| **Theoretical: Human-AI complementarity** | Kamar (2016); Bansal et al. (2019) | HITL checkpoint placement |
| **Theoretical: CMDP** | Altman (1999); Dai et al. (2023) | Dual-objective RLHF formalization |

---

## How This Answers the Research Questions

- **RQ1** (Translation accuracy): Compare LLM-generated issue specs against the expert gold-standard using the **5-dimension rubric** scores across 4 conditions. The IE cascade theory predicts that taxonomy-grounded generation (condition a) outperforms free-form (condition b), which outperforms no structuring (condition c).

- **RQ2** (Coupled response generation): Compare issue-spec-aware responses against context-unaware baselines (CoRe, RRGen) and the ablated version (RAG without issue spec). The coupling should produce more specific, accurate responses because the structured issue spec provides grounded context.

- **RQ3** (Dual-objective RLHF): Compare dual-objective (quality + compliance) against single-objective variants. The CMDP theory predicts that explicit constraint enforcement outperforms implicit learning of the compliance boundary from mixed signals.

---

## Summary of What Changed from Advisor Feedback

| Original Design | Revised Design |
|---|---|
| HITL only at Stage 5 | **HITL at 3 checkpoints** (classification, translation, final review) |
| KG + ad-hoc clustering | **Unified 3-layer framework** (KG → hierarchical clustering → schema mapping) |
| Free-form issue generation | **Taxonomy-grounded** generation (Zimmermann, Nielsen, ISO 25010) |
| Implicit dataset assumptions | **Explicit dataset strategy** with existing datasets + custom gold-standard |
| Thumbs up/down evaluation | **5-dimension expert rubric** with inter-annotator agreement |
| No theoretical grounding | **3 theoretical frameworks** (IE cascade, human-AI complementarity, CMDP) |
| Full pipeline including code resolution | **Scoped to Stages 1-3 + 4b + 5**; code resolution as future work |
| No experiment design | **3 experiments + 7 ablation studies** with defined variables and statistical tests |
| No per-step references | **Every methodology step backed by reference papers** |
| Pipeline felt like "wire-in wire-out" | Each stage justified by theory with formal claims |

The result is a rigorous, evaluable, and publishable system — one where every design decision is theoretically grounded, validated by experts, and systematically measurable.
