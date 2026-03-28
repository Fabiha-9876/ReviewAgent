# Revised Detailed Architecture — Incorporating Advisor Feedback

Here's the full architecture, redesigned with HITL woven throughout, a unified clustering-to-issue framework, literature-grounded issue taxonomies, dataset strategy, and expert rubric evaluation.

---

## The Big Picture

The original architecture had HITL only at the end (Stage 5). The revised architecture treats **human experts as active participants at every critical junction**, not just final reviewers. Think of it as a pipeline where the LLM does the heavy lifting, but humans act as **quality gates** at key checkpoints — catching errors early before they propagate downstream.

```
Raw App Reviews (thousands, noisy, multilingual, vague)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: INTAKE + HITL CHECKPOINT #1                    │
│  Classification → Aspect Extraction → Entity Extraction  │
│  Human experts verify ambiguous classifications           │
└────────────────────────┬────────────────────────────────┘
                         │  Structured review objects
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: UNIFIED CLUSTERING-TO-ISSUE FRAMEWORK          │
│  KG Construction → Hierarchical Clustering →             │
│  Standardized Issue Schema Mapping                       │
│  Priority ranking via graph centrality                   │
└────────────────────────┬────────────────────────────────┘
                         │  Prioritized issue clusters
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: REVIEW-TO-ISSUE TRANSLATION + HITL CHECKPOINT #2│
│  LLM generates structured issue specs                     │
│  Grounded in literature-based issue taxonomies            │
│  Experts review using STANDARD RUBRICS before proceeding  │
└───────────┬───────────────────────────┬─────────────────┘
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────────┐
│  STAGE 4a: AGENTIC    │   │  STAGE 4b: RESPONSE       │
│  CODE RESOLUTION      │   │  GENERATION               │
│  Plan→Navigate→Edit   │   │  RAG + resolution-aware   │
│  →Test→Validate       │   │  drafting + self-refine   │
└───────────┬───────────┘   └───────────┬───────────────┘
            │                           │
            ▼                           ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 5: HITL CHECKPOINT #3 — DUAL-OBJECTIVE FEEDBACK   │
│  Expert rubric scoring (not just thumbs up/down)         │
│  Stream 1: Quality dimensions (completeness, accuracy,   │
│            actionability, specificity, clarity)           │
│  Stream 2: Policy compliance (safety, promises, tone)    │
│  Feedback propagates BACK to Stages 1, 3, and 4b        │
└─────────────────────────────────────────────────────────┘
```

---

## Stage 1: Intake with HITL Checkpoint #1

**What happens:** Raw reviews come in — thousands of them, written in casual language, slang, multiple languages, sometimes just *"trash app 1 star"* with zero useful information.

The system runs three parallel NLP tasks:

- **Multi-label classification (RoBERTa fine-tuned):** Each review gets one or more labels. For example, *"The camera freezes and I wish you had filters"* gets labeled as both `bug_report` and `feature_request`. The model is trained on datasets like **MAALEJ** (which provides labeled app reviews across categories like bug, feature, rating, user experience).

- **Aspect-based sentiment analysis:** Extracts *what* the user is talking about and *how they feel* about it. *"Great UI but terrible battery drain"* → `{UI: positive, battery: negative}`. This goes beyond whole-review sentiment — it captures the nuance that a user can love one thing and hate another in the same sentence.

- **Entity extraction:** Pulls out concrete details — device names (`iPhone 15 Pro`), OS versions (`iOS 18.2`), specific screens (`checkout page`), feature names (`face recognition`). These become critical metadata for the knowledge graph.

**Where HITL comes in (Checkpoint #1):** Not every review gets human attention — that would defeat the purpose. Instead, the system flags **ambiguous classifications** for expert review. For example:

- A review classified as `bug_report` with only 62% confidence (below a tunable threshold)
- Reviews where multi-label predictions conflict (classified as both `praise` and `complaint` with similar scores)
- Reviews in underrepresented languages or with heavy slang where the model's confidence is low

An expert looks at these flagged cases, corrects the labels if needed, and **these corrections are logged as training data** for the next fine-tuning cycle. Over time, the model gets better at these edge cases, and fewer reviews need human attention.

**Output:** Structured review objects — each containing labels, aspect-sentiment pairs, extracted entities, and a confidence score.

**Datasets used:**
- **MAALEJ dataset** — ~4,000 manually classified app reviews (bug, feature, rating, user experience)
- **GUZMAN dataset** — aspect-level annotations for app reviews
- **RRGen's 570K review-response pairs** — for training the classifier at scale with distant supervision

---

## Stage 2: Unified Clustering-to-Issue Framework

This is where the advisor's suggestion for a **unified framework** fundamentally reshapes the original design. Instead of just building a KG and ranking, this stage now has a **three-layer process** that transforms clusters into structured representations.

### Layer 1 — Knowledge Graph Construction

Every structured review object from Stage 1 becomes nodes and edges in a graph:

- **Aspect nodes:** `login`, `camera`, `notifications`, `payment`, `battery` — these are the *things* users talk about
- **Entity nodes:** `Samsung Galaxy S24`, `Android 15`, `iOS 18`, `checkout screen` — the *context*
- **Review nodes:** Each individual review, connected to its aspects and entities
- **Sentiment edges:** Weighted edges between review nodes and aspect nodes carrying polarity and intensity (e.g., `review_142 --[negative, 0.9]--> login`)
- **Temporal edges:** When the review was posted — critical for detecting regressions (e.g., "login complaints spiked after v3.2 update")

### Layer 2 — Hierarchical Clustering

This is where deduplication happens, and it works in two levels:

- **Level 1 (Aspect-level grouping):** All reviews connected to the `login` aspect node get grouped together. This is coarse — it separates "login issues" from "camera issues" from "battery issues."

- **Level 2 (Sub-clustering within each aspect):** Within the `login` group, you might have three distinct sub-clusters:
  - Cluster A: "App crashes on login" (crash-related, 200 reviews)
  - Cluster B: "Forgot password doesn't work" (feature broken, 80 reviews)
  - Cluster C: "Login is too slow" (performance, 45 reviews)

  Sub-clustering uses both **semantic similarity** (embedding-based, comparing the meaning of reviews) and **graph structure** (reviews sharing the same entity nodes — e.g., all mentioning Samsung devices — tend to cluster together).

- **Priority ranking via graph centrality:** Using PageRank or betweenness centrality on the KG, the system ranks clusters by importance. A cluster connected to many users, multiple device types, and strong negative sentiment scores higher than a niche complaint from a single device.

### Layer 3 — Standardized Issue Schema Mapping

Here's the key addition from the advisor's feedback. Each cluster doesn't just stay as a "bag of reviews" — it gets mapped to a **standardized issue schema**:

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

This schema is **fixed and standardized** — every cluster, regardless of its content, gets mapped into this format. This is critical because it makes the downstream stages consistent and enables systematic evaluation.

**Output:** Prioritized, schema-mapped issue clusters with full KG context.

---

## Stage 3: Review-to-Issue Translation with HITL Checkpoint #2

This remains the **core novel contribution**, but now it's significantly strengthened by two advisor suggestions: **grounding in literature-based issue taxonomies** and **expert rubric-based validation**.

### How the LLM Agent Works

The LLM receives the standardized cluster schema from Stage 2 and generates a **GitHub-issue-quality specification**. But it doesn't generate in a vacuum — it's guided by **issue taxonomies from SE literature**:

- For `bug_report` type clusters → The LLM follows the **bug report template** established in literature (Zimmermann et al., 2010): summary, steps to reproduce, expected behavior, actual behavior, environment, severity. The key challenge is that users never provide steps to reproduce — the LLM must **infer** them from the review text, app context, and similar known issues.

- For `feature_request` type clusters → The LLM generates a **user story** format: *"As a [user type], I want [capability], so that [benefit]."* It also generates acceptance criteria derived from what users are asking for.

- For `performance_complaint` type clusters → Maps to **non-functional requirement** categories: latency, memory consumption, battery drain, network usage. The LLM quantifies where possible (*"Users report 10+ second load times on the checkout screen"*).

- For `usability_issue` type clusters → Aligns with **Nielsen's usability heuristics**: visibility of system status, match between system and real world, user control, consistency, error prevention, recognition over recall, flexibility, aesthetic design, error recovery, help/documentation. The LLM identifies which heuristic is violated.

- For `compatibility_issue` type clusters → Generates a **device-OS-version matrix** showing which combinations are affected, drawn from the entity data in the cluster schema.

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

This is the most important HITL checkpoint. Before this issue spec goes to the agentic code resolution system (Stage 4a), **domain experts review it using a standardized rubric**:

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
3. **Training signal** — low-scoring dimensions tell the system exactly *what* to improve (e.g., if accuracy is consistently low, the LLM needs better grounding in app documentation)

Issues scoring below a threshold get sent back to the LLM with the expert's dimension-level feedback for **regeneration**. This creates an active learning loop at Stage 3 itself.

**Output:** Validated, taxonomy-grounded, structured issue specifications ready for resolution.

---

## Stage 4a: Agentic Issue Resolution

Now the system has **high-quality, human-validated issue specs** — something no prior work has achieved from raw reviews. The multi-agent resolution system receives these and works like a software engineering team:

**Planner Agent:**
- Reads the issue spec and the codebase
- Creates a resolution strategy: *"This is a crash in authentication_service, likely introduced in the v3.2 refactor. Check the auth handler for null pointer or race condition issues. Look at the git diff between v3.1 and v3.2 for the authentication module."*
- Breaks the fix into subtasks

**Navigator Agent:**
- Searches the codebase using the affected component field from the issue spec
- Finds relevant files: `src/auth/login_handler.py`, `src/auth/session_manager.py`
- Identifies the specific function or code path likely responsible
- Uses the KG context to check if related issues (CLU-048, slow login) share code paths

**Editor Agent:**
- Writes the actual code fix based on the Planner's strategy and Navigator's findings
- If the Navigator found that v3.2 introduced a new async call in `login_handler.py` that doesn't handle a null session token, the Editor patches that specific path

**Executor Agent:**
- Runs the existing test suite to check for regressions
- Generates new test cases specifically for the fix (e.g., test login with null session token)
- If tests fail, loops back to the Editor with failure details
- Produces a final validated patch with a test report

**Output:** A code patch (diff) + test results + confidence score.

---

## Stage 4b: Response Generation (Parallel with 4a)

This runs simultaneously with Stage 4a and generates the user-facing response. The key improvement from the advisor's feedback is that this stage is now **resolution-aware** — it knows what the agentic system found and fixed.

**RAG (Retrieval-Augmented Generation):**
- Retrieves the app's past responses to similar complaints (from the RRGen dataset and the app's own history)
- Retrieves the app's FAQ, changelog, and known issues
- Retrieves the resolution status from Stage 4a (if available — since they run in parallel, Stage 4b may draft initially without it and refine once 4a completes)

**Context-aware drafting:**
- The response references the *specific* issue: *"We've identified a crash affecting login on Android devices running v3.2..."*
- If Stage 4a produced a fix: *"...and a fix will be included in our next update (v3.2.2)."*
- If Stage 4a couldn't fix it: *"...and our team is actively investigating. We'll update you once resolved."*

**Self-refinement loop:**
- The model generates a response, then critiques it: Is it too vague? Does it make unauthorized promises? Is it empathetic enough?
- It iterates 2-3 times before producing a final draft

**Output:** A draft response ready for human review.

---

## Stage 5: HITL Checkpoint #3 — Dual-Objective Feedback with Rubric Scoring

The final stage is now **richer** than simple thumbs up/down. Experts provide **dimension-level feedback** on both the code patch and the response:

### For the Response (Stream 1 — Quality)

Experts score the response on the same rubric philosophy — but adapted for responses:
- **Helpfulness:** Does it address the user's actual complaint?
- **Specificity:** Does it reference the specific issue, or is it a generic template?
- **Empathy:** Does it acknowledge the user's frustration appropriately?
- **Accuracy:** Does it correctly describe the issue status and resolution?
- **Actionability:** Does it tell the user what to do next (update the app, try a workaround, etc.)?

### For the Response (Stream 2 — Policy Compliance)

- Does it make promises the team can't keep?
- Does it leak internal information (code details, team names)?
- Does it follow the company's tone and communication guidelines?
- Is it legally safe (no admission of liability, no guarantees)?

### Feedback Propagation — The Critical Loop

This is what makes the system truly learn over time. Feedback doesn't just improve Stage 5 — it **flows backward**:

- **Back to Stage 1:** If experts consistently reclassify certain review types, the classification model gets retrained on these corrections.
- **Back to Stage 3:** Rubric scores on issue specs identify systematic weaknesses (e.g., "reproduction steps are always too vague for performance issues") → the LLM prompt/fine-tuning is adjusted.
- **Back to Stage 4b:** Response quality scores drive the RLHF training loop:
  - **KTO** initially (binary good/bad signals, works with small data)
  - **DPO** as paired preferences accumulate ("Response A is better than B")
  - **Constrained PPO** at scale (maximize helpfulness subject to compliance constraints)

---

## Dataset Strategy (Grounding the Whole System)

The advisor raised a critical question: *"Do you have datasets available?"* Here's the full strategy:

| Dataset | What It Provides | Stage |
|---|---|---|
| **RRGen (~570K pairs)** | Google Play review-response pairs for training response generation | Stages 1, 4b |
| **MAALEJ (~4K labeled)** | Multi-label classified app reviews (bug, feature, rating, UX) | Stage 1 |
| **GUZMAN** | Aspect-level annotations for app reviews | Stage 1 |
| **Open-source Android apps with GitHub repos** | Ground-truth mapping: reviews → GitHub issues → code fixes | Stages 3, 4a |
| **Custom gold-standard dataset (to be built)** | Expert-written structured issue specs paired with review clusters — the key novel artifact | Stage 3 evaluation |
| **SWE-bench** | Benchmark for evaluating agentic code resolution | Stage 4a |

The **custom gold-standard dataset** is itself a contribution — no such dataset exists. Building it involves:
1. Selecting 200-300 review clusters from the KG
2. Having 3+ experts independently write structured issue specs for each cluster
3. Measuring inter-annotator agreement
4. Using this as the evaluation benchmark for RQ1

---

## How This Answers the Research Questions

- **RQ1** (Translation accuracy): Compare LLM-generated issue specs against the expert gold-standard using the **5-dimension rubric** scores. Measure inter-annotator agreement to establish human baseline.

- **RQ2** (KG-based triage): Compare the **unified clustering framework's** priority ranking against developer-assigned priorities. Measure triage effort reduction (time to first action).

- **RQ3** (Coupled resolution + response): Compare resolution-aware responses against context-unaware baselines (CoRe, RRGen). The coupling should produce more specific, accurate responses.

- **RQ4** (Dual-objective RLHF): Compare dual-objective (quality + compliance) against single-objective. The rubric-based feedback provides richer signal than binary preferences.

- **RQ5** (End-to-end rate): Track how many raw reviews successfully traverse the full pipeline to produce a validated patch + approved response. The HITL checkpoints should improve this rate over iterations.

---

## Summary of What Changed from Advisor Feedback

| Original Design | Revised Design |
|---|---|
| HITL only at Stage 5 | **HITL at 3 checkpoints** (classification, translation, final review) |
| KG + ad-hoc clustering | **Unified 3-layer framework** (KG → hierarchical clustering → schema mapping) |
| Free-form issue generation | **Taxonomy-grounded** generation (bug reports, user stories, NFRs, usability heuristics, compatibility matrices) |
| Implicit dataset assumptions | **Explicit dataset strategy** with existing datasets + custom gold-standard construction |
| Thumbs up/down evaluation | **5-dimension expert rubric** with inter-annotator agreement and feedback propagation |

The result is a much more rigorous, evaluable, and publishable system — one where every design decision is grounded in literature, validated by experts, and systematically measurable.
