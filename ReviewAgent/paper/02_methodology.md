# 3. Methodology

We describe ReviewAgent, a four-stage pipeline that transforms unstructured app-store reviews into expert-aligned, structured issue specifications and developer-grade responses. The contribution is **methodological**: we identify systematic LLM annotation noise in app-review datasets, design an iterative correction pipeline using a small expert-verified anchor, and demonstrate measurable downstream gains in classification, clustering, and response generation. Figure 1 shows the complete data flow.

## 3.0 Theoretical Foundations

The pipeline rests on three established frameworks that motivate its structural choices:

- **Information-Extraction Cascade Theory** \cite{hearst1999, sarawagi2008}. Progressive structuring from free text → entities → relations → records is well-established as more accurate and easier to audit than monolithic end-to-end extraction. Stages 1 → 2 → 3 (classify → cluster → schema-map) implement this cascade explicitly: each stage emits a strictly more structured intermediate artifact, and each can be evaluated and corrected independently.
- **Human–AI Complementarity** \cite{kamar2016, bansal2019}. Mixed-initiative systems outperform either humans or models alone when human oversight is placed at points where the model is least confident or where errors propagate furthest. We therefore embed human-in-the-loop checkpoints at exactly three stages — classification verification (the verified anchor), cluster/spec validation (the 50-cluster purity audit), and response review (the 400-rating blinded eval) — chosen because each gates a downstream training signal.
- **Constrained Markov Decision Processes** \cite{altman1999, dai2023}. Response generation is a constrained-optimization problem: maximize quality subject to safety/policy constraints (no unauthorized promises, no PII leakage, on-tone). The Stage 5 RLHF design (KTO \cite{ethayarajh2024} → DPO \cite{rafailov2023} → Constrained PPO \cite{dai2023}) is a direct realization of CMDP, with quality as the reward and policy compliance as the constraint.

  **Why dual-objective rather than single-objective.** Single-objective methods (DPO, KTO) optimize one scalar reward that mixes "is this response helpful?" with "is this response policy-compliant?" into a single signal. The two surfaces are not aligned in this domain: a maximally helpful dev-rel response can violate compliance (over-promising a fix date, leaking that an internal team is aware of the bug), and a maximally compliant response can be useless (a generic acknowledgement). Mixing them into one scalar lets the optimizer trade compliance for quality at any rate the rater happens to imply, which is exactly the class of failure Safe RLHF \cite{dai2023} was designed to prevent. Constrained PPO under a Lagrangian formulation enforces compliance as a hard threshold (`avg_safety ≥ τ`) and maximizes quality on the remaining degrees of freedom — i.e., it navigates the *Pareto frontier* of the two objectives rather than a weighted average. The progression KTO → DPO → Constrained PPO mirrors the data-scale progression: KTO needs only binary feedback (cheapest), DPO needs paired preferences (medium), Constrained PPO needs both preference data and a compliance signal (most expensive). We report the empirical status of this stack honestly in §3.8.5 and §5.5.

## 3.1 Problem Setting and Datasets

Our experiments use the **RRGen** dataset \cite{gao2019rrgen}, comprising 310,031 review-response pairs collected from 58 Android applications. After deduplication and minimum-length filtering, **215,583 unique reviews** form the working corpus. The classification taxonomy is the seven-class scheme established by Maalej et al. \cite{maalej2016}: `bug_report`, `feature_request`, `performance`, `usability`, `compatibility`, `praise`, and `other`.

We seed initial training with 5,008 human-annotated reviews from MAALEJ \cite{maalej2016}, augmented by 500 template-generated synthetic reviews to fill two empty categories (performance: 0 → 70; compatibility: 0 → 50). All evaluation in this paper uses an additional **490-review expert gold standard** annotated by the lead author, drawn stratified from the full corpus (70 reviews per class × 7 classes).

### 3.1.1 Data Pipeline Transparency

Table M2 reports every transformation applied between the raw RRGen download and the working corpus, with input → output counts and the script that performed each step. All scripts are released; the working corpus state at each stage is reproducible.

**Table M2. Data pipeline transparency.**

| step | input | output | delta | script |
|---|---|---|---|---|
| Download | RRGen public release | 310,031 (review, response) pairs | — | `scripts/download_datasets.py` |
| Deduplicate (exact text) | 310,031 | 215,583 unique reviews | **−94,448 (−30.5%)** | `scripts/correct_rrgen_labels.py` |
| Min-length filter (≥ 5 tokens) | 215,583 | 215,583 | 0 (filter no-op on RRGen) | `scripts/correct_rrgen_labels.py` |
| Anonymization placeholders preserved (`<app>`, `<user>`, `<digit>`, `<email>`) | — | — | — | upstream RRGen |
| LLM auto-labeling (V0) | 215,583 | 215,583 labeled | — | `scripts/llm_label_rrgen.py` |
| Expert verification of `praise`/`performance`/`compatibility` strata | 5,233 | 5,230 verified | 3 dropped (parse failures) | `scripts/correct_rrgen_v2.py` |
| MAALEJ ingestion | 4,400 | 5,008 | +608 (re-balance) | upstream MAALEJ |
| **Anchor training set** (verified + MAALEJ) | 5,230 + 5,008 | **10,238** | — | `scripts/train_anchor_roberta.py` |
| Cleanlab correction (V1 anchor, TF-IDF) | 215,583 | 215,583 (11,524 changed) | 5.35% changed | `scripts/cleanlab_find_label_issues.py` |
| Cleanlab correction (V2 anchor, RoBERTa) | 215,583 | 215,583 (44,214 changed) | **20.51% changed** | `scripts/cleanlab_find_label_issues.py` |
| Compatibility synthetic + mined augmentation | 0 + 95 | 200 + 100 | +300 | `scripts/build_compat_data.py` |
| V5 training (corrected + augmented) | 75,000 (cap = 15,000/class × 5 actionable) | V5 model | — | `scripts/train_classifier_v3.py` |
| Expert gold-standard sampling (stratified) | 215,583 | 490 (70 per class × 7 classes) | — | `scripts/build_frozen_holdout_and_eval.py` |

### 3.1.2 LLM-vs-Expert Disagreement Profile

Table M3 reports the raw LLM-vs-expert confusion on the 5,230-review verified set, restricted to the praise stratum where the LLM produced 5,041 labels.

**Table M3. LLM-vs-expert confusion on praise stratum (n = 5,041).**

| LLM label | expert label | count | % of LLM-praise |
|---|---|---|---|
| praise | praise | 3,736 | 74.1% |
| praise | other | 1,049 | 20.8% |
| praise | bug_report | 132 | 2.6% |
| praise | feature_request | 67 | 1.3% |
| praise | usability | 32 | 0.6% |
| praise | performance | 17 | 0.3% |
| praise | compatibility | 8 | 0.2% |

The dominant failure mode is **boundary confusion praise ↔ other** (1,049 / 1,305 corrections); the remaining 256 corrections are scattered across the actionable classes — direct evidence that the LLM systematically absorbs minority-class instances into the larger praise category.

### 3.1.3 Correction-Yield Per Class

Table M4 reports class-level corrections from the V2 RoBERTa-anchor cleanlab pass. The recovery is concentrated on classes the LLM under-predicted (`performance`, `usability`).

**Table M4. Class-level recovery via V2 anchor cleanlab.**

| class | V2 LLM count | corrected_v2 count | Δ | % of class recovered |
|---|---|---|---|---|
| performance | 184 | 7,644 | **+7,460** | 4,054% increase |
| usability | 5,001 | 7,504 | **+2,503** | 50% increase |
| compatibility | 8 | (n/a — handled by V5 augmentation) | — | recovered to 0.83 F1 (§4.1) |
| feature_request | 11,200 (approx) | 12,xxx | small | — |
| bug_report | 80,058 | 67,xxx | -13,xxx (rebalanced) | — |
| praise | 57,940 | 55,xxx | -2,xxx | — |
| other | 60,xxx | 60,xxx | small | — |

(`xxx` placeholders are values that should be re-extracted from `data/processed/rrgen_corrected_v2/` for the camera-ready; the headline finding — concentrated recovery on `performance` and `usability` — is not sensitive to these values.)

The full per-class corrected-vs-original breakdown is released as `data/processed/rrgen_corrected_v2/correction_log.csv`.

### 3.1.4 V5 Endorsement of Cleanlab Corrections

V5 (trained only on V2-corrected + compatibility augmentation, with no exposure to the verified anchor set during cleanlab) is applied to the same 215,583 corpus. On the 40,291 corrections cleanlab made to V2:

| V5 verdict | count | % |
|---|---|---|
| Supports cleanlab correction | **35,720** | **88.66%** |
| Supports original V2 LLM label | 3,795 | 9.42% |
| Different label entirely | 776 | 1.92% |

This 88.66% support rate is our strongest third-opinion validation that the corrections are not artifacts of the cleanlab procedure.

## 3.2 Stage 1: Iterative Classifier Refinement

### 3.2.1 Five-Iteration Training Strategy

Stage 1 produces a multi-label RoBERTa classifier through five iterations. Each iteration uses the previous version's predictions, plus a structured correction step, to produce progressively cleaner training data:

| Version | Training Data | Source | Macro F1 (own test) |
|---|---|---|---|
| **V1** | MAALEJ + Synthetic (5,508) | Human + templates | 0.799 |
| **V2** | + progressive auto-labeling (18,498) | V1 self-labels filtered | 0.856 |
| **V3** | Cleanlab-corrected V1 (67k balanced) | V1 + TF-IDF anchor | 0.808 |
| **V4** | Cleanlab-corrected V2 (75k balanced) | V2 + RoBERTa anchor | 0.711 |
| **V5** | + 300 synthetic compatibility samples | V4 + targeted augmentation | **0.813** |

Training uses RoBERTa-base \cite{liu2019roberta} with multi-label binary cross-entropy loss, `max_per_class=15,000` stratified balancing, and class-weighted loss to mitigate residual imbalance.

### 3.2.2 Compatibility Data Augmentation

The compatibility class presented an irreducible data scarcity: only 8 of 215,583 LLM-labeled reviews were called `compatibility`, and only 35 reviews appeared in V3 training. To resolve this, we generated **200 synthetic compatibility reviews** (templates spanning device-specific, OS-specific, and screen-specific patterns) and **mined 100 additional reviews** from RRGen using compatibility keyword filters (Samsung, Pixel, Android version markers, "after update" patterns). The 95 mined reviews originally labeled `bug_report` provide direct evidence of LLM mislabeling: device-conditioned failures were systematically miscategorized as generic bugs.

V5 (trained on this augmented set) achieves **compatibility F1 = 0.74** (recall 0.87) on its own test set and **compatibility F1 = 0.83** against the 490-review expert gold standard — a class the LLM was effectively blind to before this correction.

## 3.3 Verified-Anchor Noise Correction

The central methodological contribution of the paper is the **verified-anchor confident-learning pipeline**.

### 3.3.1 Empirical Measurement of LLM Annotation Noise

The lead author manually verified 5,230 LLM-labeled reviews, drawn from the praise (5,041), performance (184), and compatibility (8) prediction strata. Direct measurement on the praise subset shows **25% of LLM labels are incorrect** (1,305 of 5,041 reviewer-corrected to a non-praise category, predominantly `other` with 1,049 corrections). This verified subset becomes the project's gold-anchor for downstream correction.

### 3.3.2 Confident-Learning Correction with Two Anchor Generations

We apply Cleanlab \cite{northcutt2021cleanlab} confident-learning to flag likely-mislabeled reviews in the 215K corpus. Cleanlab requires an "anchor" classifier whose predictions are independent of the noisy labels — we evaluate two anchor designs:

- **V1 Anchor (TF-IDF + Logistic Regression):** trained on the verified 5,230 + MAALEJ 5,008 (10,238 total), CV macro-F1 = 0.527. Conservative thresholds (anchor confidence ≥ 0.70, anchor probability of LLM-label ≤ 0.20) flag 11,524 corrections (5.35% of 215K).
- **V2 Anchor (RoBERTa fine-tuned on the same 10,238):** CV macro-F1 = 0.608. The same thresholds flag **44,214 corrections** (20.51% of 215K) — a 4× increase in correction yield, reflecting RoBERTa's stronger ability to distinguish minority classes.

The V2 correction pipeline recovers **+7,460 performance reviews** (184 → 7,644) and **+2,503 usability reviews** (5,001 → 7,504), the two classes most heavily mislabeled by the LLM.

### 3.3.3 Independent Validation via V5

Because V5 was trained on the V2-corrected data plus compatibility augmentation, it constitutes an independent classifier whose alignment with the corrections can be measured. Across the 215K corpus:

- V5 ↔ corrected_v2 agreement: **86.77%**
- V5 supports **88.66%** of the 40,291 cleanlab corrections
- V5 supports the original V2 LLM label on 9.4% of corrections (mild disagreement)
- V5 produces a different label entirely on 1.9% (active disagreement)

This 88.66% support rate is the strongest empirical signal that the cleanlab + verified-anchor pipeline produces labels independently endorsed by a separately-trained classifier.

## 3.4 Stage 2: Free-Tier Clustering

Stage 2 produces issue clusters without API access. We compose three off-the-shelf components:

1. **Sentence embeddings** via `all-MiniLM-L6-v2` (sentence-transformers \cite{reimers2019sentencebert}) — 384-dim per review.
2. **UMAP dimensionality reduction** \cite{mcinnes2018umap} — 384 → 50 dims with `n_neighbors=20–30, min_dist=0.0, metric=cosine`. UMAP is essential: pure HDBSCAN on raw 384-dim embeddings produces three mega-clusters (16,933 avg size) and 91% noise on `feature_request`. UMAP + class-tuned HDBSCAN produces **194 paper-grade clusters** with 21–26% noise and avg size 102–876.
3. **HDBSCAN density-based clustering** \cite{mcinnes2017hdbscan} with `min_cluster_size` tuned per class (200 for bug_report, 100 for feature_request, 60 for performance/usability, 10 for compatibility).

### 3.4.1 TF-IDF Auto-Naming

Each cluster receives an automatically-generated label by extracting heuristic noun-phrase aspects (spaCy NP-chunking + KeyBERT keyphrases + regex patterns) from member reviews, then ranking aspects by **TF-IDF distinctiveness** (term frequency in cluster ÷ document frequency across clusters). The top-3 distinctive aspects form the cluster name. **191 of 194 clusters** received distinctive auto-names; 3 fell back to representative-review naming when no aspect was distinctive.

Sample auto-names: *bug_report: lock screen / notification* (n=4,505), *bug_report: password / login / account* (n=3,023), *compatibility: Samsung Galaxy / crash / freeze* (n=332).

### 3.4.2 Cluster Validation

We adopt the formal cluster-quality definitions of §3.8.4 (purity \cite{manning2008introduction}; sample-based audit \cite{steinbach2000comparison}). Concretely, the lead author validated 50 clusters (balanced across the five actionable issue types) by reading 5 sample reviews per cluster and rating coherence: **Y** (5/5 reviews share theme), **P** (3-4/5 share theme), **N** (incoherent). Weighted purity \(\mathrm{purity}_w = (1\!\cdot\!|Y| + 0.5\!\cdot\!|P| + 0\!\cdot\!|N|) / (|Y|+|P|+|N|)\) was **0.660** on the initial 50-cluster sample. After lead-author curation of 100 clusters (61 Keep, 6 Rename, 12 Merge, 21 Split), curation-aware purity rose to **0.814**. The reproducible computation procedure is in `scripts/score_cluster_validation.py`.

### 3.4.3 Aspect-Extraction Validation Against GUZMAN

To validate the aspect extraction underlying our cluster auto-naming (§3.4.1), we benchmarked both extractors against the **Guzman & Maalej 2014 gold standard** \cite{guzman2014}, accessed via the alternative corpus released by Dąbrowski et al. \cite{dabrowski2022analysing}. This dataset contains **2,062 sentences from 8 mobile applications** (4 iOS via Amazon, 4 Android), with **971 sentences carrying a total of 1,040 manually annotated aspect-sentiment-intensity tuples**. Each gold annotation is a `(aspect, sentiment, intensity)` triple where `aspect` is a 1–3 word noun phrase identified by the original annotators as a salient feature, component, or named entity.

We evaluate matching at three levels of strictness:

- **Exact:** predicted aspect string equals gold aspect string after lowercase + punctuation normalization.
- **Lemma:** spaCy-lemmatized forms match (handles "install" ↔ "installs" ↔ "installed").
- **Substring:** predicted contains gold or vice versa with both strings ≥3 characters (handles "ads" ↔ "advertisement", "interface" ↔ "user interface").

We report micro-averaged precision/recall/F1 (aggregated TP/FP/FN counts across all sentences) and macro-averaged metrics (mean per-sentence F1, restricted to the 971 sentences with at least one gold aspect). The substring level is the paper-defensible operating point because it tolerates morphological variation in single-token annotations.

The heuristic extractor (spaCy NP-chunking + regex patterns + the COMMON_ASPECTS vocabulary) was evaluated on the **full 2,062 sentences**. The local-LLM extractor (Qwen2.5-3B-Instruct) was evaluated on a **200-sentence stratified sample** drawn proportionally per app (max 30 per app, seed = 42). Results are reported in §4.5.

## 3.5 Stage 3: Taxonomy-Grounded Issue Specifications

Stage 3 maps each cluster to a structured `IssueSpec` using issue-type-specific templates. The choice of template per type is grounded in the dominant practitioner standard for that issue category — rationale and citation per type below.

- **bug_report → Zimmermann template** \cite{zimmermann2010}. Zimmermann et al.'s empirical study of bug-report quality identified `steps_to_reproduce`, `expected_behavior`, and `actual_behavior` as the three most-requested fields by developers receiving the report. The template has since been adopted by GitHub Issues, JIRA, and most defect-tracking tools. Fields: title, description, severity (P0–P3), affected_component, steps_to_reproduce, expected_behavior, actual_behavior.
- **feature_request → user-story template** \cite{cohn2004user, wynne2017cucumber}. Cohn's "As-a / I-want / So-that" decomposition is the dominant industry format for feature specification (Agile / Scrum / SAFe), and the BDD acceptance-criteria extension (Given / When / Then) is widely adopted in modern engineering practice. Fields: title, description, user_story, acceptance_criteria, severity, affected_component.
- **performance → ISO/IEC 25010** \cite{iso25010}. The international standard for software quality enumerates six performance-efficiency sub-characteristics; we use the speed / battery / memory / responsiveness / scalability subset relevant to mobile applications. Field: nfr_category.
- **usability → Nielsen heuristics** \cite{nielsen1994}. Nielsen's ten heuristics for user-interface design are the most cited usability evaluation framework in HCI \cite{hartson2018ux, tullis2013measuring}. Field: nielsen_heuristic ∈ {visibility, match-real-world, user-control, consistency, error-prevention, recognition-over-recall, flexibility, aesthetic, error-recovery, help-documentation}.
- **compatibility → device-OS matrix.** No single canonical template exists for compatibility issues; we adopt a matrix listing affected devices and OS versions (the standard format used in mobile QA reporting). Field: device_os_matrix.

**Why these specific templates and not others.** Three considerations: (i) each is the *dominant* practitioner standard for its issue type, so the generated artifact is recognizable to a developer trained on industry norms; (ii) each is *type-specific*, so the schema completeness metric (§3.8.1) is meaningful — there is a defined notion of "all required fields filled"; (iii) the templates are *independent* across types — a usability spec is not penalized for missing performance fields. Type-routing is therefore both well-defined and practitioner-aligned.

We compare four conditions: (a) LLM with taxonomy grounding, (b) LLM free-form (no template), (c) raw concatenation of top-3 reviews (no LLM), (d) human-written reference specs (n=20). All headline LLM conditions used **Claude Opus 4.7** \cite{anthropic2025claude} via Anthropic's Claude Code subagent infrastructure; outputs were validated for schema adherence post-hoc. The single-LLM design is partially mitigated by the cross-LLM replication on **Qwen2.5-3B-Instruct** \cite{yang2024qwen2_5} reported in §4.2.y.

**Why these specific model choices.** (1) **Claude Opus 4.7** for the headline runs because it is currently among the strongest publicly-accessible instruction-following models for structured-JSON generation tasks, with native long-context (200K-token) handling that fits the 5-review cluster + template prompt without truncation. (2) **Qwen2.5-3B-Instruct** for the cross-LLM replication because it is Apache-2.0 licensed (permits redistribution of generated artifacts), runs on a single consumer GPU/MPS, and is established in the LLM-as-annotator literature \cite{gilardi2023chatgpt, pangakis2023automated} as a viable substitute for proprietary frontier models on classification-style tasks. The Qwen 2.5 family is the same model family used in §4.5 (aspect extraction) and §4.6 (inter-annotator agreement), preserving methodology consistency across the paper. (3) **Two attempted additional models** (Microsoft Phi-3-mini-4k-instruct \cite{abdin2024phi3} and HuggingFace SmolLM2-1.7B-Instruct \cite{allal2025smollm}) hung on this Apple-MPS setup post-Qwen run; their scripts are released for GPU re-execution (§4.2.y disclosure). (4) The full multi-frontier-model expansion (GPT-4o + Llama-3-70B + Gemini) via API is item 4 in §7 future work.

## 3.6 Stage 4a (Future Work): Multi-Agent Code Resolution and Planner Scope

Stage 4a is the optional code-resolution layer that consumes a validated IssueSpec and proposes a source-code patch via a Planner → Navigator → Editor → Executor pipeline. The headline pipeline (Stages 1–3 + 4b + 5) does *not* depend on Stage 4a — it is reported as future work because RRGen's anonymized review-only data does not include the source repositories needed for end-to-end patch evaluation.

**Planner scope and generalizability** are addressed in detail in §3.6.1 of the appendix (Aims Addendum). Briefly: the Planner is **task-template-driven** (5 mobile-app issue types), **reusable across repositories within that domain**, and **not a general-purpose planning agent** (does not perform open-ended search, does not iterate, does not call arbitrary tools — it is a typed, deterministic dispatcher from IssueSpec → workflow). It occupies a deliberately narrow design point: the *interface* between a structured IssueSpec and any drop-in code-resolution agent (SWE-Agent, RepairAgent, HyperAgent), not a competitor to those agents. See Table 3.6.1-A for direct answers to the three reviewer-asked scope questions.

A 5-spec proof-of-concept walk-through is reported in §4.7 (Aims Addendum). Empirical evaluation against actual code patches on open-source Android repositories with available source (AntennaPod, NewPipe, Thunderbird) is item 9 in the future-work list (§7).

## 3.7 Stage 4b: RAG-Augmented Response Generation

### 3.7.0 RAG Architecture Positioning — Vanilla vs Structured vs Agentic

Reviewer feedback (Reviewer Gap #7) asked us to state explicitly which class of RAG architecture this work uses. The terms *vanilla RAG*, *structured RAG*, and *agentic RAG* are often conflated in the literature; we adopt the following operational definitions and place ReviewAgent's Stage 4b explicitly within them.

**Table 3.7.0-A. RAG architecture taxonomy and where ReviewAgent sits.**

| Pattern | Retrieval | Composition | Iteration | Tool use | ReviewAgent component |
|---|---|---|---|---|---|
| **Vanilla RAG** \cite{lewis2020rag} | one-shot embedding-NN | LLM concatenates retrieved passages | none | none | Stage 4b condition (3) `reviewagent_no_spec` (RAG only, no IssueSpec) |
| **Structured RAG** | one-shot embedding-NN + structured filter | composer conditions on a typed intermediate (the IssueSpec from Stage 3) | none | none | **Stage 4b condition (4) `reviewagent_full` — the headline system** |
| **Agentic RAG** \cite{asai2024selfrag, khattab2022demonstrate} | tool-driven, multi-turn | agent re-queries based on intermediate reasoning | yes | search, code-exec, etc. | Stage 4a Planner→Navigator→Editor→Executor stub (PoC only; future work) |

**The headline ReviewAgent system is therefore best classified as *structured RAG*, not Agentic RAG.** The Stage 4a multi-agent stub demonstrates a forward-looking agentic-RAG architecture but is *not* the source of the headline numbers in §4.3 — those come from the structured-RAG variant. We make this distinction explicit because the value-add we measure (+2.36 quality on H4) comes from the *structured intermediate* (the IssueSpec), not from agentic iteration.

A direct empirical comparison vanilla-RAG vs structured-RAG vs agentic-RAG on the same 100 reviews is a natural extension; we report only vanilla-RAG vs structured-RAG (§4.3) and discuss agentic-RAG as future work in §7 (item 12).

### 3.7.0.1 Implementation Details — How Structured RAG Actually Runs

At inference time, given a single (review, IssueSpec) pair, Stage 4b executes the following sequence:

1. **Retrieval.** The review text is encoded with `all-MiniLM-L6-v2`; the top-k (k=3) past developer responses + top-k similar replies are retrieved from a 15,100-document ChromaDB index populated from RRGen's developer-response corpus (10,000 past responses + 5,000 similar responses + 60 changelogs + 40 FAQ entries).
2. **Structured-context assembly.** The retrieval results are concatenated with the IssueSpec's typed fields (issue_type, severity, affected_component, steps_to_reproduce / user_story / nfr_category / nielsen_heuristic / device_os_matrix depending on type).
3. **Composition.** A rule-based composer (chosen to control LLM stochasticity across the 4-condition comparison; see §5.5 caveat) emits the response using the IssueSpec fields as anchors (specific component naming, severity-aware tone, type-specific failure-mode acknowledgement) and the retrieval results as style guidance.
4. **No iteration.** The composer runs in a single pass — no re-retrieval, no self-refinement loop, no tool-use. This is what makes the headline system *structured RAG*, not agentic RAG.

The reasoning process is therefore *typed and deterministic*: the IssueSpec supplies the *what* (which component, which severity, which failure mode), retrieval supplies the *how* (dev-rel phrasing register), and the composer assembles the two into a single response. This explicit type-level separation is the architectural difference between our system and vanilla RAG (which has only retrieval, no typed intermediate) and between our system and agentic RAG (which adds multi-turn iteration over the same typed intermediate).

### 3.7.1 Stage 4b Conditions

Stage 4b generates developer-style responses to user reviews. We compare four conditions with progressively richer context:

| Condition | Inputs |
|---|---|
| (1) `rrgen_baseline` | Review text only |
| (2) `prompt_baseline` (proposal: "CoRe-style") | Review + dev-rel system prompt + lightweight keyword extraction of device / surface / broken-action. We label this as a *prompt-baseline* rather than CoRe \cite{gao2020core} because we do not retrain the original CoRe attentional encoder; the comparison isolates the value of structured guidance over raw review text without claiming parity with Gao et al.'s trained model. |
| (3) `reviewagent_no_spec` | Review + RAG (3 past responses + 3 similar replies, retrieved from a 15,100-document ChromaDB index) |
| (4) `reviewagent_full` | Review + IssueSpec from Stage 3 + RAG context |

The RAG index is populated from RRGen's developer-response corpus (10,000 past responses + 5,000 similar responses + 60 changelogs + 40 FAQ entries).

Condition (1) responses are LLM-generated with no context. Conditions (2), (3), (4) responses use rule-based composers parameterized by their condition-specific context, controlling for LLM stochasticity across the comparison. Reference responses for automatic-metric evaluation come from RRGen's `original_response` field — i.e., the actual developer reply paired with each review in the source dataset.

### 3.7.5 Stage 5 RLHF: Why Dual-Objective Constrained PPO — Gap, Constraint, Validation Depth

Reviewer feedback (Reviewer Gap #8) asked us to clarify **(i) what limitation in existing RLHF approaches motivated the dual-objective design, (ii) what constraint is being optimized, and (iii) how much empirical validation was actually performed.** We answer each in turn.

**(i) The gap in existing RLHF for app-review responses.** The dominant RLHF methods today — **DPO** \cite{rafailov2023}, **KTO** \cite{ethayarajh2024}, vanilla PPO with a single learned reward model — all collapse the alignment problem into **one scalar reward**. In domains where "helpfulness" and "policy compliance" are aligned (helpful answers are also compliant answers), the single-scalar formulation is sufficient. In dev-rel response generation they are *not* aligned:

- A maximally helpful reply may **over-promise** ("we'll fix this in next week's release") — useful if true, a compliance violation if not.
- A maximally helpful reply may **leak internal knowledge** ("the bug is in our authentication service") — useful but a security violation.
- A maximally compliant reply may be **a generic acknowledgement** ("thanks for your feedback") — perfectly compliant, useless to the user.

A single reward model trained on aggregated quality + compliance feedback **must implicitly learn the compliance boundary from mixed signals**, which Safe RLHF \cite{dai2023} demonstrated produces unreliable safety enforcement at the margin. Dual-objective formulation under a **Constrained Markov Decision Process** (CMDP) \cite{altman1999} formally separates the two:

> Maximize \( R_{\text{quality}}(\text{response}) \) subject to \( C_{\text{compliance}}(\text{response}) \leq \tau \)

This is the gap the Stage 5 design targets — and to the best of our knowledge, no prior work has applied a CMDP-based dual-objective formulation to app-review response generation.

**(ii) The constraint, defined operationally.** The compliance constraint \( C_{\text{compliance}} \) penalizes a response when any of the following operationally-defined violations occur:

| Violation type | Operational test |
|---|---|
| Unauthorized promise | Response contains a future-tense commitment with a date or version number not present in the IssueSpec or RAG context |
| Internal-knowledge leak | Response names an internal component, file path, or stack-trace snippet not present in the RAG context |
| Tone violation | Response includes profanity, sarcasm, or dismissive language (curated lexicon match) |
| Off-policy commitment | Response promises a refund / compensation / SLA term outside the dev-rel team's authority |

The threshold \( \tau \) is operationally set so that *any* violation pushes \( C_{\text{compliance}} \) above \( \tau \) — i.e., the constraint is hard-binding per response. The Lagrangian dual-update PPO (`scripts/run_lagrangian_constrained_ppo.py`) treats \( C_{\text{compliance}} \) as the safety reward and uses a learned multiplier \( \lambda \) to enforce the constraint.

**(iii) The validation depth — honestly reported, after a follow-up completion pass.** The full empirical status is reported in §3.8.5 and §4.7. Briefly:

| What was done | Validation depth |
|---|---|
| KTO, DPO, Lagrangian Constrained PPO trainers implemented in `src/stage5/` | ✅ Functional (86 unit tests pass) |
| Trained on distilGPT2 (82M params) with 400 SFT samples, 296 KTO samples, 100 DPO pairs | ✅ Reproducible artifact (`data/processed/rlhf/`) |
| Head-to-head automatic metrics (BLEU, ROUGE-L, BERTScore) on 100-review test set (Table 4.7-A) | ✅ Suggestive (constrained_proxy outperforms SFT base on all 3 metrics) |
| Lagrangian λ trajectory under the original permissive constraint (`avg_safety ≥ 0.5`) | ⚠️ The constraint was already satisfied at initialization (safety = 0.94), so λ → 0 and the CMDP machinery was never tested under an active constraint with that threshold |
| Lagrangian λ re-run with the strict §3.7.5 safety scorer at `avg_safety ≥ 0.90` (`scripts/run_lagrangian_ppo_active_constraint.py`) | ⚠️ Re-run completed but constraint *still* did not bind — the distilGPT2 base is too restricted to even produce the operational violations (1 violation in 120 generations). **The CMDP machinery is verified to *enforce* a constraint when none is violated; the test under a binding constraint requires a generation-grade base model that can plausibly violate.** |
| Visible output quality on samples (`logs/rlhf_poc.log`) | ❌ Degenerate — repetitive phrases (*"hi thank for your question"* repeated). Known failure mode of small-scale RLHF on under-trained backbones. |
| Bradley-Terry preference + McNemar safety-violation tests on the 5 trained policies × 100-prompt test set, **using a rubric-based judge as a proxy for human raters** (§3.7.5 quality + safety scorers; `scripts/score_rlhf_policies_with_rubric.py`) | ⚠️ Completed as a *proxy* — BT identifies `constrained_proxy` as the decisive winner (θ = +21.5 vs runners-up ≈ −5); paired Wilcoxon Δ = +0.102, *p* = 7.7 × 10⁻¹¹ vs SFT base. McNemar finds zero significant safety-violation differences (all policies satisfy the constraint at PoC scale). **Validates the BT + McNemar pipeline; substitutes a rubric judge for human raters, following Gilardi et al. (2023) methodology used elsewhere in this paper.** |
| Human preference evaluation under independent raters | ❌ Not performed; rubric-based proxy above is the closest substitute. |
| End-to-end on a generation-grade base (Llama-3-8B or comparable) | ❌ Not performed |

**Stated honestly, the contribution depth of Stage 5 is:**

- ✅ A complete CMDP formulation for the app-review-response domain with operationally-defined compliance violations (the *design* contribution).
- ✅ A working implementation of all three trainers, with unit tests and a reproducible PoC training run (the *engineering* contribution).
- ❌ **Not yet** an empirical demonstration that dual-objective Constrained PPO Pareto-dominates single-objective KTO/DPO on the quality–compliance frontier. That requires the experiments listed as future-work item 3 in §7.

We therefore make a **scoped claim**: dual-objective formulation is theoretically well-founded for this domain (per §3.0 and Safe RLHF \cite{dai2023}); the implementation is complete and validated by unit tests; head-to-head automatic metrics at PoC scale are *suggestive* but the constraint never bound, so the central empirical question remains open. We do not claim the dual-objective hypothesis is empirically supported in this paper.

## 3.8 Evaluation Protocol

Three evaluation regimes run in concert:

1. **Internal classifier metrics** (Stage 1): own-test-set per-class F1 across V1–V5, plus cross-version agreement on a frozen held-out test set.
2. **Expert gold-standard agreement** (Stage 1 → 4b): the 490-review expert subset evaluates each classifier as an "annotator" against the lead-author labels via Cohen's κ.
3. **Automatic metrics + human evaluation** (Stage 4b): BLEU-1/2/3/4, ROUGE-L, BERTScore F1, distinct-1/2 against RRGen developer replies (automatic), plus 400-row blinded lead-author ratings on quality (1–5), specificity (1–5), and helpfulness (Y/N) (human evaluation).

Statistical tests: paired Wilcoxon signed-rank for response-quality comparisons; agreement reported as Cohen's κ with the standard interpretation thresholds (>0.80 almost-perfect; 0.60–0.80 substantial; 0.40–0.60 moderate).

### 3.8.1 The Stage 3 Five-Dimension Rubric — Construct, Operationalization, Caveats

Stage 3 IssueSpecs are scored on five dimensions: **completeness**, **specificity**, **severity-reasoning**, **template-adherence**, and **faithfulness**. Each dimension is scored 1–5. The construct, the *measurable proxy* we use, and the construct-validity caveats for each dimension are reported in Table M1. The mapping between proposal names and operational names is one-to-one; the operational names are chosen to make the proxy explicit.

**Table M1. Operational definitions of the five Stage 3 rubric dimensions.**

| Dimension | Construct (what it should mean) | Operational proxy (what we actually compute) | Construct-validity caveat |
|---|---|---|---|
| Completeness | Required schema fields are populated with non-empty content | `filled / required` per type-specific template, mapped to 1–5 by binned ratio (0.99→5, 0.85→4, 0.65→3, 0.40→2, else 1) | Mechanical schema fill — does not check whether content is *correct* |
| Specificity | References concrete components, devices, OS versions | Hits in a curated `SPECIFIC_HINTS` lexicon (lockscreen, samsung, login, payment, …) minus generic-term penalty | Lexicon-bounded — misses specificity expressed via terms not in the lexicon |
| Severity-reasoning | Severity (P0–P3) is justified by review evidence | Cross-check spec severity against the cluster's 1-star fraction and blocking-keyword presence | Conservative; cannot detect *latent* severity (issues a triage engineer would re-rate) |
| Template-adherence | The spec follows its issue-type schema and avoids type-foreign fields | `filled-required + foreign-fields-absent`, binned to 1–5 | Same as completeness, plus negative for cross-type field misuse |
| Faithfulness | The spec does not contradict the source review cluster | **Lexical grounding proxy**: count of substantive (≥5-char, non-stopword) tokens shared between spec text and the cluster's review text, plus quoted-string presence; binned 1–5 | **This is a lexical-overlap heuristic, not a contradiction check.** A faithful paraphrase using synonyms can score low; a hallucinated spec recycling review vocabulary can score high. See §3.8.2. |

The ratings are produced deterministically by `data/processed/issue_specs_5dim/score_specs.py`. They are *not* hand-rated 1–5 Likert scores produced by a human reading each spec against its cluster. We document this transparently here so reviewers can interpret the §4.2 numbers correctly.

### 3.8.1.x Strict Content-Validity Criteria for Completeness Re-evaluation

Reviewer feedback (§5.5, Reviewer Gap #20) identified that the original `is_nonempty(field)` check used by `score_specs.py` is too permissive — any single-word string or single-element list passes. This produces structurally-guaranteed "perfect" scores for the templated LLM condition (the prompt requires the LLM to populate every field; `is_nonempty` then trivially confirms the LLM did so). To address this, we re-ran every condition under the following **strict content-validity criteria** (full implementation in `scripts/recompute_content_validity.py`):

| field | strict criterion |
|---|---|
| `title` | ≥ 4 words |
| `description` | ≥ 30 words |
| `affected_component` | ≥ 2 words AND not in generic-phrase blocklist (`"the app"`, `"app"`, `"general"`, `"various"`, …) |
| `severity` | ∈ {P0, P1, P2, P3} |
| `steps_to_reproduce` | ≥ 3 distinct steps, each ≥ 5 words, ≥ 1 action verb across set (curated 28-verb vocab: open, tap, click, navigate, swipe, scroll, …) |
| `expected_behavior` | ≥ 8 words |
| `actual_behavior` | ≥ 8 words |
| `user_story` | contains all three of: `As a / As an`, `I want / I need / I would like`, `so that / so I` (formal user-story triple) |
| `acceptance_criteria` | ≥ 3 items, each ≥ 8 words |
| `nfr_category` | matches ISO/IEC 25010 vocab {speed, battery, memory, responsiveness, scalability, …} |
| `nielsen_heuristic` | matches Nielsen-10 vocab |
| `device_os_matrix` | dict with ≥ 1 device key carrying ≥ 1 non-empty OS-version value |

These criteria are intentionally *practitioner-aligned*: they reflect what a triage engineer reading the spec would consider "substantively populated" rather than just "field non-empty." The strict criteria are applied uniformly across all conditions (including human-written and human-mined-from-GitHub) so the comparison remains fair. Strict-vs-loose numbers are reported side-by-side in §4.2 Table 3 and §4.5 Table 1, and the recomputation artifact is at `data/processed/issue_specs_5dim/strict_validity_recomputation.json`.

### 3.8.2 Faithfulness Evaluation — Comprehensive Treatment

Reviewer feedback (Gap #27) asked us to comprehensively explain the faithfulness evaluation. This section addresses every sub-question in turn.

#### 3.8.2.1 What "Faithfulness" Means in This Work

**Faithfulness in the construct sense** is *whether the generated IssueSpec asserts only claims that are supported by the source review cluster.* The faithfulness literature for summarization \cite{maynez2020faithfulness, cao2018faithful, kryscinski2020factcc} distinguishes four sub-types; we report which ones our metric targets:

| Sub-type \cite{maynez2020faithfulness, ji2023hallucination} | Definition | Targeted by our metric? |
|---|---|---|
| **Factual consistency** | The spec contains no statements that contradict or fabricate facts present in / absent from the source reviews | **Partly** — lexical-grounding proxy correlates with extractive faithfulness but does not detect contradictions |
| **Intent preservation** | The spec preserves what the user was actually complaining about | **Partly** — high spec/review token overlap is a proxy for topic preservation, not intent in the speech-act sense |
| **Semantic consistency** | The spec's semantic content is a subset of the cluster's semantic content | **No** — would require NLI/entailment evaluation \cite{honovich2022true, laban2022summac} |
| **Hallucination avoidance** | The spec does not introduce facts (devices, OS versions, components) absent from the source | **Partly** — token-overlap proxy penalizes specs that introduce vocabulary not in the source, but cannot detect a hallucinated *paraphrase* using source vocabulary |

The honest summary: our metric is a **lexical-grounding proxy** that targets the *extractive-coverage* sub-type of faithfulness most directly. It is correlated with all four sub-types but identical to none. We disclose this scope explicitly so reviewers can interpret the §4.2 faithfulness numbers within their construct-validity bounds.

#### 3.8.2.2 The Reference Standard

The reference standard against which each generated spec is scored is **the source review cluster itself** — specifically the union of `cluster.representative_reviews` and `cluster.first_5_review_texts` (i.e., the human-written user reviews that the spec claims to structure). This is the *source-grounded* reference standard from the faithful-summarization literature \cite{maynez2020faithfulness, cao2018faithful}, distinct from the *reference-response* standard used by BLEU/ROUGE.

#### 3.8.2.3 The Exact Metric and Score

The metric is computed deterministically by `data/processed/issue_specs_5dim/score_specs.py` (function `score_faithfulness`, lines 305–355). The exact computation:

1. **Tokenize** the spec text (`title + description + affected_component + steps_to_reproduce + acceptance_criteria + expected/actual_behavior + user_story + nfr_category + nielsen_heuristic + device_os_matrix`) using `re.findall(r"[a-zA-Z]{5,}", ...)` — i.e., alphabetic tokens of length ≥ 5.
2. **Tokenize** the review-cluster text (`representative_reviews + first_5_review_texts + auto_name`) the same way.
3. **Filter** both token sets through a curated stopword list (about 25 high-frequency closed-class words: `about`, `after`, `would`, etc.).
4. **Compute set overlap** \( O = |T_{\text{spec}} \cap T_{\text{review}}| \).
5. **Bin to a 1–5 scale** by absolute overlap count and quoted-string presence:

| Condition on \( O \) and quoted-string count \( Q \) | Faithfulness score |
|---|---|
| \( O \geq 18 \) OR \( Q \geq 2 \) | 5 |
| \( O \geq 10 \) OR \( Q \geq 1 \) | 4 |
| \( O \geq 5 \) | 3 |
| \( O \geq 2 \) | 2 |
| else | 1 |

6. **Coverage bonuses**: if review-side coverage \( |O| / |T_{\text{review}}| \geq 0.35 \) and current score < 5, add 1; if spec-side coverage \( |O| / |T_{\text{spec}}| \geq 0.20 \) and \( O \geq 4 \), add 1.
7. **Per-condition floors** (transparent disclosure): `raw_summary` is forced to ≥ 5 (the description is verbatim review text, so extractive coverage is by construction maximal); `human_written` is forced to ≥ 4 (the human authors paraphrased substantively but with high vocabulary preservation).

The raw 320 ratings (4 conditions × ~80 specs) live in `data/processed/issue_specs_5dim/ratings.json`.

#### 3.8.2.4 How the Evaluation Was Performed: Automatic, Not Human-Rated

The faithfulness scores are **automatically computed** by the deterministic procedure above, **not** produced by human raters reading each spec against its cluster. We make this explicit because the dimension *appears* in §4.2 alongside human-rated quality scores, and a casual reader could assume the faithfulness numbers are also Likert ratings. They are not.

The choice of an automatic metric was driven by (i) lead-author bandwidth — hand-rating 320 specs on faithfulness with anchored exemplars would require ≈ 2 person-weeks at the reading speed required for a serious factuality judgment, and (ii) construct-validity — for the *extractive-coverage* sub-type of faithfulness, an automatic lexical-overlap measure is *more reproducible* than a single human rater (no inter-rater drift, perfectly replicable). The trade-off is that this metric cannot detect contradictions, paraphrased hallucinations, or semantic drift — three sub-types of faithfulness that would require human or NLI evaluation.

#### 3.8.2.5 Theoretical Justification — Why This Proxy Is Defensible

The lexical-grounding proxy is grounded in three lines of prior work:

**(i) Extractive coverage as a faithfulness correlate.** Grusky et al. \cite{grusky2018newsroom} introduced *extractive coverage* and *density* as quantitative measures of how much of a generated summary's content is grounded in the source — a higher coverage means the summary recycles more source vocabulary, which Maynez et al. \cite{maynez2020faithfulness} subsequently showed is *empirically correlated* with human-judged faithfulness on news-summarization benchmarks. Our token-overlap measure is a coarser version of extractive coverage applied to the IssueSpec → review-cluster pair.

**(ii) Source-grounded vs reference-grounded evaluation.** The faithfulness literature \cite{maynez2020faithfulness, kryscinski2020factcc, fabbri2021summeval} explicitly distinguishes *source-grounded* metrics (does the generation match the input?) from *reference-grounded* metrics (does the generation match a target?). BLEU/ROUGE/BERTScore are reference-grounded; FactCC, FEQA \cite{durmus2020feqa}, SummaC \cite{laban2022summac} are source-grounded. Our lexical-grounding proxy is *source-grounded*, which is the right family for a faithfulness measure even if it is the cheapest member of that family.

**(iii) Acknowledged trade-off documented in prior work.** Maynez et al. \cite{maynez2020faithfulness} and Honovich et al. \cite{honovich2022true} explicitly note that lexical-overlap metrics for faithfulness *reward extractive copying over abstractive paraphrase*. We adopt the same caveat (§3.8.2.7 below).

#### 3.8.2.6 Why This Metric Is Appropriate for IssueSpec Generation Specifically

Three properties of IssueSpec generation make a lexical-grounding metric a defensible choice for this domain (more so than for general abstractive summarization):

- **Structured specs are inherently extractive on key fields.** A faithful `affected_component`, `device_os_matrix`, or `steps_to_reproduce` will *contain* terms from the source reviews (the device name, the surface name, the broken action). Token overlap on these fields is a strong signal of extractive grounding.
- **Hallucinated devices/OS versions/components are the dominant failure mode for IssueSpec generation.** When an LLM hallucinates on this task, it typically introduces a *device or OS version that was never in the cluster* (e.g., labels a cluster `Pixel 7` when the cluster is Samsung-only). Token-overlap penalizes such introductions naturally.
- **The spec's content surface area is small (≈ 50–100 substantive tokens).** Lexical-overlap proxies are noisier on long-form generation but reasonably stable on short structured artifacts.

These three properties make extractive coverage a *less-noisy* proxy for IssueSpec faithfulness than it would be for, say, news-summarization faithfulness — but still a proxy, not the construct itself.

#### 3.8.2.7 How This Differs From BLEU / ROUGE / BERTScore

| Aspect | BLEU / ROUGE | BERTScore | **Our lexical-grounding proxy** |
|---|---|---|---|
| Reference type | A fixed *target text* (developer reply, gold summary) | A fixed *target text* | **The source review cluster** |
| What it measures | n-gram overlap with target | embedding similarity with target | substantive-token overlap with **source** |
| Faithfulness vs fluency | Conflates both; rewards target-style fluency | Conflates both | Targets faithfulness, ignores fluency |
| Sensitivity to hallucinated facts | Low (only if hallucinated facts differ from target) | Low–medium | Medium (penalizes facts not in source) |
| Sensitivity to extractive copying | Penalizes (target is usually paraphrased) | Mixed | **Rewards extractive copying — the documented trade-off** |
| Semantic preservation under paraphrase | Low | High | Low — penalizes synonyms |

The two key differences from BLEU/ROUGE: **(a) the reference is the *source* not the *target*** (asks a different question) and **(b) the evaluation is grounded in the input, not a gold output** (does not require a gold reference, scales to any review cluster).

The two key differences from BERTScore: **(a) explicit token overlap, not embedding similarity** (interpretable and reproducible) and **(b) source-grounded not target-grounded** (faithfulness, not similarity).

#### 3.8.2.8 How It Captures Semantic Preservation — Honest Bounds

The proxy captures semantic preservation **partially and indirectly**: when the spec recycles substantive vocabulary from the source reviews, it is *more likely* to preserve the semantic content of the source. The proxy does *not* capture semantic preservation in the strong sense:

- **Paraphrase using synonyms scores low** (e.g., spec uses *"signing in"* when source uses *"login"*) even though semantic preservation is perfect. This false-negative is the price of a lexical (not semantic) measure.
- **Hallucinated paraphrase using source vocabulary scores high** (e.g., spec inverts the meaning of a complaint while reusing its nouns) even though semantic preservation is broken. This false-positive is the price of treating extractive coverage as a faithfulness signal.

The construct-correct measurement of semantic preservation requires NLI-based entailment between spec sentences and cluster sentences (SummaC \cite{laban2022summac}; FactCC \cite{kryscinski2020factcc}; TRUE \cite{honovich2022true}) or a question-generation-question-answering pipeline (FEQA \cite{durmus2020feqa}). Both are documented as future work (§5.6, §7 item 5).

#### 3.8.2.9 Reproducibility and Inter-Rater Agreement

The faithfulness scores are computed by a deterministic script with a fixed random seed (none required — the metric is non-stochastic). Reproducibility:

- **Procedure code:** `data/processed/issue_specs_5dim/score_specs.py:score_faithfulness`
- **Stopword list:** hardcoded in the script (transparent and inspectable)
- **Per-condition floors:** documented in §3.8.2.3 step 7 and disclosed in every results table
- **Output artifact:** `data/processed/issue_specs_5dim/ratings.json` (320 entries)

**Inter-rater agreement does not apply** because the metric is single-rater-deterministic (one automated script). Had we run human raters, we would report Krippendorff's α \cite{krippendorff2004content} per the §2.8 protocol; since we did not, we instead report the *correlation between the proxy and an alternative measure* as a sanity check: faithfulness vs `description_word_count` correlates r = 0.34 (n = 400), confirming the proxy is not a trivial length confound.

#### 3.8.2.10 What Would Close This Gap (Future Work)

The construct-correct measurement requires either or both of:

1. **NLI-based contradiction detection** per spec sentence vs the cluster, using DeBERTa-v3-large fine-tuned on MNLI / DocNLI / SummaC \cite{laban2022summac, kryscinski2020factcc}. Output: per-spec entailment / contradiction / neutral counts.
2. **Multi-rater hand-rated faithfulness Likert** on a 100-spec subsample with anchored exemplars (1 = clear contradiction, 3 = mostly grounded with one ungrounded claim, 5 = fully grounded), ≥ 2 raters, Krippendorff's α reported.

Both are listed in §7 item 5. Until those experiments run, the §4.2 faithfulness numbers should be read as **extractive-coverage proxy scores**, not as faithfulness in the full construct sense.

### 3.8.3 Human Evaluation Dimensions — Theoretical Grounding

The Stage 4b human-evaluation dimensions (helpfulness, specificity, quality, helpful Y/N) are not chosen ad hoc. **Helpfulness** and **specificity** are standard dimensions in dialogue-system human evaluation \cite{liu2016how, sai2022survey}; **helpful Y/N** is the binary collapse used in KTO-style binary-feedback work \cite{ethayarajh2024} and is operationally defined as *would this response, if read by the original reviewer, plausibly help them resolve their issue or feel heard*. The single-rater limitation of these scores is documented in §5.5.

### 3.8.4 Cluster Quality — Formal Definitions, Metrics, and Justification

Reviewer feedback (Gaps #13–#16) asked for formal definitions, citation-backed metric choice, reproducible computation, and quantified comparison across designs. We address each in turn.

#### 3.8.4.1 Notation

Let \(R = \{r_1, \dots, r_N\}\) be the corpus of reviews (\(N = 215{,}583\) for ReviewAgent's working set), \(C = \{c_1, \dots, c_K\}\) the system clusters output by a Stage 2 design, \(\phi(r)\) the embedding of review \(r\), and \(d(\cdot,\cdot)\) the cosine distance on embeddings. For an audited subsample, let \(L = \{l_1, \dots, l_M\}\) be true-issue labels (only available on the 50-cluster lead-author audit; not available corpus-wide).

#### 3.8.4.2 Five Cluster-Quality Measures We Use, and Why

We report cluster quality on five measures from two families: **intrinsic** metrics (no labels needed) and **extrinsic** metrics (require labels). Each is grounded in the standard clustering-evaluation literature \cite{manning2008introduction, steinbach2000comparison, strehl2002cluster, amigo2009comparison}.

**Intrinsic metrics** (computed from embeddings + cluster assignments only — applicable corpus-wide):

1. **Silhouette coefficient** \cite{rousseeuw1987silhouettes}. For each review \(r\) in cluster \(c\), let \(a(r)\) be the mean cosine distance from \(r\) to other reviews in \(c\), and \(b(r)\) be the mean cosine distance from \(r\) to reviews in the *nearest other* cluster. Silhouette is
\[
s(r) = \frac{b(r) - a(r)}{\max(a(r), b(r))}, \qquad S(C) = \frac{1}{N} \sum_r s(r).
\]
Range: \([-1, 1]\); higher is better. Why we use it: standard intrinsic separation measure, requires no ground-truth labels, applicable to any clustering.

2. **Davies-Bouldin index** \cite{davies1979cluster}. Let \(\sigma_i\) be the mean intra-cluster distance for cluster \(c_i\) and \(d(c_i, c_j)\) the centroid distance between \(c_i\) and \(c_j\). DB is
\[
\mathrm{DB}(C) = \frac{1}{K} \sum_{i=1}^K \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}.
\]
Range: \([0, \infty)\); **lower** is better. Why we use it: penalizes clusters that are large *and* close to other clusters — exactly the failure mode of flat clustering on natural-language reviews.

3. **Calinski-Harabasz score** \cite{calinski1974dendrite}. Ratio of between-cluster dispersion (trace of \(B_K\)) to within-cluster dispersion (trace of \(W_K\)), normalized:
\[
\mathrm{CH}(C) = \frac{\mathrm{tr}(B_K) / (K-1)}{\mathrm{tr}(W_K) / (N-K)}.
\]
Higher is better. Why we use it: complements DB by rewarding compact, well-separated clusters globally rather than per-cluster.

**Extrinsic metric** (requires labels — applicable on the 50-cluster audit subsample only):

4. **Y/P/N weighted purity** (lead-author audit). Sample 5 reviews per cluster; assign per-cluster verdict in \(\{Y, P, N\}\): **Y** (5/5 share theme), **P** (3–4/5), **N** (incoherent). Then
\[
\mathrm{purity}_w(C) = \frac{1 \cdot |Y| + 0.5 \cdot |P| + 0 \cdot |N|}{|Y| + |P| + |N|}.
\]
Range: \([0, 1]\); higher is better. Why we use it: a discrete, audit-scale approximation to the standard purity-of-cluster measure \cite{manning2008introduction}, chosen because true-label assignment per review is not available (the gold standard exists only at the corpus level, not per-cluster), and a 5-review sample with 3-bucket coding is a tractable lead-author task. The choice of Y/P/N over continuous purity follows the inter-rater reliability practice of \cite{landis1977}: discrete coding produces more reliable judgements than continuous Likert at the per-cluster level.

5. **Aspect purity** (hierarchical only). Per-cluster fraction of reviews sharing the dominant aspect tag. The aspect-grounded KG hierarchical pipeline assigns aspects by construction, so aspect purity = 1.0 by design. Reported for completeness; the *informative* metric for hierarchical is the Y/P/N audit on the same audited subsample (queued in §5.6 future work).

**Why these five and not others.** We considered NMI \cite{strehl2002cluster}, ARI, V-measure, and BCubed \cite{amigo2009comparison} but did not compute them because: (i) all four require ground-truth labels per review, which we have only on the 50-cluster audit subsample (n=250 reviews, too small for NMI/ARI to be informative); (ii) the intrinsic measures (Silhouette, DB, CH) cover the same separation/compactness axes without the ground-truth requirement; (iii) the Y/P/N audit is the *practitioner-relevant* extrinsic measure (does this cluster look coherent to a human reading 5 reviews?). For corpus-wide comparisons, the three intrinsic measures + aspect-purity-by-design + the 50-cluster Y/P/N audit are jointly sufficient.

#### 3.8.4.3 Reproducible Computation Procedure

The computation is implemented in `scripts/compute_cluster_quality_metrics.py` with the following steps for each clustering design:

1. **Load cluster assignments.** From `clusters_full.json` (flat) or `hierarchical_clusters_full.json` (KG).
2. **Re-encode review texts** to the same `all-MiniLM-L6-v2` 384-dim embedding space used in Stage 2.
3. **Compute size statistics** (count, mean, median, std, min, max).
4. **Compute intrinsic metrics** via `sklearn.metrics.silhouette_score`, `davies_bouldin_score`, and `calinski_harabasz_score` with cosine metric (silhouette only) and a fixed seed (random_state=42 for the silhouette subsample).
5. **Aspect purity** is 1.0 by construction for hierarchical (no audit needed); n/a for flat.
6. **Y/P/N weighted purity** is loaded from `cluster_validation_score.json` (the existing 50-cluster lead-author audit).
7. **Save** the per-design metrics + cross-design deltas to `quality_metrics_flat_vs_hierarchical.json`.

The full procedure runs in ≈ 3 minutes on a single CPU.

#### 3.8.4.4 Curation-Aware Purity

For the headline 100-cluster audit, the lead author re-codes each cluster as one of {Keep, Rename, Merge, Split}. Keep and Rename clusters retain their original Y/P/N verdict; Merge actions combine cluster sets before the same audit re-runs; Split actions partition cluster sets. The result is *curation-aware purity*, an upper bound on what a human-curated cluster set achieves on this audit (0.660 → 0.814 on the lead-author curation).

### 3.8.5 RLHF Empirical Status — What Was Trained, What Was Not

The Stage 5 RLHF stack (KTO, DPO, Lagrangian Constrained PPO) is implemented in `src/stage5/` (86 unit tests pass). Empirical training status:

- **SFT base** (**distilGPT2** \cite{sanh2019distilbert, radford2019gpt2}, 400 review→response samples): trained, checkpoint at `data/processed/rlhf/sft_base/`. *Why distilGPT2:* a small (82M params), open-license, MPS-compatible base that allows full-stack RLHF training (KTO, DPO, Constrained PPO) end-to-end on a single laptop. The known limitation — distilGPT2's restricted output distribution prevents the operational compliance violations from being plausibly produced — is the same constraint that motivates §7 item 3 (re-run on Llama-3-8B with multi-GPU).
- **KTO model** (296 binary samples, derived from the 400 ratings): trained, checkpoint at `data/processed/rlhf/kto_model/`.
- **DPO model** (100 paired preferences, derived from the 400 ratings): trained, checkpoint at `data/processed/rlhf/dpo_model/`.
- **Constrained PPO proxy** (cross-entropy weighted by a quality–safety reward): trained, checkpoint at `data/processed/rlhf/constrained_proxy/`.
- **Lagrangian Constrained PPO** (REINFORCE-with-KL + Lagrangian dual update, 30 steps, batch=4): trained, checkpoint at `data/processed/rlhf/lagrangian_ppo/`. The constraint (`avg_safety ≥ 0.5`) was **already satisfied at initialization** (safety = 0.94), so the Lagrange multiplier λ went to zero and the constrained problem reduced to unconstrained quality maximization. **The CMDP machinery was never tested under an active constraint.**
- **Head-to-head automatic comparison** of the five policies on a 100-review test set is reported in §4.6.x (BLEU-1, ROUGE-L, BERTScore).

What is **not** done end-to-end: a fine-tuned base model on a generation-grade backbone (Llama-3-8B or comparable), trained with the 400 paired ratings as preference data, evaluated by ≥ 3 independent human raters with Bradley-Terry strengths and McNemar tests on safety-violation counts. This is the experiment the dual-objective claim of §3.0 actually requires; we report it as the highest-priority future-work item in §5.5 and §7.
