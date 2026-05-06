# 3. Methodology

We describe ReviewAgent, a four-stage pipeline that transforms unstructured app-store reviews into expert-aligned, structured issue specifications and developer-grade responses. The contribution is **methodological**: we identify systematic LLM annotation noise in app-review datasets, design an iterative correction pipeline using a small expert-verified anchor, and demonstrate measurable downstream gains in classification, clustering, and response generation. Figure 1 shows the complete data flow.

## 3.1 Problem Setting and Datasets

Our experiments use the **RRGen** dataset \cite{gao2019rrgen}, comprising 310,031 review-response pairs collected from 58 Android applications. After deduplication and minimum-length filtering, **215,583 unique reviews** form the working corpus. The classification taxonomy is the seven-class scheme established by Maalej et al. \cite{maalej2016}: `bug_report`, `feature_request`, `performance`, `usability`, `compatibility`, `praise`, and `other`.

We seed initial training with 5,008 human-annotated reviews from MAALEJ \cite{maalej2016}, augmented by 500 template-generated synthetic reviews to fill two empty categories (performance: 0 → 70; compatibility: 0 → 50). All evaluation in this paper uses an additional **490-review expert gold standard** annotated by the lead author, drawn stratified from the full corpus (70 reviews per class × 7 classes).

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

The lead author validated 50 clusters (balanced across the five actionable issue types) by reading 5 sample reviews per cluster and rating coherence: **Y** (5/5 reviews share theme), **P** (3-4/5 share theme), **N** (incoherent). Weighted purity (Y=1, P=0.5, N=0) was **0.660** on the initial sample. After lead-author curation of 100 clusters (61 Keep, 6 Rename, 12 Merge, 21 Split), curation-aware purity rose to **0.814**.

## 3.5 Stage 3: Taxonomy-Grounded Issue Specifications

Stage 3 maps each cluster to a structured `IssueSpec` using issue-type-specific templates:

- **bug_report:** Zimmermann template \cite{zimmermann2010} — title, description, severity, affected_component, steps_to_reproduce, expected_behavior, actual_behavior.
- **feature_request:** user-story template — user_story (As-a/I-want/So-that), acceptance_criteria.
- **performance:** ISO/IEC 25010 \cite{iso25010} — nfr_category ∈ {speed, battery, memory, responsiveness, scalability}.
- **usability:** Nielsen heuristics \cite{nielsen1994} — nielsen_heuristic ∈ {visibility, match-real-world, user-control, consistency, error-prevention, recognition-over-recall, flexibility, aesthetic, error-recovery, help-documentation}.
- **compatibility:** device_os_matrix listing affected devices and OS versions.

We compare four conditions: (a) LLM with taxonomy grounding, (b) LLM free-form (no template), (c) raw concatenation of top-3 reviews (no LLM), (d) human-written reference specs (n=20). All LLM conditions used Claude Opus 4.7 via Anthropic's Claude Code subagent infrastructure; outputs were validated for schema adherence post-hoc.

## 3.6 Stage 4b: RAG-Augmented Response Generation

Stage 4b generates developer-style responses to user reviews. We compare four conditions with progressively richer context:

| Condition | Inputs |
|---|---|
| (1) `rrgen_baseline` | Review text only |
| (2) `core_baseline` | Review + general dev-rel system prompt |
| (3) `reviewagent_no_spec` | Review + RAG (3 past responses + 3 similar replies, retrieved from a 15,100-document ChromaDB index) |
| (4) `reviewagent_full` | Review + IssueSpec from Stage 3 + RAG context |

The RAG index is populated from RRGen's developer-response corpus (10,000 past responses + 5,000 similar responses + 60 changelogs + 40 FAQ entries).

Condition (1) responses are LLM-generated with no context. Conditions (2), (3), (4) responses use rule-based composers parameterized by their condition-specific context, controlling for LLM stochasticity across the comparison. Reference responses for automatic-metric evaluation come from RRGen's `original_response` field — i.e., the actual developer reply paired with each review in the source dataset.

## 3.7 Evaluation Protocol

Three evaluation regimes run in concert:

1. **Internal classifier metrics** (Stage 1): own-test-set per-class F1 across V1–V5, plus cross-version agreement on a frozen held-out test set.
2. **Expert gold-standard agreement** (Stage 1 → 4b): the 490-review expert subset evaluates each classifier as an "annotator" against the lead-author labels via Cohen's κ.
3. **Automatic metrics + human evaluation** (Stage 4b): BLEU-1/2/3/4, ROUGE-L, BERTScore F1, distinct-1/2 against RRGen developer replies (automatic), plus 400-row blinded lead-author ratings on quality (1–5), specificity (1–5), and helpfulness (Y/N) (human evaluation).

Statistical tests: paired Wilcoxon signed-rank for response-quality comparisons; agreement reported as Cohen's κ with the standard interpretation thresholds (>0.80 almost-perfect; 0.60–0.80 substantial; 0.40–0.60 moderate).
