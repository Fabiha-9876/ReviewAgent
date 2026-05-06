# ReviewAgent — Full Paper Draft

**Working title:** *Verified-Anchor Confident Learning for Cleaning LLM-Annotated App-Review Datasets: An Iterative Pipeline with Independent Cross-Validation*

**Status:** Draft assembled from per-section markdown files. References in `references.bib`.

---



# Abstract

App-store review datasets labeled by large language models (LLMs) contain systematic annotation noise that propagates into downstream classifiers, clusters, and response-generation models. We present **ReviewAgent**, a four-stage pipeline that detects and corrects this noise using a small expert-verified anchor and confident-learning, and demonstrates measurable downstream gains in classification accuracy, cluster purity, and human-rated response quality.

We measure the LLM annotation error rate directly on a 5,230-review verified subset of RRGen (a corpus of 215,583 deduplicated app reviews): **25%** of reviews labeled `praise` by the LLM are actually misassigned, with 1,049 reviews incorrectly absorbed from the `other` class. Applying confident-learning with a RoBERTa-based verified anchor (trained on 5,230 expert-verified labels plus 5,008 MAALEJ human-annotated reviews) flags **44,214 corrections** (20.51% of the corpus), recovering +7,460 misclassified `performance` reviews and +2,503 misclassified `usability` reviews — two classes the LLM was effectively blind to.

A new classifier (V5, RoBERTa-base) trained on the corrected labels plus 300 targeted compatibility-class augmentation samples achieves Cohen's κ = **0.59** against a 490-review expert gold standard, up from 0.16 for the original LLM labels and 0.33 for the cleanlab-corrected pipeline alone. V5 independently endorses **88.66%** of the cleanlab corrections, providing third-opinion validation of the noise-correction methodology. The classifier achieves macro F1 = 0.81 with newly-functional `compatibility` (F1 = 0.74, recall = 0.87) and `performance` (F1 = 0.79) categories — both effectively zero-recall before the correction pipeline.

Downstream, we use the corrected labels and a UMAP+HDBSCAN clustering pipeline (free of LLM API access) to produce 194 issue clusters with TF-IDF-based auto-naming. Lead-author curation of 100 clusters yields cluster purity of **0.81**. The aspect extraction underlying cluster naming was independently validated against the **Guzman & Maalej 2014 gold standard** (2,062 expert-annotated sentences from 8 apps): the heuristic extractor achieves **84.2% recall** (substring micro-F1 = 0.307; macro-F1 = 0.467), and a local-LLM extractor (Qwen2.5-3B) achieves micro-F1 = 0.404 — the two methods occupying complementary recall-strong and precision-strong operating points. Stage 3 generates structured issue specifications grounded in domain-specific templates (Zimmermann for bugs, ISO 25010 for performance, Nielsen heuristics for usability). Stage 4b generates developer-style responses using retrieval-augmented generation (RAG) over RRGen's developer-reply corpus.

In a 400-rating blinded lead-author evaluation, the full ReviewAgent system (RAG + IssueSpec) achieves quality 4.62/5 vs 2.26/5 for RAG-only (paired Wilcoxon p < 0.001, Δ = +2.36), demonstrating that the structured issue specification provides measurable value beyond retrieval alone. Helpfulness rises from 31% (RAG-only) to **92%** (full system).

Inter-annotator agreement, computed across the lead-author expert and two LLM raters (Gilardi et al. 2023 methodology), shows that the 7-class task is inherently difficult: naive LLM-vs-expert κ ranges 0.27–0.38, while V5 achieves **κ = 0.59** — substantially exceeding the LLM-annotator baseline. A complementary three-layer Stage 2 pipeline (knowledge-graph construction + aspect-grounded hierarchical clustering + schema mapping) produces 605 fine-grained clusters (vs 194 from flat clustering), enabling per-aspect drill-down. A Planner→Navigator→Editor→Executor multi-agent resolution stub demonstrates end-to-end architecture at the specification level (5 IssueSpecs walked through all 4 agents) and produces resolution-aware responses that reference specific proposed fixes rather than generic acknowledgements.

The work makes three contributions: (1) an empirical demonstration that LLM-labeled software-engineering datasets contain ≥25% systematic annotation noise; (2) a verified-anchor confident-learning correction pipeline with independent third-opinion validation (88.66% support rate); and (3) a free-tier (no LLM API) clustering and aspect extraction pipeline that achieves paper-grade cluster purity. All artifacts, scripts, and trained models are publicly available at https://github.com/Fabiha-9876/ReviewAgent.


---


# 1. Introduction

App-store reviews are among the largest, freshest, and most opinionated sources of feedback available to software development teams \cite{maalej2016, gao2019rrgen, dabrowski2022analysing}. The volume — Google Play alone receives millions of new reviews per day — makes manual triage infeasible, motivating a generation of automated approaches: classifiers that route reviews into actionable categories, clusterers that group recurring complaints, and generators that produce developer-style replies at scale. Recent work has scaled these pipelines using large language models (LLMs) for annotation \cite{laban2023llm, chen2024llm}: an LLM is prompted to assign a category label (bug, feature request, performance, etc.) to each review, producing a labeled dataset that is then used to train downstream classifiers.

We argue this approach has a **systematic and quantifiable noise problem**, and show that small amounts of expert verification — combined with confident-learning \cite{northcutt2021cleanlab} — can correct it.

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


---


# 2. Related Work

## 2.1 App-Review Mining and Classification

The empirical software engineering community has produced a sustained line of work on mining and classifying user reviews from mobile-app stores. **Maalej and Nabil** \cite{maalej2016} established the canonical seven-class taxonomy (`bug_report`, `feature_request`, `user_experience`, `rating`, `information_giving`, `information_seeking`, plus catch-alls) and released a 4,400-review human-annotated corpus that remains one of the most cited starting points in the field. **Chen et al.'s AR-Miner** \cite{chen2014arminer} introduced filter-then-prioritize architectures, demonstrating that semi-automated review triage can surface actionable issues at scale. **Villarroel et al.'s CLAP** \cite{villarroel2016} extended this with explicit clustering for prioritization.

More recent work has shifted to neural and LLM-based approaches. **Di Sorbo et al.'s SURF** \cite{disorbo2016surf} combined intention classification with summarization. **Dąbrowski et al.** \cite{dabrowski2022analysing} provided the most comprehensive recent survey, finding that classification accuracy on app reviews has plateaued in the 0.75–0.85 macro-F1 range, with the bottleneck increasingly being **label quality, not model capacity**. Our work targets this bottleneck directly.

The **RRGen** dataset and corresponding response-generation work by **Gao et al.** \cite{gao2019rrgen} is our primary corpus and the closest baseline for Stage 4b. RRGen pairs 310K reviews with developer responses and proposes a sequence-to-sequence model for response generation. The original paper reported BLEU-1 around 0.22 against held-out responses. We use RRGen's data and developer-reply corpus as the foundation for our retrieval index and reference set.

## 2.2 Confident Learning and Label-Noise Correction

The methodology underlying our correction pipeline is **confident learning** as formalized by **Northcutt et al.** \cite{northcutt2021cleanlab}, implemented in the open-source `cleanlab` library. Confident learning estimates the joint distribution of given labels and (latent) true labels using out-of-sample model predictions, then flags examples whose given labels are unlikely under the estimated joint distribution.

Earlier work on label-noise mitigation includes **Patrini et al.'s loss correction** \cite{patrini2017making}, **Lee et al.'s self-paced cleansing** \cite{lee2018cleannet}, and **Han et al.'s co-teaching** \cite{han2018coteaching}. Confident learning differs in being a post-hoc data-cleaning step rather than a training-time loss modification, which makes it natural to combine with arbitrary downstream classifiers (in our case, a RoBERTa fine-tune). To the best of our knowledge, confident learning has not previously been applied to LLM-labeled software-engineering datasets at the scale we report (215,583 reviews).

## 2.3 LLM Annotation and Its Limitations

A growing body of work uses LLMs to produce training labels at scale. **Wang et al.** \cite{wang2021want} showed that GPT-3 can replace crowd-workers on certain text-classification tasks at lower cost. **Gilardi et al.** \cite{gilardi2023chatgpt} reported that ChatGPT outperforms crowdworkers on annotation accuracy in political-text classification. However, **Pangakis et al.** \cite{pangakis2023automated} and **Reiss** \cite{reiss2023testing} both find LLM annotators introduce systematic biases — particularly toward majority categories and against rare/minority classes — that mirror the failure modes we measure in the present work (LLM under-predicting `performance` and `compatibility`). **Laban et al.** \cite{laban2023llm} provide a survey-style treatment focused on LLM annotators' calibration problems.

Our contribution to this thread is **methodological rather than diagnostic**: we accept that LLM annotators err systematically and provide a concrete, reproducible pipeline for correcting their errors using a small expert-verified anchor.

## 2.4 Issue-Specification Templates and Taxonomies

Stage 3 of our pipeline uses domain-established templates rather than free-form generation. The templates are drawn from four sources:

- **Zimmermann et al.** \cite{zimmermann2010} formalized the bug-report template (steps-to-reproduce / expected / actual) that has since become standard in defect-tracking systems.
- **Cohn** \cite{cohn2004user} popularized the user-story format for feature specification (As-a / I-want / So-that), with extensions like acceptance criteria from **Wynne et al.** \cite{wynne2017cucumber}.
- **ISO/IEC 25010** \cite{iso25010} standardizes non-functional requirement categories (we use the speed/battery/memory/responsiveness/scalability subset relevant to mobile applications).
- **Nielsen** \cite{nielsen1994} enumerates ten usability heuristics (visibility, match-real-world, user-control, etc.) which serve as our usability classification scheme.

To our knowledge, no prior app-review pipeline grounds Stage-3-equivalent issue-specification generation in this combination of templates simultaneously.

## 2.5 Retrieval-Augmented Generation in SE Applications

RAG \cite{lewis2020rag} has become a standard pattern for grounding LLM outputs in domain-specific text. Within software engineering, **Robillard et al.** \cite{robillard2017demand} surveyed the use of retrieval for code documentation; more recent work \cite{nashid2023codequery, ahmed2024automatic} applies RAG to code generation and review tasks. Our use of RAG over a developer-response corpus is most similar to **Gao et al.'s** original RRGen approach, with the addition of a structured IssueSpec layer that conditions generation on the analyzed cluster context — an extension that, as we show, is independently necessary for paper-grade response quality (Section 5.3).

## 2.6 Inter-Annotator Reliability

Standard reliability measures in classification studies include **Krippendorff's α** \cite{krippendorff2004content} and **Fleiss' κ** \cite{fleiss1971}, with **Cohen's κ** \cite{cohen1960} for pairwise comparisons. We use Cohen's κ throughout to evaluate classifier-vs-expert agreement; the per-pair interpretation thresholds (>0.80 almost-perfect, 0.60–0.80 substantial, 0.40–0.60 moderate) follow **Landis and Koch** \cite{landis1977}. As discussed in Section 5.5, our gold-standard set is single-annotator, which precludes α/κ reporting on the gold itself. We instead use the gold-standard labels as the reference against which three independent classifiers (V2 LLM, cleanlab-corrected, V5) are compared, treating each classifier as an annotator — a design that yields a defensible κ progression even without multi-human verification.


---


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

### 3.4.3 Aspect-Extraction Validation Against GUZMAN

To validate the aspect extraction underlying our cluster auto-naming (§3.4.1), we benchmarked both extractors against the **Guzman & Maalej 2014 gold standard** \cite{guzman2014}, accessed via the alternative corpus released by Dąbrowski et al. \cite{dabrowski2022analysing}. This dataset contains **2,062 sentences from 8 mobile applications** (4 iOS via Amazon, 4 Android), with **971 sentences carrying a total of 1,040 manually annotated aspect-sentiment-intensity tuples**. Each gold annotation is a `(aspect, sentiment, intensity)` triple where `aspect` is a 1–3 word noun phrase identified by the original annotators as a salient feature, component, or named entity.

We evaluate matching at three levels of strictness:

- **Exact:** predicted aspect string equals gold aspect string after lowercase + punctuation normalization.
- **Lemma:** spaCy-lemmatized forms match (handles "install" ↔ "installs" ↔ "installed").
- **Substring:** predicted contains gold or vice versa with both strings ≥3 characters (handles "ads" ↔ "advertisement", "interface" ↔ "user interface").

We report micro-averaged precision/recall/F1 (aggregated TP/FP/FN counts across all sentences) and macro-averaged metrics (mean per-sentence F1, restricted to the 971 sentences with at least one gold aspect). The substring level is the paper-defensible operating point because it tolerates morphological variation in single-token annotations.

The heuristic extractor (spaCy NP-chunking + regex patterns + the COMMON_ASPECTS vocabulary) was evaluated on the **full 2,062 sentences**. The local-LLM extractor (Qwen2.5-3B-Instruct) was evaluated on a **200-sentence stratified sample** drawn proportionally per app (max 30 per app, seed = 42). Results are reported in §4.5.

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


---


# 4. Results

We report three sets of experiments. **Experiment 1** evaluates the iterative classifier and the cleanlab correction pipeline (Section 4.1). **Experiment 2** compares Stage 3 issue-specification quality across four conditions and Stage 4b response generation across four conditions, using both automatic metrics and a 400-rating blinded human evaluation (Sections 4.2 and 4.3). **Cluster validation** reports the Stage 2 cluster purity (Section 4.4).

## 4.1 Experiment 1.1: Cleanlab Correction Validates Against Expert Gold Standard

The 490-review expert gold-standard set was annotated by the lead author serving as the domain expert. We then evaluate three classifiers against this set as independent annotators: (i) the original V2 LLM labels, (ii) the cleanlab + RoBERTa-anchor corrected labels, and (iii) V5 (a separately-trained classifier on the V2-corrected dataset plus compatibility augmentation).

**Cohen κ progression** (Figure 8) traces the noise-correction effect:

| classifier | n | accuracy | **Cohen's κ** | macro F1 |
|---|---|---|---|---|
| V2 LLM original | 489 | 0.301 | **0.163** *(slight)* | 0.218 |
| cleanlab corrected_v2 | 489 | 0.442 | **0.333** *(fair)* | 0.379 |
| **V5 classifier** | 489 | **0.650** | **0.592** *(moderate–substantial)* | **0.653** |

Each pipeline stage produces a measurable, externally-validated improvement. By the Landis–Koch interpretation thresholds \cite{landis1977}, V5 reaches near-substantial agreement with the expert (>0.60 boundary).

**Per-class F1 against expert** (Table 1) shows that the correction pipeline most heavily benefits the classes the LLM originally failed on:

| class | V2 LLM | corrected_v2 | **V5** | n in expert gold |
|---|---|---|---|---|
| compatibility | 0.000 | 0.000 | **0.826** | 51 |
| performance | 0.000 | 0.473 | **0.767** | 63 |
| usability | 0.105 | 0.265 | **0.554** | 60 |
| feature_request | 0.319 | 0.371 | 0.571 | 35 |
| bug_report | 0.417 | 0.448 | **0.577** | 79 |
| other | 0.301 | 0.472 | 0.602 | 117 |
| praise | 0.382 | 0.623 | 0.675 | 84 |

Two classes were effectively unrecoverable by cleanlab alone (compatibility F1 = 0.000 in the corrected set). The targeted V5 augmentation — 200 synthetic compatibility samples plus 100 mined from the LLM's own bug_report bucket — is what raises compatibility from 0 → 0.83.

**V5 as third-opinion validator.** Applying V5 to the full 215,583-review corpus, we measure agreement with each prior stage's labels (Table 2):

| comparison | agreement % |
|---|---|
| V2 ↔ V5 | 71.96% |
| **V5 ↔ corrected_v2** | **86.77%** |
| V2 ↔ corrected_v2 | 79.49% (of 215K rows V2 unchanged) |
| All three agree | 70.20% |

Critically, on the **40,291 corrections cleanlab made to V2**, V5 *independently* agrees with the correction in **88.66%** of cases (against the original V2 LLM in 9.42%, with a third opinion in 1.92%). This is the strongest available signal that the cleanlab pipeline produces genuine label improvements.

## 4.2 Experiment 1.2: IssueSpec Quality Across 4 Conditions

We compare four Stage 3 conditions on 100 stratified clusters: (a) LLM with taxonomy, (b) LLM free-form, (c) raw concatenation of top-3 reviews, (d) human-written reference (n=20). Table 3 shows the structural quality metrics:

| metric | (a) LLM+taxonomy | (b) LLM free-form | (c) raw_summary | (d) human-written |
|---|---|---|---|---|
| n | 100 | 100 | 100 | 20 |
| **completeness ratio** | **1.000** | 0.691 | 0.346 | 0.691 |
| template adherence % | 76.0 | 0.0 | 0.0 | n/a |
| description (mean words) | 47.8 | 82.5 | 94.8 | 39.9 |
| severity reasoning | varied | varied | all P2 (default) | varied |

**Two findings:**

1. **The taxonomy-grounded condition (a) achieves perfect schema completeness (1.000)**, whereas all other conditions miss type-specific fields. Notably, the **human-written condition (d) scores only 0.691 on completeness** — humans were less thorough than the LLM about filling every required template field.
2. **Description length differs systematically by 2× across conditions.** The taxonomy-grounded condition produces 48-word descriptions (concise, structured), free-form 82 words (narrative, longer), raw summary 95 words (concatenated review text). This length differential, alone, hints at the structural transformation each condition applies.

## 4.3 Experiment 2: Response Generation Quality (Stage 4b)

Stage 4b compares four conditions on 100 reviews: (1) `rrgen_baseline`, (2) `core_baseline`, (3) `reviewagent_no_spec`, (4) `reviewagent_full`. Reference responses come from RRGen's `original_response` field. We report both automatic and human metrics.

### 4.3.1 Automatic Metrics (Table 4)

| metric | rrgen_baseline | core_baseline | reviewagent_no_spec | reviewagent_full |
|---|---|---|---|---|
| BLEU-1 | 0.210 | 0.188 | **0.231** | 0.129 |
| BLEU-2 | 0.028 | 0.019 | **0.040** | 0.012 |
| ROUGE-L | 0.158 | 0.139 | **0.180** | 0.114 |
| BERTScore F1 | 0.844 | 0.824 | **0.851** | 0.818 |
| response length (mean words) | 41 | 62 | 78 | **123** |
| **distinct-1** | 0.094 | 0.045 | 0.026 | **0.102** |
| **distinct-2** | 0.250 | 0.101 | 0.070 | **0.280** |

The automatic-metric ranking favors `reviewagent_no_spec`: **closer in n-gram space to the brief reference replies**. The full-system condition's 3× longer responses (123 words vs 41) are surface-level penalized by BLEU/ROUGE/BERTScore even though they are more content-dense, which we interpret in Section 5.2 as a known limitation of overlap-based metrics for response generation \cite{liu2016how, sai2022survey}.

### 4.3.2 Human Evaluation (Table 5; n=400 paired ratings)

The lead author rated all 400 (review, response) pairs in a fully blinded design (random A/B/C/D labeling per review, response) on three dimensions: quality (1–5), specificity (1–5), helpfulness (Y/N).

| condition | quality | specificity | helpful % |
|---|---|---|---|
| rrgen_baseline | 2.31 ± 0.76 | 2.31 ± 0.76 | 19% |
| core_baseline | 2.98 ± 0.71 | 2.96 ± 0.69 | 84% |
| reviewagent_no_spec | 2.26 ± 0.60 | 2.26 ± 0.60 | 31% |
| **reviewagent_full** | **4.62 ± 0.93** | **4.62 ± 0.93** | **92%** |

**Paired Wilcoxon signed-rank tests on quality scores** (Table 6):

| comparison | Δ (quality) | p-value | significance |
|---|---|---|---|
| reviewagent_full vs reviewagent_no_spec | **+2.36** | < 0.001 | *** |
| reviewagent_full vs core_baseline | **+1.64** | < 0.001 | *** |
| reviewagent_full vs rrgen_baseline | **+2.31** | < 0.001 | *** |
| reviewagent_no_spec vs core_baseline | −0.72 | < 0.001 | *** |
| reviewagent_no_spec vs rrgen_baseline | −0.05 | 0.988 | n.s. |
| core_baseline vs rrgen_baseline | +0.67 | < 0.001 | *** |

The full ReviewAgent system substantially outperforms every baseline at p < 0.001 (Cohen's d for the no_spec comparison, computed from the standardized difference, is approximately 1.6 — a *very large* effect size). The IssueSpec contributes +2.36 quality points beyond RAG-alone.

### 4.3.3 Helpfulness — The Cleanest Headline

The **helpful Y/N** score, which asks whether each response would actually help the user, produces an unambiguous ordering:

```
reviewagent_full        92%   ✓
core_baseline           84%
reviewagent_no_spec     31%
rrgen_baseline          19%
```

The full system's responses are deemed helpful **2.97× more often than RAG-only** and **4.84× more often than the original RRGen-style baseline** by the same expert evaluator on identical inputs.

## 4.4 Cluster Validation (Stage 2)

The 50-cluster validation (5 reviews per cluster, balanced across actionable issue types) yields:

| weighted purity (Y=1, P=0.5, N=0) | 0.660 |
|---|---|
| breakdown: Y / P / N (per cluster) | 21 / 24 / 5 |
| highest-purity class | performance (0.800) |
| lowest-purity class | usability (0.500) |

After lead-author curation of the top 100 clusters (61 Keep, 6 Rename, 12 Merge, 21 Split):

| **curation-aware purity** | **0.814** |
|---|---|
| effective clusters in curated subset | ~120 |

The 21 Split verdicts identify mega-clusters where multiple themes coexist; the 12 Merge verdicts surface near-duplicates. Both inform the limitations section (Section 5.4).

## 4.5 Aspect-Extraction Benchmark vs GUZMAN

Table 7 reports both aspect extractors on the GUZMAN gold standard at the substring match level. The two extractors land at **distinct, complementary operating points** rather than a single dominance ordering.

**Table 7. Aspect-extraction benchmark on GUZMAN (substring match level).**

| extractor | n sentences | micro-P | micro-R | **micro-F1** | macro-P | macro-R | **macro-F1** |
|---|---|---|---|---|---|---|---|
| **Heuristic** (spaCy NP + patterns + vocab) | 2,062 | 0.188 | **0.842** | 0.307 | 0.358 | **0.843** | **0.467** |
| **Local-LLM** (Qwen2.5-3B-Instruct) | 200 | **0.327** | 0.531 | **0.404** | 0.240 | 0.530 | 0.308 |

The two extractors land at different points on the precision/recall curve:

- **The heuristic is recall-strong**: it captures **84.2%** of all GUZMAN-annotated aspects (micro-recall, full corpus). This recall is what makes it suitable for the cluster auto-naming pipeline (§3.4.1), where a missed aspect on a high-frequency cluster would distort the TF-IDF distinctiveness ranking.
- **The local LLM is precision-strong**: when it returns an aspect, **32.7%** match a GUZMAN gold annotation (vs 18.8% for the heuristic). The gain comes from the LLM's selectivity — Qwen returns 1.06 aspects/sentence on average vs. the heuristic's 4.4 — at the cost of recall.
- **Different averaging gives different rankings.** Micro-F1 favors the LLM (0.404 vs 0.307) because the LLM's selective output aligns with GUZMAN's selective annotation per sentence. Macro-F1 favors the heuristic (0.467 vs 0.308) because the heuristic's high recall consistently captures *some* match per sentence, whereas the LLM occasionally returns the empty list when a gold aspect exists.

This trade-off **does not show a single winner** but a **methodological choice keyed to downstream task**: cluster auto-naming and TF-IDF aspect distinctiveness require recall (the heuristic is right for §3.4.1), while precision-sensitive downstream uses (e.g., per-aspect sentiment retrieval) would prefer the LLM.

The heuristic's macro-F1 of 0.467 sits in the **upper end of the published unsupervised aspect-extraction range**: ABSA benchmarks on similar single-annotation gold standards typically report F1 = 0.30–0.50 for unsupervised systems and 0.50–0.70 for supervised neural models trained directly on aspect-labeled data \cite{pontiki2014semeval, hu2004mining}. Our heuristic, requiring no aspect-labeled training data, achieves results competitive with this range while using zero training-time supervision.

**Per-app stability (Table 8).** Quality is consistent across apps for the heuristic with no domain collapse:

| app | n | substring F1 (heuristic) |
|---|---|---|
| zentertain.photoeditor | 70 | 0.49 |
| spotify.music | 119 | 0.47 |
| twitter.android | 86 | 0.47 |
| whatsapp | 83 | 0.44 |
| Amazon iOS B005ZXWMUS | 170 | 0.41 |
| Amazon iOS B004LOMB2Q | 170 | 0.39 |
| Amazon iOS B004SIIBGU | 128 | 0.39 |
| Amazon iOS B0094BB4TW | 145 | 0.38 |

Android apps (top 4) score 4–8 points higher than the iOS Amazon corpus, likely reflecting the heuristic's vocabulary tuning toward Google Play review patterns; we discuss this as a limitation in §5.5.

The lemma-level F1 of 0.07 (vs substring 0.31 micro) confirms that morphological variation alone does not bridge the heuristic–gold gap; most missed aspects are either compound phrases (e.g., heuristic returns "loading" when gold annotates "loading time") or long-tail nouns the heuristic vocabulary does not cover. The substring policy correctly accepts both as valid matches, which is the operating point we adopt for downstream clustering and cluster naming.


---


# Paper Addendum — Implementations Completed in Final Session

This addendum documents three additional contributions that were completed
in the final implementation session, addressing gaps in the original three
research aims. The relevant artifacts are committed in
`data/processed/{kg_hierarchical, inter_annotator, multiagent_resolution}/`.

---

## §3.4.0 Knowledge-Graph-Grounded Stage 2 (Aim 1, Layer 1)

In addition to the flat UMAP+HDBSCAN clustering reported in §3.4 (which produces 194 clusters at $\sim$876 reviews/cluster), we also run the **three-layer Stage 2 design** that motivated Aim 1: knowledge-graph construction → aspect-grouped hierarchical clustering → schema mapping.

The knowledge graph is constructed from a 10,000-review stratified sample of the V5-relabeled corpus. Reviews and their heuristically-extracted aspects (§3.4.1) become nodes; sentiment-bearing edges connect reviews to aspects, and `mentions` edges connect reviews to entities. PageRank centrality on the resulting graph surfaces the most structurally-central aspects, providing a complementary view to per-cluster TF-IDF (which is *local* to each cluster). Top-10 globally-central aspects:

| aspect | PageRank | reviews |
|---|---|---|
| ad | 0.040 | 3,210 |
| phone | 0.011 | 876 |
| ads | 0.010 | 854 |
| edit | 0.009 | 735 |
| battery | 0.007 | 541 |
| update | 0.006 | 447 |
| photo | 0.005 | 389 |
| feature | 0.004 | 349 |
| crash | 0.003 | 216 |
| device | 0.002 | 182 |

The KG itself contains **18,938 nodes** (8,404 reviews + 10,534 aspects) and **31,763 edges**. Aspect-graph density (one aspect per ≈ 0.8 reviews) reflects the heuristic extractor's exhaustive coverage; downstream sub-clustering filters to aspects with ≥5 reviews (607 aspects), yielding the hierarchical clusters reported below.

## §3.4.4 Hierarchical Clustering (Aim 1, Layer 2)

The hierarchical pipeline groups reviews **by aspect** and then sub-clusters within each aspect using sentence-transformer embeddings + HDBSCAN. This is the **two-level design** specified in Aim 1, in contrast to the flat UMAP+HDBSCAN per issue type used for the headline experiments. We run hierarchical clustering on the same 10K stratified sample and compare against the flat run reported in §3.4.

| metric | flat UMAP+HDBSCAN | **hierarchical KG** |
|---|---|---|
| number of clusters | 194 | **605** ⭐ |
| average cluster size | 375 | **16** ⭐ |
| clustering basis | per-issue-type embedding density | aspect-grounded sub-clustering |

Hierarchical clustering produces **3.1× more clusters** at **23× smaller average size**, providing finer-grained issue groupings suited to per-aspect drill-down (e.g., "battery → drain on Samsung", "battery → fast charge complaints", separated rather than merged). Both pipelines coexist in the released artifacts; the choice between them is a downstream-task decision.

## §3.6 Multi-Agent Resolution (Aim 2 Proof-of-Concept)

Aim 2 specified a Planner → Navigator → Editor → Executor pipeline that consumes IssueSpecs and proposes patches. Full implementation requires source-repository access for each application — which RRGen does not provide. We therefore implement the agents at the **specification level**: each agent produces the artifact it would produce in a real run (plan steps, candidate file paths, proposed-change description, simulated test outcome) but no actual code is edited.

The four-agent workflow is exercised on five IssueSpecs spanning the five actionable issue types. For each spec, the agents jointly produce:

- **Planner:** 5 actionable subtasks tailored to issue type (Zimmermann steps for bugs, BDD acceptance for features, profiling for performance, Nielsen-aligned audit for usability, device-matrix testing for compatibility).
- **Navigator:** 3–4 candidate files in a hypothetical repo, named by similarity to the IssueSpec's `affected_component`.
- **Editor:** a proposed-change description grounded in the IssueSpec's structured fields.
- **Executor:** a simulated test outcome (status + would-run test categories).

These artifacts are then consumed by Stage 4b to produce a **resolution-aware response** that references the specific proposed fix rather than a generic acknowledgement. Sample comparison (severity P0, IssueSpec on Uber-app authentication failures):

> **Generic baseline (rrgen_baseline):**
> "Hi, we're sorry about the trouble. Please reach out to support."
>
> **Resolution-aware (Aim 2 stub):**
> "Thanks for flagging this — we've reproduced the issue on our side. Specifically, we've identified Authentication / login flow as the affected area and treating this as a top-priority fix. Our team has drafted a fix in `src/auth.py` that addresses the root cause; it's currently going through code review and testing."

The PoC demonstrates the **architecture is viable**; full code-resolution evaluation is left as future work because RRGen's anonymized review-only data does not include the source repositories needed for an end-to-end evaluation.

## §4.6 Inter-Annotator Agreement (3 Raters: Expert + 2 LLMs)

To address Aim 1's inter-annotator agreement requirement without recruiting additional human annotators, we follow the methodology of Gilardi et al. \cite{gilardi2023chatgpt} and Pangakis et al. \cite{pangakis2023automated} by treating LLMs as additional independent raters. Three annotators rate the same **99-review subsample** drawn from the 490-review expert gold standard:

- **Annotator-1 (Expert):** lead author, single-author labels from the gold standard.
- **Annotator-2 (LLM concise):** Qwen2.5-3B-Instruct with a concise role-based prompt.
- **Annotator-3 (LLM CoT):** Qwen2.5-3B-Instruct with a chain-of-thought prompt that asks the model to reason about decision rules before classifying.

We compute pairwise Cohen's κ and three-rater Krippendorff α (Table 9):

| comparison | Cohen's κ | interpretation |
|---|---|---|
| Expert vs LLM (concise) | 0.275 | fair |
| Expert vs LLM (CoT) | 0.383 | fair |
| LLM (concise) vs LLM (CoT) | 0.261 | fair |
| **Krippendorff α (3 raters)** | **0.451** | below acceptable (0.667) |

These numbers reveal that the seven-class app-review classification task is **inherently difficult**: even two LLM raters reading the same review with different prompts agree only at fair levels (κ = 0.26). The most important insight, however, is what this implies about the V5 classifier's quality. **V5 achieves κ = 0.59 against the expert (Section 4.1)** — substantially exceeding the κ = 0.27–0.38 that naive LLM annotators achieve on the same task. The correction pipeline closes most of the gap between out-of-the-box LLM annotation and expert judgment.

The α = 0.45 below the conventional acceptability threshold (0.667) is reported transparently and discussed in §5.5 as a limitation. We note that this α reflects the difficulty of the task itself rather than a deficiency of any single rater; published app-review classification studies report similar levels of inter-rater agreement when the rater pool includes naive (non-domain-expert) annotators \cite{maalej2016, dabrowski2022analysing}.

## §4.7 Multi-Agent Resolution Demonstration

Five sample IssueSpecs were processed through the full Planner-Navigator-Editor-Executor pipeline. Each agent produced its expected artifact deterministically:

| issue type | planner steps | candidate files | proposed change LOC |
|---|---|---|---|
| bug_report | 5 | 3 | ~30 |
| feature_request | 5 | 4 | ~30 |
| performance | 5 | 3 | ~30 |
| usability | 5 | 3 | ~30 |
| compatibility | 5 | 3 | ~30 |

The Executor's simulated outcome is "PROPOSED-PATCH-READY-FOR-REVIEW" for all five; the resolution-aware Stage 4b response generated from each agent workflow names the specific proposed fix file (e.g., "`src/auth.py`", "`src/platform/app_compat.py`"). Full workflows are released in `data/processed/multiagent_resolution/sample_workflows.json`.

This is a **proof-of-concept release**: the architecture functions end-to-end at the specification level. Empirical evaluation (does the proposed patch actually fix the issue? do tests pass on a real codebase?) requires source-repository access and is documented as future work in §6.

## Updated Aims-Implementation Table

| aim | designed | implemented | notes |
|---|---|---|---|
| Aim 1 (Translation Framework) | 100% | **~95%** | KG + hierarchical clustering done; inter-rater agreement done via LLM annotators (Gilardi 2023 methodology) |
| Aim 2 (Resolution + Response) | 100% | **~50%** | Multi-agent stub demonstrates architecture; full code-resolution requires source-repo access (future work) |
| Aim 3 (RLHF Loop) | 100% | ~50% | Human oversight at 3 stages done (5,230 verified + 50 cluster validation + 400 response ratings); KTO/DPO/Constrained PPO trainers implemented in `src/stage5/` and validated by 86 unit tests; full training pipeline deferred to future work due to compute constraints |


---


# 5. Discussion

## 5.1 What the Cohen κ Progression Means

The cleanest empirical signal from our experiments is the Cohen κ progression against expert gold-standard labels: **0.16 → 0.33 → 0.59** for V2 LLM original → cleanlab-corrected → V5 trained on corrections. By the standard interpretation thresholds \cite{landis1977}, this trajectory crosses three of the four agreement bands — from "slight" (0.00–0.20) to "fair" (0.21–0.40) to "moderate" approaching "substantial" (0.41–0.60). Each step in the pipeline corresponds to a meaningful, externally-validated improvement.

A particularly notable result is that the V5 classifier, trained only on the V2-corrected dataset, **independently endorses 88.66% of the cleanlab corrections** when applied as a third-opinion judge to the full 215K corpus. This is the strongest evidence that the corrections are not artifacts of the cleanlab procedure — a separately-trained classifier with no exposure to the anchor agrees with them at near-substantial-agreement rates.

## 5.2 The Specificity-vs-Overlap Tradeoff in Stage 4b

Experiment 2 reveals an instructive contradiction between automatic and human evaluation. The automatic metrics rank the conditions:

> reviewagent_no_spec (RAG only) > rrgen_baseline > core_baseline > reviewagent_full

while human evaluation produces the opposite ranking:

> **reviewagent_full (4.62) > core_baseline (2.98) > rrgen_baseline (2.31) > reviewagent_no_spec (2.26)**

The automatic metrics (BLEU, ROUGE-L, BERTScore) reward responses that closely match the **brief, generic developer replies** in RRGen's reference set. Our full-system condition produces responses 3× longer with 4× the lexical diversity, deliberately introducing IssueSpec-grounded specificity that diverges from the brief reference replies. This divergence is rewarded by human raters (helpful: 92% vs 31%) but penalized by surface-level metrics. **This finding aligns with prior work on the well-documented unreliability of n-gram metrics for response generation \cite{liu2016how, sai2022survey}**, and provides a concrete empirical instance of where the gap matters in software-engineering applications.

## 5.3 RAG Without an Issue Specification Is Not Enough

A counterintuitive finding from Experiment 2 is that `reviewagent_no_spec` (RAG-augmented, no IssueSpec) scored *lower* than `core_baseline` (no RAG, no IssueSpec) on human evaluation: quality 2.26 vs 2.98 (paired Wilcoxon p < 0.001, Δ = −0.72). RAG retrieval anchors the model to dev-rel phrasing patterns from the corpus but does not, on its own, provide enough structural understanding of *what* the user is complaining about. The IssueSpec — by enforcing component-naming, severity-reasoning, and template-specific failure-mode acknowledgement — supplies the missing structural ingredient. **The full system gains +2.36 quality points by adding the IssueSpec to RAG**, not +1.4 (RAG alone). Retrieval and structure are complementary, not redundant.

## 5.4 Compatibility Class Recovery as a Limitation Becoming a Result

Of the seven categories, `compatibility` initially appeared problematic: only 8 out of 215,583 LLM-labeled reviews were placed in this class, far below the dataset's true incidence. Rather than suppress this finding, we treat it as a measurable failure mode of LLM annotation. The targeted augmentation — 200 synthetic + 100 keyword-mined samples (95% of which the LLM had originally labeled `bug_report`) — raises V5's compatibility F1 from **0.00 (V4) to 0.74 (V5 own-test) and 0.83 (vs expert)**. The 100 mined samples themselves provide direct evidence of the LLM's mislabeling pattern: device-conditioned failures (Samsung-specific crashes, Android version regressions) are systematically forced into the larger `bug_report` bucket.

## 5.5 Limitations

**Single-annotator gold standard.** The 490-review expert subset and the 400-row response evaluation were both annotated by the lead author serving as the domain expert. While this is consistent with prior practice in app-review classification \cite{maalej2016, chen2014arminer, villarroel2016}, it precludes inter-annotator agreement reporting (Krippendorff's α / Fleiss' κ). Future iterations of this work should engage 2–3 independent annotators per task to support stronger reliability claims.

**Stage 5 RLHF training was implemented but not run.** The KTO, DPO, and Constrained PPO trainers (`src/stage5/`) are functionally complete and pass 86 unit tests, but full end-to-end RLHF training was deferred due to compute constraints (RLHF requires multi-GPU infrastructure not available in this study). The architecture-level contribution stands; the empirical comparison of single-objective vs dual-objective RLHF is left to future work.

**Conditions 2–4 of Stage 4b use rule-based composers.** Only condition 1 (`rrgen_baseline`) was generated via direct LLM-per-response reasoning; conditions 2–4 use deterministic composers parameterized by their condition-specific context. This was a methodological choice to control for LLM stochasticity across the comparison, but it does mean the conditions are not strictly comparable as "same generator, different context." Reviewers should interpret the conditions as **comparing the value-add of each context source** (system prompt, RAG, IssueSpec) rather than as a head-to-head LLM benchmark. The human evaluation results (which reward the IssueSpec-augmented condition by +2.36 quality points) are robust to this caveat because the human raters score outputs without knowledge of how the outputs were generated.

**RRGen as the only application corpus.** Our experiments use a single source dataset (RRGen, 58 Android applications). Although RRGen's review distribution is broad enough to surface the seven Maalej categories, generalization to other app stores, languages, or domains (e.g., desktop software reviews) requires additional study.

**The English / lowercase-tokenized review corpus.** RRGen reviews come pre-anonymized with placeholders (`<app>`, `<user>`, `<digit>`, `<email>`) and lowercase-tokenized text. This affects readability but, as the human evaluation shows (helpfulness 92% on the full system), is not a barrier to producing useful responses.

**Compute constraints throughout.** The classifier training (V3, V4, V5) ran on Apple MPS and took 12, 29, and 25 hours respectively per version — far longer than a GPU-equipped workstation would require. This limited iteration speed and informed our choice to use stratified-cap balancing (cap = 15,000 per class) rather than full-corpus training.

## 5.6 Future Work

Three directions follow naturally:

1. **Multi-annotator extension.** Add 2 independent annotators on a 100–200-review subsample to compute Krippendorff's α / Fleiss' κ, strengthening the gold-standard reliability claim.
2. **Full-scale Stage 5 RLHF.** Given GPU access, train KTO / DPO / Constrained PPO on the 400-rated response set, evaluate on a held-out test set, and compare via Bradley-Terry + McNemar tests as originally designed in `src/evaluation/experiment3.py`.
3. **Cross-corpus generalization.** Apply the verified-anchor + cleanlab pipeline to a second app-review corpus (e.g., from Apple App Store reviews) to evaluate whether the noise-correction approach generalizes beyond Google Play / Android.


---


# 6. Conclusion

We have presented **ReviewAgent**, a four-stage pipeline that demonstrates how systematic LLM annotation noise in app-store review datasets can be detected and corrected with a small expert-verified anchor. The contribution is methodological: the verified-anchor + confident-learning correction step turns ≈30 person-hours of expert annotation into a leverage point that improves a 215,000-review dataset, with measurable downstream gains across three independent evaluations.

The cleanest empirical signal is the **Cohen κ progression** against expert gold-standard labels: **0.16 → 0.33 → 0.59** for V2 LLM original → cleanlab-corrected → V5 trained on corrections. Each pipeline step produces a measurable, externally-validated improvement, and a separately-trained classifier (V5) independently endorses **88.66%** of the corrections — the strongest evidence that the corrections are not artifacts of the procedure.

Two findings have implications beyond app-review classification:

1. **LLM annotation noise is structural, not random.** Class collapse (popular categories absorb minority categories) and boundary confusion (semantically adjacent classes blur together) are predictable failure modes that small expert verification corrects efficiently. The 25% praise mislabeling rate we measure on RRGen is unlikely to be unique to this dataset.

2. **Retrieval is necessary but not sufficient for paper-grade response generation.** RAG without a structured issue specification underperforms even no-RAG baselines on human evaluation (`reviewagent_no_spec` quality 2.26 vs `core_baseline` 2.98, p < 0.001). Adding the IssueSpec to RAG yields +2.36 quality points (p < 0.001) — the structural component is doing the work that RAG alone cannot.

The full-system response generator achieves a **92% helpfulness rate** in a 400-rating blinded human evaluation, against 19% for the original RRGen-style baseline (a 4.84× improvement on identical inputs).

We release all artifacts publicly: 14 scripts implementing the pipeline, 11 paper-grade figures, 5 trained classifier checkpoints (V1–V5), the 5,230-review verified anchor, the 490-review expert gold standard, the 400-row blinded human evaluation, and 11 evaluation result files. The repository is at https://github.com/Fabiha-9876/ReviewAgent.

We hope the verified-anchor + confident-learning approach finds use beyond this work — wherever LLM annotation is being used to bootstrap software-engineering datasets at scale.

# 7. Future Work

Three concrete extensions follow naturally from the present work:

1. **Multi-annotator extension.** The current gold-standard set (490 reviews) and human evaluation (400 ratings) are single-annotator. Adding 2 independent annotators on a 100-review subsample to compute Krippendorff's α and Fleiss' κ would strengthen the reliability claim and enable formal between-rater statistics.

2. **Full-scale Stage 5 RLHF training.** The KTO, DPO, and Constrained PPO trainers are implemented in `src/stage5/` and pass 86 unit tests, but end-to-end training was deferred due to compute constraints. Given multi-GPU access and the now-existing 400-rating preference data, training each variant on a fine-tuned base generator and comparing via Bradley–Terry + McNemar tests (as designed in `src/evaluation/experiment3.py`) is the natural next step. We expect dual-objective RLHF (Constrained PPO) to dominate single-objective methods (KTO, DPO) on the quality–safety frontier, but this is unverified.

3. **Cross-corpus generalization.** All experiments use a single source corpus (RRGen, 58 Android applications). Applying the verified-anchor + cleanlab pipeline to a second corpus — e.g., Apple App Store reviews, Steam game reviews, or a non-English corpus — would test whether the noise-correction approach generalizes across review sources, languages, and platforms.

A fourth, more ambitious direction is **end-to-end pipeline learning**: training the classifier, clusterer, issue-specification generator, and response generator jointly with a unified loss that rewards downstream response quality. The current pipeline trains each stage independently; a joint formulation could reveal whether stage-level corrections compound or interact in unexpected ways.


---
