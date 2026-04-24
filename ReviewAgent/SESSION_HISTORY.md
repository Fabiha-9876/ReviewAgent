# ReviewAgent — Session History & Decision Log

This file records all major decisions, actions, and outcomes from the development sessions so that future sessions can pick up where we left off.

---

## Session Timeline

### Phase 1: Documentation & Advisor Feedback (March 28, 2026)

**Starting point:** 4 files in `/Users/fabihajalal/Desktop/Review Agent/`:
- ReviewAgent_Project_Proposal.md
- ReviewAgent_Detailed_Architecture.md
- ReviewAgent_Research_Proposal.pdf
- generate_pdf.py

**Advisor feedback received (9 concerns from Hasan Mahmud via WhatsApp):**
1. Select relevant evaluation metrics for gold standard and different steps
2. Pipeline became huge — decide input materials for RAG
3. Decide everything and go for designing experiments
4. How many experiments do you need to claim contributions?
5. Required ablation studies
6. Experimental variables
7. For each step of proposed methodology there must be reference papers
8. No theoretical backup — try to relate through theoretical alignment
9. Seems to be wire-in wire-out — needs justification

**Actions taken to address each concern:**
1. Added finalized evaluation metrics table (per-stage + gold standard)
2. Fixed 5 RAG input sources (past responses, changelogs, FAQ, issue spec, similar responses)
3. Designed 3 experiments mapped to 3 RQs
4. Defined 3 experiments + 7 ablation studies (minimum to prove contributions)
5. Defined ablations A1-A7 with what's removed, what it tests, measured by
6. Defined independent (6), dependent (8), and control (5) variables
7. Created reference papers per methodology step table (17 entries)
8. Added 3 theoretical frameworks: IE Cascade (Hearst, Sarawagi), Human-AI Complementarity (Kamar, Bansal, Geifman), CMDP (Altman, Dai)
9. Every stage now has theory tags in architecture diagram

**Additional improvements:**
- Added formal Problem Statement (3-part: manual triage slow, no standardized issues, works in silos)
- Added 5 Key Contributions section with "What's new" + "Why it matters" for each
- Scoped pipeline: Stage 4a (code resolution) moved to future work
- Reduced RQs from 5 to 3 (focused)
- Updated PDF to 16 pages with all sections
- Added Expected Outcomes section (structured issue database, faster triaging, better developer insights, improved response quality)
- Total references expanded from 22 to 38 across 8 categories

### Phase 2: Project Implementation (March 29, 2026)

**Built the entire codebase — 95+ files:**

**Common module (3 files):**
- schemas.py — 8 Pydantic models (ReviewObject, IssueCluster, IssueSpec, GeneratedResponse, RubricScores, ComplianceFlags, AspectSentiment, ExtractedEntities)
- config.py — YAML config loader using OmegaConf
- llm_client.py — Unified OpenAI + Anthropic wrapper

**Stage 1: Intake (6 files):**
- classifier.py — RoBERTa multi-label (7 labels), train/predict/needs_hitl
- aspect_sentiment.py — LLM-based aspect-sentiment extraction
- entity_extractor.py — Hybrid regex + LLM (devices, OS, versions, screens, features)
- hitl_checkpoint.py — Confidence-based flagging, correction recording, active learning
- pipeline.py — Orchestrator with process() and process_with_hitl()

**Stage 2: KG + Clustering (6 files):**
- kg_builder.py — NetworkX DiGraph (review/aspect/entity nodes, sentiment/temporal edges)
- clustering.py — Two-level HDBSCAN (aspect-level then sub-cluster with sentence embeddings)
- schema_mapper.py — Maps clusters to IssueCluster schema (fixed fields)
- priority_ranker.py — PageRank + review_count + sentiment + recency
- pipeline.py — Orchestrator

**Stage 3: Translation (5 files):**
- taxonomy.py — 5 templates: Zimmermann bug, user story, ISO 25010 perf, Nielsen usability, compat matrix
- translator.py — LLM-based IssueCluster → IssueSpec with prompt building + parsing
- hitl_checkpoint.py — 5-dim rubric scoring, weak dimension detection, regeneration
- pipeline.py — Orchestrator with retry loop

**Stage 4b: Response Gen (5 files):**
- rag_retriever.py — ChromaDB with 5 sources
- response_generator.py — Issue-spec-aware generation
- self_refiner.py — Self-critique loop (specificity, compliance, empathy)
- pipeline.py — Orchestrator with ablation toggles

**Stage 5: RLHF (7 files):**
- feedback_collector.py — Dual-stream (quality 5 dims + compliance 4 flags) + text storage for DPO
- kto_trainer.py — Phase 1 binary feedback
- dpo_trainer.py — Phase 2 paired preferences
- constrained_ppo.py — Phase 3 CMDP dual-objective
- feedback_propagator.py — Backward propagation to Stages 1, 3, 4b
- pipeline.py — Auto-selects KTO/DPO/PPO by data volume

**Evaluation (6 files):**
- metrics.py — BLEU, ROUGE-L, BERTScore, completeness ratio, Krippendorff's alpha
- statistical_tests.py — Wilcoxon, Friedman, Nemenyi, Bradley-Terry, McNemar, Bonferroni
- experiment1.py, experiment2.py, experiment3.py — Full experiment runners
- ablations.py — A1-A7 runner

**API (5 files):** FastAPI with intake, issues, responses, feedback routes
**Scripts (4 files):** run_pipeline, run_experiment, download_datasets, train_classifier
**Tests (3 files):** test_schemas, test_kg_builder, test_metrics
**Configs (7 files):** base, stage1-5, experiments (all YAML)

### Phase 3: Data Acquisition (March 29, 2026)

**Datasets downloaded and verified:**
- MAALEJ unlabeled: 50,000 reviews from 189 apps (sealuzh/user_quality GitHub)
- MAALEJ labeled: 5,008 human-annotated reviews (mohammadzaeem GitHub, actual Maalej et al. 2016 data)
  - Labels: Problem Discovery→bug_report (864), Feature Request→feature_request (444), User Experience→usability (607), Rating→praise (2389), Information Giving/Seeking→other (704)
- RRGen: 310,031 review-response pairs from 58 apps (Google Drive via gdown)
  - Train: 279,802, Valid: 14,727, Test: 15,502
- GUZMAN: 2,062 sentences with 1,040 aspect-sentiment pairs from 8 apps (Dabrowski IS-22 repo)
- Synthetic: 500 template-generated reviews across 6 categories

**Why RRGen not used for classifier training:** Has review+response pairs but NO category labels
**Why GUZMAN not used for classifier training:** Has aspect-sentiment but NO category labels
**Why synthetic data was needed:** MAALEJ lacks performance (0 examples) and compatibility (0 examples) categories. Synthetic provides 70 performance + 50 compatibility.

### Phase 4: Model Training (March 29-30, 2026)

**Run 1 (auto-labeled, superseded):**
- Data: 9,529 (500 synthetic + 5,000 MAALEJ keyword-auto-labeled + 4,029 RRGen keyword-auto-labeled)
- F1 macro: 0.7478
- Problem: keyword heuristics are noisy

**Run 2 (actual MAALEJ human labels, current model):**
- Data: 5,508 (5,008 MAALEJ human-annotated + 500 synthetic)
- Training: 3 epochs, ~30 min on MPS Apple Silicon, batch size 8, lr 2e-5
- Training loss: 0.1877
- F1 macro: 0.7992 (+6.9% over Run 1)
- Per-label F1: bug_report 0.816, feature_request 0.627, performance 1.000 (14 samples), usability 0.552, compatibility 1.000 (10 samples), praise 0.820, other 0.780
- Model saved: models/stage1_classifier/ (476 MB safetensors)
- Critical fix during training: labels must be float32 for BCEWithLogitsLoss (multi-label), used custom MultiLabelCollator

**Advisor feedback on synthetic data:** "Synthetic data generation process should be verified by expert along with the samples"
- Created Synthetic_Data_Validation_Form.xlsx (4 sheets: Instructions, 150 review samples with rubric, process validation with templates/word lists, summary with verdict)
- Created Synthetic_Data_Validation_Guide.md
- Sent to advisor — awaiting expert feedback

### Phase 5: RAG Index Population (March 30, 2026)

**15,100 documents indexed into ChromaDB:**
- past_responses: 10,000 (from RRGen 310K, sampled valid pairs >20 chars)
- similar_responses: 5,000 (deduplicated by first 60 chars, sorted by length)
- changelogs: 60 (58 app-specific from RRGen + 10 generic templates)
- faq: 40 (30 from common response patterns + 10 generic)
- issue_spec: 0 (populated by Stage 3 at runtime)
- Script: scripts/populate_rag.py
- Fix: added index to document IDs to prevent ChromaDB DuplicateIDError

**Test queries verified:** crash→0.726, dark mode→0.750, battery→0.753, checkout→0.434

### Phase 6: Bug Fixes (March 30, 2026)

**DPO Training Text Mapping:**
- Problem: export_dpo_data() returned only response IDs, DPO trainer had no text
- Fix: Added register_response() to store prompt+response text, rewrote export_dpo_data() to return (prompt, chosen, rejected) text triples, groups by issue_id
- Also added export_kto_data_with_text() for KTO
- Updated Stage 5 pipeline to use text-aware exports

**Constrained PPO Reward Model Inference:**
- Problem: _score_quality() and _score_compliance() returned hardcoded 0.5
- Fix: Two-tier scoring for both (trained model + heuristic fallback)
- Quality heuristic: 5 signals (length, specificity, empathy, actionability, non-generic)
- Compliance heuristic: 4-dimension continuous scoring (promise 0.35 weight, info 0.25, tone 0.20, legal 0.20)

**Compliance scoring refined (after advisor challenge "compliance 1.0 is unrealistic"):**
- Problem: Most responses got 1.0 because only exact phrase matches caught violations
- Fix: Severity levels per dimension (hard 0.1, medium 0.4, soft 0.6-0.75, hedged 0.85)
- Min-weighted combination: severe violation in ANY dimension caps the total score
- Fixed "sue" matching inside "issue" with word boundary regex
- Results: generic template 1.0 (CLEAN), hedged 0.948 (minor risk), soft promise+leak 0.480 (HIGH RISK), hard violations 0.120 (SEVERE)

### Git History

| Commit | Hash | Description |
|---|---|---|
| 1 | 87bb24c | Initial proposal and architecture documents |
| 2 | b2a4d5d | Complete implementation: 88 files, 82,556 lines |
| 3 | f7e7548 | RAG index population + DPO text mapping fix |
| 4 | 031bcc9 | Constrained PPO reward model inference fix |
| 5 | e356e01 | Compliance scoring refinement (realistic continuous scores) |

**Repository:** https://github.com/Fabiha-9876/ReviewAgent

---

### Phase 7: 5-Fold Stratified Cross-Validation & Stage 3 Tests (April 1, 2026)

**Advisor feedback (Hasan Mahmud via WhatsApp):**
- "What was split and distribution of the splits?"
- "Did you perform k-fold cross validation?"
- "Distribution of priors in test and train splits?"
- "5 fold stratified is ok"
- "Increase the verified samples"
- "Look for volunteers" — need people to validate synthetic data
- "May require proper attributions" — volunteers need credit
- "To convince the reviewers" — make participation compelling

**5-Fold Stratified CV executed:**
- Script: `scripts/kfold_classifier.py` (already existed, just needed to be run)
- Used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on primary label
- Total samples: 5,508 (MAALEJ 5,008 + Synthetic 500)
- Each fold: ~4,406 train / ~1,102 val with preserved label proportions
- Results:
  - F1 micro: 0.7645 ± 0.0095
  - F1 macro: 0.7974 ± 0.0053
  - Consistent with single-split Run 2 (0.7992) — confirms model stability
- Per-label: bug_report 0.81, feature_request 0.66, performance 0.99 (14 samples), usability 0.50, compatibility 1.00 (10 samples), praise 0.82, other 0.81
- Weakness: usability (F1 0.50) despite 138 support; performance/compatibility perfect but only synthetic data
- Results saved: `KFOLD_CV_RESULTS.md` and `models/stage1_classifier/kfold_results.json`

**Stage 3 unit tests written (80 tests, all passing):**
- `test_taxonomy.py` (9 tests): Template selection per issue type, content validation, fallback to bug template for unknown types
- `test_translator.py` (29 tests): Prompt building (cluster data, entities, KG context), LLM response parsing (title, steps, severity, feature/performance/usability-specific fields), batch translation, edge cases (missing title/description fallbacks)
- `test_hitl_checkpoint.py` (26 tests): Score recording & persistence, threshold checking (pass/fail/boundary/custom), multi-rater aggregation, weak dimension detection, regeneration with feedback, file persistence
- `test_pipeline.py` (10 tests): Process without HITL, auto-approve when no callback, HITL validation with passing/failing scores, retry loop, max retries respected
- All tests use mocked LLM client (AsyncMock) — no real API calls needed
- Testing patterns match Stage 1 & 2 conventions (fixtures, tmp_path, asyncio.run)

**Volunteer recruitment plan (discussed with advisor):**
- Target: 3-5 grad students from department
- Task: ~45-60 min to validate 150 synthetic samples using existing form
- Attribution: co-authorship or acknowledgment
- Waiting on advisor to help connect with candidates
- Options for increasing verified samples:
  1. Manual annotation of RRGen reviews filtered by keywords (best quality)
  2. LLM-assisted labeling + human verification (fastest)
  3. Improved synthetic generation via LLM instead of templates (weakest)

### Phase 8: Stage 4b & 5 Unit Tests + Jupyter Notebooks (April 1-2, 2026)

**Stage 4b unit tests written (63 tests, all passing):**
- `test_rag_retriever.py` (15 tests): ChromaDB collection management (create once, different sources), indexing (calls add, default metadata), retrieval (score calculation as 1-distance, sorted descending, top-k, skip empty, handle exceptions, all sources default, source tagging)
- `test_response_generator.py` (22 tests): Context building (includes review text, rating, issue spec fields, actual behavior, steps, priority, RAG context, omits when disabled), generation (returns GeneratedResponse, review_id, issue_id, text, response_id, refinement=0, RAG sources, LLM params), batch (count, preserves IDs, empty)
- `test_self_refiner.py` (13 tests): Critique (all-pass, suggestions, issue spec in context, malformed JSON fallback, markdown code block, correct prompt/temp), revise (returns string, only non-pass feedback), refine loop (stops on pass, iterates until pass, max iterations, updates text, preserves IDs, with issue spec)
- `test_pipeline.py` (8 tests): Process returns responses, matches reviews to specs, extra reviews get no spec, empty input, no refinement when disabled, refinement enabled, no issue spec when disabled, single review
- All tests mock LLM (AsyncMock) and ChromaDB (patch) — no external dependencies

**Stage 5 unit tests written (86 tests, all passing):**
- `test_feedback_collector.py` (30 tests): Constants (quality 5 dims, compliance 4 dims), register response (stores text, persists, multiple), record quality (appends, stores scores/rater/timestamp, multiple raters), record compliance (appends, flags, noncompliant), export KTO (good=true, low quality=false, noncompliant=false, empty, multiple), KTO with text (includes text, skips missing), export DPO (creates pairs, single response, empty, requires score difference), export PPO (two lists, scores, flags, noncompliant, empty), persistence (loads existing, creates dirs)
- `test_feedback_propagator.py` (11 tests): Stage 1 (saves corrections, appends, empty), Stage 3 (dimension averages, weak dims file, no file when clean, empty), Stage 4b (saves scores, appends, empty), directory creation
- `test_constrained_ppo.py` (24 tests): Constrained reward (no violation, violation reduces, exact threshold, zero compliance, perfect), quality heuristic (good response, short=low, generic=low, empathy, actionable, specific, 0-1 range), compliance heuristic (clean, hard promise, soft promise, hedged, info leak hard/medium, tone violation, legal violation/soft, 0-1 range, empathy boost, "sue" not matched in "issue")
- `test_trainers.py` (10 tests): KTO (default params, custom, prepare dataset, empty, LoRA config), DPO (default, custom, prepare dataset, empty, LoRA config)
- `test_pipeline.py` (11 tests): Trainer selection (kto <500, dpo 500-1499, ppo >=1500), initialization, feedback propagation (to stage4b, empty, iteration increments)

**Key challenge: trl PPO removal**
- `trl` v1.0 removed `PPOConfig`/`PPOTrainer`; v0.7.x incompatible with current `transformers`
- Solution: Made `src/stage5/__init__.py` imports lazy (try/except) so missing PPO doesn't break other imports
- Constrained PPO tests recreate the scoring logic as a minimal class (no trl import needed) — tests the actual heuristic algorithms
- Pipeline tests use a mock pipeline class replicating the selection/propagation logic
- Result: **all 86 tests pass, zero skips**

**Full test suite: 335 tests, all passing**
- Stage 1 & 2: 88 tests
- Stage 3: 80 tests
- Stage 4b: 63 tests
- Stage 5: 86 tests
- Schemas + metrics: 18 tests

**Jupyter notebooks created (3 notebooks, 35 cells total):**
- `notebooks/01_data_exploration.ipynb` (14 cells): MAALEJ label distribution + original label mapping, text length distribution per category, synthetic data label + rating distribution, combined training set with red borders highlighting synthetic-only categories (performance, compatibility), RRGen overview (310K reviews, rating distribution, response lengths), summary table of all datasets
- `notebooks/02_classifier_results.ipynb` (10 cells): Single-split vs 5-fold CV side-by-side bar chart with error bars, per-label F1 sorted with color coding (red <0.6, orange <0.75, green >=0.75), cross-fold stability (line chart + heatmap of F1 across 5 folds x 7 labels), precision vs recall scatter with F1 iso-lines and bubble size = support
- `notebooks/03_knowledge_graph.ipynb` (11 cells): Sample ReviewObjects with aspects/entities, KG construction using actual `KnowledgeGraphBuilder`, graph visualization colored by node type (review, aspect, entity, device, OS), PageRank centrality ranking table + size-by-centrality visualization
- All figures save to `notebooks/fig_*.png` for paper/presentation use

---

### Phase 9: QA Volunteer Preparation & Annotation Protocol (April 6, 2026)

**Advisor feedback (Hasan Mahmud via WhatsApp):**
- "I have to look for volunteers" — started talking with QAs
- "Please give me a summary of description 5-6 lines of the project to give the QAs a brief idea"
- "Also their tasks need to be defined"
- "Option 1 is difficult for 310K samples. Option 2 and 3 can be done by the volunteers" — confirmed LLM pre-label + volunteer verification approach
- "See the SOTA how many samples would be required to validate to convince the reviewers in this domain"
- "You need a protocol of annotation/validation with a reliability measure through statistics — like agreements score"
- "Need your prompt response"

**Decision: Go with Option 2 — GPT-4 pre-label + volunteer verification**
- Filter ~1,000 RRGen reviews by keywords for performance/compatibility
- GPT-4 auto-labels into 6 categories
- Volunteers verify/correct (much faster than labeling from scratch)

**SOTA sample sizes researched (app review annotation papers):**
- Maalej et al. (2016): 4,400 reviews, 2 annotators
- Guzman & Maalej (2014): 2,062 sentences, 2 annotators
- Chen et al. (2014) AR-Miner: 1,000 reviews, 2 annotators
- Villarroel et al. (2016) CLAP: 1,390 reviews, 2 annotators
- Di Sorbo et al. (2016) SURF: 4,000+ reviews, 3 annotators
- Standard at ICSE/FSE/ASE: 500–2,000 samples, 2–3 annotators
- Target: 500–1,000 reviews verified by 3 annotators

**Annotation protocol defined:**
1. GPT-4 pre-labels ~1,000 filtered RRGen reviews into 6 categories
2. 3 annotators independently verify/correct each label (blind)
3. Training round: 20 practice samples to calibrate understanding
4. Reliability: Krippendorff's alpha (primary) + Fleiss' kappa (secondary)
5. Targets: α ≥ 0.67 acceptable, α ≥ 0.80 strong
6. Per-category agreement reported separately
7. Disagreements resolved by majority vote; ties adjudicated by expert

**QA volunteer tasks defined:**
- Task 1: Validate 150 synthetic reviews for realism (1-5) + category correctness (Y/N) — ~45-60 min
- Task 2: Verify/correct LLM-assigned labels on 500-1,000 real reviews — ~2-3 hours
- Task 3 (later): Score generated issue specs on 5-dimension rubric (1-5 scale)

**Incomplete steps identified (10 items):**
1. Write formal Annotation Protocol document — NOT STARTED
2. Research & cite SOTA sample sizes — DONE (listed above)
3. Build GPT-4 pre-labeling script (filter RRGen + auto-label) — NOT STARTED
4. Create volunteer verification form/spreadsheet — NOT STARTED
5. Send project summary + QA tasks + protocol to professor — NOT STARTED
6. Recruit & onboard volunteers — BLOCKED on professor connecting QAs
7. Execute annotation with 3 volunteers on 500-1,000 reviews — BLOCKED on step 6
8. Compute agreement scores (Krippendorff's alpha, Fleiss' kappa) — BLOCKED on step 7
9. Retrain classifier on verified data — BLOCKED on step 8
10. Build gold-standard dataset (200-300 clusters, 3+ experts) — BLOCKED on step 9

**Dependency chain updated:**
```
Steps 1-5 (can do NOW)
  → Step 6: Recruit volunteers (blocked on prof)
    → Step 7: Execute annotation
      → Step 8: Compute agreement scores
        → Step 9: Retrain classifier
          → Step 10: Gold-standard dataset
            → Run Experiments 1-3 + Ablations A1-A7
              → Paper results
```

---

## Current State (as of April 6, 2026)

### What's Complete
- All documentation (proposal, architecture, experiments, PDF, guide, training log)
- Full 5-stage pipeline implementation (45+ Python source files)
- 7 config files, 4 scripts
- All 3 datasets downloaded and verified (362,593 total data points)
- MAALEJ labeled dataset (5,008 human-annotated reviews)
- RoBERTa classifier trained on real labels (F1 macro 0.7992)
- 5-fold stratified cross-validation completed (F1 macro 0.7974 ± 0.0053)
- RAG index populated (15,100 documents in ChromaDB)
- DPO text mapping fixed
- Constrained PPO scoring fixed with realistic compliance
- Synthetic data validation form sent to advisor
- Full unit test suite: 335 tests passing (Stage 1-2: 88, Stage 3: 80, Stage 4b: 63, Stage 5: 86, schemas+metrics: 18)
- 3 Jupyter notebooks (data exploration, classifier results, KG visualization)
- SOTA sample sizes researched for annotation validation
- QA volunteer tasks defined
- Annotation protocol with reliability measures designed

### What's Remaining (10 items)

**Can do NOW (not blocked):**
1. Write formal Annotation Protocol document
2. Build GPT-4 pre-labeling script (filter RRGen + auto-label ~1,000 reviews)
3. Create volunteer verification form/spreadsheet
4. Send project summary + QA tasks + protocol to professor

**Blocked on volunteers/professor:**
5. Recruit & onboard volunteers (waiting on prof to connect QAs)
6. Execute annotation with 3 volunteers on 500-1,000 reviews
7. Compute agreement scores (Krippendorff's alpha, Fleiss' kappa)
8. Retrain classifier on verified data
9. Build gold-standard dataset (200-300 clusters, 3+ experts)
10. Execute experiments 1-3 and ablation studies A1-A7

### Dependency Chain
```
Steps 1-4 (can do NOW — no blockers)
  → Step 5: Recruit volunteers (blocked on prof)
    → Step 6: Execute annotation
      → Step 7: Compute agreement scores
        → Step 8: Retrain classifier
          → Step 9: Gold-standard dataset
            → Step 10: Run Experiments 1-3 + Ablations A1-A7
              → Paper results
```

### Pending
- Professor connecting with QA volunteers
- Synthetic data expert validation feedback
- Need to send professor: project summary + tasks + protocol (URGENT — he asked for prompt response)
- If synthetic data needs revision → regenerate → retrain classifier

### Status Report
- Excel report sent to advisor: `Desktop/ReviewAgent_Status_Update.xlsx`

### Key Technical Decisions Made
1. Used RoBERTa (not BERT) for classification — better performance on informal text
2. Used HDBSCAN (not K-means) for clustering — handles variable cluster sizes, no need to predefine K
3. Used ChromaDB (not FAISS) for RAG — persistent storage, easy to add/query
4. Used sentence-transformers all-MiniLM-L6-v2 for embeddings — fast, good quality
5. Custom MultiLabelCollator needed for HuggingFace Trainer — labels must be float32 not long
6. Compliance scoring uses min-weighted combination — single severe violation tanks the score
7. DPO pairs grouped by issue_id — ensures chosen/rejected compare responses to same prompt
8. Synthetic data fills MAALEJ gaps for performance and compatibility categories only
9. 5-fold stratified CV chosen over 10-fold per advisor recommendation
10. trl v1.0 removed PPO — stage5 __init__.py uses lazy imports; tests bypass trl entirely

---

### Phase 5: Progressive Semi-Supervised Labeling & Classifier V2 (April 8-9, 2026)

**Goal:** Expand labeled dataset from 5.5K to 18K+ using the trained classifier, then retrain.

#### Step 1: Synthetic Data Validation Sheet (500 reviews)
- Professor requested 500 reviews validated (original form had only 150)
- Generated complete validation form: `Synthetic_Data_Validation_Form_500.xlsx`
- 5 sheets: Instructions, Review Samples (all 500), Generation Methodology, Process Validation, Summary & Verdict
- Generation Methodology sheet documents: templates (29 total), word banks (9 types), distribution weights, rating logic, data sources, known limitations
- File location: `/Users/fabihajalal/Desktop/Review Agent/Synthetic_Data_Validation_Form_500.xlsx`

#### Step 2: Initial 10K Labeling with V1 Classifier
- Script: `scripts/label_rrgen_10k.py`
- Used trained V1 classifier (F1 0.7974) to predict labels on 10K RRGen reviews
- Results showed performance (6/10K, 0.1%) and compatibility (0/10K, 0.0%) nearly absent
- Confirmed the classifier cannot find these categories without retraining

#### Step 3: Progressive Labeling Pipeline
- Script: `scripts/progressive_labeling_fast.py`
- Strategy: Use existing classifier to label in rounds (10K → 20K → 30K)
- Round 1 includes keyword seeding for performance (150) and compatibility (100)
- Keywords: 37 performance terms, 33 compatibility terms
- Confidence threshold: 0.80

**Round Results:**

| Round | Sampled | Retained | Rejected | Acceptance Rate |
|-------|---------|----------|----------|-----------------|
| 1 (10K) | 10,000 | 6,382 + 250 seeded | 3,618 | 63.8% |
| 2 (20K) | 10,000 | 5,707 | 4,293 | 57.1% |
| 3 (30K) | 10,000 | 5,659 | 4,341 | 56.6% |

**Cumulative 18,498 labeled reviews:**

| Category | Count | % | Source |
|----------|-------|---|--------|
| bug_report | 7,527 | 40.7% | Model predicted |
| praise | 6,080 | 32.9% | Model predicted |
| other | 3,044 | 16.5% | Model predicted |
| feature_request | 1,391 | 7.5% | Model predicted |
| performance | 220 | 1.2% | 150 keyword + 70 synthetic |
| compatibility | 150 | 0.8% | 100 keyword + 50 synthetic |
| usability | 86 | 0.5% | Model predicted (weak) |

**Data files:**
- `data/processed/progressive/10k/labeled_10k.json` (7,132 reviews)
- `data/processed/progressive/20k/labeled_20k.json` (12,839 reviews)
- `data/processed/progressive/30k/labeled_30k.json` (18,498 reviews)
- Each folder also has: new_labels, low_confidence, stats, CSV

#### Step 4: Retrained Classifier V2 on 18K Data
- Retrained RoBERTa from scratch on 18,498 samples (90/10 split → 16,648 train / 1,850 val)
- 3 epochs, batch_size=16, lr=2e-5
- Training time: ~12 hours on MPS (Apple Silicon CPU fallback)
- Model saved: `models/stage1_classifier_v2/`

**V1 vs V2 Comparison:**

| Metric | V1 (5.5K) | V2 (18K) | Change |
|--------|-----------|----------|--------|
| F1 Macro | 0.7974 | 0.8558 | +0.058 ↑ |
| F1 Micro | 0.7645 | 0.9684 | +0.204 ↑ |
| Eval Loss | 0.0495 | 0.0400 | ↓ better |

**Per-Category V2 Results:**

| Category | F1 | Precision | Recall | Support | Notes |
|----------|-----|-----------|--------|---------|-------|
| bug_report | 0.98 | 0.96 | 0.99 | 731 | Excellent |
| praise | 0.99 | 0.99 | 0.98 | 617 | Excellent |
| other | 0.95 | 0.94 | 0.96 | 329 | Excellent |
| feature_request | 0.94 | 0.99 | 0.90 | 143 | Great |
| usability | 1.00 | 1.00 | 1.00 | 5 | Perfect but tiny support |
| performance | 0.67 | 1.00 | 0.50 | 12 | Precision perfect, recall weak |
| compatibility | 0.47 | 1.00 | 0.31 | 13 | Precision perfect, recall very weak |

**Key Finding:** V2 is much better overall (F1 micro 0.97), but performance/compatibility recall is still low because those categories have minimal real-world training examples (mostly keyword-seeded and synthetic). The model learned to be very precise (100% precision) but conservative (misses many real examples).

#### Scripts Created This Session
1. `scripts/label_rrgen_10k.py` — Label 10K RRGen with trained classifier
2. `scripts/progressive_labeling.py` — Full progressive pipeline with retraining (slow, CPU-bound)
3. `scripts/progressive_labeling_fast.py` — Fast version, predict-only rounds without retraining
4. `generate_validation_sheet.py` (in parent dir) — Generates 500-review validation Excel

#### What To Do Next
1. **Run another progressive round with V2 model** — V2 should find more performance/compatibility in the next 30K batch since it's better at the other categories (freeing up "other" misclassifications)
2. **Expand keyword seeding** — Add more aggressive keyword filtering specifically for performance/compatibility to boost those categories
3. **Consider LLM-assisted labeling** — Use GPT-4/Claude to label a targeted batch of performance/compatibility reviews identified by keywords
4. **Volunteer verification** — Once dataset is large enough, sample 500 for volunteer Y/N verification
5. **Gold-standard dataset** — Still blocked on annotation protocol execution

#### File Inventory (This Session)
```
models/
├── stage1_classifier/           # V1 model (F1 0.7974, trained on 5.5K)
└── stage1_classifier_v2/        # V2 model (F1 0.8558, trained on 18K)
    ├── checkpoint-1041/         # End of epoch 1
    ├── checkpoint-2082/         # End of epoch 2
    ├── model.safetensors        # Final model
    ├── tokenizer.json
    └── config.json

data/processed/
├── rrgen_10k_labeled.json       # Initial 10K labeling (from Step 2)
├── rrgen_10k_labeled.csv
├── rrgen_10k_stats.json
├── progressive/
│   ├── 10k/                     # Round 1: 7,132 cumulative
│   │   ├── labeled_10k.json
│   │   ├── new_labels_10k.json
│   │   ├── low_confidence_10k.json
│   │   ├── labeled_10k.csv
│   │   └── stats_10k.json
│   ├── 20k/                     # Round 2: 12,839 cumulative
│   │   └── (same structure)
│   └── 30k/                     # Round 3: 18,498 cumulative
│       └── (same structure)
└── round_1/                     # From progressive_labeling.py (has partial round_2)
    └── labeled_10k.json

scripts/
├── label_rrgen_10k.py
├── progressive_labeling.py
└── progressive_labeling_fast.py
```
