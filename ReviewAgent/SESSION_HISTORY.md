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

## Current State (as of March 31, 2026)

### What's Complete
- All documentation (proposal, architecture, experiments, PDF, guide, training log)
- Full 5-stage pipeline implementation (45+ Python source files)
- 7 config files, 4 scripts, 3 test files
- All 3 datasets downloaded and verified (362,593 total data points)
- MAALEJ labeled dataset (5,008 human-annotated reviews)
- RoBERTa classifier trained on real labels (F1 macro 0.7992)
- RAG index populated (15,100 documents in ChromaDB)
- DPO text mapping fixed
- Constrained PPO scoring fixed with realistic compliance
- Synthetic data validation form sent to advisor

### What's Remaining (9 items)
1. Unit Tests — Stage 1 (classifier, aspect_sentiment, entity_extractor, pipeline)
2. Unit Tests — Stage 3 (taxonomy, translator, hitl_checkpoint)
3. Unit Tests — Stage 4b (rag_retriever, response_generator, self_refiner)
4. Unit Tests — Stage 5 (feedback_collector, feedback_propagator)
5. Jupyter Notebooks (data exploration, KG viz, experiment results)
6. Gold-Standard Dataset Construction (200-300 clusters, 3+ experts — weeks)
7. Dockerfile
8. CI/CD Pipeline
9. Frontend Dashboard + Multi-language

### Pending
- Synthetic data expert validation feedback from advisor
- If synthetic data needs revision → regenerate → retrain classifier

### Key Technical Decisions Made
1. Used RoBERTa (not BERT) for classification — better performance on informal text
2. Used HDBSCAN (not K-means) for clustering — handles variable cluster sizes, no need to predefine K
3. Used ChromaDB (not FAISS) for RAG — persistent storage, easy to add/query
4. Used sentence-transformers all-MiniLM-L6-v2 for embeddings — fast, good quality
5. Custom MultiLabelCollator needed for HuggingFace Trainer — labels must be float32 not long
6. Compliance scoring uses min-weighted combination — single severe violation tanks the score
7. DPO pairs grouped by issue_id — ensures chosen/rejected compare responses to same prompt
8. Synthetic data fills MAALEJ gaps for performance and compatibility categories only
