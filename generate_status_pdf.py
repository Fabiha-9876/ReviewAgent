from fpdf import FPDF


class StatusPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.ln(4)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def status_item(self, number, title, status, description):
        self.set_text_color(30, 30, 30)
        self.set_font("Helvetica", "B", 10)
        self.cell(10, 6, f"{number}.", align="R")
        self.cell(3, 6, "")
        self.set_font("Helvetica", "B", 10)
        self.write(6, title)
        # Status badge
        if status == "DONE":
            self.set_text_color(0, 128, 0)
            self.write(6, f"  [{status}]")
        else:
            self.set_text_color(200, 0, 0)
            self.write(6, f"  [{status}]")
        self.ln(7)
        self.set_text_color(30, 30, 30)
        self.set_font("Helvetica", "", 9)
        self.cell(13, 5, "")
        self.multi_cell(0, 5, description)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        line_h = 6
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 9, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows:
            max_h = 9
            for i, cell in enumerate(row):
                lines = self.multi_cell(col_widths[i] - 4, line_h, cell, split_only=True)
                h = len(lines) * line_h + 3
                if h > max_h:
                    max_h = h
            x_start = self.get_x()
            y_start = self.get_y()
            if y_start + max_h > 270:
                self.add_page()
                y_start = self.get_y()
            if fill:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                x = x_start + sum(col_widths[:i])
                self.set_xy(x, y_start)
                self.rect(x, y_start, col_widths[i], max_h, style="DF")
                self.set_xy(x + 2, y_start + 1.5)
                self.multi_cell(col_widths[i] - 4, line_h, cell)
            self.set_xy(x_start, y_start + max_h)
            fill = not fill
        self.ln(5)


pdf = StatusPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ---- TITLE PAGE ----
pdf.add_page()
pdf.ln(40)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(20, 60, 120)
pdf.multi_cell(0, 12, "ReviewAgent", align="C")
pdf.ln(3)
pdf.set_font("Helvetica", "", 14)
pdf.set_text_color(50, 50, 50)
pdf.multi_cell(0, 7, "Project Status Report\nComplete & Incomplete Items", align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "I", 11)
pdf.set_text_color(80, 80, 80)
pdf.multi_cell(0, 7, "70 files created | 15 items remaining", align="C")
pdf.ln(30)
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(100, 100, 100)
pdf.multi_cell(0, 6, "Prepared by: Fabiha Jalal\nDate: March 2026", align="C")

# ---- COMPLETED: DOCUMENTATION ----
pdf.add_page()
pdf.section_title("COMPLETED ITEMS")

pdf.sub_title("A. Documentation (6 files)")
headers = ["#", "File", "Description"]
rows = [
    ["1", "ReviewAgent_Project_Proposal.md", "Full proposal: problem statement, 5 contributions, 3 RQs, theoretical foundations, 38 references"],
    ["2", "ReviewAgent_Detailed_Architecture.md", "Technical deep-dive: all stages with theory tags, JSON schemas, examples"],
    ["3", "ReviewAgent_Experimental_Design.md", "3 experiments, 7 ablations, statistical tests, sample size justification"],
    ["4", "ReviewAgent_Research_Proposal.pdf", "16-page polished PDF with all 15 sections"],
    ["5", "generate_pdf.py", "PDF generation script using fpdf"],
    ["6", "IMPLEMENTATION_GUIDE.md", "Beginner-friendly guide explaining every file, setup, and usage"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("B. Project Setup (3 files)")
rows = [
    ["7", "pyproject.toml", "30+ dependencies: torch, transformers, networkx, chromadb, trl, fastapi, etc."],
    ["8", ".env.example", "API key template for OpenAI, Anthropic, HuggingFace"],
    ["9", ".gitignore", "Ignores data/, models/, venv/, __pycache__/"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("C. Configuration (7 files)")
rows = [
    ["10", "configs/base.yaml", "Shared: LLM provider, model name, thresholds"],
    ["11", "configs/stage1.yaml", "RoBERTa: 7 labels, training params, confidence threshold 0.7"],
    ["12", "configs/stage2.yaml", "KG node/edge types, HDBSCAN params, priority weights"],
    ["13", "configs/stage3.yaml", "Translation temperature 0.3, HITL min score 3.0"],
    ["14", "configs/stage4b.yaml", "ChromaDB path, 5 RAG sources, 3 refinement iterations"],
    ["15", "configs/stage5.yaml", "LoRA config, KTO/DPO/PPO hyperparams, feedback dimensions"],
    ["16", "configs/experiments.yaml", "All 3 experiments + 7 ablations fully configured"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

# ---- COMPLETED: SOURCE CODE ----
pdf.add_page()
pdf.sub_title("D. Common Module (3 files)")
rows = [
    ["17", "src/common/schemas.py", "8 Pydantic models: ReviewObject, AspectSentiment, ExtractedEntities, IssueCluster, IssueSpec, RubricScores, ComplianceFlags, GeneratedResponse"],
    ["18", "src/common/config.py", "YAML config loader using OmegaConf"],
    ["19", "src/common/llm_client.py", "Unified OpenAI + Anthropic wrapper: generate() and generate_structured()"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("E. Stage 1: Intake (6 files)")
rows = [
    ["20", "src/stage1/classifier.py", "RoBERTa multi-label classifier: train(), predict(), needs_hitl(), save/load"],
    ["21", "src/stage1/aspect_sentiment.py", "LLM-based aspect-sentiment extraction: analyze(), analyze_batch()"],
    ["22", "src/stage1/entity_extractor.py", "Hybrid regex + LLM: devices, OS, versions, screens, features"],
    ["23", "src/stage1/hitl_checkpoint.py", "Confidence flagging, correction recording, retraining data export"],
    ["24", "src/stage1/pipeline.py", "Orchestrator: process() and process_with_hitl()"],
    ["25", "src/stage1/__init__.py", "Module exports"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("F. Stage 2: KG + Clustering (6 files)")
rows = [
    ["26", "src/stage2/kg_builder.py", "NetworkX DiGraph: add_reviews(), aspect nodes, pagerank, export/load"],
    ["27", "src/stage2/clustering.py", "Two-level HDBSCAN: Level 1 by aspect, Level 2 sub-clustering"],
    ["28", "src/stage2/schema_mapper.py", "Maps to IssueCluster: infer type, select reps, merge entities, temporal"],
    ["29", "src/stage2/priority_ranker.py", "Weighted: PageRank + review_count + sentiment + recency"],
    ["30", "src/stage2/pipeline.py", "Orchestrator: KG -> Clustering -> Schema -> Ranking"],
    ["31", "src/stage2/__init__.py", "Module exports"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.add_page()
pdf.sub_title("G. Stage 3: Translation -- Core Contribution (5 files)")
rows = [
    ["32", "src/stage3/taxonomy.py", "5 templates: Zimmermann bug, user story, ISO 25010, Nielsen, compat matrix"],
    ["33", "src/stage3/translator.py", "LLM IssueCluster -> IssueSpec: translate(), prompt building, parsing"],
    ["34", "src/stage3/hitl_checkpoint.py", "5-dim rubric, threshold check, weak dims, regeneration with feedback"],
    ["35", "src/stage3/pipeline.py", "Orchestrator: process() and process_with_hitl() with retry loop"],
    ["36", "src/stage3/__init__.py", "Module exports"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("H. Stage 4b: Response Generation (5 files)")
rows = [
    ["37", "src/stage4b/rag_retriever.py", "ChromaDB: 5 sources, index_source(), retrieve() with top-k"],
    ["38", "src/stage4b/response_generator.py", "Issue-spec-aware: builds context from RAG + spec, generates response"],
    ["39", "src/stage4b/self_refiner.py", "Self-critique: specificity/compliance/empathy, revises 2-3 times"],
    ["40", "src/stage4b/pipeline.py", "Orchestrator: ablation toggles for RAG, spec, refinement"],
    ["41", "src/stage4b/__init__.py", "Module exports"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("I. Stage 5: RLHF (7 files)")
rows = [
    ["42", "src/stage5/feedback_collector.py", "Dual-stream: quality (5 dims) + compliance (4 flags), KTO/DPO/PPO export"],
    ["43", "src/stage5/kto_trainer.py", "KTO with LoRA via TRL: binary good/bad training"],
    ["44", "src/stage5/dpo_trainer.py", "DPO with LoRA via TRL: paired preference training"],
    ["45", "src/stage5/constrained_ppo.py", "Constrained PPO (CMDP): dual reward models, constrained optimization"],
    ["46", "src/stage5/feedback_propagator.py", "Backward: corrections -> Stage 1, rubric -> Stage 3, quality -> Stage 4b"],
    ["47", "src/stage5/pipeline.py", "Auto-selects KTO/DPO/PPO by data volume"],
    ["48", "src/stage5/__init__.py", "Module exports"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

# ---- COMPLETED: EVALUATION ----
pdf.add_page()
pdf.sub_title("J. Evaluation Module (6 files)")
rows = [
    ["49", "src/evaluation/metrics.py", "BLEU, ROUGE-L, BERTScore, completeness ratio, Krippendorff alpha"],
    ["50", "src/evaluation/statistical_tests.py", "Wilcoxon, Friedman, Nemenyi, Bradley-Terry, McNemar, Bonferroni"],
    ["51", "src/evaluation/experiment1.py", "Exp 1: 4 conditions, rubric eval, Wilcoxon comparison, report"],
    ["52", "src/evaluation/experiment2.py", "Exp 2: 4 conditions, auto + human metrics, Friedman + Nemenyi"],
    ["53", "src/evaluation/experiment3.py", "Exp 3: 3 RLHF variants, Bradley-Terry + McNemar, iteration tracking"],
    ["54", "src/evaluation/ablations.py", "A1-A7 runner: full vs ablated comparison, report table"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("K. API (5 files)")
rows = [
    ["55", "api/main.py", "FastAPI app: health check, 4 route groups"],
    ["56", "api/routes/intake.py", "POST /reviews/intake -- process through Stage 1"],
    ["57", "api/routes/issues.py", "GET /issues -- list/retrieve generated issue specs"],
    ["58", "api/routes/responses.py", "POST /responses/generate -- generate response for a review"],
    ["59", "api/routes/feedback.py", "POST /feedback/quality + /feedback/compliance"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("L. Scripts, Tests, Data (7 files)")
rows = [
    ["60", "scripts/run_pipeline.py", "End-to-end: Stage 1 -> 2 -> 3 -> 4b with JSON output"],
    ["61", "scripts/run_experiment.py", "CLI: python3 run_experiment.py [1|2|3|ablations]"],
    ["62", "scripts/download_datasets.py", "Downloads MAALEJ, RRGen, GUZMAN + generates synthetic data"],
    ["63", "tests/test_schemas.py", "All Pydantic models: ReviewObject, RubricScores, IssueSpec, etc."],
    ["64", "tests/test_stage2/test_kg_builder.py", "KG: add_reviews, aspects, pagerank, export/load"],
    ["65", "tests/test_evaluation/test_metrics.py", "Completeness, rubric agg, Wilcoxon, Friedman, McNemar, B-T"],
    ["66", "data/raw/sample_reviews.json", "500 synthetic reviews (balanced across 6 categories)"],
]
pdf.add_table(headers, rows, col_widths=[10, 70, 110])

pdf.sub_title("M. Downloaded Datasets (ALL VERIFIED)")
headers_d = ["#", "Dataset", "Details"]
rows_d = [
    ["67", "data/raw/maalej/maalej_reviews.json", "50,000 reviews from 189 apps, 12.3 MB, all ratings 1-5"],
    ["68", "data/raw/rrgen/rrgen_reviews.json", "310,031 review-response pairs from 58 apps, 144 MB"],
    ["69", "data/raw/rrgen/data/ (23 files)", "Train 280K + Valid 15K + Test 16K + embeddings, 1.7 GB total"],
    ["70", "data/raw/guzman/guzman_reviews.json", "2,062 sentences, 1,040 aspect-sentiment pairs from 8 apps"],
    ["71", "data/raw/guzman/Ground_truth.xlsx", "Source xlsx with annotated reviews (319 KB)"],
    ["72", "data/processed/training_data.json", "Training-ready format: 500 texts + labels + ratings"],
]
pdf.add_table(headers_d, rows_d, col_widths=[10, 70, 110])

# ---- SUMMARY OF COMPLETED ----
pdf.add_page()
pdf.sub_title("Completed Summary")
headers_s = ["Category", "Files", "Status"]
rows_s = [
    ["Documentation", "6", "DONE"],
    ["Project Setup", "3", "DONE"],
    ["Configuration", "7", "DONE"],
    ["Common Module", "3", "DONE"],
    ["Stage 1: Intake", "6", "DONE"],
    ["Stage 2: KG + Clustering", "6", "DONE"],
    ["Stage 3: Translation", "5", "DONE"],
    ["Stage 4b: Response Gen", "5", "DONE"],
    ["Stage 5: RLHF", "7", "DONE"],
    ["Evaluation Module", "6", "DONE"],
    ["API Routes", "5", "DONE"],
    ["Scripts", "3 (pipeline + experiment + download)", "DONE"],
    ["Tests", "3", "DONE"],
    ["Sample + Synthetic Data", "2", "DONE"],
    ["Downloaded Datasets", "6 (MAALEJ + RRGen + GUZMAN)", "DONE - ALL VERIFIED"],
    ["Init files", "17", "DONE"],
    ["TOTAL", "78 content + 17 init = 95", "ALL DONE"],
]
pdf.add_table(headers_s, rows_s, col_widths=[80, 40, 70])

# ---- RECENTLY COMPLETED ----
pdf.add_page()
pdf.section_title("RECENTLY COMPLETED (Previously Incomplete)")

pdf.status_item(1, "Dataset Download Script", "DONE",
    "File: scripts/download_datasets.py. Downloads MAALEJ, RRGen, GUZMAN datasets + generates synthetic "
    "data. Supports individual or batch download. All 3 datasets verified and processed into JSON.")

pdf.status_item(2, "MAALEJ Dataset Download", "DONE",
    "50,000 app reviews from 189 apps downloaded from sealuzh/user_quality GitHub repo. "
    "12.3 MB JSON. Ratings 1-5, avg 106 chars/review. Zero empty entries. Integrity: PASS.")

pdf.status_item(3, "RRGen Dataset Download", "DONE",
    "310,031 review-response pairs from 58 apps downloaded from Google Drive via gdown. "
    "144 MB JSON + 1.7 GB raw data (train 280K, valid 15K, test 16K + embeddings). "
    "Avg review 74 chars, avg response 229 chars. Integrity: PASS.")

pdf.status_item(4, "GUZMAN Dataset Download", "DONE",
    "2,062 annotated sentences from 8 apps (Spotify, Twitter, WhatsApp, etc.) downloaded from "
    "Dabrowski IS-22 repo (same methodology as Guzman & Maalej 2014). 1,040 aspect-sentiment pairs "
    "(292 positive, 638 neutral, 110 negative). 894 unique aspects. Integrity: PASS.")

pdf.status_item(5, "Synthetic Sample Data", "DONE",
    "500 balanced synthetic reviews generated across 6 categories: bug 30%, feature 18%, usability 17%, "
    "performance 14%, praise 12%, compatibility 10%. Training-ready format also generated.")

pdf.status_item(6, "Fine-tune RoBERTa Classifier (Run 1: Auto-Labels)", "DONE",
    "First training run on 9,529 reviews (500 synthetic + 5,000 MAALEJ auto-labeled via keyword "
    "heuristics + 4,029 RRGen auto-labeled). F1 micro: 0.8355, F1 macro: 0.7478. Completed in ~50 min "
    "on MPS. Limitation: auto-labeling is noisy and approximate.")

pdf.status_item(7, "Fine-tune RoBERTa Classifier (Run 2: Real MAALEJ Labels)", "DONE",
    "Retrained on actual human-annotated MAALEJ data (5,008 reviews from Maalej et al. 2016 paper, "
    "downloaded from github.com/mohammadzaeem/classification_of_app_reviews) + 500 synthetic = 5,508 total. "
    "Original labels: Problem Discovery->bug_report (864), Feature Request->feature_request (444), "
    "User Experience->usability (607), Rating->praise (2389), Information Giving/Seeking->other (704). "
    "Synthetic data fills gaps for performance (70) and compatibility (50) categories missing from MAALEJ. "
    "F1 macro improved from 0.7478 to 0.7992 (+6.9%). Model saved to models/stage1_classifier/. "
    "Training log: TRAINING_LOG.md.")

pdf.status_item(8, "MAALEJ Labeled Dataset Downloaded", "DONE",
    "Original Maalej et al. (2016) labeled dataset: app_reviews.csv (5,008 reviews with human annotations) "
    "+ Maalej_Dataset.xlsx (3,691 reviews) + Pan_Dataset.xlsx (1,390 reviews). "
    "Source: github.com/mohammadzaeem/classification_of_app_reviews.")

# ---- INCOMPLETE ITEMS ----
pdf.add_page()
pdf.section_title("INCOMPLETE ITEMS (12 Remaining)")

pdf.body_text(
    "Datasets downloaded, classifier trained on real MAALEJ labels. "
    "RRGen (310K) not used for training because it has no category labels -- used for RAG in Stage 4b. "
    "GUZMAN (2K) not used for training because it has aspect-sentiment annotations, not category labels "
    "-- used for aspect sentiment in Stage 1 and KG edges in Stage 2."
)

pdf.status_item(1, "RAG Index Population", "INCOMPLETE",
    "File needed: scripts/populate_rag.py. Must load 5 RAG sources (RRGen 310K past responses, changelogs, "
    "FAQ, etc.) and index into ChromaDB. Datasets now available. Without this, RAG returns empty. "
    "Priority: HIGH. Effort: 1-2 hours. Blocked by: Nothing.")

pdf.status_item(2, "Constrained PPO Reward Model Inference", "INCOMPLETE",
    "File: src/stage5/constrained_ppo.py. Methods _score_quality() and _score_compliance() return "
    "hardcoded 0.5 instead of actual model inference. PPO training produces meaningless rewards. "
    "Priority: MEDIUM. Effort: 2-3 hours. Blocked by: Needs trained reward models from feedback data.")

pdf.status_item(3, "DPO Training Text Mapping", "INCOMPLETE",
    "File: src/stage5/feedback_collector.py export_dpo_data() returns response IDs as pairs but DPO trainer "
    "needs actual prompt+response texts. The ID-to-text mapping is not wired up. "
    "Priority: MEDIUM. Effort: 1 hour. Blocked by: Nothing.")

pdf.status_item(4, "Gold-Standard Dataset Construction", "INCOMPLETE",
    "Manual research activity: 200-300 review clusters must be scored by 3+ domain experts writing structured "
    "issue specs. This is Contribution 5 in the proposal and primary evaluation artifact for Experiment 1. "
    "Priority: HIGH. Effort: Weeks. Blocked by: Pipeline running on real data to produce clusters.")

pdf.add_page()
pdf.status_item(5, "Unit Tests -- Stage 1", "INCOMPLETE",
    "Directory exists: tests/test_stage1/ (only __init__.py). Need: test_classifier.py, "
    "test_aspect_sentiment.py, test_entity_extractor.py, test_pipeline.py. "
    "Priority: MEDIUM. Effort: 1-2 hours. Blocked by: Nothing.")

pdf.status_item(6, "Unit Tests -- Stage 3", "INCOMPLETE",
    "Directory exists: tests/test_stage3/ (only __init__.py). Need: test_taxonomy.py, "
    "test_translator.py, test_hitl_checkpoint.py. "
    "Priority: MEDIUM. Effort: 1-2 hours. Blocked by: Nothing.")

pdf.status_item(7, "Unit Tests -- Stage 4b", "INCOMPLETE",
    "Directory exists: tests/test_stage4b/ (only __init__.py). Need: test_rag_retriever.py, "
    "test_response_generator.py, test_self_refiner.py. "
    "Priority: MEDIUM. Effort: 1-2 hours. Blocked by: Nothing.")

pdf.status_item(8, "Unit Tests -- Stage 5", "INCOMPLETE",
    "Directory exists: tests/test_stage5/ (only __init__.py). Need: test_feedback_collector.py, "
    "test_feedback_propagator.py. "
    "Priority: MEDIUM. Effort: 1-2 hours. Blocked by: Nothing.")

pdf.status_item(9, "Jupyter Notebooks", "INCOMPLETE",
    "Directory exists: notebooks/ (empty). Need: 01_data_exploration.ipynb, 02_stage1_training.ipynb, "
    "03_kg_visualization.ipynb, 04_experiment_results.ipynb. "
    "Priority: MEDIUM. Effort: 2-3 hours.")

pdf.status_item(10, "Dockerfile", "INCOMPLETE",
    "Files needed: Dockerfile + docker-compose.yaml. Containerize the API for deployment. "
    "Priority: LOW. Effort: 1-2 hours. Blocked by: Nothing.")

pdf.status_item(11, "CI/CD Pipeline", "INCOMPLETE",
    "File needed: .github/workflows/test.yml. Run tests + ruff linting on every push. "
    "Priority: LOW. Effort: 1 hour. Blocked by: Nothing.")

pdf.status_item(12, "Frontend Dashboard for HITL + Multi-language Support", "INCOMPLETE",
    "Web UI (Streamlit/React) for HITL checkpoints + XLM-RoBERTa for multilingual reviews. "
    "Priority: LOW. Effort: Days. Blocked by: Nothing.")

# ---- PRIORITY ORDER ----
pdf.add_page()
pdf.section_title("Recommended Priority Order")

headers_p = ["Priority", "Item", "Why First", "Effort", "Blocked By"]
rows_p = [
    ["1", "RAG index population (#1)", "Stage 4b generic without RAG", "1-2 hrs", "Nothing"],
    ["2", "Fix DPO text mapping (#3)", "Quick fix, unblocks RLHF", "1 hr", "Nothing"],
    ["3", "Fix PPO reward placeholders (#2)", "Unblocks Constrained PPO", "2-3 hrs", "Feedback data"],
    ["4", "Unit tests (#5-8)", "Catch bugs before experiments", "4-6 hrs", "Nothing"],
    ["5", "Jupyter notebooks (#9)", "Needed for exploration + paper", "2-3 hrs", "Nothing"],
    ["6", "Gold-standard dataset (#4)", "Contribution 5, needed for Exp 1", "Weeks", "Real clusters"],
    ["7-12", "Docker, CI/CD, frontend, multilingual", "Nice-to-have, not blocking", "Days", "Nothing"],
]
pdf.add_table(headers_p, rows_p, col_widths=[18, 52, 52, 28, 40])

# ---- FINAL SUMMARY ----
pdf.ln(10)
pdf.sub_title("Final Summary")

headers_f = ["Metric", "Count"]
rows_f = [
    ["Total files created", "97+ (source + configs + data + model)"],
    ["Source code files", "45 (non-init Python files)"],
    ["Config files", "7"],
    ["Documentation files", "7 (+ TRAINING_LOG.md)"],
    ["Downloaded datasets", "3 (MAALEJ 50K + RRGen 310K + GUZMAN 2K)"],
    ["MAALEJ labeled dataset", "5,008 human-annotated reviews (actual paper data)"],
    ["Total data points available", "362,593"],
    ["Trained model", "RoBERTa classifier, F1 macro 0.7992, 476 MB"],
    ["Test files", "3 (+ 4 more needed)"],
    ["Originally incomplete", "15"],
    ["Completed so far", "8 (datasets + classifier training x2 + labeled data)"],
    ["Currently incomplete", "12"],
    ["High priority remaining", "1 (RAG index)"],
    ["Medium priority remaining", "7"],
    ["Low priority remaining", "3"],
]
pdf.add_table(headers_f, rows_f, col_widths=[80, 110])

# Save
output_path = "/Users/fabihajalal/Desktop/Review Agent/ReviewAgent_Status_Report.pdf"
pdf.output(output_path)
print(f"PDF saved to: {output_path}")
