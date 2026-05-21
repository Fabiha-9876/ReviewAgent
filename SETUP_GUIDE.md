# IssueSpec / ReviewAgent — Setup & Reproducibility Guide

This document walks a new collaborator (or reviewer) through the full project
end-to-end: cloning the code, downloading data and models, running each pipeline
stage, and reproducing every numerical claim in the CIKM 2026 paper.

---

## 1. What is this project?

**Paper:** *IssueSpec: A Framework for Structured Review-to-Issue Translation*
(CIKM 2026 submission).

**One-sentence summary:** A five-stage pipeline that converts noisy app-store
reviews into typed, developer-routable GitHub-issue-quality specifications via
knowledge-graph clustering, LLM-based template-filled IR generation, and
CMDP-grounded RLHF response alignment.

**Five stages:**
1. **Intake, classification, aspect extraction** (RoBERTa V5 + cleanlab)
2. **Three-layer knowledge graph** (aspect → cluster → PageRank prioritize)
3. **Review-to-issue translation** (LLM agent fills 5 standards-body templates)
4. **RAG response generation** (uses validated IssueSpec as first-class source)
5. **CMDP-grounded RLHF** (KTO → DPO → constrained-proxy → Lagrangian PPO)

---

## 2. Folder layout (what's where)

After everything is set up, the project root looks like this:

```
Review Agent/
├── ReviewAgent/                ← main code repository (cloned from GitHub)
│   ├── paper/                  ← paper LaTeX + figures
│   │   └── IssueSpec/          ← final paper: main.tex + figures + IssueSpec.pdf
│   ├── scripts/                ← ~80 pipeline scripts (Stage 1–5 + ablations)
│   ├── src/                    ← core library
│   ├── verify_paper_results.py ← reproduces every numerical claim
│   ├── data/                   ← (downloaded separately — see §4)
│   │   ├── raw/                ← RRGen 310,031 reviews CSV
│   │   └── processed/          ← anchor, gold, ratings, RLHF metrics
│   └── models/                 ← (downloaded separately — see §4)
│       ├── stage1_classifier_v5/   ← headline V5 RoBERTa (κ = 0.59)
│       ├── anchor_roberta/         ← anchor classifier
│       └── ...
├── RRGen/                      ← original Gao et al. 2019 RRGen code (reference)
├── RRGen_Annotation/           ← annotation tooling
├── RRGen_Full_Dataset.csv      ← raw 310,031 review-response pairs (46 MB)
├── ReviewAgent_Detailed_Architecture.md
├── ReviewAgent_Experimental_Design.md
└── SETUP_GUIDE.md              ← (this file)
```

---

## 3. Prerequisites

| Tool | Minimum version | Why |
|---|---|---|
| **Python** | 3.10+ | Project tooling |
| **Git** | 2.x | Clone repo |
| **pip** | 22+ | Install dependencies |
| **LaTeX** (optional) | TeX Live 2023+ | Compile the paper PDF |
| **CUDA GPU** (optional) | 12.x, 16 GB+ VRAM | Re-train V5 / run RLHF |
| **Docker** (optional) | 24+ | Pinned-environment reproducibility |

Recommended for full pipeline retraining:
- Single GPU machine (e.g., RTX 3090 / A100)
- ~50 GB free disk (project + models)
- Anthropic API key (Claude Opus 4.7 — for Stage 3 generation)
- Optional: Hugging Face Inference Endpoint or local Llama-3.3-70B + Qwen2.5

---

## 4. Where to get the code, data, and models

### 4.1 Code (GitHub)

```bash
cd ~/Desktop
git clone https://github.com/Fabiha-9876/ReviewAgent.git "Review Agent/ReviewAgent"
cd "Review Agent/ReviewAgent"
```

The repo includes:
- All Python scripts (`scripts/`, `src/`)
- The final paper LaTeX (`paper/IssueSpec/main.tex` + `IssueSpec.pdf`)
- Figure-generation scripts and PNGs
- `verify_paper_results.py` (reproduces every numerical claim)
- `Dockerfile` (optional pinned environment)
- Documentation (`ANNOTATION_PROTOCOL.md`, `IMPLEMENTATION_GUIDE.md`)

The repo **excludes** large files: `data/`, `models/`, `logs/`. These are
released separately (see §4.2–§4.4).

### 4.2 Raw dataset (RRGen)

The raw 310,031 review-response pairs come from:
- **Original release:** Gao et al., *"Automating App Review Response Generation"*,
  ASE 2019. See https://github.com/CuriousG102/rrgen.
- **Pre-processed CSV** (single 46 MB file): `RRGen_Full_Dataset.csv` — placed
  one level above the cloned `ReviewAgent/` folder (so the relative path
  `../RRGen_Full_Dataset.csv` works from `ReviewAgent/`).

Place it at: `Review Agent/RRGen_Full_Dataset.csv`

### 4.3 Processed data (~2 GB)

The processed JSON files (verified anchor, 100-cluster benchmark, 400-row
human ratings, RLHF head-to-head results, cluster quality metrics, etc.) are
released as a single bundle on Zenodo.

**Download:** `issuespec-data-bundle.tar.gz` (~10 MB)
- **Zenodo DOI:** [10.5281/zenodo.20320410](https://doi.org/10.5281/zenodo.20320410)

**Extract into:**
```bash
cd "Review Agent/ReviewAgent"
# download from the Zenodo record above, then:
tar -xzf ~/Downloads/issuespec-data-bundle.tar.gz
# This creates ./data/processed/ with all the JSON artifacts
```

This bundle contains everything `verify_paper_results.py` needs to reproduce
every numerical claim in the paper (no GPU or API keys required).

### 4.4 Model checkpoints

The headline V5 production classifier (the κ = 0.592 model used throughout the
paper) is hosted on the Hugging Face Hub:

| Model | What | HF path |
|---|---|---|
| **V5 (headline)** | Production classifier (κ = 0.592) | [`Fabiha9876/issuespec-v5-classifier`](https://huggingface.co/Fabiha9876/issuespec-v5-classifier) |

**Download + load V5:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok   = AutoTokenizer.from_pretrained("Fabiha9876/issuespec-v5-classifier")
model = AutoModelForSequenceClassification.from_pretrained("Fabiha9876/issuespec-v5-classifier")
```

The intermediate V1–V4 checkpoints and the anchor RoBERTa head (~21 GB total)
are available on request; V5 alone suffices to reproduce all Stage-1 results
in the paper. To re-train any version from scratch, see §7.2.

For most reproductions you only need **V5** (the headline production model).

---

## 5. Environment setup

### 5.1 Install Python dependencies

```bash
cd "Review Agent/ReviewAgent"
pip install -e .          # uses pyproject.toml
# or:
pip install -r requirements.txt    # if a flat requirements file is preferred
```

Key packages: `torch`, `transformers`, `sentence-transformers`, `umap-learn`,
`hdbscan`, `cleanlab`, `networkx`, `chromadb`, `scikit-learn`, `pandas`.

### 5.2 (Optional) Use Docker for pinned environment

```bash
cd "Review Agent/ReviewAgent"
docker build -t issuespec .
docker run --gpus all -v "$(pwd):/work" -it issuespec bash
```

### 5.3 Set up API keys (only for Stage 3 LLM calls)

Copy `.env.example` to `.env` and fill in:
```
ANTHROPIC_API_KEY=sk-ant-...           # for Claude Opus Stage 3
HF_TOKEN=hf_...                         # for Llama-3.3-70B via Groq, Qwen models
GROQ_API_KEY=gsk_...                    # for Llama-3.3-70B (cross-family control)
```

If you skip these, the saved outputs (in `data/processed/issue_specs/`) still
let `verify_paper_results.py` reproduce every paper number.

---

## 6. Quick start — verify the paper without re-running training

This is the fastest path. Takes < 1 minute. No GPU, no API keys needed.

```bash
cd "Review Agent/ReviewAgent"

# 1. Download the data bundle (10 MB) from Zenodo: 10.5281/zenodo.20320410
#    https://doi.org/10.5281/zenodo.20320410
tar -xzf ~/Downloads/issuespec-data-bundle.tar.gz

# 2. Run all 10 verification segments
python3 verify_paper_results.py

# Or run a single segment:
python3 verify_paper_results.py 5    # Stage 3 SpecCov scorer only
python3 verify_paper_results.py 9    # Stage 5 RLHF policies only
```

Expected output: every numerical claim from the paper printed alongside the
recomputed value, all matching within rounding.

---

## 7. Full pipeline reproduction (from raw RRGen to paper results)

This re-runs the entire five-stage pipeline. Allow ~6 hours on a single
GPU-equipped workstation.

**One-command orchestrator** (`run_pipeline.sh`) runs the stages in order:
```bash
./run_pipeline.sh            # all five stages
./run_pipeline.sh 2          # only Stage 2
./run_pipeline.sh 2 3 4      # Stages 2-4
./run_pipeline.sh verify     # just re-verify paper numbers (needs data bundle)
```
The script does pre-flight checks and prints the expected headline result after
each stage. The per-stage commands below are what it runs internally.

### 7.1 Place the raw dataset
Put `RRGen_Full_Dataset.csv` at `Review Agent/RRGen_Full_Dataset.csv`.

> Each script carries a docstring describing its exact inputs and outputs.
> Run scripts from the repository root. Stage 1, 3, and 5 need an LLM (Gemini/
> Qwen/Claude) and/or a GPU; Stage 2 is CPU-bound.

### 7.2 Stage 1 — Intake, classification, aspect extraction

```bash
cd "Review Agent/ReviewAgent"

# V2 labels: LLM-label the keyword-filtered RRGen corpus
python3 scripts/llm_label_rrgen.py

# Find suspected label errors against the verified anchor (cleanlab)
python3 scripts/cleanlab_find_label_issues.py

# Second-pass correction using the anchor RoBERTa head
python3 scripts/correct_rrgen_v2.py

# Ingest human-verified labels into the training set
python3 scripts/ingest_verified_labels.py

# Compatibility-class augmentation (200 synthetic + 100 mined)
python3 scripts/build_compat_data.py

# Train + cross-validate the V5 RoBERTa classifier
python3 scripts/kfold_classifier.py

# Cross-protocol generalization check on Maalej's labels
python3 scripts/_eval_v5_on_maalej.py
```

Expected: V5 reaches κ ≈ 0.59 on the 490-review expert gold (Table 8).

### 7.3 Stage 2 — Three-layer knowledge graph

```bash
# Aspect extraction (local Qwen2.5-3B, no API key needed)
python3 scripts/extract_aspects_local_llm.py

# Aspect-grounded sub-clustering: UMAP + class-specific HDBSCAN → 605 clusters
python3 scripts/cluster_phase1b_umap_hdbscan.py

# Cluster quality metrics (Table 11: DB, CH, silhouette)
python3 scripts/compute_cluster_quality_metrics.py

# Count-controlled ablation (A1b) and LLM-judge purity audit
python3 scripts/ablation_a1b_fine_flat_vs_kg.py
python3 scripts/audit_hierarchical_cluster_purity_llm.py
```

### 7.4 Stage 3 — LLM-based IR generation

Uses a local Qwen2.5-3B for the cross-LLM comparison; the headline Claude run
needs `ANTHROPIC_API_KEY`.

```bash
# Generate IssueSpecs across LLMs (taxonomy / free-form / raw conditions)
python3 scripts/multi_llm_stage3_comparison.py

# SpecCov extractive-coverage faithfulness scorer
python3 scripts/speccov.py \
    --specs data/processed/issue_specs/specs_with_taxonomy.json \
    --clusters data/processed/issue_specs/sample_100_clusters.json \
    --out data/processed/speccov_scores.json

# 5-dimension rubric (Qwen-as-judge) + GitHub-issue baseline
python3 scripts/_qwen_judge_5dim_rubric.py
python3 scripts/mine_github_issues.py
```

### 7.5 Stage 4 — RAG response generation

```bash
# Build the 15,100-document ChromaDB index over the five sources
python3 scripts/populate_rag.py

# Generate responses for the full system and the no-spec ablation
python3 scripts/generate_reviewagent_full.py
python3 scripts/generate_reviewagent_no_spec.py

# A5 no-RAG ablation + agentic-vs-vanilla feasibility study
python3 scripts/run_ablation_a5_no_rag.py
python3 scripts/_agentic_vs_vanilla_rag.py      # n = 10, max 2 iterations
```

Human evaluation: open the rating workbooks under `human_work/` and follow
`ANNOTATION_PROTOCOL.md`.

### 7.6 Stage 5 — CMDP-grounded RLHF

Requires GPU. distilGPT2 proof-of-concept, ~30 minutes.

```bash
# Train the policies (SFT-base, KTO, DPO)
python3 scripts/run_rlhf_proof_of_concept.py

# Constrained-PPO variants
python3 scripts/run_constrained_ppo_proxy.py
python3 scripts/run_lagrangian_constrained_ppo.py

# Head-to-head evaluation + rubric scoring
python3 scripts/run_rlhf_head_to_head.py
python3 scripts/score_rlhf_policies_with_rubric.py
```

Expected (§5.3): constrained-proxy BLEU-1 0.137 (+52% over SFT-base 0.090).

### 7.7 Compile the paper

```bash
cd paper/IssueSpec
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# IssueSpec.pdf — 14 pages, matches the submitted PDF
```

---

## 8. Verifying everything matches the paper

The `verify_paper_results.py` script re-runs the computation behind ten
families of paper claims and prints each value side-by-side with the paper
claim. Ten segments cover:

| # | Segment | Verifies |
|---|---|---|
| 1 | Corpus + provenance | 215,583 working corpus, 5,230 anchor, 79.49/18.08/2.43% |
| 2 | Stage 1 κ progression | Table 8: V2 0.163, cleanlab 0.333, V5 0.592 |
| 3 | Stage 2 cluster quality | Table 11: DB, CH, 5.4× and 1.9× ratios |
| 4 | A1b ablation | §5.5 count-matched flat-605 vs KG-605 |
| 5 | SpecCov scorer | §4.4 4.16/3.33/5.00/4.00 (all 4 conditions) |
| 6 | Stage 4 human eval | Table 10: 2.31/2.98/2.26/4.62; +2.36 paired Δ |
| 7 | A5 no-RAG ablation | §5.5 ΔBLEU/ROUGE/BERTScore |
| 8 | Agentic vs vanilla | §5.2 0.58→0.70, 0%→60% citation |
| 9 | Stage 5 RLHF | §5.3 five policies + +52% BLEU-1 gain |
| 10 | Inter-rater α | Table 4: α = 0.451 (99 reviews) |

Run all at once:
```bash
python3 verify_paper_results.py
```

Or a single segment:
```bash
python3 verify_paper_results.py 5      # SpecCov only
python3 verify_paper_results.py 9      # RLHF only
```

---

## 9. Folder-by-folder reference

| Folder | What lives here |
|---|---|
| `paper/IssueSpec/` | Final paper LaTeX, figures, compiled PDF |
| `paper/build_v2/` | Figure-generation Python scripts + intermediate PNGs |
| `paper/experiments/` | Pre-submission experiment scripts + curation rubric + labmate handoff bundle |
| `scripts/` | All pipeline scripts (Stage 1–5, ablations, scorers) |
| `scripts/speccov.py` | Standalone SpecCov faithfulness scorer |
| `scripts/run_rlhf_head_to_head.py` | Stage 5 head-to-head policy evaluation |
| `scripts/audit_hierarchical_cluster_purity_llm.py` | LLM-as-judge cluster purity |
| `scripts/compute_3rater_krippendorff.py` | Inter-rater agreement on 99 reviews |
| `scripts/_agentic_vs_vanilla_rag.py` | Agentic-RAG vs vanilla feasibility study |
| `src/` | Core library modules (taxonomies, schema, utils) |
| `api/` | Optional API wrappers |
| `configs/` | YAML configuration files |
| `notebooks/` | Exploratory Jupyter notebooks |
| `annotator_materials/` | Templates and instructions for human-in-the-loop annotation |
| `human_work/` | Lead-author rating spreadsheets (anonymized) |
| `tests/` | Unit tests |
| `data/raw/` | Place `RRGen_Full_Dataset.csv` here (or symlink from parent dir) |
| `data/processed/` | All processed JSON outputs (released bundle) |
| `models/` | V1–V5 + anchor classifier checkpoints (Hugging Face download) |
| `verify_paper_results.py` | Reproduces every numerical claim in the paper |
| `verify_paper_results.ipynb` | Notebook version for cell-by-cell exploration |

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: data/processed/...` | Data bundle not extracted | Re-download and extract (§4.3) |
| `OSError: cannot find Fabiha9876/issuespec-v5-classifier` | HF download failed | Check internet / try `HF_TOKEN` |
| LaTeX compile fails with `\Bbbk` redefined | Old `amssymb` conflict | Already handled by `\let\Bbbk\relax` in preamble |
| Out of GPU memory during V5 training | < 16 GB VRAM | Lower batch size in `configs/stage1.yaml` |
| `verify_paper_results.py` reports mismatch | Stale data files | Re-download data bundle |
| Stage 3 LLM call fails | `ANTHROPIC_API_KEY` missing | Set in `.env`; or skip (use saved outputs) |

---

## 11. Data

Data: the review text comes from the RRGen corpus (Gao et al., "Automating App
Review Response Generation", ASE 2019) and remains under its original terms;
refer to the original authors. The other files are derived artifacts (labels,
IssueSpecs, results, metrics).

This work is under review at CIKM 2026 and is not yet published.

---

## 12. Where to ask for help

- **Code / scripts:** open an issue at https://github.com/Fabiha-9876/ReviewAgent/issues
- **Data / models:** check `RELEASE.md` in the cloned repo
- **Paper claims:** the `verify_paper_results.py` script returns the
  ground-truth recomputed value for every numerical claim.
