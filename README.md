# IssueSpec — A Framework for Structured Review-to-Issue Translation

Code, data, and models for the CIKM 2026 paper *IssueSpec: A Framework for
Structured Review-to-Issue Translation*.

IssueSpec is a five-stage pipeline that converts noisy app-store reviews into
typed, developer-routable issue specifications, then uses them to generate
aligned, compliant developer responses:

1. **Intake & Classification** — RoBERTa multi-label classifier + verified-anchor correction
2. **Three-Layer Knowledge Graph** — aspect-grounded clustering + PageRank prioritization
3. **Review-to-Issue Translation** — LLM agent fills five standards-body templates (validated by SpecCov)
4. **Spec-aware RAG** — response generation over five fixed sources
5. **CMDP-grounded RLHF** — constrained alignment (KTO → DPO → Constrained PPO)

The compiled paper is in [`paper/IssueSpec/IssueSpec.pdf`](paper/IssueSpec/IssueSpec.pdf).

---

## Quick start — verify every paper number (no GPU, no API keys)

```bash
git clone https://github.com/Fabiha-9876/ReviewAgent.git
cd ReviewAgent

# Download the data bundle (10 MB) from Zenodo and extract it here:
#   https://doi.org/10.5281/zenodo.20320410
tar -xzf issuespec-data-bundle.tar.gz

# Reproduce every numerical claim in the paper (~1 minute)
python3 verify_paper_results.py
```

Each of the ten segments prints the recomputed value next to the paper claim.

## Run the full pipeline

```bash
./run_pipeline.sh            # all five stages, in order
./run_pipeline.sh 2          # only Stage 2
./run_pipeline.sh verify     # just re-verify paper numbers
```

See [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for full setup, prerequisites, and
stage-by-stage instructions.

---

## Released artifacts

| Artifact | Location |
|---|---|
| **Code + paper** | this repository |
| **Data bundle** (verification, 10 MB) | Zenodo DOI [10.5281/zenodo.20320410](https://doi.org/10.5281/zenodo.20320410) |
| **V5 classifier** (κ = 0.592) | Hugging Face [`Fabiha9876/issuespec-v5-classifier`](https://huggingface.co/Fabiha9876/issuespec-v5-classifier) |
| **Raw RRGen data** | `RRGen_Full_Dataset.csv` (in this repo) + Gao et al., ASE 2019 |

```python
# Load the V5 classifier directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok   = AutoTokenizer.from_pretrained("Fabiha9876/issuespec-v5-classifier")
model = AutoModelForSequenceClassification.from_pretrained("Fabiha9876/issuespec-v5-classifier")
```

---

## Repository layout

```
.
├── ReviewAgent/
│   ├── paper/IssueSpec/        Final paper (main.tex, IssueSpec.pdf, figures)
│   ├── scripts/                ~90 pipeline scripts (Stage 1-5, ablations, scorers)
│   ├── src/                    Core library modules
│   ├── api/ · configs/         API wrappers, YAML configs
│   ├── tests/                  Unit tests
│   ├── verify_paper_results.py 10-segment paper-claim verifier
│   └── run_pipeline.sh         One-command Stage 1-5 orchestrator
├── RRGen_Full_Dataset.csv      Raw 310,031 review-response pairs (46 MB)
├── SETUP_GUIDE.md              Full reproducibility walkthrough
├── ReviewAgent_Detailed_Architecture.md
└── ReviewAgent_Experimental_Design.md
```

> Large artifacts (`models/` ~21 GB, `data/` ~8.8 GB: embeddings, vector DB,
> RLHF rollouts) are not stored in git — they regenerate from the code, and the
> 10 MB verification bundle on Zenodo covers every reported number.

---

## Headline results

| Result | Value |
|---|---|
| Stage-1 classifier (Cohen's κ vs expert gold) | 0.16 → **0.59** |
| Stage-3 template-fill (vs human GitHub issues) | **0.96** vs 0.53 |
| Stage-3 IssueSpec rubric | **3.89 / 5** |
| Stage-4 IssueSpec-in-RAG quality gain | **+2.36** Likert (p < 0.001) |
| Stage-5 constrained-proxy BLEU-1 over SFT | **+52%** |
| Cross-family replication | 4 LLMs, 3 families (rank order preserved) |
