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

---

## Quick start — verify every paper number (no GPU, no API keys)

```bash
git clone https://github.com/Fabiha-9876/ReviewAgent.git
cd ReviewAgent/ReviewAgent      # the code lives one level down

# Download the data bundle (18 MB) from Zenodo and extract it here:
#   https://doi.org/10.5281/zenodo.21982774
tar -xzf issuespec-data-bundle-v2.tar.gz   # or download it from the Zenodo DOI below

# Reproduce every numerical claim in the paper (~1 minute)
python3 verify_paper_results.py
```

Each of the eighteen segments prints the recomputed value next to the paper claim. Segments
that need a data file you do not have print `[SKIP]` and name the file rather than crashing.

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
| **Data bundle** (verification, 18 MB, v2.0) | Zenodo DOI [10.5281/zenodo.21982774](https://doi.org/10.5281/zenodo.21982774) |
| *Superseded bundle (pre-correction, do not use)* | [10.5281/zenodo.20320410](https://doi.org/10.5281/zenodo.20320410) |
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
│   ├── verify_paper_results.py 18-segment paper-claim verifier
│   └── run_pipeline.sh         One-command Stage 1-5 orchestrator
├── RRGen_Full_Dataset.csv      Raw 310,031 review-response pairs (46 MB)
├── SETUP_GUIDE.md              Full reproducibility walkthrough
├── ReviewAgent_Detailed_Architecture.md
└── ReviewAgent_Experimental_Design.md
```

> Large artifacts (`models/` ~21 GB, `data/` ~8.8 GB: embeddings, vector DB,
> RLHF rollouts) are not stored in git — they regenerate from the code, and the
> verification bundle on Zenodo covers every reported number.

---

## Headline results

> **Correction, please read before using these numbers.** An earlier version of this work
> reported a +2.36 Likert-point response-quality gain from supplying the IssueSpec. That
> comparison was confounded: its two arms were written by different deterministic template
> composers rather than by one model, so it measured the composer as much as the specification.
> A corrected same-model comparison gives **+0.03 (p = 0.38)**. The superseded response files
> are kept in `data/processed/responses/` and marked; see that folder's README.

| Result | Value | Status |
|---|---|---|
| Stage-1 classifier (Cohen's κ vs 3-rater human gold) | 0.163 → **0.592** | holds |
| Stage-1 on the 307 reviews the classifier never saw | **0.616** | holds |
| Stage-3 template-fill, with vs without type routing | **0.96** vs 0.69 | holds |
| Stage-3 rubric, all 100 specs, LLM judge | **3.94 / 5** | holds |
| Stage-4 IssueSpec-in-RAG quality gain | **+0.03** Likert (p = 0.38) | **withdrawn, was +2.36** |
| Stage-2 knowledge graph vs count-matched flat clustering | flat wins on DB, CH, silhouette | negative |
| SpecCov vs human faithfulness judgement | ρ = 0.15 (p = 0.26) | negative, withdrawn |
| Stage-5 CMDP quality-vs-compliance trade-off | not demonstrated | negative |
| Cross-family replication | 4 LLMs, 3 families | ordering only, not significant at n=14 |
