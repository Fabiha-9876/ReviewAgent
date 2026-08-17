# IST submission package

Elsevier *Information and Software Technology* (ISSN 0950-5849) version of the IssueSpec
paper, converted from the CIKM 2026 `acmart` source in `../IssueSpec/main.tex`.

## Files

| File | Purpose |
|---|---|
| `main_ist.tex` | Manuscript, `elsarticle` single-column preprint (Your Paper Your Way) |
| `main_ist.pdf` | Compiled PDF, 36 pages incl. references and appendix |
| `highlights.txt` | Standalone Highlights file, 5 bullets, each under 85 characters |
| `references.bib` | Unchanged from the CIKM source, 111 entries |
| `elsarticle.cls`, `elsarticle-num.bst` | Generated locally from CTAN `elsarticle.dtx` |
| `fig*.png`, `fig_architecture.pdf` | Figures, copied unchanged |

## Build

```bash
pdflatex main_ist && bibtex main_ist && pdflatex main_ist && pdflatex main_ist
```

Compiles with zero errors, zero undefined citations, zero undefined references.

## What changed from the CIKM version

- `acmart[sigconf]` replaced by `elsarticle[preprint,12pt,authoryear]`, single column.
- Abstract rewritten as an IST **structured abstract** with the five mandatory headings
  (Context, Objective, Method, Results, Conclusions). IST does not process papers without it.
- CCS concepts and CCSXML block dropped, Elsevier uses free keywords via `\begin{keyword}`.
- ACM `\keywords` replaced by `\begin{keyword} ... \sep ...`.
- `figure*` / `table*` converted to `figure` / `table` for the single-column layout.
- Bibliography style `ACM-Reference-Format` to `elsarticle-num`.
- `\begin{acks}` replaced by a plain `Acknowledgements` section.
- Added the sections Elsevier requires at submission: CRediT authorship contribution
  statement, Declaration of competing interest, Data availability.
- GenAI disclosure kept, reworded from CIKM policy to Elsevier policy.
- Line numbers enabled (`lineno`), standard for Elsevier review copies.
- Author block de-anonymized in structure. IST is single blind, so real names go in.

## Before you submit, still to do

1. Confirm the CRediT roles. Six authors are filled in with a first guess at each role,
   the middle three and the three supervisors especially should be checked.
2. Confirm Fabiha Jalal as the single corresponding author. The submission sheet marked
   all six, which Elsevier's system treats as a form artifact rather than a real intent.
3. Decide the Data availability wording, the current text points at the Zenodo DOI,
   the Hugging Face model, and the RRGen upstream terms.
4. Add a cover letter, IST does not require one but it helps for a framing-heavy paper.
5. Confirm the venue framing in the body text. Any remaining phrasing aimed at a
   conference audience should read as a journal contribution.
6. Consider whether the appendix stays inline or moves to supplementary material.
