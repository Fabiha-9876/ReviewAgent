# Annotation Protocol — ReviewAgent Project

**Version:** 1.0
**Date:** April 29, 2026
**Author:** Fabiha Jalal
**Advisor:** Hasan Mahmud

---

## 1. Purpose

This protocol defines the procedure, materials, and reliability measures for human verification of LLM-assigned labels in the ReviewAgent project. It ensures that the gold-standard subset used to validate the classifier and correction pipeline is reproducible and meets the inter-annotator agreement standards expected by ICSE / FSE / ASE-grade publications.

## 2. Background

The project trains a multi-label review classifier across seven categories: `bug_report`, `feature_request`, `performance`, `usability`, `compatibility`, `praise`, `other`. The full RRGen corpus (~310K reviews) was machine-labeled by an iteratively-trained RoBERTa classifier (V2). Direct measurement on a 5,230-review sample showed an LLM error rate of approximately 25% on praise predictions, motivating a correction pipeline (cleanlab + verified-anchor) and a multi-annotator gold-standard build.

## 3. Sample Sizes — SOTA Reference

Our target sample size and annotator count are aligned with established work in app-review classification:

| paper | reviews | annotators |
|---|---|---|
| Maalej et al. (2016) | 4,400 | 2 |
| Guzman & Maalej (2014) | 2,062 | 2 |
| Chen et al. (2014) AR-Miner | 1,000 | 2 |
| Villarroel et al. (2016) CLAP | 1,390 | 2 |
| Di Sorbo et al. (2016) SURF | 4,000+ | 3 |
| **ICSE/FSE/ASE convention** | **500–2,000** | **2–3** |

**ReviewAgent target:** 500–1,000 reviews verified by 3 annotators.

## 4. Annotators

- **Number:** 3 independent annotators, recruited from the department.
- **Profile:** Graduate students with English fluency and basic familiarity with mobile app reviews.
- **Compensation:** Authorship credit (co-author or acknowledgment, depending on contribution scale) per advisor's directive.
- **Training:** Each annotator completes a 20-review calibration round before starting the main task. Disagreements in the calibration round are reviewed jointly to align category understanding.

## 5. Materials

Each annotator receives:

- **Task spreadsheet** (one per annotator): `[id, text, rating, app_id, llm_predicted_label, correct_yn, correct_label_if_no, comments]`
- **Category definitions** (Section 6 below)
- **Decision rules** (Section 7 below)
- **Calibration set** (20 reviews, shared across all annotators)

Annotation interface: Excel/Numbers spreadsheet. Annotators do not see each other's labels during the main task.

## 6. Category Definitions

| label | definition |
|---|---|
| `bug_report` | Crashes, errors, broken features, malfunctions not tied to a specific device |
| `feature_request` | Explicit request for new functionality or improvement of existing feature |
| `performance` | Speed, battery, memory, lag, loading times, responsiveness |
| `usability` | Confusing UI, poor navigation, hard-to-find features (without functional break) |
| `compatibility` | Device-specific or OS-specific issues (e.g., "crashes only on Samsung S22", "broken since Android 13") |
| `praise` | Positive feedback, compliments, satisfaction |
| `other` | Information-giving, off-topic, or doesn't fit any above category |

## 7. Decision Rules

To minimize boundary ambiguity:

1. **`slow` / `lag` is performance, NOT bug_report** — these are degraded-but-functional, not broken.
2. **`crash on my Samsung` is compatibility** (device-specific). `crash` without device context is bug_report.
3. **`would be nice if X` or `please add X` is feature_request**, even if phrased as a complaint.
4. **`hard to find the X button` is usability**, not bug_report.
5. **Multi-aspect reviews:** assign the *primary* category if one dominates; otherwise mark all that apply (multi-label).
6. **Spam / non-English / nonsense:** label as `other`.
7. **When in doubt:** mark `correct_yn = N`, write the alternative in `correct_label_if_no`, and add a one-line comment explaining the ambiguity.

## 8. Reliability Measures

After all 3 annotators complete the task:

### 8.1 Primary metric — Krippendorff's α
- Computed across all 3 annotators on the full set
- Treats labels as nominal categories
- **Acceptance threshold:** α ≥ 0.67 (acceptable), α ≥ 0.80 (strong)

### 8.2 Secondary metric — Fleiss' κ
- Reports per-category Fleiss' κ to surface specific weak boundaries
- Useful for diagnosing which categories need clearer definitions

### 8.3 Per-pair Cohen's κ
- All three pairwise (A↔B, A↔C, B↔C) Cohen's κ values reported
- Helps detect annotator-specific bias

### 8.4 Agreement on machine-flagged correction set
- Separately compute α on the subset where the cleanlab correction pipeline disagreed with the original LLM label
- Establishes whether annotators agree with the corrections (validating the noise-modeling claim)

## 9. Disagreement Resolution

1. **Agreement** (≥2 of 3 annotators agree): adopt the majority label.
2. **3-way disagreement:** flagged for adjudication by Fabiha Jalal (lead) with advisor consultation.
3. **Resolved label** stored in a separate column for traceability; original 3 annotator labels preserved.

## 10. Outputs and Reporting

For the paper, we will report:

| artifact | description |
|---|---|
| `gold_standard_500.json` | Final adjudicated labels on 500–1,000 reviews |
| `interannotator_agreement.json` | All α, κ, pairwise scores |
| `per_category_disagreement.csv` | Categories ranked by disagreement rate |
| `correction_validation.json` | Annotator agreement on the cleanlab-flagged subset |

Reported in the paper's methodology section as:
> "We collected 3-annotator labels on N=[500-1000] reviews drawn from the LLM-labeled RRGen corpus. Inter-annotator agreement was Krippendorff's α=[X] (acceptable threshold ≥ 0.67); per-category Fleiss' κ ranged from [Y] to [Z]. The gold-standard set was used to evaluate the noise-modeling pipeline at [W]% precision."

## 11. Volunteer Tasks Summary

| task | scope | est. time | output |
|---|---|---|---|
| **Task 1 — Synthetic data validation** | 150 generated reviews × Y/N realism + category | 45–60 min | Confirmation that synthetic data is realistic |
| **Task 2 — Real-review verification** | 500–1,000 RRGen reviews × Y/N + correct label | 2–3 hours | Gold-standard for classifier eval |
| **Task 3 — Issue-spec scoring** *(later phase)* | Score generated specs on 5-dim rubric | TBD | Stage 3 evaluation data |

## 12. Sampling Strategy for Task 2

The 500–1,000 review sample is drawn from the LLM-labeled 215,583 RRGen corpus using:

- **Stratified by predicted label** (~equal samples from all 7 categories despite class imbalance)
- **Confidence-stratified within each label** (mix of high-conf, medium-conf, low-conf predictions)
- **App-stratified** (no single app dominates the sample)
- Random seed: 42 (reproducible)

This ensures the gold-standard set covers the full range of LLM behavior, including the boundary cases the cleanlab pipeline is meant to correct.

## 13. Privacy and Ethics

- All reviews are public app-store data (RRGen dataset, Gao et al.).
- No personal information is collected from annotators beyond name + email for authorship attribution.
- Annotators may withdraw at any time.

---

**Contact:** Fabiha Jalal — farhansaif488@gmail.com
**Advisor:** Hasan Mahmud
**Project repository:** https://github.com/Fabiha-9876/ReviewAgent
