# Email Draft — to Hasan Mahmud (Advisor)

**To:** [advisor's email]
**From:** Fabiha Jalal — farhansaif488@gmail.com
**Subject:** ReviewAgent — project status update + annotation protocol + QA tasks

---

Dear Sir,

Thank you for your patience. I want to give you a consolidated update on the ReviewAgent project and follow up on your earlier requests for a project summary, defined QA tasks, and an annotation protocol with reliability measures.

## 1. Project Summary (5 lines)

ReviewAgent is an end-to-end pipeline that turns unstructured app-store reviews into structured issue specifications and developer-grade responses. It combines a 5-stage architecture (intake → KG-clustering → translation → response generation → RLHF) with a noise-aware data correction step that uses a small expert-verified subset to identify and correct mislabels in a much larger LLM-annotated dataset. The system achieves macro F1 = 0.81 across seven review categories on RRGen-scale data, including a previously failing compatibility class (F1 0.00 → 0.74). The work supports three research questions: (RQ1) whether structured taxonomy grounding improves issue-spec quality, (RQ2) whether issue-spec-conditioned response generation outperforms direct review→response, and (RQ3) whether dual-objective RLHF beats single-objective alignment. The classifier and noise-correction pipeline are now production-ready; the remaining experiments are gated on multi-annotator gold-standard data.

## 2. What's Done Since Last Update

### Data and labeling
- Full 310K RRGen corpus deduped and machine-labeled → **215,583 reviews**.
- I personally verified **5,230 reviews** (in `.numbers` files), measured LLM error rate ≈ 25% on praise predictions.
- Built a verified-anchor noise-modeling pipeline (cleanlab + RoBERTa anchor) → **44,214 corrections applied** to the 215K dataset.
- Independent V5 classifier supports **88.66%** of those corrections — strong validation that the noise-modeling pipeline is correct.

### Classifier (5 iterations)
| version | training data | macro F1 | notable |
|---|---|---|---|
| V1 | MAALEJ + synthetic (5.5K) | 0.799 | initial baseline |
| V2 | progressive auto-labeled (18K) | 0.856 | used to label full 215K |
| V3 | cleanlab-corrected (67K bal.) | 0.808 | first correction-aware model |
| V4 | RoBERTa-anchor-corrected (75K bal.) | 0.711 | validates correction pipeline |
| **V5** | V4 data + 300 synthetic compat | **0.813** | **production model** |

V5 is the first version where all 7 classes work (compatibility F1 = 0.74 vs V4's 0.00).

### Stage 2 (clustering)
- Without API access, ran a fully local pipeline: sentence-transformers + UMAP + HDBSCAN → **194 paper-grade clusters** with TF-IDF aspect-based auto-naming.
- Cluster examples: "lock-screen ads complaints" (4,505 reviews), "login from new phone" (3,023), "Samsung Galaxy crash/freeze" (332).
- Heuristic aspect extraction (spaCy + KeyBERT) on 113K reviews + Qwen2.5-3B local-LLM aspect extraction on 1K sample for gold-standard cross-validation (substring F1 = 0.53).

### Code quality
- 335 unit tests passing across all stages.
- All scripts and reproducible artifacts committed to GitHub.
- Documented in `SESSION_HISTORY.md` (732 lines, all decisions logged).

## 3. Annotation Protocol (attached)

Attached as `ANNOTATION_PROTOCOL.md`. Key details:

- **Sample size:** 500–1,000 reviews (per ICSE/FSE/ASE convention — Maalej, Guzman, Chen, Villarroel papers cited).
- **Annotators:** 3 independent graduate students.
- **Reliability measures:** Krippendorff's α (primary, threshold ≥ 0.67), Fleiss' κ (per-category secondary), pairwise Cohen's κ (per-annotator bias).
- **Calibration:** 20-review training round before main task.
- **Disagreement:** majority adoption with adjudication for 3-way disagreement.
- **Stratified sampling:** by predicted label, confidence band, and source app, with seed 42 for reproducibility.

## 4. Volunteer Tasks (defined)

| task | what they do | time per volunteer | status |
|---|---|---|---|
| Task 1 — Synthetic validation | 150 templated reviews × Y/N realism | ~45–60 min | sheet ready (`Synthetic_Data_Validation_Form_500.xlsx`) |
| Task 2 — Real-review verification | 500–1,000 RRGen reviews × Y/N + correct label | ~2–3 hours | protocol ready, sample to be drawn |
| Task 3 — Issue-spec scoring | 5-dim rubric on generated specs | TBD | follows after Stage 3 ground truth |

**Compensation:** authorship credit (co-author or acknowledgment) per your earlier directive on attribution.

## 5. What I'd Like to Request

1. **Connect me with 3 QA volunteers** for Task 2 — this is the single biggest blocker right now. Once volunteers complete the verification, I can compute α/κ scores, retrain the classifier on the gold-standard set, and run the three experiments and seven ablations.
2. **Confirm the protocol** is acceptable, or suggest revisions.
3. **Feedback on the synthetic-data validation form** I sent earlier (`Synthetic_Data_Validation_Form_500.xlsx`) — if you have time to review it.

## 6. What I'm Doing in Parallel (Unblocked Work)

- Hand-validating the top 50 clusters for paper-ready cluster purity stats.
- Writing the methodology section (the cleanlab + verified-anchor + V5-validation pipeline is a real, quantifiable contribution).
- Generating final figures (V1→V5 progression, before/after class distributions, correction policy diagram).

I am happy to set up a short meeting if it's easier to discuss in person.

Best regards,
Fabiha Jalal

---

**Attachments:**
- `ANNOTATION_PROTOCOL.md` — full annotation protocol with reliability measures
- `SESSION_HISTORY.md` — complete project decision log (optional, for reference)
- `Synthetic_Data_Validation_Form_500.xlsx` — already sent earlier, awaiting your review
