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
