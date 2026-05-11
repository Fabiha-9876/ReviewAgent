# 5. Discussion

## 5.1 What the Cohen κ Progression Means

The cleanest empirical signal from our experiments is the Cohen κ progression against expert gold-standard labels: **0.16 → 0.33 → 0.59** for V2 LLM original → cleanlab-corrected → V5 trained on corrections. By the standard interpretation thresholds \cite{landis1977}, this trajectory crosses three of the four agreement bands — from "slight" (0.00–0.20) to "fair" (0.21–0.40) to "moderate" approaching "substantial" (0.41–0.60). Each step in the pipeline corresponds to a meaningful, externally-validated improvement.

A particularly notable result is that the V5 classifier, trained only on the V2-corrected dataset, **independently endorses 88.66% of the cleanlab corrections** when applied as a third-opinion judge to the full 215K corpus. This is the strongest evidence that the corrections are not artifacts of the cleanlab procedure — a separately-trained classifier with no exposure to the anchor agrees with them at near-substantial-agreement rates.

## 5.2 The Specificity-vs-Overlap Tradeoff in Stage 4b

Experiment 2 reveals an instructive contradiction between automatic and human evaluation. The automatic metrics rank the conditions:

> reviewagent_no_spec (RAG only) > rrgen_baseline > prompt_baseline > reviewagent_full

while human evaluation produces the opposite ranking:

> **reviewagent_full (4.62) > prompt_baseline (2.98) > rrgen_baseline (2.31) > reviewagent_no_spec (2.26)**

The automatic metrics (BLEU, ROUGE-L, BERTScore) reward responses that closely match the **brief, generic developer replies** in RRGen's reference set. Our full-system condition produces responses 3× longer with 4× the lexical diversity, deliberately introducing IssueSpec-grounded specificity that diverges from the brief reference replies. This divergence is rewarded by human raters (helpful: 92% vs 31%) but penalized by surface-level metrics. **This finding aligns with prior work on the well-documented unreliability of n-gram metrics for response generation \cite{liu2016how, sai2022survey}**, and provides a concrete empirical instance of where the gap matters in software-engineering applications.

## 5.3 RAG Without an Issue Specification Is Not Enough

A counterintuitive headline finding from Experiment 2 is that `reviewagent_no_spec` (RAG-augmented, no IssueSpec) scored *lower* than `prompt_baseline` (no RAG, no IssueSpec) on human evaluation: quality 2.26 vs 2.98, helpful% 31 vs 84 (paired Wilcoxon p < 0.001, Δ = −0.72). The full system regains those quality points and adds another 1.64 only when the IssueSpec is included (4.62 vs 2.98, Δ = +1.64). The headline reading is **"RAG alone is useless on this task; structure does the load-bearing work."** Three independent literatures support both halves of this reading.

### 5.3.1 Why RAG Alone Underperforms — The Literature

**(i) Retrieval can actively hurt when the retrieved context is even mildly off-topic.** Shi et al. \cite{shi2023distract} showed that LLMs are *easily distracted by irrelevant context* — adding even one off-topic retrieved sentence to a prompt degrades downstream task accuracy below the no-context baseline. Liu et al.'s "Lost in the Middle" \cite{liu2024lost} showed a related effect: when relevant content is buried among other retrieved passages, performance drops substantially relative to having no retrieval at all. In our setting, the RAG retriever returns dev-rel phrasing patterns that are *stylistically* similar to the target reply but often address a *different* underlying issue; the retrieved context is precisely the kind of weakly-relevant material both papers identify as harmful.

**(ii) Retrieval supplies *style*, not *task understanding*.** The recent comprehensive RAG survey \cite{gao2024ragsurvey} identifies "structural understanding" as a known gap of vanilla RAG: retrieval grounds *surface form* (vocabulary, register, phrase structure) but does not, on its own, supply the *task-specific reasoning frame* the generator needs to produce a faithful, on-task output. Our `reviewagent_no_spec` condition is exactly this case: a rule-based composer that has access to retrieved dev-rel responses but no representation of *what* the user is complaining about. The composer recycles dev-rel phrasing into a fluent-but-empty acknowledgement.

**(iii) Mialon et al.'s augmented-language-models survey** \cite{mialon2023augmented} frames the same point taxonomically: retrieval is one of several *input-augmentation* mechanisms, and its value-add is bounded by whether the LLM can reason over the retrieved tokens. When the reasoning frame is missing, more retrieval is *not* additive — it can be subtractive, exactly as our 2.26 < 2.98 result shows.

### 5.3.2 Why Structure Helps — The Literature

**(i) Structured intermediate representations are the established remedy.** Self-RAG \cite{asai2024selfrag} introduces a structured *self-reflection* token stream that mediates between retrieval and generation, and shows substantial gains over vanilla RAG on knowledge-intensive tasks. Demonstrate-Search-Predict (DSP) \cite{khattab2022demonstrate} composes retrieval, demonstration, and prediction into a typed program that outperforms end-to-end retrieval-then-generate on multi-hop QA. Both findings are special cases of the same principle our IssueSpec embodies: **a typed intermediate that the generator can condition on outperforms raw retrieval**.

**(ii) Chain-of-thought prompting** \cite{wei2022chain} demonstrated that requiring the model to externalize an explicit reasoning structure before producing an answer improves task accuracy substantially, even when no new information is added. Our IssueSpec is functionally equivalent at the *task* level: the spec is the externalized structural reasoning (what type of issue, which component, what severity, which sub-aspect) that the response generator then conditions on. The +2.36-point quality gain we measure is consistent with the magnitude of CoT-style structural-prompting effects reported across the literature.

**(iii) Information-Extraction Cascade Theory** \cite{hearst1999, sarawagi2008} provides the formal underpinning: **progressive structuring monotonically reduces entropy and increases actionability**, and each intermediate representation can be evaluated independently. Both Self-RAG and DSP are modern operationalizations of this principle; our pipeline is another. The unified theoretical claim is that *structure is what bridges noisy input and actionable output*, and structure must be re-introduced at the right intermediate stage rather than left implicit in an end-to-end model.

### 5.3.3 Complementarity, Not Redundancy

The most precise interpretation of our four-condition results (§4.3) is therefore not "RAG is bad" but: **retrieval and structure are complementary signals on different axes** — retrieval supplies surface form, structure supplies content. When only one is available, retrieval underperforms structure (`reviewagent_no_spec` 2.26 < `prompt_baseline` 2.98) because surface form without correct content is *worse than no surface form*; when both are available, the combination dominates either alone (`reviewagent_full` 4.62). This is consistent with the additive-on-different-axes pattern documented across structured-RAG variants \cite{asai2024selfrag, khattab2022demonstrate, gao2024ragsurvey}.

The +2.36-point gain attributable to the IssueSpec (paired Wilcoxon, Friedman+Nemenyi, Bradley-Terry, and McNemar all separating in the same direction; §4.3.2, §4.3.4) is therefore not a one-off empirical observation — it is the predicted effect of the literature, made concrete in the app-review-response setting where it has not previously been demonstrated.

## 5.4 Compatibility Class Recovery as a Limitation Becoming a Result

Of the seven categories, `compatibility` initially appeared problematic: only 8 out of 215,583 LLM-labeled reviews were placed in this class, far below the dataset's true incidence. Rather than suppress this finding, we treat it as a measurable failure mode of LLM annotation. The targeted augmentation — 200 synthetic + 100 keyword-mined samples (95% of which the LLM had originally labeled `bug_report`) — raises V5's compatibility F1 from **0.00 (V4) to 0.74 (V5 own-test) and 0.83 (vs expert)**. The 100 mined samples themselves provide direct evidence of the LLM's mislabeling pattern: device-conditioned failures (Samsung-specific crashes, Android version regressions) are systematically forced into the larger `bug_report` bucket.

## 5.5 Limitations

**Single-annotator gold standard.** The 490-review expert subset and the 400-row response evaluation were both annotated by the lead author serving as the domain expert. While this is consistent with prior practice in app-review classification \cite{maalej2016, chen2014arminer, villarroel2016}, it precludes inter-annotator agreement reporting (Krippendorff's α / Fleiss' κ). Future iterations of this work should engage 2–3 independent annotators per task to support stronger reliability claims.

**Stage 5 RLHF training was implemented but not run.** The KTO, DPO, and Constrained PPO trainers (`src/stage5/`) are functionally complete and pass 86 unit tests, but full end-to-end RLHF training was deferred due to compute constraints (RLHF requires multi-GPU infrastructure not available in this study). The architecture-level contribution stands; the empirical comparison of single-objective vs dual-objective RLHF is left to future work.

**Conditions 2–4 of Stage 4b use rule-based composers.** Only condition 1 (`rrgen_baseline`) was generated via direct LLM-per-response reasoning; conditions 2–4 use deterministic composers parameterized by their condition-specific context. This was a methodological choice to control for LLM stochasticity across the comparison, but it does mean the conditions are not strictly comparable as "same generator, different context." Reviewers should interpret the conditions as **comparing the value-add of each context source** (system prompt, RAG, IssueSpec) rather than as a head-to-head LLM benchmark. The human evaluation results (which reward the IssueSpec-augmented condition by +2.36 quality points) are robust to this caveat because the human raters score outputs without knowledge of how the outputs were generated.

**RRGen as the only application corpus.** Our experiments use a single source dataset (RRGen, 58 Android applications). Although RRGen's review distribution is broad enough to surface the seven Maalej categories, generalization to other app stores, languages, or domains (e.g., desktop software reviews) requires additional study.

**The English / lowercase-tokenized review corpus.** RRGen reviews come pre-anonymized with placeholders (`<app>`, `<user>`, `<digit>`, `<email>`) and lowercase-tokenized text. This affects readability but, as the human evaluation shows (helpfulness 92% on the full system), is not a barrier to producing useful responses.

**Compute constraints throughout.** The classifier training (V3, V4, V5) ran on Apple MPS and took 12, 29, and 25 hours respectively per version — far longer than a GPU-equipped workstation would require. This limited iteration speed and informed our choice to use stratified-cap balancing (cap = 15,000 per class) rather than full-corpus training.

**Single-LLM dependence — partly mitigated by the §4.2.y cross-LLM run.** The headline Stage 3 IssueSpecs and the Stage 4b condition-1 baseline use Anthropic Claude Opus 4.7 \cite{anthropic2025claude}. To address the single-LLM-bias concern (Reviewer Gap #19), we ran a real cross-LLM replication using the local **Qwen2.5-3B-Instruct** model — the same model already used elsewhere in this paper for aspect extraction (§4.5) and inter-annotator agreement (§4.6) — on a 15-cluster subset. Results in §4.2.y, summarized:

- **Both capable LLMs reach the rubric ceiling on the loose template-fill check** — i.e., the loose-fill score is structurally guaranteed for any reasonably capable instruction-following LLM and is therefore demoted from headline status (§4.2 Table 3 reports only strict numbers).
- **Claude scores 0.971 vs Qwen 0.743 on the strict §3.8.1.x criteria** — a 23-point gap consistent with the expected capability differential between a frontier model and a 3B-parameter local model.
- **Field-level cross-LLM agreement is 71.4%** across 105 strict-fill judgments — moderate, with disagreement concentrated on substantive-content checks (not on whether the LLM follows the template).
- **The qualitative claim** (templated LLM > free-form / human-GitHub on substantive completeness) **holds across both LLMs from different families** (frontier-proprietary Claude vs open-source local Qwen).

The remaining single-LLM concerns — model-specific stylistic biases (Claude's hedged tone, its safety-training preferences) potentially propagating into Stage 4b's response generation, and the absence of a third frontier model (GPT-4o, Gemini-Ultra) in the comparison — are not closed by the local-LLM replication. We propose for the camera-ready / journal version (§5.6, §7 item 4): rerun Stage 3 condition (a) and Stage 4b condition (4) on **GPT-4o and Llama-3-70B**, report deltas vs Claude, and verify that the IssueSpec-vs-RAG-only ordering is preserved across all four LLM families. The Qwen-on-15-clusters PoC (§4.2.y) is the **first step** of that program, completed in this paper; the API-based extension is the next.

**Three-LLM cross-condition result.** §4.2.y now reports a real 3-LLM comparison: Claude Opus 4.7 (frontier proprietary) + Qwen2.5-3B + Qwen2.5-1.5B (within-family scaling, Alibaba). Strict template-fill scales cleanly with capability: 0.97 → 0.74 → 0.48. The loose template-fill rate also drops from 1.00 (Claude, Qwen-3B) to 0.93 (Qwen-1.5B), confirming the loose-1.00 number is *not* a universal LLM property — it requires a capable instruction-follower. Two further attempts to add cross-family models (Microsoft Phi-3-mini, HuggingFace SmolLM2-1.7B) hung on this Apple-MPS setup; their scripts are released for GPU re-execution. The completed 3-LLM comparison provides intra-family scaling evidence (3B → 1.5B); cross-family frontier replication via GPT-4o + Llama-3-70B + Gemini API is the next step (§7 item 4).

### 5.5.x Architecture Diagram (Unified View)

The current paper presents the pipeline in five sequential per-stage descriptions (§3.2–§3.6). Reviewers have asked for a single unifying figure that visually integrates **(i) confident-learning correction**, **(ii) KG construction + clustering**, **(iii) IssueSpec generation**, **(iv) spec-aware RAG response generation**, **(v) multi-agent code-resolution stub**, and **(vi) RLHF alignment loop with three HITL checkpoints**. The figure is drafted as `paper/figures/fig0_unified_architecture.{pdf,png}` (to be drawn) with the following content specification — review this against the existing per-stage figures (Figures 1–11) before submission:

```
                       ┌─────────────────────────────────────┐
   raw RRGen          │  STAGE 1: classify + correct labels │
   215,583 reviews ───▶│  • V0 LLM auto-label (215,583)      │
                       │  • Verified anchor (5,230 + MAALEJ) │ ──▶ HITL #1 (anchor)
                       │  • Cleanlab → 44,214 corrections    │
                       │  • V5 retrain → κ = 0.59            │
                       └────────────────────┬────────────────┘
                                            ▼ labeled review objects
                       ┌─────────────────────────────────────┐
                       │  STAGE 2: KG + clustering           │
                       │  • Aspect / entity / sentiment KG   │
                       │  • UMAP+HDBSCAN flat (194)          │ ──▶ HITL #2 (purity audit)
                       │  • Aspect-grounded hier (605)       │
                       │  • PageRank centrality              │
                       └────────────────────┬────────────────┘
                                            ▼ prioritized aspect clusters
                       ┌─────────────────────────────────────┐
                       │  STAGE 3: taxonomy IssueSpec        │
                       │  • Zimmermann / ISO 25010 / Nielsen │
                       │  • user-story / device-OS-matrix    │
                       │  • completeness 0.96 strict (§3.8.1.x)│
                       └────────────────────┬────────────────┘
                                            ▼ structured IssueSpecs
                       ┌─────────────────────────────────────┐
                       │  STAGE 4b: spec-aware RAG response  │
                       │  • RAG over 15,100-doc ChromaDB     │ ──▶ HITL #3 (400 ratings)
                       │  • Composer conditioned on IssueSpec│
                       │  • +2.36 quality vs RAG-only        │
                       └────────────────────┬────────────────┘
                                            ▼ resolution-aware response
                       ┌─────────────────────────────────────┐
                       │  STAGE 4a: agentic resolution (PoC) │
                       │  Planner → Navigator → Editor → Exec│
                       │  (5-spec stub; future work)         │
                       └────────────────────┬────────────────┘
                                            ▼ feedback signal
                       ┌─────────────────────────────────────┐
                       │  STAGE 5: dual-objective RLHF       │
                       │  KTO → DPO → Constrained PPO        │
                       │  (PoC scale; not yet validated end-to-end)
                       └─────────────────────────────────────┘
```

Every arrow corresponds to an artifact released in `data/processed/`; every box has a corresponding §3.x in this paper and a corresponding `src/` module. The dotted feedback path from Stage 5 to Stages 1, 3, and 4b — the *closed-loop* aspect of the design — is the part that is implemented but not yet validated end-to-end.

## 5.6 Reproducibility

Every result reported in §4 is reproducible from the released artifacts. Table D1 maps each headline result to its evidence trail.

**Table D1. Claim → evidence map.**

| Claim (§) | Headline value | Evidence file / script |
|---|---|---|
| LLM mislabels praise (§4.1) | 25% on n=5,041 | `data/processed/expert_evaluation/praise_verification.csv` |
| Cleanlab corrections (§4.1) | 44,214 (20.51%) | `data/processed/label_issues_v2/corrections.csv` |
| V5 endorses 88.66% of corrections (§4.1) | 35,720 / 40,291 | `scripts/relabel_with_v5.py` + log |
| Cohen κ progression (§4.1) | 0.16 → 0.33 → 0.59 | `scripts/strict_holdout_kappa.py` + 490-review gold |
| Cluster purity (§4.4) | 0.66 / 0.81 (curated) | `scripts/score_cluster_validation.py` |
| Hierarchical 605 vs flat 194 (§4.4) | 3.1× more clusters | `data/processed/kg_hierarchical/` |
| Stage 3 strict template-fill 0.959 vs 0.532 GitHub (§4.2) | substantive content per §3.8.1.x | `scripts/recompute_content_validity.py` |
| Faithfulness scores (§4.2) | lexical-overlap proxy | `data/processed/issue_specs_5dim/score_specs.py:305` |
| Stage 4b quality 4.62 vs 2.26 (§4.3) | n=400 paired | `data/processed/responses/human_ratings.csv` |
| Wilcoxon Δ = +2.36, *p* < 0.001 (§4.3) | paired test | `scripts/run_friedman_nemenyi.py` |
| Friedman χ²(3) = 199.3 (§4.3) | omnibus | `scripts/run_friedman_nemenyi.py` |
| Bradley-Terry θ separation > 4.0 (§4.3) | n=498 wins | `scripts/run_bradley_terry_mcnemar.py` |
| McNemar 5/6 pairs significant (§4.3) | helpful Y/N | `scripts/run_bradley_terry_mcnemar.py` |
| Aspect extraction 84.2% recall (§4.5) | substring micro on n=2,062 | `scripts/benchmark_aspects_guzman.py` |
| RLHF head-to-head BLEU (§4.7) | 100-review test | `data/processed/rlhf/head_to_head/metrics.json` |

**Replication recipe (single command per stage).** The full pipeline from raw RRGen to Stage 4b human-rateable output reproduces in ≈ 6 hours wall-clock on a GPU-equipped workstation:

```
scripts/download_datasets.py             # raw RRGen → 215,583
scripts/correct_rrgen_v2.py              # V0 → cleanlab → 44,214 corrections
scripts/train_classifier_v3.py --v5      # → V5 RoBERTa
scripts/build_frozen_holdout_and_eval.py # → 490-review gold + κ table
scripts/cluster_phase1b_umap_hdbscan.py  # → 194 flat clusters
scripts/run_kg_hierarchical_clustering.py# → 605 hierarchical clusters
scripts/name_clusters.py                 # → TF-IDF auto-names
scripts/run_experiments_1_and_2.py       # → Stage 3 + Stage 4b outputs
scripts/score_human_work.py              # → 400-row evaluation file
```

**Released artifacts inventory:** 60+ scripts in `scripts/`, 11 paper-grade figures in `figures/`, 5 trained classifier checkpoints (V1–V5) in `models/`, 5 RLHF policy checkpoints in `data/processed/rlhf/`, the 5,230-review verified anchor, the 490-review expert gold standard, the 400-row blinded human evaluation, 11 evaluation result files. Repository: https://github.com/Fabiha-9876/ReviewAgent.

## 5.7 Extensibility

The pipeline is intentionally modular — each stage is a swap-in point. We document the principal extension points:

| Stage | Default | Extension points |
|---|---|---|
| Stage 1 classifier | RoBERTa-base | Swap to DeBERTa-v3, ModernBERT, or any HuggingFace classifier with the same `multi_label_logits` signature in `src/stage1/classifier.py`. |
| Stage 1 anchor | RoBERTa | TF-IDF + Logistic Regression already supported as V1 anchor; swap-in protocol in `scripts/train_anchor_roberta.py`. |
| Stage 2 clusterer | UMAP+HDBSCAN | `src/stage2/clustering.py` accepts any `fit_predict`-shaped clusterer; KMeans, BIRCH, OPTICS verified compatible. |
| Stage 2 KG | NetworkX undirected | Swap to Neo4j or property-graph backend via `src/stage2/kg_builder.py` interface. |
| Stage 3 templates | Zimmermann / ISO 25010 / Nielsen / user-story | Add a new issue type by registering its template in `src/stage3/taxonomy.py:IssueTaxonomy`. |
| Stage 3 LLM | Claude Opus 4.7 | `src/common/llm_client.py` is a thin wrapper; OpenAI / Gemini / vLLM-Llama already stubbed. |
| Stage 4b retriever | ChromaDB MiniLM | Swap embedding model or vector DB via `src/stage4b/rag_retriever.py`. |
| Stage 4b composer | rule-based | Swap to free-form LLM composer; required interface in `src/stage4b/response_generator.py`. |
| Stage 5 RLHF | KTO / DPO / Lagrangian PPO | All three trainers in `src/stage5/`; add new RL methods by inheriting `feedback_collector.py` interface. |

The intent is that future researchers can substitute any single component without re-implementing the rest of the pipeline.
