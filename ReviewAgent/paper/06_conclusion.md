# 6. Conclusion

We have presented **ReviewAgent**, a four-stage pipeline that demonstrates how systematic LLM annotation noise in app-store review datasets can be detected and corrected with a small expert-verified anchor. The contribution is methodological: the verified-anchor + confident-learning correction step turns ≈30 person-hours of expert annotation into a leverage point that improves a 215,000-review dataset, with measurable downstream gains across three independent evaluations.

The cleanest empirical signal is the **Cohen κ progression** against expert gold-standard labels: **0.16 → 0.33 → 0.59** for V2 LLM original → cleanlab-corrected → V5 trained on corrections. Each pipeline step produces a measurable, externally-validated improvement, and a separately-trained classifier (V5) independently endorses **88.66%** of the corrections — the strongest evidence that the corrections are not artifacts of the procedure.

Two findings have implications beyond app-review classification:

1. **LLM annotation noise is structural, not random.** Class collapse (popular categories absorb minority categories) and boundary confusion (semantically adjacent classes blur together) are predictable failure modes that small expert verification corrects efficiently. The 25% praise mislabeling rate we measure on RRGen is unlikely to be unique to this dataset.

2. **Retrieval is necessary but not sufficient for paper-grade response generation.** RAG without a structured issue specification underperforms even no-RAG baselines on human evaluation (`reviewagent_no_spec` quality 2.26 vs `prompt_baseline` 2.98, p < 0.001). Adding the IssueSpec to RAG yields +2.36 quality points (p < 0.001) — the structural component is doing the work that RAG alone cannot.

The full-system response generator achieves a **92% helpfulness rate** in a 400-rating blinded human evaluation, against 19% for the original RRGen-style baseline (a 4.84× improvement on identical inputs).

We release all artifacts publicly: 14 scripts implementing the pipeline, 11 paper-grade figures, 5 trained classifier checkpoints (V1–V5), the 5,230-review verified anchor, the 490-review expert gold standard, the 400-row blinded human evaluation, and 11 evaluation result files. The repository is at https://github.com/Fabiha-9876/ReviewAgent.

We hope the verified-anchor + confident-learning approach finds use beyond this work — wherever LLM annotation is being used to bootstrap software-engineering datasets at scale.

# 7. Future Work

The limitations identified in §5.5 map directly to a prioritized list of next experiments. We rank by dependency and impact: items higher in the list close the most reviewer-visible gaps.

## 7.1 High-Priority (Closes Major Reviewer Concerns)

1. **Multi-annotator gold standard.** Add 2 independent expert annotators on a 100-review subsample of the 490-review gold standard. Compute Krippendorff's α and pairwise Cohen's κ to bound the single-rater bias of all §4 results. This is the highest-leverage near-term item — closes §5.5(A), strengthens the κ progression of §4.1, and enables proper inter-rater reliability in §4.3's 400-rating evaluation.

2. **Multi-annotator Stage 4b human evaluation.** Recruit 2 additional raters for a 100-row subset of the 400 (review, response) pairs. Re-compute helpfulness, specificity, and quality with proper Krippendorff α and Fleiss κ. This addresses the construct-validity concern of §4.3.3 (rater designed the system).

3. **Full-scale Stage 5 RLHF on a generation-grade base.** Replace the distilGPT2 PoC with Llama-3-8B-Instruct (or Mistral-7B-Instruct). Use the existing 400-rating preference data for DPO and Constrained-PPO training; collect a held-out 100-row test set; evaluate via Bradley-Terry preference and McNemar safety-violation tests as designed in `src/evaluation/experiment3.py`. This is what would actually test the dual-objective claim of §3.0.

4. **Multi-LLM replication of Stage 3 and Stage 4b.** Rerun condition (a) of Stage 3 and condition (4) of Stage 4b on GPT-4o and Llama-3-70B. Verify the IssueSpec-vs-RAG-only ordering is preserved across LLMs; report any LLM-specific deltas. Closes §5.5 single-LLM dependence.

## 7.2 Medium-Priority (Closes Construct Gaps)

5. **NLI-based faithfulness.** Replace the lexical-overlap proxy of §3.7.2 with a per-spec-sentence textual-entailment check against the cluster (using DeBERTa-v3-large fine-tuned on MNLI / DocNLI). Compare scores against a 100-spec subset hand-rated for faithfulness on a Likert scale by ≥ 2 raters. This is the construct-correct measurement we currently substitute with a proxy.

6. **Hierarchical purity audit.** Run the same 50-cluster Y/P/N audit on the 605-cluster hierarchical output to confirm H2's "comparable purity at finer granularity" claim that is currently only provisional.

7. **Anchor-size ablation.** The verified-anchor budget (5,230 + 5,008 = 10,238) was chosen pragmatically. Run the cleanlab pipeline at anchor sizes ∈ {1K, 2.5K, 5K, 7.5K, 10K} and report the κ progression as a function of annotation budget. This generalizes the methodology — telling future researchers how much expert annotation actually matters.

## 7.3 Long-Horizon Extensions

8. **Cross-corpus generalization.** Apply verified-anchor + cleanlab to a second corpus — Apple App Store reviews, Steam game reviews, or a non-English (e.g., Mandarin or Spanish) corpus. Tests whether the noise-correction approach generalizes across review sources, platforms, and languages.

9. **Full agentic resolution (Stage 4a).** Replace the 5-spec stub with a real Planner→Navigator→Editor→Executor loop on open-source Android apps where source repositories are available (e.g., AntennaPod, Signal, Wikipedia). The IssueSpecs from Stage 3 would feed into a SWE-Agent-style loop \cite{nashid2023codequery, ahmed2024automatic} producing actual patches; success measured via patch-acceptance and test-pass rate.

10. **Closed-loop deployment study.** Deploy the spec-aware response generator to a live app's developer-response queue. Measure (a) reviewer reply rate, (b) reviewer-rated satisfaction, (c) developer-time-saved per response. This is the only way to validate the §4.3.3 "helpful" claim against an *outcome*, not a *predicted* outcome.

11. **End-to-end pipeline learning.** Train the classifier, clusterer, issue-specification generator, and response generator jointly with a unified loss that rewards downstream response quality. The current pipeline trains each stage independently; a joint formulation could reveal whether stage-level corrections compound.

12. **Agentic RAG comparison.** Add a fourth Stage 4b condition: an *agentic* RAG loop (multi-turn retrieval, self-refinement, tool use) and benchmark against vanilla RAG and structured RAG. Quantifies whether agentic iteration adds value beyond structured intermediates.

13. **Per-domain template library.** Extend the Stage 3 taxonomy beyond mobile-app issue types — backend incidents (post-mortem template), UX research findings (Hartson template), accessibility issues (WCAG template). Each addition broadens the framework's domain coverage.
