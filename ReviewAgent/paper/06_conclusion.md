# 6. Conclusion

We have presented **ReviewAgent**, a four-stage pipeline that demonstrates how systematic LLM annotation noise in app-store review datasets can be detected and corrected with a small expert-verified anchor. The contribution is methodological: the verified-anchor + confident-learning correction step turns ≈30 person-hours of expert annotation into a leverage point that improves a 215,000-review dataset, with measurable downstream gains across three independent evaluations.

The cleanest empirical signal is the **Cohen κ progression** against expert gold-standard labels: **0.16 → 0.33 → 0.59** for V2 LLM original → cleanlab-corrected → V5 trained on corrections. Each pipeline step produces a measurable, externally-validated improvement, and a separately-trained classifier (V5) independently endorses **88.66%** of the corrections — the strongest evidence that the corrections are not artifacts of the procedure.

Two findings have implications beyond app-review classification:

1. **LLM annotation noise is structural, not random.** Class collapse (popular categories absorb minority categories) and boundary confusion (semantically adjacent classes blur together) are predictable failure modes that small expert verification corrects efficiently. The 25% praise mislabeling rate we measure on RRGen is unlikely to be unique to this dataset.

2. **Retrieval is necessary but not sufficient for paper-grade response generation.** RAG without a structured issue specification underperforms even no-RAG baselines on human evaluation (`reviewagent_no_spec` quality 2.26 vs `core_baseline` 2.98, p < 0.001). Adding the IssueSpec to RAG yields +2.36 quality points (p < 0.001) — the structural component is doing the work that RAG alone cannot.

The full-system response generator achieves a **92% helpfulness rate** in a 400-rating blinded human evaluation, against 19% for the original RRGen-style baseline (a 4.84× improvement on identical inputs).

We release all artifacts publicly: 14 scripts implementing the pipeline, 11 paper-grade figures, 5 trained classifier checkpoints (V1–V5), the 5,230-review verified anchor, the 490-review expert gold standard, the 400-row blinded human evaluation, and 11 evaluation result files. The repository is at https://github.com/Fabiha-9876/ReviewAgent.

We hope the verified-anchor + confident-learning approach finds use beyond this work — wherever LLM annotation is being used to bootstrap software-engineering datasets at scale.

# 7. Future Work

Three concrete extensions follow naturally from the present work:

1. **Multi-annotator extension.** The current gold-standard set (490 reviews) and human evaluation (400 ratings) are single-annotator. Adding 2 independent annotators on a 100-review subsample to compute Krippendorff's α and Fleiss' κ would strengthen the reliability claim and enable formal between-rater statistics.

2. **Full-scale Stage 5 RLHF training.** The KTO, DPO, and Constrained PPO trainers are implemented in `src/stage5/` and pass 86 unit tests, but end-to-end training was deferred due to compute constraints. Given multi-GPU access and the now-existing 400-rating preference data, training each variant on a fine-tuned base generator and comparing via Bradley–Terry + McNemar tests (as designed in `src/evaluation/experiment3.py`) is the natural next step. We expect dual-objective RLHF (Constrained PPO) to dominate single-objective methods (KTO, DPO) on the quality–safety frontier, but this is unverified.

3. **Cross-corpus generalization.** All experiments use a single source corpus (RRGen, 58 Android applications). Applying the verified-anchor + cleanlab pipeline to a second corpus — e.g., Apple App Store reviews, Steam game reviews, or a non-English corpus — would test whether the noise-correction approach generalizes across review sources, languages, and platforms.

A fourth, more ambitious direction is **end-to-end pipeline learning**: training the classifier, clusterer, issue-specification generator, and response generator jointly with a unified loss that rewards downstream response quality. The current pipeline trains each stage independently; a joint formulation could reveal whether stage-level corrections compound or interact in unexpected ways.
