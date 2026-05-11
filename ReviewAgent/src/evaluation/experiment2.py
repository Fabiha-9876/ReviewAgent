"""Experiment 2: Coupled vs Uncoupled Response Generation (RQ2).

Tests 4 response generation conditions.
Statistical test: Friedman + Nemenyi post-hoc.
"""

from __future__ import annotations

import json
import asyncio
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from src.common.schemas import ReviewObject, IssueSpec, GeneratedResponse
from src.common.llm_client import LLMClient
from src.stage4b.rag_retriever import RAGRetriever
from src.stage4b.pipeline import Stage4bPipeline
from .metrics import compute_bleu, compute_rouge_l, compute_bert_score
from .statistical_tests import friedman_test, nemenyi_posthoc


@dataclass
class Experiment2Results:
    condition_responses: dict[str, list[GeneratedResponse]] = field(default_factory=dict)
    automatic_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    human_scores: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    statistical_tests: dict = field(default_factory=dict)


class Experiment2Runner:
    """Runs Experiment 2: Coupled vs Uncoupled Response Generation."""

    CONDITIONS = ["rrgen_baseline", "prompt_baseline", "reviewagent_no_spec", "reviewagent_full"]

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: RAGRetriever | None = None,
        output_dir: str = "data/processed/experiment2",
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = Experiment2Results()

    async def run(
        self,
        reviews: list[ReviewObject],
        issue_specs: list[IssueSpec],
        reference_responses: list[str] | None = None,
    ) -> Experiment2Results:
        """Generate responses under all 4 conditions."""
        print(f"Running Experiment 2 on {len(reviews)} reviews...")

        # Condition (a): RRGen baseline — no RAG, no issue spec
        print("  Condition (a): RRGen baseline...")
        pipeline_a = Stage4bPipeline(self.llm, retriever=None)
        self.results.condition_responses["rrgen_baseline"] = await pipeline_a.process(
            issue_specs, reviews, include_rag=False, include_issue_spec=False, refine=False
        )

        # Condition (b): CoRe baseline — app context but no structured issue spec
        print("  Condition (b): CoRe baseline...")
        pipeline_b = Stage4bPipeline(self.llm, retriever=self.retriever)
        self.results.condition_responses["prompt_baseline"] = await pipeline_b.process(
            issue_specs, reviews, include_rag=True, include_issue_spec=False, refine=False
        )

        # Condition (c): ReviewAgent WITHOUT issue spec (RAG only)
        print("  Condition (c): ReviewAgent without issue spec...")
        pipeline_c = Stage4bPipeline(self.llm, retriever=self.retriever)
        self.results.condition_responses["reviewagent_no_spec"] = await pipeline_c.process(
            issue_specs, reviews, include_rag=True, include_issue_spec=False, refine=True
        )

        # Condition (d): ReviewAgent WITH issue spec (full system)
        print("  Condition (d): ReviewAgent full system...")
        pipeline_d = Stage4bPipeline(self.llm, retriever=self.retriever)
        self.results.condition_responses["reviewagent_full"] = await pipeline_d.process(
            issue_specs, reviews, include_rag=True, include_issue_spec=True, refine=True
        )

        # Compute automatic metrics if references available
        if reference_responses:
            for cond, responses in self.results.condition_responses.items():
                preds = [r.text for r in responses]
                refs = reference_responses[:len(preds)]
                self.results.automatic_metrics[cond] = {
                    "bleu": compute_bleu(preds, refs),
                    "rouge_l": compute_rouge_l(preds, refs),
                    **{f"bert_{k}": v for k, v in compute_bert_score(preds, refs).items()},
                }
                print(f"  {cond}: BLEU={self.results.automatic_metrics[cond]['bleu']:.3f}")

        self._save()
        return self.results

    def evaluate(self, human_scores: dict[str, list[dict[str, int]]]) -> None:
        """Store human evaluation scores.

        Args:
            human_scores: {condition: [{"helpfulness": 1-5, "specificity": 1-5, ...}, ...]}
        """
        self.results.human_scores = human_scores

    def compare(self) -> dict:
        """Run Friedman test + Nemenyi post-hoc."""
        results = {}

        # Compare on each human dimension
        dims = ["helpfulness", "specificity", "empathy", "accuracy"]
        for dim in dims:
            conditions_data = []
            cond_names = []
            for cond in self.CONDITIONS:
                if cond in self.results.human_scores:
                    scores = [s[dim] for s in self.results.human_scores[cond]]
                    conditions_data.append(scores)
                    cond_names.append(cond)

            if len(conditions_data) >= 3:
                friedman = friedman_test(conditions_data)
                results[dim] = {"friedman": friedman}
                if friedman["significant"]:
                    posthoc = nemenyi_posthoc(conditions_data, cond_names)
                    results[dim]["nemenyi"] = posthoc

        # Compare on automatic metrics
        if self.results.automatic_metrics:
            for metric in ["bleu", "rouge_l", "bert_f1"]:
                values = {
                    cond: m.get(metric, 0)
                    for cond, m in self.results.automatic_metrics.items()
                }
                results[f"auto_{metric}"] = values

        self.results.statistical_tests = results
        return results

    def report(self) -> str:
        """Generate summary report."""
        lines = ["# Experiment 2: Coupled vs Uncoupled Response Generation (RQ2)\n"]

        # Automatic metrics
        if self.results.automatic_metrics:
            lines.append("## Automatic Metrics")
            lines.append("| Condition | BLEU | ROUGE-L | BERTScore F1 |")
            lines.append("|---|---|---|---|")
            for cond, m in self.results.automatic_metrics.items():
                lines.append(f"| {cond} | {m.get('bleu', 0):.3f} | {m.get('rouge_l', 0):.3f} | {m.get('bert_f1', 0):.3f} |")

        # Human scores
        if self.results.human_scores:
            lines.append("\n## Human Evaluation Scores")
            dims = ["helpfulness", "specificity", "empathy", "accuracy"]
            lines.append("| Condition | " + " | ".join(dims) + " |")
            lines.append("|" + "---|" * (len(dims) + 1))
            for cond, scores in self.results.human_scores.items():
                avgs = {d: np.mean([s[d] for s in scores]) for d in dims}
                lines.append(f"| {cond} | " + " | ".join(f"{v:.2f}" for v in avgs.values()) + " |")

        report = "\n".join(lines)
        (self.output_dir / "experiment2_report.md").write_text(report)
        return report

    def _save(self):
        for cond, responses in self.results.condition_responses.items():
            path = self.output_dir / f"{cond}_responses.json"
            path.write_text(json.dumps([r.model_dump(mode="json") for r in responses], indent=2))
