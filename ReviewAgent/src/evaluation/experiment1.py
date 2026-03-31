"""Experiment 1: Review-to-Issue Translation Quality (RQ1).

Tests 4 conditions on 100 clusters using 5-dimension rubric scoring.
Statistical test: Paired Wilcoxon signed-rank with Bonferroni correction.
"""

from __future__ import annotations

import json
import asyncio
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from src.common.schemas import IssueCluster, IssueSpec
from src.common.llm_client import LLMClient
from src.stage3.taxonomy import IssueTaxonomy
from src.stage3.translator import ReviewToIssueTranslator
from .metrics import (
    compute_completeness_ratio,
    compute_bert_score,
    compute_krippendorff_alpha,
    aggregate_rubric_scores,
)
from .statistical_tests import paired_wilcoxon, bonferroni_correction


@dataclass
class Experiment1Results:
    """Results for Experiment 1."""

    condition_specs: dict[str, list[IssueSpec]] = field(default_factory=dict)
    rubric_scores: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    completeness_ratios: dict[str, list[float]] = field(default_factory=dict)
    bert_scores: dict[str, dict] = field(default_factory=dict)
    krippendorff_alpha: float = 0.0
    statistical_tests: list[dict] = field(default_factory=list)


class Experiment1Runner:
    """Runs Experiment 1: Review-to-Issue Translation Quality."""

    CONDITIONS = ["llm_with_taxonomy", "llm_free_form", "raw_summary", "human_written"]

    def __init__(
        self,
        llm_client: LLMClient,
        output_dir: str = "data/processed/experiment1",
    ):
        self.llm = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.taxonomy = IssueTaxonomy()
        self.translator = ReviewToIssueTranslator(llm_client, self.taxonomy)
        self.results = Experiment1Results()

    async def run(
        self,
        clusters: list[IssueCluster],
        human_specs: list[IssueSpec] | None = None,
    ) -> Experiment1Results:
        """Generate issue specs under all 4 conditions."""
        print(f"Running Experiment 1 on {len(clusters)} clusters...")

        # Condition (a): LLM + taxonomy grounding (full system)
        print("  Condition (a): LLM with taxonomy grounding...")
        self.results.condition_specs["llm_with_taxonomy"] = (
            await self.translator.translate_batch(clusters, use_taxonomy=True)
        )

        # Condition (b): LLM without taxonomy (free-form)
        print("  Condition (b): LLM free-form...")
        self.results.condition_specs["llm_free_form"] = (
            await self.translator.translate_batch(clusters, use_taxonomy=False)
        )

        # Condition (c): Raw summary (no LLM structuring)
        print("  Condition (c): Raw summary...")
        raw_specs = []
        for cluster in clusters:
            raw_specs.append(IssueSpec(
                issue_id=f"RAW-{cluster.cluster_id}",
                cluster_id=cluster.cluster_id,
                title=f"{cluster.aspect} issue",
                issue_type=cluster.issue_type,
                description="\n".join(cluster.representative_reviews[:3]),
                environment=cluster.entities,
                severity="P2",
                priority_score=cluster.priority_score,
            ))
        self.results.condition_specs["raw_summary"] = raw_specs

        # Condition (d): Human-written (gold standard)
        if human_specs:
            self.results.condition_specs["human_written"] = human_specs
        else:
            print("  WARNING: No human-written specs provided. Using LLM+taxonomy as proxy.")
            self.results.condition_specs["human_written"] = (
                self.results.condition_specs["llm_with_taxonomy"]
            )

        # Compute completeness ratios
        for cond, specs in self.results.condition_specs.items():
            self.results.completeness_ratios[cond] = [
                compute_completeness_ratio(s) for s in specs
            ]
            print(f"  {cond} avg completeness: {np.mean(self.results.completeness_ratios[cond]):.2f}")

        self._save()
        return self.results

    def evaluate(
        self, rubric_scores_by_condition: dict[str, list[list[dict[str, int]]]]
    ) -> None:
        """Process rubric scores from raters.

        Args:
            rubric_scores_by_condition: {condition: [[rater1_scores], [rater2_scores], ...]}
                where each score is {"completeness": 1-5, "accuracy": 1-5, ...}
        """
        # Aggregate scores across raters per condition
        for cond, rater_scores_list in rubric_scores_by_condition.items():
            n_items = len(rater_scores_list[0])
            aggregated = []
            for i in range(n_items):
                item_scores = [rater_scores_list[r][i] for r in range(len(rater_scores_list))]
                aggregated.append(aggregate_rubric_scores(item_scores))
            self.results.rubric_scores[cond] = aggregated

        # Compute Krippendorff's alpha
        # Build ratings matrix: (n_raters, n_items * n_dimensions)
        first_cond = list(rubric_scores_by_condition.keys())[0]
        n_raters = len(rubric_scores_by_condition[first_cond])
        all_ratings = []
        for rater_idx in range(n_raters):
            rater_row = []
            for cond in self.CONDITIONS:
                if cond in rubric_scores_by_condition:
                    for item_scores in rubric_scores_by_condition[cond][rater_idx]:
                        for dim_score in item_scores.values():
                            rater_row.append(dim_score)
            all_ratings.append(rater_row)

        ratings_matrix = np.array(all_ratings, dtype=float)
        self.results.krippendorff_alpha = compute_krippendorff_alpha(ratings_matrix)
        print(f"  Krippendorff's alpha: {self.results.krippendorff_alpha:.3f}")

    def compare(self) -> list[dict]:
        """Run pairwise Wilcoxon tests with Bonferroni correction."""
        comparisons = [
            ("llm_with_taxonomy", "llm_free_form"),
            ("llm_with_taxonomy", "raw_summary"),
            ("llm_with_taxonomy", "human_written"),
            ("llm_free_form", "raw_summary"),
            ("llm_free_form", "human_written"),
            ("raw_summary", "human_written"),
        ]

        p_values = []
        results = []
        for cond_a, cond_b in comparisons:
            if cond_a not in self.results.rubric_scores or cond_b not in self.results.rubric_scores:
                continue
            scores_a = [np.mean(list(s.values())) for s in self.results.rubric_scores[cond_a]]
            scores_b = [np.mean(list(s.values())) for s in self.results.rubric_scores[cond_b]]

            test_result = paired_wilcoxon(scores_a, scores_b)
            test_result["comparison"] = f"{cond_a} vs {cond_b}"
            results.append(test_result)
            p_values.append(test_result["p_value"])

        # Bonferroni correction
        corrections = bonferroni_correction(p_values)
        for i, result in enumerate(results):
            result["bonferroni_significant"] = corrections[i]["significant"]
            result["corrected_alpha"] = corrections[i]["corrected_alpha"]

        self.results.statistical_tests = results
        return results

    def report(self) -> str:
        """Generate a summary report."""
        lines = ["# Experiment 1: Review-to-Issue Translation Quality (RQ1)\n"]

        # Completeness ratios
        lines.append("## Completeness Ratios")
        for cond, ratios in self.results.completeness_ratios.items():
            lines.append(f"- {cond}: {np.mean(ratios):.3f} (+/- {np.std(ratios):.3f})")

        # Rubric scores
        if self.results.rubric_scores:
            lines.append("\n## Mean Rubric Scores (per dimension)")
            dims = list(self.results.rubric_scores[list(self.results.rubric_scores.keys())[0]][0].keys())
            header = "| Condition | " + " | ".join(dims) + " | Mean |"
            lines.append(header)
            lines.append("|" + "---|" * (len(dims) + 2))
            for cond, scores in self.results.rubric_scores.items():
                dim_avgs = {d: np.mean([s[d] for s in scores]) for d in dims}
                overall = np.mean(list(dim_avgs.values()))
                row = f"| {cond} | " + " | ".join(f"{v:.2f}" for v in dim_avgs.values()) + f" | {overall:.2f} |"
                lines.append(row)

        # Krippendorff's alpha
        if self.results.krippendorff_alpha > 0:
            lines.append(f"\n## Inter-Annotator Agreement")
            lines.append(f"Krippendorff's alpha: {self.results.krippendorff_alpha:.3f}")

        # Statistical tests
        if self.results.statistical_tests:
            lines.append("\n## Statistical Comparisons (Wilcoxon + Bonferroni)")
            lines.append("| Comparison | p-value | Cliff's delta | Significant |")
            lines.append("|---|---|---|---|")
            for t in self.results.statistical_tests:
                sig = "Yes" if t.get("bonferroni_significant", t["significant"]) else "No"
                lines.append(f"| {t['comparison']} | {t['p_value']:.4f} | {t['cliffs_delta']:.3f} | {sig} |")

        report = "\n".join(lines)
        (self.output_dir / "experiment1_report.md").write_text(report)
        return report

    def _save(self):
        for cond, specs in self.results.condition_specs.items():
            path = self.output_dir / f"{cond}_specs.json"
            path.write_text(json.dumps([s.model_dump(mode="json") for s in specs], indent=2))
