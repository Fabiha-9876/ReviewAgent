"""Ablation studies A1-A7: remove one component and measure impact."""

from __future__ import annotations

import json
import asyncio
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from src.common.schemas import ReviewObject, IssueCluster, IssueSpec
from src.common.llm_client import LLMClient
from src.stage2.pipeline import Stage2Pipeline
from src.stage2.schema_mapper import SchemaMapper
from src.stage3.pipeline import Stage3Pipeline
from src.stage4b.pipeline import Stage4bPipeline
from src.stage4b.rag_retriever import RAGRetriever
from .metrics import compute_completeness_ratio, compute_bleu, compute_rouge_l


@dataclass
class AblationResult:
    ablation_id: str
    name: str
    what_removed: str
    full_system_score: float = 0.0
    ablated_score: float = 0.0
    delta: float = 0.0
    metric_name: str = ""
    details: dict = field(default_factory=dict)


class AblationRunner:
    """Runs ablation studies A1-A7."""

    ABLATIONS = {
        "A1": {"name": "No KG", "what_removed": "Knowledge graph (skip Stage 2)"},
        "A2": {"name": "No hierarchical clustering", "what_removed": "Two-level hierarchy (flat clustering)"},
        "A3": {"name": "No taxonomy grounding", "what_removed": "Literature-grounded templates"},
        "A4": {"name": "No HITL at Stage 3", "what_removed": "Expert validation checkpoint"},
        "A5": {"name": "No RAG", "what_removed": "Retrieval augmented generation"},
        "A6": {"name": "No issue spec in response", "what_removed": "Coupling between Stage 3 and 4b"},
        "A7": {"name": "Single-stream feedback", "what_removed": "Dual-objective decomposition"},
    }

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: RAGRetriever | None = None,
        output_dir: str = "data/processed/ablations",
    ):
        self.llm = llm_client
        self.retriever = retriever
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[AblationResult] = []

    async def run_ablation(
        self,
        ablation_id: str,
        reviews: list[ReviewObject],
        clusters: list[IssueCluster],
        full_system_specs: list[IssueSpec],
        full_system_responses: list[str] | None = None,
    ) -> AblationResult:
        """Run a single ablation study."""
        info = self.ABLATIONS[ablation_id]
        print(f"  Running {ablation_id}: {info['name']}...")

        result = AblationResult(
            ablation_id=ablation_id,
            name=info["name"],
            what_removed=info["what_removed"],
        )

        if ablation_id == "A1":
            # No KG: feed reviews directly to Stage 3 without clustering
            # Create one giant cluster per review label
            from collections import Counter
            label_groups: dict[str, list[ReviewObject]] = {}
            for r in reviews:
                for label in r.labels:
                    label_groups.setdefault(label, []).append(r)

            flat_clusters = []
            mapper = SchemaMapper()
            for label, group_reviews in label_groups.items():
                cluster = IssueCluster(
                    cluster_id=f"FLAT-{label}",
                    issue_type=label if label in ["bug_report", "feature_request", "performance", "usability", "compatibility"] else "bug_report",
                    aspect=label,
                    sub_category="flat",
                    review_ids=[r.review_id for r in group_reviews],
                    review_count=len(group_reviews),
                    representative_reviews=[r.text for r in group_reviews[:3]],
                    entities=group_reviews[0].entities,
                )
                flat_clusters.append(cluster)

            stage3 = Stage3Pipeline(self.llm)
            ablated_specs = await stage3.process(flat_clusters)
            full_scores = [compute_completeness_ratio(s) for s in full_system_specs]
            ablated_scores = [compute_completeness_ratio(s) for s in ablated_specs]
            result.full_system_score = float(np.mean(full_scores))
            result.ablated_score = float(np.mean(ablated_scores))
            result.metric_name = "completeness_ratio"

        elif ablation_id == "A3":
            # No taxonomy: translate without templates
            stage3 = Stage3Pipeline(self.llm, use_taxonomy=False)
            ablated_specs = await stage3.process(clusters)
            full_scores = [compute_completeness_ratio(s) for s in full_system_specs]
            ablated_scores = [compute_completeness_ratio(s) for s in ablated_specs]
            result.full_system_score = float(np.mean(full_scores))
            result.ablated_score = float(np.mean(ablated_scores))
            result.metric_name = "completeness_ratio"

        elif ablation_id == "A5":
            # No RAG: generate responses without retrieval
            pipeline = Stage4bPipeline(self.llm, retriever=None)
            matched_reviews = reviews[:len(full_system_specs)]
            responses = await pipeline.process(
                full_system_specs, matched_reviews, include_rag=False, include_issue_spec=True
            )
            result.metric_name = "response_quality"
            result.ablated_score = len(responses)  # Placeholder
            result.full_system_score = len(responses)

        elif ablation_id == "A6":
            # No issue spec: RAG but no structured issue
            pipeline = Stage4bPipeline(self.llm, retriever=self.retriever)
            matched_reviews = reviews[:len(full_system_specs)]
            responses = await pipeline.process(
                full_system_specs, matched_reviews, include_rag=True, include_issue_spec=False
            )
            result.metric_name = "response_quality"
            result.ablated_score = len(responses)
            result.full_system_score = len(responses)

        else:
            # A2, A4, A7: require more complex setup
            result.metric_name = "pending"
            result.details = {"note": f"{ablation_id} requires specific setup for full evaluation"}

        result.delta = result.full_system_score - result.ablated_score
        self.results.append(result)
        return result

    async def run_all(
        self,
        reviews: list[ReviewObject],
        clusters: list[IssueCluster],
        full_system_specs: list[IssueSpec],
    ) -> list[AblationResult]:
        """Run all 7 ablation studies."""
        print("Running all ablation studies...")
        for aid in self.ABLATIONS:
            await self.run_ablation(aid, reviews, clusters, full_system_specs)
        self._save()
        return self.results

    def report(self) -> str:
        """Generate comparison table."""
        lines = ["# Ablation Study Results\n"]
        lines.append("| ID | Name | What Removed | Metric | Full System | Ablated | Delta |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in self.results:
            lines.append(
                f"| {r.ablation_id} | {r.name} | {r.what_removed} | {r.metric_name} | "
                f"{r.full_system_score:.3f} | {r.ablated_score:.3f} | {r.delta:+.3f} |"
            )

        report = "\n".join(lines)
        (self.output_dir / "ablation_report.md").write_text(report)
        return report

    def _save(self):
        data = [
            {
                "ablation_id": r.ablation_id,
                "name": r.name,
                "what_removed": r.what_removed,
                "metric": r.metric_name,
                "full_system_score": r.full_system_score,
                "ablated_score": r.ablated_score,
                "delta": r.delta,
            }
            for r in self.results
        ]
        (self.output_dir / "ablation_results.json").write_text(json.dumps(data, indent=2))
