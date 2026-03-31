"""LLM-based review-to-issue translation — the core novel contribution."""

from __future__ import annotations

import uuid
import json

from src.common.schemas import IssueCluster, IssueSpec, ExtractedEntities
from src.common.llm_client import LLMClient
from .taxonomy import IssueTaxonomy


class ReviewToIssueTranslator:
    """Converts IssueCluster objects into structured IssueSpec using an LLM."""

    def __init__(self, llm_client: LLMClient, taxonomy: IssueTaxonomy | None = None):
        self.llm = llm_client
        self.taxonomy = taxonomy or IssueTaxonomy()

    async def translate(
        self,
        cluster: IssueCluster,
        kg_context: dict | None = None,
        use_taxonomy: bool = True,
    ) -> IssueSpec:
        """Generate an IssueSpec from an IssueCluster."""
        if use_taxonomy:
            system_prompt = self.taxonomy.get_template(cluster.issue_type)
        else:
            system_prompt = (
                "You are a software engineer. Convert the following cluster of user reviews "
                "into a structured issue specification. Include a title, description, "
                "steps to reproduce, severity, and affected component."
            )

        user_prompt = self._build_prompt(cluster, kg_context)

        raw = await self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=2048,
        )

        return self._parse_response(raw, cluster)

    async def translate_batch(
        self,
        clusters: list[IssueCluster],
        kg_context_map: dict[str, dict] | None = None,
        use_taxonomy: bool = True,
    ) -> list[IssueSpec]:
        """Translate multiple clusters."""
        import asyncio

        tasks = []
        for cluster in clusters:
            ctx = (kg_context_map or {}).get(cluster.cluster_id)
            tasks.append(self.translate(cluster, ctx, use_taxonomy))
        return await asyncio.gather(*tasks)

    def _build_prompt(self, cluster: IssueCluster, kg_context: dict | None) -> str:
        """Construct the full user prompt from cluster data."""
        sections = [
            f"## Issue Cluster: {cluster.cluster_id}",
            f"**Issue Type:** {cluster.issue_type}",
            f"**Aspect:** {cluster.aspect}",
            f"**Sub-category:** {cluster.sub_category}",
            f"**Review Count:** {cluster.review_count}",
            f"**Sentiment Distribution:** {json.dumps(cluster.sentiment_distribution)}",
            f"**Priority Score:** {cluster.priority_score}",
            "",
            "## Representative Reviews:",
        ]
        for i, review_text in enumerate(cluster.representative_reviews, 1):
            sections.append(f'{i}. "{review_text}"')

        sections.append("")
        sections.append("## Extracted Entities:")
        if cluster.entities.devices:
            sections.append(f"- Devices: {', '.join(cluster.entities.devices)}")
        if cluster.entities.os_versions:
            sections.append(f"- OS Versions: {', '.join(cluster.entities.os_versions)}")
        if cluster.entities.app_versions:
            sections.append(f"- App Versions: {', '.join(cluster.entities.app_versions)}")
        if cluster.entities.screens:
            sections.append(f"- Screens: {', '.join(cluster.entities.screens)}")

        if cluster.temporal_pattern:
            sections.append(f"\n## Temporal Pattern: {cluster.temporal_pattern}")

        if kg_context:
            sections.append(f"\n## Knowledge Graph Context:")
            sections.append(f"Related clusters: {json.dumps(kg_context)}")

        sections.append(
            "\n\nGenerate a complete, structured issue specification based on the above data. "
            "Use markdown formatting."
        )
        return "\n".join(sections)

    def _parse_response(self, raw: str, cluster: IssueCluster) -> IssueSpec:
        """Parse LLM output into an IssueSpec."""
        lines = raw.strip().split("\n")

        title = ""
        description = ""
        steps = []
        expected = None
        actual = None
        severity = "P2"
        component = ""
        user_story = None
        acceptance_criteria = None
        nfr_category = None
        nielsen_heuristic = None

        current_section = None
        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            if "title" in lower and ("**" in stripped or "#" in stripped):
                title = stripped.split(":", 1)[-1].strip().strip("*#").strip()
                current_section = "title"
            elif "steps to reproduce" in lower or "reproduction" in lower:
                current_section = "steps"
            elif "expected behavior" in lower or "expected" in lower and "actual" not in lower:
                current_section = "expected"
            elif "actual behavior" in lower:
                current_section = "actual"
            elif "description" in lower and ("**" in stripped or "#" in stripped):
                current_section = "description"
            elif "severity" in lower and ("**" in stripped or "#" in stripped):
                for p in ["P0", "P1", "P2", "P3"]:
                    if p in stripped:
                        severity = p
                        break
                current_section = None
            elif "affected component" in lower or "component" in lower:
                component = stripped.split(":", 1)[-1].strip().strip("*").strip()
                current_section = None
            elif "user story" in lower:
                current_section = "user_story"
            elif "acceptance criteria" in lower:
                current_section = "acceptance"
            elif "nfr category" in lower:
                nfr_category = stripped.split(":", 1)[-1].strip().strip("*").strip()
                current_section = None
            elif "nielsen" in lower or "heuristic" in lower:
                nielsen_heuristic = stripped.split(":", 1)[-1].strip().strip("*").strip()
                current_section = None
            elif current_section == "steps" and stripped.startswith(("1", "2", "3", "4", "5", "6", "-")):
                step = stripped.lstrip("0123456789.-) ").strip()
                if step:
                    steps.append(step)
            elif current_section == "expected" and stripped:
                expected = (expected or "") + " " + stripped
            elif current_section == "actual" and stripped:
                actual = (actual or "") + " " + stripped
            elif current_section == "description" and stripped:
                description += " " + stripped
            elif current_section == "user_story" and stripped:
                user_story = (user_story or "") + " " + stripped
            elif current_section == "acceptance" and stripped.startswith(("1", "2", "3", "-")):
                if acceptance_criteria is None:
                    acceptance_criteria = []
                acceptance_criteria.append(stripped.lstrip("0123456789.-) ").strip())

        return IssueSpec(
            issue_id=f"ISS-{uuid.uuid4().hex[:6].upper()}",
            cluster_id=cluster.cluster_id,
            title=title or f"{cluster.aspect} — {cluster.sub_category}",
            issue_type=cluster.issue_type,
            description=description.strip() or "See representative reviews.",
            steps_to_reproduce=steps or None,
            expected_behavior=expected.strip() if expected else None,
            actual_behavior=actual.strip() if actual else None,
            user_story=user_story.strip() if user_story else None,
            acceptance_criteria=acceptance_criteria,
            nfr_category=nfr_category,
            nielsen_heuristic=nielsen_heuristic,
            environment=cluster.entities,
            severity=severity,
            affected_component=component,
            priority_score=cluster.priority_score,
        )
