"""Stage 3 Pipeline: Review-to-Issue Translation with HITL validation."""

from __future__ import annotations

from src.common.schemas import IssueCluster, IssueSpec
from src.common.llm_client import LLMClient
from .taxonomy import IssueTaxonomy
from .translator import ReviewToIssueTranslator
from .hitl_checkpoint import Stage3HITLCheckpoint


class Stage3Pipeline:
    """Orchestrates the review-to-issue translation process."""

    def __init__(
        self,
        llm_client: LLMClient,
        taxonomy: IssueTaxonomy | None = None,
        hitl: Stage3HITLCheckpoint | None = None,
        use_taxonomy: bool = True,
    ):
        self.translator = ReviewToIssueTranslator(llm_client, taxonomy or IssueTaxonomy())
        self.hitl = hitl or Stage3HITLCheckpoint()
        self.use_taxonomy = use_taxonomy

    async def process(
        self,
        clusters: list[IssueCluster],
        kg_context_map: dict[str, dict] | None = None,
    ) -> list[IssueSpec]:
        """Translate clusters into issue specs (without HITL)."""
        return await self.translator.translate_batch(
            clusters, kg_context_map, use_taxonomy=self.use_taxonomy
        )

    async def process_with_hitl(
        self,
        clusters: list[IssueCluster],
        score_callback=None,
        kg_context_map: dict[str, dict] | None = None,
        max_retries: int = 2,
    ) -> list[IssueSpec]:
        """Translate clusters with HITL rubric validation.

        Args:
            score_callback: function(spec: IssueSpec) -> dict[str, int]
                Returns rubric scores {dimension: 1-5} for a spec.
                If None, all specs are auto-approved.
        """
        specs = await self.process(clusters, kg_context_map)

        if score_callback is None:
            return specs

        cluster_map = {c.cluster_id: c for c in clusters}
        validated = []

        for spec in specs:
            for attempt in range(max_retries + 1):
                # Get expert scores
                scores = score_callback(spec)
                self.hitl.record_scores(spec.issue_id, scores, rater_id="expert")
                spec.rubric_scores = self.hitl.aggregate_scores(spec.issue_id)

                if self.hitl.check_threshold(spec):
                    spec.validated = True
                    break

                if attempt < max_retries:
                    cluster = cluster_map.get(spec.cluster_id)
                    if cluster:
                        ctx = (kg_context_map or {}).get(spec.cluster_id)
                        spec = await self.hitl.regenerate_with_feedback(
                            spec, self.translator, cluster, ctx
                        )

            validated.append(spec)

        return validated
