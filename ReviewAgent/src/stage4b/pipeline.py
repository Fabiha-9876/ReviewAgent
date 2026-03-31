"""Stage 4b Pipeline: Issue-spec-aware response generation with self-refinement."""

from __future__ import annotations

from src.common.schemas import ReviewObject, IssueSpec, GeneratedResponse
from src.common.llm_client import LLMClient
from .rag_retriever import RAGRetriever
from .response_generator import ResponseGenerator
from .self_refiner import SelfRefiner


class Stage4bPipeline:
    """Orchestrates response generation with RAG and self-refinement."""

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: RAGRetriever | None = None,
        max_refinement_iterations: int = 3,
    ):
        self.generator = ResponseGenerator(llm_client, retriever)
        self.refiner = SelfRefiner(llm_client, max_iterations=max_refinement_iterations)

    async def process(
        self,
        issue_specs: list[IssueSpec],
        reviews: list[ReviewObject],
        include_rag: bool = True,
        include_issue_spec: bool = True,
        refine: bool = True,
    ) -> list[GeneratedResponse]:
        """Generate responses for reviews linked to issue specs.

        Each review is matched to its issue spec. If there are more reviews than
        specs, extra reviews get no issue context.
        """
        import asyncio

        tasks = []
        for i, review in enumerate(reviews):
            spec = issue_specs[i] if include_issue_spec and i < len(issue_specs) else None
            tasks.append(
                self._process_single(review, spec, include_rag, refine)
            )
        return await asyncio.gather(*tasks)

    async def _process_single(
        self,
        review: ReviewObject,
        issue_spec: IssueSpec | None,
        include_rag: bool,
        refine: bool,
    ) -> GeneratedResponse:
        """Generate and optionally refine a single response."""
        response = await self.generator.generate(review, issue_spec, include_rag)
        if refine:
            response = await self.refiner.refine(response, issue_spec)
        return response
