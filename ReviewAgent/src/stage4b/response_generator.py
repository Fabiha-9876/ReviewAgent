"""Issue-spec-aware response generation."""

from __future__ import annotations

import uuid

from src.common.schemas import ReviewObject, IssueSpec, GeneratedResponse
from src.common.llm_client import LLMClient
from .rag_retriever import RAGRetriever

SYSTEM_PROMPT = """You are a customer support specialist for a mobile app. Generate a helpful,
empathetic, and specific response to a user's app review.

Guidelines:
- Reference the SPECIFIC issue the user is experiencing (not a generic response)
- Be empathetic — acknowledge the user's frustration
- If a fix or workaround exists, mention it concretely
- Do NOT make promises you can't keep (e.g., "will be fixed in the next update" unless confirmed)
- Do NOT leak internal information (code details, team names, internal tools)
- Suggest concrete next steps for the user (update the app, try a workaround, contact support)
- Keep the response concise (3-5 sentences)
- Maintain a professional but warm tone"""


class ResponseGenerator:
    """Generates user-facing responses that are aware of structured issue specs."""

    def __init__(self, llm_client: LLMClient, retriever: RAGRetriever | None = None):
        self.llm = llm_client
        self.retriever = retriever

    async def generate(
        self,
        review: ReviewObject,
        issue_spec: IssueSpec | None = None,
        include_rag: bool = True,
    ) -> GeneratedResponse:
        """Generate a response to a user review."""
        context = self._build_context(review, issue_spec, include_rag)

        text = await self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=context,
            temperature=0.4,
            max_tokens=512,
        )

        rag_sources = []
        if include_rag and self.retriever:
            rag_sources = [s for s in RAGRetriever.SOURCES]
        if issue_spec:
            rag_sources.append("issue_spec")

        return GeneratedResponse(
            response_id=str(uuid.uuid4()),
            issue_id=issue_spec.issue_id if issue_spec else "",
            review_id=review.review_id,
            text=text.strip(),
            rag_sources_used=rag_sources,
            refinement_iterations=0,
        )

    async def generate_batch(
        self,
        reviews: list[ReviewObject],
        issue_specs: list[IssueSpec | None],
        include_rag: bool = True,
    ) -> list[GeneratedResponse]:
        """Generate responses for multiple reviews."""
        import asyncio

        tasks = [
            self.generate(review, spec, include_rag)
            for review, spec in zip(reviews, issue_specs)
        ]
        return await asyncio.gather(*tasks)

    def _build_context(
        self,
        review: ReviewObject,
        issue_spec: IssueSpec | None,
        include_rag: bool,
    ) -> str:
        """Assemble the prompt context."""
        sections = [
            f"## User Review (Rating: {review.rating}/5):",
            f'"{review.text}"',
        ]

        # Add issue spec context (the key coupling)
        if issue_spec:
            sections.append(f"\n## Structured Issue Analysis:")
            sections.append(f"- **Issue:** {issue_spec.title}")
            sections.append(f"- **Type:** {issue_spec.issue_type}")
            sections.append(f"- **Severity:** {issue_spec.severity}")
            sections.append(f"- **Affected Component:** {issue_spec.affected_component}")
            if issue_spec.actual_behavior:
                sections.append(f"- **Known Problem:** {issue_spec.actual_behavior}")
            if issue_spec.steps_to_reproduce:
                sections.append(f"- **Reproduction:** {'; '.join(issue_spec.steps_to_reproduce[:3])}")
            sections.append(f"- **Affected Users:** ~{issue_spec.priority_score * 100:.0f}% priority")

        # Add RAG context
        if include_rag and self.retriever:
            query = review.text
            if issue_spec:
                query += f" {issue_spec.title}"
            docs = self.retriever.retrieve(query, top_k=3)
            if docs:
                sections.append("\n## Reference Information:")
                for doc in docs:
                    sections.append(f"- [{doc.source}]: {doc.text[:200]}")

        sections.append(
            "\n\nGenerate a helpful, empathetic response to this user's review. "
            "Reference the specific issue identified above."
        )
        return "\n".join(sections)
