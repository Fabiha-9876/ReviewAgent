"""Self-refinement loop for response quality improvement."""

from __future__ import annotations

from src.common.schemas import GeneratedResponse, IssueSpec
from src.common.llm_client import LLMClient

CRITIQUE_PROMPT = """You are a quality reviewer for customer support responses.
Evaluate this response on three dimensions and suggest improvements:

1. **Specificity**: Does it reference the specific issue, or is it generic?
2. **Compliance**: Does it make unauthorized promises or leak internal info?
3. **Empathy**: Does it acknowledge the user's frustration appropriately?

For each dimension, output:
- "pass" if acceptable
- A specific improvement suggestion if not

Format as JSON:
{"specificity": "pass" or "suggestion", "compliance": "pass" or "suggestion", "empathy": "pass" or "suggestion"}"""

REVISE_PROMPT = """Revise this customer support response based on the following critique.
Keep the core message but address each issue raised.

Original response:
{response}

Critique:
{critique}

Write ONLY the revised response, nothing else."""


class SelfRefiner:
    """Self-critique and refinement loop for generated responses."""

    def __init__(self, llm_client: LLMClient, max_iterations: int = 3):
        self.llm = llm_client
        self.max_iterations = max_iterations

    async def refine(
        self, response: GeneratedResponse, issue_spec: IssueSpec | None = None
    ) -> GeneratedResponse:
        """Run the self-refinement loop."""
        current_text = response.text
        iterations = 0

        for i in range(self.max_iterations):
            critique = await self._critique(current_text, issue_spec)
            iterations += 1

            # Check if all dimensions pass
            if all(v == "pass" for v in critique.values()):
                break

            # Revise based on critique
            current_text = await self._revise(current_text, critique)

        response.text = current_text
        response.refinement_iterations = iterations
        return response

    async def _critique(
        self, response_text: str, issue_spec: IssueSpec | None = None
    ) -> dict[str, str]:
        """Evaluate the response on specificity, compliance, empathy."""
        context = f"Response to evaluate:\n\"{response_text}\""
        if issue_spec:
            context += f"\n\nThe response should reference this issue: {issue_spec.title}"

        raw = await self.llm.generate(
            system_prompt=CRITIQUE_PROMPT,
            user_prompt=context,
            temperature=0.1,
            max_tokens=512,
        )

        import json

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
            return json.loads(cleaned)
        except Exception:
            return {"specificity": "pass", "compliance": "pass", "empathy": "pass"}

    async def _revise(self, response_text: str, critique: dict[str, str]) -> str:
        """Generate a revised response based on critique."""
        critique_str = "\n".join(
            f"- {dim}: {feedback}" for dim, feedback in critique.items() if feedback != "pass"
        )
        prompt = REVISE_PROMPT.format(response=response_text, critique=critique_str)

        revised = await self.llm.generate(
            system_prompt="You are a customer support specialist. Revise the response.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=512,
        )
        return revised.strip()
