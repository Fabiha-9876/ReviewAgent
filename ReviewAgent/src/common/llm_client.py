"""Unified LLM API wrapper supporting OpenAI and Anthropic."""

from __future__ import annotations

import json
import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Unified interface for calling LLM APIs."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key: str | None = None,
    ):
        self.provider = provider
        self.model = model

        if provider == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            from anthropic import AsyncAnthropic

            self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a text response from the LLM."""
        if self.provider == "openai":
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            return response.content[0].text

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.2,
    ) -> T:
        """Generate a structured response that conforms to a Pydantic model."""
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)
        full_prompt = (
            f"{user_prompt}\n\n"
            f"Respond ONLY with valid JSON matching this schema:\n{schema_str}"
        )

        raw = await self.generate(system_prompt, full_prompt, temperature=temperature)

        # Extract JSON from potential markdown code blocks
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        return response_model.model_validate_json(text)
