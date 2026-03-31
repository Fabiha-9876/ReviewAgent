"""Aspect-based sentiment analysis for app reviews."""

from __future__ import annotations

from src.common.schemas import AspectSentiment
from src.common.llm_client import LLMClient

SYSTEM_PROMPT = """You are an aspect-based sentiment analyzer for mobile app reviews.
For each review, extract all aspects (features/components the user mentions) and their sentiment.

Return a JSON array of objects with fields:
- "aspect": the feature/component mentioned (lowercase, e.g., "login", "battery", "ui", "camera")
- "sentiment": one of "positive", "negative", or "neutral"
- "intensity": a float 0.0-1.0 indicating how strong the sentiment is

Examples:
- "Great UI but terrible battery drain" → [{"aspect":"ui","sentiment":"positive","intensity":0.8},{"aspect":"battery","sentiment":"negative","intensity":0.9}]
- "App works fine" → [{"aspect":"general","sentiment":"positive","intensity":0.5}]

Return ONLY the JSON array, no other text."""


class AspectSentimentAnalyzer:
    """Extracts aspect-sentiment pairs from review text using an LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    async def analyze(self, text: str) -> list[AspectSentiment]:
        """Extract aspects and sentiments from a single review."""
        try:
            raw = await self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=f"Analyze this review:\n\"{text}\"",
                temperature=0.1,
                max_tokens=512,
            )
            import json

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            items = json.loads(cleaned)
            return [
                AspectSentiment(
                    aspect=item["aspect"].lower(),
                    sentiment=item["sentiment"],
                    intensity=min(1.0, max(0.0, float(item["intensity"]))),
                )
                for item in items
            ]
        except Exception:
            return [AspectSentiment(aspect="unknown", sentiment="neutral", intensity=0.5)]

    async def analyze_batch(self, texts: list[str]) -> list[list[AspectSentiment]]:
        """Analyze multiple reviews."""
        import asyncio

        tasks = [self.analyze(text) for text in texts]
        return await asyncio.gather(*tasks)
