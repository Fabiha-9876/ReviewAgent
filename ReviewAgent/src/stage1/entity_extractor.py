"""Entity extraction from app reviews using regex + LLM hybrid approach."""

from __future__ import annotations

import re

from src.common.schemas import ExtractedEntities
from src.common.llm_client import LLMClient

DEVICE_PATTERNS = [
    r"(iPhone\s*\d+\s*(?:Pro|Max|Plus|Mini)?)",
    r"(iPad\s*(?:Pro|Air|Mini)?(?:\s*\d+)?)",
    r"(Samsung\s*Galaxy\s*\w+\s*\w*)",
    r"(Google\s*Pixel\s*\d+\w?)",
    r"(OnePlus\s*\d+\w?)",
    r"(Huawei\s*\w+)",
    r"(Xiaomi\s*\w+)",
]

OS_PATTERNS = [
    r"(Android\s*\d+\.?\d*)",
    r"(iOS\s*\d+\.?\d*)",
]

VERSION_PATTERNS = [
    r"(?:v|version\s*)(\d+\.\d+(?:\.\d+)?)",
]

SYSTEM_PROMPT = """Extract entities from this app review. Return JSON with:
- "devices": list of device names mentioned
- "os_versions": list of OS versions mentioned
- "app_versions": list of app versions mentioned
- "screens": list of app screens/pages mentioned (e.g., "login screen", "checkout page")
- "features": list of app features mentioned (e.g., "face recognition", "dark mode")
Return ONLY the JSON object. Use empty lists if nothing found."""


class EntityExtractor:
    """Hybrid regex + LLM entity extraction."""

    def __init__(self, llm_client: LLMClient | None = None, use_llm: bool = True):
        self.llm = llm_client
        self.use_llm = use_llm and llm_client is not None

    def _regex_extract(self, text: str) -> ExtractedEntities:
        """Extract entities using regex patterns."""
        devices = []
        for pattern in DEVICE_PATTERNS:
            devices.extend(re.findall(pattern, text, re.IGNORECASE))

        os_versions = []
        for pattern in OS_PATTERNS:
            os_versions.extend(re.findall(pattern, text, re.IGNORECASE))

        app_versions = []
        for pattern in VERSION_PATTERNS:
            app_versions.extend(re.findall(pattern, text, re.IGNORECASE))

        return ExtractedEntities(
            devices=sorted(set(devices)),
            os_versions=sorted(set(os_versions)),
            app_versions=sorted(set(f"v{v}" if not v.startswith("v") else v for v in app_versions)),
        )

    async def extract(self, text: str) -> ExtractedEntities:
        """Extract entities from a single review."""
        regex_entities = self._regex_extract(text)

        if not self.use_llm:
            return regex_entities

        try:
            import json

            raw = await self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=f'Review: "{text}"',
                temperature=0.1,
                max_tokens=512,
            )
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            data = json.loads(cleaned)
            llm_entities = ExtractedEntities(
                devices=data.get("devices", []),
                os_versions=data.get("os_versions", []),
                app_versions=data.get("app_versions", []),
                screens=data.get("screens", []),
                features=data.get("features", []),
            )
            return regex_entities.merge(llm_entities)

        except Exception:
            return regex_entities

    async def extract_batch(self, texts: list[str]) -> list[ExtractedEntities]:
        """Extract entities from multiple reviews."""
        import asyncio

        return await asyncio.gather(*[self.extract(t) for t in texts])
